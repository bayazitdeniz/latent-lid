#!/usr/bin/env python3
"""Find a safe training batch size for tuned-lens fitting."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
from typing import Any

import torch

from latents import get_hf_layer_hidden_state
from fit_tuned_lens import (
    DEFAULT_FINEWEB_DATASET_DIR,
    DEFAULT_FINEWEB_TRAIN_SPLIT,
    DEFAULT_PUD_DATA_ROOT,
    DEFAULT_PUD_TEXT_COLUMN,
    TranslatorBank,
    _autocast_context,
    _build_padded_token_batch,
    _rows_contain_text,
    _get_hf_causal_lm,
    _infer_model_max_length,
    _iter_layer_chunks,
    _layer_chunk_reg_loss,
    _load_pud_prompt_records,
    _load_fineweb_split_rows,
    _masked_kl_and_ce,
    _resolve_amp_dtype,
    _resolve_effective_max_length,
    _resolve_fineweb_window_cache_dir,
    _resolve_hidden_size,
    _resolve_layer_indices,
    _window_tokenize_text_rows,
)
from lenses import RawLogitLens, prepare_unembed_info
from utils import load_nnsight_model, set_seed


################################################################################
# Result storage
################################################################################

def _load_json_dict(path: Path) -> dict[str, Any]:
    if path.exists():
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Expected top-level JSON object in {path}")
        return data
    return {}


def _merge_result_into_json(
    *,
    out_path: Path,
    model_name: str,
    dataset_key: str,
    result_payload: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        data = _load_json_dict(out_path)
        model_entry = data.get(model_name, {})
        if not isinstance(model_entry, dict):
            model_entry = {}
        model_entry[str(dataset_key)] = dict(result_payload)
        data[str(model_name)] = model_entry
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True))
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


################################################################################
# Data loading and batch-size checks
################################################################################

def _load_probe_rows(
    *,
    model,
    dataset_source: str,
    languages: list[str],
    max_examples_per_lang: int,
    seed: int,
    max_length: int | None,
    fineweb_dataset_dir: Path,
) -> tuple[list[dict[str, Any]], int | None]:
    tokenizer = model.tokenizer
    if dataset_source == "pud":
        records = _load_pud_prompt_records(
            data_root=DEFAULT_PUD_DATA_ROOT,
            languages=languages,
            text_column=DEFAULT_PUD_TEXT_COLUMN,
            max_samples_per_lang=max_examples_per_lang,
            seed=seed,
        )
        prompts = [record["text"] for record in records]
        rows: list[dict[str, Any]] = []
        for lang in languages:
            lang_prompts = [record["text"] for record in records if record["lang"] == lang][:max_examples_per_lang]
            for prompt in lang_prompts:
                rows.append({"lang": lang, "text": prompt})
        inferred_max_length = _resolve_effective_max_length(
            max_length,
            _infer_model_max_length(model, tokenizer),
        )
    else:
        rows = _load_fineweb_split_rows(
            dataset_dir=fineweb_dataset_dir,
            split_name=DEFAULT_FINEWEB_TRAIN_SPLIT,
            languages=languages,
            max_examples_per_lang=max_examples_per_lang,
            seed=seed,
        )
        inferred_max_length = _resolve_effective_max_length(
            max_length,
            _infer_model_max_length(model, tokenizer),
        )
        if _rows_contain_text(rows):
            if inferred_max_length is None:
                raise ValueError("FineWeb batch probing requires a finite max_length.")
            cache_dir = _resolve_fineweb_window_cache_dir(
                dataset_dir=fineweb_dataset_dir,
                model_name=getattr(model, "model_name", None) or "",
                revision=getattr(model, "revision", None) or "main",
                max_length=int(inferred_max_length),
            )
            if cache_dir.exists():
                rows = _load_fineweb_split_rows(
                    dataset_dir=cache_dir,
                    split_name=DEFAULT_FINEWEB_TRAIN_SPLIT,
                    languages=languages,
                    max_examples_per_lang=max_examples_per_lang,
                    seed=seed,
                )
            else:
                rows = _window_tokenize_text_rows(
                    rows,
                    tokenizer=tokenizer,
                    max_length=int(inferred_max_length),
                    desc="Window-tokenizing FineWeb probe split",
                )
    token_rows: list[dict[str, Any]] = []
    if _rows_contain_text(rows):
        assert inferred_max_length is not None
        token_rows = _window_tokenize_text_rows(
            rows,
            tokenizer=tokenizer,
            max_length=int(inferred_max_length),
            desc="Window-tokenizing probe rows",
        )
    else:
        token_rows = rows
    token_rows = sorted(token_rows, key=lambda row: len(row["input_ids"]), reverse=True)
    return token_rows, inferred_max_length


def _try_tuned_lens_batch_size(
    *,
    model,
    token_rows: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
    seed: int,
    layers: list[int] | None,
    layers_per_step: int,
    amp_dtype: torch.dtype | None,
) -> bool:
    set_seed(seed)
    tokenizer = model.tokenizer
    causal_lm = _get_hf_causal_lm(model)
    hidden_size = _resolve_hidden_size(causal_lm)
    layer_indices = _resolve_layer_indices(model, layers)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

    probe_rows = token_rows[:batch_size]
    if len(probe_rows) < batch_size:
        raise ValueError(f"Not enough probe rows ({len(probe_rows)}) to test batch size {batch_size}.")

    translators = TranslatorBank(hidden_size=hidden_size, layer_indices=layer_indices).to(device)
    optimizer = torch.optim.AdamW(translators.parameters(), lr=1e-3, weight_decay=0.0)
    unembed_info = prepare_unembed_info(model, device=device)
    raw_lens = RawLogitLens(unembed_info=unembed_info, apply_final_norm=True)
    eye = torch.eye(hidden_size, device=device)
    try:
        input_ids, attention_mask = _build_padded_token_batch(
            probe_rows,
            pad_token_id=int(pad_token_id),
            device=device,
        )
        with torch.no_grad(), _autocast_context(device, amp_dtype):
            outputs = causal_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            teacher_logits = outputs.logits[:, :-1, :].detach()
            target_ids = input_ids[:, 1:]
            token_mask = attention_mask[:, 1:].bool()

        optimizer.zero_grad(set_to_none=True)
        n_layers = max(len(layer_indices), 1)
        for layer_chunk in _iter_layer_chunks(layer_indices, int(layers_per_step)):
            chunk_loss = torch.zeros((), device=device)
            for layer_idx in layer_chunk:
                hidden = get_hf_layer_hidden_state(
                    outputs.hidden_states,
                    int(layer_idx),
                )[:, :-1, :].detach()
                with _autocast_context(device, amp_dtype):
                    translated = translators(hidden, int(layer_idx))
                    student_logits = raw_lens.project_layer(translated, int(layer_idx))
                    layer_kl, _ = _masked_kl_and_ce(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        target_ids=target_ids,
                        token_mask=token_mask,
                        temperature=1.0,
                    )
                chunk_loss = chunk_loss + (layer_kl / n_layers)
                del hidden
                del translated
                del student_logits
                del layer_kl
            chunk_reg_loss = _layer_chunk_reg_loss(
                translators=translators,
                eye=eye,
                layer_chunk=layer_chunk,
                identity_reg_weight=1e-4,
                bias_reg_weight=1e-4,
                device=device,
            )
            total_chunk_loss = chunk_loss + chunk_reg_loss
            total_chunk_loss.backward()
            del chunk_reg_loss
            del chunk_loss
            del total_chunk_loss
        optimizer.step()
        del teacher_logits
        del target_ids
        del token_mask
        del outputs
        del input_ids
        del attention_mask
        return True
    except RuntimeError as exc:
        if "CUDA out of memory" not in str(exc):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False
    finally:
        del translators
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    parser = argparse.ArgumentParser("Infer a safe tuned-lens training batch size for a model and dataset source.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset-source", choices=["pud", "fineweb"], default="fineweb")
    parser.add_argument("--batch-size-key", default=None)
    parser.add_argument("--fineweb-dataset-dir", default=str(DEFAULT_FINEWEB_DATASET_DIR))
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--max-examples-per-lang", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--layers-per-step", type=int, default=1)
    parser.add_argument("--amp-dtype", default="bfloat16")
    parser.add_argument("--candidates", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--out-json", default="logs/tuned_lens_batch_sizes.json")
    parser.add_argument("--merge-into", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    model = load_nnsight_model(
        model_name=args.model_name,
        revision=args.revision,
        device=args.device,
        seed=args.seed,
    )
    setattr(model, "model_name", args.model_name)
    setattr(model, "revision", args.revision)
    amp_dtype = _resolve_amp_dtype(torch.device(args.device), args.amp_dtype)
    candidates = sorted(args.candidates, reverse=args.descending)
    token_rows, inferred_max_length = _load_probe_rows(
        model=model,
        dataset_source=args.dataset_source,
        languages=list(args.languages),
        max_examples_per_lang=int(args.max_examples_per_lang),
        seed=args.seed,
        max_length=args.max_length,
        fineweb_dataset_dir=Path(args.fineweb_dataset_dir),
    )
    if args.max_length is not None and inferred_max_length is not None and int(args.max_length) > int(inferred_max_length):
        print(
            f"[auto_tuned_lens_batch] clamped requested max_length={int(args.max_length)} "
            f"down to model-supported max_length={int(inferred_max_length)}"
        )

    best = min(candidates) if candidates else 1
    print(f"[auto_tuned_lens_batch] dataset_source={args.dataset_source} candidates={candidates}")
    for cand in candidates:
        print(f"[auto_tuned_lens_batch] try batch_size={cand}")
        ok = _try_tuned_lens_batch_size(
            model=model,
            token_rows=token_rows,
            batch_size=int(cand),
            device=torch.device(args.device),
            seed=args.seed,
            layers=args.layers,
            layers_per_step=int(args.layers_per_step),
            amp_dtype=amp_dtype,
        )
        if ok:
            best = int(cand)
            print(f"[auto_tuned_lens_batch] ok batch_size={cand}")
            if args.descending:
                break
        else:
            print(f"[auto_tuned_lens_batch] OOM batch_size={cand}")
            if not args.descending:
                break

    result_payload = {
        "batch_size": best,
        "dataset_source": args.dataset_source,
        "fineweb_dataset_dir": args.fineweb_dataset_dir if args.dataset_source == "fineweb" else None,
        "max_length": inferred_max_length,
        "n_probe_rows": len(token_rows),
        "layers_per_step": int(args.layers_per_step),
        "amp_dtype": args.amp_dtype,
    }
    _merge_result_into_json(
        out_path=Path(args.out_json),
        model_name=args.model_name,
        dataset_key=args.batch_size_key or args.dataset_source,
        result_payload=result_payload,
    )
    if args.merge_into not in {None, "", "none"} and Path(args.merge_into) != Path(args.out_json):
        _merge_result_into_json(
            out_path=Path(args.merge_into),
            model_name=args.model_name,
            dataset_key=args.batch_size_key or args.dataset_source,
            result_payload=result_payload,
        )
    print(best)


if __name__ == "__main__":
    main()
