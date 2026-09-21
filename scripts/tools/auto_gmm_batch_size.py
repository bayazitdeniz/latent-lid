#!/usr/bin/env python3
"""Find a safe NNsight trace batch size for GMM fitting."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from gmm_setup import DEFAULT_GMM_SETUP_REGISTRY_PATH, get_gmm_setup
from latents import collect_sequence_latents, get_transformer_layers
from utils import load_nnsight_model, set_seed


################################################################################
# Data loading and batch-size checks
################################################################################

def _load_prompts(data_root: Path, lang: str, text_column: str, max_samples: int) -> list[str]:
    csv_path = data_root / lang / "train.csv"
    if not csv_path.exists():
        fallback = data_root / lang / "clean.csv"
        if fallback.exists():
            csv_path = fallback
        else:
            raise FileNotFoundError(f"Missing calibration CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        for candidate in ("blank_prompt_translation", "blank_prompt_original"):
            if candidate in df.columns:
                text_column = candidate
                break
        else:
            raise ValueError(
                f"Column '{text_column}' not found in {csv_path}; "
                f"available columns: {list(df.columns)}"
            )
    prompts = df[text_column].dropna().astype(str).tolist()
    return prompts[: max_samples]


def _load_prompts_from_roots(
    data_roots: list[Path],
    lang: str,
    text_column: str,
    max_samples: int,
) -> list[str]:
    errors: list[str] = []
    for data_root in data_roots:
        try:
            return _load_prompts(data_root, lang, text_column, max_samples)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else "no data roots provided"
    raise FileNotFoundError(f"Missing calibration data for language '{lang}': {detail}")


def _select_longest_prompts(model, prompts: list[str], k: int):
    if k <= 0:
        return prompts, []
    tokenizer = model.tokenizer
    lengths = []
    for p in prompts:
        lengths.append((len(tokenizer(p).input_ids), p))
    lengths.sort(reverse=True, key=lambda x: x[0])
    return [p for _, p in lengths[:k]], lengths


def _try_batch_size(
    model,
    prompts: list[str],
    trace_batch_size: int,
    layers: list[int] | None,
    *,
    debug_label: str,
    prompt_token_lengths: list[int],
) -> bool:
    try:
        collect_sequence_latents(
            model=model,
            prompts=prompts,
            layer_indices=layers,
            trace_batch_size=trace_batch_size,
            show_progress=False,
            debug_label=debug_label,
            debug_token_lengths=prompt_token_lengths,
        )
        return True
    except RuntimeError as exc:
        if "CUDA out of memory" not in str(exc):
            raise
        print(f"[auto_batch] {debug_label} CUDA OOM: {exc}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def _select_probe_layers(model, requested_layers: list[int] | None) -> list[int]:
    if requested_layers:
        return [list(requested_layers)[-1]]
    layers = get_transformer_layers(model)
    if len(layers) == 0:
        raise ValueError("No transformer layers available to probe.")
    return [len(layers) - 1]


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
    setup_name: str | None,
    result_payload: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        data = _load_json_dict(out_path)
        if setup_name:
            model_entry = data.get(model_name, {})
            if not isinstance(model_entry, dict) or "batch_size" in model_entry:
                model_entry = {}
            model_entry[setup_name] = dict(result_payload)
            data[model_name] = model_entry
        else:
            data[model_name] = dict(result_payload)
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True))
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    parser = argparse.ArgumentParser("Infer a safe NNsight trace batch size for a model + dataset.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--setup-name", default=None)
    parser.add_argument("--gmm-setup-registry", default=str(DEFAULT_GMM_SETUP_REGISTRY_PATH))
    parser.add_argument("--data-root", default="data/pud_holdout/pud_langs_train")
    parser.add_argument("--extra-data-root", nargs="*", default=[])
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--text-column", default="blank_prompt_translation")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--probe-prompts", type=int, default=100)
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512])
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Try batch sizes from largest to smallest for faster success detection.",
    )
    parser.add_argument("--out-json", default="logs/gmm_batch_sizes.json")
    args = parser.parse_args()

    if args.setup_name:
        setup = get_gmm_setup(args.setup_name, args.gmm_setup_registry)
        args.data_root = setup["data_root"]
        args.extra_data_root = list(setup["extra_data_roots"])
        args.languages = list(setup["languages"])
        args.text_column = setup["text_column"]
        args.max_samples = int(setup["max_samples"])
        args.seed = int(setup["seed"])
    if not args.languages:
        raise ValueError("Either --languages or --setup-name is required.")

    set_seed(args.seed)
    model = load_nnsight_model(
        model_name=args.model_name,
        revision=args.revision,
        device=args.device,
        seed=args.seed,
    )
    probe_layers = _select_probe_layers(model, args.layers)
    print(f"[auto_batch] probing layer_indices={probe_layers}", flush=True)

    data_roots = [Path(args.data_root)] + [Path(p) for p in args.extra_data_root]
    per_lang_best: dict[str, int | None] = {}
    per_lang_lengths: dict[str, dict] = {}
    candidates = sorted(args.candidates, reverse=args.descending)
    failed_lang: str | None = None
    skipped_langs: list[str] = []

    for lang_idx, lang in enumerate(args.languages):
        prompts = _load_prompts_from_roots(
            data_roots=data_roots,
            lang=lang,
            text_column=args.text_column,
            max_samples=args.max_samples,
        )
        prompts, lengths = _select_longest_prompts(model, prompts, args.probe_prompts)
        probe_token_lengths = [int(length) for length, _ in lengths[: len(prompts)]]
        if lengths:
            top = lengths[0][0]
            median = sorted([l for l, _ in lengths])[len(lengths) // 2]
            per_lang_lengths[lang] = {"max_tokens": int(top), "median_tokens": int(median)}

        best = None
        print(f"[auto_batch] lang={lang} testing candidates={candidates}")
        for cand in candidates:
            print(f"[auto_batch] lang={lang} try batch_size={cand}")
            debug_label = (
                f"model={args.model_name}"
                f" setup={args.setup_name or 'manual'}"
                f" lang={lang}"
                f" candidate={cand}"
            )
            ok = _try_batch_size(
                model,
                prompts,
                cand,
                probe_layers,
                debug_label=debug_label,
                prompt_token_lengths=probe_token_lengths,
            )
            if ok:
                best = cand
                print(f"[auto_batch] lang={lang} ok batch_size={cand}")
                if args.descending:
                    break
            else:
                print(f"[auto_batch] lang={lang} OOM batch_size={cand}")
                if not args.descending:
                    break
        if best is None:
            print(f"[auto_batch] lang={lang} failed: all candidate batch sizes OOM")
        per_lang_best[lang] = best
        if best is None:
            failed_lang = lang
            skipped_langs = list(args.languages[lang_idx + 1 :])
            break

    failed_langs = [failed_lang] if failed_lang is not None else []
    valid_lang_batches = [batch_size for batch_size in per_lang_best.values() if batch_size is not None]
    any_language_oom = bool(failed_langs)
    best = None if any_language_oom else min(valid_lang_batches)

    result_payload = {
        "batch_size": best,
        "per_lang": per_lang_best,
        "per_lang_token_lengths": per_lang_lengths,
        "probe_layers": probe_layers,
        "failed": any_language_oom,
    }
    if any_language_oom:
        result_payload["failure_reason"] = "all_candidates_oom"
        result_payload["failed_languages"] = failed_langs
        result_payload["skipped_languages"] = skipped_langs
    if args.setup_name:
        result_payload["setup_name"] = args.setup_name

    _merge_result_into_json(
        out_path=Path(args.out_json),
        model_name=args.model_name,
        setup_name=args.setup_name,
        result_payload=result_payload,
    )

    print(best if best is not None else "FAILED")


if __name__ == "__main__":
    main()
