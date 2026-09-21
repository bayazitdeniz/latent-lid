"""Fit per-layer tuned-lens affine translators from multilingual text."""

import argparse
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from transformers import logging as transformers_logging

from latents import get_hf_layer_hidden_state
from lenses import (
    RawLogitLens,
    build_tuned_lens_base_dir,
    hash_training_config,
    next_tuned_lens_run_dir,
    prepare_unembed_info,
    tuned_lens_layer_filename,
)
from run_eval_utils import format_prompts_for_tokenizer
from utils import load_nnsight_model, sanitize_model_id, set_seed


DEFAULT_PUD_DATA_ROOT = Path("data/pud_holdout/pud_langs_train")
DEFAULT_PUD_TEXT_COLUMN = "prompt"
DEFAULT_FINEWEB_DATASET_DIR = Path("data/fineweb_lens_27lang_55m")
DEFAULT_FINEWEB_TRAIN_SPLIT = "train"
DEFAULT_FINEWEB_VAL_SPLIT = "validation"
DEFAULT_FINEWEB_WINDOW_CACHE_SUBDIR = "window_cache"
FINEWEB_LANGUAGE_ALIASES = {
    "pt_br": "pt",
}
DEFAULT_TUNED_LENS_OUT_DIR_BY_SOURCE = {
    "pud": Path("logs/tuned_lens_pud"),
    "fineweb": Path("logs/tuned_lens_fineweb27"),
}


################################################################################
# Fitting data loading
################################################################################

def _fineweb_source_lang(lang: str) -> str:
    """Map a requested language to its FineWeb source label."""
    return FINEWEB_LANGUAGE_ALIASES.get(str(lang), str(lang))


def _fineweb_lang_matches(row_lang: str, requested_lang: str) -> bool:
    """Return whether a FineWeb row matches the requested language."""
    row_lang = str(row_lang)
    requested_lang = str(requested_lang)
    return row_lang == requested_lang or row_lang == _fineweb_source_lang(requested_lang)


def _load_pud_prompt_records(
    data_root: Path,
    languages: Sequence[str],
    text_column: str,
    max_samples_per_lang: int | None,
    seed: int,
) -> list[dict[str, str]]:
    """Load deterministically sampled PUD prompts for each language."""
    prompt_jsonl = data_root.parent / "pud_prompts_train.jsonl"
    if prompt_jsonl.exists():
        rows: list[dict[str, str]] = []
        with prompt_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                prompt_id = str(obj.get("id", ""))
                lang = prompt_id.split("_", 1)[0] if "_" in prompt_id else None
                if lang not in languages:
                    continue
                text = obj.get("prompt_text") or obj.get("text")
                if text is None:
                    continue
                rows.append({"lang": lang, "text": str(text)})
        if rows:
            out: list[dict[str, str]] = []
            for lang in languages:
                lang_rows = [row for row in rows if row["lang"] == lang]
                if not lang_rows:
                    continue
                sample_n = (
                    len(lang_rows)
                    if max_samples_per_lang is None
                    else min(max_samples_per_lang, len(lang_rows))
                )
                sampled_idx = pd.Series(range(len(lang_rows))).sample(n=sample_n, random_state=seed)
                out.extend(lang_rows[int(i)] for i in sampled_idx.tolist())
            return out

    records: list[dict[str, str]] = []
    for lang in languages:
        csv_path = data_root / lang / "train.csv"
        if not csv_path.exists():
            fallback = data_root / lang / "clean.csv"
            if fallback.exists():
                csv_path = fallback
            else:
                raise FileNotFoundError(f"Missing calibration CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in {csv_path}; available columns: {list(df.columns)}"
            )
        if max_samples_per_lang is not None:
            df = df.sample(n=min(max_samples_per_lang, len(df)), random_state=seed)
        for text in df[text_column].dropna().astype(str).tolist():
            records.append({"lang": lang, "text": text})
    if not records:
        raise ValueError("No calibration prompts loaded.")
    return records


def _iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _resolve_pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    raise ValueError("Tokenizer must define either pad_token_id or eos_token_id for padded token batches.")


def _load_fineweb_split_rows(
    dataset_dir: Path,
    split_name: str,
    languages: Sequence[str],
    max_examples_per_lang: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Load deterministically sampled FineWeb text or token rows."""
    dataset_dict = load_from_disk(str(dataset_dir))
    if split_name not in dataset_dict:
        raise ValueError(
            f"Split '{split_name}' not found in dataset at {dataset_dir}; "
            f"available splits: {list(dataset_dict.keys())}"
        )
    rows = [dict(row) for row in dataset_dict[split_name]]
    if not rows:
        raise ValueError(f"Split '{split_name}' is empty in dataset at {dataset_dir}.")

    out: list[dict[str, Any]] = []
    for lang in languages:
        lang_rows = [row for row in rows if _fineweb_lang_matches(str(row.get("lang")), str(lang))]
        if not lang_rows:
            raise ValueError(f"No examples for language '{lang}' in split '{split_name}' at {dataset_dir}.")
        if max_examples_per_lang is not None:
            sample_n = min(int(max_examples_per_lang), len(lang_rows))
            sampled_idx = pd.Series(range(len(lang_rows))).sample(n=sample_n, random_state=seed)
            lang_rows = [lang_rows[int(i)] for i in sampled_idx.tolist()]
        for row in lang_rows:
            text = row.get("text")
            input_ids = row.get("input_ids")
            if isinstance(text, str) and text:
                out.append({"lang": lang, "text": text})
                continue
            if isinstance(input_ids, list) and input_ids:
                out.append(
                    {
                        "lang": lang,
                        "input_ids": [int(token_id) for token_id in input_ids],
                    }
                )
                continue
            raise ValueError(
                f"Row in split '{split_name}' for language '{lang}' must contain non-empty text or input_ids."
            )
    if not out:
        raise ValueError(f"No tokenized examples loaded from split '{split_name}' at {dataset_dir}.")
    return out


def _build_padded_token_batch(
    rows: Sequence[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad token rows and return input IDs with their attention mask."""
    max_len = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    attention_mask = []
    for row in rows:
        ids = list(row["input_ids"])
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
    )


def _rows_contain_text(rows: Sequence[dict[str, Any]]) -> bool:
    """Return whether dataset rows contain text rather than token IDs."""
    if not rows:
        raise ValueError("Expected non-empty dataset rows.")
    sample = rows[0]
    if "text" in sample:
        return True
    if "input_ids" in sample:
        return False
    raise ValueError("Dataset rows must contain either text or input_ids.")


def _select_balanced_rows(
    rows: Sequence[dict[str, Any]],
    *,
    languages: Sequence[str],
    max_examples_per_lang: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], int | None]:
    """Sample the same capped number of validation rows per language."""
    if max_examples_per_lang is None:
        return [dict(row) for row in rows], None
    per_lang_rows: dict[str, list[dict[str, Any]]] = {}
    for lang in languages:
        lang_rows = [dict(row) for row in rows if str(row.get("lang")) == str(lang)]
        if not lang_rows:
            raise ValueError(f"No rows available for language '{lang}' in balanced subset selection.")
        per_lang_rows[str(lang)] = lang_rows
    effective_cap = min(int(max_examples_per_lang), min(len(lang_rows) for lang_rows in per_lang_rows.values()))
    selected: list[dict[str, Any]] = []
    for offset, lang in enumerate(languages):
        lang_rows = per_lang_rows[str(lang)]
        sampled_idx = pd.Series(range(len(lang_rows))).sample(n=effective_cap, random_state=seed + offset)
        selected.extend(lang_rows[int(i)] for i in sampled_idx.tolist())
    return selected, effective_cap


def _build_balanced_train_rows(
    rows: Sequence[dict[str, Any]],
    *,
    languages: Sequence[str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Shuffle per language, then interleave training rows in rotating round-robin order."""
    per_lang_rows: dict[str, list[dict[str, Any]]] = {str(lang): [] for lang in languages}
    for row in rows:
        row_lang = str(row.get("lang"))
        matched_lang = row_lang if row_lang in per_lang_rows else None
        if matched_lang is None:
            matches = [str(lang) for lang in languages if _fineweb_lang_matches(row_lang, str(lang))]
            if len(matches) == 1:
                matched_lang = matches[0]
        if matched_lang is None:
            continue
        normalized = dict(row)
        normalized["lang"] = matched_lang
        per_lang_rows[matched_lang].append(normalized)

    missing = [str(lang) for lang in languages if not per_lang_rows[str(lang)]]
    if missing:
        raise ValueError(
            "Cannot balance tuned-lens training rows; missing examples for languages: "
            + ", ".join(missing)
        )

    counts = {str(lang): len(per_lang_rows[str(lang)]) for lang in languages}
    min_count = min(counts.values())
    max_count = max(counts.values())
    if min_count != max_count:
        print(
            "Balancing tuned-lens training rows with strict round-robin/drop-remainder: "
            f"min={min_count}, max={max_count}; dropping extras to {min_count}/lang."
        )
    else:
        print(
            "Balancing tuned-lens training rows with strict round-robin: "
            f"{min_count}/lang across {len(languages)} languages."
        )

    shuffled_by_lang: dict[str, list[dict[str, Any]]] = {}
    for offset, lang in enumerate(languages):
        lang_key = str(lang)
        sampled_idx = pd.Series(range(len(per_lang_rows[lang_key]))).sample(
            n=min_count,
            random_state=seed + offset,
        )
        shuffled_by_lang[lang_key] = [per_lang_rows[lang_key][int(i)] for i in sampled_idx.tolist()]

    balanced: list[dict[str, Any]] = []
    lang_order = [str(lang) for lang in languages]
    for example_idx in range(min_count):
        rotation = example_idx % len(lang_order)
        rotated_langs = lang_order[rotation:] + lang_order[:rotation]
        for lang in rotated_langs:
            balanced.append(shuffled_by_lang[lang][example_idx])
    return balanced, counts, min_count


def _extract_batch_langs(batch_rows: Sequence[dict[str, Any]]) -> list[str]:
    return [str(row["lang"]) for row in batch_rows]


def _window_tokenize_text_rows(
    rows: Sequence[dict[str, Any]],
    *,
    tokenizer,
    max_length: int | None,
    desc: str,
) -> list[dict[str, Any]]:
    """Tokenize text rows into non-overlapping, model-length windows."""
    if max_length is None:
        raise ValueError("max_length must be set before window-tokenizing text rows.")
    out: list[dict[str, Any]] = []
    for row in tqdm(rows, desc=desc, unit="example"):
        original_verbosity = transformers_logging.get_verbosity()
        try:
            transformers_logging.set_verbosity_error()
            encoded = tokenizer(
                str(row["text"]),
                add_special_tokens=True,
                truncation=False,
            )
        finally:
            transformers_logging.set_verbosity(original_verbosity)
        input_ids = [int(token_id) for token_id in encoded["input_ids"]]
        if not input_ids:
            continue
        for start in range(0, len(input_ids), int(max_length)):
            window_ids = input_ids[start : start + int(max_length)]
            if len(window_ids) < 2:
                continue
            out.append({"lang": str(row["lang"]), "input_ids": window_ids})
    if not out:
        raise ValueError("Window tokenization produced no usable token windows.")
    return out


@dataclass
class PreparedFittingData:
    """Resolved training and lightweight-validation rows for one fitting run."""

    uses_fineweb: bool
    train_rows: list[dict[str, Any]]
    light_validation_rows: list[dict[str, Any]]
    rows_contain_token_ids: bool
    pad_token_id: int | None
    effective_max_length: int | None
    effective_light_val_examples: int | None
    train_rows_per_lang_before_balance: dict[str, int]
    train_examples_per_lang: int
    n_train_examples: int
    n_val_examples: int
    calibration_data: Path


def _prepare_fitting_data(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
) -> PreparedFittingData:
    """Load, format, and deterministically balance fitting rows."""
    uses_fineweb = args.dataset_source == "fineweb"
    fineweb_dataset_dir = Path(args.fineweb_dataset_dir)
    pad_token_id = _resolve_pad_token_id(tokenizer) if uses_fineweb else None
    train_prompt_rows: list[dict[str, Any]] = []
    val_prompt_rows: list[dict[str, Any]] = []
    train_token_rows: list[dict[str, Any]] = []
    val_token_rows: list[dict[str, Any]] = []
    rows_contain_text = False

    if uses_fineweb:
        train_token_rows = _load_fineweb_split_rows(
            dataset_dir=fineweb_dataset_dir,
            split_name=DEFAULT_FINEWEB_TRAIN_SPLIT,
            languages=args.languages,
            max_examples_per_lang=args.max_samples_per_lang,
            seed=args.seed,
        )
        val_token_rows = _load_fineweb_split_rows(
            dataset_dir=fineweb_dataset_dir,
            split_name=DEFAULT_FINEWEB_VAL_SPLIT,
            languages=args.languages,
            max_examples_per_lang=args.max_samples_per_lang,
            seed=args.seed + 1,
        )
        rows_contain_text = _rows_contain_text(train_token_rows)
        if _rows_contain_text(val_token_rows) != rows_contain_text:
            raise ValueError(
                "Train and validation dataset splits must use the same row format "
                "(text or input_ids)."
            )
        if rows_contain_text:
            train_prompt_rows = [dict(row) for row in train_token_rows]
            val_prompt_rows = [dict(row) for row in val_token_rows]
    else:
        records = _load_pud_prompt_records(
            data_root=DEFAULT_PUD_DATA_ROOT,
            languages=args.languages,
            text_column=DEFAULT_PUD_TEXT_COLUMN,
            max_samples_per_lang=args.max_samples_per_lang,
            seed=args.seed,
        )
        raw_prompts = [record["text"] for record in records]
        prompts, _ = format_prompts_for_tokenizer(tokenizer, raw_prompts)
        if not prompts:
            raise ValueError("No prompts available for tuned-lens fitting.")
        if len(prompts) != len(records):
            raise ValueError("Formatted prompts length does not match original prompt records.")
        prompt_rows = [
            {"lang": str(record["lang"]), "text": str(prompt)}
            for record, prompt in zip(records, prompts)
        ]
        shuffled_idx = (
            pd.Series(range(len(prompt_rows)))
            .sample(frac=1.0, random_state=args.seed)
            .tolist()
        )
        shuffled = [prompt_rows[int(idx)] for idx in shuffled_idx]
        val_count = max(1, int(len(shuffled) * float(args.val_split)))
        val_prompt_rows = shuffled[:val_count]
        train_prompt_rows = shuffled[val_count:]
        if not train_prompt_rows:
            raise ValueError(
                "Validation split consumed all prompts; increase calibration data or "
                "lower --val-split."
            )

    inferred_model_max_length = _infer_model_max_length(model, tokenizer)
    effective_max_length = _resolve_effective_max_length(
        args.max_length,
        inferred_model_max_length,
    )
    if uses_fineweb and rows_contain_text and effective_max_length is not None:
        if args.max_length is None:
            print(
                f"Using inferred max_length={effective_max_length} for "
                f"dataset_source=fineweb with model {args.model_name}."
            )
        elif (
            inferred_model_max_length is not None
            and int(args.max_length) > int(effective_max_length)
        ):
            print(
                f"Clamped requested max_length={int(args.max_length)} down to "
                f"model-supported max_length={effective_max_length} for model "
                f"{args.model_name}."
            )

        fineweb_cache_dir = _resolve_fineweb_window_cache_dir(
            dataset_dir=fineweb_dataset_dir,
            model_name=args.model_name,
            revision=args.revision,
            max_length=int(effective_max_length),
        )
        if fineweb_cache_dir.exists():
            print(f"Loading cached FineWeb windows from {fineweb_cache_dir}")
            train_token_rows = _load_fineweb_split_rows(
                dataset_dir=fineweb_cache_dir,
                split_name=DEFAULT_FINEWEB_TRAIN_SPLIT,
                languages=args.languages,
                max_examples_per_lang=args.max_samples_per_lang,
                seed=args.seed,
            )
            val_token_rows = _load_fineweb_split_rows(
                dataset_dir=fineweb_cache_dir,
                split_name=DEFAULT_FINEWEB_VAL_SPLIT,
                languages=args.languages,
                max_examples_per_lang=args.max_samples_per_lang,
                seed=args.seed + 1,
            )
        else:
            train_token_rows = _window_tokenize_text_rows(
                train_token_rows,
                tokenizer=tokenizer,
                max_length=effective_max_length,
                desc="Window-tokenizing FineWeb train split",
            )
            val_token_rows = _window_tokenize_text_rows(
                val_token_rows,
                tokenizer=tokenizer,
                max_length=effective_max_length,
                desc="Window-tokenizing FineWeb validation split",
            )
            fineweb_cache_dir.parent.mkdir(parents=True, exist_ok=True)
            DatasetDict(
                {
                    DEFAULT_FINEWEB_TRAIN_SPLIT: Dataset.from_list(train_token_rows),
                    DEFAULT_FINEWEB_VAL_SPLIT: Dataset.from_list(val_token_rows),
                }
            ).save_to_disk(str(fineweb_cache_dir))
            print(f"Saved cached FineWeb windows to {fineweb_cache_dir}")
        rows_contain_text = False
        print(
            f"Prepared {len(train_token_rows)} FineWeb train windows and "
            f"{len(val_token_rows)} validation windows for model {args.model_name}."
        )

    rows_contain_token_ids = uses_fineweb and not rows_contain_text
    if rows_contain_token_ids:
        light_validation_rows, effective_light_val_examples = _select_balanced_rows(
            val_token_rows,
            languages=args.languages,
            max_examples_per_lang=int(args.light_val_examples_per_lang),
            seed=args.seed + 10_000,
        )
        print(
            f"Prepared lightweight validation subset with {effective_light_val_examples} "
            f"examples per language ({len(light_validation_rows)} total windows)."
        )
    else:
        light_validation_rows, effective_light_val_examples = _select_balanced_rows(
            val_prompt_rows,
            languages=args.languages,
            max_examples_per_lang=int(args.light_val_examples_per_lang),
            seed=args.seed + 10_000,
        )
        print(
            f"Prepared lightweight validation subset with {effective_light_val_examples} "
            f"examples per language ({len(light_validation_rows)} total prompts)."
        )
    print(
        f"Validation cadence: every {int(args.val_every_steps)} steps. "
        f"Checkpoint cadence: every {int(args.save_every_steps)} steps."
    )

    unbalanced_train_rows = train_token_rows if rows_contain_token_ids else train_prompt_rows
    n_train_examples = len(train_token_rows) if uses_fineweb else len(train_prompt_rows)
    n_val_examples = len(light_validation_rows)
    train_rows, rows_per_lang, examples_per_lang = _build_balanced_train_rows(
        unbalanced_train_rows,
        languages=args.languages,
        seed=args.seed + 20_000,
    )
    return PreparedFittingData(
        uses_fineweb=uses_fineweb,
        train_rows=train_rows,
        light_validation_rows=light_validation_rows,
        rows_contain_token_ids=rows_contain_token_ids,
        pad_token_id=pad_token_id,
        effective_max_length=effective_max_length,
        effective_light_val_examples=effective_light_val_examples,
        train_rows_per_lang_before_balance=rows_per_lang,
        train_examples_per_lang=examples_per_lang,
        n_train_examples=n_train_examples,
        n_val_examples=n_val_examples,
        calibration_data=(fineweb_dataset_dir if uses_fineweb else DEFAULT_PUD_DATA_ROOT),
    )


################################################################################
# CLI and configuration resolution
################################################################################

def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "t", "yes", "y"}


def _build_parser() -> argparse.ArgumentParser:
    """Build the tuned-lens fitting CLI parser."""
    parser = argparse.ArgumentParser(
        "Fit per-layer tuned-lens affine translators from calibration prompts."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["ar", "cs", "en", "fr", "hi", "is", "id", "pt", "es"],
    )
    parser.add_argument("--dataset-source", choices=["pud", "fineweb"], default="fineweb")
    parser.add_argument("--fineweb-dataset-dir", default=str(DEFAULT_FINEWEB_DATASET_DIR))
    parser.add_argument("--max-samples-per-lang", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--identity-reg-weight", type=float, default=1e-4)
    parser.add_argument("--bias-reg-weight", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--light-val-examples-per-lang", type=int, default=32)
    parser.add_argument("--val-every-steps", type=int, default=100)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--apply-final-norm", type=_parse_bool, default=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--layers-per-step", type=int, default=1)
    parser.add_argument("--amp-dtype", default="bfloat16")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", default="llid-tuned-lens")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate step and checkpoint settings before fitting starts."""
    if int(args.max_train_steps) <= 0:
        raise ValueError("--max-train-steps must be >= 1")
    if int(args.val_every_steps) <= 0:
        raise ValueError("--val-every-steps must be >= 1")
    if int(args.save_every_steps) <= 0:
        raise ValueError("--save-every-steps must be >= 1")
    if int(args.save_every_steps) % int(args.val_every_steps) != 0:
        raise ValueError("--save-every-steps must be a positive multiple of --val-every-steps.")


################################################################################
# Model and lens setup
################################################################################


def _resolve_layer_indices(model, requested_layers: Sequence[int] | None) -> list[int]:
    """Resolve requested transformer layers, defaulting to every layer."""
    hf_model = getattr(model, "_model", model)
    config = getattr(hf_model, "config", None)
    if config is None or getattr(config, "num_hidden_layers", None) is None:
        raise ValueError("Unable to infer num_hidden_layers from model config.")
    total_layers = int(config.num_hidden_layers)
    if requested_layers:
        return [int(layer_idx) for layer_idx in requested_layers]
    return list(range(total_layers))


def _resolve_trainable_layers(model, requested_layers: Sequence[int] | None) -> tuple[list[int], list[int], int]:
    """Resolve requested and trainable layers, excluding the raw-head final layer."""
    resolved_layers = _resolve_layer_indices(model, requested_layers)
    hf_model = getattr(model, "_model", model)
    config = getattr(hf_model, "config", None)
    total_layers = int(config.num_hidden_layers)
    final_layer_idx = total_layers - 1
    trainable_layers = [int(layer_idx) for layer_idx in resolved_layers if int(layer_idx) != final_layer_idx]
    if not trainable_layers:
        raise ValueError(
            "No trainable tuned-lens layers remain after excluding the final transformer layer. "
            "Request at least one non-final layer."
        )
    return resolved_layers, trainable_layers, final_layer_idx


def _get_hf_causal_lm(model):
    """Return the underlying Hugging Face causal language model."""
    return getattr(model, "_model", model)


def _resolve_hidden_size(hf_model) -> int:
    """Resolve hidden size across supported model configuration conventions."""
    config = getattr(hf_model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Unable to infer hidden size from model config.")


def _infer_model_max_length(model, tokenizer) -> int | None:
    hf_model = _get_hf_causal_lm(model)
    config = getattr(hf_model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if tokenizer_max is None:
        return None
    tokenizer_max = int(tokenizer_max)
    if tokenizer_max >= 1_000_000_000:
        return None
    return tokenizer_max


def _resolve_effective_max_length(
    requested_max_length: int | None,
    model_max_length: int | None,
) -> int | None:
    """Clamp a requested length to the model-supported context length."""
    if requested_max_length is None:
        return model_max_length
    if model_max_length is None:
        return int(requested_max_length)
    return min(int(requested_max_length), int(model_max_length))


def _resolve_amp_dtype(device: torch.device, amp_dtype: str | None) -> torch.dtype | None:
    """Resolve optional CUDA autocast dtype from its CLI name."""
    if device.type != "cuda" or amp_dtype in {None, "", "none"}:
        return None
    key = str(amp_dtype).lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}")


def _autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _iter_layer_chunks(layer_indices: Sequence[int], layers_per_step: int) -> Iterable[list[int]]:
    """Yield consecutive layer groups in the requested order.

    Training backpropagates through one group at a time to limit the peak memory
    used by the lens projections.
    """
    if int(layers_per_step) <= 0:
        raise ValueError("--layers-per-step must be >= 1")
    for start in range(0, len(layer_indices), int(layers_per_step)):
        yield [int(layer_idx) for layer_idx in layer_indices[start : start + int(layers_per_step)]]


class TranslatorBank(nn.Module):
    """Transform each selected layer with its own identity-initialized linear map."""

    def __init__(self, hidden_size: int, layer_indices: Sequence[int]) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layer_indices = [int(layer_idx) for layer_idx in layer_indices]
        self.layers = nn.ModuleDict()
        eye = torch.eye(self.hidden_size)
        for layer_idx in self.layer_indices:
            linear = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            with torch.no_grad():
                linear.weight.copy_(eye)
                linear.bias.zero_()
            self.layers[str(layer_idx)] = linear

    def forward(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.layers[str(int(layer_idx))](hidden_states)


def _masked_kl_and_ce(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_ids: torch.Tensor,
    token_mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute temperature-scaled KL and target-token CE over valid tokens."""
    flat_mask = token_mask.reshape(-1)
    if not bool(flat_mask.any()):
        raise ValueError("token_mask must include at least one valid token.")
    flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])[flat_mask]
    flat_student = student_logits.reshape(-1, student_logits.shape[-1])[flat_mask]
    flat_targets = target_ids.reshape(-1)[flat_mask]
    t = float(temperature)
    teacher_log_probs = torch.log_softmax(flat_teacher / t, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = torch.log_softmax(flat_student / t, dim=-1)
    token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1) * (t * t)
    loss_kl = token_kl.mean()
    loss_ce = F.cross_entropy(flat_student, flat_targets)
    return loss_kl, loss_ce


def _layer_chunk_reg_loss(
    *,
    translators: TranslatorBank,
    eye: torch.Tensor,
    layer_chunk: Sequence[int],
    identity_reg_weight: float,
    bias_reg_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute identity and bias regularization for one layer chunk."""
    reg_loss = torch.zeros((), device=device)
    if float(identity_reg_weight) > 0.0:
        reg_loss = reg_loss + float(identity_reg_weight) * sum(
            (translators.layers[str(int(layer_idx))].weight - eye).pow(2).mean()
            for layer_idx in layer_chunk
        )
    if float(bias_reg_weight) > 0.0:
        reg_loss = reg_loss + float(bias_reg_weight) * sum(
            translators.layers[str(int(layer_idx))].bias.pow(2).mean()
            for layer_idx in layer_chunk
        )
    return reg_loss


def _lang_index_map(batch_langs: Sequence[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx, lang in enumerate(batch_langs):
        out.setdefault(str(lang), []).append(int(idx))
    return out


def _mean_across_layers(values_by_layer: dict[int, float]) -> float:
    if not values_by_layer:
        return 0.0
    return float(sum(values_by_layer.values()) / max(len(values_by_layer), 1))


################################################################################
# Validation
################################################################################

def _evaluate_rows(
    *,
    rows: Sequence[dict[str, Any]],
    rows_contain_token_ids: bool,
    batch_size: int,
    tokenizer,
    pad_token_id: int | None,
    causal_lm,
    translators: TranslatorBank,
    unembed_info,
    layer_indices: Sequence[int],
    temperature: float,
    apply_final_norm: bool,
    device: torch.device,
    max_length: int | None,
    amp_dtype: torch.dtype | None,
    layers_per_step: int,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    """Evaluate translators on text or pretokenized rows without changing row order."""
    translators.eval()
    raw_lens = RawLogitLens(
        unembed_info=unembed_info,
        apply_final_norm=apply_final_norm,
    )
    total_kl = {int(layer_idx): 0.0 for layer_idx in layer_indices}
    total_ce = {int(layer_idx): 0.0 for layer_idx in layer_indices}
    total_tokens = {int(layer_idx): 0 for layer_idx in layer_indices}
    total_kl_by_lang = {int(layer_idx): {} for layer_idx in layer_indices}
    total_ce_by_lang = {int(layer_idx): {} for layer_idx in layer_indices}
    total_tokens_by_lang = {int(layer_idx): {} for layer_idx in layer_indices}
    val_pbar = None
    if progress_desc is not None:
        total_batches = max((len(rows) + int(batch_size) - 1) // int(batch_size), 1)
        val_pbar = tqdm(total=total_batches, desc=progress_desc, unit="batch", leave=False)
    with torch.inference_mode():
        for batch_rows in _iter_batches(rows, batch_size):
            batch_lang_to_indices = _lang_index_map(_extract_batch_langs(batch_rows))
            if rows_contain_token_ids:
                if pad_token_id is None:
                    raise ValueError("pad_token_id is required for pretokenized validation rows.")
                input_ids, attention_mask = _build_padded_token_batch(
                    batch_rows,
                    pad_token_id=pad_token_id,
                    device=device,
                )
            else:
                prompt_batch = [str(row["text"]) for row in batch_rows]
                encoded = tokenizer(
                    prompt_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=max_length is not None,
                    max_length=max_length,
                    add_special_tokens=True,
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
            with _autocast_context(device, amp_dtype):
                outputs = causal_lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            teacher_logits = outputs.logits[:, :-1, :].detach()
            target_ids = input_ids[:, 1:]
            token_mask = attention_mask[:, 1:].bool()
            n_tokens = int(token_mask.sum().item())
            for layer_chunk in _iter_layer_chunks(layer_indices, layers_per_step):
                for layer_idx in layer_chunk:
                    hidden = get_hf_layer_hidden_state(
                        outputs.hidden_states,
                        int(layer_idx),
                    )[:, :-1, :].detach()
                    with _autocast_context(device, amp_dtype):
                        translated = translators(hidden, int(layer_idx))
                        student_logits = raw_lens.project_layer(translated, int(layer_idx))
                        layer_kl, layer_ce = _masked_kl_and_ce(
                            student_logits=student_logits,
                            teacher_logits=teacher_logits,
                            target_ids=target_ids,
                            token_mask=token_mask,
                            temperature=temperature,
                        )
                    total_kl[int(layer_idx)] += float(layer_kl.item()) * n_tokens
                    total_ce[int(layer_idx)] += float(layer_ce.item()) * n_tokens
                    total_tokens[int(layer_idx)] += n_tokens
                    for lang, indices in batch_lang_to_indices.items():
                        lang_teacher_logits = teacher_logits[indices]
                        lang_target_ids = target_ids[indices]
                        lang_token_mask = token_mask[indices]
                        lang_tokens = int(lang_token_mask.sum().item())
                        if lang_tokens == 0:
                            continue
                        lang_student_logits = student_logits[indices]
                        lang_kl, lang_ce = _masked_kl_and_ce(
                            student_logits=lang_student_logits,
                            teacher_logits=lang_teacher_logits,
                            target_ids=lang_target_ids,
                            token_mask=lang_token_mask,
                            temperature=temperature,
                        )
                        total_kl_by_lang[int(layer_idx)][lang] = (
                            total_kl_by_lang[int(layer_idx)].get(lang, 0.0)
                            + float(lang_kl.item()) * lang_tokens
                        )
                        total_ce_by_lang[int(layer_idx)][lang] = (
                            total_ce_by_lang[int(layer_idx)].get(lang, 0.0)
                            + float(lang_ce.item()) * lang_tokens
                        )
                        total_tokens_by_lang[int(layer_idx)][lang] = (
                            total_tokens_by_lang[int(layer_idx)].get(lang, 0) + lang_tokens
                        )
                        del lang_student_logits
                        del lang_teacher_logits
                        del lang_target_ids
                        del lang_token_mask
                        del lang_kl
                        del lang_ce
                    del hidden
                    del translated
                    del student_logits
                    del layer_kl
                    del layer_ce
            del teacher_logits
            del target_ids
            del token_mask
            del outputs
            del input_ids
            del attention_mask
            if val_pbar is not None:
                val_pbar.update(1)
                val_pbar.set_postfix(tokens=n_tokens)
    if val_pbar is not None:
        val_pbar.close()

    metrics = {}
    for layer_idx in layer_indices:
        denom = max(total_tokens[int(layer_idx)], 1)
        metrics[int(layer_idx)] = {
            "kl": total_kl[int(layer_idx)] / denom,
            "ce": total_ce[int(layer_idx)] / denom,
            "n_tokens": total_tokens[int(layer_idx)],
        }
    mean_kl_by_lang: dict[str, float] = {}
    mean_ce_by_lang: dict[str, float] = {}
    langs_present = sorted(
        {lang for layer_map in total_tokens_by_lang.values() for lang in layer_map}
    )
    for lang in langs_present:
        layer_kl_values = {}
        layer_ce_values = {}
        for layer_idx in layer_indices:
            lang_tokens = total_tokens_by_lang[int(layer_idx)].get(lang, 0)
            if lang_tokens <= 0:
                continue
            layer_kl_values[int(layer_idx)] = total_kl_by_lang[int(layer_idx)][lang] / lang_tokens
            layer_ce_values[int(layer_idx)] = total_ce_by_lang[int(layer_idx)][lang] / lang_tokens
        if layer_kl_values:
            mean_kl_by_lang[lang] = _mean_across_layers(layer_kl_values)
            mean_ce_by_lang[lang] = _mean_across_layers(layer_ce_values)
    return {
        "by_layer": metrics,
        "mean_kl_by_lang": mean_kl_by_lang,
        "mean_ce_by_lang": mean_ce_by_lang,
    }


################################################################################
# Tracking and artifact output
################################################################################


def _maybe_init_wandb(args, training_config: dict[str, Any], out_root: Path | None):
    """Initialize optional W&B tracking for one fitting run."""
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging requested but wandb is not installed.") from exc

    run_name = args.wandb_run_name or f"tunedlens-{args.model_name.split('/')[-1]}-{args.revision}"
    tags = list(args.wandb_tags or [])
    if "tuned_lens" not in tags:
        tags.append("tuned_lens")
    config_payload = dict(training_config)
    config_payload.update(
        {
            "wandb_enabled": True,
            "out_dir": str(out_root) if out_root is not None else args.out_dir,
        }
    )
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=tags,
        config=config_payload,
    )


def _resolve_tuned_lens_out_dir(dataset_source: str, out_dir: str | None) -> Path:
    """Resolve an explicit or dataset-specific artifact root."""
    if out_dir not in {None, "", "none"}:
        return Path(str(out_dir))
    return DEFAULT_TUNED_LENS_OUT_DIR_BY_SOURCE[str(dataset_source)]


def _resolve_fineweb_window_cache_dir(
    dataset_dir: Path,
    model_name: str,
    revision: str,
    max_length: int,
) -> Path:
    """Build the model & window size specific cache path for FineWeb token windows."""
    return (
        Path(dataset_dir)
        / DEFAULT_FINEWEB_WINDOW_CACHE_SUBDIR
        / sanitize_model_id(model_name)
        / str(revision)
        / f"maxlen_{int(max_length)}"
    )


def _compute_periodic_steps(total_steps: int, every_steps: int) -> list[int]:
    """Schedule periodic validation or checkpoint steps, including the final step."""
    if total_steps <= 0 or every_steps <= 0:
        return []
    out = set(range(int(every_steps), total_steps + 1, int(every_steps)))
    out.add(int(total_steps))
    return sorted(out)


def _write_tuned_lens_snapshot(
    out_root: Path,
    *,
    layer_indices: Sequence[int],
    hidden_size: int,
    model_name: str,
    training_config_hash: str,
    validation_kl_by_layer: dict[int, float],
    validation_ce_by_layer: dict[int, float],
    state_by_layer: dict[int, dict[str, torch.Tensor]],
    config_payload: dict[str, Any],
) -> None:
    """Write one complete tuned-lens snapshot and its resolved configuration."""
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for layer_idx in layer_indices:
        out_path = out_root / tuned_lens_layer_filename(int(layer_idx))
        torch.save(
            {
                "A": state_by_layer[int(layer_idx)]["A"],
                "b": state_by_layer[int(layer_idx)]["b"],
                "layer_idx": int(layer_idx),
                "hidden_size": int(hidden_size),
                "model_id": model_name,
                "training_config_hash": training_config_hash,
                "validation_kl": float(validation_kl_by_layer[int(layer_idx)]),
                "validation_ce": float(validation_ce_by_layer[int(layer_idx)]),
            },
            out_path,
        )


def _build_training_config(
    *,
    args: argparse.Namespace,
    fitting_data: PreparedFittingData,
    layer_indices: Sequence[int],
    requested_layer_indices: Sequence[int],
    skipped_final_layers: Sequence[int],
    final_layer_idx: int,
    hidden_size: int,
) -> dict[str, Any]:
    """Build the resolved configuration stored with every snapshot."""
    return {
        "model_name_or_path": args.model_name,
        "revision": args.revision,
        "layers": [int(layer_idx) for layer_idx in layer_indices],
        "requested_layers": [int(layer_idx) for layer_idx in requested_layer_indices],
        "skipped_final_layers": list(skipped_final_layers),
        "final_layer_idx": int(final_layer_idx),
        "final_layer_policy": "raw_model_head",
        "hidden_size": hidden_size,
        "dataset_source": str(args.dataset_source),
        "calibration_data": str(fitting_data.calibration_data),
        "dataset_train_split": (
            DEFAULT_FINEWEB_TRAIN_SPLIT if fitting_data.uses_fineweb else None
        ),
        "dataset_val_split": (
            DEFAULT_FINEWEB_VAL_SPLIT if fitting_data.uses_fineweb else None
        ),
        "languages": list(args.languages),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "max_train_steps": int(args.max_train_steps),
        "temperature": float(args.temperature),
        "val_split": float(args.val_split),
        "identity_reg_weight": float(args.identity_reg_weight),
        "bias_reg_weight": float(args.bias_reg_weight),
        "early_stopping_patience": int(args.early_stopping_patience),
        "light_val_examples_per_lang": int(args.light_val_examples_per_lang),
        "train_balance_policy": "strict_round_robin_drop_remainder",
        "train_examples_per_lang": int(fitting_data.train_examples_per_lang),
        "train_rows_per_lang_before_balance": {
            str(lang): int(fitting_data.train_rows_per_lang_before_balance[str(lang)])
            for lang in args.languages
        },
        "val_every_steps": int(args.val_every_steps),
        "save_every_steps": int(args.save_every_steps),
        "apply_final_norm": bool(args.apply_final_norm),
        "max_length": fitting_data.effective_max_length,
        "layers_per_step": int(args.layers_per_step),
        "amp_dtype": args.amp_dtype,
        "wandb_project": args.wandb_project if args.wandb else None,
    }


def _write_final_artifacts(
    *,
    args: argparse.Namespace,
    fitting_data: PreparedFittingData,
    final_out_root: Path,
    layer_indices: Sequence[int],
    hidden_size: int,
    training_config: dict[str, Any],
    training_config_hash: str,
    metrics_rows: Sequence[dict[str, object]],
    best_layer_kl: dict[int, float],
    best_layer_ce: dict[int, float],
    best_layer_state: dict[int, dict[str, torch.Tensor]],
    checkpoint_steps: Sequence[int],
) -> None:
    """Write final metrics and the best per-layer translator snapshot."""
    final_out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = final_out_root / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        for row in metrics_rows:
            f.write(json.dumps(row) + "\n")

    validation_kl_by_layer = {
        int(layer_idx): float(best_layer_kl[int(layer_idx)])
        for layer_idx in layer_indices
    }
    validation_ce_by_layer = {
        int(layer_idx): float(best_layer_ce[int(layer_idx)])
        for layer_idx in layer_indices
    }
    config_payload = {
        **training_config,
        "training_config_hash": training_config_hash,
        "validation_kl_by_layer": validation_kl_by_layer,
        "validation_ce_by_layer": validation_ce_by_layer,
        "n_train_examples": fitting_data.n_train_examples,
        "n_val_examples": fitting_data.n_val_examples,
        "final_transform_mode": "apply_final_norm" if args.apply_final_norm else "none",
        "wandb_enabled": bool(args.wandb),
        "best_snapshot_type": "best",
        "checkpoint_steps": sorted(checkpoint_steps),
    }
    _write_tuned_lens_snapshot(
        final_out_root,
        layer_indices=layer_indices,
        hidden_size=hidden_size,
        model_name=args.model_name,
        training_config_hash=training_config_hash,
        validation_kl_by_layer=validation_kl_by_layer,
        validation_ce_by_layer=validation_ce_by_layer,
        state_by_layer=best_layer_state,
        config_payload=config_payload,
    )


################################################################################
# Training
################################################################################

def _fit_tuned_lens(args: argparse.Namespace) -> None:
    """Run one tuned-lens fitting workflow from resolved CLI arguments."""
    device = torch.device(args.device)
    dtype = None if args.dtype in {None, "", "none"} else getattr(torch, str(args.dtype))

    model = load_nnsight_model(
        model_name=args.model_name,
        revision=args.revision,
        device=args.device,
        seed=args.seed,
        dtype=dtype,
    )
    tokenizer = model.tokenizer
    causal_lm = _get_hf_causal_lm(model)
    amp_dtype = _resolve_amp_dtype(device, args.amp_dtype)
    for param in causal_lm.parameters():
        param.requires_grad_(False)
    causal_lm.eval()
    fitting_data = _prepare_fitting_data(args=args, model=model, tokenizer=tokenizer)

    requested_layer_indices, layer_indices, final_layer_idx = _resolve_trainable_layers(model, args.layers)
    skipped_final_layers = [
        int(layer_idx)
        for layer_idx in requested_layer_indices
        if int(layer_idx) == int(final_layer_idx)
    ]
    if skipped_final_layers:
        print(
            "Skipping final transformer layer for tuned-lens training; "
            "runtime tuned_lens uses raw model-head projection for that layer."
        )
    hf_model = _get_hf_causal_lm(model)
    hidden_size = _resolve_hidden_size(hf_model)
    translators = TranslatorBank(hidden_size=hidden_size, layer_indices=layer_indices).to(device)
    optimizer = torch.optim.AdamW(translators.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    unembed_info = prepare_unembed_info(model, device=device)
    eye = torch.eye(hidden_size, device=device)

    training_config = _build_training_config(
        args=args,
        fitting_data=fitting_data,
        layer_indices=layer_indices,
        requested_layer_indices=requested_layer_indices,
        skipped_final_layers=skipped_final_layers,
        final_layer_idx=final_layer_idx,
        hidden_size=hidden_size,
    )
    training_config_hash = hash_training_config(training_config)
    resolved_out_dir = _resolve_tuned_lens_out_dir(args.dataset_source, args.out_dir)
    tuned_lens_base_dir = build_tuned_lens_base_dir(
        args.model_name,
        args.revision,
        root=resolved_out_dir,
    )
    final_out_root = next_tuned_lens_run_dir(tuned_lens_base_dir)
    wandb_run = _maybe_init_wandb(args, training_config, out_root=resolved_out_dir)

    metrics_rows: list[dict[str, object]] = []
    best_layer_kl = {int(layer_idx): float("inf") for layer_idx in layer_indices}
    best_layer_ce = {int(layer_idx): float("inf") for layer_idx in layer_indices}
    best_layer_state = {
        int(layer_idx): {
            "A": translators.layers[str(int(layer_idx))].weight.detach().cpu().clone(),
            "b": translators.layers[str(int(layer_idx))].bias.detach().cpu().clone(),
        }
        for layer_idx in layer_indices
    }
    best_mean_val_kl = float("inf")
    validations_without_improvement = 0
    num_batches_per_epoch = len(
        list(_iter_batches(fitting_data.train_rows, int(args.batch_size)))
    )
    layer_chunks = list(_iter_layer_chunks(layer_indices, int(args.layers_per_step)))
    num_layer_chunks = len(layer_chunks)
    total_train_steps = min(int(args.max_train_steps), int(args.num_epochs) * num_batches_per_epoch)
    planned_train_examples = int(total_train_steps) * int(args.batch_size)
    if planned_train_examples < len(args.languages):
        print(
            "WARNING: tuned-lens training will see fewer examples than languages "
            f"({planned_train_examples} examples across {len(args.languages)} languages). "
            "Increase --max-train-steps or --batch-size for complete language coverage."
        )
    else:
        print(
            f"Tuned-lens training exposure is balanced over the first {planned_train_examples} "
            f"examples ({len(args.languages)} languages; batch_size={int(args.batch_size)})."
        )
    validation_steps = set(_compute_periodic_steps(total_train_steps, int(args.val_every_steps)))
    checkpoint_steps = set(_compute_periodic_steps(total_train_steps, int(args.save_every_steps)))
    global_step = 0
    stop_training = False

    raw_lens = RawLogitLens(unembed_info=unembed_info, apply_final_norm=bool(args.apply_final_norm))

    try:
        for epoch in range(int(args.num_epochs)):
            translators.train()
            running_train_kl = {int(layer_idx): 0.0 for layer_idx in layer_indices}
            running_train_tokens = {int(layer_idx): 0 for layer_idx in layer_indices}
            running_reg_loss = 0.0
            running_total_loss = 0.0
            batch_count = 0
            remaining_steps = max(total_train_steps - global_step, 0)
            epoch_pbar = tqdm(
                total=max(min(num_batches_per_epoch, remaining_steps), 1),
                desc=f"Epoch {epoch+1}",
                unit="batch",
            )

            for batch in _iter_batches(fitting_data.train_rows, int(args.batch_size)):
                if global_step >= total_train_steps:
                    stop_training = True
                    break
                if fitting_data.rows_contain_token_ids:
                    batch_rows = batch
                    input_ids, attention_mask = _build_padded_token_batch(
                        batch_rows,
                        pad_token_id=int(fitting_data.pad_token_id),
                        device=device,
                    )
                else:
                    batch_rows = batch
                    prompt_texts = [str(row["text"]) for row in batch_rows]
                    encoded = tokenizer(
                        prompt_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=fitting_data.effective_max_length is not None,
                        max_length=fitting_data.effective_max_length,
                        add_special_tokens=True,
                    )
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded["attention_mask"].to(device)

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
                valid_token_count = int(token_mask.sum().item())
                batch_total_loss_value = 0.0
                batch_reg_loss_value = 0.0
                batch_train_kl_sum = 0.0
                batch_train_kl_by_lang = {str(lang): {} for lang in args.languages}
                batch_train_kl_by_layer = {}
                batch_lang_to_indices = _lang_index_map(_extract_batch_langs(batch_rows))
                n_layers = max(len(layer_indices), 1)
                chunk_pbar = tqdm(
                    total=max(num_layer_chunks, 1),
                    desc=f"Epoch {epoch+1} batch {batch_count + 1} layers",
                    unit="chunk",
                    leave=False,
                )
                for chunk_idx, layer_chunk in enumerate(layer_chunks, start=1):
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
                                temperature=float(args.temperature),
                            )
                        chunk_loss = chunk_loss + (layer_kl / n_layers)
                        batch_train_kl_sum += float(layer_kl.item())
                        batch_train_kl_by_layer[int(layer_idx)] = float(layer_kl.item())
                        running_train_kl[int(layer_idx)] += float(layer_kl.item()) * valid_token_count
                        running_train_tokens[int(layer_idx)] += valid_token_count
                        for lang, indices in batch_lang_to_indices.items():
                            lang_student_logits = student_logits[indices]
                            lang_teacher_logits = teacher_logits[indices]
                            lang_target_ids = target_ids[indices]
                            lang_token_mask = token_mask[indices]
                            lang_tokens = int(lang_token_mask.sum().item())
                            if lang_tokens == 0:
                                continue
                            lang_kl, _ = _masked_kl_and_ce(
                                student_logits=lang_student_logits,
                                teacher_logits=lang_teacher_logits,
                                target_ids=lang_target_ids,
                                token_mask=lang_token_mask,
                                temperature=float(args.temperature),
                            )
                            batch_train_kl_by_lang[str(lang)][int(layer_idx)] = float(lang_kl.item())
                            del lang_student_logits
                            del lang_teacher_logits
                            del lang_target_ids
                            del lang_token_mask
                            del lang_kl
                        del hidden
                        del translated
                        del student_logits
                        del layer_kl
                    chunk_reg_loss = _layer_chunk_reg_loss(
                        translators=translators,
                        eye=eye,
                        layer_chunk=layer_chunk,
                        identity_reg_weight=float(args.identity_reg_weight),
                        bias_reg_weight=float(args.bias_reg_weight),
                        device=device,
                    )
                    total_chunk_loss = chunk_loss + chunk_reg_loss
                    total_chunk_loss.backward()
                    batch_total_loss_value += float(total_chunk_loss.item())
                    batch_reg_loss_value += float(chunk_reg_loss.item())
                    del chunk_reg_loss
                    del chunk_loss
                    del total_chunk_loss
                    chunk_pbar.update(1)
                    chunk_pbar.set_postfix(tokens=valid_token_count, chunk=f"{chunk_idx}/{num_layer_chunks}")
                chunk_pbar.close()
                optimizer.step()
                global_step += 1

                running_reg_loss += batch_reg_loss_value
                running_total_loss += batch_total_loss_value
                batch_count += 1
                epoch_pbar.update(1)
                epoch_pbar.set_postfix(tokens=valid_token_count, mean_kl=f"{(batch_train_kl_sum / n_layers):.4f}")
                if wandb_run is not None:
                    train_log = {
                        "epoch": epoch + 1,
                        "global_step": int(global_step),
                        "train/mean_kl": batch_train_kl_sum / n_layers,
                        "train/total_loss": batch_total_loss_value,
                        "train/reg_loss": batch_reg_loss_value,
                        "train/tokens": valid_token_count,
                    }
                    for layer_idx in layer_indices:
                        if int(layer_idx) in batch_train_kl_by_layer:
                            train_log[f"train/layer_{int(layer_idx):02d}_kl"] = batch_train_kl_by_layer[int(layer_idx)]
                    for lang, values_by_layer in batch_train_kl_by_lang.items():
                        if values_by_layer:
                            train_log[f"train/mean_kl_by_lang/{lang}"] = _mean_across_layers(values_by_layer)
                    wandb_run.log(
                        train_log,
                        step=int(global_step),
                    )
                del teacher_logits
                del target_ids
                del token_mask
                del outputs
                del input_ids
                del attention_mask

                if global_step in validation_steps:
                    translators.eval()
                    print(
                        f"Running lightweight validation at step {global_step} "
                        f"with {fitting_data.effective_light_val_examples} examples per language."
                    )
                    val_metrics = _evaluate_rows(
                        rows=fitting_data.light_validation_rows,
                        rows_contain_token_ids=fitting_data.rows_contain_token_ids,
                        batch_size=int(args.batch_size),
                        tokenizer=tokenizer,
                        pad_token_id=fitting_data.pad_token_id,
                        causal_lm=causal_lm,
                        translators=translators,
                        unembed_info=unembed_info,
                        layer_indices=layer_indices,
                        temperature=float(args.temperature),
                        apply_final_norm=bool(args.apply_final_norm),
                        device=device,
                        max_length=fitting_data.effective_max_length,
                        amp_dtype=amp_dtype,
                        layers_per_step=int(args.layers_per_step),
                        progress_desc=f"Validation step {global_step}",
                    )
                    val_layer_metrics = val_metrics["by_layer"]
                    mean_val_kl = float(
                        sum(
                            val_layer_metrics[layer_idx]["kl"]
                            for layer_idx in layer_indices
                        )
                        / max(len(layer_indices), 1)
                    )
                    improved = mean_val_kl < best_mean_val_kl - 1e-8
                    if improved:
                        best_mean_val_kl = mean_val_kl
                        validations_without_improvement = 0
                    else:
                        validations_without_improvement += 1

                    checkpoint_log: dict[str, Any] = {
                        "epoch": epoch + 1,
                        "global_step": int(global_step),
                        "train/mean_total_loss": running_total_loss / max(batch_count, 1),
                        "train/mean_reg_loss": running_reg_loss / max(batch_count, 1),
                        "val/mean_kl": mean_val_kl,
                        "train/n_examples": fitting_data.n_train_examples,
                        "val/n_examples": fitting_data.n_val_examples,
                        "early_stop/validations_without_improvement": validations_without_improvement,
                        "early_stop/improved": int(improved),
                    }
                    current_state = {
                        int(layer_idx): {
                            "A": translators.layers[str(int(layer_idx))].weight.detach().cpu().clone(),
                            "b": translators.layers[str(int(layer_idx))].bias.detach().cpu().clone(),
                        }
                        for layer_idx in layer_indices
                    }
                    validation_kl_by_layer = {}
                    validation_ce_by_layer = {}
                    for layer_idx in layer_indices:
                        train_tokens = max(running_train_tokens[int(layer_idx)], 1)
                        train_kl = running_train_kl[int(layer_idx)] / train_tokens
                        val_kl = float(val_layer_metrics[int(layer_idx)]["kl"])
                        val_ce = float(val_layer_metrics[int(layer_idx)]["ce"])
                        validation_kl_by_layer[int(layer_idx)] = val_kl
                        validation_ce_by_layer[int(layer_idx)] = val_ce
                        if val_kl < best_layer_kl[int(layer_idx)]:
                            best_layer_kl[int(layer_idx)] = val_kl
                            best_layer_ce[int(layer_idx)] = val_ce
                            best_layer_state[int(layer_idx)] = {
                                "A": current_state[int(layer_idx)]["A"].clone(),
                                "b": current_state[int(layer_idx)]["b"].clone(),
                            }
                        metrics_rows.append(
                            {
                                "epoch": epoch + 1,
                                "global_step": int(global_step),
                                "layer": int(layer_idx),
                                "train_kl": train_kl,
                                "val_kl": val_kl,
                                "val_ce": val_ce,
                                "n_train_tokens": int(running_train_tokens[int(layer_idx)]),
                                "n_val_tokens": int(val_layer_metrics[int(layer_idx)]["n_tokens"]),
                                "early_stopped": False,
                            }
                        )
                        checkpoint_log[f"train/layer_{int(layer_idx):02d}_kl"] = train_kl
                        checkpoint_log[f"val/layer_{int(layer_idx):02d}_kl"] = val_kl
                        checkpoint_log[f"val/layer_{int(layer_idx):02d}_ce"] = val_ce
                        checkpoint_log[f"tokens/layer_{int(layer_idx):02d}_train"] = int(
                            running_train_tokens[int(layer_idx)]
                        )
                        checkpoint_log[f"tokens/layer_{int(layer_idx):02d}_val"] = int(
                            val_layer_metrics[int(layer_idx)]["n_tokens"]
                        )
                    for lang, lang_mean_kl in val_metrics["mean_kl_by_lang"].items():
                        checkpoint_log[f"val/mean_kl_by_lang/{lang}"] = float(lang_mean_kl)

                    if wandb_run is not None:
                        wandb_run.log(checkpoint_log, step=int(global_step))

                    if global_step in checkpoint_steps:
                        print(f"Saving checkpoint at step {global_step}.")
                        checkpoint_root = final_out_root / "checkpoints" / f"step_{global_step:06d}"
                        checkpoint_config_payload = {
                            **training_config,
                            "training_config_hash": training_config_hash,
                            "validation_kl_by_layer": validation_kl_by_layer,
                            "validation_ce_by_layer": validation_ce_by_layer,
                            "n_train_examples": fitting_data.n_train_examples,
                            "n_val_examples": fitting_data.n_val_examples,
                            "final_transform_mode": "apply_final_norm" if args.apply_final_norm else "none",
                            "wandb_enabled": bool(args.wandb),
                            "checkpoint_epoch": int(epoch + 1),
                            "checkpoint_step": int(global_step),
                            "checkpoint_type": "periodic_step",
                        }
                        _write_tuned_lens_snapshot(
                            checkpoint_root,
                            layer_indices=layer_indices,
                            hidden_size=hidden_size,
                            model_name=args.model_name,
                            training_config_hash=training_config_hash,
                            validation_kl_by_layer=validation_kl_by_layer,
                            validation_ce_by_layer=validation_ce_by_layer,
                            state_by_layer=current_state,
                            config_payload=checkpoint_config_payload,
                        )
                    translators.train()

                    if validations_without_improvement >= int(args.early_stopping_patience):
                        stop_training = True
                        break
                if global_step >= total_train_steps:
                    stop_training = True
                    break
            epoch_pbar.close()
            if stop_training:
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    if metrics_rows and validations_without_improvement >= int(args.early_stopping_patience):
        last_global_step = max(int(row["global_step"]) for row in metrics_rows)
        for row in metrics_rows:
            if int(row["global_step"]) == last_global_step:
                row["early_stopped"] = True
    _write_final_artifacts(
        args=args,
        fitting_data=fitting_data,
        final_out_root=final_out_root,
        layer_indices=layer_indices,
        hidden_size=hidden_size,
        training_config=training_config,
        training_config_hash=training_config_hash,
        metrics_rows=metrics_rows,
        best_layer_kl=best_layer_kl,
        best_layer_ce=best_layer_ce,
        best_layer_state=best_layer_state,
        checkpoint_steps=checkpoint_steps,
    )

    print(f"Saved tuned lens artifacts to {final_out_root}")


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    _validate_args(args)
    _fit_tuned_lens(args)


if __name__ == "__main__":
    main()
