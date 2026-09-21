#!/usr/bin/env python3
"""Build the balanced and uncapped INCLUDE evaluation subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import get_dataset_split_names, load_dataset


DEFAULT_DATASET_NAME = "CohereLabs/include-base-44"
DEFAULT_LANGS = ["ar", "es", "fi", "fr", "hi", "id", "pt", "ru", "tr", "zh"]
DEFAULT_DOMAINS = ["Social Science", "Arts & Humanities", "STEM"]
CHARACTER_SURFACE_TOKEN_LANGS = {"zh", "ja", "th"}

LANG_CONFIGS = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "hi": "Hindi",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "zh": "Chinese",
}


################################################################################
# CLI
################################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build an INCLUDE subset for LLID evaluation.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--langs", nargs="+", default=list(DEFAULT_LANGS), choices=sorted(LANG_CONFIGS))
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--splits", nargs="+", default=None, help="Dataset splits to include. Defaults to all available splits.")
    parser.add_argument(
        "--sampling",
        choices=["balanced_min", "cap", "all"],
        default="all",
        help=(
            "Row selection policy per (lang, domain): balanced_min downsamples every cell to the global minimum; "
            "cap keeps up to --max-per-cell per cell; all keeps every row."
        ),
    )
    parser.add_argument("--max-per-cell", type=int, default=None, help="Required for --sampling=cap; optional cap for balanced_min.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/include_eval_base_8lang_3domain")
    return parser.parse_args()


################################################################################
# Prompt records
################################################################################

def dataset_variant(dataset_name: str) -> str:
    return str(dataset_name).rsplit("/", 1)[-1]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return slug or "unknown"


def stable_json_hash(payload: Mapping[str, Any], *, n_chars: int = 16) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:n_chars]


def get_options(row: Mapping[str, Any]) -> list[str]:
    if "choices" in row and row.get("choices") is not None:
        choices = row.get("choices")
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except json.JSONDecodeError:
                choices = [choices]
        options = [str(choice) for choice in list(choices)]
    else:
        options = [
            str(row.get("option_a", "")),
            str(row.get("option_b", "")),
            str(row.get("option_c", "")),
            str(row.get("option_d", "")),
        ]
    if len(options) != 4 or any(option == "" for option in options):
        raise ValueError("INCLUDE row must have exactly four non-empty options.")
    return options


def render_question_only(row: Mapping[str, Any]) -> str:
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError("INCLUDE row is missing a non-empty question.")
    return question


def render_minimal_mcq(row: Mapping[str, Any]) -> str:
    question = render_question_only(row)
    options = get_options(row)
    labels = ["A", "B", "C", "D"]
    option_lines = [f"{label}. {option.strip()}" for label, option in zip(labels, options)]
    return question + "\n\n" + "\n".join(option_lines)


def simple_surface_tokens(text: str, *, lang: str | None = None) -> list[str]:
    if lang in CHARACTER_SURFACE_TOKEN_LANGS:
        return [char for char in str(text) if not char.isspace()]
    return str(text).split()


def row_hash_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    options = get_options(row)
    return {
        "dataset_name": row.get("dataset_name"),
        "hf_config": row.get("hf_config"),
        "source_split": row.get("source_split"),
        "source_row_idx": row.get("source_row_idx"),
        "language": row.get("language"),
        "domain": row.get("domain"),
        "subject": row.get("subject"),
        "question": row.get("question"),
        "options": options,
        "answer": row.get("answer"),
    }


def make_row_hash(row: Mapping[str, Any]) -> str:
    return stable_json_hash(row_hash_payload(row))


def make_prompt_id(row: Mapping[str, Any]) -> str:
    return "-".join(
        [
            slugify(str(row.get("dataset_variant") or dataset_variant(str(row.get("dataset_name"))))),
            str(row["lang"]),
            slugify(str(row["domain"])),
            str(row["row_hash"]),
        ]
    )


def prompt_payload(row: Mapping[str, Any], prompt_style: str) -> dict[str, Any]:
    if prompt_style == "question_only":
        prompt_text = render_question_only(row)
    elif prompt_style == "minimal_mcq":
        prompt_text = render_minimal_mcq(row)
    else:
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")
    return {
        "id": row["id"],
        "prompt_style": prompt_style,
        "prompt_text": prompt_text,
        "lang": row["lang"],
        "source_lang": row["lang"],
        "target_lang": row["lang"],
        "surface_tokens": simple_surface_tokens(prompt_text, lang=str(row["lang"])),
    }


def metadata_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    options = get_options(row)
    answer = row.get("answer")
    return {
        "id": row["id"],
        "row_hash": row["row_hash"],
        "dataset_name": row["dataset_name"],
        "dataset_variant": row["dataset_variant"],
        "hf_config": row["hf_config"],
        "source_split": row["source_split"],
        "source_row_idx": int(row["source_row_idx"]),
        "lang": row["lang"],
        "language": row.get("language"),
        "country": row.get("country"),
        "domain": row.get("domain"),
        "subject": row.get("subject"),
        "level": row.get("level"),
        "regional_feature": row.get("regional_feature"),
        "question": row.get("question"),
        "option_a": options[0],
        "option_b": options[1],
        "option_c": options[2],
        "option_d": options[3],
        "answer": None if pd.isna(answer) else int(answer),
    }


################################################################################
# Loading and sampling
################################################################################

def load_include_rows(
    *,
    dataset_name: str,
    langs: Sequence[str],
    splits: Sequence[str] | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    load_issues: list[dict[str, Any]] = []
    requested_splits = None if splits is None else set(str(split) for split in splits)
    for lang in langs:
        config_name = LANG_CONFIGS[lang]
        available_splits = get_dataset_split_names(dataset_name, config_name)
        split_names = [split for split in available_splits if requested_splits is None or split in requested_splits]
        for split in split_names:
            try:
                ds = load_dataset(dataset_name, config_name, split=split, verification_mode="no_checks")
            except ValueError as exc:
                if "corresponds to no data" not in str(exc):
                    raise
                load_issues.append(
                    {
                        "dataset_name": dataset_name,
                        "hf_config": config_name,
                        "lang": lang,
                        "split": split,
                        "issue": str(exc),
                    }
                )
                continue
            frame = ds.to_pandas()
            frame["dataset_name"] = dataset_name
            frame["dataset_variant"] = dataset_variant(dataset_name)
            frame["hf_config"] = config_name
            frame["source_split"] = split
            frame["source_row_idx"] = list(range(len(frame)))
            frame["lang"] = lang
            frames.append(frame)
    if not frames:
        raise RuntimeError("No INCLUDE rows loaded for the requested languages/splits.")
    return pd.concat(frames, ignore_index=True), load_issues


def filter_rows(frame: pd.DataFrame, *, langs: Sequence[str], domains: Sequence[str]) -> pd.DataFrame:
    filtered = frame[frame["lang"].isin(langs) & frame["domain"].isin(domains)].copy()
    if filtered.empty:
        raise ValueError("No INCLUDE rows remain after language/domain filtering.")
    return filtered


def _records_by_cell(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["lang"]), str(record["domain"]))].append(record)
    return grouped


def select_records_by_cell(
    records: Sequence[dict[str, Any]],
    *,
    langs: Sequence[str],
    domains: Sequence[str],
    seed: int,
    sampling: str,
    max_per_cell: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = _records_by_cell(records)
    expected_cells = [(lang, domain) for lang in langs for domain in domains]
    missing = [cell for cell in expected_cells if len(grouped.get(cell, [])) == 0]
    if missing:
        raise ValueError(f"Missing INCLUDE rows for requested (lang, domain) cells: {missing}")

    available_by_cell = {f"{lang}::{domain}": len(grouped[(lang, domain)]) for lang, domain in expected_cells}
    if sampling not in {"balanced_min", "cap", "all"}:
        raise ValueError(f"Unsupported sampling policy: {sampling}")
    if max_per_cell is not None:
        if int(max_per_cell) <= 0:
            raise ValueError("--max-per-cell must be positive when provided.")
    if sampling == "cap" and max_per_cell is None:
        raise ValueError("--max-per-cell is required when --sampling=cap.")

    target_per_cell = None
    if sampling == "balanced_min":
        target_per_cell = min(available_by_cell.values())
        if max_per_cell is not None:
            target_per_cell = min(target_per_cell, int(max_per_cell))

    selected: list[dict[str, Any]] = []
    selected_ids_by_cell: dict[str, list[str]] = {}
    for lang, domain in expected_cells:
        cell_key = f"{lang}::{domain}"
        cell_rows = sorted(grouped[(lang, domain)], key=lambda row: str(row["row_hash"]))
        rng = random.Random(f"{int(seed)}::{lang}::{domain}")
        rng.shuffle(cell_rows)
        if sampling == "all":
            n_keep = len(cell_rows)
        elif sampling == "cap":
            assert max_per_cell is not None
            n_keep = min(len(cell_rows), int(max_per_cell))
        else:
            assert target_per_cell is not None
            n_keep = int(target_per_cell)
        chosen = sorted(cell_rows[:n_keep], key=lambda row: str(row["id"]))
        selected.extend(chosen)
        selected_ids_by_cell[cell_key] = [str(row["id"]) for row in chosen]

    selected = sorted(selected, key=lambda row: str(row["id"]))
    stats = {
        "sampling": sampling,
        "target_per_cell": None if target_per_cell is None else int(target_per_cell),
        "max_per_cell": max_per_cell,
        "available_by_cell": available_by_cell,
        "selected_by_cell": {cell: len(ids) for cell, ids in selected_ids_by_cell.items()},
        "selected_ids_by_cell": selected_ids_by_cell,
    }
    return selected, stats


def balance_records(
    records: Sequence[dict[str, Any]],
    *,
    langs: Sequence[str],
    domains: Sequence[str],
    seed: int,
    max_per_cell: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Backward-compatible strict balancing helper used by older tests/scripts."""
    return select_records_by_cell(
        records,
        langs=langs,
        domains=domains,
        seed=seed,
        sampling="balanced_min",
        max_per_cell=max_per_cell,
    )


def prepare_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        # Validate option shape before the row can enter balancing.
        get_options(record)
        record["row_hash"] = make_row_hash(record)
        record["id"] = make_prompt_id(record)
        records.append(record)
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Stable INCLUDE ids are not unique; row hash/id construction needs adjustment.")
    return records


################################################################################
# Outputs
################################################################################

def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_metadata_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        pd.DataFrame(list(rows)).to_parquet(path, index=False)
    except Exception as exc:
        print(f"Warning: could not write parquet metadata to {path}: {type(exc).__name__}: {exc}")
        return False
    return True


def build_subset(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    frame, load_issues = load_include_rows(
        dataset_name=args.dataset_name,
        langs=list(args.langs),
        splits=args.splits,
    )
    filtered = filter_rows(frame, langs=list(args.langs), domains=list(args.domains))
    prepared = prepare_records(filtered)
    selected, balance_stats = select_records_by_cell(
        prepared,
        langs=list(args.langs),
        domains=list(args.domains),
        seed=int(args.seed),
        sampling=str(args.sampling),
        max_per_cell=args.max_per_cell,
    )

    metadata_rows = [metadata_payload(row) for row in selected]
    question_rows = [prompt_payload(row, "question_only") for row in selected]
    mcq_rows = [prompt_payload(row, "minimal_mcq") for row in selected]

    n_question = write_jsonl(out_dir / "prompts_question_only.jsonl", question_rows)
    n_mcq = write_jsonl(out_dir / "prompts_minimal_mcq.jsonl", mcq_rows)
    n_meta = write_jsonl(out_dir / "metadata.jsonl", metadata_rows)
    wrote_parquet = write_metadata_parquet(out_dir / "metadata.parquet", metadata_rows)

    stats = {
        "dataset_name": args.dataset_name,
        "dataset_variant": dataset_variant(args.dataset_name),
        "langs": list(args.langs),
        "domains": list(args.domains),
        "splits": None if args.splits is None else list(args.splits),
        "sampling": str(args.sampling),
        "seed": int(args.seed),
        "max_per_cell": args.max_per_cell,
        "n_loaded_rows": int(len(frame)),
        "n_filtered_rows": int(len(filtered)),
        "n_selected_rows": int(len(selected)),
        "n_prompts_question_only": int(n_question),
        "n_prompts_minimal_mcq": int(n_mcq),
        "n_metadata_rows": int(n_meta),
        "wrote_metadata_parquet": bool(wrote_parquet),
        "load_issues": load_issues,
        **balance_stats,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    args = parse_args()
    stats = build_subset(args)
    print(f"Saved INCLUDE subset to: {args.out_dir}")
    print(json.dumps({k: stats[k] for k in ["n_selected_rows", "sampling", "target_per_cell", "max_per_cell", "langs", "domains"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
