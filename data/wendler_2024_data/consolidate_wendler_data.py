#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


DATASET_RE = re.compile(r"^(?P<model_size>[^_]+)_(?P<input_lang>[^_]+)_(?P<target_lang>[^_]+)_dataset\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate the imported Wendler et al. 2024 data into easier-to-read CSVs."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "llm-latent-language",
        help="Path to the imported llm-latent-language repo.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed",
        help="Directory for consolidated CSV outputs.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def drop_uninformative_error_column(
    rows: list[dict[str, object]], fieldnames: list[str]
) -> tuple[list[dict[str, object]], list[str]]:
    """Drop error when it is absent or only says that there is no error."""
    if "error" not in fieldnames:
        return rows, fieldnames
    error_values = {
        str(row.get("error", "")).strip()
        for row in rows
        if str(row.get("error", "")).strip()
    }
    if error_values and error_values != {"no error"}:
        return rows, fieldnames
    filtered_rows = [{key: value for key, value in row.items() if key != "error"} for row in rows]
    filtered_fieldnames = [field for field in fieldnames if field != "error"]
    return filtered_rows, filtered_fieldnames


def normalize_lang(path_lang: str, row: dict[str, str], source_path: Path) -> str:
    row_lang = row.get("lang", "")
    if row_lang and row_lang != path_lang:
        raise ValueError(f"Language mismatch in {source_path}: folder={path_lang}, row lang={row_lang}")
    return path_lang


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def collect_source_tables(repo_dir: Path, file_name: str, display_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    langs_dir = repo_dir / "data" / "langs"
    for path in sorted(langs_dir.glob(f"*/{file_name}")):
        path_lang = path.parent.name
        for row_idx, row in enumerate(read_csv(path)):
            lang = normalize_lang(path_lang, row, path)
            rows.append(
                {
                    "source_file": display_path(path, display_root),
                    "source_row": row_idx,
                    "lang": lang,
                    "concept_id": row.get("word_original", ""),
                    **row,
                }
            )
    return rows


def collect_cloze_rows(clean_rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in clean_rows:
        rows.append(
            {
                "source_file": row.get("source_file", ""),
                "source_row": row.get("source_row", ""),
                "lang": row.get("lang", ""),
                "concept_id": row.get("concept_id", ""),
                "target_text": row.get("word_translation", ""),
                "english_text": row.get("word_original", ""),
                "blank_prompt_original": row.get("blank_prompt_original", ""),
                "blank_prompt_translation": row.get("blank_prompt_translation", ""),
                "blank_prompt_translation_masked": row.get("blank_prompt_translation_masked", ""),
                "error": row.get("error", ""),
            }
        )
    return rows


def collect_generated_tasks(
    repo_dir: Path, display_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    translation_rows: list[dict[str, object]] = []
    copy_rows: list[dict[str, object]] = []
    inputs_dir = repo_dir / "R" / "data" / "inputs"
    for path in sorted(inputs_dir.glob("*_dataset.csv")):
        match = DATASET_RE.match(path.name)
        if match is None:
            continue
        meta = match.groupdict()
        input_lang = meta["input_lang"]
        target_lang = meta["target_lang"]
        task = "copy" if input_lang == target_lang else "translation"
        for row_idx, row in enumerate(read_csv(path)):
            out_row = {
                "source_file": display_path(path, display_root),
                "source_row": row_idx,
                "task": task,
                "model_size": meta["model_size"],
                "input_lang": input_lang,
                "target_lang": target_lang,
                "prompt": row.get("prompt", ""),
                "out_token_id": row.get("out_token_id", ""),
                "out_token_str": row.get("out_token_str", ""),
                "latent_token_id": row.get("latent_token_id", ""),
                "latent_token_str": row.get("latent_token_str", ""),
                "in_token_id": row.get("in_token_id", ""),
                "in_token_str": row.get("in_token_str", ""),
            }
            if task == "copy":
                copy_rows.append(out_row)
            else:
                translation_rows.append(out_row)
    return translation_rows, copy_rows


def summarize_outputs(out_dir: Path, outputs: dict[str, list[dict[str, object]]]) -> None:
    summary = {}
    for name, rows in outputs.items():
        entry: dict[str, object] = {"rows": len(rows)}
        langs = sorted({str(row.get("lang")) for row in rows if row.get("lang")})
        if langs:
            entry["languages"] = langs
        input_langs = sorted({str(row.get("input_lang")) for row in rows if row.get("input_lang")})
        target_langs = sorted({str(row.get("target_lang")) for row in rows if row.get("target_lang")})
        pairs = sorted(
            {
                f"{row.get('input_lang')}->{row.get('target_lang')}"
                for row in rows
                if row.get("input_lang") and row.get("target_lang")
            }
        )
        if input_langs:
            entry["input_languages"] = input_langs
        if target_langs:
            entry["target_languages"] = target_langs
        if pairs:
            entry["language_pairs"] = pairs
        summary[name] = entry
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_dir = args.repo_dir
    out_dir = args.out_dir

    if not repo_dir.exists():
        raise FileNotFoundError(f"Imported repo not found: {repo_dir}")

    display_root = repo_dir.parent
    text_rows = collect_source_tables(repo_dir, "text.csv", display_root)
    clean_rows = collect_source_tables(repo_dir, "clean.csv", display_root)
    cloze_rows = collect_cloze_rows(clean_rows)
    translation_rows, copy_rows = collect_generated_tasks(repo_dir, display_root)

    source_fieldnames = [
        "source_file",
        "source_row",
        "lang",
        "concept_id",
        "word_original",
        "word_translation",
        "blank_prompt_original",
        "blank_prompt_translation",
        "blank_prompt_translation_masked",
        "error",
    ]
    task_fieldnames = [
        "source_file",
        "source_row",
        "task",
        "model_size",
        "input_lang",
        "target_lang",
        "prompt",
        "out_token_id",
        "out_token_str",
        "latent_token_id",
        "latent_token_str",
        "in_token_id",
        "in_token_str",
    ]
    cloze_fieldnames = [
        "source_file",
        "source_row",
        "lang",
        "concept_id",
        "target_text",
        "english_text",
        "blank_prompt_original",
        "blank_prompt_translation",
        "blank_prompt_translation_masked",
        "error",
    ]

    outputs = {
        "source_text": text_rows,
        "source_clean": clean_rows,
        "translation": translation_rows,
        "copy": copy_rows,
        "cloze": cloze_rows,
    }
    text_rows, text_fieldnames = drop_uninformative_error_column(text_rows, source_fieldnames)
    clean_rows, clean_fieldnames = drop_uninformative_error_column(clean_rows, source_fieldnames)
    cloze_rows, cloze_fieldnames = drop_uninformative_error_column(cloze_rows, cloze_fieldnames)

    write_csv(out_dir / "source_text.csv", text_rows, text_fieldnames)
    write_csv(out_dir / "source_clean.csv", clean_rows, clean_fieldnames)
    write_csv(out_dir / "translation.csv", translation_rows, task_fieldnames)
    write_csv(out_dir / "copy.csv", copy_rows, task_fieldnames)
    write_csv(out_dir / "cloze.csv", cloze_rows, cloze_fieldnames)
    summarize_outputs(out_dir, outputs)

    for name, rows in outputs.items():
        print(f"{name}: {len(rows)} rows")
    print(f"Wrote consolidated outputs to: {out_dir}")


if __name__ == "__main__":
    main()
