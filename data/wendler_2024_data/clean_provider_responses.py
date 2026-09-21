#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from wendler_extend_utils import (
    EXTENDED_DIR,
    FILTER_MODES,
    PROVIDER_DIR,
    TARGET_LANGUAGES,
    has_token_prefix,
    mask_cloze,
    normalize_space,
    read_json,
    write_csv,
    write_json,
)


RAW_FIELDNAMES = [
    "source_file",
    "source_row",
    "provider",
    "chunk_id",
    "lang",
    "concept_id",
    "word_original",
    "word_translation",
    "blank_prompt_original",
    "blank_prompt_translation",
    "blank_prompt_translation_masked",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse and clean raw provider responses.")
    parser.add_argument("--responses-dir", type=Path, default=PROVIDER_DIR / "deepl_responses")
    parser.add_argument("--out-dir", type=Path, default=EXTENDED_DIR)
    parser.add_argument("--default-filter-mode", choices=FILTER_MODES, default="none")
    return parser.parse_args()


def translations_from_response(data: dict[str, Any]) -> list[str]:
    response = data.get("response", {})
    translations = response.get("translations", []) if isinstance(response, dict) else []
    return [str(item.get("text", "")) for item in translations if isinstance(item, dict)]


def parse_raw_rows(responses_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(responses_dir.glob("*.json")):
        data = read_json(path)
        request = data.get("request", {})
        items = request.get("items", [])
        texts = translations_from_response(data)
        if len(texts) != len(items):
            rows.append(
                {
                    "source_file": str(path),
                    "source_row": "",
                    "provider": str(data.get("provider", "deepl")),
                    "chunk_id": str(data.get("chunk_id", path.stem)),
                    "lang": str(request.get("lang", "")),
                    "concept_id": "",
                    "word_original": "",
                    "word_translation": "",
                    "blank_prompt_original": "",
                    "blank_prompt_translation": "",
                    "blank_prompt_translation_masked": "",
                    "error": f"translation_count_mismatch:{len(texts)}!={len(items)}",
                }
            )
            continue
        grouped: dict[str, dict[str, str]] = {}
        request_texts = [str(text) for text in request.get("texts", [])]
        source_by_concept: dict[str, str] = {}
        cloze_by_concept: dict[str, str] = {}
        for item_index, item in enumerate(items):
            original_text = (
                str(item.get("text", ""))
                if isinstance(item, dict) and item.get("text")
                else request_texts[item_index]
                if item_index < len(request_texts)
                else ""
            )
            concept_id = str(item.get("concept_id", ""))
            field = str(item.get("field", ""))
            if field == "word_original":
                source_by_concept[concept_id] = original_text
            elif field == "blank_prompt_original":
                cloze_by_concept[concept_id] = original_text
        for item, translated in zip(items, texts):
            concept_id = str(item.get("concept_id", ""))
            field = str(item.get("field", ""))
            grouped.setdefault(
                concept_id,
                {
                    "source_file": str(path),
                    "source_row": "",
                    "provider": str(data.get("provider", "deepl")),
                    "chunk_id": str(data.get("chunk_id", path.stem)),
                    "lang": str(request.get("lang", "")),
                    "concept_id": concept_id,
                    "word_original": source_by_concept.get(concept_id, concept_id),
                    "word_translation": "",
                    "blank_prompt_original": "",
                    "blank_prompt_translation": "",
                    "blank_prompt_translation_masked": "",
                    "error": "",
                },
            )
            if field == "word_original":
                grouped[concept_id]["word_translation"] = normalize_space(translated)
            elif field == "blank_prompt_original":
                grouped[concept_id]["blank_prompt_original"] = cloze_by_concept.get(concept_id, "")
                grouped[concept_id]["blank_prompt_translation"] = normalize_space(translated)
        for row_idx, row in enumerate(grouped.values()):
            row["source_row"] = str(row_idx)
            row["blank_prompt_translation_masked"] = mask_cloze(
                row["blank_prompt_translation"], row["word_translation"]
            )
            rows.append(row)
    return rows


def rejection_reasons(row: dict[str, str], mode: str, by_concept: dict[str, list[dict[str, str]]]) -> list[str]:
    reasons: list[str] = []
    if row.get("error"):
        reasons.append(row["error"])
    if row.get("lang") not in TARGET_LANGUAGES:
        reasons.append("unknown_language")
    if not row.get("concept_id") or not row.get("word_translation") or not row.get("blank_prompt_translation"):
        reasons.append("missing_required_field")
    if "___" not in row.get("blank_prompt_translation_masked", ""):
        reasons.append("missing_mask")
    if (
        mode in {"not_match_english", "not_match_all"}
        and row.get("word_translation")
        and row["word_translation"].casefold() == row.get("word_original", "").casefold()
    ):
        reasons.append("translation_matches_english")
    if mode in {"not_match_english", "not_match_all"} and has_token_prefix(
        row.get("word_translation", ""), row.get("word_original", "")
    ):
        reasons.append("token_prefix_matches_english")
    if mode == "not_match_all":
        for other in by_concept.get(row.get("concept_id", ""), []):
            if other is row:
                continue
            if has_token_prefix(row.get("word_translation", ""), other.get("word_translation", "")):
                reasons.append("token_prefix_matches_other_language")
                break
    return reasons


def clean_rows(raw_rows: list[dict[str, str]], mode: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_concept: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        by_concept[row.get("concept_id", "")].append(row)
    duplicate_counts = Counter((row.get("lang", ""), row.get("word_translation", "")) for row in raw_rows)

    kept: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for row in raw_rows:
        reasons = rejection_reasons(row, mode, by_concept)
        if row.get("word_translation") and duplicate_counts[(row.get("lang", ""), row.get("word_translation", ""))] > 1:
            reasons.append("duplicate_translation_in_language")
        if reasons:
            rejected.append({**row, "filter_mode": mode, "rejection_reason": ";".join(sorted(set(reasons)))})
        else:
            kept.append(row)
    return kept, rejected


def main() -> None:
    args = parse_args()
    raw_rows = parse_raw_rows(args.responses_dir)
    write_csv(args.out_dir / "raw_text.csv", raw_rows, RAW_FIELDNAMES)
    write_csv(args.out_dir / "source_text.csv", raw_rows, RAW_FIELDNAMES)

    all_rejected: list[dict[str, str]] = []
    token_flags: list[dict[str, str]] = []
    summary: dict[str, object] = {
        "raw_rows": len(raw_rows),
        "filters": {},
    }
    for mode in FILTER_MODES:
        kept, rejected = clean_rows(raw_rows, mode)
        mode_dir = args.out_dir / f"filter_{mode}"
        write_csv(mode_dir / "source_clean.csv", kept, RAW_FIELDNAMES)
        all_rejected.extend(rejected)
        summary["filters"][mode] = {"kept": len(kept), "rejected": len(rejected)}
        for row in rejected:
            if "token_prefix" in row.get("rejection_reason", ""):
                token_flags.append(row)

    default_clean = args.out_dir / f"filter_{args.default_filter_mode}" / "source_clean.csv"
    default_rows = clean_rows(raw_rows, args.default_filter_mode)[0]
    write_csv(args.out_dir / "source_clean.csv", default_rows, RAW_FIELDNAMES)
    write_csv(
        args.out_dir / "rejected_rows.csv",
        all_rejected,
        RAW_FIELDNAMES + ["filter_mode", "rejection_reason"],
    )
    write_csv(
        args.out_dir / "token_overlap_flags.csv",
        token_flags,
        RAW_FIELDNAMES + ["filter_mode", "rejection_reason"],
    )
    write_json(args.out_dir / "validation_summary.json", summary)
    print(f"Wrote raw rows and cleaned tables to {args.out_dir}; default clean table: {default_clean}")


if __name__ == "__main__":
    main()
