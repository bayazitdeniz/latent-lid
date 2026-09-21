#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from wendler_extend_utils import EXTENDED_DIR, default_seed_path, read_csv, write_csv


FIELDNAMES = [
    "concept_id",
    "word_original",
    "zh_word_translation",
    "blank_prompt_original",
    "zh_blank_prompt_translation",
    "zh_blank_prompt_translation_masked",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the 139-concept Chinese seed table.")
    parser.add_argument("--seed-csv", type=Path, default=default_seed_path())
    parser.add_argument("--out-dir", type=Path, default=EXTENDED_DIR)
    return parser.parse_args()


def build_seed_rows(seed_csv: Path) -> list[dict[str, str]]:
    rows = read_csv(seed_csv)
    seen: set[str] = set()
    output: list[dict[str, str]] = []
    for row in rows:
        concept = row["word_original"].strip()
        if concept in seen:
            continue
        seen.add(concept)
        output.append(
            {
                "concept_id": concept,
                "word_original": concept,
                "zh_word_translation": row.get("word_translation", "").strip(),
                "blank_prompt_original": row.get("blank_prompt_original", "").strip(),
                "zh_blank_prompt_translation": row.get("blank_prompt_translation", "").strip(),
                "zh_blank_prompt_translation_masked": row.get(
                    "blank_prompt_translation_masked", ""
                ).strip(),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    rows = build_seed_rows(args.seed_csv)
    if len(rows) != 139:
        raise ValueError(f"Expected 139 unique Chinese seed concepts, found {len(rows)}")
    out_path = args.out_dir / "seed_concepts.csv"
    write_csv(out_path, rows, FIELDNAMES)
    print(f"Wrote {len(rows)} seed concepts to {out_path}")


if __name__ == "__main__":
    main()
