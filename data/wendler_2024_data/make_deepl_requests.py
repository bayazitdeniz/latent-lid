#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from wendler_extend_utils import EXTENDED_DIR, PROVIDER_DIR, TARGET_LANGUAGES, read_csv, stable_chunks, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create chunked DeepL request artifacts.")
    parser.add_argument("--seed-csv", type=Path, default=EXTENDED_DIR / "seed_concepts.csv")
    parser.add_argument("--out-dir", type=Path, default=PROVIDER_DIR / "deepl_requests")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--languages", nargs="*", default=sorted(TARGET_LANGUAGES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--write-test-requests",
        action="store_true",
        help="Also write a tiny request set for smoke-testing DeepL calls.",
    )
    parser.add_argument(
        "--test-out-dir",
        type=Path,
        default=PROVIDER_DIR / "deepl_test_requests",
        help="Directory for smoke-test request artifacts.",
    )
    parser.add_argument(
        "--test-languages",
        nargs="*",
        default=["ja", "tr"],
        help="Languages to include in the smoke-test request set.",
    )
    parser.add_argument(
        "--test-chunks-per-language",
        type=int,
        default=1,
        help="Number of chunks per smoke-test language.",
    )
    return parser.parse_args()


def request_record(lang: str, chunk_index: int, rows: list[dict[str, str]]) -> dict[str, object]:
    meta = TARGET_LANGUAGES[lang]
    texts: list[str] = []
    items: list[dict[str, str]] = []
    for row in rows:
        word_text = row["word_original"]
        cloze_text = row["blank_prompt_original"]
        texts.append(word_text)
        items.append(
            {
                "concept_id": row["concept_id"],
                "field": "word_original",
                "text": word_text,
                "context": cloze_text,
            }
        )
        texts.append(cloze_text)
        items.append(
            {
                "concept_id": row["concept_id"],
                "field": "blank_prompt_original",
                "text": cloze_text,
                "context": "",
            }
        )
    chunk_id = f"{lang}_{chunk_index:03d}"
    return {
        "chunk_id": chunk_id,
        "lang": lang,
        "target_lang": meta["deepl"],
        "source_lang": "EN",
        "request_version": "deepl_context_v1",
        "texts": texts,
        "items": items,
        "concept_ids": [row["concept_id"] for row in rows],
        "character_count": sum(len(text) for text in texts),
        "context_character_count": sum(len(row["blank_prompt_original"]) for row in rows),
    }


def build_records(
    seed_rows: list[dict[str, str]], languages: list[str], chunk_size: int
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for lang in languages:
        if lang not in TARGET_LANGUAGES:
            raise ValueError(f"Unknown target language: {lang}")
        for chunk_index, chunk in enumerate(stable_chunks(seed_rows, chunk_size)):
            records.append(request_record(lang, chunk_index, chunk))
    return records


def write_records(
    records: list[dict[str, object]],
    out_dir: Path,
    *,
    dry_run: bool,
    force: bool,
    languages: list[str],
    concepts: int,
    chunk_size: int,
    label: str,
) -> dict[str, object]:
    skipped = 0
    for record in records:
        out_path = out_dir / f"{record['chunk_id']}.json"
        if out_path.exists() and not force:
            skipped += 1
            continue
        if not dry_run:
            write_json(out_path, record)

    manifest = {
        "chunks": len(records),
        "languages": languages,
        "concepts": concepts,
        "chunk_size": chunk_size,
        "total_characters": sum(int(record["character_count"]) for record in records),
        "total_context_characters": sum(
            int(record.get("context_character_count", 0)) for record in records
        ),
        "request_version": "deepl_context_v1",
    }
    if not dry_run:
        write_json(out_dir / "manifest.json", manifest)
        write_jsonl(out_dir / "requests.jsonl", records)
    print(
        f"{'Would write' if dry_run else 'Wrote'} {len(records)} {label} chunks "
        f"({manifest['total_characters']} characters); skipped {skipped} existing chunks."
    )
    return manifest


def main() -> None:
    args = parse_args()
    seed_rows = read_csv(args.seed_csv)
    if len({row["concept_id"] for row in seed_rows}) != len(seed_rows):
        raise ValueError("Duplicate concept_id values found in seed CSV")

    records = build_records(seed_rows, args.languages, args.chunk_size)
    write_records(
        records,
        args.out_dir,
        dry_run=args.dry_run,
        force=args.force,
        languages=args.languages,
        concepts=len(seed_rows),
        chunk_size=args.chunk_size,
        label="DeepL request",
    )
    if args.write_test_requests:
        test_records = build_records(seed_rows, args.test_languages, args.chunk_size)
        selected_records: list[dict[str, object]] = []
        for lang in args.test_languages:
            selected_records.extend(
                record
                for record in test_records
                if record["lang"] == lang and int(str(record["chunk_id"]).rsplit("_", 1)[1]) < args.test_chunks_per_language
            )
        write_records(
            selected_records,
            args.test_out_dir,
            dry_run=args.dry_run,
            force=args.force,
            languages=args.test_languages,
            concepts=min(len(seed_rows), args.chunk_size * args.test_chunks_per_language),
            chunk_size=args.chunk_size,
            label="DeepL smoke-test request",
        )


if __name__ == "__main__":
    main()
