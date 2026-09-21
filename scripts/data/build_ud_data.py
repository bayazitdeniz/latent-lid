#!/usr/bin/env python3
"""Build deterministic UD train and held-out evaluation data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from tqdm import tqdm

from scripts.data.pud_data_source import (
    UD_EXTRA_LANGS,
    UD_EXTRA_TREEBANKS,
    ensure_ud_tarball,
    sentence_records_from_ud_treebank,
)
from utils import set_seed


_LIST_ITEM_RE = re.compile(r"^\d+(?:\.\d+)*\.$")
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


################################################################################
# CLI
################################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build deterministic non-parallel UD train/test holdout assets.")
    parser.add_argument("--langs", nargs="+", default=list(UD_EXTRA_LANGS))
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--ud-revision", default="2.17")
    parser.add_argument(
        "--ud-url",
        default="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-6036/ud-treebanks-v2.17.tgz",
    )
    parser.add_argument("--download-timeout", type=int, default=600)
    parser.add_argument("--ud-cache-path", default=None)
    return parser.parse_args()


################################################################################
# Filtering and splitting
################################################################################

def _split_records(
    records: list[dict[str, object]],
    *,
    train_size: int,
    test_size: int,
    split_seed: int,
    lang: str,
) -> tuple[list[tuple[int, dict[str, object]]], list[tuple[int, dict[str, object]]]]:
    if train_size + test_size > len(records):
        raise ValueError(
            f"Not enough texts for train/test split for lang={lang}: "
            f"need {train_size + test_size}, found {len(records)}."
        )
    indexed = list(enumerate(records))
    rng = random.Random(str(split_seed))
    rng.shuffle(indexed)
    return indexed[:train_size], indexed[train_size : train_size + test_size]


def _sample_from_split(
    records: list[dict[str, object]],
    *,
    size: int,
    split_seed: int,
    lang: str,
    split: str,
) -> list[tuple[int, dict[str, object]]]:
    if size > len(records):
        raise ValueError(f"Not enough {split} texts for lang={lang}: need {size}, found {len(records)}.")
    indexed = list(enumerate(records))
    rng = random.Random(f"{split_seed}:{lang}:{split}")
    rng.shuffle(indexed)
    return indexed[:size]


def _filter_reason(record: dict[str, object]) -> str | None:
    text = str(record["text"]).strip()
    if not text:
        return "empty"
    if _LIST_ITEM_RE.fullmatch(text):
        return "list_item"

    word_tokens = [tok for tok in record.get("surface_tokens", []) if _WORD_RE.search(str(tok))]
    alpha_like = [str(tok).casefold() for tok in word_tokens if any(ch.isalpha() for ch in str(tok))]
    if text.startswith("(") and text.endswith(")") and len(alpha_like) <= 2:
        return "parenthetical"
    if text.endswith(":") and len(alpha_like) <= 3:
        return "trailing_colon"
    if (text.endswith("...") or text.endswith("…")) and len(alpha_like) <= 3:
        return "ellipsis"
    if len(alpha_like) <= 2:
        return "very_short"
    if len(alpha_like) <= 3 and len(set(alpha_like)) == 1:
        return "repeated_interjection"
    return None


def _filter_records(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int], list[dict[str, object]]]:
    kept: list[dict[str, object]] = []
    filtered_counts: dict[str, int] = {}
    filtered_rows: list[dict[str, object]] = []
    for record in records:
        reason = _filter_reason(record)
        if reason is None:
            kept.append(record)
            continue
        filtered_counts[reason] = filtered_counts.get(reason, 0) + 1
        filtered_rows.append(
            {
                "text": record["text"],
                "surface_tokens": record["surface_tokens"],
                "ud_split": record.get("ud_split"),
                "filter_reason": reason,
            }
        )
    return kept, filtered_counts, filtered_rows


################################################################################
# Outputs
################################################################################

def _write_prompts_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f_out:
        for obj in payloads:
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote prompts JSONL: {path}")


def _write_train_csv(root: Path, lang: str, records: list[dict[str, object]]) -> None:
    lang_dir = root / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    csv_path = lang_dir / "train.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["prompt"])
        for record in records:
            writer.writerow([str(record["text"]).replace("\n", " ")])
    print(f"Wrote train.csv: {csv_path}")


def _write_rejected_rows(root: Path, lang: str, cfg: dict[str, str], rows: list[dict[str, object]]) -> None:
    out_path = root / f"{lang}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        for row in rows:
            payload = {
                "source_lang": lang,
                "treebank": f"UD_{cfg['treebank']}",
                "source": "ud",
                "is_parallel": False,
                **row,
            }
            f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"Wrote rejected rows: {out_path}")


def _prompt_payload(
    *,
    lang: str,
    cfg: dict[str, str],
    split: str,
    offset: int,
    record: dict[str, object],
) -> dict[str, object]:
    return {
        "id": f"{cfg['file_prefix']}-{split}-{offset}",
        "prompt_text": record["text"],
        "surface_tokens": record["surface_tokens"],
        "source_lang": lang,
        "target_lang": lang,
        "treebank": f"UD_{cfg['treebank']}",
        "source": "ud",
        "is_parallel": False,
        "ud_split": record.get("ud_split"),
    }


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    set_seed(42)
    args = parse_args()
    if args.train_size <= 0 or args.test_size <= 0:
        raise ValueError("Train and test sizes must be > 0.")

    tar_path = ensure_ud_tarball(
        ud_revision=args.ud_revision,
        ud_url=args.ud_url,
        download_timeout=args.download_timeout,
        cache_path=args.ud_cache_path,
    )

    split_root = Path(args.out_dir) / "ud_holdout"
    train_root = split_root / "ud_langs_train"
    rejected_root = split_root / "rejected"
    train_prompt_payloads: list[dict] = []
    test_prompt_payloads: list[dict] = []
    split_meta: dict[str, object] = {
        "split": "train_test",
        "source": "ud",
        "is_parallel": False,
        "split_seed": args.split_seed,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "langs": list(args.langs),
        "per_language": {},
    }

    train_counts: dict[str, int] = {}
    test_counts: dict[str, int] = {}
    available_counts: dict[str, int] = {}

    for lang in tqdm(args.langs, desc="Extracting UD holdout", unit="lang"):
        if lang not in UD_EXTRA_TREEBANKS:
            raise ValueError(f"Unsupported non-parallel UD language: {lang}")
        cfg = UD_EXTRA_TREEBANKS[lang]
        all_records = sentence_records_from_ud_treebank(
            tar_path,
            treebank=cfg["treebank"],
            file_prefix=cfg["file_prefix"],
            splits=("train", "dev", "test"),
        )
        available_counts[lang] = len(all_records)
        filtered_records, filtered_counts, filtered_rows = _filter_records(all_records)

        if lang == "mr":
            train_rows, test_rows = _split_records(
                filtered_records,
                train_size=args.train_size,
                test_size=args.test_size,
                split_seed=args.split_seed,
                lang=lang,
            )
            split_policy = "combined_deterministic"
        else:
            train_records_source = [r for r in filtered_records if r.get("ud_split") == "train"]
            test_records_source = [r for r in filtered_records if r.get("ud_split") == "test"]
            train_rows = _sample_from_split(
                train_records_source,
                size=args.train_size,
                split_seed=args.split_seed,
                lang=lang,
                split="train",
            )
            test_rows = _sample_from_split(
                test_records_source,
                size=args.test_size,
                split_seed=args.split_seed,
                lang=lang,
                split="test",
            )
            split_policy = "upstream_train_test"

        train_records = [record for _, record in train_rows]
        test_records = [record for _, record in test_rows]
        _write_train_csv(train_root, lang, train_records)
        _write_rejected_rows(rejected_root, lang, cfg, filtered_rows)

        for i, record in enumerate(train_records):
            train_prompt_payloads.append(_prompt_payload(lang=lang, cfg=cfg, split="train", offset=i, record=record))
        for i, record in enumerate(test_records):
            test_prompt_payloads.append(_prompt_payload(lang=lang, cfg=cfg, split="test", offset=i, record=record))

        train_counts[lang] = len(train_records)
        test_counts[lang] = len(test_records)
        split_meta["per_language"][lang] = {
            "mode": "train_test",
            "split_policy": split_policy,
            "treebank": f"UD_{cfg['treebank']}",
            "file_prefix": cfg["file_prefix"],
            "n_total": len(all_records),
            "n_filtered": len(all_records) - len(filtered_records),
            "filtered_counts": filtered_counts,
            "n_usable": len(filtered_records),
            "n_train": len(train_records),
            "n_test": len(test_records),
            "train_indices": [idx for idx, _ in train_rows],
            "test_indices": [idx for idx, _ in test_rows],
        }

    _write_prompts_jsonl(split_root / "ud_prompts_train.jsonl", train_prompt_payloads)
    _write_prompts_jsonl(split_root / "ud_prompts_test.jsonl", test_prompt_payloads)
    (split_root / "ud_split_meta.json").write_text(
        json.dumps(split_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote split metadata: {split_root / 'ud_split_meta.json'}")
    print(f"Completed UD holdout build under: {split_root}")
    print("Available source sentences per language:")
    for lang in args.langs:
        print(f"  {lang}: {available_counts.get(lang, 0)}")
    print(f"Train JSONL total prompts: {sum(train_counts.values())}")
    print(f"Test JSONL total prompts: {sum(test_counts.values())}")


if __name__ == "__main__":
    main()
