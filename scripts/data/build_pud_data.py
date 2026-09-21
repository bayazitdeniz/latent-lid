#!/usr/bin/env python3
"""Build deterministic PUD train (for GMM) and held-out evaluation data."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from scripts.data.pud_data_source import PUD_LANGS, ensure_ud_tarball, sentence_records_from_ud_tar
from utils import set_seed


################################################################################
# CLI
################################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build deterministic PUD train/test splits and train-only calibration CSVs.")
    parser.add_argument(
        "--langs",
        nargs="+",
        default=list(PUD_LANGS),
        help="Language codes to build. Defaults to all supported PUD languages.",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Base data directory. Writes canonical split outputs under data/pud_holdout.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=100,
        help="Per-language train size.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Per-language held-out eval size.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for deterministic train/test splitting.",
    )
    parser.add_argument(
        "--ud-revision",
        default="2.17",
        help="UD dataset revision tag on HF.",
    )
    parser.add_argument(
        "--source",
        choices=["ud_tarball", "hf"],
        default="ud_tarball",
        help="Data source: canonical UD tarball (ud_tarball) or Hugging Face mirror (hf).",
    )
    parser.add_argument(
        "--ud-url",
        default="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-6036/ud-treebanks-v2.17.tgz",
        help="UD treebanks tarball URL (used when --source=ud_tarball).",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=600,
        help="Download timeout in seconds for the UD tarball.",
    )
    parser.add_argument(
        "--ud-cache-path",
        default=None,
        help="Optional on-disk cache path for the UD tarball.",
    )
    return parser.parse_args()


################################################################################
# Data loading and splitting
################################################################################

def _records_from_hf(lang: str, ud_revision: str) -> list[dict[str, object]]:
    cfg = f"{lang}_pud"
    ds = load_dataset("commul/universal_dependencies", cfg, revision=ud_revision)["train"]
    records = []
    for ex in ds.select(range(len(ds))):
        record = {"text": ex["text"]}
        if "tokens" in ex and ex["tokens"] is not None:
            record["surface_tokens"] = list(ex["tokens"])
        records.append(record)
    return records


################################################################################
# Outputs
################################################################################

def _write_prompts_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f_out:
        for obj in payloads:
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote prompts JSONL: {path}")


def _write_train_csv(root: Path, lang: str, texts: list[str]) -> None:
    lang_dir = root / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    csv_path = lang_dir / "train.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["prompt"])
        for text in texts:
            writer.writerow([text.replace("\n", " ")])
    print(f"Wrote train.csv: {csv_path}")


def _resolve_split_sizes(args: argparse.Namespace) -> tuple[int, int]:
    train_size = args.train_size
    test_size = args.test_size
    if train_size <= 0:
        raise ValueError("Train size must be > 0.")
    if test_size <= 0:
        raise ValueError("Test size must be > 0.")
    return train_size, test_size


def _split_texts(
    texts: list[object],
    *,
    train_size: int,
    test_size: int,
    split_seed: int,
    lang: str,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    if train_size + test_size > len(texts):
        raise ValueError(
            f"Not enough texts for train/test split for lang={lang}: "
            f"need {train_size + test_size}, found {len(texts)}."
        )
    indexed = list(enumerate(texts))
    rng = random.Random(str(split_seed))
    rng.shuffle(indexed)
    train_rows = indexed[:train_size]
    test_rows = indexed[train_size : train_size + test_size]
    return train_rows, test_rows


def _resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    split_root = Path(args.out_dir) / "pud_holdout"
    return {
        "split_root": split_root,
        "out_prompts_test": split_root / "pud_prompts_test.jsonl",
        "out_prompts_train": split_root / "pud_prompts_train.jsonl",
        "out_gmm_root_train": split_root / "pud_langs_train",
        "out_split_meta": split_root / "pud_split_meta.json",
    }


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    set_seed(42)
    args = parse_args()
    train_size, test_size = _resolve_split_sizes(args)
    resolved_paths = _resolve_output_paths(args)
    split_root = resolved_paths["split_root"]
    split_root.mkdir(parents=True, exist_ok=True)

    tar_path = None
    if args.source == "ud_tarball":
        tar_path = ensure_ud_tarball(
            ud_revision=args.ud_revision,
            ud_url=args.ud_url,
            download_timeout=args.download_timeout,
            cache_path=args.ud_cache_path,
        )

    available_counts: dict[str, int] = {}
    train_counts: dict[str, int] = {}
    test_counts: dict[str, int] = {}
    test_prompt_payloads: list[dict] = []
    train_prompt_payloads: list[dict] = []
    split_meta: dict[str, object] = {
        "split": "train_test",
        "split_unit": "sentence_index",
        "split_seed": args.split_seed,
        "train_size": train_size,
        "test_size": test_size,
        "langs": list(args.langs),
        "per_language": {},
    }

    lang_iter = tqdm(args.langs, desc="Extracting PUD", unit="lang")
    for lang in lang_iter:
        if args.source == "hf":
            records = _records_from_hf(lang, args.ud_revision)
        else:
            if tar_path is None:
                raise RuntimeError("UD tarball not available.")
            records = sentence_records_from_ud_tar(lang, tar_path)

        available_counts[lang] = len(records)
        train_rows, test_rows = _split_texts(
            records,
            train_size=train_size,
            test_size=test_size,
            split_seed=args.split_seed,
            lang=lang,
        )
        train_records = [record for _, record in train_rows]
        test_records = [record for _, record in test_rows]
        train_texts = [record["text"] for record in train_records]
        test_texts = [record["text"] for record in test_records]

        for i, record in enumerate(train_records):
            payload = {"id": f"{cfg}-train-{i}", "prompt_text": record["text"]}
            payload["surface_tokens"] = record["surface_tokens"]
            train_prompt_payloads.append(payload)
        for i, record in enumerate(test_records):
            payload = {"id": f"{cfg}-{i}", "prompt_text": record["text"]}
            payload["surface_tokens"] = record["surface_tokens"]
            test_prompt_payloads.append(payload)

        train_counts[lang] = len(train_texts)
        test_counts[lang] = len(test_texts)
        _write_train_csv(resolved_paths["out_gmm_root_train"], lang, train_texts)
        split_meta["per_language"][lang] = {
            "mode": "train_test",
            "n_total": len(records),
            "n_train": len(train_texts),
            "n_test": len(test_texts),
            "train_indices": [idx for idx, _ in train_rows],
            "test_indices": [idx for idx, _ in test_rows],
        }

    _write_prompts_jsonl(resolved_paths["out_prompts_train"], train_prompt_payloads)
    _write_prompts_jsonl(resolved_paths["out_prompts_test"], test_prompt_payloads)
    resolved_paths["out_split_meta"].parent.mkdir(parents=True, exist_ok=True)
    resolved_paths["out_split_meta"].write_text(
        json.dumps(split_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote split metadata: {resolved_paths['out_split_meta']}")

    print(f"Completed split build under: {split_root}")
    print("Available source sentences per language:")
    for lang in args.langs:
        print(f"  {lang}: {available_counts.get(lang, 0)}")
    print(f"Available source total sentences: {sum(available_counts.values())}")
    print("Train JSONL prompts per language:")
    for lang in args.langs:
        print(f"  {lang}: {train_counts.get(lang, 0)}")
    print(f"Train JSONL total prompts: {sum(train_counts.values())}")
    print("Test JSONL prompts per language:")
    for lang in args.langs:
        print(f"  {lang}: {test_counts.get(lang, 0)}")
    print(f"Test JSONL total prompts: {sum(test_counts.values())}")
    for lang in args.langs:
        if train_counts.get(lang, 0) != train_size:
            print(f"WARNING: {lang} train CSV has {train_counts.get(lang, 0)} sentences (expected {train_size}).")
        if test_counts.get(lang, 0) != test_size:
            print(f"WARNING: {lang} test prompts file has {test_counts.get(lang, 0)} sentences (expected {test_size}).")


if __name__ == "__main__":
    main()
