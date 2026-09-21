#!/usr/bin/env python3
"""Build the balanced FineWeb/FineWeb2 corpus used to fit tuned lenses."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


LANGS = {
    "ar": {"dataset": "HuggingFaceFW/fineweb-2", "name": "arb_Arab", "label": "Arabic"},
    "bg": {"dataset": "HuggingFaceFW/fineweb-2", "name": "bul_Cyrl", "label": "Bulgarian"},
    "cs": {"dataset": "HuggingFaceFW/fineweb-2", "name": "ces_Latn", "label": "Czech"},
    "de": {"dataset": "HuggingFaceFW/fineweb-2", "name": "deu_Latn", "label": "German"},
    "en": {"dataset": "HuggingFaceFW/fineweb", "name": None, "label": "English"},
    "es": {"dataset": "HuggingFaceFW/fineweb-2", "name": "spa_Latn", "label": "Spanish"},
    "fa": {"dataset": "HuggingFaceFW/fineweb-2", "name": "fas_Arab", "label": "Persian"},
    "fi": {"dataset": "HuggingFaceFW/fineweb-2", "name": "fin_Latn", "label": "Finnish"},
    "fr": {"dataset": "HuggingFaceFW/fineweb-2", "name": "fra_Latn", "label": "French"},
    "gl": {"dataset": "HuggingFaceFW/fineweb-2", "name": "glg_Latn", "label": "Galician"},
    "hi": {"dataset": "HuggingFaceFW/fineweb-2", "name": "hin_Deva", "label": "Hindi"},
    "id": {"dataset": "HuggingFaceFW/fineweb-2", "name": "ind_Latn", "label": "Indonesian"},
    "is": {"dataset": "HuggingFaceFW/fineweb-2", "name": "isl_Latn", "label": "Icelandic"},
    "it": {"dataset": "HuggingFaceFW/fineweb-2", "name": "ita_Latn", "label": "Italian"},
    "ja": {"dataset": "HuggingFaceFW/fineweb-2", "name": "jpn_Jpan", "label": "Japanese"},
    "ko": {"dataset": "HuggingFaceFW/fineweb-2", "name": "kor_Hang", "label": "Korean"},
    "mr": {"dataset": "HuggingFaceFW/fineweb-2", "name": "mar_Deva", "label": "Marathi"},
    "pl": {"dataset": "HuggingFaceFW/fineweb-2", "name": "pol_Latn", "label": "Polish"},
    "pt": {"dataset": "HuggingFaceFW/fineweb-2", "name": "por_Latn", "label": "Portuguese"},
    "ru": {"dataset": "HuggingFaceFW/fineweb-2", "name": "rus_Cyrl", "label": "Russian"},
    "sr": {"dataset": "HuggingFaceFW/fineweb-2", "name": "srp_Cyrl", "label": "Serbian"},
    "sv": {"dataset": "HuggingFaceFW/fineweb-2", "name": "swe_Latn", "label": "Swedish"},
    "th": {"dataset": "HuggingFaceFW/fineweb-2", "name": "tha_Thai", "label": "Thai"},
    "tr": {"dataset": "HuggingFaceFW/fineweb-2", "name": "tur_Latn", "label": "Turkish"},
    "uk": {"dataset": "HuggingFaceFW/fineweb-2", "name": "ukr_Cyrl", "label": "Ukrainian"},
    "ur": {"dataset": "HuggingFaceFW/fineweb-2", "name": "urd_Arab", "label": "Urdu"},
    "zh": {"dataset": "HuggingFaceFW/fineweb-2", "name": "cmn_Hani", "label": "Mandarin Chinese"},
}
LANG_ALIASES = {
    # The synthetic data uses pt_br, but FineWeb2 provides generic Portuguese.
    # Keep this alias local to FineWeb corpus construction.
    "pt_br": "pt",
}
LANG_CHOICES = sorted(set(LANGS) | set(LANG_ALIASES))

DEFAULT_TOKENIZER = "swiss-ai/Apertus-8B-2509"
DEFAULT_OUT_DIR = "data/fineweb_lens_27lang_55m"
DEFAULT_LANGS = (
    "ar",
    "bg",
    "cs",
    "de",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "gl",
    "hi",
    "id",
    "is",
    "it",
    "ja",
    "ko",
    "mr",
    "pl",
    "pt_br",
    "ru",
    "sr",
    "sv",
    "th",
    "tr",
    "uk",
    "ur",
    "zh",
)

DEFAULT_STREAM_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0


################################################################################
# CLI
################################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build a balanced multilingual tuned-lens corpus from FineWeb/FineWeb2.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--train-tokens-per-lang", type=int, default=4_096_000)
    parser.add_argument("--val-tokens-per-lang", type=int, default=1_024_000)
    parser.add_argument("--test-tokens-per-lang", type=int, default=1_024_000)
    parser.add_argument("--langs", nargs="+", default=list(DEFAULT_LANGS), choices=LANG_CHOICES)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-docs-per-lang", type=int, default=None)
    parser.add_argument(
        "--reuse-dataset-dir",
        action="append",
        default=[],
        help="Existing saved FineWeb corpus to reuse matching language rows from. Repeatable.",
    )
    parser.add_argument(
        "--lang-cache-dir",
        type=str,
        default=None,
        help="Directory for per-split/per-language intermediate caches. Defaults to <out-dir>_lang_cache.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


################################################################################
# Streaming and chunking
################################################################################

def round_down_to_seq_len(n_tokens: int, seq_len: int) -> int:
    return (int(n_tokens) // int(seq_len)) * int(seq_len)


def load_stream(
    dataset_name: str,
    config_name: str | None,
    *,
    seed: int,
    shuffle_buffer: int,
):
    kwargs = {"split": "train", "streaming": True}
    if config_name is None:
        ds = load_dataset(dataset_name, **kwargs)
    else:
        ds = load_dataset(dataset_name, name=config_name, **kwargs)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return ds


def iter_chunks_for_language(
    *,
    lang: str,
    tokenizer,
    seq_len: int,
    min_chars: int,
    seed: int,
    shuffle_buffer: int,
    max_docs: int | None,
) -> Iterator[tuple[str, int, dict[str, str]]]:
    source_lang = LANG_ALIASES.get(lang, lang)
    cfg = LANGS[source_lang]
    ds = load_stream(
        dataset_name=cfg["dataset"],
        config_name=cfg["name"],
        seed=seed,
        shuffle_buffer=shuffle_buffer,
    )

    buffer_text = ""
    buffer_ids: list[int] = []
    buffer_offsets: list[tuple[int, int]] = []
    docs_seen = 0
    separator_text = "\n\n"

    for row in ds:
        docs_seen += 1
        if max_docs is not None and docs_seen > int(max_docs):
            break
        text = str(row.get("text") or "")
        if len(text) < int(min_chars):
            continue
        piece = text if not buffer_text else f"{separator_text}{text}"
        encoded = tokenizer(
            piece,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        piece_ids = list(encoded["input_ids"])
        piece_offsets = list(encoded["offset_mapping"])
        if not piece_ids:
            continue
        base_char = len(buffer_text)
        buffer_text += piece
        buffer_ids.extend(int(token_id) for token_id in piece_ids)
        buffer_offsets.extend((base_char + int(start), base_char + int(end)) for start, end in piece_offsets)
        while len(buffer_ids) >= seq_len:
            end_char = int(buffer_offsets[seq_len - 1][1])
            chunk_text = buffer_text[:end_char]
            buffer_text = buffer_text[end_char:]
            buffer_ids = buffer_ids[seq_len:]
            buffer_offsets = [(start - end_char, end - end_char) for start, end in buffer_offsets[seq_len:]]
            yield chunk_text, seq_len, {
                "lang": lang,
                "source_lang": source_lang,
                "language": cfg["label"],
                "source_dataset": cfg["dataset"],
                "source_config": cfg["name"] or "default",
            }


def collect_split_for_language(
    *,
    lang: str,
    split_name: str,
    target_tokens: int,
    tokenizer,
    seq_len: int,
    min_chars: int,
    seed: int,
    shuffle_buffer: int,
    max_docs: int | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    target_tokens = round_down_to_seq_len(target_tokens, seq_len)
    target_chunks = target_tokens // int(seq_len)
    split_seed_offsets = {"train": 0, "validation": 10_000, "test": 20_000}
    lang_seed = int(seed) + split_seed_offsets[split_name] + sum(ord(ch) for ch in lang)

    rows: list[dict[str, object]] = []
    n_tokens = 0
    n_chunks = 0
    chunk_iter = iter_chunks_for_language(
        lang=lang,
        tokenizer=tokenizer,
        seq_len=seq_len,
        min_chars=min_chars,
        seed=lang_seed,
        shuffle_buffer=shuffle_buffer,
        max_docs=max_docs,
    )
    pbar = tqdm(total=target_chunks, desc=f"{split_name}:{lang}", unit="chunk")
    for text, n_reference_tokens, meta in chunk_iter:
        rows.append(
            {
                "text": text,
                "lang": meta["lang"],
                "source_lang": meta["source_lang"],
                "language": meta["language"],
                "source_dataset": meta["source_dataset"],
                "source_config": meta["source_config"],
                "num_reference_tokens": int(n_reference_tokens),
            }
        )
        n_chunks += 1
        n_tokens += int(n_reference_tokens)
        pbar.update(1)
        if n_chunks >= target_chunks:
            break
    pbar.close()
    return rows, {
        "split": split_name,
        "lang": lang,
        "chunks": n_chunks,
        "tokens": n_tokens,
        "target_chunks": target_chunks,
        "target_tokens": target_tokens,
        "shortfall_chunks": max(target_chunks - n_chunks, 0),
        "shortfall_tokens": max(target_tokens - n_tokens, 0),
        "met_target": n_chunks >= target_chunks,
    }


def collect_split_for_language_with_retries(
    *,
    lang: str,
    split_name: str,
    target_tokens: int,
    tokenizer,
    seq_len: int,
    min_chars: int,
    seed: int,
    shuffle_buffer: int,
    max_docs: int | None,
    max_retries: int = DEFAULT_STREAM_RETRIES,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    last_error: Exception | None = None
    for attempt_idx in range(int(max_retries) + 1):
        try:
            return collect_split_for_language(
                lang=lang,
                split_name=split_name,
                target_tokens=target_tokens,
                tokenizer=tokenizer,
                seq_len=seq_len,
                min_chars=min_chars,
                seed=seed,
                shuffle_buffer=shuffle_buffer,
                max_docs=max_docs,
            )
        except Exception as exc:
            last_error = exc
            if attempt_idx >= int(max_retries):
                break
            wait_seconds = float(retry_backoff_seconds) * float(2 ** attempt_idx)
            print(
                f"Warning: {split_name}:{lang} failed on attempt {attempt_idx + 1}/{int(max_retries) + 1} "
                f"with {type(exc).__name__}: {exc}. Retrying in {wait_seconds:.1f}s..."
            )
            time.sleep(wait_seconds)
    assert last_error is not None
    raise last_error


################################################################################
# Cached and reused rows
################################################################################

def _lang_cache_path(cache_dir: Path, split_name: str, lang: str) -> Path:
    return cache_dir / split_name / lang


def _normalize_reused_row(
    row: dict[str, object],
    *,
    requested_lang: str,
    source_lang: str,
    seq_len: int,
) -> dict[str, object]:
    out = dict(row)
    out["lang"] = requested_lang
    out["source_lang"] = str(out.get("source_lang") or source_lang)
    out["num_reference_tokens"] = int(out.get("num_reference_tokens") or seq_len)
    return out


def _stats_for_rows(
    *,
    rows: list[dict[str, object]],
    lang: str,
    split_name: str,
    target_tokens: int,
    seq_len: int,
    source: str,
    reuse_dataset_dir: str | None = None,
) -> dict[str, object]:
    target_tokens = round_down_to_seq_len(target_tokens, seq_len)
    target_chunks = target_tokens // int(seq_len)
    n_chunks = len(rows)
    n_tokens = sum(int(row.get("num_reference_tokens") or seq_len) for row in rows)
    stats = {
        "split": split_name,
        "lang": lang,
        "chunks": n_chunks,
        "tokens": n_tokens,
        "target_chunks": target_chunks,
        "target_tokens": target_tokens,
        "shortfall_chunks": max(target_chunks - n_chunks, 0),
        "shortfall_tokens": max(target_tokens - n_tokens, 0),
        "met_target": n_chunks >= target_chunks,
        "source": source,
    }
    if reuse_dataset_dir is not None:
        stats["reuse_dataset_dir"] = reuse_dataset_dir
    return stats


def load_cached_split_for_language(
    *,
    cache_dir: Path,
    lang: str,
    split_name: str,
    target_tokens: int,
    seq_len: int,
) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    cache_path = _lang_cache_path(cache_dir, split_name, lang)
    if not cache_path.exists():
        return None
    dataset = load_from_disk(str(cache_path))
    rows = [dict(row) for row in dataset]
    target_chunks = round_down_to_seq_len(target_tokens, seq_len) // int(seq_len)
    if len(rows) < target_chunks:
        return None
    rows = rows[:target_chunks]
    return rows, _stats_for_rows(
        rows=rows,
        lang=lang,
        split_name=split_name,
        target_tokens=target_tokens,
        seq_len=seq_len,
        source="cache",
    )


def load_reused_split_for_language(
    *,
    reuse_dataset_dirs: list[Path],
    lang: str,
    split_name: str,
    target_tokens: int,
    seq_len: int,
) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    source_lang = LANG_ALIASES.get(lang, lang)
    target_chunks = round_down_to_seq_len(target_tokens, seq_len) // int(seq_len)
    for dataset_dir in reuse_dataset_dirs:
        if not dataset_dir.exists():
            continue
        try:
            dataset_dict = load_from_disk(str(dataset_dir))
        except Exception:
            continue
        if split_name not in dataset_dict:
            continue
        rows = []
        for row in dataset_dict[split_name]:
            row_lang = str(row.get("lang") or "")
            row_source_lang = str(row.get("source_lang") or row_lang)
            if row_lang not in {lang, source_lang} and row_source_lang not in {lang, source_lang}:
                continue
            rows.append(
                _normalize_reused_row(
                    dict(row),
                    requested_lang=lang,
                    source_lang=source_lang,
                    seq_len=seq_len,
                )
            )
            if len(rows) >= target_chunks:
                break
        if len(rows) >= target_chunks:
            return rows, _stats_for_rows(
                rows=rows[:target_chunks],
                lang=lang,
                split_name=split_name,
                target_tokens=target_tokens,
                seq_len=seq_len,
                source="reuse",
                reuse_dataset_dir=str(dataset_dir),
            )
    return None


def save_split_language_cache(
    cache_dir: Path,
    split_name: str,
    lang: str,
    rows: list[dict[str, object]],
) -> None:
    cache_path = _lang_cache_path(cache_dir, split_name, lang)
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(cache_path))


def _rebalance_split_rows(
    per_language_rows: dict[str, list[dict[str, object]]],
    per_language_stats: list[dict[str, object]],
    seq_len: int,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    if not per_language_stats:
        return [], None

    shared_chunks = min(int(stat["chunks"]) for stat in per_language_stats)
    requested_chunks = min(int(stat["target_chunks"]) for stat in per_language_stats)
    if shared_chunks <= 0:
        raise RuntimeError("Could not collect any full chunks for at least one language; refusing to save an empty balanced split.")

    warning = None
    if shared_chunks < requested_chunks:
        limiting = [str(stat["lang"]) for stat in per_language_stats if int(stat["chunks"]) == shared_chunks]
        warning = {
            "split": str(per_language_stats[0]["split"]),
            "requested_chunks_per_lang": requested_chunks,
            "effective_chunks_per_lang": shared_chunks,
            "requested_tokens_per_lang": requested_chunks * int(seq_len),
            "effective_tokens_per_lang": shared_chunks * int(seq_len),
            "limiting_languages": limiting,
        }
        print(
            f"Warning: split '{warning['split']}' rebalanced down to {shared_chunks} chunks "
            f"({warning['effective_tokens_per_lang']} tokens) per language; limiting languages: {', '.join(limiting)}"
        )

    trimmed_rows: list[dict[str, object]] = []
    for stat in per_language_stats:
        lang = str(stat["lang"])
        rows = per_language_rows[lang][:shared_chunks]
        trimmed_rows.extend(rows)
        stat["effective_chunks"] = shared_chunks
        stat["effective_tokens"] = shared_chunks * int(seq_len)
        stat["trimmed_chunks"] = max(int(stat["chunks"]) - shared_chunks, 0)
        stat["trimmed_tokens"] = max(int(stat["tokens"]) - (shared_chunks * int(seq_len)), 0)
    return trimmed_rows, warning


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    lang_cache_dir = Path(args.lang_cache_dir) if args.lang_cache_dir else Path(f"{args.out_dir}_lang_cache")
    reuse_dataset_dirs = [Path(path) for path in args.reuse_dataset_dir]

    # Load the target tokenizer up front because token budgets are enforced
    # in the model's own tokenization space, not by document count.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )

    # Define per-language token budgets for each split. Each language will be
    # sampled independently to the same budget before we merge splits together.
    split_budgets = {
        "train": int(args.train_tokens_per_lang),
        "validation": int(args.val_tokens_per_lang),
        "test": int(args.test_tokens_per_lang),
    }
    all_splits: dict[str, list[dict[str, object]]] = defaultdict(list)
    stats: dict[str, object] = {
        "tokenizer": args.tokenizer,
        "seq_len": int(args.seq_len),
        "langs": list(args.langs),
        "budgets_requested": split_budgets,
        "budgets_effective": {
            split: round_down_to_seq_len(tokens, int(args.seq_len))
            for split, tokens in split_budgets.items()
        },
        "per_language": [],
        "rebalance_warnings": [],
        "reuse_dataset_dirs": [str(path) for path in reuse_dataset_dirs],
        "lang_cache_dir": str(lang_cache_dir),
    }

    # Build each split one language at a time so the final corpus is balanced
    # by tokenizer tokens rather than by documents or raw bytes.
    for split_name, tokens_per_lang in split_budgets.items():
        split_rows_by_lang: dict[str, list[dict[str, object]]] = {}
        split_stats: list[dict[str, object]] = []
        for lang in args.langs:
            cached = load_cached_split_for_language(
                cache_dir=lang_cache_dir,
                lang=lang,
                split_name=split_name,
                target_tokens=tokens_per_lang,
                seq_len=int(args.seq_len),
            )
            if cached is not None:
                rows, lang_stats = cached
                print(f"Reusing cached {split_name}:{lang} from {lang_cache_dir}")
            else:
                reused = load_reused_split_for_language(
                    reuse_dataset_dirs=reuse_dataset_dirs,
                    lang=lang,
                    split_name=split_name,
                    target_tokens=tokens_per_lang,
                    seq_len=int(args.seq_len),
                )
                if reused is not None:
                    rows, lang_stats = reused
                    print(f"Reusing existing corpus rows for {split_name}:{lang}")
                else:
                    rows, lang_stats = collect_split_for_language_with_retries(
                        lang=lang,
                        split_name=split_name,
                        target_tokens=tokens_per_lang,
                        tokenizer=tokenizer,
                        seq_len=int(args.seq_len),
                        min_chars=int(args.min_chars),
                        seed=int(args.seed),
                        shuffle_buffer=int(args.shuffle_buffer),
                        max_docs=args.max_docs_per_lang,
                    )
                    lang_stats["source"] = "stream"
                save_split_language_cache(lang_cache_dir, split_name, lang, rows)
            split_rows_by_lang[lang] = rows
            split_stats.append(lang_stats)
            stats["per_language"].append(lang_stats)

        balanced_rows, warning = _rebalance_split_rows(
            split_rows_by_lang,
            split_stats,
            seq_len=int(args.seq_len),
        )
        if warning is not None:
            stats["rebalance_warnings"].append(warning)
        all_splits[split_name].extend(balanced_rows)

        # Shuffle only after balanced collection so each split is mixed across
        # languages while preserving the exact per-language token budget.
        rng = random.Random(int(args.seed) + len(split_name))
        rng.shuffle(all_splits[split_name])

    # Materialize the in-memory rows as a Hugging Face DatasetDict so the tuned
    # lens fitter can load the corpus directly from disk later on.
    dataset_dict = DatasetDict(
        {split_name: Dataset.from_list(rows) for split_name, rows in all_splits.items()}
    )
    dataset_dict.save_to_disk(args.out_dir)

    # Write a compact accounting summary so we can verify that the saved corpus
    # actually matches the requested token and example balance per language.
    stats["final"] = {
        split_name: {
            "examples": len(rows),
            "tokens": sum(int(row["num_reference_tokens"]) for row in rows),
            "tokens_by_lang": {
                lang: sum(int(row["num_reference_tokens"]) for row in rows if row["lang"] == lang)
                for lang in args.langs
            },
            "examples_by_lang": {
                lang: sum(1 for row in rows if row["lang"] == lang)
                for lang in args.langs
            },
        }
        for split_name, rows in all_splits.items()
    }

    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Saved DatasetDict to: {args.out_dir}")
    print(json.dumps(stats["final"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
