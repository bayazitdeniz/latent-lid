#!/usr/bin/env python
"""Dump tokenizer-level views of the canonical PUD train/test prompts for many models."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_MODELS = [
    "gpt2",
    "gpt2-xl",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "swiss-ai/Apertus-8B-2509",
    "swiss-ai/Apertus-8B-Instruct-2509",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "CohereLabs/aya-23-8B",
    "utter-project/EuroLLM-9B",
    "utter-project/EuroLLM-9B-Instruct",
]

SPLIT_TO_FILENAME = {
    "train": "pud_prompts_train.jsonl",
    "test": "pud_prompts_test.jsonl",
}


################################################################################
# Inputs and tokenization
################################################################################

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Dump tokenizer outputs for canonical PUD train/test prompts.")
    parser.add_argument("--pud-root", default="data/pud_holdout")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    return parser


def load_prompt_rows(path: Path, max_prompts: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_prompts is not None and len(rows) >= max_prompts:
                break
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def tokenizer_kwargs(tokenizer) -> dict:
    kwargs = {
        "add_special_tokens": False,
        "return_attention_mask": False,
    }
    if getattr(tokenizer, "is_fast", False):
        kwargs["return_offsets_mapping"] = True
    return kwargs


def tokenize_rows(rows: Iterable[dict], tokenizer) -> list[dict]:
    payloads = []
    kwargs = tokenizer_kwargs(tokenizer)
    for row in rows:
        text = row.get("prompt_text") or row.get("text")
        if text is None:
            raise ValueError("Each JSONL row must include 'prompt_text' or 'text'.")
        enc = tokenizer(text, **kwargs)
        token_ids = enc["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
        payload = {
            "id": row.get("id"),
            "prompt_text": text,
            "token_ids": token_ids,
            "tokens": tokens,
            "n_tokens": len(token_ids),
        }
        if getattr(tokenizer, "is_fast", False):
            payload["word_ids"] = enc.word_ids()
            payload["offset_mapping"] = enc["offset_mapping"]
        payloads.append(payload)
    return payloads


################################################################################
# Outputs and entrypoint
################################################################################

def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tokenizer(model_name: str, revision: str):
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            use_fast=True,
            trust_remote_code=True,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )


def main() -> None:
    args = build_parser().parse_args()
    pud_root = Path(args.pud_root)
    model_names = args.models or DEFAULT_MODELS

    split_rows = {
        split: load_prompt_rows(pud_root / SPLIT_TO_FILENAME[split], args.max_prompts)
        for split in args.splits
    }

    tokenized_by_split = {split: [{} for _ in rows] for split, rows in split_rows.items()}

    for model_name in model_names:
        print(f"Loading tokenizer: model={model_name} revision={args.revision}")
        tokenizer = load_tokenizer(model_name, args.revision)
        for split, rows in split_rows.items():
            tokenized = tokenize_rows(rows, tokenizer)
            for idx, payload in enumerate(tokenized):
                tokenized_by_split[split][idx][model_name] = {
                    "token_ids": payload["token_ids"],
                    "tokens": payload["tokens"],
                    "n_tokens": payload["n_tokens"],
                }
                if "word_ids" in payload:
                    tokenized_by_split[split][idx][model_name]["word_ids"] = payload["word_ids"]
                if "offset_mapping" in payload:
                    tokenized_by_split[split][idx][model_name]["offset_mapping"] = payload["offset_mapping"]

    for split, rows in split_rows.items():
        merged_rows = []
        for idx, row in enumerate(rows):
            text = row.get("prompt_text") or row.get("text")
            merged_rows.append(
                {
                    "id": row.get("id"),
                    "prompt_text": text,
                    "tokenized_dict": tokenized_by_split[split][idx],
                }
            )
        out_path = pud_root / f"tokenized_pud_prompts_{split}.jsonl"
        write_jsonl(out_path, merged_rows)
        print(f"Wrote tokenized prompts: {out_path}")


if __name__ == "__main__":
    main()
