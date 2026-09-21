#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from target_string_artifacts import (
    BYTE_FALLBACK_LANGS,
    COMMON_SPACE_PREFIX_MARKERS,
    TOKENIZER_META_PATH,
    detect_space_prefix_markers,
    infer_start_token_strategy,
)


DEFAULT_TOKENIZERS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh repo-local tokenizer metadata used for Wendler-style Start(w) preprocessing."
    )
    parser.add_argument("--models", nargs="*", default=DEFAULT_TOKENIZERS)
    parser.add_argument("--out", type=Path, default=TOKENIZER_META_PATH)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "tokenizers": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def tokenization_probe(tokenizer, text: str) -> dict[str, Any]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = list(encoded["input_ids"])
    return {
        "text": text,
        "input_ids": [int(token_id) for token_id in token_ids[:12]],
        "tokens": tokenizer.convert_ids_to_tokens(token_ids[:12], skip_special_tokens=False),
    }


def inspect_tokenizer(model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {
        "local_files_only": bool(args.local_files_only),
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.revision:
        kwargs["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    vocab = tokenizer.get_vocab()
    markers = detect_space_prefix_markers(tokenizer)
    start_token_strategy = infer_start_token_strategy(markers)
    marker_counts = {
        marker: sum(1 for token in vocab if token.startswith(marker) and len(token) > len(marker))
        for marker in COMMON_SPACE_PREFIX_MARKERS
    }
    byte_tokens_present = {
        lang: bool(token in vocab)
        for lang, token in {
            "zh": "<0xE8>",
            "ru": "<0xD0>",
        }.items()
        if lang in BYTE_FALLBACK_LANGS
    }
    return {
        "status": "ok",
        "model_name": model_name,
        "revision": args.revision or "default",
        "tokenizer_class": tokenizer.__class__.__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": int(len(vocab)),
        "space_prefix_markers": list(markers),
        "start_token_strategy": start_token_strategy,
        "space_prefix_marker_counts": marker_counts,
        "byte_fallback_probe_tokens_present": byte_tokens_present,
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0) or 0),
        "special_tokens_map": dict(getattr(tokenizer, "special_tokens_map", {}) or {}),
        "probes": [
            tokenization_probe(tokenizer, "flower"),
            tokenization_probe(tokenizer, " flower"),
            tokenization_probe(tokenizer, "花"),
            tokenization_probe(tokenizer, " 花"),
        ],
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    meta = load_existing(args.out)
    meta.setdefault("version", 1)
    tokenizers = meta.setdefault("tokenizers", {})
    for model_name in args.models:
        print(f"Inspecting tokenizer: {model_name}", flush=True)
        try:
            tokenizers[model_name] = inspect_tokenizer(model_name, args)
        except Exception as exc:  # pragma: no cover - records external loading failures
            tokenizers[model_name] = {
                "status": "error",
                "model_name": model_name,
                "revision": args.revision or "default",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  failed: {exc}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
