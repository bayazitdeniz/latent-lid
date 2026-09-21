#!/usr/bin/env python
"""Compute bits per UTF-8 byte for causal LMs on repo corpora."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PUD_SOURCES = {"pud9", "pud9_ud6", "pud21", "pud21_ud6"}
PUD21_LANGS = {
    "ar",
    "cs",
    "de",
    "en",
    "es",
    "fi",
    "fr",
    "gl",
    "hi",
    "id",
    "is",
    "it",
    "ja",
    "ko",
    "pl",
    "pt",
    "ru",
    "sv",
    "th",
    "tr",
    "zh",
}
PUD9_LANGS = {"ar", "de", "en", "es", "fr", "hi", "id", "ru", "zh"}


################################################################################
# CLI and corpus records
################################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", action="append", required=True, help="HF model id/path. Repeat for multiple models.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data-source", choices=sorted(PUD_SOURCES), default="pud21_ud6")
    parser.add_argument("--pud-root", default="data/pud_holdout")
    parser.add_argument("--ud-root", default="data/ud_holdout")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=0, help="Total cap after language filtering. 0 means uncapped.")
    parser.add_argument("--max-examples-per-lang", type=int, default=0, help="Per-language cap. 0 means uncapped.")
    parser.add_argument("--max-length", type=int, default=0, help="Tokenizer max length cap. 0 uses model/tokenizer limit.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device-map", default=None, help="Optional transformers device_map, e.g. auto.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepend-bos", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--output-csv", default="logs/bpb/pud21_ud6_bpb.csv")
    parser.add_argument("--output-jsonl", default="logs/bpb/pud21_ud6_bpb.jsonl")
    parser.add_argument("--line-output-csv", default="logs/bpb/pud21_ud6_line_scores.csv")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def lang_from_row(row: dict) -> str:
    if row.get("source_lang"):
        return str(row["source_lang"])
    row_id = str(row.get("id", ""))
    if "_pud-" in row_id:
        return row_id.split("_pud-", 1)[0]
    return row_id.split("_", 1)[0]


def load_records(args: argparse.Namespace) -> list[dict]:
    pud_path = Path(args.pud_root) / f"pud_prompts_{args.split}.jsonl"
    ud_path = Path(args.ud_root) / f"ud_prompts_{args.split}.jsonl"

    langs = PUD21_LANGS if args.data_source.startswith("pud21") else PUD9_LANGS
    include_ud = args.data_source.endswith("_ud6")

    records: list[dict] = []
    for row in read_jsonl(pud_path):
        lang = lang_from_row(row)
        if lang in langs:
            records.append(
                {
                    "id": row.get("id"),
                    "lang": lang,
                    "parallel_id": parallel_id_from_row(row),
                    "source": row.get("source", "pud"),
                    "text": row.get("prompt_text") or row.get("text") or "",
                }
            )
    if include_ud:
        for row in read_jsonl(ud_path):
            lang = lang_from_row(row)
            records.append(
                {
                    "id": row.get("id"),
                    "lang": lang,
                    "parallel_id": parallel_id_from_row(row),
                    "source": row.get("source", "ud"),
                    "text": row.get("prompt_text") or row.get("text") or "",
                }
            )

    if args.max_examples_per_lang > 0:
        kept: list[dict] = []
        counts: dict[str, int] = defaultdict(int)
        for record in records:
            if counts[record["lang"]] >= args.max_examples_per_lang:
                continue
            kept.append(record)
            counts[record["lang"]] += 1
        records = kept
    if args.max_examples > 0:
        records = records[: args.max_examples]
    return records


################################################################################
# Model scoring
################################################################################

def parallel_id_from_row(row: dict) -> str:
    row_id = str(row.get("id", ""))
    if "_pud-" in row_id:
        return "pud-" + row_id.split("_pud-", 1)[1]
    return row_id


def model_slug(model_name: str) -> str:
    known = {
        "gpt2": "gpt2",
        "gpt2-xl": "gpt2xl",
        "openai-community/gpt2-xl": "gpt2xl",
        "meta-llama/Llama-3.1-8B": "llama318b",
        "meta-llama/Llama-3.1-8B-Instruct": "llama318binstr",
        "swiss-ai/Apertus-8B-2509": "apertus8b",
        "swiss-ai/Apertus-8B-Instruct-2509": "apertus8binstr",
        "mistralai/Mistral-Nemo-Instruct-2407": "nemo",
        "CohereLabs/aya-23-8B": "aya238b",
        "utter-project/EuroLLM-9B": "eurollm9b",
        "utter-project/EuroLLM-9B-Instruct": "eurollm9binstr",
        "allenai/OLMo-2-1124-7B": "olmo211247b",
    }
    if model_name in known:
        return known[model_name]
    if "Llama-2-7b-hf" in model_name:
        return "llama27b"
    return "".join(ch for ch in model_name.rsplit("/", 1)[-1].lower() if ch.isalnum())


def torch_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def should_prepend_bos(tokenizer, setting: str) -> bool:
    if setting == "true":
        return True
    if setting == "false":
        return False
    return tokenizer.bos_token_id is not None


@torch.no_grad()
def score_batch(model, tokenizer, records: list[dict], *, max_length: int, prepend_bos: bool, device: torch.device) -> list[dict]:
    texts = [r["text"] for r in records]
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=max_length > 0,
        max_length=max_length or None,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if prepend_bos:
        bos = torch.full((input_ids.shape[0], 1), tokenizer.bos_token_id, dtype=input_ids.dtype)
        bos_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)
        attention_mask = torch.cat([bos_mask, attention_mask], dim=1)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    labels = input_ids.masked_fill(attention_mask == 0, -100)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    token_loss = flat_loss.view(shift_labels.shape)
    token_mask = shift_labels != -100
    nll = token_loss.masked_fill(~token_mask, 0.0).sum(dim=1).cpu()
    n_tokens = token_mask.sum(dim=1).cpu()

    rows = []
    for record, nll_value, token_count in zip(records, nll.tolist(), n_tokens.tolist(), strict=True):
        byte_count = len(record["text"].encode("utf-8"))
        rows.append(
            {
                **record,
                "nll_nats": float(nll_value),
                "logprob_nats": -float(nll_value),
                "logprob_bits": -float(nll_value) / math.log(2),
                "n_pred_tokens": int(token_count),
                "n_bytes": int(byte_count),
                "n_chars": int(len(record["text"])),
            }
        )
    return rows


################################################################################
# Aggregation and outputs
################################################################################

def aggregate(rows: list[dict], *, model_name: str, data_source: str, split: str) -> list[dict]:
    groups: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "nll_nats": 0.0,
            "logprob_nats": 0.0,
            "n_bytes": 0,
            "n_chars": 0,
            "n_pred_tokens": 0,
            "n_examples": 0,
        }
    )
    for row in rows:
        for key in ("__overall__", row["lang"]):
            groups[key]["nll_nats"] += row["nll_nats"]
            groups[key]["logprob_nats"] += row["logprob_nats"]
            groups[key]["n_bytes"] += row["n_bytes"]
            groups[key]["n_chars"] += row["n_chars"]
            groups[key]["n_pred_tokens"] += row["n_pred_tokens"]
            groups[key]["n_examples"] += 1

    out = []
    for lang, values in sorted(groups.items()):
        n_bytes = values["n_bytes"]
        bpb = values["nll_nats"] / math.log(2) / n_bytes if n_bytes else math.nan
        n_chars = values["n_chars"]
        n_tokens = values["n_pred_tokens"]
        n_examples = values["n_examples"]
        mean_line_logprob_nats = values["logprob_nats"] / n_examples if n_examples else math.nan
        mean_line_logprob_bits = mean_line_logprob_nats / math.log(2) if n_examples else math.nan
        mean_line_nll_nats = values["nll_nats"] / n_examples if n_examples else math.nan
        nll_bits_per_char = values["nll_nats"] / math.log(2) / n_chars if n_chars else math.nan
        mean_token_nll_nats = values["nll_nats"] / n_tokens if n_tokens else math.nan
        out.append(
            {
                "model_name": model_name,
                "model_slug": model_slug(model_name),
                "data_source": data_source,
                "split": split,
                "lang": "overall" if lang == "__overall__" else lang,
                "bits_per_byte": bpb,
                "nll_bits_per_char": nll_bits_per_char,
                "mean_token_nll_nats": mean_token_nll_nats,
                "mean_line_logprob_nats": mean_line_logprob_nats,
                "mean_line_logprob_bits": mean_line_logprob_bits,
                "mean_line_nll_nats": mean_line_nll_nats,
                "nll_nats": values["nll_nats"],
                "logprob_nats": values["logprob_nats"],
                "n_bytes": int(n_bytes),
                "n_chars": int(n_chars),
                "n_pred_tokens": int(values["n_pred_tokens"]),
                "n_examples": int(n_examples),
            }
        )
    return out


def write_outputs(rows: list[dict], line_rows: list[dict], csv_path: Path, jsonl_path: Path, line_csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    line_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "model_slug",
        "data_source",
        "split",
        "lang",
        "bits_per_byte",
        "nll_bits_per_char",
        "mean_token_nll_nats",
        "mean_line_logprob_nats",
        "mean_line_logprob_bits",
        "mean_line_nll_nats",
        "nll_nats",
        "logprob_nats",
        "n_bytes",
        "n_chars",
        "n_pred_tokens",
        "n_examples",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    line_fieldnames = [
        "model_name",
        "model_slug",
        "data_source",
        "split",
        "id",
        "parallel_id",
        "source",
        "lang",
        "logprob_nats",
        "logprob_bits",
        "nll_nats",
        "n_bytes",
        "n_chars",
        "n_pred_tokens",
        "bits_per_byte",
        "nll_bits_per_char",
        "mean_token_nll_nats",
    ]
    with line_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=line_fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in line_fieldnames} for row in line_rows)


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    args = parse_args()
    records = load_records(args)
    if not records:
        raise ValueError(f"No records loaded for {args.data_source} split={args.split}.")
    print(f"Loaded {len(records)} records for {args.data_source}/{args.split}.")

    all_summary_rows = []
    all_line_rows = []
    for model_name in args.model_name:
        print(f"Loading {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=args.revision, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
        model_kwargs = {
            "revision": args.revision,
            "trust_remote_code": args.trust_remote_code,
            "dtype": torch_dtype(args.dtype),
        }
        if args.device_map:
            model_kwargs["device_map"] = args.device_map
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        if not args.device_map:
            model.to(args.device)
        device = next(model.parameters()).device
        prepend_bos = should_prepend_bos(tokenizer, args.prepend_bos)

        scored_rows = []
        for start in tqdm(range(0, len(records), args.batch_size), desc=f"Scoring {model_slug(model_name)}"):
            batch = records[start : start + args.batch_size]
            scored_rows.extend(score_batch(model, tokenizer, batch, max_length=args.max_length, prepend_bos=prepend_bos, device=device))
        for row in scored_rows:
            row["model_name"] = model_name
            row["model_slug"] = model_slug(model_name)
            row["data_source"] = args.data_source
            row["split"] = args.split
            row["bits_per_byte"] = row["nll_nats"] / math.log(2) / row["n_bytes"] if row["n_bytes"] else math.nan
            row["nll_bits_per_char"] = row["nll_nats"] / math.log(2) / row["n_chars"] if row["n_chars"] else math.nan
            row["mean_token_nll_nats"] = row["nll_nats"] / row["n_pred_tokens"] if row["n_pred_tokens"] else math.nan
        all_line_rows.extend(scored_rows)
        summary_rows = aggregate(scored_rows, model_name=model_name, data_source=args.data_source, split=args.split)
        all_summary_rows.extend(summary_rows)
        overall = next(row for row in summary_rows if row["lang"] == "overall")
        print(f"{model_slug(model_name)} overall BPB: {overall['bits_per_byte']:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_outputs(all_summary_rows, all_line_rows, Path(args.output_csv), Path(args.output_jsonl), Path(args.line_output_csv))
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_jsonl}")
    print(f"Wrote {args.line_output_csv}")


if __name__ == "__main__":
    main()
