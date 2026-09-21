#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from .wendler_extend_utils import read_csv, read_jsonl, write_json, write_jsonl
except ImportError:  # pragma: no cover - script entry fallback
    from wendler_extend_utils import read_csv, read_jsonl, write_json, write_jsonl

TASK_NAMES = ("translation", "copy", "cloze")
DEFAULT_SPACE_PREFIX_MARKERS = ("▁",)
COMMON_SPACE_PREFIX_MARKERS = ("▁", "Ġ")
BYTE_FALLBACK_LANGS = {"zh", "ru"}
TOKENIZER_META_PATH = Path(__file__).with_name("tokenizer_meta.json")
START_TOKEN_STRATEGIES = {"vocab_prefix", "tokenizer_encode_prefix"}
EMPTY_QUOTED_TARGET_RE = re.compile(r'(:\s*)""\s*$')
CLOZE_BLANK = "___"


def infer_task_langs_tgtstring(row: dict[str, str], task: str) -> tuple[str, str]:
    """Resolve prompt and target languages from a target-string task row."""
    if task == "translation":
        prompt_lang = row.get("input_lang", "").strip()
        tgt_lang = row.get("target_lang", "").strip()
    elif task == "copy":
        lang = (row.get("target_lang") or row.get("input_lang") or "").strip()
        prompt_lang = lang
        tgt_lang = lang
    elif task == "cloze":
        lang = row.get("lang", "").strip()
        prompt_lang = lang
        tgt_lang = lang
    else:
        raise ValueError(f"Unsupported task: {task}")
    if not prompt_lang or not tgt_lang:
        raise ValueError(f"Missing language metadata for task={task}")
    return prompt_lang, tgt_lang


def prompt_id_from_task_row(row: dict[str, str], task: str) -> str:
    if task == "translation":
        return (
            f"translation:{row.get('input_lang','')}:{row.get('target_lang','')}:"
            f"{row.get('source_file','')}:{row.get('source_row','')}"
        )
    if task == "copy":
        return (
            f"copy:{row.get('target_lang','')}:{row.get('source_file','')}:"
            f"{row.get('source_row','')}"
        )
    if task == "cloze":
        return (
            f"cloze:{row.get('lang','')}:{row.get('concept_id','')}:"
            f"{row.get('source_file','')}:{row.get('source_row','')}"
        )
    raise ValueError(f"Unsupported task: {task}")


def prompt_text_from_task_row(row: dict[str, str], task: str) -> str:
    if task in {"translation", "copy"}:
        prompt = row.get("prompt", "")
        prompt = EMPTY_QUOTED_TARGET_RE.sub(r'\1"', prompt.strip())
    elif task == "cloze":
        prompt = row.get("prompt", "")
        if prompt:
            prompt = prompt.strip()
        else:
            prompt = row.get("blank_prompt_translation_masked", "")
            prompt = prompt.strip()
            blank_idx = prompt.find(CLOZE_BLANK)
            if blank_idx > 0:
                prompt = prompt[:blank_idx]
    else:
        raise ValueError(f"Unsupported task: {task}")
    if not prompt:
        raise ValueError(f"Missing prompt text for task={task}")
    return prompt


def target_text_from_task_row(row: dict[str, str], task: str) -> str:
    if task in {"translation", "copy"}:
        target = row.get("out_token_str", "")
    elif task == "cloze":
        target = row.get("target_text", "")
    else:
        raise ValueError(f"Unsupported task: {task}")
    target = target.strip()
    if not target:
        raise ValueError(f"Missing target text for task={task}")
    return target


def concept_id_from_task_row(row: dict[str, str], task: str) -> str:
    if task == "cloze":
        concept_id = row.get("concept_id", "").strip()
        if concept_id:
            return concept_id
    concept_id = (row.get("latent_token_str") or "").strip()
    if concept_id:
        return concept_id
    concept_id = (row.get("in_token_str") or "").strip()
    if concept_id:
        return concept_id
    concept_id = (row.get("tgt_text") or row.get("target_text") or row.get("out_token_str") or "").strip()
    if concept_id:
        return concept_id
    raise ValueError(f"Unable to infer concept_id for task={task}")


def target_string_path_for_task_csv(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_target_string.jsonl")


def safe_tokenizer_artifact_name(tokenizer_name: str) -> str:
    return (
        tokenizer_name.strip()
        .replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def target_string_start_token_dir_for_tokenizer(directory: Path, tokenizer_name: str) -> Path:
    return directory / "target_string_start_tokens" / safe_tokenizer_artifact_name(tokenizer_name)


def target_string_start_token_path_for_task_csv(csv_path: Path, tokenizer_name: str) -> Path:
    return target_string_start_token_dir_for_tokenizer(csv_path.parent, tokenizer_name) / (
        f"{csv_path.stem}_start_tokens.jsonl"
    )


def target_string_concept_start_token_path(directory: Path, tokenizer_name: str) -> Path:
    return target_string_start_token_dir_for_tokenizer(directory, tokenizer_name) / "concept_start_tokens.jsonl"


def token_prefixes(token_str: str) -> list[str]:
    return [token_str[:idx] for idx in range(1, len(token_str) + 1)]


def start_token_candidate_strings(
    token_str: str,
    *,
    space_prefix_markers: tuple[str, ...] = DEFAULT_SPACE_PREFIX_MARKERS,
) -> list[str]:
    prefixes = token_prefixes(token_str)
    candidates: list[str] = []
    for marker in space_prefix_markers:
        candidates.extend(f"{marker}{prefix}" for prefix in prefixes)
    candidates.extend(prefixes)
    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def detect_space_prefix_markers(tokenizer: Any) -> tuple[str, ...]:
    vocab = tokenizer.get_vocab()
    detected = [
        marker
        for marker in COMMON_SPACE_PREFIX_MARKERS
        if any(token.startswith(marker) and len(token) > len(marker) for token in vocab)
    ]
    return tuple(detected)


def load_tokenizer_meta(path: Path = TOKENIZER_META_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "tokenizers": {}}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def tokenizer_names_from_meta(path: Path = TOKENIZER_META_PATH) -> list[str]:
    meta = load_tokenizer_meta(path)
    tokenizers = meta.get("tokenizers", {})
    if not isinstance(tokenizers, dict):
        raise ValueError(f"Malformed tokenizer metadata in {path}: expected a tokenizers object.")
    names = [
        str(name)
        for name, entry in tokenizers.items()
        if isinstance(entry, dict) and entry.get("status") == "ok"
    ]
    if not names:
        raise ValueError(
            f"No usable tokenizer entries found in {path}. "
            "Run data/wendler_2024_data/refresh_tokenizer_meta.py first."
        )
    return names


def infer_start_token_strategy(space_prefix_markers: tuple[str, ...]) -> str:
    if "Ġ" in space_prefix_markers and "▁" not in space_prefix_markers:
        return "tokenizer_encode_prefix"
    return "vocab_prefix"


def tokenizer_start_token_meta(
    tokenizer_name: str,
    *,
    meta_path: Path = TOKENIZER_META_PATH,
) -> tuple[tuple[str, ...], str]:
    meta = load_tokenizer_meta(meta_path)
    entry = meta.get("tokenizers", {}).get(tokenizer_name)
    if not isinstance(entry, dict):
        raise ValueError(
            f"Missing tokenizer metadata for {tokenizer_name!r} in {meta_path}. "
            "Run data/wendler_2024_data/refresh_tokenizer_meta.py for this tokenizer first."
        )
    if entry.get("status") != "ok":
        raise ValueError(
            f"Tokenizer metadata for {tokenizer_name!r} in {meta_path} is not usable: "
            f"status={entry.get('status')!r}, error={entry.get('error', '')!r}."
        )
    markers = entry.get("space_prefix_markers")
    if not isinstance(markers, list):
        raise ValueError(
            f"Tokenizer metadata for {tokenizer_name!r} is missing space_prefix_markers."
        )
    strategy = entry.get("start_token_strategy")
    if strategy not in START_TOKEN_STRATEGIES:
        raise ValueError(
            f"Tokenizer metadata for {tokenizer_name!r} has invalid start_token_strategy={strategy!r}."
        )
    return tuple(str(marker) for marker in markers), str(strategy)


def add_unique_token_id(token_ids: list[int], token_strings: list[str], token_id: int, token_str: str) -> None:
    if token_id not in token_ids:
        token_ids.append(int(token_id))
        token_strings.append(token_str)


def unicode_prefix_token_string(token_str: str) -> str | None:
    if not token_str:
        return None
    first_byte = token_str.encode("utf-8")[0]
    return f"<0x{first_byte:02X}>"


def wendler_start_tokens_for_text(
    tokenizer: Any,
    token_str: str,
    lang: str,
    *,
    space_prefix_markers: tuple[str, ...] = DEFAULT_SPACE_PREFIX_MARKERS,
    start_token_strategy: str = "vocab_prefix",
) -> dict[str, Any]:
    vocab = tokenizer.get_vocab()
    id_to_token = {int(token_id): token for token, token_id in vocab.items()}
    token_ids: list[int] = []
    token_strings: list[str] = []
    if start_token_strategy == "vocab_prefix":
        for candidate in start_token_candidate_strings(
            token_str,
            space_prefix_markers=space_prefix_markers,
        ):
            if candidate in vocab:
                add_unique_token_id(token_ids, token_strings, int(vocab[candidate]), candidate)
    elif start_token_strategy == "tokenizer_encode_prefix":
        for prefix in token_prefixes(token_str):
            for candidate in (prefix, f" {prefix}"):
                encoded = tokenizer(candidate, add_special_tokens=False)["input_ids"]
                if encoded:
                    token_id = int(encoded[0])
                    add_unique_token_id(token_ids, token_strings, token_id, id_to_token.get(token_id, str(token_id)))
    else:
        raise ValueError(f"Unsupported start_token_strategy={start_token_strategy!r}")
    if lang in BYTE_FALLBACK_LANGS:
        byte_token = unicode_prefix_token_string(token_str)
        if byte_token is not None and byte_token in vocab:
            add_unique_token_id(token_ids, token_strings, int(vocab[byte_token]), byte_token)
    return {
        "text": token_str,
        "lang": lang,
        "start_token_strategy": start_token_strategy,
        "start_token_ids": token_ids,
        "start_token_strs": token_strings,
        "start_token_count": len(token_ids),
    }


def wendler_start_tokens_for_texts(
    tokenizer: Any,
    text_langs: list[tuple[str, str]],
    *,
    space_prefix_markers: tuple[str, ...] = DEFAULT_SPACE_PREFIX_MARKERS,
    start_token_strategy: str = "vocab_prefix",
    encode_batch_size: int = 4096,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Compute Start(w) entries for many (text, lang) pairs with shared tokenizer work."""
    unique_text_langs = list(dict.fromkeys((str(text), str(lang)) for text, lang in text_langs))
    if start_token_strategy == "vocab_prefix":
        return {
            (text, lang): wendler_start_tokens_for_text(
                tokenizer,
                text,
                lang,
                space_prefix_markers=space_prefix_markers,
                start_token_strategy=start_token_strategy,
            )
            for text, lang in unique_text_langs
        }
    if start_token_strategy != "tokenizer_encode_prefix":
        raise ValueError(f"Unsupported start_token_strategy={start_token_strategy!r}")

    vocab = tokenizer.get_vocab()
    id_to_token = {int(token_id): token for token, token_id in vocab.items()}
    token_ids_by_key: dict[tuple[str, str], list[int]] = {key: [] for key in unique_text_langs}
    token_strs_by_key: dict[tuple[str, str], list[str]] = {key: [] for key in unique_text_langs}
    candidates: list[str] = []
    candidate_keys: list[tuple[str, str]] = []
    for text, lang in unique_text_langs:
        for prefix in token_prefixes(text):
            for candidate in (prefix, f" {prefix}"):
                candidates.append(candidate)
                candidate_keys.append((text, lang))

    for start in range(0, len(candidates), encode_batch_size):
        batch = candidates[start : start + encode_batch_size]
        encoded_batch = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for offset, encoded in enumerate(encoded_batch):
            if not encoded:
                continue
            key = candidate_keys[start + offset]
            token_id = int(encoded[0])
            add_unique_token_id(
                token_ids_by_key[key],
                token_strs_by_key[key],
                token_id,
                id_to_token.get(token_id, str(token_id)),
            )

    for text, lang in unique_text_langs:
        if lang in BYTE_FALLBACK_LANGS:
            byte_token = unicode_prefix_token_string(text)
            if byte_token is not None and byte_token in vocab:
                add_unique_token_id(
                    token_ids_by_key[(text, lang)],
                    token_strs_by_key[(text, lang)],
                    int(vocab[byte_token]),
                    byte_token,
                )

    return {
        (text, lang): {
            "text": text,
            "lang": lang,
            "start_token_strategy": start_token_strategy,
            "start_token_ids": token_ids_by_key[(text, lang)],
            "start_token_strs": token_strs_by_key[(text, lang)],
            "start_token_count": len(token_ids_by_key[(text, lang)]),
        }
        for text, lang in unique_text_langs
    }


def build_target_string_records(task_rows: list[dict[str, str]], task: str) -> list[dict[str, Any]]:
    concept_entries: dict[str, dict[str, set[str]]] = {}
    normalized_rows: list[dict[str, Any]] = []
    for row in task_rows:
        prompt_lang, tgt_lang = infer_task_langs_tgtstring(row, task)
        prompt_id = prompt_id_from_task_row(row, task)
        prompt = prompt_text_from_task_row(row, task)
        tgt_text = target_text_from_task_row(row, task)
        concept_id = concept_id_from_task_row(row, task)
        concept_entries.setdefault(concept_id, {}).setdefault(tgt_text, set()).add(tgt_lang)
        normalized_rows.append(
            {
                "prompt_id": prompt_id,
                "concept_id": concept_id,
                "task": task,
                "prompt": prompt,
                "prompt_lang": prompt_lang,
                "tgt_lang": tgt_lang,
                "tgt_text": tgt_text,
            }
        )

    records: list[dict[str, Any]] = []
    for row in normalized_rows:
        grouped = [
            {"text": text, "langs": sorted(langs)}
            for text, langs in sorted(concept_entries[row["concept_id"]].items(), key=lambda item: item[0])
        ]
        unique_only = [entry for entry in grouped if len(entry["langs"]) == 1]
        record = dict(row)
        record.update(
            {
                "menu_strings_grouped": grouped,
                "menu_strings_unique": unique_only,
                "menu_has_ambiguity": any(len(entry["langs"]) > 1 for entry in grouped),
                "menu_group_count": len(grouped),
                "menu_unique_count": len(unique_only),
            }
        )
        records.append(record)
    return records


def load_target_string_records(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def union_start_tokens(entries: list[dict[str, Any]]) -> tuple[list[int], list[str]]:
    token_ids: list[int] = []
    token_strings: list[str] = []
    for entry in entries:
        for token_id, token_str in zip(entry["start_token_ids"], entry["start_token_strs"]):
            add_unique_token_id(token_ids, token_strings, int(token_id), str(token_str))
    return token_ids, token_strings


def build_concept_start_token_records(
    target_string_records: list[dict[str, Any]],
    tokenizer: Any,
    tokenizer_name: str,
    *,
    space_prefix_markers: tuple[str, ...] = DEFAULT_SPACE_PREFIX_MARKERS,
    start_token_strategy: str = "vocab_prefix",
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    concept_text_langs: dict[str, dict[str, set[str]]] = {}
    for record in target_string_records:
        concept_id = str(record["concept_id"])
        text_langs = concept_text_langs.setdefault(concept_id, {})
        for group in record["menu_strings_grouped"]:
            text = str(group["text"])
            text_langs.setdefault(text, set()).update(str(lang) for lang in group["langs"])

    start_token_inputs: list[tuple[str, str]] = []
    for concept_id, text_langs in concept_text_langs.items():
        start_token_inputs.append((concept_id, "en"))
        for text, langs in text_langs.items():
            start_token_inputs.extend((text, lang) for lang in langs)
    start_token_cache = wendler_start_tokens_for_texts(
        tokenizer,
        start_token_inputs,
        space_prefix_markers=space_prefix_markers,
        start_token_strategy=start_token_strategy,
    )

    records: list[dict[str, Any]] = []
    concept_items = sorted(concept_text_langs.items(), key=lambda item: item[0])
    if show_progress:
        concept_items = tqdm(
            concept_items,
            desc=f"{safe_tokenizer_artifact_name(tokenizer_name)} concepts",
            unit="concept",
            leave=False,
        )
    for concept_id, text_langs in concept_items:
        latent_text = concept_id
        latent_tokens = start_token_cache[(latent_text, "en")]
        latent_token_id_set = set(latent_tokens["start_token_ids"])
        grouped: list[dict[str, Any]] = []
        for text, langs in sorted(text_langs.items(), key=lambda item: item[0]):
            group_langs = sorted(langs)
            lang_token_entries = [
                dict(start_token_cache[(text, lang)])
                for lang in group_langs
            ]
            for entry in lang_token_entries:
                overlaps_latent = bool(latent_token_id_set.intersection(entry["start_token_ids"]))
                entry["overlaps_latent_start_tokens"] = overlaps_latent
                entry["wendler_keep"] = bool(
                    entry["start_token_ids"]
                    and latent_tokens["start_token_ids"]
                    and (entry["lang"] == "en" or not overlaps_latent)
                )
            group_token_ids, group_token_strs = union_start_tokens(lang_token_entries)
            grouped.append(
                {
                    "text": text,
                    "langs": group_langs,
                    "start_token_ids": group_token_ids,
                    "start_token_strs": group_token_strs,
                    "start_token_count": len(group_token_ids),
                    "per_lang_start_tokens": lang_token_entries,
                    "overlaps_latent_start_tokens": bool(latent_token_id_set.intersection(group_token_ids)),
                }
            )
        records.append(
            {
                "concept_id": concept_id,
                "tokenizer_name": tokenizer_name,
                "space_prefix_markers": list(space_prefix_markers),
                "start_token_strategy": start_token_strategy,
                "byte_fallback_langs": sorted(BYTE_FALLBACK_LANGS),
                "latent_text": latent_text,
                "latent_lang": "en",
                "latent_start_token_ids": latent_tokens["start_token_ids"],
                "latent_start_token_strs": latent_tokens["start_token_strs"],
                "latent_start_token_count": latent_tokens["start_token_count"],
                "menu_start_tokens_grouped": grouped,
            }
        )
    return records


def materialize_start_token_records_from_concepts(
    target_string_records: list[dict[str, Any]],
    concept_start_token_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    concepts = {str(record["concept_id"]): record for record in concept_start_token_records}
    records: list[dict[str, Any]] = []
    for record in target_string_records:
        concept_id = str(record["concept_id"])
        if concept_id not in concepts:
            raise ValueError(f"Missing concept start-token record for concept_id={concept_id}")
        concept_record = concepts[concept_id]
        concept_groups = {
            str(group["text"]): group
            for group in concept_record["menu_start_tokens_grouped"]
        }
        latent_token_ids = list(concept_record["latent_start_token_ids"])
        grouped = []
        target_group = None
        target_lang_entry = None
        for group in record["menu_strings_grouped"]:
            text = str(group["text"])
            group_langs = [str(lang) for lang in group["langs"]]
            if text not in concept_groups:
                raise ValueError(f"Missing concept menu text={text!r} for concept_id={concept_id}")
            concept_group = concept_groups[text]
            per_lang_source = {
                str(entry["lang"]): entry
                for entry in concept_group["per_lang_start_tokens"]
            }
            lang_token_entries = []
            for lang in group_langs:
                if lang not in per_lang_source:
                    raise ValueError(
                        f"Missing concept menu lang={lang!r} for concept_id={concept_id}, text={text!r}"
                    )
                lang_token_entries.append(per_lang_source[lang])
            group_token_ids, group_token_strs = union_start_tokens(lang_token_entries)
            group_record = {
                "text": text,
                "langs": group_langs,
                "start_token_ids": group_token_ids,
                "start_token_strs": group_token_strs,
                "start_token_count": len(group_token_ids),
                "per_lang_start_tokens": lang_token_entries,
                "overlaps_latent_start_tokens": bool(set(latent_token_ids).intersection(group_token_ids)),
            }
            grouped.append(group_record)
            if text == str(record["tgt_text"]) and str(record["tgt_lang"]) in group_langs:
                target_group = group_record
                target_lang_entry = per_lang_source[str(record["tgt_lang"])]
        if target_group is None or target_lang_entry is None:
            raise ValueError(f"Could not locate target group for prompt_id={record['prompt_id']}")
        target_start_token_ids = list(target_group["start_token_ids"])
        records.append(
            {
                "prompt_id": record["prompt_id"],
                "concept_id": record["concept_id"],
                "task": record["task"],
                "tokenizer_name": concept_record["tokenizer_name"],
                "space_prefix_markers": concept_record["space_prefix_markers"],
                "start_token_strategy": concept_record["start_token_strategy"],
                "byte_fallback_langs": concept_record["byte_fallback_langs"],
                "latent_text": concept_record["latent_text"],
                "latent_lang": concept_record["latent_lang"],
                "latent_start_token_ids": latent_token_ids,
                "latent_start_token_strs": list(concept_record["latent_start_token_strs"]),
                "latent_start_token_count": int(concept_record["latent_start_token_count"]),
                "tgt_text": record["tgt_text"],
                "tgt_lang": record["tgt_lang"],
                "tgt_start_token_ids": target_start_token_ids,
                "tgt_start_token_strs": list(target_group["start_token_strs"]),
                "tgt_start_token_count": len(target_start_token_ids),
                "tgt_lang_start_token_ids": list(target_lang_entry["start_token_ids"]),
                "tgt_lang_start_token_strs": list(target_lang_entry["start_token_strs"]),
                "tgt_lang_start_token_count": int(target_lang_entry["start_token_count"]),
                "tgt_overlaps_latent_start_tokens": bool(target_lang_entry["overlaps_latent_start_tokens"]),
                "wendler_keep": bool(target_lang_entry["wendler_keep"]),
                "menu_start_tokens_grouped": grouped,
            }
        )
    return records


def build_start_token_records(
    target_string_records: list[dict[str, Any]],
    tokenizer: Any,
    tokenizer_name: str,
    *,
    space_prefix_markers: tuple[str, ...] = DEFAULT_SPACE_PREFIX_MARKERS,
    start_token_strategy: str = "vocab_prefix",
) -> list[dict[str, Any]]:
    concept_records = build_concept_start_token_records(
        target_string_records,
        tokenizer,
        tokenizer_name,
        space_prefix_markers=space_prefix_markers,
        start_token_strategy=start_token_strategy,
    )
    return materialize_start_token_records_from_concepts(target_string_records, concept_records)


def load_start_token_records(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def merge_start_token_records(
    target_string_records: list[dict[str, Any]],
    start_token_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_prompt_id = {str(record["prompt_id"]): record for record in start_token_records}
    merged: list[dict[str, Any]] = []
    for record in target_string_records:
        prompt_id = str(record["prompt_id"])
        if prompt_id not in by_prompt_id:
            raise ValueError(f"Missing start-token record for prompt_id={prompt_id}")
        merged.append({**record, **by_prompt_id[prompt_id]})
    return merged


def write_start_token_artifact_for_csv(
    csv_path: Path,
    tokenizer: Any,
    tokenizer_name: str,
) -> Path | None:
    task = csv_path.stem
    if task not in TASK_NAMES or not csv_path.exists():
        return None
    space_prefix_markers, start_token_strategy = tokenizer_start_token_meta(tokenizer_name)
    target_string_path = target_string_path_for_task_csv(csv_path)
    if target_string_path.exists():
        target_string_records = load_target_string_records(target_string_path)
    else:
        target_string_records = build_target_string_records(read_csv(csv_path), task)
    records = build_start_token_records(
        target_string_records,
        tokenizer,
        tokenizer_name,
        space_prefix_markers=space_prefix_markers,
        start_token_strategy=start_token_strategy,
    )
    out_path = target_string_start_token_path_for_task_csv(csv_path, tokenizer_name)
    write_jsonl(out_path, records)
    return out_path


def target_string_records_for_task_csv(csv_path: Path) -> list[dict[str, Any]] | None:
    task = csv_path.stem
    if task not in TASK_NAMES or not csv_path.exists():
        return None
    target_string_path = target_string_path_for_task_csv(csv_path)
    if target_string_path.exists():
        return load_target_string_records(target_string_path)
    return build_target_string_records(read_csv(csv_path), task)


def write_start_token_artifacts_for_dir(
    directory: Path,
    tokenizer: Any,
    tokenizer_name: str,
    *,
    show_progress: bool = False,
) -> list[Path]:
    space_prefix_markers, start_token_strategy = tokenizer_start_token_meta(tokenizer_name)
    records_by_task: dict[str, list[dict[str, Any]]] = {}
    combined_records: list[dict[str, Any]] = []
    for task in TASK_NAMES:
        task_records = target_string_records_for_task_csv(directory / f"{task}.csv")
        if task_records is None:
            continue
        records_by_task[task] = task_records
        combined_records.extend(task_records)
    concept_records = build_concept_start_token_records(
        combined_records,
        tokenizer,
        tokenizer_name,
        space_prefix_markers=space_prefix_markers,
        start_token_strategy=start_token_strategy,
        show_progress=show_progress,
    )
    concept_path = target_string_concept_start_token_path(directory, tokenizer_name)
    write_jsonl(concept_path, concept_records)

    written: list[Path] = []
    task_items = list(records_by_task.items())
    if show_progress:
        task_items = tqdm(
            task_items,
            desc=f"{safe_tokenizer_artifact_name(tokenizer_name)} task artifacts",
            unit="task",
            leave=False,
        )
    for task, target_string_records in task_items:
        records = materialize_start_token_records_from_concepts(
            target_string_records,
            concept_records,
        )
        out_path = target_string_start_token_path_for_task_csv(directory / f"{task}.csv", tokenizer_name)
        write_jsonl(out_path, records)
        written.append(out_path)
    write_json(
        target_string_start_token_dir_for_tokenizer(directory, tokenizer_name) / "manifest.json",
        {
            "tokenizer_name": tokenizer_name,
            "tokenizer_class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer.get_vocab()),
            "space_prefix_markers": list(space_prefix_markers),
            "space_prefix_markers_source": "tokenizer_meta",
            "start_token_strategy": start_token_strategy,
            "start_token_strategy_source": "tokenizer_meta",
            "byte_fallback_langs": sorted(BYTE_FALLBACK_LANGS),
            "concept_artifact": concept_path.name,
            "concept_count": len(concept_records),
            "tasks": [path.name for path in written],
        },
    )
    return [concept_path, *written]


def write_target_string_artifact_for_csv(csv_path: Path) -> Path | None:
    task = csv_path.stem
    if task not in TASK_NAMES or not csv_path.exists():
        return None
    rows = read_csv(csv_path)
    records = build_target_string_records(rows, task)
    out_path = target_string_path_for_task_csv(csv_path)
    write_jsonl(out_path, records)
    return out_path


def write_target_string_artifacts_for_dir(directory: Path) -> list[Path]:
    written: list[Path] = []
    for task in TASK_NAMES:
        out_path = write_target_string_artifact_for_csv(directory / f"{task}.csv")
        if out_path is not None:
            written.append(out_path)
    return written
