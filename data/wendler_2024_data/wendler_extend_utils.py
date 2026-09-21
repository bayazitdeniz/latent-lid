#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / "llm-latent-language"
PROCESSED_DIR = BASE_DIR / "processed"
EXTENDED_DIR = BASE_DIR / "extended"
MERGED_DIR = BASE_DIR / "merged"
PROVIDER_DIR = EXTENDED_DIR / "provider_artifacts"

FILTER_MODES = ("none", "not_match_english", "not_match_all")

TARGET_LANGUAGES: dict[str, dict[str, str]] = {
    "ar": {"name": "Arabic", "deepl": "AR"},
    "hi": {"name": "Hindi", "deepl": "HI"},
    "ko": {"name": "Korean", "deepl": "KO"},
    "ja": {"name": "Japanese", "deepl": "JA"},
    "id": {"name": "Indonesian", "deepl": "ID"},
    "is": {"name": "Icelandic", "deepl": "IS"},
    "sv": {"name": "Swedish", "deepl": "SV"},
    "es": {"name": "Spanish", "deepl": "ES"},
    "gl": {"name": "Galician", "deepl": "GL"},
    "it": {"name": "Italian", "deepl": "IT"},
    "pt_br": {"name": "Portuguese Brazilian", "deepl": "PT-BR"},
    "cs": {"name": "Czech", "deepl": "CS"},
    "pl": {"name": "Polish", "deepl": "PL"},
    "tr": {"name": "Turkish", "deepl": "TR"},
    "fi": {"name": "Finnish", "deepl": "FI"},
    "th": {"name": "Thai", "deepl": "TH"},
    "uk": {"name": "Ukrainian", "deepl": "UK"},
    "bg": {"name": "Bulgarian", "deepl": "BG"},
    "sr": {"name": "Serbian", "deepl": "SR"},
    "ur": {"name": "Urdu", "deepl": "UR"},
    "fa": {"name": "Persian", "deepl": "FA"},
    "mr": {"name": "Marathi", "deepl": "MR"},
    # DeepL-supported candidates not in the current README target set:
    # "da": {"name": "Danish", "deepl": "DA"},
    # "el": {"name": "Greek", "deepl": "EL"},
    # "et": {"name": "Estonian", "deepl": "ET"},
    # "hu": {"name": "Hungarian", "deepl": "HU"},
    # "lt": {"name": "Lithuanian", "deepl": "LT"},
    # "lv": {"name": "Latvian", "deepl": "LV"},
    # "nb": {"name": "Norwegian Bokmal", "deepl": "NB"},
    # "nl": {"name": "Dutch", "deepl": "NL"},
    # "ro": {"name": "Romanian", "deepl": "RO"},
    # "sk": {"name": "Slovak", "deepl": "SK"},
    # "sl": {"name": "Slovenian", "deepl": "SL"},
    # "vi": {"name": "Vietnamese", "deepl": "VI"},
}

LANG_LABELS = {
    "en": "English",
    "de": "Deutsch",
    "fr": "Francais",
    "ru": "Русский",
    "zh": "中文",
    **{lang: meta["name"] for lang, meta in TARGET_LANGUAGES.items()},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stable_chunks(rows: list[dict[str, str]], chunk_size: int) -> Iterable[list[dict[str, str]]]:
    for idx in range(0, len(rows), chunk_size):
        yield rows[idx : idx + chunk_size]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_token(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").casefold(), flags=re.UNICODE)


def has_token_prefix(a: str, b: str) -> bool:
    norm_a = normalize_token(a)
    norm_b = normalize_token(b)
    if not norm_a or not norm_b:
        return False
    return norm_a.startswith(norm_b) or norm_b.startswith(norm_a)


ANSWER_MARKER_RE = re.compile(
    r"(answer|cevap|yanıt|yanit|答え|答えなさい|答案|回答|答)", re.IGNORECASE
)


def replace_span(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]


def mask_cloze(text: str, word: str) -> str:
    if not text:
        return ""
    answer_match = ANSWER_MARKER_RE.search(text)
    answer_start = answer_match.start() if answer_match else len(text)

    dangling_japanese_quote = re.match(r"^([^」]{1,40})」", text)
    if dangling_japanese_quote:
        return replace_span(
            text,
            dangling_japanese_quote.start(1),
            dangling_japanese_quote.end(1),
            "___",
        )

    quote_patterns = [
        (re.compile(r'"([^"]+)"'), '"___"'),
        (re.compile(r"“([^”]+)”"), "“___”"),
        (re.compile(r"「([^」]+)」"), "「___」"),
        (re.compile(r"『([^』]+)』"), "『___』"),
    ]
    for pattern, replacement in quote_patterns:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        before_answer = [match for match in matches if match.start() < answer_start]
        if before_answer:
            match = before_answer[0]
            return replace_span(text, match.start(), match.end(), replacement)

    if word:
        before_answer_text = text[:answer_start]
        word_match = re.search(re.escape(word), before_answer_text, re.IGNORECASE)
        if word_match:
            return replace_span(text, word_match.start(), word_match.end(), "___")
        word_match = re.search(re.escape(word), text, re.IGNORECASE)
        if word_match:
            return replace_span(text, word_match.start(), word_match.end(), "___")
    for pattern, replacement in quote_patterns:
        match = pattern.search(text)
        if match:
            return replace_span(text, match.start(), match.end(), replacement)
    return text


def deterministic_examples(
    rows: list[dict[str, str]],
    current: dict[str, str],
    target_key: str,
    prompt_seed: int,
    k: int,
) -> list[dict[str, str]]:
    candidates = [row for row in rows if row.get("concept_id") != current.get("concept_id")]
    rng = random.Random(f"{prompt_seed}:{current.get('concept_id')}:{target_key}")
    rng.shuffle(candidates)
    return candidates[:k]


def prompt_line(input_lang: str, input_text: str, target_lang: str, target_text: str | None = None) -> str:
    prefix = f'{LANG_LABELS.get(input_lang, input_lang)}: "{input_text}" - {LANG_LABELS.get(target_lang, target_lang)}: "'
    if target_text is None:
        return prefix
    return f'{prefix}{target_text}"'


def default_seed_path() -> Path:
    return REPO_DIR / "data" / "langs" / "zh" / "clean.csv"
