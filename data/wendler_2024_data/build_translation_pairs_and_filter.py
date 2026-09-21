#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm

try:
    from .target_string_artifacts import (
        tokenizer_names_from_meta,
        write_start_token_artifacts_for_dir,
        write_target_string_artifacts_for_dir,
    )
    from .wendler_extend_utils import (
        EXTENDED_DIR,
        PROCESSED_DIR,
        prompt_line,
        read_csv,
        write_csv,
        write_json,
    )
except ImportError:  # pragma: no cover - script entry fallback
    from target_string_artifacts import (
        tokenizer_names_from_meta,
        write_start_token_artifacts_for_dir,
        write_target_string_artifacts_for_dir,
    )
    from wendler_extend_utils import (
        EXTENDED_DIR,
        PROCESSED_DIR,
        prompt_line,
        read_csv,
        write_csv,
        write_json,
    )


TASK_FIELDNAMES = [
    "source_file",
    "source_row",
    "task",
    "model_size",
    "input_lang",
    "target_lang",
    "prompt",
    "out_token_id",
    "out_token_str",
    "latent_token_id",
    "latent_token_str",
    "in_token_id",
    "in_token_str",
]

CLOZE_FIELDNAMES = [
    "source_file",
    "source_row",
    "lang",
    "concept_id",
    "prompt",
    "target_text",
    "english_text",
    "blank_prompt_original",
    "blank_prompt_translation",
    "blank_prompt_translation_masked",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build common-concept synthetic tasks: all directed pairwise translation prompts, "
            "plus balanced copy and cloze rows over concepts shared by every language."
        )
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--extended-dir", type=Path, default=EXTENDED_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/wendler_2024_data/common69"),
    )
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument(
        "--shots",
        type=int,
        default=4,
        help=(
            "Number of completed in-context examples before the query. "
            "Wendler-style translation/copy prompts contain 4 answered examples plus a fifth query line."
        ),
    )
    parser.add_argument(
        "--cloze-shots",
        type=int,
        default=2,
        help="Number of completed cloze demonstrations before the query.",
    )
    return parser.parse_args()


def source_rows_by_language(processed_dir: Path, extended_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    by_lang: dict[str, dict[str, dict[str, str]]] = {}
    for path in [processed_dir / "source_clean.csv", extended_dir / "source_clean.csv"]:
        for row in read_csv(path):
            lang = row["lang"].strip()
            concept_id = (row.get("concept_id") or row.get("word_original") or "").strip()
            if not lang or not concept_id:
                raise ValueError(f"Missing lang/concept_id in {path}: {row}")
            by_lang.setdefault(lang, {})[concept_id] = row
    return by_lang


def common_concepts(by_lang: dict[str, dict[str, dict[str, str]]]) -> list[str]:
    if not by_lang:
        return []
    common = set.intersection(*(set(rows) for rows in by_lang.values()))
    return sorted(common)


def pair_examples(
    *,
    concepts: list[str],
    src_rows: dict[str, dict[str, str]],
    tgt_rows: dict[str, dict[str, str]],
    current_concept: str,
    src_lang: str,
    tgt_lang: str,
    prompt_seed: int,
    shots: int,
) -> list[dict[str, str]]:
    candidates = [
        {
            "concept_id": concept,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "src_text": src_rows[concept]["word_translation"],
            "tgt_text": tgt_rows[concept]["word_translation"],
        }
        for concept in concepts
        if concept != current_concept
    ]
    rng = random.Random(f"{prompt_seed}:translation:{src_lang}:{tgt_lang}:{current_concept}")
    rng.shuffle(candidates)
    return candidates[:shots]


def build_pair_prompt(
    *,
    concepts: list[str],
    src_rows: dict[str, dict[str, str]],
    tgt_rows: dict[str, dict[str, str]],
    current_concept: str,
    src_lang: str,
    tgt_lang: str,
    prompt_seed: int,
    shots: int,
) -> str:
    examples = pair_examples(
        concepts=concepts,
        src_rows=src_rows,
        tgt_rows=tgt_rows,
        current_concept=current_concept,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        prompt_seed=prompt_seed,
        shots=shots,
    )
    lines = [
        prompt_line(example["src_lang"], example["src_text"], example["tgt_lang"], example["tgt_text"])
        for example in examples
    ]
    lines.append(prompt_line(src_lang, src_rows[current_concept]["word_translation"], tgt_lang))
    return "\n".join(lines)


def build_translation_rows(
    by_lang: dict[str, dict[str, dict[str, str]]],
    concepts: list[str],
    *,
    prompt_seed: int,
    shots: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    langs = sorted(by_lang)
    for src_lang in langs:
        for tgt_lang in langs:
            if src_lang == tgt_lang:
                continue
            src_rows = by_lang[src_lang]
            tgt_rows = by_lang[tgt_lang]
            for concept_idx, concept in enumerate(concepts):
                src_row = src_rows[concept]
                tgt_row = tgt_rows[concept]
                rows.append(
                    {
                        "source_file": "pairwise_common_concepts",
                        "source_row": str(concept_idx),
                        "task": "translation",
                        "model_size": "pairwise_common",
                        "input_lang": src_lang,
                        "target_lang": tgt_lang,
                        "prompt": build_pair_prompt(
                            concepts=concepts,
                            src_rows=src_rows,
                            tgt_rows=tgt_rows,
                            current_concept=concept,
                            src_lang=src_lang,
                            tgt_lang=tgt_lang,
                            prompt_seed=prompt_seed,
                            shots=shots,
                        ),
                        "out_token_id": "",
                        "out_token_str": tgt_row["word_translation"],
                        "latent_token_id": "",
                        "latent_token_str": concept,
                        "in_token_id": "",
                        "in_token_str": src_row["word_translation"],
                    }
                )
    return rows


def build_copy_prompt(
    *,
    concepts: list[str],
    lang_rows: dict[str, dict[str, str]],
    current_concept: str,
    lang: str,
    prompt_seed: int,
    shots: int,
) -> str:
    candidates = [
        {
            "concept_id": concept,
            "text": lang_rows[concept]["word_translation"],
        }
        for concept in concepts
        if concept != current_concept
    ]
    rng = random.Random(f"{prompt_seed}:copy:{lang}:{current_concept}")
    rng.shuffle(candidates)
    lines = [
        prompt_line(lang, example["text"], lang, example["text"])
        for example in candidates[:shots]
    ]
    lines.append(prompt_line(lang, lang_rows[current_concept]["word_translation"], lang))
    return "\n".join(lines)


def build_copy_rows(
    by_lang: dict[str, dict[str, dict[str, str]]],
    concepts: list[str],
    *,
    prompt_seed: int,
    shots: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for lang in sorted(by_lang):
        lang_rows = by_lang[lang]
        for concept_idx, concept in enumerate(concepts):
            row = lang_rows[concept]
            rows.append(
                {
                    "source_file": "common_concepts",
                    "source_row": str(concept_idx),
                    "task": "copy",
                    "model_size": "common",
                    "input_lang": lang,
                    "target_lang": lang,
                    "prompt": build_copy_prompt(
                        concepts=concepts,
                        lang_rows=lang_rows,
                        current_concept=concept,
                        lang=lang,
                        prompt_seed=prompt_seed,
                        shots=shots,
                    ),
                    "out_token_id": "",
                    "out_token_str": row["word_translation"],
                    "latent_token_id": "",
                    "latent_token_str": concept,
                    "in_token_id": "",
                    "in_token_str": row["word_translation"],
                }
            )
    return rows


def cloze_query_prompt(masked_prompt: str, lang: str, target_text: str = "") -> str:
    if "：" in masked_prompt:
        return masked_prompt.split("：", 1)[0] + ': "'
    if ":" in masked_prompt:
        return masked_prompt.split(":", 1)[0] + ': "'
    marker_match = re.search(r"(答え(?:は|なさい)?[「\"])", masked_prompt)
    if marker_match:
        return masked_prompt[: marker_match.end()]
    target_text = target_text.strip()
    if target_text:
        blank_idx = masked_prompt.find("___")
        target_idx = masked_prompt.find(target_text, max(blank_idx, 0))
        if target_idx > 0:
            return masked_prompt[:target_idx]
    blank_idx = masked_prompt.find("___")
    if blank_idx == 0:
        return ""
    if blank_idx > 0:
        return masked_prompt[:blank_idx]
    raise ValueError(f"Cannot build cloze query prompt for lang={lang}: {masked_prompt!r}")


def build_cloze_prompt(
    *,
    concepts: list[str],
    lang_rows: dict[str, dict[str, str]],
    current_concept: str,
    lang: str,
    prompt_seed: int,
    shots: int,
) -> str:
    candidates = [
        lang_rows[concept]
        for concept in concepts
        if concept != current_concept and lang_rows[concept].get("blank_prompt_translation_masked", "").strip()
    ]
    rng = random.Random(f"{prompt_seed}:cloze:{lang}:{current_concept}")
    rng.shuffle(candidates)
    examples = [row["blank_prompt_translation_masked"].strip() for row in candidates[:shots]]
    current_row = lang_rows[current_concept]
    current = current_row["blank_prompt_translation_masked"].strip()
    return "\n".join([*examples, cloze_query_prompt(current, lang, current_row.get("word_translation", ""))])


def build_cloze_rows(
    by_lang: dict[str, dict[str, dict[str, str]]],
    concepts: list[str],
    *,
    prompt_seed: int,
    shots: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for lang in sorted(by_lang):
        lang_rows = by_lang[lang]
        for concept_idx, concept in enumerate(concepts):
            row = lang_rows[concept]
            rows.append(
                {
                    "source_file": "common_concepts",
                    "source_row": str(concept_idx),
                    "lang": lang,
                    "concept_id": concept,
                    "prompt": build_cloze_prompt(
                        concepts=concepts,
                        lang_rows=lang_rows,
                        current_concept=concept,
                        lang=lang,
                        prompt_seed=prompt_seed,
                        shots=shots,
                    ),
                    "target_text": row["word_translation"],
                    "english_text": row["word_original"],
                    "blank_prompt_original": row.get("blank_prompt_original", ""),
                    "blank_prompt_translation": row.get("blank_prompt_translation", ""),
                    "blank_prompt_translation_masked": row.get("blank_prompt_translation_masked", ""),
                    "error": row.get("error", ""),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    by_lang = source_rows_by_language(args.processed_dir, args.extended_dir)
    concepts = common_concepts(by_lang)
    translation_rows = build_translation_rows(by_lang, concepts, prompt_seed=args.prompt_seed, shots=args.shots)
    copy_rows = build_copy_rows(by_lang, concepts, prompt_seed=args.prompt_seed, shots=args.shots)
    cloze_rows = build_cloze_rows(by_lang, concepts, prompt_seed=args.prompt_seed, shots=args.cloze_shots)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "translation.csv", translation_rows, TASK_FIELDNAMES)
    write_csv(args.out_dir / "copy.csv", copy_rows, TASK_FIELDNAMES)
    write_csv(args.out_dir / "cloze.csv", cloze_rows, CLOZE_FIELDNAMES)
    target_string_paths = write_target_string_artifacts_for_dir(args.out_dir)
    start_token_paths: dict[str, list[str]] = {}
    tokenizer_names = tokenizer_names_from_meta()
    for tokenizer_name in tqdm(tokenizer_names, desc="Tokenizer start-token artifacts", unit="tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        written = write_start_token_artifacts_for_dir(
            args.out_dir,
            tokenizer,
            tokenizer_name,
            show_progress=True,
        )
        start_token_paths[tokenizer_name] = [str(path) for path in written]
    write_json(
        args.out_dir / "summary.json",
        {
            "task": "translation",
            "language_count": len(by_lang),
            "languages": sorted(by_lang),
            "common_concept_count": len(concepts),
            "common_concepts": concepts,
            "directed_pair_count": len(by_lang) * (len(by_lang) - 1),
            "rows": {
                "translation": len(translation_rows),
                "copy": len(copy_rows),
                "cloze": len(cloze_rows),
            },
            "prompt_seed": args.prompt_seed,
            "shots": args.shots,
            "cloze_shots": args.cloze_shots,
            "target_string_artifacts": [str(path) for path in target_string_paths],
            "start_token_artifacts": start_token_paths,
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "languages": len(by_lang),
                "common_concepts": len(concepts),
                "rows": {
                    "translation": len(translation_rows),
                    "copy": len(copy_rows),
                    "cloze": len(cloze_rows),
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
