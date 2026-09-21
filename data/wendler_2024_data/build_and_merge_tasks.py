#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from .target_string_artifacts import write_target_string_artifacts_for_dir
    from .wendler_extend_utils import (
        EXTENDED_DIR,
        FILTER_MODES,
        MERGED_DIR,
        PROCESSED_DIR,
        deterministic_examples,
        prompt_line,
        read_csv,
        write_csv,
        write_json,
    )
except ImportError:  # pragma: no cover - script entry fallback
    from target_string_artifacts import write_target_string_artifacts_for_dir
    from wendler_extend_utils import (
        EXTENDED_DIR,
        FILTER_MODES,
        MERGED_DIR,
        PROCESSED_DIR,
        deterministic_examples,
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
    parser = argparse.ArgumentParser(description="Build merged task CSVs from processed baseline and extended clean rows.")
    parser.add_argument("--extended-dir", type=Path, default=EXTENDED_DIR)
    parser.add_argument("--merged-dir", type=Path, default=MERGED_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--default-filter-mode", choices=FILTER_MODES, default="none")
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


def build_prompt(rows: list[dict[str, str]], current: dict[str, str], task: str, prompt_seed: int, shots: int) -> str:
    examples = deterministic_examples(rows, current, task, prompt_seed, shots)
    lines: list[str] = []
    for example in examples:
        if task == "copy":
            lines.append(prompt_line(example["lang"], example["word_translation"], example["lang"], example["word_translation"]))
        else:
            lines.append(prompt_line("en", example["word_original"], example["lang"], example["word_translation"]))
    if task == "copy":
        lines.append(prompt_line(current["lang"], current["word_translation"], current["lang"]))
    else:
        lines.append(prompt_line("en", current["word_original"], current["lang"]))
    return "\n".join(lines)


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


def build_cloze_prompt(rows: list[dict[str, str]], current: dict[str, str], prompt_seed: int, shots: int) -> str:
    examples = [
        row["blank_prompt_translation_masked"].strip()
        for row in deterministic_examples(rows, current, "cloze", prompt_seed, shots)
        if row.get("blank_prompt_translation_masked", "").strip()
    ]
    current_masked = current.get("blank_prompt_translation_masked", "").strip()
    return "\n".join(
        [*examples, cloze_query_prompt(current_masked, current.get("lang", ""), current.get("word_translation", ""))]
    )


def build_tasks(
    source_rows: list[dict[str, str]],
    prompt_seed: int,
    shots: int,
    cloze_shots: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    translation: list[dict[str, str]] = []
    copy: list[dict[str, str]] = []
    cloze: list[dict[str, str]] = []
    for idx, row in enumerate(source_rows):
        common = {
            "source_file": row.get("source_file", ""),
            "source_row": row.get("source_row", str(idx)),
            "model_size": "extended",
            "out_token_id": "",
            "out_token_str": row.get("word_translation", ""),
            "latent_token_id": "",
            "latent_token_str": row.get("word_original", ""),
            "in_token_id": "",
            "in_token_str": row.get("word_original", ""),
        }
        translation.append(
            {
                **common,
                "task": "translation",
                "input_lang": "en",
                "target_lang": row.get("lang", ""),
                "prompt": build_prompt(source_rows, row, "translation", prompt_seed, shots),
            }
        )
        copy.append(
            {
                **common,
                "task": "copy",
                "input_lang": row.get("lang", ""),
                "target_lang": row.get("lang", ""),
                "prompt": build_prompt(source_rows, row, "copy", prompt_seed, shots),
            }
        )
        cloze.append(
            {
                "source_file": row.get("source_file", ""),
                "source_row": row.get("source_row", str(idx)),
                "lang": row.get("lang", ""),
                "concept_id": row.get("concept_id", ""),
                "prompt": build_cloze_prompt(source_rows, row, prompt_seed, cloze_shots),
                "target_text": row.get("word_translation", ""),
                "english_text": row.get("word_original", ""),
                "blank_prompt_original": row.get("blank_prompt_original", ""),
                "blank_prompt_translation": row.get("blank_prompt_translation", ""),
                "blank_prompt_translation_masked": row.get("blank_prompt_translation_masked", ""),
                "error": row.get("error", ""),
            }
        )
    return translation, copy, cloze


def merge_baseline(processed_dir: Path, name: str, new_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    baseline_path = processed_dir / f"{name}.csv"
    baseline = read_csv(baseline_path) if baseline_path.exists() else []
    return baseline + new_rows


def write_mode_outputs(args: argparse.Namespace, mode: str) -> dict[str, int]:
    source_dir = args.extended_dir / f"filter_{mode}"
    mode_dir = args.merged_dir / f"filter_{mode}"
    source_path = source_dir / "source_clean.csv"
    source_rows = read_csv(source_path) if source_path.exists() else []
    translation, copy, cloze = build_tasks(source_rows, args.prompt_seed, args.shots, args.cloze_shots)

    merged_translation = merge_baseline(args.processed_dir, "translation", translation)
    merged_copy = merge_baseline(args.processed_dir, "copy", copy)
    merged_cloze = merge_baseline(args.processed_dir, "cloze", cloze)

    write_csv(mode_dir / "translation.csv", merged_translation, TASK_FIELDNAMES)
    write_csv(mode_dir / "copy.csv", merged_copy, TASK_FIELDNAMES)
    write_csv(mode_dir / "cloze.csv", merged_cloze, CLOZE_FIELDNAMES)
    write_target_string_artifacts_for_dir(mode_dir)
    if mode == args.default_filter_mode:
        write_csv(args.merged_dir / "translation.csv", merged_translation, TASK_FIELDNAMES)
        write_csv(args.merged_dir / "copy.csv", merged_copy, TASK_FIELDNAMES)
        write_csv(args.merged_dir / "cloze.csv", merged_cloze, CLOZE_FIELDNAMES)
        write_target_string_artifacts_for_dir(args.merged_dir)
    return {
        "source_clean": len(source_rows),
        "new_translation": len(translation),
        "new_copy": len(copy),
        "new_cloze": len(cloze),
        "translation": len(merged_translation),
        "copy": len(merged_copy),
        "cloze": len(merged_cloze),
    }


def main() -> None:
    args = parse_args()
    summary = {"default_filter_mode": args.default_filter_mode, "prompt_seed": args.prompt_seed, "filters": {}}
    write_target_string_artifacts_for_dir(args.processed_dir)
    for mode in FILTER_MODES:
        summary["filters"][mode] = write_mode_outputs(args, mode)
    processed_summary = args.processed_dir / "summary.json"
    if processed_summary.exists():
        summary["baseline"] = json.loads(processed_summary.read_text(encoding="utf-8"))
    write_json(args.merged_dir / "summary.json", summary)
    print(f"Wrote merged task CSVs to {args.merged_dir}")


if __name__ == "__main__":
    main()
