"""Load prompts, select scoring positions, and write evaluation artifacts."""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from args import (
    infer_prompt_lang_from_id,
    is_synthetic_data_source,
    synthetic_base_task,
    synthetic_target_lang_filter,
)
from metrics import (
    compute_langdist_metrics,
    compute_langdist_prompt_pivot_mask,
)
from data.wendler_2024_data.target_string_artifacts import (
    build_target_string_records,
    infer_task_langs_tgtstring,
    load_start_token_records,
    load_target_string_records,
    merge_start_token_records,
    prompt_id_from_task_row,
    prompt_text_from_task_row,
    target_string_start_token_path_for_task_csv,
    target_string_path_for_task_csv,
)


################################################################################
# Prompt data and loading
################################################################################

@dataclass
class PromptData:
    """Prompt text and language metadata consumed by evaluation."""

    prompts: list[str]
    prompt_ids: list[str]
    prompt_langs: list[str | None]
    task_langs_by_prompt: list[tuple[str, ...]]
    surface_tokens_by_prompt: list[list[str] | None]
    target_string_records: dict[str, dict[str, Any]] | None = None


def load_synthetic_prompt_data(
    path: Path,
    data_source: str,
    max_prompts: int | None,
    max_prompts_per_lang: int | None = None,
    selected_task_langs: set[str] | None = None,
    start_token_artifact_key: str | None = None,
) -> PromptData:
    """Load target-string prompts and their task-language metadata from CSV."""
    prompts: list[str] = []
    prompt_ids: list[str] = []
    prompt_langs: list[str | None] = []
    task_langs_by_prompt: list[tuple[str, ...]] = []
    surface_tokens_by_prompt: list[list[str] | None] = []
    target_string_records: dict[str, dict[str, Any]] = {}
    per_lang_counts: dict[str, int] = {}
    base_task = synthetic_base_task(data_source)
    target_lang_filter = synthetic_target_lang_filter(data_source)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    companion_path = target_string_path_for_task_csv(path)
    if companion_path.exists():
        records = load_target_string_records(companion_path)
    else:
        records = build_target_string_records(rows, base_task)
    if start_token_artifact_key:
        start_token_path = target_string_start_token_path_for_task_csv(path, start_token_artifact_key)
        if not start_token_path.exists():
            raise ValueError(
                "Missing target-string start-token artifact for "
                f"model/tokenizer key={start_token_artifact_key!r}, task={base_task}: {start_token_path}. "
                "Build it during preprocessing with build_translation_pairs_and_filter.py after adding "
                "the tokenizer to data/wendler_2024_data/tokenizer_meta.json."
            )
        records = merge_start_token_records(records, load_start_token_records(start_token_path))
    all_target_string_records = {str(record["prompt_id"]): record for record in records}
    for row in rows:
        source_lang, target_lang = infer_task_langs_tgtstring(row, base_task)
        source_lang = source_lang or None
        target_lang = target_lang or None
        # Keep only rows whose source and target belong to the selected subset.
        if selected_task_langs is not None:
            if source_lang not in selected_task_langs:
                continue
            if target_lang is not None and target_lang not in selected_task_langs:
                continue
        if target_lang_filter is not None and target_lang != target_lang_filter:
            continue
        prompt_id = prompt_id_from_task_row(row, base_task)
        prompt_text = prompt_text_from_task_row(row, base_task)
        if max_prompts_per_lang is not None:
            lang_key = source_lang or target_lang or ""
            count = per_lang_counts.get(lang_key, 0)
            if count >= max_prompts_per_lang:
                continue
            per_lang_counts[lang_key] = count + 1

        target_lang_key = target_lang or source_lang
        if not target_lang_key:
            raise ValueError(f"Synthetic {data_source} row is missing language metadata.")

        prompts.append(prompt_text)
        prompt_ids.append(prompt_id)
        prompt_langs.append(source_lang)
        task_langs = tuple(sorted({lang for lang in [source_lang, target_lang] if lang}))
        task_langs_by_prompt.append(task_langs)
        surface_tokens_by_prompt.append(None)
        if prompt_id not in all_target_string_records:
            raise ValueError(f"Missing target-string companion record for prompt_id={prompt_id}")
        target_string_records[prompt_id] = all_target_string_records[prompt_id]
        if max_prompts is not None and len(prompts) >= max_prompts:
            break

    if not prompts:
        filter_detail = "" if target_lang_filter is None else f" for target_lang={target_lang_filter}"
        raise ValueError(f"No synthetic prompts found in {path}{filter_detail}")
    return PromptData(
        prompts=prompts,
        prompt_ids=prompt_ids,
        prompt_langs=prompt_langs,
        task_langs_by_prompt=task_langs_by_prompt,
        surface_tokens_by_prompt=surface_tokens_by_prompt,
        target_string_records=target_string_records,
    )


def infer_task_langs_openended(
    obj: Mapping[str, Any],
    prompt_id: str | None,
) -> tuple[str | None, str | None]:
    """Resolve source and target roles for shared task-language aggregation.

    Current open-ended datasets use the same language for both roles. Keeping
    the pair matches the target-string loader and lets downstream code collect
    all task-valid languages through one interface. Older PUD records without
    language fields fall back to the language prefix in the prompt ID.
    """
    source_lang = obj.get("source_lang") or obj.get("lang")
    target_lang = obj.get("target_lang") or source_lang
    if source_lang is None and prompt_id is not None:
        inferred = infer_prompt_lang_from_id(str(prompt_id))
        source_lang = inferred
        target_lang = inferred if target_lang is None else target_lang
    return source_lang, target_lang


def load_jsonl_prompt_data(
    paths: Sequence[Path],
    max_prompts: int | None,
    max_prompts_per_lang: int | None = None,
    *,
    selected_task_langs: set[str] | None = None,
) -> PromptData:
    """Load open-ended prompts and language metadata from JSONL files."""
    prompts: list[str] = []
    prompt_ids: list[str] = []
    prompt_langs: list[str | None] = []
    task_langs_by_prompt: list[tuple[str, ...]] = []
    surface_tokens_by_prompt: list[list[str] | None] = []
    per_lang_counts: dict[str, int] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("prompt_text") or obj.get("text")
                prompt_id = obj.get("id")
                if text is None:
                    raise ValueError("Each JSONL row must include 'prompt_text' or 'text'.")
                source_lang, target_lang = infer_task_langs_openended(
                    obj,
                    str(prompt_id) if prompt_id is not None else None,
                )
                # Filter the selected dataset subset before applying per-language caps.
                if selected_task_langs is not None and source_lang not in selected_task_langs:
                    continue
                if max_prompts_per_lang is not None:
                    if prompt_id is None:
                        raise ValueError("Each JSONL row must include 'id' when max_prompts_per_lang is used.")
                    lang = source_lang or infer_prompt_lang_from_id(str(prompt_id))
                    count = per_lang_counts.get(lang, 0)
                    if count >= max_prompts_per_lang:
                        continue
                    per_lang_counts[lang] = count + 1
                prompts.append(text)
                prompt_ids.append("" if prompt_id is None else str(prompt_id))
                prompt_langs.append(source_lang)
                task_langs = tuple(sorted({lang for lang in [source_lang, target_lang] if lang}))
                task_langs_by_prompt.append(task_langs)
                surface_tokens = obj.get("surface_tokens")
                if not isinstance(surface_tokens, list):
                    raise ValueError("Each prompt row must include 'surface_tokens' as a list.")
                surface_tokens_by_prompt.append(list(surface_tokens))
                if max_prompts is not None and len(prompts) >= max_prompts:
                    break
        if max_prompts is not None and len(prompts) >= max_prompts:
            break
    if not prompts:
        joined = ", ".join(str(path) for path in paths)
        raise ValueError(f"No prompts found in {joined}")
    return PromptData(
        prompts=prompts,
        prompt_ids=prompt_ids,
        prompt_langs=prompt_langs,
        task_langs_by_prompt=task_langs_by_prompt,
        surface_tokens_by_prompt=surface_tokens_by_prompt,
    )


################################################################################
# Prompt formatting and token alignment
################################################################################

def compute_chunk_ids_for_offsets(
    offsets: Sequence[tuple[int, int]],
    chunk_spans: Sequence[tuple[int, int]],
) -> list[int | None]:
    """Map each token offset span to a chunk id; special tokens remain None."""
    chunk_ids: list[int | None] = []
    chunk_idx = 0
    for token_start, token_end in offsets:
        if token_end <= token_start:
            chunk_ids.append(None)
            continue
        while chunk_idx < len(chunk_spans) and chunk_spans[chunk_idx][1] <= token_start:
            chunk_idx += 1
        assigned = None
        probe_idx = chunk_idx
        while probe_idx < len(chunk_spans):
            chunk_start, chunk_end = chunk_spans[probe_idx]
            if chunk_start >= token_end:
                break
            if token_start < chunk_end and token_end > chunk_start:
                assigned = probe_idx
            probe_idx += 1
        chunk_ids.append(assigned)
    return chunk_ids


def compute_surface_token_spans(
    text: str,
    surface_tokens: Sequence[str],
    *,
    start_pos: int = 0,
) -> list[tuple[int, int]]:
    """Align UD/PUD-style surface tokens to raw-text spans in order."""
    spans: list[tuple[int, int]] = []
    cursor = start_pos
    for token in surface_tokens:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        start = text.find(token, cursor)
        if start < 0:
            raise ValueError(f"Could not align surface token {token!r} within text starting at {cursor}.")
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    return spans


def format_prompts_for_tokenizer(
    tokenizer,
    prompts: Sequence[str],
) -> tuple[list[str], list[int]]:
    """Format prompts with the tokenizer chat template when present."""
    if not getattr(tokenizer, "chat_template", None):
        return list(prompts), [0 for _ in prompts]

    formatted_prompts: list[str] = []
    content_start_offsets: list[int] = []
    bos_token = getattr(tokenizer, "bos_token", None)
    for prompt in prompts:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if bos_token and formatted.startswith(bos_token):
            formatted = formatted[len(bos_token):]
        start = formatted.find(prompt)
        if start < 0:
            raise ValueError("Could not locate raw prompt inside chat-formatted prompt.")
        formatted_prompts.append(formatted)
        content_start_offsets.append(start)
    return formatted_prompts, content_start_offsets


def get_prompt_content_end_token_positions(
    tokenizer,
    prompts: Sequence[str],
    raw_prompts: Sequence[str],
    content_start_offsets: Sequence[int],
) -> list[int]:
    """Return the token position ending each raw prompt inside its formatted prompt."""
    if not (len(prompts) == len(raw_prompts) == len(content_start_offsets)):
        raise ValueError("prompts, raw_prompts, and content_start_offsets must have the same length.")

    positions: list[int] = []
    for prompt, raw_prompt, content_start in zip(prompts, raw_prompts, content_start_offsets):
        content_end = int(content_start) + len(raw_prompt)
        try:
            enc = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
        except (NotImplementedError, TypeError):
            enc = tokenizer(prompt, add_special_tokens=True)
        offsets = enc.get("offset_mapping")
        pos = None
        if offsets is not None:
            for idx, offset in enumerate(offsets):
                start, end = int(offset[0]), int(offset[1])
                if start == end:
                    continue
                if start >= int(content_start) and end <= content_end:
                    pos = idx
        if pos is None:
            prefix_ids = tokenizer(prompt[:content_end], add_special_tokens=True)["input_ids"]
            full_ids = enc["input_ids"]
            if not prefix_ids:
                raise ValueError("Could not identify a token position for the raw prompt content.")
            pos = min(len(prefix_ids) - 1, len(full_ids) - 1)
        positions.append(int(pos))
    return positions


def should_anchor_to_raw_prompt_content(data_source: str) -> bool:
    """Return whether scoring should use the raw prompt end inside a formatted prompt.

    Synthetic Wendler prompts are next-token prefixes, so the intended scoring
    position is the end of the raw task prefix even when chat formatting appends
    generation-control text. PUD/INCLUDE open-ended decoding and repr methods
    score at the model's formatted generation position instead.
    """
    return is_synthetic_data_source(data_source)


def get_surface_word_ids_for_prompts(
    tokenizer,
    prompts: list[str],
    surface_tokens_by_prompt: Sequence[Sequence[str] | None] | None = None,
    content_start_offsets: Sequence[int] | None = None,
) -> list[list[int] | None]:
    """Return tokenizer-aligned chunk ids from prompt-level surface tokens."""
    chunk_ids = []
    for prompt_idx, prompt in enumerate(prompts):
        if getattr(tokenizer, "is_fast", False):
            # Keep special tokens so word ids stay aligned with the traced latent sequence.
            enc = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
            offsets = enc.get("offset_mapping")
            if offsets is not None:
                surface_tokens = None if surface_tokens_by_prompt is None else surface_tokens_by_prompt[prompt_idx]
                if not surface_tokens:
                    raise ValueError("surface_tokens_by_prompt is required for tokenizer-aligned chunk ids.")
                start_pos = 0 if content_start_offsets is None else content_start_offsets[prompt_idx]
                chunk_spans = compute_surface_token_spans(prompt, surface_tokens, start_pos=start_pos)
                chunk_ids.append(
                    compute_chunk_ids_for_offsets(offsets, chunk_spans)
                )
                continue
        chunk_ids.append(None)
    return chunk_ids


################################################################################
# Scoring-position selection
################################################################################

def get_ordered_word_ids(word_ids: Sequence[int | None] | None) -> list[int]:
    """Return unique non-None ids in first-appearance order."""
    ordered: list[int] = []
    if word_ids is None:
        return ordered
    seen = set()
    for wid in word_ids:
        if wid is None or wid in seen:
            continue
        ordered.append(int(wid))
        seen.add(int(wid))
    return ordered


def get_word_end_token_positions(word_ids: Sequence[int | None] | None) -> list[int]:
    """Return the final token index for each chunk/word id in sequence order."""
    if word_ids is None:
        return []
    ordered = get_ordered_word_ids(word_ids)
    if not ordered:
        return []
    last_positions: dict[int, int] = {}
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        last_positions[int(wid)] = idx
    return [last_positions[wid] for wid in ordered]


def _resolve_fraction(mode: str) -> float:
    if mode == "frac_50":
        return 0.50
    if mode == "frac_75":
        return 0.75
    raise ValueError(f"Unsupported fractional mode: {mode}")


def select_fractional_token_position(
    length: int,
    mode: str,
    *,
    word_ids: Sequence[int | None] | None = None,
) -> int:
    """Select a fractional sequence position, preferring surface-word ends."""
    if length <= 0:
        raise ValueError("length must be positive.")
    frac = _resolve_fraction(mode)
    word_end_positions = get_word_end_token_positions(word_ids)
    if word_end_positions:
        word_idx = min(int(frac * max(len(word_end_positions) - 1, 0)), len(word_end_positions) - 1)
        return word_end_positions[word_idx]
    return min(int(frac * max(length - 1, 0)), length - 1)


def select_scoring_token_position(
    seq_len: int,
    mode: str,
    *,
    word_ids: Sequence[int | None] | None = None,
) -> int:
    """Select the token position represented by a scoring aggregation mode."""
    if mode == "last_token":
        return seq_len - 1
    if mode in {"frac_50", "frac_75"}:
        return select_fractional_token_position(seq_len, mode, word_ids=word_ids)
    raise ValueError(f"Unsupported decoding token aggregation mode for position selection: {mode}")


def select_rollout_token_positions(
    seq_len: int,
    mode: str,
    *,
    rollout_k: int,
    rollout_word_cnt: int,
    word_ids: Sequence[int | None] | None = None,
) -> list[int]:
    """Select prompt positions whose next-token predictions form a rollout."""
    if rollout_k < 0:
        raise ValueError("rollout_k must be non-negative.")
    if rollout_word_cnt < 0:
        raise ValueError("rollout_word_cnt must be non-negative.")
    anchor = select_scoring_token_position(seq_len, mode, word_ids=word_ids)
    max_predict_pos = seq_len - 2
    if anchor > max_predict_pos:
        return []
    if rollout_word_cnt > 0:
        if word_ids is None:
            end = min(anchor + rollout_word_cnt, max_predict_pos)
            return list(range(anchor, end + 1))
        future_word_ids: list[int] = []
        seen = set()
        for target_pos in range(anchor + 1, seq_len):
            wid = word_ids[target_pos]
            if wid is None or wid in seen:
                continue
            future_word_ids.append(int(wid))
            seen.add(int(wid))
            if len(future_word_ids) >= rollout_word_cnt + 1:
                break
        if not future_word_ids:
            return [anchor]
        selected_word_ids = set(future_word_ids)
        selected = [
            pos
            for pos in range(anchor, max_predict_pos + 1)
            if word_ids[pos + 1] is not None and int(word_ids[pos + 1]) in selected_word_ids
        ]
        return selected or [anchor]
    end = min(anchor + rollout_k, max_predict_pos)
    return list(range(anchor, end + 1))


################################################################################
# Metric row builders
################################################################################

def append_mean_layer_metric_rows(
    rows: list[dict[str, Any]],
    *,
    checkpoint_id: str,
    method: str,
    n_prompts: int,
    layer_indices: Sequence[int],
    metrics: Mapping[str, torch.Tensor],
) -> None:
    """Append one prompt-mean row per metric and layer."""
    for key, tensor in metrics.items():
        values = tensor.detach().cpu().numpy()
        for layer_offset, layer_idx in enumerate(layer_indices):
            rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "method": method,
                    "layer": int(layer_idx),
                    "D": None,
                    "H": None,
                    "A": None,
                    "P": None,
                    "n_prompts": n_prompts,
                    key: float(values[:, layer_offset].mean()),
                }
            )


def build_langdist_summary_rows(
    *,
    checkpoint_id: str,
    method: str,
    layer_indices: Sequence[int],
    langdist: Mapping[str, torch.Tensor],
    n_prompts: int,
    decoding_mapping: str | None = None,
    decoding_lid_backend: str | None = None,
    decoding_lens: str | None = None,
) -> list[dict[str, Any]]:
    """Build per-layer aggregate dominance and entropy rows from a langdist."""
    metrics = compute_langdist_metrics(langdist)
    dominance = metrics["dominance"].detach().cpu().numpy()
    entropy = metrics["entropy"].detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for layer_offset, layer_idx in enumerate(layer_indices):
        row = {
            "checkpoint_id": checkpoint_id,
            "method": method,
        }
        if decoding_mapping is not None:
            row["decoding_mapping"] = decoding_mapping
        if decoding_lid_backend is not None:
            row["decoding_lid_backend"] = decoding_lid_backend
        if decoding_lens is not None:
            row["decoding_lens"] = decoding_lens
        row.update(
            {
                "layer": int(layer_idx),
                "D": float(dominance[:, layer_offset].mean()),
                "H": float(entropy[:, layer_offset].mean()),
                "A": None,
                "P": None,
                "n_prompts": n_prompts,
            }
        )
        rows.append(row)
    return rows


def build_pivot_summary_rows(
    *,
    checkpoint_id: str,
    method: str,
    layer_indices: Sequence[int],
    langdist: Mapping[str, torch.Tensor],
    n_prompts: int,
    task_langs_by_prompt: Sequence[Sequence[str]],
    decoding_mapping: str | None = None,
    decoding_lid_backend: str | None = None,
    decoding_lens: str | None = None,
) -> list[dict[str, Any]]:
    """Build per-layer aggregate pivot-rate rows from langdist."""
    pivot = compute_langdist_prompt_pivot_mask(
        langdist,
        task_langs_by_prompt,
    ).detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    for layer_offset, layer_idx in enumerate(layer_indices):
        row = {
            "checkpoint_id": checkpoint_id,
            "method": method,
        }
        if decoding_mapping is not None:
            row["decoding_mapping"] = decoding_mapping
        if decoding_lid_backend is not None:
            row["decoding_lid_backend"] = decoding_lid_backend
        if decoding_lens is not None:
            row["decoding_lens"] = decoding_lens
        row.update(
            {
                "layer": int(layer_idx),
                "D": None,
                "H": None,
                "A": None,
                "P": float(pivot[:, layer_offset].mean()),
                "n_prompts": n_prompts,
            }
        )
        rows.append(row)
    return rows


def build_prompt_metric_rows(
    *,
    checkpoint_id: str,
    method: str,
    layer_indices: Sequence[int],
    langdist: Mapping[str, torch.Tensor],
    task_langs_by_prompt: Sequence[Sequence[str]],
    prompt_ids: Sequence[str] | None = None,
    prompt_langs: Sequence[str | None] | None = None,
    decoding_mapping: str | None = None,
    decoding_lid_backend: str | None = None,
    decoding_lens: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a prompt-by-layer langdist into long-form prompt metrics."""
    langs = sorted(langdist.keys())
    stacked = torch.stack([langdist[lang] for lang in langs], dim=-1)
    argmax = stacked.argmax(dim=-1).detach().cpu().numpy()
    metrics = compute_langdist_metrics(langdist)
    dominance = metrics["dominance"].detach().cpu().numpy()
    entropy = metrics["entropy"].detach().cpu().numpy()
    pivot = compute_langdist_prompt_pivot_mask(
        langdist,
        task_langs_by_prompt,
    ).detach().cpu().numpy()

    n_prompts = int(stacked.shape[0])
    prompt_ids = list(prompt_ids) if prompt_ids else [None] * n_prompts
    prompt_langs = list(prompt_langs) if prompt_langs else [None] * n_prompts

    rows: list[dict[str, Any]] = []
    for prompt_idx in range(n_prompts):
        for layer_offset, layer_idx in enumerate(layer_indices):
            row = {
                "checkpoint_id": checkpoint_id,
                "method": method,
                "decoding_mapping": decoding_mapping,
                "decoding_lid_backend": decoding_lid_backend,
                "decoding_lens": decoding_lens,
                "prompt_idx": int(prompt_idx),
                "prompt_id": prompt_ids[prompt_idx],
                "prompt_lang": prompt_langs[prompt_idx],
                "layer": int(layer_idx),
                "dominant_lang": langs[int(argmax[prompt_idx, layer_offset])],
                "dominance": float(dominance[prompt_idx, layer_offset]),
                "entropy": float(entropy[prompt_idx, layer_offset]),
                "pivot": bool(pivot[prompt_idx, layer_offset]),
            }
            rows.append(row)
    return rows


################################################################################
# Language-distribution artifact building
################################################################################


def build_method_langdist_artifact(
    langdist: Mapping[str, torch.Tensor],
    *,
    decoding_mapping: str | None = None,
    decoding_lid_backend: str | None = None,
    decoding_lens: str | None = None,
    target_score_semantics: Any = None,
    target_mass_semantics: Any = None,
    target_dist_semantics: Any = None,
    target_string_scoring_mode: Any = None,
    target_string_execution_path: Any = None,
    extra_langdist_views: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    """Build one method entry for the full language-distribution artifact."""
    langs = sorted(langdist)
    method_artifact: dict[str, Any] = {
        "langs": langs,
        "probs": torch.stack([langdist[lang] for lang in langs], dim=-1)
        .detach()
        .cpu()
        .to(torch.float16),
        "decoding_mapping": decoding_mapping,
        "decoding_lid_backend": decoding_lid_backend,
        "decoding_lens": decoding_lens,
    }

    optional_metadata = {
        "target_score_semantics": target_score_semantics,
        "target_mass_semantics": target_mass_semantics,
        "target_dist_semantics": target_dist_semantics,
        "target_string_scoring_mode": target_string_scoring_mode,
        "target_string_execution_path": target_string_execution_path,
    }
    method_artifact.update(
        {
            key: value
            for key, value in optional_metadata.items()
            if value is not None
        }
    )
    if extra_langdist_views:
        for view_name, view_langdist in extra_langdist_views.items():
            if not view_langdist:
                continue
            view_langs = sorted(view_langdist)
            method_artifact[view_name] = {
                "langs": view_langs,
                "probs": torch.stack([view_langdist[lang] for lang in view_langs], dim=-1)
                .detach()
                .cpu()
                .to(torch.float16),
            }
    return method_artifact


################################################################################
# Output writing
################################################################################

def prepare_output_root(cfg: Mapping[str, Any]) -> Path:
    """Create the run output directory and save the resolved config."""
    output_root = Path(cfg["output_dir"]) / cfg["exp_id"]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.json").write_text(
        json.dumps(dict(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_root


def write_revision_artifacts(
    *,
    output_root: Path,
    revision: str,
    cfg: Mapping[str, Any],
    full_lang_probs_payload: Mapping[str, Any],
    target_string_artifact: Mapping[str, Any] | None,
) -> None:
    """Write per-revision probability and target-string artifacts."""
    if cfg["save_full_lang_probs"] and full_lang_probs_payload["methods"]:
        torch.save(full_lang_probs_payload, output_root / f"lang_probs_{revision}.pt")
    if target_string_artifact is not None:
        torch.save(
            target_string_artifact,
            output_root / f"target_string_details_{revision}.pt",
        )


def write_run_outputs(
    *,
    output_root: Path,
    cfg: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    prompt_metric_rows: Sequence[Mapping[str, Any]],
    repr_meta_payload: Mapping[str, Any] | None,
    decoding_meta_payload: Mapping[str, Any] | None,
    decoded_samples_all: Sequence[Mapping[str, Any]],
) -> Path:
    """Write final metrics, metadata, and decoded text artifacts."""
    parquet_path = output_root / "metrics.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)

    if prompt_metric_rows:
        prompt_df = pd.DataFrame(prompt_metric_rows)
        prompt_df.to_parquet(output_root / "prompt_metrics.parquet", index=False)
    if repr_meta_payload:
        (output_root / "repr_meta.json").write_text(
            json.dumps(repr_meta_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if decoding_meta_payload:
        (output_root / "decoding_meta.json").write_text(
            json.dumps(decoding_meta_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if decoded_samples_all:
        decoded_path = output_root / "decoded_samples.jsonl"
        with decoded_path.open("w", encoding="utf-8") as f:
            for row in decoded_samples_all:
                f.write(json.dumps(row) + "\n")
    return parquet_path
