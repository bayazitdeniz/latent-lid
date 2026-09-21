"""Score synthetic target-string menus with Start(w) or teacher forcing."""

import gc
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

from lenses import UnembedInfo, prepare_unembed_info
from latents import collect_position_latents, collect_sequence_latents

PROB_EPS = 1e-12
PROGRESS_IS_TTY = sys.stderr.isatty()


################################################################################
# Result containers
################################################################################

@dataclass
class TargetStringResult:
    """Language scores, diagnostics, and artifacts from target-string scoring."""

    lang_probs_dec: dict[str, torch.Tensor]
    lang_mass_dec: dict[str, torch.Tensor]
    lang_dist_dec: dict[str, torch.Tensor]
    lang_mass_key: str
    lang_dist_key: str
    menu_lang_probs_unique_only: dict[str, torch.Tensor]
    menu_ambiguous_mass: torch.Tensor
    menu_unique_mass: torch.Tensor
    prompt_target_records: list[dict[str, Any]]
    target_string_artifact: dict[str, Any]
    decoding_meta: dict[str, Any]
    tgt_sum_logprob: torch.Tensor | None = None
    tgt_prob: torch.Tensor | None = None
    tgt_mean_logprob: torch.Tensor | None = None
    tgt_mean_prob: torch.Tensor | None = None
    tgt_first_token_vocab_entropy_bits: torch.Tensor | None = None
    tgt_vocab_entropy_bits: torch.Tensor | None = None
    tgt_token_count_cpu: list[int] | None = None


@dataclass
class TeacherForcedCandidates:
    """Candidate sequences and per-prompt score buffers for teacher forcing."""

    sequences: list[str]
    metadata: list[dict[str, Any]]
    sequence_token_lengths: list[int]
    group_sum_logprobs_by_prompt: list[torch.Tensor]
    group_first_token_logprobs_by_prompt: list[torch.Tensor]
    group_token_counts_by_prompt: list[list[int]]


@dataclass
class TeacherForcedScores:
    """Target-level scores and token diagnostics from teacher forcing."""

    tgt_sum_logprob: torch.Tensor
    tgt_token_count: torch.Tensor
    tgt_tokens_by_prompt: list[list[int] | None]
    tgt_token_strs_by_prompt: list[list[str] | None]
    tgt_step_logprobs_by_prompt: list[torch.Tensor | None]
    tgt_step_probs_by_prompt: list[torch.Tensor | None]
    tgt_step_vocab_entropy_bits_by_prompt: list[torch.Tensor | None]


################################################################################
# Shared target-string helpers
################################################################################

def resolve_prompt_target_string_records(
    *,
    prompt_ids: Sequence[str],
    target_string_records: Mapping[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return precomputed target-string records in prompt order."""
    if target_string_records is None:
        raise ValueError("Precomputed target-string records are required for target-string scoring.")
    prompt_target_records: list[dict[str, Any]] = []
    for prompt_id in prompt_ids:
        if prompt_id not in target_string_records:
            raise ValueError(f"Missing target-string companion record for prompt_id={prompt_id}.")
        prompt_target_records.append(target_string_records[prompt_id])
    return prompt_target_records


def get_menu_langs(prompt_target_records: Sequence[dict[str, Any]]) -> list[str]:
    """Return all languages covered by grouped target-string menu entries."""
    return sorted(
        {
            lang
            for record in prompt_target_records
            for group in record["menu_strings_grouped"]
            for lang in group["langs"]
        }
    )


def build_target_string_record_base(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build fields shared by target-string artifact records across scoring modes."""
    return {
        "prompt_id": record["prompt_id"],
        "concept_id": record["concept_id"],
        "prompt_lang": record["prompt_lang"],
        "tgt_lang": record["tgt_lang"],
        "tgt_text": record["tgt_text"],
        "menu_strings_grouped": record["menu_strings_grouped"],
        "menu_strings_unique": record["menu_strings_unique"],
    }


def build_target_string_artifact(
    *,
    checkpoint_id: str,
    layer_indices: Sequence[int],
) -> dict[str, Any]:
    """Build the common target-string artifact shell."""
    return {
        "checkpoint_id": checkpoint_id,
        "layer_indices": [int(layer_idx) for layer_idx in layer_indices],
        "records": [],
    }


def add_menu_mass_summary(
    decoding_meta: dict[str, Any],
    *,
    menu_ambiguous_mass: torch.Tensor,
    menu_unique_mass: torch.Tensor,
) -> None:
    """Add shared menu ambiguity/uniqueness summaries to decoding metadata."""
    decoding_meta.update(
        {
            "mean_menu_ambiguous_mass_by_layer": (
                menu_ambiguous_mass.mean(dim=0).detach().cpu().tolist()
            ),
            "mean_menu_unique_mass_by_layer": (
                menu_unique_mass.mean(dim=0).detach().cpu().tolist()
            ),
        }
    )


def finite_exp(logprobs: torch.Tensor) -> torch.Tensor:
    """Exponentiate finite log-probs and map non-finite entries to zero."""
    probs = torch.exp(logprobs)
    return torch.where(torch.isfinite(logprobs), probs, torch.zeros_like(probs))


def build_menu_lang_masses(
    *,
    prompt_target_records: Sequence[dict[str, Any]],
    menu_group_probs: Sequence[torch.Tensor],
    all_menu_langs: Sequence[str],
    device: torch.device | str,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Aggregate menu group probabilities into per-language probability masses."""
    if len(prompt_target_records) != len(menu_group_probs):
        raise ValueError("prompt_target_records and menu_group_probs must have the same length.")
    if not prompt_target_records:
        raise ValueError("No target-string records available.")
    n_prompts = len(prompt_target_records)
    n_layers = int(menu_group_probs[0].shape[0])
    lang_masses = {
        lang: torch.zeros((n_prompts, n_layers), device=device)
        for lang in all_menu_langs
    }
    unique_lang_masses = {
        lang: torch.zeros((n_prompts, n_layers), device=device)
        for lang in all_menu_langs
    }
    ambiguous_mass = torch.zeros((n_prompts, n_layers), device=device)
    unique_mass = torch.zeros((n_prompts, n_layers), device=device)
    for prompt_idx, (record, group_probs) in enumerate(
        zip(prompt_target_records, menu_group_probs)
    ):
        if int(group_probs.shape[0]) != n_layers:
            raise ValueError("All menu_group_probs entries must have the same layer count.")
        for group_idx, group in enumerate(record["menu_strings_grouped"]):
            langs = list(group["langs"])
            prob = group_probs[:, group_idx]
            if len(langs) == 1:
                unique_mass[prompt_idx] += prob
                unique_lang_masses[langs[0]][prompt_idx] += prob
            else:
                ambiguous_mass[prompt_idx] += prob
            share = prob / float(len(langs))
            for lang in langs:
                lang_masses[lang][prompt_idx] += share
    return lang_masses, unique_lang_masses, ambiguous_mass, unique_mass


################################################################################
# Start(w) target-string scoring (paper default)
################################################################################

def normalize_lang_log_masses(
    lang_log_masses: Mapping[str, torch.Tensor],
    *,
    empty_policy: str = "uniform",
    warn_label: str | None = None,
) -> dict[str, torch.Tensor]:
    """Normalize per-language log masses without probability-space underflow.

    `start_tokens_only` can assign extremely small finite probability mass to every
    candidate language in early layers. Normalizing after `exp(log_mass)` can then
    produce rows that sum to almost zero. This helper keeps normalization in log
    space and only exponentiates after subtracting the row log normalizer.

    If a row has no finite language log mass at all, `empty_policy="uniform"` marks
    it as maximally uninformative instead of producing zero entropy/dominance.
    """
    if not lang_log_masses:
        return {}
    if empty_policy not in {"uniform", "nan"}:
        raise ValueError(f"Unsupported empty_policy={empty_policy!r}")
    langs = sorted(lang_log_masses)
    stacked = torch.stack([lang_log_masses[lang] for lang in langs], dim=-1)
    log_denom = torch.logsumexp(stacked, dim=-1, keepdim=True)
    finite = torch.isfinite(log_denom)
    dist = torch.exp(stacked - log_denom)
    if not finite.all():
        n_empty = int((~finite).sum().item())
        total = int(finite.numel())
        label = f" for {warn_label}" if warn_label else ""
        print(
            f"[warn] {n_empty:,}/{total:,} language log-mass rows{label} are all -inf; "
            f"using empty_policy={empty_policy!r}.",
            flush=True,
        )
        if empty_policy == "uniform":
            fallback = torch.full_like(dist, 1.0 / float(len(langs)))
        else:
            fallback = torch.full_like(dist, float("nan"))
        dist = torch.where(finite, dist, fallback)
    return {lang: dist[..., idx] for idx, lang in enumerate(langs)}


def build_menu_lang_log_masses(
    *,
    prompt_target_records: Sequence[dict[str, Any]],
    menu_group_logprobs: Sequence[torch.Tensor],
    all_menu_langs: Sequence[str],
    device: torch.device | str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Aggregate menu group log-probabilities into per-language log masses."""
    if len(prompt_target_records) != len(menu_group_logprobs):
        raise ValueError("prompt_target_records and menu_group_logprobs must have the same length.")
    if not prompt_target_records:
        raise ValueError("No target-string records available.")
    n_prompts = len(prompt_target_records)
    n_layers = int(menu_group_logprobs[0].shape[0])
    lang_log_masses = {
        lang: torch.full(
            (n_prompts, n_layers),
            -float("inf"),
            device=device,
            dtype=torch.float64,
        )
        for lang in all_menu_langs
    }
    unique_lang_log_masses = {
        lang: torch.full(
            (n_prompts, n_layers),
            -float("inf"),
            device=device,
            dtype=torch.float64,
        )
        for lang in all_menu_langs
    }
    for prompt_idx, (record, group_logprobs) in enumerate(
        zip(prompt_target_records, menu_group_logprobs)
    ):
        group_logprobs = group_logprobs.to(device=device, dtype=torch.float64)
        if int(group_logprobs.shape[0]) != n_layers:
            raise ValueError("All menu_group_logprobs entries must have the same layer count.")
        for group_idx, group in enumerate(record["menu_strings_grouped"]):
            langs = list(group["langs"])
            if not langs:
                continue
            logprob = group_logprobs[:, group_idx]
            if len(langs) == 1:
                lang = langs[0]
                unique_lang_log_masses[lang][prompt_idx] = torch.logaddexp(
                    unique_lang_log_masses[lang][prompt_idx],
                    logprob,
                )
            share_logprob = logprob - math.log(float(len(langs)))
            for lang in langs:
                lang_log_masses[lang][prompt_idx] = torch.logaddexp(
                    lang_log_masses[lang][prompt_idx],
                    share_logprob,
                )
    return lang_log_masses, unique_lang_log_masses


def prepare_start_token_groups(
    prompt_target_records: Sequence[dict[str, Any]],
    *,
    require_wendler_keep: bool,
) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Validate Start(w) artifacts and prepare token-id groups plus group sizes."""
    start_token_groups_by_prompt: list[list[list[int]]] = []
    start_token_counts_by_prompt: list[list[int]] = []
    prompt_record_iter = prompt_target_records
    if PROGRESS_IS_TTY:
        prompt_record_iter = tqdm(
            prompt_record_iter,
            desc=f"Validating start-token menus ({len(prompt_target_records)} prompts)",
            unit="prompt",
            leave=False,
        )
    else:
        print(
            f"[target_string] Validating start-token menus for "
            f"{len(prompt_target_records)} prompts",
            flush=True,
        )
    for record in prompt_record_iter:
        grouped = record["menu_strings_grouped"]
        start_grouped = record.get("menu_start_tokens_grouped")
        if not grouped:
            raise ValueError(f"No grouped menu strings for prompt_id={record['prompt_id']}")
        if not isinstance(start_grouped, list):
            raise ValueError(f"Missing menu_start_tokens_grouped for prompt_id={record['prompt_id']}")
        if len(start_grouped) != len(grouped):
            raise ValueError(f"Start-token group count mismatch for prompt_id={record['prompt_id']}")
        if require_wendler_keep and not bool(record.get("wendler_keep", False)):
            raise ValueError(
                f"Prompt {record['prompt_id']} is marked wendler_keep=false "
                "(empty target/latent Start(w) set or target/English start-token overlap)."
            )
        prompt_groups: list[list[int]] = []
        prompt_counts: list[int] = []
        for group_idx, group in enumerate(grouped):
            start_group = start_grouped[group_idx]
            if (
                str(start_group.get("text", "")) != str(group["text"])
                or list(start_group.get("langs", [])) != list(group.get("langs", []))
            ):
                raise ValueError(
                    f"Start-token artifact group mismatch for prompt_id={record['prompt_id']}, group={group_idx}"
                )
            start_token_ids = [
                int(token_id)
                for token_id in start_group.get("start_token_ids", [])
            ]
            prompt_groups.append(start_token_ids)
            prompt_counts.append(len(start_token_ids))
        start_token_groups_by_prompt.append(prompt_groups)
        start_token_counts_by_prompt.append(prompt_counts)
    return start_token_groups_by_prompt, start_token_counts_by_prompt


def _score_start_token_groups(
    *,
    prompt_idx: int,
    prompt_latents_cpu: torch.Tensor,
    group_token_ids: Sequence[Sequence[int]],
    layer_indices: Sequence[int],
    decoding_lens_obj: Any,
    unembed_info: UnembedInfo,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score one prompt's Start(w) token groups for every selected layer."""
    n_layers = len(layer_indices)
    prompt_latents = prompt_latents_cpu.to(unembed_info.device)
    group_logprobs = torch.full(
        (n_layers, len(group_token_ids)),
        -float("inf"),
        device=unembed_info.device,
    )
    vocab_entropy_bits = torch.empty(n_layers, device=unembed_info.device)
    token_tensors = [
        torch.tensor(token_ids, dtype=torch.long, device=unembed_info.device)
        if token_ids
        else None
        for token_ids in group_token_ids
    ]
    for layer_offset, layer_idx in enumerate(layer_indices):
        hidden = prompt_latents[layer_offset : layer_offset + 1]
        logits = decoding_lens_obj.project_layer(hidden, int(layer_idx))[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs_for_entropy = log_probs.double()
        probs_for_entropy = torch.exp(log_probs_for_entropy)
        vocab_entropy_bits[layer_offset] = (
            -(probs_for_entropy * log_probs_for_entropy).sum() / math.log(2.0)
        )
        vocab_prob_sum = probs_for_entropy.sum()
        if (
            not torch.isfinite(vocab_entropy_bits[layer_offset])
            or not torch.isfinite(vocab_prob_sum)
            or torch.abs(vocab_prob_sum - 1.0) > 1e-4
        ):
            print(
                "[warn] target_string start_tokens_only vocab entropy diagnostic: "
                f"prompt_idx={prompt_idx}, layer={int(layer_idx)}, "
                f"prob_sum={float(vocab_prob_sum.detach().cpu())}, "
                f"entropy_bits={float(vocab_entropy_bits[layer_offset].detach().cpu())}",
                flush=True,
            )
        for group_idx, token_ids in enumerate(token_tensors):
            if token_ids is not None:
                group_logprobs[layer_offset, group_idx] = torch.logsumexp(
                    log_probs[token_ids],
                    dim=0,
                )
    return group_logprobs, finite_exp(group_logprobs), vocab_entropy_bits


def run_start_token_scoring(
    *,
    model: Any,
    prompts: Sequence[str],
    prompt_ids: Sequence[str],
    target_string_records: Mapping[str, dict[str, Any]] | None,
    layer_indices: Sequence[int],
    trace_batch_size: int,
    decoding_lens_obj: Any,
    require_wendler_keep: bool,
    rev: str,
    unembed_device: str | torch.device | None = None,
    prompt_token_lengths: Sequence[int] | None = None,
    prompt_anchor_positions: Sequence[int] | None = None,
) -> TargetStringResult:
    """Run paper-default Start(w) scoring from streamed prompt latents."""
    if not prompt_ids:
        raise ValueError("prompt_ids are required for target_string mapping.")
    if len(prompts) != len(prompt_ids):
        raise ValueError("prompts and prompt_ids must have the same length.")
    if not layer_indices:
        raise ValueError("layer_indices are required for target-string scoring.")
    if trace_batch_size <= 0:
        raise ValueError("trace_batch_size must be >= 1 for target-string scoring.")
    if prompt_token_lengths is not None and len(prompt_token_lengths) != len(prompts):
        raise ValueError("prompt_token_lengths must match prompts.")
    if prompt_anchor_positions is not None and len(prompt_anchor_positions) != len(prompts):
        raise ValueError("prompt_anchor_positions must match prompts.")

    prompt_target_records = resolve_prompt_target_string_records(
        prompt_ids=prompt_ids,
        target_string_records=target_string_records,
    )
    all_menu_langs = get_menu_langs(prompt_target_records)
    if not all_menu_langs:
        raise ValueError("No target-string menu languages found.")

    # Prepare Start(w) groups before tracing so malformed artifacts fail early.
    start_token_groups_by_prompt, start_token_counts_by_prompt = prepare_start_token_groups(
        prompt_target_records,
        require_wendler_keep=require_wendler_keep,
    )

    print("[target_string] Preparing unembedding weights", flush=True)
    unembed_info = getattr(decoding_lens_obj, "unembed_info", None)
    if unembed_info is None:
        unembed_info = prepare_unembed_info(model, device=unembed_device)
    if getattr(unembed_info, "normalized_weight", None) is not None:
        # Start(w) scoring uses log probabilities from the raw lm_head only.
        # Drop token-energy stats to reduce peak memory while preserving the
        # same log_softmax computation as the original implementation.
        unembed_info.normalized_weight = torch.empty(
            0,
            dtype=unembed_info.weight.dtype,
            device=unembed_info.device,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    n_layers = len(layer_indices)
    menu_start_token_logprobs_cpu: list[torch.Tensor] = []
    menu_start_token_probs_cpu: list[torch.Tensor] = []
    menu_group_probs: list[torch.Tensor] = []
    tgt_first_token_vocab_entropy_bits_by_prompt: list[torch.Tensor] = []

    if prompt_token_lengths is None:
        prompt_token_lengths = [
            len(model.tokenizer(prompt, add_special_tokens=True)["input_ids"])
            for prompt in prompts
        ]
    print(
        f"[target_string] Streaming prompt-only Start(w) scoring "
        f"({len(prompts)} prompts, {n_layers} layers)",
        flush=True,
    )
    total_batches = (len(prompts) + trace_batch_size - 1) // trace_batch_size
    batch_starts = range(0, len(prompts), trace_batch_size)
    batch_iter = batch_starts
    if PROGRESS_IS_TTY:
        batch_iter = tqdm(
            batch_starts,
            total=total_batches,
            desc=f"Tracing/scoring Start(w) batches ({len(prompts)} prompts, {n_layers} layers)",
            unit="batch",
        )
    for batch_start in batch_iter:
        batch_end = min(batch_start + trace_batch_size, len(prompts))
        batch_latents, _ = collect_position_latents(
            model=model,
            prompts=prompts[batch_start:batch_end],
            layer_indices=layer_indices,
            position=-1,
            positions_by_prompt=(
                None
                if prompt_anchor_positions is None
                else prompt_anchor_positions[batch_start:batch_end]
            ),
            sequence_lengths=prompt_token_lengths[batch_start:batch_end],
            trace_batch_size=trace_batch_size,
            show_progress=False,
        )
        batch_latents = batch_latents.detach().cpu()
        for local_idx in range(batch_end - batch_start):
            prompt_idx = batch_start + local_idx
            group_logprobs, group_probs, vocab_entropy_bits = _score_start_token_groups(
                prompt_idx=prompt_idx,
                prompt_latents_cpu=batch_latents[local_idx],
                group_token_ids=start_token_groups_by_prompt[prompt_idx],
                layer_indices=layer_indices,
                decoding_lens_obj=decoding_lens_obj,
                unembed_info=unembed_info,
            )
            menu_group_probs.append(group_probs.detach().cpu())
            menu_start_token_logprobs_cpu.append(
                group_logprobs.detach().cpu().to(torch.float32)
            )
            menu_start_token_probs_cpu.append(
                group_probs.detach().cpu().to(torch.float64)
            )
            tgt_first_token_vocab_entropy_bits_by_prompt.append(
                vocab_entropy_bits.detach().cpu().to(torch.float32)
            )
            del group_logprobs, group_probs, vocab_entropy_bits
        del batch_latents
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert menu scores into language masses, normalized distributions, and artifacts.
    lang_mass_start_tokens, _, menu_ambiguous_mass, menu_unique_mass = (
        build_menu_lang_masses(
            prompt_target_records=prompt_target_records,
            menu_group_probs=menu_group_probs,
            all_menu_langs=all_menu_langs,
            device=torch.device("cpu"),
        )
    )
    lang_log_mass_start_tokens, unique_log_mass_start_tokens_by_lang = build_menu_lang_log_masses(
        prompt_target_records=prompt_target_records,
        menu_group_logprobs=menu_start_token_logprobs_cpu,
        all_menu_langs=all_menu_langs,
        device=torch.device("cpu"),
    )
    lang_dist_start_tokens = normalize_lang_log_masses(
        lang_log_mass_start_tokens,
        empty_policy="uniform",
        warn_label="target_string start_tokens_only",
    )
    unique_dist_start_tokens_by_lang = normalize_lang_log_masses(
        unique_log_mass_start_tokens_by_lang,
        empty_policy="uniform",
        warn_label="target_string start_tokens_only unique-only",
    )
    tgt_first_token_vocab_entropy_bits = torch.stack(
        tgt_first_token_vocab_entropy_bits_by_prompt,
        dim=0,
    )
    decoding_meta = {
        "target_score_semantics": "prompt_only_next_token_start_set_logprob",
        "target_mass_semantics": "wendler_start_token_probability_mass",
        "target_dist_semantics": "language_normalized_wendler_start_token_mass",
        "target_string_scoring_mode": "start_tokens_only",
        "target_string_execution_path": "prompt_only_start_tokens_only",
        "mean_tgt_first_token_vocab_entropy_bits_by_layer": tgt_first_token_vocab_entropy_bits.mean(dim=0).detach().cpu().tolist(),
        "mean_tgt_vocab_entropy_bits_by_layer": tgt_first_token_vocab_entropy_bits.mean(dim=0).detach().cpu().tolist(),
    }
    add_menu_mass_summary(
        decoding_meta,
        menu_ambiguous_mass=menu_ambiguous_mass,
        menu_unique_mass=menu_unique_mass,
    )
    target_string_artifact = build_target_string_artifact(
        checkpoint_id=rev,
        layer_indices=layer_indices,
    )
    for prompt_idx, record in enumerate(prompt_target_records):
        target_string_artifact["records"].append(
            {
                **build_target_string_record_base(record),
                "menu_start_token_logprobs": menu_start_token_logprobs_cpu[prompt_idx],
                "menu_start_token_probs": menu_start_token_probs_cpu[prompt_idx],
                "menu_start_token_counts": start_token_counts_by_prompt[prompt_idx],
                "tgt_first_token_vocab_entropy_bits": tgt_first_token_vocab_entropy_bits[prompt_idx].detach().cpu().to(torch.float32),
                "tgt_vocab_entropy_bits": tgt_first_token_vocab_entropy_bits[prompt_idx].detach().cpu().to(torch.float32),
                "menu_start_tokens_grouped": record.get("menu_start_tokens_grouped"),
                "wendler_keep": record.get("wendler_keep"),
                "lang_mass_start_tokens": {
                    lang: values[prompt_idx].detach().cpu().to(torch.float32)
                    for lang, values in lang_mass_start_tokens.items()
                },
                "lang_log_mass_start_tokens": {
                    lang: values[prompt_idx].detach().cpu().to(torch.float32)
                    for lang, values in lang_log_mass_start_tokens.items()
                },
                "lang_dist_start_tokens": {
                    lang: values[prompt_idx].detach().cpu().to(torch.float32)
                    for lang, values in lang_dist_start_tokens.items()
                },
            }
        )
    return TargetStringResult(
        lang_probs_dec=lang_dist_start_tokens,
        lang_mass_dec=lang_mass_start_tokens,
        lang_dist_dec=lang_dist_start_tokens,
        lang_mass_key="lang_mass_start_tokens",
        lang_dist_key="lang_dist_start_tokens",
        menu_lang_probs_unique_only=unique_dist_start_tokens_by_lang,
        menu_ambiguous_mass=menu_ambiguous_mass,
        menu_unique_mass=menu_unique_mass,
        prompt_target_records=prompt_target_records,
        target_string_artifact=target_string_artifact,
        decoding_meta=decoding_meta,
        tgt_first_token_vocab_entropy_bits=tgt_first_token_vocab_entropy_bits,
        tgt_vocab_entropy_bits=tgt_first_token_vocab_entropy_bits,
    )

################################################################################
# Multi-token teacher-forced target-string scoring (experimental! not in the paper)
################################################################################

@torch.no_grad()
def score_teacher_forced_one_candidate(
    *,
    latents: torch.Tensor,
    prompt_len: int,
    target_tokens: torch.Tensor,
    unembed_info: UnembedInfo,
    decoding_lens: Any,
    layer_indices: Sequence[int],
    return_vocab_entropy_bits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return per-layer log-probabilities and optional vocabulary entropy."""
    if latents.ndim != 3:
        raise ValueError("latents must be (layers, seq, hidden).")
    if decoding_lens is None:
        raise ValueError("decoding_lens is required for score_teacher_forced_one_candidate.")
    if layer_indices is None:
        raise ValueError("layer_indices are required for score_teacher_forced_one_candidate.")
    if len(layer_indices) != int(latents.shape[0]):
        raise ValueError("layer_indices must match the latent layer dimension.")
    if target_tokens.numel() == 0:
        raise ValueError("target_tokens must be non-empty.")
    device = unembed_info.device
    target_tokens = target_tokens.to(device)
    per_layer = []
    entropy_per_layer = []
    for layer_idx in range(latents.shape[0]):
        hidden = latents[layer_idx].to(device)
        logits = decoding_lens.project_layer(hidden, int(layer_indices[layer_idx]))
        log_probs = torch.log_softmax(logits, dim=-1)
        step_logp = []
        step_entropy = []
        for t_idx in range(target_tokens.numel()):
            pos = prompt_len - 1 + t_idx
            if pos < 0 or pos >= log_probs.shape[0]:
                raise IndexError("Teacher-forced position out of range.")
            token_id = int(target_tokens[t_idx].item())
            step_logp.append(log_probs[pos, token_id])
            if return_vocab_entropy_bits:
                probs = log_probs[pos].exp()
                step_entropy.append(
                    (-probs * log_probs[pos]).sum() / torch.log(torch.tensor(2.0, device=device))
                )
        per_layer.append(torch.stack(step_logp))
        if return_vocab_entropy_bits:
            entropy_per_layer.append(torch.stack(step_entropy))
    step_logprobs = torch.stack(per_layer, dim=0)
    vocab_entropy_bits = (
        torch.stack(entropy_per_layer, dim=0)
        if return_vocab_entropy_bits
        else None
    )
    return step_logprobs, vocab_entropy_bits


def decode_token_strings(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    """Convert token ids into display strings using the tokenizer if possible."""
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
        return [str(token) for token in tokens]
    return [str(tokenizer.decode([int(token_id)])) for token_id in token_ids]


def normalize_lang_masses(
    lang_masses: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize probability-space language masses across languages."""
    if not lang_masses:
        return {}
    langs = sorted(lang_masses)
    stacked = torch.stack([lang_masses[lang] for lang in langs], dim=-1)
    denom = stacked.sum(dim=-1, keepdim=True).clamp_min(PROB_EPS)
    dist = stacked / denom
    return {lang: dist[..., idx] for idx, lang in enumerate(langs)}


def build_teacher_forced_candidates(
    *,
    model: Any,
    prompts: Sequence[str],
    prompt_target_records: Sequence[dict[str, Any]],
    layer_indices: Sequence[int],
) -> TeacherForcedCandidates:
    """Build prompt+target candidate strings and per-candidate metadata."""
    if len(prompts) != len(prompt_target_records):
        raise ValueError("prompts and prompt_target_records must have the same length.")
    if not layer_indices:
        raise ValueError("layer_indices are required for teacher-forced scoring.")
    sequences: list[str] = []
    metadata: list[dict[str, Any]] = []
    prompt_group_sum_logprobs: list[torch.Tensor] = []
    prompt_group_first_token_logprobs: list[torch.Tensor] = []
    prompt_group_token_counts: list[list[int]] = []
    prompt_record_iter = zip(prompts, prompt_target_records)
    if PROGRESS_IS_TTY:
        prompt_record_iter = tqdm(
            prompt_record_iter,
            desc=f"Building target-string candidates ({len(prompts)} prompts)",
            unit="prompt",
            total=len(prompts),
            leave=False,
        )
    else:
        print(f"[target_string] Building candidates for {len(prompts)} prompts", flush=True)
    for idx, (prompt, record) in enumerate(prompt_record_iter):
        prompt_tokens = model.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)
        grouped = record["menu_strings_grouped"]
        if not grouped:
            raise ValueError(f"No grouped menu strings for prompt_id={record['prompt_id']}")
        prompt_group_sum_logprobs.append(torch.full((len(layer_indices), len(grouped)), -float("inf")))
        prompt_group_first_token_logprobs.append(torch.full((len(layer_indices), len(grouped)), -float("inf")))
        prompt_group_token_counts.append([0 for _ in grouped])
        target_group_idx = -1
        for group_idx, group in enumerate(grouped):
            target = str(group["text"])
            target_tokens = model.tokenizer(target, add_special_tokens=False)["input_ids"]
            if len(target_tokens) == 0:
                raise ValueError(
                    "Grouped target text tokenized to zero tokens for "
                    f"prompt_id={record['prompt_id']}, text={target!r}"
                )
            is_tgt = target == record["tgt_text"] and record["tgt_lang"] in group["langs"]
            if is_tgt:
                target_group_idx = group_idx
            sequences.append(prompt + target)
            metadata.append(
                {
                    "prompt_idx": idx,
                    "group_idx": group_idx,
                    "prompt_len": prompt_len,
                    "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
                    "is_tgt": is_tgt,
                }
            )
        if target_group_idx < 0:
            raise ValueError(
                "Could not locate tgt_text inside grouped menu for "
                f"prompt_id={record['prompt_id']}"
            )

    if not sequences:
        raise ValueError("No target sequences constructed for target_string mapping.")
    if PROGRESS_IS_TTY:
        sequence_token_iter = tqdm(
            sequences,
            desc=f"Tokenizing target-string candidates ({len(sequences)} candidates)",
            unit="candidate",
            leave=False,
        )
    else:
        print(f"[target_string] Tokenizing {len(sequences)} candidates", flush=True)
        sequence_token_iter = sequences
    sequence_token_lengths = [
        len(model.tokenizer(sequence, add_special_tokens=True)["input_ids"])
        for sequence in sequence_token_iter
    ]
    return TeacherForcedCandidates(
        sequences=sequences,
        metadata=metadata,
        sequence_token_lengths=sequence_token_lengths,
        group_sum_logprobs_by_prompt=prompt_group_sum_logprobs,
        group_first_token_logprobs_by_prompt=prompt_group_first_token_logprobs,
        group_token_counts_by_prompt=prompt_group_token_counts,
    )


def score_teacher_forced_all_candidates(
    *,
    model: Any,
    candidates: TeacherForcedCandidates,
    n_prompts: int,
    layer_indices: Sequence[int],
    trace_batch_size: int,
    unembed_info: UnembedInfo,
    decoding_lens_obj: Any,
) -> TeacherForcedScores:
    """Fill teacher-forced menu scores and return target-level diagnostics."""
    if trace_batch_size <= 0:
        raise ValueError("trace_batch_size must be >= 1 for teacher-forced scoring.")
    tgt_sum_logprob = torch.full(
        (n_prompts, len(layer_indices)),
        -float("inf"),
        device=unembed_info.device,
    )
    tgt_token_count = torch.zeros(n_prompts, device=unembed_info.device)
    tgt_tokens_by_prompt: list[list[int] | None] = [None] * n_prompts
    tgt_token_strs_by_prompt: list[list[str] | None] = [None] * n_prompts
    tgt_step_logprobs_by_prompt: list[torch.Tensor | None] = [None] * n_prompts
    tgt_step_probs_by_prompt: list[torch.Tensor | None] = [None] * n_prompts
    tgt_step_vocab_entropy_bits_by_prompt: list[torch.Tensor | None] = [None] * n_prompts

    target_batch_starts = range(0, len(candidates.sequences), trace_batch_size)
    target_batch_count = (len(candidates.sequences) + trace_batch_size - 1) // trace_batch_size
    target_batch_iter = enumerate(target_batch_starts, start=1)
    if PROGRESS_IS_TTY:
        target_batch_iter = enumerate(
            tqdm(
                target_batch_starts,
                desc=(
                    f"Scoring target strings ({len(candidates.sequences)} candidates, "
                    f"{len(layer_indices)} layers)"
                ),
                unit="batch",
                total=target_batch_count,
            ),
            start=1,
        )
    else:
        print(
            f"[target_string] Scoring {len(candidates.sequences)} candidates"
            f" over {len(layer_indices)} layers in {target_batch_count} batches",
            flush=True,
        )
    log_every = max(1, target_batch_count // 20)
    for batch_idx, batch_start in target_batch_iter:
        if not PROGRESS_IS_TTY and (
            batch_idx == 1 or batch_idx == target_batch_count or batch_idx % log_every == 0
        ):
            print(
                f"[target_string] Scoring batch {batch_idx}/{target_batch_count}",
                flush=True,
            )
        batch_end = min(batch_start + trace_batch_size, len(candidates.sequences))
        target_latents_batch, _ = collect_sequence_latents(
            model=model,
            prompts=candidates.sequences[batch_start:batch_end],
            layer_indices=layer_indices,
            sequence_lengths=candidates.sequence_token_lengths[batch_start:batch_end],
            trace_batch_size=trace_batch_size,
            show_progress=False,
        )
        for latents_seq, meta in zip(
            target_latents_batch,
            candidates.metadata[batch_start:batch_end],
        ):
            step_logprobs, step_vocab_entropy_bits = score_teacher_forced_one_candidate(
                latents=latents_seq,
                prompt_len=meta["prompt_len"],
                target_tokens=meta["target_tokens"],
                unembed_info=unembed_info,
                decoding_lens=decoding_lens_obj,
                layer_indices=layer_indices,
                return_vocab_entropy_bits=bool(meta["is_tgt"]),
            )
            per_layer = step_logprobs.sum(dim=-1)
            idx = meta["prompt_idx"]
            group_idx = meta["group_idx"]
            candidates.group_sum_logprobs_by_prompt[idx][:, group_idx] = (
                per_layer.detach().cpu()
            )
            candidates.group_first_token_logprobs_by_prompt[idx][:, group_idx] = (
                step_logprobs[:, 0].detach().cpu()
            )
            target_len = int(meta["target_tokens"].numel())
            candidates.group_token_counts_by_prompt[idx][group_idx] = target_len
            if meta["is_tgt"]:
                tgt_sum_logprob[idx] = per_layer
                tgt_token_count[idx] = float(target_len)
                tgt_tokens = meta["target_tokens"].detach().cpu().tolist()
                tgt_tokens_by_prompt[idx] = tgt_tokens
                tgt_token_strs_by_prompt[idx] = decode_token_strings(model.tokenizer, tgt_tokens)
                tgt_step_logprobs_by_prompt[idx] = step_logprobs.detach().cpu().to(torch.float32)
                tgt_step_probs_by_prompt[idx] = torch.exp(step_logprobs).detach().cpu().to(torch.float32)
                tgt_step_vocab_entropy_bits_by_prompt[idx] = (
                    step_vocab_entropy_bits.detach().cpu().to(torch.float32)
                    if step_vocab_entropy_bits is not None
                    else None
                )
        del target_latents_batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (tgt_token_count == 0).any():
        raise ValueError("Missing tgt_token_count for at least one target-string prompt.")
    return TeacherForcedScores(
        tgt_sum_logprob=tgt_sum_logprob,
        tgt_token_count=tgt_token_count,
        tgt_tokens_by_prompt=tgt_tokens_by_prompt,
        tgt_token_strs_by_prompt=tgt_token_strs_by_prompt,
        tgt_step_logprobs_by_prompt=tgt_step_logprobs_by_prompt,
        tgt_step_probs_by_prompt=tgt_step_probs_by_prompt,
        tgt_step_vocab_entropy_bits_by_prompt=tgt_step_vocab_entropy_bits_by_prompt,
    )


def run_teacher_forced_scoring(
    *,
    model: Any,
    prompts: Sequence[str],
    prompt_ids: Sequence[str],
    target_string_records: Mapping[str, dict[str, Any]] | None,
    layer_indices: Sequence[int],
    trace_batch_size: int,
    decoding_lens_obj: Any,
    rev: str,
    unembed_device: str | torch.device | None = None,
) -> TargetStringResult:
    """Run experimental teacher-forced scoring for full target strings."""
    if not prompt_ids:
        raise ValueError("prompt_ids are required for target_string mapping.")
    if len(prompts) != len(prompt_ids):
        raise ValueError("prompts and prompt_ids must have the same length.")
    if trace_batch_size <= 0:
        raise ValueError("trace_batch_size must be >= 1 for target-string scoring.")
    prompt_target_records = resolve_prompt_target_string_records(
        prompt_ids=prompt_ids,
        target_string_records=target_string_records,
    )
    all_menu_langs = get_menu_langs(prompt_target_records)
    if not all_menu_langs:
        raise ValueError("No target-string menu languages found.")

    # Build prompt+target candidates before tracing; their order defines menu columns.
    candidates = build_teacher_forced_candidates(
        model=model,
        prompts=prompts,
        prompt_target_records=prompt_target_records,
        layer_indices=layer_indices,
    )

    print("[target_string] Preparing unembedding weights", flush=True)
    unembed_info = prepare_unembed_info(model, device=unembed_device)
    # Score every candidate sequence with teacher forcing.
    teacher_forced_scores = score_teacher_forced_all_candidates(
        model=model,
        candidates=candidates,
        n_prompts=len(prompts),
        layer_indices=layer_indices,
        trace_batch_size=trace_batch_size,
        unembed_info=unembed_info,
        decoding_lens_obj=decoding_lens_obj,
    )
    tgt_sum_logprob = teacher_forced_scores.tgt_sum_logprob
    tgt_token_count = teacher_forced_scores.tgt_token_count
    tgt_tokens_by_prompt = teacher_forced_scores.tgt_tokens_by_prompt
    tgt_token_strs_by_prompt = teacher_forced_scores.tgt_token_strs_by_prompt
    tgt_step_logprobs_by_prompt = teacher_forced_scores.tgt_step_logprobs_by_prompt
    tgt_step_probs_by_prompt = teacher_forced_scores.tgt_step_probs_by_prompt
    tgt_step_vocab_entropy_bits_by_prompt = teacher_forced_scores.tgt_step_vocab_entropy_bits_by_prompt

    menu_sum_logprobs_cpu = []
    menu_first_token_logprobs_cpu = []
    menu_group_probs: list[torch.Tensor] = []
    menu_probs_cpu = []
    menu_first_token_probs_cpu = []
    # Convert teacher-forced menu log-probs into language masses and artifacts.
    for prompt_idx, record in enumerate(prompt_target_records):
        group_logprobs = candidates.group_sum_logprobs_by_prompt[prompt_idx].to(unembed_info.device)
        group_first_token_logprobs = (
            candidates.group_first_token_logprobs_by_prompt[prompt_idx].to(
                unembed_info.device
            )
        )
        group_probs = finite_exp(group_logprobs)
        group_first_token_probs = finite_exp(group_first_token_logprobs)
        menu_sum_logprobs_cpu.append(group_logprobs.detach().cpu().to(torch.float32))
        menu_first_token_logprobs_cpu.append(group_first_token_logprobs.detach().cpu().to(torch.float32))
        menu_group_probs.append(group_probs)
        menu_probs_cpu.append(group_probs.detach().cpu().to(torch.float32))
        menu_first_token_probs_cpu.append(group_first_token_probs.detach().cpu().to(torch.float64))
    (
        lang_mass_teacher_forced,
        unique_mass_teacher_forced_by_lang,
        menu_ambiguous_mass,
        menu_unique_mass,
    ) = build_menu_lang_masses(
        prompt_target_records=prompt_target_records,
        menu_group_probs=menu_group_probs,
        all_menu_langs=all_menu_langs,
        device=unembed_info.device,
    )
    lang_dist_teacher_forced = normalize_lang_masses(lang_mass_teacher_forced)
    unique_dist_teacher_forced_by_lang = normalize_lang_masses(unique_mass_teacher_forced_by_lang)

    tgt_prob = torch.exp(tgt_sum_logprob)
    tgt_mean_logprob = tgt_sum_logprob / tgt_token_count.unsqueeze(-1).clamp_min(1.0)
    tgt_mean_prob = torch.exp(tgt_mean_logprob)
    tgt_vocab_entropy_bits = torch.stack(
        [
            value.mean(dim=-1).to(unembed_info.device)
            for value in tgt_step_vocab_entropy_bits_by_prompt
            if value is not None
        ],
        dim=0,
    )
    tgt_first_token_vocab_entropy_bits = torch.stack(
        [
            value[:, 0].to(unembed_info.device)
            for value in tgt_step_vocab_entropy_bits_by_prompt
            if value is not None
        ],
        dim=0,
    )
    if int(tgt_vocab_entropy_bits.shape[0]) != len(prompts):
        raise ValueError("Missing target vocab entropy for at least one target-string prompt.")
    if int(tgt_first_token_vocab_entropy_bits.shape[0]) != len(prompts):
        raise ValueError("Missing first-token vocab entropy for at least one target-string prompt.")
    tgt_token_count_cpu = [int(value) for value in tgt_token_count.detach().cpu().tolist()]
    decoding_meta = {
        "target_score_semantics": "teacher_forced_sequence_logprob",
        "target_mass_semantics": "teacher_forced_sequence_probability_mass",
        "target_dist_semantics": "language_normalized_teacher_forced_sequence_mass",
        "target_string_scoring_mode": "multi_token_teacher_forced",
        "target_string_execution_path": "multi_token_teacher_forced",
        "avg_tgt_token_count": float(sum(tgt_token_count_cpu) / max(len(tgt_token_count_cpu), 1)),
        "max_tgt_token_count": max(tgt_token_count_cpu) if tgt_token_count_cpu else 0,
        "single_token_targets": sum(1 for value in tgt_token_count_cpu if value == 1),
        "multi_token_targets": sum(1 for value in tgt_token_count_cpu if value > 1),
        "mean_tgt_sum_logprob_by_layer": tgt_sum_logprob.mean(dim=0).detach().cpu().tolist(),
        "mean_tgt_prob_by_layer": tgt_prob.mean(dim=0).detach().cpu().tolist(),
        "mean_tgt_mean_logprob_by_layer": tgt_mean_logprob.mean(dim=0).detach().cpu().tolist(),
        "mean_tgt_mean_prob_by_layer": tgt_mean_prob.mean(dim=0).detach().cpu().tolist(),
        "mean_tgt_first_token_vocab_entropy_bits_by_layer": tgt_first_token_vocab_entropy_bits.mean(dim=0).detach().cpu().tolist(),
        "mean_tgt_vocab_entropy_bits_by_layer": tgt_vocab_entropy_bits.mean(dim=0).detach().cpu().tolist(),
    }
    add_menu_mass_summary(
        decoding_meta,
        menu_ambiguous_mass=menu_ambiguous_mass,
        menu_unique_mass=menu_unique_mass,
    )
    target_string_artifact = build_target_string_artifact(
        checkpoint_id=rev,
        layer_indices=layer_indices,
    )
    for prompt_idx, record in enumerate(prompt_target_records):
        target_string_artifact["records"].append(
            {
                **build_target_string_record_base(record),
                "tgt_sum_logprob": tgt_sum_logprob[prompt_idx].detach().cpu().to(torch.float32),
                "tgt_prob": tgt_prob[prompt_idx].detach().cpu().to(torch.float32),
                "tgt_mean_logprob": tgt_mean_logprob[prompt_idx].detach().cpu().to(torch.float32),
                "tgt_mean_prob": tgt_mean_prob[prompt_idx].detach().cpu().to(torch.float32),
                "tgt_token_count": tgt_token_count_cpu[prompt_idx],
                "tgt_tokens": tgt_tokens_by_prompt[prompt_idx],
                "tgt_token_strs": tgt_token_strs_by_prompt[prompt_idx],
                "tgt_step_logprobs": tgt_step_logprobs_by_prompt[prompt_idx],
                "tgt_step_probs": tgt_step_probs_by_prompt[prompt_idx],
                "tgt_step_vocab_entropy_bits": tgt_step_vocab_entropy_bits_by_prompt[prompt_idx],
                "tgt_first_token_vocab_entropy_bits": tgt_first_token_vocab_entropy_bits[prompt_idx].detach().cpu().to(torch.float32),
                "tgt_vocab_entropy_bits": tgt_vocab_entropy_bits[prompt_idx].detach().cpu().to(torch.float32),
                "menu_teacher_forced_sum_logprobs": menu_sum_logprobs_cpu[prompt_idx],
                "menu_teacher_forced_first_token_logprobs": menu_first_token_logprobs_cpu[prompt_idx],
                "menu_teacher_forced_probs": menu_probs_cpu[prompt_idx],
                "menu_teacher_forced_first_token_probs": menu_first_token_probs_cpu[prompt_idx],
                "menu_teacher_forced_token_counts": (
                    candidates.group_token_counts_by_prompt[prompt_idx]
                ),
                "lang_mass_teacher_forced": {
                    lang: values[prompt_idx].detach().cpu().to(torch.float32)
                    for lang, values in lang_mass_teacher_forced.items()
                },
                "lang_dist_teacher_forced": {
                    lang: values[prompt_idx].detach().cpu().to(torch.float32)
                    for lang, values in lang_dist_teacher_forced.items()
                },
            }
        )
    return TargetStringResult(
        lang_probs_dec=lang_dist_teacher_forced,
        lang_mass_dec=lang_mass_teacher_forced,
        lang_dist_dec=lang_dist_teacher_forced,
        lang_mass_key="lang_mass_teacher_forced",
        lang_dist_key="lang_dist_teacher_forced",
        menu_lang_probs_unique_only=unique_dist_teacher_forced_by_lang,
        menu_ambiguous_mass=menu_ambiguous_mass,
        menu_unique_mass=menu_unique_mass,
        prompt_target_records=prompt_target_records,
        target_string_artifact=target_string_artifact,
        decoding_meta=decoding_meta,
        tgt_sum_logprob=tgt_sum_logprob,
        tgt_prob=tgt_prob,
        tgt_mean_logprob=tgt_mean_logprob,
        tgt_mean_prob=tgt_mean_prob,
        tgt_first_token_vocab_entropy_bits=tgt_first_token_vocab_entropy_bits,
        tgt_vocab_entropy_bits=tgt_vocab_entropy_bits,
        tgt_token_count_cpu=tgt_token_count_cpu,
    )


################################################################################
# Target-string entrypoint function
################################################################################

def run_target_string_decoding(
    *,
    model: Any,
    prompts: Sequence[str],
    prompt_ids: Sequence[str],
    target_string_records: Mapping[str, dict[str, Any]] | None,
    layer_indices: Sequence[int],
    trace_batch_size: int,
    decoding_lens_obj: Any,
    target_string_scoring_mode: str,
    require_wendler_keep: bool,
    rev: str,
    unembed_device: str | torch.device | None = None,
    prompt_token_lengths: Sequence[int] | None = None,
    prompt_anchor_positions: Sequence[int] | None = None,
) -> TargetStringResult:
    """Dispatch target-string evaluation to the configured scoring mode."""
    if target_string_scoring_mode == "start_tokens_only":
        return run_start_token_scoring(
            model=model,
            prompts=prompts,
            prompt_ids=prompt_ids,
            target_string_records=target_string_records,
            layer_indices=layer_indices,
            trace_batch_size=trace_batch_size,
            decoding_lens_obj=decoding_lens_obj,
            require_wendler_keep=require_wendler_keep,
            rev=rev,
            unembed_device=unembed_device,
            prompt_token_lengths=prompt_token_lengths,
            prompt_anchor_positions=prompt_anchor_positions,
        )
    if target_string_scoring_mode == "multi_token_teacher_forced":
        return run_teacher_forced_scoring(
            model=model,
            prompts=prompts,
            prompt_ids=prompt_ids,
            target_string_records=target_string_records,
            layer_indices=layer_indices,
            trace_batch_size=trace_batch_size,
            decoding_lens_obj=decoding_lens_obj,
            rev=rev,
            unembed_device=unembed_device,
        )
    raise ValueError(f"Unsupported target_string_scoring_mode={target_string_scoring_mode!r}")
