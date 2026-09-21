"""Evaluate one decoding or representation LLID pipeline for one revision."""

import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from args import (
    build_parser,
    get_explicit_cli_keys,
    infer_task_langs,
    is_include_data_source,
    is_pud_data_source,
    is_synthetic_data_source,
    load_config,
    pud_data_source_languages,
    resolve_config,
    resolve_synthetic_csv_path,
    validate_pud_prompt_languages,
)
from decode_then_classify import (
    DecodeThenClassifyScorer,
    decode_then_classify_rollout_streamed,
    decode_then_classify_singlepos_streamed,
    score_singlepos_latents,
)
from latents import collect_position_latents, get_transformer_layers
from lenses import build_decoding_lens
from lid import build_lid_backend
from representation import (
    canonical_gmm_task_langs,
    compute_repr_langdist_from_latents,
    compute_repr_langdist_streamed,
    load_gmm_layer,
    resolve_repr_gmm_dir,
    validate_repr_gmm_dir,
)
from run_eval_utils import (
    PromptData,
    append_mean_layer_metric_rows,
    build_langdist_summary_rows,
    build_method_langdist_artifact,
    build_pivot_summary_rows,
    build_prompt_metric_rows,
    format_prompts_for_tokenizer,
    get_prompt_content_end_token_positions,
    get_surface_word_ids_for_prompts,
    load_jsonl_prompt_data,
    load_synthetic_prompt_data,
    prepare_output_root,
    should_anchor_to_raw_prompt_content,
    write_revision_artifacts,
    write_run_outputs,
)
from target_string import TargetStringResult, run_target_string_decoding
from utils import load_nnsight_model, set_seed


@dataclass
class EvaluationResult:
    """Outputs produced by one decoding or representation evaluation pipeline."""

    method_name: str
    rows: list[dict[str, Any]]
    prompt_metric_rows: list[dict[str, Any]]
    method_artifact: dict[str, Any] | None
    metadata: dict[str, Any]
    target_string_artifact: dict[str, Any] | None
    decoded_samples: list[dict[str, Any]]


################################################################################
# Shared evaluation helpers
################################################################################

def _resolve_layer_indices(
    model: Any,
    requested_layers: Sequence[int] | None,
) -> list[int]:
    """Resolve requested layer indices, defaulting to every transformer layer."""
    model_layers = get_transformer_layers(model)
    layer_indices = (
        list(range(len(model_layers)))
        if requested_layers is None
        else list(requested_layers)
    )
    if not layer_indices:
        raise ValueError("No transformer layers available to evaluate.")
    return layer_indices


def _build_target_string_metric_tensors(
    target_result: TargetStringResult,
) -> dict[str, torch.Tensor]:
    """Collect target-string tensors that are averaged into summary rows."""
    metric_tensors = {
        "menu_ambiguous_mass": target_result.menu_ambiguous_mass,
        "menu_unique_mass": target_result.menu_unique_mass,
    }
    if target_result.tgt_sum_logprob is not None:
        metric_tensors.update(
            {
                "tgt_sum_logprob": target_result.tgt_sum_logprob,
                "tgt_prob": target_result.tgt_prob,
                "tgt_mean_logprob": target_result.tgt_mean_logprob,
                "tgt_mean_prob": target_result.tgt_mean_prob,
            }
        )
    if target_result.tgt_first_token_vocab_entropy_bits is not None:
        metric_tensors["tgt_first_token_vocab_entropy_bits"] = (
            target_result.tgt_first_token_vocab_entropy_bits
        )
    if target_result.tgt_vocab_entropy_bits is not None:
        metric_tensors["tgt_vocab_entropy_bits"] = target_result.tgt_vocab_entropy_bits
    return metric_tensors


def _add_target_string_method_artifacts(
    method_artifact: dict[str, Any],
    target_result: TargetStringResult,
    scoring_mode: str,
) -> None:
    """Add target-string-specific tensors and menu scores to a method artifact."""
    method_artifact[target_result.lang_mass_key] = {
        lang: values.detach().cpu().to(torch.float32)
        for lang, values in target_result.lang_mass_dec.items()
    }
    method_artifact[target_result.lang_dist_key] = {
        lang: values.detach().cpu().to(torch.float32)
        for lang, values in target_result.lang_dist_dec.items()
    }

    records = target_result.target_string_artifact["records"]
    if scoring_mode == "multi_token_teacher_forced":
        method_artifact["menu_teacher_forced_sum_logprobs"] = [
            record["menu_teacher_forced_sum_logprobs"] for record in records
        ]
        method_artifact["menu_teacher_forced_first_token_logprobs"] = [
            record["menu_teacher_forced_first_token_logprobs"] for record in records
        ]
        method_artifact["menu_teacher_forced_probs"] = [
            record["menu_teacher_forced_probs"] for record in records
        ]
        method_artifact["menu_teacher_forced_first_token_probs"] = [
            record["menu_teacher_forced_first_token_probs"] for record in records
        ]
        method_artifact["menu_teacher_forced_token_counts"] = [
            record["menu_teacher_forced_token_counts"] for record in records
        ]
    elif scoring_mode == "start_tokens_only":
        method_artifact["menu_start_token_logprobs"] = [
            record["menu_start_token_logprobs"] for record in records
        ]
        method_artifact["menu_start_token_probs"] = [
            record["menu_start_token_probs"] for record in records
        ]
        method_artifact["menu_start_token_counts"] = [
            record["menu_start_token_counts"] for record in records
        ]


def _add_target_string_prompt_fields(
    prompt_rows: list[dict[str, Any]],
    target_result: TargetStringResult,
    layer_indices: Sequence[int],
    scoring_mode: str,
) -> None:
    """Add target-string menu and score fields to prompt-level metric rows."""
    layer_to_offset = {
        int(layer_idx): layer_offset
        for layer_offset, layer_idx in enumerate(layer_indices)
    }
    prompt_target_records = target_result.prompt_target_records
    menu_ambiguous_mass = target_result.menu_ambiguous_mass
    menu_unique_mass = target_result.menu_unique_mass
    tgt_sum_logprob = target_result.tgt_sum_logprob
    tgt_first_token_vocab_entropy_bits = target_result.tgt_first_token_vocab_entropy_bits
    tgt_vocab_entropy_bits = target_result.tgt_vocab_entropy_bits

    for row in prompt_rows:
        prompt_idx = int(row["prompt_idx"])
        layer_offset = layer_to_offset[int(row["layer"])]
        record = prompt_target_records[prompt_idx]
        update = {
            "concept_id": record["concept_id"],
            "target_string_scoring_mode": scoring_mode,
            "menu_has_ambiguity": bool(record["menu_has_ambiguity"]),
            "menu_group_count": int(record["menu_group_count"]),
            "menu_unique_count": int(record["menu_unique_count"]),
            "menu_ambiguous_mass": float(
                menu_ambiguous_mass[prompt_idx, layer_offset].item()
            ),
            "menu_unique_mass": float(
                menu_unique_mass[prompt_idx, layer_offset].item()
            ),
        }
        if tgt_sum_logprob is not None:
            update.update(
                {
                    "tgt_sum_logprob": float(
                        tgt_sum_logprob[prompt_idx, layer_offset].item()
                    ),
                    "tgt_prob": float(
                        target_result.tgt_prob[prompt_idx, layer_offset].item()
                    ),
                    "tgt_mean_logprob": float(
                        target_result.tgt_mean_logprob[
                            prompt_idx, layer_offset
                        ].item()
                    ),
                    "tgt_mean_prob": float(
                        target_result.tgt_mean_prob[
                            prompt_idx, layer_offset
                        ].item()
                    ),
                    "tgt_token_count": int(
                        target_result.tgt_token_count_cpu[prompt_idx]
                    ),
                }
            )
        if tgt_first_token_vocab_entropy_bits is not None:
            update["tgt_first_token_vocab_entropy_bits"] = float(
                tgt_first_token_vocab_entropy_bits[prompt_idx, layer_offset].item()
            )
        if tgt_vocab_entropy_bits is not None:
            update["tgt_vocab_entropy_bits"] = float(
                tgt_vocab_entropy_bits[prompt_idx, layer_offset].item()
            )
        row.update(update)


################################################################################
# Decoding evaluation
################################################################################

def _evaluate_decoding(
    *,
    cfg: Mapping[str, Any],
    rev: str,
    model: Any,
    prompts: Sequence[str],
    prompt_data: PromptData,
    layer_indices: Sequence[int],
    prompt_token_lengths: Sequence[int],
    prompt_anchor_positions: Sequence[int] | None,
    latents: torch.Tensor | None,
    word_ids_by_prompt: Sequence[Sequence[int | None] | None] | None,
) -> EvaluationResult:
    """Run the configured decoding method and assemble its output artifacts."""
    decoding_mapping = cfg["decoding_mapping"]
    decoding_lid_backend = cfg["decoding_lid_backend"]
    decoding_token_agg = cfg["decoding_token_agg"]
    n_prompts = len(prompts)
    rows: list[dict[str, Any]] = []
    prompt_metric_rows: list[dict[str, Any]] = []
    decoded_samples: list[dict[str, Any]] = []
    target_result = None
    target_string_artifact = None

    decode_start = time.time()
    print(f"[{rev}] Stage 3/4: decoding metrics ({decoding_mapping})")
    tuned_lens_dir = None
    if cfg["decoding_lens"] == "tuned_lens":
        tuned_lens_dir = cfg["tuned_lens_dir"].format(rev=rev)
    decoding_lens_obj, decoding_meta = build_decoding_lens(
        model,
        decoding_lens=cfg["decoding_lens"],
        apply_final_norm=bool(cfg["decoding_apply_final_norm"]),
        tuned_lens_dir=tuned_lens_dir,
        device=cfg["device"],
    )

    if decoding_mapping == "target_string":
        target_result = run_target_string_decoding(
            model=model,
            prompts=prompts,
            prompt_ids=prompt_data.prompt_ids,
            target_string_records=prompt_data.target_string_records,
            layer_indices=layer_indices,
            trace_batch_size=int(cfg["trace_batch_size"]),
            decoding_lens_obj=decoding_lens_obj,
            target_string_scoring_mode=cfg["target_string_scoring_mode"],
            require_wendler_keep=bool(cfg["target_string_require_wendler_keep"]),
            unembed_device=cfg["device"],
            rev=rev,
            prompt_token_lengths=prompt_token_lengths,
            prompt_anchor_positions=prompt_anchor_positions,
        )
        lang_probs_dec = target_result.lang_probs_dec
        target_string_artifact = target_result.target_string_artifact
        decoding_meta.update(target_result.decoding_meta)
        append_mean_layer_metric_rows(
            rows,
            checkpoint_id=rev,
            method="decoding_target_string",
            n_prompts=n_prompts,
            layer_indices=layer_indices,
            metrics=_build_target_string_metric_tensors(target_result),
        )
    elif decoding_mapping == "decode_then_classify":
        decode_mode = cfg["decoding_decode_mode"]
        topk = int(cfg["decoding_topk"])
        scorer = DecodeThenClassifyScorer(
            model.tokenizer,
            build_lid_backend(decoding_lid_backend),
            candidate_languages=cfg["lid_candidate_langs"],
        )
        if decode_mode in {"rollout_argmax", "rollout_sample"}:
            lang_probs_dec, decoded_samples = decode_then_classify_rollout_streamed(
                prompts=prompts,
                layer_indices=layer_indices,
                prompt_token_lengths=prompt_token_lengths,
                trace_batch_size=int(cfg["trace_batch_size"]),
                decode_mode=decode_mode,
                token_agg=decoding_token_agg,
                rollout_k=cfg["decoding_rollout_k"],
                rollout_word_cnt=cfg["decoding_rollout_word_cnt"],
                rollout_top_p=cfg["decoding_rollout_top_p"],
                rollout_num_samples=cfg["decoding_rollout_num_samples"],
                model=model,
                decoding_lens=decoding_lens_obj,
                scorer=scorer,
                word_ids_list=word_ids_by_prompt,
                checkpoint_id=rev,
                decoding_lens_name=cfg["decoding_lens"],
            )
        else:
            print(
                f"[{rev}] Stage 3/4 detail: projecting to vocab and scoring "
                f"{len(prompts)} prompts x {len(layer_indices)} layers "
                f"with mode={decode_mode}, topk={topk}"
            )
            if latents is None:
                # Stream full traces, then select one position per prompt.
                lang_probs_dec = decode_then_classify_singlepos_streamed(
                    prompts=prompts,
                    layer_indices=layer_indices,
                    prompt_token_lengths=prompt_token_lengths,
                    trace_batch_size=int(cfg["trace_batch_size"]),
                    token_agg=decoding_token_agg,
                    model=model,
                    decoding_lens=decoding_lens_obj,
                    scorer=scorer,
                    word_ids_list=word_ids_by_prompt,
                    decode_mode=decode_mode,
                    topk=topk,
                    desc=f"Streaming decode traces ({decode_mode})",
                )
            else:
                # Score the single-position latents collected in Stage 2.
                lang_probs_dec = score_singlepos_latents(
                    latents=latents,
                    layer_indices=layer_indices,
                    decoding_lens=decoding_lens_obj,
                    scorer=scorer,
                    decode_mode=decode_mode,
                    topk=topk,
                    desc=f"Decode/LID {decode_mode} ({rev})",
                )
    else:
        raise ValueError(f"Unsupported decoding_mapping: {decoding_mapping}")
    print(f"[{rev}] Stage 3/4 complete in {time.time() - decode_start:.1f}s")

    rows.extend(
        build_langdist_summary_rows(
            checkpoint_id=rev,
            method="decoding",
            layer_indices=layer_indices,
            langdist=lang_probs_dec,
            n_prompts=n_prompts,
            decoding_mapping=decoding_mapping,
            decoding_lid_backend=decoding_lid_backend,
            decoding_lens=cfg["decoding_lens"],
        )
    )

    method_artifact = None
    if cfg["save_full_lang_probs"]:
        method_artifact = build_method_langdist_artifact(
            langdist=lang_probs_dec,
            decoding_mapping=decoding_mapping,
            decoding_lid_backend=decoding_lid_backend,
            decoding_lens=cfg["decoding_lens"],
            target_score_semantics=decoding_meta.get("target_score_semantics"),
            target_mass_semantics=decoding_meta.get("target_mass_semantics"),
            target_dist_semantics=decoding_meta.get("target_dist_semantics"),
            target_string_scoring_mode=decoding_meta.get(
                "target_string_scoring_mode"
            ),
            target_string_execution_path=decoding_meta.get(
                "target_string_execution_path"
            ),
            extra_langdist_views=(
                {"menu_lang_probs_unique_only": target_result.menu_lang_probs_unique_only}
                if target_result is not None
                else None
            ),
        )
        if target_result is not None:
            _add_target_string_method_artifacts(
                method_artifact,
                target_result,
                cfg["target_string_scoring_mode"],
            )

    if cfg["write_prompt_metrics"]:
        prompt_metric_rows = build_prompt_metric_rows(
            checkpoint_id=rev,
            method="decoding",
            layer_indices=layer_indices,
            langdist=lang_probs_dec,
            prompt_ids=prompt_data.prompt_ids,
            prompt_langs=prompt_data.prompt_langs,
            decoding_mapping=decoding_mapping,
            decoding_lid_backend=decoding_lid_backend,
            decoding_lens=cfg["decoding_lens"],
            task_langs_by_prompt=prompt_data.task_langs_by_prompt,
        )
        if target_result is not None:
            _add_target_string_prompt_fields(
                prompt_metric_rows,
                target_result,
                layer_indices,
                cfg["target_string_scoring_mode"],
            )

    rows.extend(
        build_pivot_summary_rows(
            checkpoint_id=rev,
            method="decoding_pivot",
            layer_indices=layer_indices,
            langdist=lang_probs_dec,
            n_prompts=n_prompts,
            task_langs_by_prompt=prompt_data.task_langs_by_prompt,
            decoding_mapping=decoding_mapping,
            decoding_lid_backend=decoding_lid_backend,
            decoding_lens=cfg["decoding_lens"],
        )
    )
    return EvaluationResult(
        method_name="decoding",
        rows=rows,
        prompt_metric_rows=prompt_metric_rows,
        method_artifact=method_artifact,
        metadata=decoding_meta,
        target_string_artifact=target_string_artifact,
        decoded_samples=decoded_samples,
    )


################################################################################
# Representation evaluation
################################################################################

def _evaluate_representation(
    *,
    cfg: Mapping[str, Any],
    rev: str,
    model: Any,
    prompts: Sequence[str],
    prompt_data: PromptData,
    layer_indices: Sequence[int],
    prompt_token_lengths: Sequence[int],
    latents: torch.Tensor | None,
    word_ids_by_prompt: Sequence[Sequence[int | None] | None] | None,
    need_full_repr: bool,
    repr_gmm_dir: str | Path,
    gmm_task_langs_by_prompt: Sequence[Sequence[str]],
) -> EvaluationResult:
    """Run representation GMM scoring and assemble its output artifacts."""
    n_prompts = len(prompts)
    rows: list[dict[str, Any]] = []
    prompt_metric_rows: list[dict[str, Any]] = []
    repr_meta = {"gmm_mode": None, "pca_k_by_layer": {}, "word_ids_used": False}
    repr_start = time.time()
    print(f"[{rev}] Stage 3/4: representation metrics")

    if need_full_repr:
        print(
            f"[{rev}] Stage 3/4 detail: scoring repr posteriors across "
            f"{len(prompts)} prompts x {len(layer_indices)} layers "
            f"(token_agg={cfg['repr_token_agg']}, unit={cfg['repr_unit']})"
        )
        if cfg["repr_unit"] == "token":
            scoring_word_ids_by_prompt = word_ids_by_prompt
        elif cfg["repr_unit"] == "subtoken":
            # Subtoken-unit repr ignores word boundaries.
            scoring_word_ids_by_prompt = [None for _ in prompts]
        else:
            raise ValueError(f"Unsupported repr_unit: {cfg['repr_unit']}")

        lang_probs_repr, word_ids_used, pca_k_by_layer, gmm_mode = (
            compute_repr_langdist_streamed(
                model=model,
                prompts=prompts,
                layer_indices=layer_indices,
                prompt_token_lengths=prompt_token_lengths,
                trace_batch_size=int(cfg["trace_batch_size"]),
                gmm_dir=repr_gmm_dir,
                repr_priors=cfg["repr_priors"],
                repr_unit=cfg["repr_unit"],
                repr_token_agg=cfg["repr_token_agg"],
                repr_rollout_k=int(cfg["repr_rollout_k"]),
                word_ids_by_prompt=scoring_word_ids_by_prompt,
            )
        )
        repr_meta["word_ids_used"] = word_ids_used
        if cfg["repr_unit"] == "token":
            repr_meta["word_ids_source"] = (
                "word_ids" if word_ids_used else "fallback"
            )
        else:
            repr_meta["word_ids_source"] = "subtoken"
        repr_meta["pca_k_by_layer"] = pca_k_by_layer
        repr_meta["gmm_mode"] = gmm_mode
    else:
        print(
            f"[{rev}] Stage 3/4 detail: scoring repr posteriors across "
            f"{latents.shape[0]} prompts x {len(layer_indices)} layers "
            "(last-token path)"
        )
        priors_override = (
            "uniform" if cfg["repr_priors"] == "uniform" else "empirical"
        )
        lang_probs_repr = compute_repr_langdist_from_latents(
            latents=latents,
            layer_indices=layer_indices,
            gmm_dir=repr_gmm_dir,
            use_priors=True,
            priors_override=priors_override,
        )
        for layer_idx in layer_indices:
            loaded = load_gmm_layer(Path(repr_gmm_dir) / f"layer_{layer_idx}.pt")
            if repr_meta["gmm_mode"] is None:
                repr_meta["gmm_mode"] = loaded.mode
            if loaded.pca is not None:
                repr_meta["pca_k_by_layer"][int(layer_idx)] = loaded.pca.k
        repr_meta["word_ids_source"] = "subtoken"

    rows.extend(
        build_langdist_summary_rows(
            checkpoint_id=rev,
            method="repr_gmm",
            layer_indices=layer_indices,
            langdist=lang_probs_repr,
            n_prompts=n_prompts,
        )
    )
    method_artifact = None
    if cfg["save_full_lang_probs"]:
        method_artifact = build_method_langdist_artifact(
            langdist=lang_probs_repr,
            decoding_mapping=None,
        )
    if cfg["write_prompt_metrics"]:
        prompt_metric_rows = build_prompt_metric_rows(
            checkpoint_id=rev,
            method="repr_gmm",
            layer_indices=layer_indices,
            langdist=lang_probs_repr,
            prompt_ids=prompt_data.prompt_ids,
            prompt_langs=prompt_data.prompt_langs,
            task_langs_by_prompt=gmm_task_langs_by_prompt,
        )
    print(f"[{rev}] Stage 3/4 complete in {time.time() - repr_start:.1f}s")

    rows.extend(
        build_pivot_summary_rows(
            checkpoint_id=rev,
            method="repr_gmm_pivot",
            layer_indices=layer_indices,
            langdist=lang_probs_repr,
            n_prompts=n_prompts,
            task_langs_by_prompt=gmm_task_langs_by_prompt,
        )
    )
    repr_meta["pca_used"] = bool(repr_meta["pca_k_by_layer"])
    return EvaluationResult(
        method_name="repr_gmm",
        rows=rows,
        prompt_metric_rows=prompt_metric_rows,
        method_artifact=method_artifact,
        metadata=repr_meta,
        target_string_artifact=None,
        decoded_samples=[],
    )


################################################################################
# Entrypoint
################################################################################

def main():
    # 1) Parse args with defaults from default.json
    default_cfg = load_config("configs/default.json")
    parser = build_parser(default_cfg)
    args = parser.parse_args()

    # 2) Resolve config/overrides
    explicit_cli_keys = get_explicit_cli_keys(parser, sys.argv[1:])
    explicit_keys = set(explicit_cli_keys)
    cfg = dict(default_cfg)
    for key in explicit_cli_keys:
        cfg[key] = getattr(args, key)

    cfg = resolve_config(cfg, explicit_keys)

    set_seed(cfg["seed"])

    if (
        cfg["max_prompts"] is not None
        and cfg["max_prompts_per_lang"] is not None
    ):
        raise ValueError("Set only one of max_prompts or max_prompts_per_lang, not both.")

    # 3) Load prompt data and resolve languages
    decoding_mapping = cfg["decoding_mapping"]
    prompt_paths = []
    if cfg["prompts_path"]:
        prompt_paths.append(Path(cfg["prompts_path"]))
    prompt_paths.extend(Path(p) for p in cfg.get("extra_prompts_paths", []))
    if is_pud_data_source(cfg["data_source"]) or is_include_data_source(cfg["data_source"]):
        selected_task_langs = (
            pud_data_source_languages(cfg["data_source"], cfg["gmm_setup_registry"])
            if is_pud_data_source(cfg["data_source"])
            else None
        )
        prompt_data = load_jsonl_prompt_data(
            prompt_paths,
            cfg["max_prompts"],
            cfg["max_prompts_per_lang"],
            selected_task_langs=selected_task_langs,
        )
        if is_pud_data_source(cfg["data_source"]):
            validate_pud_prompt_languages(
                cfg["data_source"],
                prompt_data.prompt_langs,
                cfg["gmm_setup_registry"],
            )
    elif is_synthetic_data_source(cfg["data_source"]):
        synthetic_path = resolve_synthetic_csv_path(
            cfg["synthetic_root"], cfg["data_source"]
        )
        prompt_data = load_synthetic_prompt_data(
            synthetic_path,
            cfg["data_source"],
            cfg["max_prompts"],
            cfg["max_prompts_per_lang"],
            selected_task_langs=set(cfg.get("synthetic_langs") or []) or None,
            start_token_artifact_key=(
                str(cfg["model_name"])
                if decoding_mapping == "target_string"
                and cfg.get("target_string_scoring_mode") == "start_tokens_only"
                else None
            ),
        )
    else:
        raise ValueError(f"Unsupported data_source: {cfg['data_source']}")
    raw_prompts = prompt_data.prompts
    prompt_ids = prompt_data.prompt_ids
    prompt_langs = prompt_data.prompt_langs
    task_langs_by_prompt = prompt_data.task_langs_by_prompt
    surface_tokens_by_prompt = prompt_data.surface_tokens_by_prompt
    task_langs = infer_task_langs(task_langs_by_prompt, prompt_langs)
    if not task_langs:
        raise ValueError("Could not infer task languages from prompt metadata.")
    requested_lid_candidate_langs = sorted(
        set(cfg.get("lid_candidate_langs") or [])
    )
    if requested_lid_candidate_langs:
        missing_task_langs = sorted(
            set(task_langs) - set(requested_lid_candidate_langs)
        )
        if missing_task_langs:
            raise ValueError(
                "--lid-candidate-langs must include every task language inferred from "
                f"the prompt metadata; missing: {missing_task_langs}"
            )
        lid_candidate_langs = requested_lid_candidate_langs
    else:
        lid_candidate_langs = task_langs
    cfg["lid_candidate_langs"] = lid_candidate_langs
    gmm_task_langs_by_prompt = canonical_gmm_task_langs(
        task_langs_by_prompt
    )
    if (
        cfg["do_decoding"]
        and decoding_mapping == "target_string"
        and any(not prompt_id for prompt_id in prompt_ids)
    ):
        raise ValueError("Each prompt row must include 'id' for target_string mapping.")

    # 4) Experiment logging
    output_root = prepare_output_root(cfg)

    # 5) Evaluate the requested revision
    rev = cfg["revision"]
    rev_start = time.time()
    repr_gmm_dir = None
    repr_gmm_setup_name = None
    repr_gmm_setup = None
    if cfg["do_repr"]:
        repr_gmm_dir, repr_gmm_setup_name, repr_gmm_setup = resolve_repr_gmm_dir(
            cfg,
            prompt_langs=prompt_langs,
            revision=rev,
        )
        print(f"[{rev}] Repr precheck: validating GMM artifacts in {repr_gmm_dir}")
        gmm_check = validate_repr_gmm_dir(
            repr_gmm_dir,
            expected_prompt_langs=prompt_langs,
            expected_setup_name=repr_gmm_setup_name,
            expected_setup=repr_gmm_setup,
        )
        print(
            f"[{rev}] Repr precheck complete: {gmm_check['n_layers']} layer files, "
            f"languages={gmm_check['languages']}"
        )
    
    # 5.1) Model loading + tracing
    print(f"[{rev}] Stage 1/4: loading model")
    model = load_nnsight_model(
        model_name=cfg["model_name"],
        revision=rev,
        device=cfg["device"],
        seed=cfg["seed"],
    )
    model.sampled_rollout_prompt_batch_size = int(cfg["sampled_rollout_prompt_batch_size"])
    prompts, prompt_content_start_offsets = format_prompts_for_tokenizer(
        model.tokenizer,
        raw_prompts,
    )
    prompt_token_lengths = [
        len(model.tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in prompts
    ]
    prompt_anchor_positions = None
    if should_anchor_to_raw_prompt_content(cfg["data_source"]):
        prompt_anchor_positions = get_prompt_content_end_token_positions(
            model.tokenizer,
            prompts,
            raw_prompts,
            prompt_content_start_offsets,
        )
    decoding_mapping = cfg["decoding_mapping"]
    decoding_token_agg = cfg["decoding_token_agg"]

    need_full_repr = False
    need_full_decoding = False
    if cfg["do_repr"]:
        need_full_repr = (
            (
                cfg["repr_unit"] == "token"
                and not is_synthetic_data_source(cfg["data_source"])
            )
            or (cfg["repr_token_agg"] != "last_token")
            or (int(cfg["repr_rollout_k"]) > 0)
        )
    elif cfg["do_decoding"] and decoding_mapping == "decode_then_classify":
        # Rollout scores multiple prompt positions, so it still needs full traces
        # even when the anchor aggregation mode itself is "last_token".
        need_full_decoding = (
            decoding_token_agg != "last_token"
            or cfg["decoding_decode_mode"] in {"rollout_argmax", "rollout_sample"}
        )
    stream_full_latents = need_full_repr or need_full_decoding
    need_prompt_latents = (
        cfg["do_repr"] or decoding_mapping == "decode_then_classify"
    )

    latents = None
    word_ids_by_prompt = None
    latent_start = time.time()

    if not need_prompt_latents:
        print(
            f"[{rev}] Stage 2/4: skipping prompt latent collection "
            "(target_string decoding)"
        )
    elif stream_full_latents:
        print(
            f"[{rev}] Stage 2/4: deferring full latent collection to streamed "
            "scoring"
        )
    else:
        print(f"[{rev}] Stage 2/4: collecting latents")

    if not need_prompt_latents or stream_full_latents:
        layer_indices = _resolve_layer_indices(model, cfg["layers"])

    if cfg["do_repr"]:
        needs_word_ids = need_full_repr and (
            cfg["repr_unit"] == "token"
            or cfg["repr_token_agg"] in {"frac_50", "frac_75"}
            or int(cfg["repr_rollout_k"]) > 0
        )
    elif cfg["do_decoding"] and decoding_mapping == "decode_then_classify":
        needs_word_ids = (
            decoding_token_agg in {"frac_50", "frac_75"}
            or cfg["decoding_decode_mode"] in {"rollout_argmax", "rollout_sample"}
        )
    else:
        needs_word_ids = False

    if need_prompt_latents and stream_full_latents and needs_word_ids:
        word_ids_by_prompt = get_surface_word_ids_for_prompts(
            model.tokenizer,
            prompts,
            surface_tokens_by_prompt=surface_tokens_by_prompt,
            content_start_offsets=prompt_content_start_offsets,
        )
    if need_prompt_latents and not stream_full_latents:
        latents, layer_indices = collect_position_latents(
            model=model,
            prompts=prompts,
            layer_indices=cfg["layers"],
            position=-1,
            positions_by_prompt=prompt_anchor_positions,
            sequence_lengths=prompt_token_lengths,
            trace_batch_size=int(cfg["trace_batch_size"]),
        )
    print(f"[{rev}] Stage 2/4 complete in {time.time() - latent_start:.1f}s")

    # Checkpoint metadata wraps the output of the selected evaluation pipeline.
    full_lang_probs_payload = {
        "checkpoint_id": rev,
        "layer_indices": [int(layer_idx) for layer_idx in layer_indices],
        "prompt_ids": list(prompt_ids),
        "prompt_langs": list(prompt_langs),
        "methods": {},
    }

    # 5.2) Run the selected evaluation pipeline
    if cfg["do_decoding"]:
        result = _evaluate_decoding(
            cfg=cfg,
            rev=rev,
            model=model,
            prompts=prompts,
            prompt_data=prompt_data,
            layer_indices=layer_indices,
            prompt_token_lengths=prompt_token_lengths,
            prompt_anchor_positions=prompt_anchor_positions,
            latents=latents,
            word_ids_by_prompt=word_ids_by_prompt,
        )
    else:
        result = _evaluate_representation(
            cfg=cfg,
            rev=rev,
            model=model,
            prompts=prompts,
            prompt_data=prompt_data,
            layer_indices=layer_indices,
            prompt_token_lengths=prompt_token_lengths,
            latents=latents,
            word_ids_by_prompt=word_ids_by_prompt,
            need_full_repr=need_full_repr,
            repr_gmm_dir=repr_gmm_dir,
            gmm_task_langs_by_prompt=gmm_task_langs_by_prompt,
        )
    if result.method_artifact is not None:
        full_lang_probs_payload["methods"][result.method_name] = (
            result.method_artifact
        )

    del model
    write_revision_artifacts(
        output_root=output_root,
        revision=rev,
        cfg=cfg,
        full_lang_probs_payload=full_lang_probs_payload,
        target_string_artifact=result.target_string_artifact,
    )
    print(f"[{rev}] Stage 4/4: finished revision in {time.time() - rev_start:.1f}s")

    # 6) Save outputs
    parquet_path = write_run_outputs(
        output_root=output_root,
        cfg=cfg,
        rows=result.rows,
        prompt_metric_rows=result.prompt_metric_rows,
        repr_meta_payload=result.metadata if cfg["do_repr"] else None,
        decoding_meta_payload=result.metadata if cfg["do_decoding"] else None,
        decoded_samples_all=result.decoded_samples,
    )
    print(f"Wrote {parquet_path}", flush=True)
    if cfg.get("hard_exit_after_success"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
