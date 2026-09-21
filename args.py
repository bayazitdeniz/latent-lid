"""CLI and config resolution for run_eval."""

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from gmm_setup import load_gmm_setup_registry
from lenses import build_tuned_lens_base_dir


PAPER_REPR_DEFAULTS = {
    "repr_priors": "uniform",
    "repr_pca": "layerwise",
    "repr_pca_variance": 0.98,
    "repr_cov": "diag",
    "repr_unit": "token",
}
DATASET_RUNTIME_PARAM_KEYS = (
    "max_prompts",
    "max_prompts_per_lang",
    "trace_batch_size",
    "sampled_rollout_prompt_batch_size",
)
PROMPT_CAP_KEYS = ("max_prompts", "max_prompts_per_lang")


################################################################################
# Data-source resolution
################################################################################

INCLUDE_DATA_SOURCES = (
    "include_10lang_3domain_all",
    "include_10lang_3domain_cap30",
)
INCLUDE_GMM_SETUP = "include_10lang_en"
INCLUDE_PROMPT_STYLE_FILES = {
    "question_only": "prompts_question_only.jsonl",
    "minimal_mcq": "prompts_minimal_mcq.jsonl",
}
PUD_DATA_SOURCES = (
    "pud9",
    "pud9_ud6",
    "pud21",
    "pud21_ud6",
)


def parse_data_source_arg(value: str) -> str:
    static_sources = {*PUD_DATA_SOURCES, *INCLUDE_DATA_SOURCES, "copy", "translation", "cloze"}
    if value in static_sources:
        return value
    if value.startswith("translation_to_") and value.removeprefix("translation_to_").strip():
        return value
    allowed = ", ".join(sorted(static_sources))
    raise argparse.ArgumentTypeError(
        f"invalid data_source={value!r}; expected one of: {allowed}, or translation_to_<lang>"
    )


def is_synthetic_data_source(data_source: str) -> bool:
    return data_source in {"copy", "translation", "cloze"} or data_source.startswith("translation_to_")


def synthetic_base_task(data_source: str) -> str:
    if data_source.startswith("translation_to_"):
        return "translation"
    return data_source


def synthetic_target_lang_filter(data_source: str) -> str | None:
    if data_source.startswith("translation_to_"):
        lang = data_source.removeprefix("translation_to_").strip()
        if not lang:
            raise ValueError("translation_to_<lang> data_source requires a non-empty target language.")
        return lang
    return None


def is_include_data_source(data_source: str) -> bool:
    return data_source in INCLUDE_DATA_SOURCES


def is_pud_data_source(data_source: str) -> bool:
    # Keep legacy config files readable, but new CLI choices use setup-named PUD sources.
    return data_source == "pud" or data_source in PUD_DATA_SOURCES


def pud_data_source_languages(data_source: str, registry_path: str | Path) -> set[str] | None:
    if data_source not in PUD_DATA_SOURCES:
        return None
    registry = load_gmm_setup_registry(registry_path)
    if data_source not in registry:
        raise ValueError(f"Unknown PUD data_source={data_source!r}; available setups: {sorted(registry)}")
    return set(registry[data_source]["languages"])


def validate_pud_prompt_languages(
    data_source: str,
    prompt_langs: Sequence[str | None],
    registry_path: str | Path,
) -> None:
    """Reject prompt languages outside the selected PUD setup."""
    expected_langs = pud_data_source_languages(data_source, registry_path)
    if expected_langs is None:
        return
    observed_langs = {lang for lang in prompt_langs if lang}
    extra_langs = observed_langs - expected_langs
    if extra_langs:
        raise ValueError(
            f"data_source={data_source} expects prompt languages to be a subset of "
            f"{sorted(expected_langs)}, but loaded unexpected languages {sorted(extra_langs)}. "
            "Use a matching prompts_path or choose the PUD data_source for this prompt file."
        )


def resolve_pud_prompts_path(pud_root: str | Path, split_mode: str) -> Path:
    root = Path(pud_root)
    if split_mode == "overlap":
        return root / "pud_prompts_train.jsonl"
    if split_mode == "heldout":
        return root / "pud_prompts_test.jsonl"
    raise ValueError(f"Unsupported pud_split_mode: {split_mode}")


def resolve_ud_prompts_path(ud_root: str | Path, split_mode: str) -> Path:
    root = Path(ud_root)
    if split_mode == "overlap":
        return root / "ud_prompts_train.jsonl"
    if split_mode == "heldout":
        return root / "ud_prompts_test.jsonl"
    raise ValueError(f"Unsupported pud_split_mode: {split_mode}")


def resolve_include_prompts_path(include_root: str | Path, data_source: str, prompt_style: str) -> Path:
    if data_source not in INCLUDE_DATA_SOURCES:
        raise ValueError(f"Unsupported INCLUDE data_source: {data_source}")
    if prompt_style not in INCLUDE_PROMPT_STYLE_FILES:
        raise ValueError(f"Unsupported INCLUDE prompt style: {prompt_style}")
    return Path(include_root) / data_source / INCLUDE_PROMPT_STYLE_FILES[prompt_style]


def resolve_synthetic_csv_path(synthetic_root: str | Path, data_source: str) -> Path:
    base_task = synthetic_base_task(data_source)
    if base_task not in {"copy", "translation", "cloze"}:
        raise ValueError(f"Unsupported synthetic data_source: {data_source}")
    return Path(synthetic_root) / f"{base_task}.csv"


def infer_task_langs(
    task_langs_by_prompt: Sequence[Sequence[str]] | None,
    prompt_langs: Sequence[str | None] | None,
) -> list[str]:
    """Return the task-language union inferred from prompt metadata."""
    langs = {
        str(lang)
        for task_langs in (task_langs_by_prompt or [])
        for lang in task_langs
        if lang
    }
    if not langs:
        langs = {str(lang) for lang in (prompt_langs or []) if lang}
    return sorted(langs)


def infer_prompt_lang_from_id(prompt_id: str) -> str:
    """Infer a prompt language from a language-prefixed prompt ID."""
    if "_" in prompt_id:
        return prompt_id.split("_", 1)[0]
    if "-" in prompt_id:
        return prompt_id.split("-", 1)[0]
    return prompt_id


################################################################################
# CLI and config inputs
################################################################################

def load_config(path: str | None) -> dict[str, Any]:
    """Load JSON config if provided; return empty dict otherwise."""
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run decoding + repr LLID evaluation.")
    
    # general eval args
    parser.add_argument(
        "--dataset-runtime-params-config",
        default=defaults["dataset_runtime_params_config"],
        help="Dataset/model runtime parameters for prompt caps and batch sizes. Explicit CLI args take precedence.",
    )
    parser.add_argument("--model-name", default=defaults["model_name"], help="HF model name or local path.")
    parser.add_argument(
        "--hard-exit-after-success",
        action="store_true",
        default=None,
        help=(
            "After all artifacts are written, flush output and call os._exit(0). "
            "Useful for native CUDA/model finalizer segfaults during interpreter shutdown."
        ),
    )
    parser.add_argument("--revision", default=defaults["revision"], help="Single revision to evaluate.")
    parser.add_argument(
        "--data-source",
        type=parse_data_source_arg,
        default=defaults["data_source"],
        help="Dataset source to evaluate. Supports PUD/INCLUDE names, copy/cloze/translation, and translation_to_<lang>.",
    )
    parser.add_argument("--prompts-path", default=defaults["prompts_path"], help="JSONL prompts path.")
    parser.add_argument(
        "--extra-prompts-paths",
        nargs="*",
        default=defaults["extra_prompts_paths"],
        help="Additional JSONL prompt paths to append after prompts-path.",
    )
    parser.add_argument(
        "--pud-root",
        default=defaults["pud_root"],
        help="Root directory for canonical PUD split assets.",
    )
    parser.add_argument(
        "--synthetic-root",
        default=defaults["synthetic_root"],
        help="Root directory for Wendler-style synthetic task CSVs.",
    )
    parser.add_argument(
        "--synthetic-langs",
        nargs="*",
        default=defaults.get("synthetic_langs", []),
        help="Optional source/target language filter for synthetic CSV tasks.",
    )
    parser.add_argument(
        "--ud-root",
        default=defaults["ud_root"],
        help="Root directory for non-parallel UD holdout assets.",
    )
    parser.add_argument(
        "--include-nonparallel-ud",
        type=lambda v: v.lower() in {"1", "true", "t", "yes", "y"},
        default=defaults["include_nonparallel_ud"],
        help="Append non-parallel UD holdout prompts matching pud-split-mode.",
    )
    parser.add_argument(
        "--include-root",
        default=defaults["include_root"],
        help="Root directory containing materialized INCLUDE subset directories.",
    )
    parser.add_argument(
        "--include-prompt-style",
        choices=sorted(INCLUDE_PROMPT_STYLE_FILES),
        default=defaults["include_prompt_style"],
        help="Prompt rendering to use for materialized INCLUDE subsets.",
    )
    parser.add_argument(
        "--pud-split-mode",
        choices=["overlap", "heldout"],
        default=defaults["pud_split_mode"],
        help="For canonical PUD split assets: overlap evaluates on train prompts, heldout evaluates on test prompts.",
    )
    parser.add_argument("--output-dir", default=defaults["output_dir"], help="Output root directory for logs.")
    parser.add_argument("--exp-id", default=defaults["exp_id"], help="Experiment id used for logs folder.")
    parser.add_argument("--max-prompts", type=int, default=defaults["max_prompts"], help="Max prompts to load.")
    parser.add_argument(
        "--max-prompts-per-lang",
        type=int,
        default=defaults["max_prompts_per_lang"],
        help="Optional per-language cap (requires prompt ids).",
    )
    # Keep --eval-langs as a backward-compatible alias for existing run commands.
    parser.add_argument(
        "--lid-candidate-langs",
        "--eval-langs",
        nargs="*",
        default=defaults.get(
            "lid_candidate_langs",
            defaults.get("eval_langs", []),
        ),
        help=(
            "Optional candidate language set for LLID scoring. "
            "If omitted, candidates are inferred from prompt task languages."
        ),
    )
    parser.add_argument("--device", default=defaults["device"], help="Device map for model loading.")
    parser.add_argument("--seed", type=int, default=defaults["seed"], help="Random seed for model loading.")
    parser.add_argument("--layers", nargs="*", type=int, default=defaults["layers"], help="Layer indices to probe.")
    parser.add_argument("--trace-batch-size", type=int, default=defaults["trace_batch_size"])
    parser.add_argument(
        "--sampled-rollout-prompt-batch-size",
        type=int,
        default=defaults["sampled_rollout_prompt_batch_size"],
    )

    # artifact/output args
    parser.add_argument(
        "--write-prompt-metrics",
        action="store_true",
        default=None,
        help="Write per-prompt per-layer metrics for downstream visualization.",
    )
    parser.add_argument(
        "--save-full-lang-probs",
        action="store_true",
        default=None,
        help="Save full per-language prompt/layer distributions in a compact tensor artifact.",
    )

    # repr-specific args
    parser.add_argument(
        "--do-repr",
        type=lambda v: v.lower() in {"1", "true", "t", "yes", "y"},
        default=defaults["do_repr"],
        help="Enable repr metrics (true/false).",
    )
    parser.add_argument("--repr-priors", choices=["uniform", "empirical"], default=defaults["repr_priors"])
    parser.add_argument("--repr-pca", choices=["none", "layerwise"], default=defaults["repr_pca"])
    parser.add_argument("--repr-pca-variance", type=float, default=defaults["repr_pca_variance"])
    parser.add_argument("--repr-cov", choices=["diag", "full"], default=defaults["repr_cov"])
    parser.add_argument(
        "--repr-token-agg",
        choices=["last_token", "all_tokens", "frac_50", "frac_75"],
        default=defaults["repr_token_agg"],
    )
    parser.add_argument("--repr-rollout-k", type=int, default=defaults["repr_rollout_k"])
    parser.add_argument("--repr-unit", choices=["subtoken", "token"], default=defaults["repr_unit"])
    parser.add_argument("--repr-gmm-dir", default=defaults["repr_gmm_dir"])
    parser.add_argument("--repr-gmm-setup", default=defaults["repr_gmm_setup"])
    parser.add_argument("--gmm-setup-registry", default=defaults["gmm_setup_registry"])

    # decoding-specific args
    parser.add_argument(
        "--do-decoding",
        type=lambda v: v.lower() in {"1", "true", "t", "yes", "y"},
        default=defaults["do_decoding"],
        help="Enable decoding metrics (true/false).",
    )
    parser.add_argument(
        "--decoding-lens",
        choices=["raw_logitlens", "tuned_lens"],
        default=defaults["decoding_lens"],
    )
    parser.add_argument(
        "--decoding-apply-final-norm",
        type=lambda v: v.lower() in {"1", "true", "t", "yes", "y"},
        default=defaults["decoding_apply_final_norm"],
    )
    parser.add_argument(
        "--tuned-lens-root",
        default=defaults["tuned_lens_root"],
        help="Artifact root used to infer a tuned-lens directory from model and revision.",
    )
    parser.add_argument(
        "--tuned-lens-dir",
        default=defaults["tuned_lens_dir"],
        help="Exact tuned-lens directory; overrides model/revision inference.",
    )
    parser.add_argument("--tuned-lens-temperature", type=float, default=defaults["tuned_lens_temperature"])
    parser.add_argument(
        "--decoding-token-agg",
        choices=["last_token", "all_mean", "frac_50", "frac_75"],
        default=defaults["decoding_token_agg"],
    )
    parser.add_argument(
        "--decoding-mapping",
        choices=["target_string", "decode_then_classify"],
        default=defaults["decoding_mapping"],
    )

    # target-string decoding args
    parser.add_argument(
        "--target-string-scoring-mode",
        choices=["multi_token_teacher_forced", "start_tokens_only"],
        default=defaults["target_string_scoring_mode"],
        help=(
            "For decoding_mapping=target_string, choose prompt-only Wendler Start(w) "
            "scoring or optional teacher-forced full target-string scoring."
        ),
    )
    parser.add_argument(
        "--target-string-require-wendler-keep",
        type=lambda v: v.lower() in {"1", "true", "t", "yes", "y"},
        default=defaults["target_string_require_wendler_keep"],
        help="Fail if a prompt is marked as filtered by Wendler-style target/English start-token overlap checks.",
    )
    # Decode-then-classify args
    parser.add_argument(
        "--decoding-lid-backend",
        choices=["langid", "glotlid"],
        default=defaults["decoding_lid_backend"],
    )
    parser.add_argument(
        "--decoding-decode-mode",
        choices=["topk_weighted", "rollout", "rollout_argmax", "rollout_sample"],
        default=defaults["decoding_decode_mode"],
    )
    parser.add_argument("--decoding-topk", type=int, default=defaults["decoding_topk"])
    parser.add_argument("--decoding-rollout-k", type=int, default=defaults["decoding_rollout_k"])
    parser.add_argument(
        "--decoding-rollout-word-cnt",
        type=int,
        default=defaults["decoding_rollout_word_cnt"],
    )
    parser.add_argument("--decoding-rollout-top-p", type=float, default=defaults["decoding_rollout_top_p"])
    parser.add_argument("--decoding-rollout-num-samples", type=int, default=defaults["decoding_rollout_num_samples"])

    return parser


def get_explicit_cli_keys(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> list[str]:
    """Return explicitly supplied option destinations in parser order."""
    explicit_keys = []
    for action in parser._actions:
        if not action.option_strings:
            continue
        if any(arg == opt or arg.startswith(f"{opt}=") for arg in argv for opt in action.option_strings):
            explicit_keys.append(action.dest)
    return explicit_keys


################################################################################
# Config resolution
################################################################################

def _apply_dataset_runtime_params(cfg: dict[str, Any], explicit_keys: set[str]) -> None:
    """Apply dataset/model runtime parameters unless explicitly overridden."""
    path = cfg.get("dataset_runtime_params_config")
    if not path:
        return

    defaults_cfg = load_config(path)
    dataset_key = (
        "translation"
        if str(cfg["data_source"]).startswith("translation_to_")
        else cfg["data_source"]
    )
    dataset_cfg = (defaults_cfg.get("datasets") or {}).get(dataset_key)
    if not dataset_cfg:
        return

    resolved: dict[str, Any] = {}
    resolved.update(dataset_cfg.get("default") or {})

    model_cfgs = dataset_cfg.get("models") or {}
    model_cfg = model_cfgs.get(cfg["model_name"])
    if model_cfg:
        resolved.update(model_cfg)

    prompt_cap_explicit = any(key in explicit_keys for key in PROMPT_CAP_KEYS)
    for key in DATASET_RUNTIME_PARAM_KEYS:
        if prompt_cap_explicit and key in PROMPT_CAP_KEYS:
            continue
        if key not in explicit_keys and key in resolved:
            cfg[key] = resolved[key]

    if "max_prompts" in explicit_keys and "max_prompts_per_lang" not in explicit_keys:
        cfg["max_prompts_per_lang"] = 0
    if "max_prompts_per_lang" in explicit_keys and "max_prompts" not in explicit_keys:
        cfg["max_prompts"] = 0


def _apply_repr_defaults(cfg: dict[str, Any]) -> None:
    for key, val in PAPER_REPR_DEFAULTS.items():
        if cfg.get(key) is None:
            cfg[key] = val


def _apply_synthetic_eval_policy(cfg: dict[str, Any], explicit_keys: set[str]) -> None:
    """Apply synthetic-task defaults and validate the enabled stage."""
    if not is_synthetic_data_source(cfg["data_source"]):
        return

    # Synthetic target-string decoding defaults to the paper Start(w) setting.
    if cfg["do_decoding"] and "decoding_mapping" not in explicit_keys:
        cfg["decoding_mapping"] = "target_string"
    if cfg["do_decoding"] and cfg["decoding_token_agg"] != "last_token":
        raise ValueError(
            f"Synthetic data_source={cfg['data_source']} requires decoding_token_agg=last_token."
        )
    if cfg["do_repr"] and cfg["repr_token_agg"] != "last_token":
        raise ValueError(
            f"Synthetic data_source={cfg['data_source']} requires repr_token_agg=last_token."
        )
    if cfg["do_repr"] and int(cfg["repr_rollout_k"]) != 0:
        raise ValueError(
            f"Synthetic data_source={cfg['data_source']} requires repr_rollout_k=0."
        )


def resolve_config(cfg: dict[str, Any], explicit_keys: set[str]) -> dict[str, Any]:
    """Resolve defaults and validate one evaluation run configuration."""
    cfg = dict(cfg)

    # Preserve backward compatibility with configs written before the rename.
    legacy_eval_langs = cfg.pop("eval_langs", None)
    if "lid_candidate_langs" not in cfg:
        cfg["lid_candidate_langs"] = legacy_eval_langs or []
    cfg.pop("decoding_targets_path", None)

    # Select exactly one evaluation pipeline.
    if cfg["do_decoding"] and cfg["do_repr"]:
        raise ValueError("You can only run do_decoding or do_repr at a time, not both.")
    if not cfg["do_decoding"] and not cfg["do_repr"]:
        raise ValueError("At least one of do_decoding or do_repr must be true.")

    # Resolve data-source-derived setup and prompt paths.
    if is_pud_data_source(cfg["data_source"]) and cfg["data_source"] != "pud":
        if cfg.get("repr_gmm_setup") and cfg["repr_gmm_setup"] != cfg["data_source"]:
            raise ValueError(
                f"data_source={cfg['data_source']} requires matching repr_gmm_setup; "
                f"got {cfg['repr_gmm_setup']!r}."
            )
        cfg["repr_gmm_setup"] = cfg["data_source"]
        if cfg["data_source"].endswith("_ud6"):
            cfg["include_nonparallel_ud"] = True

    if is_include_data_source(cfg["data_source"]):
        registry = load_gmm_setup_registry(cfg["gmm_setup_registry"])
        requested_lid_langs = set(cfg.get("lid_candidate_langs") or [])
        if requested_lid_langs and "en" not in requested_lid_langs:
            raise ValueError("INCLUDE lid_candidate_langs must include English ('en').")
        if not requested_lid_langs:
            cfg["lid_candidate_langs"] = registry[INCLUDE_GMM_SETUP]["languages"]
        if cfg["do_repr"]:
            if cfg.get("repr_gmm_setup") and cfg["repr_gmm_setup"] != INCLUDE_GMM_SETUP:
                raise ValueError(
                    f"INCLUDE representation requires repr_gmm_setup={INCLUDE_GMM_SETUP!r}; "
                    f"got {cfg['repr_gmm_setup']!r}."
                )
            cfg["repr_gmm_setup"] = INCLUDE_GMM_SETUP

    if is_pud_data_source(cfg["data_source"]) and not cfg["prompts_path"] and cfg["pud_root"]:
        cfg["prompts_path"] = str(resolve_pud_prompts_path(cfg["pud_root"], cfg["pud_split_mode"]))
    if is_include_data_source(cfg["data_source"]) and not cfg["prompts_path"]:
        cfg["prompts_path"] = str(
            resolve_include_prompts_path(
                cfg["include_root"],
                cfg["data_source"],
                cfg["include_prompt_style"],
            )
        )

    # Validate shared settings.
    if not cfg["model_name"]:
        raise ValueError("model_name is required.")
    if (is_pud_data_source(cfg["data_source"]) or is_include_data_source(cfg["data_source"])) and not cfg["prompts_path"]:
        raise ValueError(f"prompts_path is required when data_source={cfg['data_source']}.")
    if not is_pud_data_source(cfg["data_source"]):
        if cfg.get("extra_prompts_paths"):
            raise ValueError("extra_prompts_paths are only supported for PUD data sources.")
        if cfg.get("include_nonparallel_ud"):
            raise ValueError("include_nonparallel_ud is only supported for PUD data sources.")

    # Validate decoding-specific settings.
    if (
        cfg["do_decoding"]
        and cfg["decoding_lens"] == "tuned_lens"
        and not cfg.get("tuned_lens_dir")
    ):
        cfg["tuned_lens_dir"] = str(
            build_tuned_lens_base_dir(
                cfg["model_name"],
                "{rev}",
                root=cfg["tuned_lens_root"],
            )
        )
    if cfg["decoding_lens"] not in {"raw_logitlens", "tuned_lens"}:
        raise NotImplementedError(f"Unsupported decoding_lens: {cfg['decoding_lens']}")
    if cfg["decoding_lid_backend"] not in {"langid", "glotlid"}:
        raise ValueError(
            f"Unsupported decoding_lid_backend: {cfg['decoding_lid_backend']}. "
            "Must be 'langid' or 'glotlid'."
        )
    if (
        cfg["do_decoding"]
        and cfg["decoding_decode_mode"] in {"rollout", "rollout_argmax", "rollout_sample"}
        and cfg["decoding_token_agg"] == "last_token"
        and int(cfg["decoding_rollout_word_cnt"]) > 0
    ):
        raise ValueError(
            "decoding_token_agg=last_token cannot be combined with decoding_rollout_word_cnt > 0 "
            "for rollout_argmax/rollout_sample."
        )

    _apply_repr_defaults(cfg)
    _apply_synthetic_eval_policy(cfg, explicit_keys)
    if (
        cfg["do_decoding"]
        and cfg["decoding_mapping"] == "target_string"
        and not is_synthetic_data_source(cfg["data_source"])
    ):
        raise ValueError(
            "decoding_mapping=target_string requires a synthetic data source "
            "(copy, cloze, translation, or translation_to_<lang>)."
        )
    _apply_dataset_runtime_params(cfg, explicit_keys)

    repr_rollout_k = int(cfg["repr_rollout_k"])
    if repr_rollout_k < 0:
        raise ValueError("repr_rollout_k must be non-negative.")
    if cfg["do_repr"] and repr_rollout_k > 0:
        if cfg["repr_token_agg"] == "all_tokens":
            raise ValueError(
                "repr_rollout_k cannot be combined with repr_token_agg=all_tokens. "
                "Use last_token/frac_50/frac_75 to define the rollout anchor."
            )

    # Normalize values consumed by the evaluation runner.
    if cfg["max_prompts"] <= 0:
        cfg["max_prompts"] = None
    if cfg["max_prompts_per_lang"] <= 0:
        cfg["max_prompts_per_lang"] = None

    cfg["extra_prompts_paths"] = list(cfg.get("extra_prompts_paths") or [])
    cfg["gmm_setup_registry"] = str(cfg["gmm_setup_registry"])
    if cfg.get("include_nonparallel_ud"):
        cfg["extra_prompts_paths"].append(str(resolve_ud_prompts_path(cfg["ud_root"], cfg["pud_split_mode"])))
    cfg["extra_prompts_paths"] = list(dict.fromkeys(cfg["extra_prompts_paths"]))
    cfg["lid_candidate_langs"] = sorted(
        set(cfg.get("lid_candidate_langs") or [])
    )

    if not cfg["exp_id"]:
        now = time.time()
        millis = int((now - int(now)) * 1000)
        suffix = uuid.uuid4().hex[:4]
        cfg["exp_id"] = (
            time.strftime("exp-%Y%m%d-%H%M%S", time.localtime(now))
            + f"-{millis:03d}-{suffix}"
        )

    return cfg
