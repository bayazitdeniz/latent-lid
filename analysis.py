"""Load evaluation artifacts and build analysis tables."""

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

PROB_EPS = 1e-12

SYNTHETIC_MODEL_NAME_ORDER = [
    "gpt2",
    "gpt2-xl",
    "Llama-2-7b-hf",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "swiss-ai/Apertus-8B-2509",
    "swiss-ai/Apertus-8B-Instruct-2509",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "CohereLabs/aya-23-8B",
    "utter-project/EuroLLM-9B",
    "utter-project/EuroLLM-9B-Instruct",
]

MULTILINGUAL_PERFORMANCE_ASCENDING_MODEL_ORDER = [
    "GPT-2 XL",
    "GPT-2",
    "Llama-2-7B",
    "OLMo-2-1124-7B",
    "Aya-23-8B",
    "EuroLLM-9B",
    "EuroLLM-9B-Instruct",
    "Llama-3.1-8B",
    "Mistral-Nemo-Instruct",
    "Llama-3.1-8B-Instruct",
    "Apertus-8B",
    "Apertus-8B-Instruct",
]

# OLMo-2 is not part of the original 11-model synthetic suite. Keep this
# compatibility subset ordered by the same multilingual-performance ranking.
SYNTHETIC_MODEL_DISPLAY_ORDER = [
    model
    for model in MULTILINGUAL_PERFORMANCE_ASCENDING_MODEL_ORDER
    if model != "OLMo-2-1124-7B"
]

MAIN_PAPER_SYNTHETIC_MODELS = ["Llama-2-7B", "Aya-23-8B", "Apertus-8B"]
SYNTHETIC_ANCHOR_LANGS = ["ar", "hi", "zh", "ru", "en", "fr"]
OPENENDED_METHOD_ORDER = ["repr", "raw-rmax", "raw-rtopp"]
OPENENDED_METHOD_ORDER_WITH_TUNED = [
    "repr",
    "raw-rmax",
    "raw-rtopp",
    "tuned-rmax",
    "tuned-rtopp",
]
DOMAIN_COMPARISON_DATA_SOURCES = [
    "include_10lang_3domain_cap30",
    "pud9",
    "pud9_ud6",
    "pud21",
    "pud21_ud6",
]
INCLUDE_DOMAIN_DATA_SOURCE = "include_10lang_3domain_cap30"
DOMAIN_CATEGORY_LABELS = {
    "arts_humanities": "Arts/Humanities",
    "social_science": "Social Sciences",
    "social_sciences": "Social Sciences",
    "stem": "STEM",
    "pud9": "PUD9",
    "pud9_ud6": "PUD9 + UD6",
    "pud21": "PUD21",
    "pud21_ud6": "PUD21 + UD6",
}

BASE_INSTRUCT_MODEL_PAIRS = {
    "EuroLLM": {
        "base": "EuroLLM-9B",
        "instruct": "EuroLLM-9B-Instruct",
    },
    "Llama-3.1": {
        "base": "Llama-3.1-8B",
        "instruct": "Llama-3.1-8B-Instruct",
    },
    "Apertus": {
        "base": "Apertus-8B",
        "instruct": "Apertus-8B-Instruct",
    },
}

BASE_INSTRUCT_FAMILIES = list(BASE_INSTRUCT_MODEL_PAIRS)
BASE_INSTRUCT_VARIANT_LABELS = {
    "base": "Base",
    "instruct": "Instruct",
}
BASE_INSTRUCT_DATA_SOURCES = [
    "pud9",
    "pud21",
    INCLUDE_DOMAIN_DATA_SOURCE,
]

_SYNTHETIC_MODEL_DISPLAY_NAMES = {
    "gpt2": "GPT-2",
    "gpt2-xl": "GPT-2 XL",
    "meta-llama/Llama-3.1-8B": "Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "swiss-ai/Apertus-8B-2509": "Apertus-8B",
    "swiss-ai/Apertus-8B-Instruct-2509": "Apertus-8B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407": "Mistral-Nemo-Instruct",
    "CohereLabs/aya-23-8B": "Aya-23-8B",
    "utter-project/EuroLLM-9B": "EuroLLM-9B",
    "utter-project/EuroLLM-9B-Instruct": "EuroLLM-9B-Instruct",
}


################################################################################
# Shared labels and table coordinates
################################################################################

def synthetic_model_display_name(model_name):
    """Return the paper-facing name for a synthetic-suite model."""
    if not isinstance(model_name, str):
        return model_name
    if "Llama-2-7b-hf" in model_name:
        return "Llama-2-7B"
    return _SYNTHETIC_MODEL_DISPLAY_NAMES.get(model_name, model_name.rsplit("/", 1)[-1])


def translation_target_from_data_source(data_source):
    if not isinstance(data_source, str) or not data_source.startswith("translation_to_"):
        return None
    return data_source.removeprefix("translation_to_")


def add_normalized_layers(
    df,
    *,
    group_col="display_model_name",
    layer_col="layer",
    num_bins=20,
    out_norm_col="layer_norm",
    out_bin_col="layer_bin",
):
    """Add normalized layer coordinates for cross-model plots."""
    if df.empty or layer_col not in df.columns:
        return df.copy()
    out = df.copy()
    if group_col in out.columns:
        max_layer = out.groupby(group_col, dropna=False)[layer_col].transform("max")
    else:
        max_layer = out[layer_col].max()
    max_layer = pd.Series(max_layer, index=out.index).replace(0, 1)
    out[out_norm_col] = out[layer_col] / max_layer
    out[out_bin_col] = np.clip(
        np.round(out[out_norm_col] * (num_bins - 1)),
        0,
        num_bins - 1,
    ).astype(int)
    return out


################################################################################
# Run discovery and artifact loading
################################################################################

def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _canonicalize_path(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value


def _display_model_name(model_name: Any) -> Any:
    if not isinstance(model_name, str):
        return model_name
    return model_name.rsplit("/", 1)[-1]
    try:
        return str(Path(value).resolve())
    except OSError:
        return value


def _load_prompt_index(prompts_path: str | Path) -> pd.DataFrame:
    prompts_path = Path(prompts_path)
    if not prompts_path.exists():
        return pd.DataFrame(columns=["prompt_idx", "prompt_id_from_source", "prompt_lang_from_source"])
    rows = []
    with prompts_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt_id = obj.get("id")
            prompt_lang = None
            if isinstance(prompt_id, str):
                if "_" in prompt_id:
                    prompt_lang = prompt_id.split("_", 1)[0]
                elif "-" in prompt_id:
                    prompt_lang = prompt_id.split("-", 1)[0]
                else:
                    prompt_lang = prompt_id
            rows.append(
                {
                    "prompt_idx": idx,
                    "prompt_id_from_source": prompt_id,
                    "prompt_lang_from_source": prompt_lang,
                }
            )
    return pd.DataFrame(rows)

def _backfill_prompt_metadata(frame: pd.DataFrame, prompts_path: str | Path) -> pd.DataFrame:
    if "prompt_idx" not in frame.columns:
        return frame
    prompt_index = _load_prompt_index(prompts_path)
    if prompt_index.empty:
        return frame

    merged = frame.merge(prompt_index, on="prompt_idx", how="left")
    if "prompt_id" in merged.columns:
        merged["prompt_id"] = merged["prompt_id"].where(merged["prompt_id"].notna(), merged["prompt_id_from_source"])
    else:
        merged["prompt_id"] = merged["prompt_id_from_source"]
    if "prompt_lang" in merged.columns:
        merged["prompt_lang"] = merged["prompt_lang"].where(
            merged["prompt_lang"].notna(), merged["prompt_lang_from_source"]
        )
    else:
        merged["prompt_lang"] = merged["prompt_lang_from_source"]
    return merged.drop(columns=["prompt_id_from_source", "prompt_lang_from_source"])


def _get_nested(config: Mapping[str, Any], key: str) -> Any:
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _normalize_filter_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _config_matches(config: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = _get_nested(config, key)
        if callable(expected):
            if not expected(actual):
                return False
            continue
        if _normalize_filter_value(actual) != _normalize_filter_value(expected):
            return False
    return True


def discover_eval_runs(
    log_root: str | Path = "logs/evals",
    config_filters: Mapping[str, Any] | None = None,
    require_prompt_metrics: bool = True,
) -> pd.DataFrame:
    """Enumerate eval runs and expose config fields for filtering/selection."""
    log_root = Path(log_root)
    rows = []
    exp_dirs = sorted(log_root.iterdir()) if log_root.exists() else []
    display_root = log_root
    try:
        display_root = log_root.resolve().relative_to(Path(__file__).resolve().parent)
    except ValueError:
        pass
    print(f"Discovering eval runs under {display_root} ({len(exp_dirs):,} directories)...", flush=True)
    for exp_dir in tqdm(exp_dirs, desc="Discovering eval runs", unit="run"):
        if not exp_dir.is_dir():
            continue
        config_path = exp_dir / "config.json"
        metrics_path = exp_dir / "metrics.parquet"
        prompt_metrics_path = exp_dir / "prompt_metrics.parquet"
        if not config_path.exists() or not metrics_path.exists():
            continue
        if require_prompt_metrics and not prompt_metrics_path.exists():
            continue
        config = _read_json(config_path)
        if config_filters and not _config_matches(config, config_filters):
            continue
        run_files = [config_path, metrics_path]
        if prompt_metrics_path.exists():
            run_files.append(prompt_metrics_path)
        run_mtime_ns = max(path.stat().st_mtime_ns for path in run_files)
        rows.append(
            {
                "exp_id": exp_dir.name,
                "exp_path": str(exp_dir),
                "run_mtime_ns": int(run_mtime_ns),
                "run_mtime": run_mtime_ns / 1_000_000_000,
                "config_path": str(config_path),
                "metrics_path": str(metrics_path),
                "prompt_metrics_path": str(prompt_metrics_path) if prompt_metrics_path.exists() else None,
                "model_name": config.get("model_name"),
                "revision": config.get("revision"),
                "prompts_path": config.get("prompts_path"),
                "do_decoding": config.get("do_decoding"),
                "do_repr": config.get("do_repr"),
                "decoding_mapping": config.get("decoding_mapping"),
                "decoding_lens": config.get("decoding_lens"),
                "repr_priors": config.get("repr_priors"),
                "repr_pca": config.get("repr_pca"),
                "repr_cov": config.get("repr_cov"),
                "repr_unit": config.get("repr_unit"),
                "repr_token_agg": config.get("repr_token_agg"),
                "config": config,
            }
        )
    return pd.DataFrame(rows)


def select_latest_unique_runs(
    manifest: pd.DataFrame,
    *,
    extra_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Keep exactly one run per analysis mode bucket, preferring the most recently
    written completed run when filesystem timestamps are available.

    Buckets are defined by model/checkpoint/method-relevant config so repeated reruns
    do not contaminate downstream summaries.
    """
    if manifest.empty:
        return manifest.copy()

    extra_keys = list(extra_keys or [])
    df = manifest.copy()
    if "prompts_path" in df.columns:
        df["prompts_path"] = df["prompts_path"].map(_canonicalize_path)
    df["mode_family"] = np.where(df["do_repr"].fillna(False), "repr_gmm", "decoding")
    group_keys = [
        "model_name",
        "revision",
        "prompts_path",
        "mode_family",
        "decoding_mapping",
        "decoding_lens",
        "repr_priors",
        "repr_pca",
        "repr_cov",
        "repr_unit",
        "repr_token_agg",
        *extra_keys,
    ]
    group_keys = [key for key in group_keys if key in df.columns]
    sort_cols = [col for col in ["run_mtime_ns", "exp_id"] if col in df.columns]
    df = df.sort_values(sort_cols)
    latest = df.groupby(group_keys, dropna=False, as_index=False).tail(1)
    out_sort_cols = [col for col in ["model_name", "mode_family", "run_mtime_ns", "exp_id"] if col in latest.columns]
    return latest.sort_values(out_sort_cols).reset_index(drop=True)


def load_selected_runs(
    log_root: str | Path = "logs/evals",
    *,
    config_filters: Mapping[str, Any] | None = None,
    table: str = "metrics",
    require_prompt_metrics: bool | None = None,
    exp_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load metrics from matching runs and append config metadata columns."""
    if table not in {"metrics", "prompt_metrics"}:
        raise ValueError("table must be 'metrics' or 'prompt_metrics'.")
    if require_prompt_metrics is None:
        require_prompt_metrics = True

    manifest = discover_eval_runs(
        log_root=log_root,
        config_filters=config_filters,
        require_prompt_metrics=require_prompt_metrics,
    )
    if exp_ids is not None:
        manifest = manifest[manifest["exp_id"].isin(list(exp_ids))].copy()
    if manifest.empty:
        return pd.DataFrame()

    frames = []
    path_col = "metrics_path" if table == "metrics" else "prompt_metrics_path"
    for row in manifest.itertuples(index=False):
        table_path = getattr(row, path_col)
        if not table_path:
            continue
        frame = pd.read_parquet(table_path)
        frame["exp_id"] = row.exp_id
        frame["exp_path"] = row.exp_path
        frame["model_name"] = row.model_name
        frame["display_model_name"] = _display_model_name(row.model_name)
        frame["run_revision"] = row.revision
        frame["prompts_path"] = row.prompts_path
        frame["config_decoding_mapping"] = row.decoding_mapping
        frame["config_decoding_lens"] = getattr(row, "decoding_lens", None)
        frame["config_repr_priors"] = getattr(row, "repr_priors", None)
        frame["config_repr_pca"] = getattr(row, "repr_pca", None)
        frame["config_repr_cov"] = getattr(row, "repr_cov", None)
        frame["config_repr_unit"] = getattr(row, "repr_unit", None)
        frame["config_repr_token_agg"] = getattr(row, "repr_token_agg", None)
        if table == "prompt_metrics" and row.prompts_path:
            frame = _backfill_prompt_metadata(frame, row.prompts_path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_lang_probs_artifact(path: str | Path) -> dict:
    """Load a saved language-distribution artifact on CPU."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)


################################################################################
# Synthetic target-string analysis
################################################################################


def load_synthetic_target_string_prompt_metrics(manifest):
    """Load prompt_metrics.parquet for selected synthetic target-string runs."""
    if manifest.empty:
        return pd.DataFrame()
    frames = []
    print(f"Loading prompt metrics from {len(manifest):,} runs...", flush=True)
    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Loading prompt metrics",
        unit="run",
    ):
        prompt_metrics_path = getattr(row, "prompt_metrics_path", None)
        if not prompt_metrics_path:
            continue
        frame = pd.read_parquet(prompt_metrics_path)
        frame["exp_id"] = row.exp_id
        frame["exp_path"] = row.exp_path
        frame["model_name"] = row.model_name
        frame["display_model_name"] = row.display_model_name
        frame["run_revision"] = row.revision
        frame["data_source"] = row.data_source
        frame["translation_target_lang"] = row.translation_target_lang
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "method" in out.columns:
        out = out[out["method"] == "decoding"].copy()
    return out.reset_index(drop=True)


def _artifact_records(artifact):
    if isinstance(artifact, dict) and "records" in artifact:
        return artifact.get("records") or [], artifact.get("layer_indices")
    if isinstance(artifact, list):
        return artifact, None
    raise TypeError(f"Unsupported target-string artifact type: {type(artifact)!r}")


def _selected_languages(group_langs, target_langs):
    langs = list(group_langs or [])
    if target_langs is None:
        return langs
    target_langs = set(target_langs)
    return [lang for lang in langs if lang in target_langs]


def _target_string_menu_needs(include_values):
    """Return which target-string score families are needed for the requested columns."""
    return {
        "mean_prob": bool({"menu_mean_prob", "menu_mean_prob_share"} & include_values),
        "menu_prob": bool({"menu_prob", "menu_prob_share"} & include_values),
        "first_token_logprob": bool({"menu_first_token_logprob", "menu_first_token_prob"} & include_values),
        "first_token_prob": bool({"menu_first_token_prob", "menu_first_token_prob_share"} & include_values),
        "start_token_logprob": bool({"menu_start_token_logprob", "menu_start_token_prob"} & include_values),
        "start_token_prob": bool({"menu_start_token_prob", "menu_start_token_prob_share"} & include_values),
    }


def _finite_exp(logprobs):
    """Exponentiate finite log-probabilities and map non-finite entries to zero."""
    probs = torch.exp(logprobs)
    return torch.where(torch.isfinite(logprobs), probs, torch.zeros_like(probs))


def _target_string_score_tensors(rec, needs):
    """Load and validate score tensors for one target-string prompt record."""
    sum_logprobs = rec.get("menu_teacher_forced_sum_logprobs", rec.get("menu_sum_logprobs"))
    start_token_logprobs = rec.get("menu_start_token_logprobs")
    if sum_logprobs is None and start_token_logprobs is None:
        return None

    score_shape = None
    if sum_logprobs is not None:
        sum_logprobs = torch.as_tensor(sum_logprobs).detach().cpu().double()
        if sum_logprobs.ndim != 2:
            return None
        score_shape = sum_logprobs.shape

    if start_token_logprobs is not None:
        start_token_logprobs = torch.as_tensor(start_token_logprobs).detach().cpu().double()
        if start_token_logprobs.ndim != 2:
            return None
        if score_shape is None:
            score_shape = start_token_logprobs.shape

    if score_shape is None:
        return None

    token_counts = None
    if needs["mean_prob"] and sum_logprobs is not None:
        token_counts = torch.as_tensor(
            rec.get("menu_teacher_forced_token_counts", rec.get("menu_token_counts")),
            dtype=torch.float64,
        )
        if token_counts.numel() != score_shape[1]:
            return None

    menu_probs = None
    if needs["menu_prob"]:
        menu_probs = rec.get("menu_teacher_forced_probs", rec.get("menu_probs"))
        if menu_probs is not None:
            menu_probs = torch.as_tensor(menu_probs).detach().cpu().double()

    first_token_logprobs = rec.get(
        "menu_teacher_forced_first_token_logprobs",
        rec.get("menu_first_token_logprobs"),
    )
    first_token_probs = None
    if first_token_logprobs is not None:
        first_token_logprobs = torch.as_tensor(first_token_logprobs).detach().cpu().double()
        if needs["first_token_prob"]:
            first_token_probs = _finite_exp(first_token_logprobs)
    else:
        first_token_probs = rec.get(
            "menu_teacher_forced_first_token_probs",
            rec.get("menu_first_token_probs"),
        )
        if first_token_probs is not None:
            first_token_probs = torch.as_tensor(first_token_probs).detach().cpu().double()

    if start_token_logprobs is not None and needs["start_token_prob"]:
        start_token_probs = _finite_exp(start_token_logprobs)
    else:
        start_token_probs = rec.get("menu_start_token_probs")
        if start_token_probs is not None:
            start_token_probs = torch.as_tensor(start_token_probs).detach().cpu().double()

    return {
        "score_shape": score_shape,
        "sum_logprobs": sum_logprobs,
        "token_counts": token_counts,
        "menu_probs": menu_probs,
        "first_token_logprobs": first_token_logprobs,
        "first_token_probs": first_token_probs,
        "start_token_logprobs": start_token_logprobs,
        "start_token_probs": start_token_probs,
    }


def _target_string_base_menu_row(run, rec, group, prompt_lang, layer, token_count, lang_count):
    """Build the shared dataframe fields for one run/prompt/menu/layer row."""
    return {
        "exp_id": run.exp_id,
        "model_name": run.model_name,
        "display_model_name": run.display_model_name,
        "revision": run.revision,
        "data_source": run.data_source,
        "translation_target_lang": run.translation_target_lang,
        "prompt_id": rec.get("prompt_id"),
        "prompt_lang": prompt_lang,
        "concept_id": rec.get("concept_id"),
        "true_tgt_lang": rec.get("tgt_lang"),
        "layer": int(layer),
        "tgt_lang": None,
        "tgt_text": group.get("text"),
        "menu_token_count": token_count,
        "menu_group_lang_count": lang_count,
    }


def _append_target_string_score_columns(base, scores, needs, layer_pos, group_idx, token_count, lang_count):
    """Attach requested target-string score columns to a base row in place."""
    sum_logprobs = scores["sum_logprobs"]
    token_counts = scores["token_counts"]
    menu_probs = scores["menu_probs"]
    first_token_logprobs = scores["first_token_logprobs"]
    first_token_probs = scores["first_token_probs"]
    start_token_logprobs = scores["start_token_logprobs"]
    start_token_probs = scores["start_token_probs"]

    if needs["mean_prob"] and sum_logprobs is not None and token_counts is not None:
        mean_logprob = sum_logprobs[:, group_idx] / max(token_count, 1.0)
        mean_prob = torch.exp(mean_logprob)
        base["menu_mean_logprob"] = float(mean_logprob[layer_pos])
        base["menu_mean_prob"] = float(mean_prob[layer_pos])
        base["menu_mean_prob_share"] = float(mean_prob[layer_pos]) / lang_count

    if needs["menu_prob"]:
        prob = (
            menu_probs[:, group_idx]
            if menu_probs is not None
            else torch.exp(sum_logprobs[:, group_idx])
            if sum_logprobs is not None
            else None
        )
        if prob is not None:
            base["menu_prob"] = float(prob[layer_pos])
            base["menu_prob_share"] = float(prob[layer_pos]) / lang_count

    if first_token_logprobs is not None and needs["first_token_logprob"]:
        base["menu_first_token_logprob"] = float(first_token_logprobs[layer_pos, group_idx])
    if first_token_probs is not None and needs["first_token_prob"]:
        base["menu_first_token_prob"] = float(first_token_probs[layer_pos, group_idx])
        base["menu_first_token_prob_share"] = float(first_token_probs[layer_pos, group_idx]) / lang_count
    if start_token_logprobs is not None and needs["start_token_logprob"]:
        base["menu_start_token_logprob"] = float(start_token_logprobs[layer_pos, group_idx])
    if start_token_probs is not None and needs["start_token_prob"]:
        base["menu_start_token_prob"] = float(start_token_probs[layer_pos, group_idx])
        base["menu_start_token_prob_share"] = float(start_token_probs[layer_pos, group_idx]) / lang_count


def _target_string_record_menu_rows(run, rec, layer_indices, target_langs, needs):
    """Expand one target-string prompt record into long-form dataframe rows."""
    grouped = rec.get("menu_strings_grouped") or []
    scores = _target_string_score_tensors(rec, needs)
    if scores is None:
        return []

    layers = layer_indices if layer_indices is not None else range(scores["score_shape"][0])
    rows = []
    for group_idx, group in enumerate(grouped):
        langs = _selected_languages(group.get("langs"), target_langs)
        if not langs:
            continue
        lang_count = max(len(group.get("langs") or []), 1)
        token_counts = scores["token_counts"]
        token_count = float(token_counts[group_idx]) if token_counts is not None else np.nan

        for layer_pos, layer in enumerate(layers):
            base = _target_string_base_menu_row(
                run,
                rec,
                group,
                rec.get("prompt_lang"),
                layer,
                token_count,
                lang_count,
            )
            _append_target_string_score_columns(
                base,
                scores,
                needs,
                layer_pos,
                group_idx,
                token_count,
                lang_count,
            )
            for lang in langs:
                row = dict(base)
                row["tgt_lang"] = lang
                rows.append(row)
    return rows


def _ensure_target_string_menu_columns(out, include_values):
    """Add requested score columns as NaN when no loaded artifact supplied them."""
    for col in [
        "menu_mean_logprob",
        "menu_mean_prob",
        "menu_mean_prob_share",
        "menu_prob",
        "menu_prob_share",
        "menu_first_token_logprob",
        "menu_first_token_prob",
        "menu_first_token_prob_share",
        "menu_start_token_logprob",
        "menu_start_token_prob",
        "menu_start_token_prob_share",
    ]:
        if col in include_values and col not in out.columns:
            out[col] = np.nan
    return out


def load_target_string_menu_dataframe(
    manifest,
    *,
    data_sources=None,
    models=None,
    prompt_langs=None,
    target_langs=None,
    include_values=(
        "menu_mean_prob_share",
        "menu_mean_prob",
        "menu_prob_share",
        "menu_prob",
        "menu_first_token_prob_share",
        "menu_first_token_prob",
        "menu_first_token_logprob",
        "menu_start_token_prob_share",
        "menu_start_token_prob",
        "menu_start_token_logprob",
    ),
):
    """
    Expand target-string menu artifacts into a long dataframe.

    Probability-share columns divide each menu string's probability equally across
    all language labels attached to that string, matching the paper plotting plan.
    """
    if manifest.empty:
        return pd.DataFrame()

    data_sources = set(data_sources) if data_sources is not None else None
    models = set(models) if models is not None else None
    prompt_langs = set(prompt_langs) if prompt_langs is not None else None
    target_langs = set(target_langs) if target_langs is not None else None
    include_values = set(include_values)

    selected = manifest.copy()
    if data_sources is not None:
        selected = selected[selected["data_source"].isin(data_sources)]
    if models is not None:
        selected = selected[selected["display_model_name"].isin(models)]
    print(f"Expanding target-string menu artifacts from {len(selected):,} runs...", flush=True)

    needs = _target_string_menu_needs(include_values)
    rows = []
    for run in tqdm(
        selected.itertuples(index=False),
        total=len(selected),
        desc="Loading target-string artifacts",
        unit="run",
    ):
        print(f"[target-string] loading {run.exp_id}", flush=True)
        artifact_path = os.path.join(run.exp_path, f"target_string_details_{run.revision}.pt")
        if not os.path.exists(artifact_path):
            print(f"[target-string] missing artifact for {run.exp_id}: {artifact_path}", flush=True)
            continue
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        records, layer_indices = _artifact_records(artifact)
        print(f"[target-string] expanding {run.exp_id}: {len(records):,} prompts", flush=True)

        for rec in tqdm(
            records,
            total=len(records),
            desc=f"Expanding {run.exp_id}",
            unit="prompt",
            leave=False,
        ):
            prompt_lang = rec.get("prompt_lang")
            if prompt_langs is not None and prompt_lang not in prompt_langs:
                continue
            rows.extend(_target_string_record_menu_rows(run, rec, layer_indices, target_langs, needs))

    out = pd.DataFrame(rows)
    return _ensure_target_string_menu_columns(out, include_values)


def summarize_target_string_menu_entries(
    df,
    *,
    group_cols=("display_model_name", "prompt_lang", "tgt_lang"),
    value_col=None,
    entry_cols=(
        "data_source",
        "display_model_name",
        "prompt_lang",
        "tgt_lang",
        "prompt_id",
        "concept_id",
        "tgt_text",
    ),
):
    """Count unique target-string menu entries for a filtered plotting dataframe."""
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "entries"])
    frame = df.copy()
    if value_col is not None and value_col in frame.columns:
        frame = frame[frame[value_col].notna()].copy()
    present_entry_cols = [col for col in entry_cols if col in frame.columns]
    present_group_cols = [col for col in group_cols if col in frame.columns]
    if present_entry_cols:
        frame = frame.drop_duplicates(present_entry_cols)
    if not present_group_cols:
        return pd.DataFrame({"entries": [len(frame)]})
    return (
        frame.groupby(present_group_cols, dropna=False)
        .size()
        .rename("entries")
        .reset_index()
        .sort_values(present_group_cols)
        .reset_index(drop=True)
    )


################################################################################
# Open-ended analysis
################################################################################

def openended_method_label_from_config(config):
    """Return the paper-facing estimator label for an open-ended run."""
    if not isinstance(config, dict):
        return None
    if config.get("do_repr") is True:
        return "repr"
    if config.get("do_decoding") is True:
        decode_mode = config.get("decoding_decode_mode")
        lens = config.get("decoding_lens")
        if lens == "raw_logitlens":
            lens_prefix = "raw"
        elif lens == "tuned_lens":
            lens_prefix = "tuned"
        else:
            return None
        if decode_mode == "rollout_argmax":
            return f"{lens_prefix}-rmax"
        if decode_mode == "rollout_sample":
            return f"{lens_prefix}-rtopp"
    return None


def extract_include_domain(prompt_id):
    if not isinstance(prompt_id, str) or "-" not in prompt_id:
        return None
    parts = prompt_id.split("-")
    if len(parts) < 4:
        return None
    return "-".join(parts[2:-1])


def load_openended_run_manifest(
    log_root="logs/evals",
    *,
    data_sources=None,
    methods=None,
    require_prompt_metrics=True,
):
    """Discover the latest PUD/INCLUDE runs with paper-facing labels."""
    data_sources = list(data_sources or DOMAIN_COMPARISON_DATA_SOURCES)
    methods = list(methods or OPENENDED_METHOD_ORDER)

    manifest = discover_eval_runs(log_root=log_root, require_prompt_metrics=require_prompt_metrics)
    if manifest.empty:
        return manifest
    manifest = manifest.copy()
    manifest["data_source"] = manifest["config"].map(lambda c: c.get("data_source") if isinstance(c, dict) else None)
    manifest["method_label"] = manifest["config"].map(openended_method_label_from_config)
    manifest["decoding_decode_mode"] = manifest["config"].map(
        lambda c: c.get("decoding_decode_mode") if isinstance(c, dict) else None
    )
    manifest["include_prompt_style"] = manifest["config"].map(
        lambda c: c.get("include_prompt_style") if isinstance(c, dict) else None
    )
    manifest["pud_split_mode"] = manifest["config"].map(
        lambda c: c.get("pud_split_mode") if isinstance(c, dict) else None
    )
    manifest = manifest[
        manifest["data_source"].isin(data_sources)
        & manifest["method_label"].isin(methods)
    ].copy()
    if manifest.empty:
        return manifest.reset_index(drop=True)

    manifest["display_model_name"] = manifest["model_name"].map(synthetic_model_display_name)
    manifest = select_latest_unique_runs(
        manifest,
        extra_keys=["data_source", "method_label", "decoding_decode_mode", "include_prompt_style", "pud_split_mode"],
    )
    manifest = (
        manifest.sort_values(["data_source", "display_model_name", "method_label", "exp_id"])
        .groupby(["data_source", "display_model_name", "method_label"], dropna=False, as_index=False)
        .tail(1)
    )
    model_order = {name: idx for idx, name in enumerate(SYNTHETIC_MODEL_DISPLAY_ORDER)}
    method_order = {name: idx for idx, name in enumerate(methods)}
    source_order = {name: idx for idx, name in enumerate(data_sources)}
    manifest["display_model_sort"] = manifest["display_model_name"].map(model_order).fillna(len(model_order)).astype(int)
    manifest["method_sort"] = manifest["method_label"].map(method_order).fillna(len(method_order)).astype(int)
    manifest["data_source_sort"] = manifest["data_source"].map(source_order).fillna(len(source_order)).astype(int)
    return manifest.sort_values(
        ["data_source_sort", "method_sort", "display_model_sort", "exp_id"]
    ).reset_index(drop=True)


def load_openended_prompt_metrics(
    manifest,
    *,
    columns=None,
    add_domain_columns=True,
    show_progress=False,
):
    """Load prompt metrics from selected PUD/INCLUDE runs.

    ``columns`` projects Parquet columns before materialization. Callers can
    disable domain parsing when the INCLUDE/PUD domain labels are unused.
    """
    if manifest.empty:
        return pd.DataFrame()
    frames = []
    rows = list(manifest.itertuples(index=False))
    iterator = tqdm(
        rows,
        desc="Loading prompt metrics",
        unit="run",
        disable=not show_progress,
    )
    for row in iterator:
        prompt_metrics_path = getattr(row, "prompt_metrics_path", None)
        if not prompt_metrics_path:
            continue
        frame = pd.read_parquet(prompt_metrics_path, columns=columns)
        frame["exp_id"] = row.exp_id
        frame["exp_path"] = row.exp_path
        frame["model_name"] = row.model_name
        frame["display_model_name"] = row.display_model_name
        frame["run_revision"] = row.revision
        frame["data_source"] = row.data_source
        frame["method_label"] = row.method_label
        frame["decoding_decode_mode"] = row.decoding_decode_mode
        frame["include_prompt_style"] = row.include_prompt_style
        frame["pud_split_mode"] = row.pud_split_mode
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if add_domain_columns and "prompt_id" in out.columns:
        out["include_domain"] = out["prompt_id"].map(extract_include_domain)
    if add_domain_columns:
        out["domain_category"] = out.get(
            "include_domain",
            pd.Series(index=out.index, dtype=object),
        )
        pud_domain_sources = [
            source for source in DOMAIN_COMPARISON_DATA_SOURCES
            if source != INCLUDE_DOMAIN_DATA_SOURCE
        ]
        pud_domain_mask = out["data_source"].isin(pud_domain_sources)
        out.loc[pud_domain_mask, "domain_category"] = out.loc[pud_domain_mask, "data_source"]
        out["domain_category_label"] = out["domain_category"].map(DOMAIN_CATEGORY_LABELS)
        out["domain_category_label"] = out["domain_category_label"].fillna(
            out["domain_category"]
        )
    return out.reset_index(drop=True)


def build_pivot_destination_tables(
    prompt_df,
    *,
    data_source=INCLUDE_DOMAIN_DATA_SOURCE,
    method="repr",
    models=None,
    task_lang_col="prompt_lang",
    exclude_english_prompts=True,
):
    """Split prompt-layer pivots into English and non-English destinations."""
    required = {
        "data_source",
        "method_label",
        "display_model_name",
        "layer",
        "dominant_lang",
        "dominance",
        task_lang_col,
    }
    missing = sorted(required - set(prompt_df.columns))
    if missing:
        raise KeyError(f"Missing columns required for pivot-destination analysis: {missing}")

    events = prompt_df[
        prompt_df["data_source"].eq(data_source)
        & prompt_df["method_label"].eq(method)
    ].copy()
    if models is not None:
        events = events[events["display_model_name"].isin(list(models))].copy()
    if exclude_english_prompts:
        events = events[events[task_lang_col].ne("en")].copy()

    events["total_pivot"] = events["dominant_lang"].ne(events[task_lang_col])
    if "pivot" in events.columns and events["pivot"].notna().any():
        stored_pivot = events["pivot"].fillna(False).astype(bool)
        mismatches = stored_pivot.ne(events["total_pivot"])
        if mismatches.any():
            raise ValueError(
                "Stored pivot labels disagree with dominant-language argmaxes for "
                f"{int(mismatches.sum())} prompt-layer rows."
            )

    events["english_pivot"] = events["total_pivot"] & events["dominant_lang"].eq("en")
    events["other_pivot"] = events["total_pivot"] & events["dominant_lang"].ne("en")
    decomposed = events["english_pivot"].astype(int) + events["other_pivot"].astype(int)
    if not decomposed.eq(events["total_pivot"].astype(int)).all():
        raise AssertionError("Pivot destination rates do not decompose exactly.")

    rate_cols = ["english_pivot", "other_pivot", "total_pivot", "dominance"]
    rates = (
        events.groupby(["display_model_name", "method_label", "layer"], dropna=False)[rate_cols]
        .mean()
        .reset_index()
    )
    return events.reset_index(drop=True), rates


################################################################################
# Base-instruct comparisons
################################################################################

def add_base_instruct_model_columns(df):
    """Annotate rows with base-vs-instruct model family and variant labels."""
    out = df.copy()
    display_to_pair = {}
    for family, variants in BASE_INSTRUCT_MODEL_PAIRS.items():
        for variant, display_model_name in variants.items():
            display_to_pair[display_model_name] = (family, variant)

    pairs = out["display_model_name"].map(display_to_pair)
    out["base_instruct_family"] = pairs.map(lambda item: item[0] if isinstance(item, tuple) else None)
    out["base_instruct_variant"] = pairs.map(lambda item: item[1] if isinstance(item, tuple) else None)
    out["base_instruct_variant_label"] = out["base_instruct_variant"].map(
        BASE_INSTRUCT_VARIANT_LABELS
    )
    return out


def summarize_base_instruct_methods(
    prompt_df,
    *,
    data_sources=None,
    families=None,
    methods=("repr", "raw-rtopp"),
    metrics=("pivot", "entropy"),
    agreement_methods=("repr", "raw-rtopp"),
    layer_cols=("layer", "layer_norm"),
):
    """Build a plot-ready base/instruct summary for multiple estimators.

    Method metrics retain the estimator label. Agreement is computed from
    prompt-level dominant-language argmaxes before aggregation.
    """
    if prompt_df.empty:
        return pd.DataFrame()

    df = add_base_instruct_model_columns(prompt_df)
    df = df[df["base_instruct_family"].notna() & df["base_instruct_variant"].notna()].copy()
    if data_sources is not None:
        df = df[df["data_source"].isin(list(data_sources))].copy()
    if families is not None:
        df = df[df["base_instruct_family"].isin(list(families))].copy()
    df = df[df["method_label"].isin(list(methods))].copy()
    if df.empty:
        return pd.DataFrame()

    present_layer_cols = [col for col in layer_cols if col in df.columns]
    if not present_layer_cols:
        raise ValueError("No requested layer columns are present in prompt metrics.")
    group_cols = [
        "data_source",
        "base_instruct_family",
        "base_instruct_variant",
        "base_instruct_variant_label",
        *present_layer_cols,
    ]
    frames = []
    for method_label in methods:
        method_df = df[df["method_label"].eq(method_label)]
        for metric in metrics:
            if method_df.empty or metric not in method_df.columns:
                continue
            summary = (
                method_df.groupby(group_cols, dropna=False)[metric]
                .agg(value="mean", n="count")
                .reset_index()
            )
            summary["method_label"] = method_label
            summary["metric"] = metric
            frames.append(summary)

    agreement_left, agreement_right = agreement_methods
    agreement_df = df[df["method_label"].isin(agreement_methods)].copy()
    if not agreement_df.empty and "dominant_lang" in agreement_df.columns:
        join_keys = [
            "data_source",
            "display_model_name",
            "base_instruct_family",
            "base_instruct_variant",
            "base_instruct_variant_label",
            *present_layer_cols,
            "prompt_idx",
        ]
        if "prompt_id" in agreement_df.columns:
            join_keys.append("prompt_id")
        wide = (
            agreement_df[join_keys + ["method_label", "dominant_lang"]]
            .drop_duplicates()
            .pivot_table(
                index=join_keys,
                columns="method_label",
                values="dominant_lang",
                aggfunc="first",
            )
            .reset_index()
        )
        if agreement_left in wide.columns and agreement_right in wide.columns:
            wide["agreement"] = (wide[agreement_left] == wide[agreement_right]).astype(float)
            summary = (
                wide.groupby(group_cols, dropna=False)["agreement"]
                .agg(value="mean", n="count")
                .reset_index()
            )
            summary["method_label"] = f"agreement_{agreement_left}_vs_{agreement_right}"
            summary["metric"] = "agreement"
            frames.append(summary)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


################################################################################
# Language-distribution summaries
################################################################################

def story_layer_window_mask(
    df: pd.DataFrame,
    *,
    layer_norm_col: str = "layer_norm",
    layer_min: float | None = 0.25,
    layer_max: float | None = 0.75,
    include_min: bool = True,
    include_max: bool = True,
) -> pd.Series:
    """Return a boolean mask for normalized layer-window filtering."""
    if df.empty:
        return pd.Series([], index=df.index, dtype=bool)
    if layer_norm_col not in df.columns:
        raise ValueError(f"Missing normalized layer column {layer_norm_col!r}.")
    mask = pd.Series(True, index=df.index)
    values = df[layer_norm_col]
    if layer_min is not None:
        mask &= values.ge(layer_min) if include_min else values.gt(layer_min)
    if layer_max is not None:
        mask &= values.le(layer_max) if include_max else values.lt(layer_max)
    return mask


def add_story_category_mass_columns(
    category_df: pd.DataFrame,
    *,
    prob_col: str,
    category_col: str = "prob_category",
    lang_count_col: str = "category_lang_count",
    prob_sum_col: str = "category_prob_sum",
) -> pd.DataFrame:
    """Add total category mass from per-language category probabilities."""
    out = category_df.copy()
    if out.empty:
        return out
    if lang_count_col not in out.columns:
        out[lang_count_col] = 1
    if prob_col in out.columns:
        out[prob_sum_col] = out[prob_col] * out[lang_count_col].fillna(1)
    return out


def diagnose_story_distribution_sums(
    category_df: pd.DataFrame,
    *,
    prob_col: str,
    group_cols: Sequence[str] = ("dataset", "method_label", "display_model_name"),
    prompt_col: str = "prompt_id",
    layer_col: str = "layer",
    category_col: str = "prob_category",
    lang_count_col: str = "category_lang_count",
) -> pd.DataFrame:
    """Check whether category probabilities reconstruct a full language distribution."""
    if category_df.empty:
        return pd.DataFrame()
    needed = [*group_cols, prompt_col, layer_col, category_col, prob_col]
    missing = [col for col in needed if col not in category_df.columns]
    if missing:
        raise ValueError(f"Missing columns for distribution diagnostic: {missing}")
    df = add_story_category_mass_columns(
        category_df,
        prob_col=prob_col,
        category_col=category_col,
        lang_count_col=lang_count_col,
    )
    key_cols = [*group_cols, prompt_col, layer_col]
    sums = (
        df.groupby(key_cols, dropna=False)["category_prob_sum"]
        .sum()
        .rename("reconstructed_mass")
        .reset_index()
    )
    return (
        sums.groupby(list(group_cols), dropna=False)["reconstructed_mass"]
        .agg(
            rows="count",
            mean="mean",
            min="min",
            max="max",
            max_abs_err_from_1=lambda s: (s - 1.0).abs().max(),
        )
        .reset_index()
    )


def micro_average_langdist_categories(
    category_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    category_col: str = "prob_category",
    prompt_col: str = "prompt_id",
    layer_norm_col: str = "layer_norm",
    category_count_col: str = "category_lang_count",
    category_sum_col: str = "category_prob_sum",
    category_sq_sum_col: str = "category_prob_sq_sum",
    layer_min: float | None = 0.25,
    layer_max: float | None = 0.75,
    include_min: bool = True,
    include_max: bool = True,
    keep_category_lang_count: bool = False,
) -> pd.DataFrame:
    """Pool prompt-layer-language probabilities before averaging.

    The cached category table stores the count, sum, and squared-value sum of
    the individual language probabilities in each prompt-layer category. These
    sufficient statistics recover the exact pooled mean and standard error
    without expanding the table back to one row per candidate language.
    """
    out_cols = [
        *group_cols,
        category_col,
        "mean_prob",
        "stderr_prob",
        "std_prob",
        "var_prob",
        "n_prompts",
        "n_values",
    ]
    if keep_category_lang_count:
        out_cols.append("mean_category_lang_count")
    if category_df.empty:
        return pd.DataFrame(columns=out_cols)

    required = [
        *group_cols,
        category_col,
        prompt_col,
        layer_norm_col,
        category_count_col,
        category_sum_col,
        category_sq_sum_col,
    ]
    missing = [col for col in required if col not in category_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for micro averaging: {missing}")

    df = category_df[story_layer_window_mask(
        category_df,
        layer_norm_col=layer_norm_col,
        layer_min=layer_min,
        layer_max=layer_max,
        include_min=include_min,
        include_max=include_max,
    )].copy()
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    df["_micro_count"] = pd.to_numeric(df[category_count_col], errors="coerce")
    df["_micro_sum"] = pd.to_numeric(df[category_sum_col], errors="coerce")
    df["_micro_sq_sum"] = pd.to_numeric(df[category_sq_sum_col], errors="coerce")
    valid = (
        df["_micro_count"].gt(0)
        & df["_micro_sum"].notna()
        & df["_micro_sq_sum"].notna()
    )
    df = df[valid].copy()
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    summary_keys = [*group_cols, category_col]
    summary = (
        df.groupby(summary_keys, dropna=False)
        .agg(
            total_prob=("_micro_sum", "sum"),
            total_prob_sq=("_micro_sq_sum", "sum"),
            n_values=("_micro_count", "sum"),
            n_prompts=(prompt_col, "nunique"),
        )
        .reset_index()
    )
    summary["mean_prob"] = summary["total_prob"] / summary["n_values"]
    centered_sum_sq = (
        summary["total_prob_sq"]
        - summary["total_prob"].pow(2) / summary["n_values"]
    ).clip(lower=0.0)
    variance_denom = (summary["n_values"] - 1).clip(lower=1)
    summary["var_prob"] = centered_sum_sq / variance_denom
    summary.loc[summary["n_values"].le(1), "var_prob"] = 0.0
    summary["std_prob"] = np.sqrt(summary["var_prob"])
    summary["stderr_prob"] = summary["std_prob"] / np.sqrt(summary["n_values"].clip(lower=1))

    if keep_category_lang_count:
        lang_counts = (
            df.groupby(summary_keys, dropna=False)[category_count_col]
            .mean()
            .rename("mean_category_lang_count")
            .reset_index()
        )
        summary = summary.merge(lang_counts, on=summary_keys, how="left")
    return summary[out_cols]


def micro_average_story_row_values(
    category_df: pd.DataFrame,
    *,
    prob_col: str,
    group_cols: Sequence[str],
    category_col: str = "prob_category",
    prompt_col: str = "prompt_id",
    layer_norm_col: str = "layer_norm",
    layer_min: float | None = 0.25,
    layer_max: float | None = 0.75,
    include_min: bool = True,
    include_max: bool = True,
    keep_category_lang_count: bool = False,
) -> pd.DataFrame:
    """Pool one scalar per prompt-layer row before averaging."""
    out_cols = [
        *group_cols,
        category_col,
        "mean_prob",
        "stderr_prob",
        "std_prob",
        "var_prob",
        "n_prompts",
        "n_values",
    ]
    if keep_category_lang_count:
        out_cols.append("mean_category_lang_count")
    if category_df.empty:
        return pd.DataFrame(columns=out_cols)

    required = [*group_cols, category_col, prompt_col, layer_norm_col, prob_col]
    missing = [col for col in required if col not in category_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for pooled row averaging: {missing}")

    df = category_df[story_layer_window_mask(
        category_df,
        layer_norm_col=layer_norm_col,
        layer_min=layer_min,
        layer_max=layer_max,
        include_min=include_min,
        include_max=include_max,
    )].copy()
    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")
    df = df[df[prob_col].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    summary_keys = [*group_cols, category_col]
    summary = (
        df.groupby(summary_keys, dropna=False)[prob_col]
        .agg(mean_prob="mean", std_prob="std", var_prob="var", n_values="count")
        .reset_index()
    )
    summary["std_prob"] = summary["std_prob"].fillna(0.0)
    summary["var_prob"] = summary["var_prob"].fillna(0.0)
    summary["stderr_prob"] = summary["std_prob"] / np.sqrt(summary["n_values"].clip(lower=1))
    prompt_counts = (
        df.groupby(summary_keys, dropna=False)[prompt_col]
        .nunique()
        .rename("n_prompts")
        .reset_index()
    )
    summary = summary.merge(prompt_counts, on=summary_keys, how="left")
    if keep_category_lang_count and "category_lang_count" in df.columns:
        lang_counts = (
            df.groupby(summary_keys, dropna=False)["category_lang_count"]
            .mean()
            .rename("mean_category_lang_count")
            .reset_index()
        )
        summary = summary.merge(lang_counts, on=summary_keys, how="left")
    return summary[out_cols]


def build_undetermined_langdist_rows(
    artifact: Mapping[str, Any],
    *,
    method: str,
    n_eff_threshold: float,
    margin_threshold: float,
    label: str = "undet",
) -> pd.DataFrame:
    """Assign an undetermined label using effective support and top-two margin."""
    methods = artifact.get("methods", {})
    if method not in methods:
        raise KeyError(f"Method '{method}' not found in lang-probs artifact.")

    payload = methods[method]
    langs = list(payload["langs"])
    probs = payload["probs"].to(torch.float32)
    if probs.ndim != 3:
        raise ValueError("Expected probs tensor with shape (prompt, layer, lang).")

    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    top1 = sorted_probs[..., 0]
    top2 = sorted_probs[..., 1] if probs.shape[-1] > 1 else torch.zeros_like(top1)
    margin = top1 - top2
    entropy = -(probs * probs.clamp_min(PROB_EPS).log()).sum(dim=-1)
    n_eff = torch.exp(entropy)
    undetermined_mask = (n_eff >= float(n_eff_threshold)) & (margin <= float(margin_threshold))

    prompt_ids = list(artifact.get("prompt_ids") or [None] * probs.shape[0])
    prompt_langs = list(artifact.get("prompt_langs") or [None] * probs.shape[0])
    layer_indices = list(artifact.get("layer_indices") or list(range(probs.shape[1])))
    checkpoint_id = artifact.get("checkpoint_id")

    rows = []
    for prompt_idx in range(probs.shape[0]):
        for layer_offset, layer_idx in enumerate(layer_indices):
            top1_idx = int(sorted_idx[prompt_idx, layer_offset, 0].item())
            top2_idx = int(sorted_idx[prompt_idx, layer_offset, 1].item()) if probs.shape[-1] > 1 else top1_idx
            dominant_lang = langs[top1_idx]
            rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "method": method,
                    "prompt_idx": int(prompt_idx),
                    "prompt_id": prompt_ids[prompt_idx],
                    "prompt_lang": prompt_langs[prompt_idx],
                    "layer": int(layer_idx),
                    "dominant_lang": dominant_lang,
                    "assigned_lang": label if bool(undetermined_mask[prompt_idx, layer_offset].item()) else dominant_lang,
                    "top1_lang": dominant_lang,
                    "top1_prob": float(top1[prompt_idx, layer_offset].item()),
                    "top2_lang": langs[top2_idx],
                    "top2_prob": float(top2[prompt_idx, layer_offset].item()),
                    "top2_margin": float(margin[prompt_idx, layer_offset].item()),
                    "n_eff": float(n_eff[prompt_idx, layer_offset].item()),
                    "undetermined": bool(undetermined_mask[prompt_idx, layer_offset].item()),
                }
            )
    return pd.DataFrame(rows)
