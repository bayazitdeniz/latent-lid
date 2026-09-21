"""Resolve named GMM setups and validate their fitted artifact metadata."""

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from utils import sanitize_model_id


DEFAULT_GMM_SETUP_REGISTRY_PATH = Path("configs/gmm_setups.json")


################################################################################
# Setup registry
################################################################################

def _normalize_languages(languages: Iterable[str | None]) -> list[str]:
    """Return sorted, unique, non-empty language names for setup comparison."""
    return sorted(
        {
            str(lang)
            for lang in languages
            if lang is not None and str(lang)
        }
    )


def load_gmm_setup_registry(
    path: str | Path = DEFAULT_GMM_SETUP_REGISTRY_PATH,
) -> dict[str, dict]:
    """Load the named GMM setups from a JSON registry."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_gmm_setup(
    name: str,
    path: str | Path = DEFAULT_GMM_SETUP_REGISTRY_PATH,
) -> dict:
    """Return one named setup from the registry."""
    setups = load_gmm_setup_registry(path)
    if name not in setups:
        raise KeyError(f"Unknown GMM setup '{name}'. Available setups: {sorted(setups)}")
    return setups[name]


def infer_gmm_setup_name(
    prompt_langs: Sequence[str | None],
    registry_source: str | Path | Mapping[str, dict] = DEFAULT_GMM_SETUP_REGISTRY_PATH,
) -> str:
    """Infer the unique setup whose languages exactly match the prompt languages."""
    prompt_set = _normalize_languages(prompt_langs)
    setups = (
        registry_source
        if isinstance(registry_source, Mapping)
        else load_gmm_setup_registry(registry_source)
    )
    matches = [
        name
        for name, setup in setups.items()
        if _normalize_languages(setup["languages"]) == prompt_set
    ]
    if not matches:
        raise ValueError(
            "No GMM setup matches the evaluated prompt language set exactly. "
            f"Prompt langs: {prompt_set}. Available setups: {sorted(setups)}"
        )
    if len(matches) > 1:
        raise ValueError(
            "Multiple GMM setups match the evaluated prompt language set exactly. "
            f"Prompt langs: {prompt_set}. Matching setups: {sorted(matches)}"
        )
    return matches[0]


################################################################################
# Artifact paths and manifests
################################################################################

def build_gmm_artifact_dir(
    base_dir: str | Path,
    setup_name: str,
    model_name: str,
    revision: str,
) -> Path:
    """Build the artifact directory for one setup, model, and revision."""
    return Path(base_dir) / setup_name / sanitize_model_id(model_name) / revision


def build_gmm_manifest(
    *,
    setup_name: str | None,
    model_name: str,
    revision: str,
    languages: Sequence[str],
    data_root: str,
    extra_data_roots: Sequence[str],
    text_column: str,
    max_samples: int,
    seed: int,
    repr_priors: str,
    repr_pca: str,
    repr_pca_variance: float,
    repr_cov: str,
    repr_unit: str,
) -> dict:
    """Build the resolved setup metadata stored with fitted GMM artifacts."""
    return {
        "schema_version": 1,
        "setup_name": setup_name,
        "model_name": model_name,
        "revision": revision,
        "languages": _normalize_languages(languages),
        "data_root": str(data_root),
        "extra_data_roots": [str(root) for root in extra_data_roots],
        "text_column": str(text_column),
        "max_samples": int(max_samples),
        "seed": int(seed),
        "repr_priors": str(repr_priors),
        "repr_pca": str(repr_pca),
        "repr_pca_variance": float(repr_pca_variance),
        "repr_cov": str(repr_cov),
        "repr_unit": str(repr_unit),
    }


def get_gmm_setup_comparison_view(setup: Mapping[str, object]) -> dict:
    """Return normalized setup fields for registry-to-manifest comparison."""
    return {
        "languages": _normalize_languages(setup.get("languages", [])),
        "data_root": str(setup.get("data_root", "")),
        "extra_data_roots": [str(root) for root in setup.get("extra_data_roots", [])],
        "text_column": str(setup.get("text_column", "prompt")),
        "max_samples": int(setup.get("max_samples", 0)),
        "seed": int(setup.get("seed", 0)),
        "repr_priors": str(setup.get("repr_priors", "")),
        "repr_pca": str(setup.get("repr_pca", "")),
        "repr_pca_variance": float(setup.get("repr_pca_variance", 0.0)),
        "repr_cov": str(setup.get("repr_cov", "")),
        "repr_unit": str(setup.get("repr_unit", "")),
    }
