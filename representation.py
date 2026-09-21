"""Representation GMM parameters, fitting primitives, artifact loading, and scoring."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from gmm_setup import (
    build_gmm_artifact_dir,
    get_gmm_setup_comparison_view,
    infer_gmm_setup_name,
    load_gmm_setup_registry,
)
from latents import mean_pool_latents_by_word, stream_sequence_latent_batches
from metrics import concat_langdist_chunks
from run_eval_utils import (
    get_ordered_word_ids,
    select_fractional_token_position,
    select_rollout_token_positions,
)

COVARIANCE_EPS = 1e-12
PROB_EPS = 1e-12


################################################################################
# GMM and PCA parameter types
################################################################################

@dataclass
class GaussianParams:
    """Parameters for one language's Gaussian distribution."""

    mean: torch.Tensor
    cov: torch.Tensor
    n_samples: int


@dataclass
class PCAParams:
    """Layerwise PCA projection stored with a GMM artifact."""

    mean: torch.Tensor
    components: torch.Tensor
    explained_var_ratio: torch.Tensor
    k: int
    var_threshold: float


@dataclass
class LoadedGMMLayer:
    """All language Gaussians and preprocessing metadata for one layer."""

    covariance_type: str
    layer_index: int
    params: dict[str, GaussianParams]  # Language label -> Gaussian parameters.
    mode: str = "legacy"
    priors: dict[str, float] | None = None
    pca: PCAParams | None = None
    meta: dict[str, object] | None = None


################################################################################
# Fitting primitives
################################################################################

def fit_language_gaussians(
    latents_by_lang: Mapping[str, torch.Tensor],
    covariance_type: str = "diag",
) -> dict[str, GaussianParams]:
    """Fit one Gaussian per language from ``(samples, hidden_dim)`` latents."""
    if covariance_type not in {"diag", "full"}:
        raise ValueError("covariance_type must be 'diag' or 'full'.")

    params: dict[str, GaussianParams] = {}
    for lang, latents in latents_by_lang.items():
        if latents.ndim != 2:
            raise ValueError(f"Latents for {lang} must be 2D (n_samples, hidden_dim).")
        if latents.shape[0] < 2:
            raise ValueError(f"Need at least 2 samples to fit covariance for {lang}.")

        mean = latents.mean(dim=0)
        centered = latents - mean

        if covariance_type == "diag":
            var = centered.pow(2).mean(dim=0).clamp_min(COVARIANCE_EPS)
            cov = var
        else:
            cov = (centered.T @ centered) / max(latents.shape[0] - 1, 1)
            cov = cov + torch.eye(cov.shape[0], device=cov.device) * COVARIANCE_EPS

        params[lang] = GaussianParams(mean=mean, cov=cov, n_samples=int(latents.shape[0]))

    return params


def compute_pca(latents: torch.Tensor, var_threshold: float) -> PCAParams:
    """Fit PCA and retain the components needed to reach a variance threshold."""
    if latents.ndim != 2:
        raise ValueError("PCA expects a 2D tensor (n_samples, dim).")
    mean = latents.mean(dim=0)
    x = latents - mean
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    var = (s ** 2) / max(latents.shape[0] - 1, 1)
    total = var.sum()
    ratio = var / total
    cum = torch.cumsum(ratio, dim=0)
    k = int((cum < var_threshold).sum().item() + 1)
    components = vh[:k]
    return PCAParams(
        mean=mean,
        components=components,
        explained_var_ratio=ratio[:k],
        k=k,
        var_threshold=var_threshold,
    )


################################################################################
# Posterior scoring
################################################################################

def apply_pca(latents: torch.Tensor, pca: PCAParams) -> torch.Tensor:
    """Project latents with a fitted PCA transformation."""
    mean = pca.mean.to(latents.device, latents.dtype)
    components = pca.components.to(latents.device, latents.dtype)
    return (latents - mean) @ components.T


def _log_prob_diag(latents: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    diff = latents - mean
    log_det = var.log().sum(dim=-1)
    quad = (diff.pow(2) / var).sum(dim=-1)
    const = torch.log(torch.tensor(2.0 * math.pi, device=latents.device, dtype=latents.dtype))
    return -0.5 * (quad + log_det + latents.shape[-1] * const)


def _log_prob_full(latents: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    diff = latents - mean
    chol = torch.linalg.cholesky(cov)
    solve = torch.cholesky_solve(diff.unsqueeze(-1), chol).squeeze(-1)
    quad = (diff * solve).sum(dim=-1)
    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    const = torch.log(torch.tensor(2.0 * math.pi, device=latents.device, dtype=latents.dtype))
    return -0.5 * (quad + log_det + latents.shape[-1] * const)


def compute_lang_posteriors(
    latents: torch.Tensor,
    loaded: LoadedGMMLayer,
    use_priors: bool = True,
    priors_override: Mapping[str, float] | str | None = None,
) -> dict[str, torch.Tensor]:
    """Return normalized language posteriors for each input latent."""
    if latents.ndim != 2:
        raise ValueError("Latents must be 2D (n_samples, hidden_dim).")
    params = loaded.params
    if not params:
        raise ValueError("No Gaussian params loaded.")

    if loaded.pca is not None:
        latents = apply_pca(latents, loaded.pca)

    langs = sorted(params.keys())
    log_probs = []
    sample_counts = []
    for lang in langs:
        p = params[lang]
        mean = p.mean.to(latents.device)
        cov = p.cov.to(latents.device)
        if loaded.covariance_type == "diag":
            log_p = _log_prob_diag(latents, mean, cov)
        else:
            log_p = _log_prob_full(latents, mean, cov)
        log_probs.append(log_p)
        sample_counts.append(p.n_samples)

    log_probs = torch.stack(log_probs, dim=-1)
    if priors_override is not None:
        use_priors = True

    if use_priors:
        if priors_override == "uniform":
            prior = torch.full(
                (len(langs),),
                1.0 / len(langs),
                device=latents.device,
                dtype=log_probs.dtype,
            )
        elif priors_override == "empirical":
            prior = torch.tensor(
                sample_counts,
                device=latents.device,
                dtype=log_probs.dtype,
            )
        elif isinstance(priors_override, Mapping):
            prior = torch.tensor(
                [priors_override.get(l, 0.0) for l in langs],
                device=latents.device,
                dtype=log_probs.dtype,
            )
        elif loaded.priors is not None:
            prior = torch.tensor(
                [loaded.priors.get(l, 0.0) for l in langs],
                device=latents.device,
                dtype=log_probs.dtype,
            )
        else:
            prior = torch.tensor(
                sample_counts,
                device=latents.device,
                dtype=log_probs.dtype,
            )
        prior = prior / prior.sum()
        log_probs = log_probs + torch.log(prior.clamp_min(PROB_EPS))

    log_probs = log_probs - log_probs.max(dim=-1, keepdim=True).values
    probs = torch.softmax(log_probs, dim=-1)
    return {lang: probs[..., i] for i, lang in enumerate(langs)}


################################################################################
# GMM artifact resolution and validation
################################################################################

REPR_GMM_LANGUAGE_ALIASES = {
    "pt_br": "pt",
}


def load_gmm_layer(path: str | Path) -> LoadedGMMLayer:
    """Load one layer's current or legacy GMM artifact."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Support PyTorch versions that predate the weights_only argument.
        payload = torch.load(path, map_location="cpu")

    # Current artifacts nest Gaussian parameters and optional PCA metadata.
    if "gaussians" in payload:
        mode = payload.get("mode", "paper")
        meta = payload.get("meta")
        gauss = payload["gaussians"]
        covariance_type = gauss.get("covariance_type", "full")
        layer_index = (
            int(meta.get("layer", payload.get("layer_index", 0)))
            if meta
            else int(payload.get("layer_index", 0))
        )
        priors = gauss.get("priors")
        params: dict[str, GaussianParams] = {}
        for lang, entries in gauss["languages"].items():
            params[lang] = GaussianParams(
                mean=entries["mean"].float(),
                cov=entries["cov"].float(),
                n_samples=int(entries.get("n_samples", 0)),
            )
        pca = None
        if "pca" in payload:
            pca_payload = payload["pca"]
            pca = PCAParams(
                mean=pca_payload["mean"].float(),
                components=pca_payload["components"].float(),
                explained_var_ratio=pca_payload["explained_var_ratio"].float(),
                k=int(pca_payload["k"]),
                var_threshold=float(pca_payload["var_threshold"]),
            )
        return LoadedGMMLayer(
            covariance_type=covariance_type,
            layer_index=layer_index,
            params=params,
            mode=mode,
            priors=priors,
            pca=pca,
            meta=meta,
        )

    # Legacy artifacts store the language mapping at the top level.
    covariance_type = payload.get("covariance_type", "diag")
    layer_index = int(payload.get("layer_index", 0))
    params = {}
    for lang, entries in payload["languages"].items():
        params[lang] = GaussianParams(
            mean=entries["mean"].float(),
            cov=entries["cov"].float(),
            n_samples=int(entries["n_samples"]),
        )
    return LoadedGMMLayer(
        covariance_type=covariance_type,
        layer_index=layer_index,
        params=params,
        mode=payload.get("mode", "legacy"),
    )


def canonical_repr_gmm_lang(lang: str | None) -> str | None:
    """Map a dataset language label to its representation GMM label."""
    if lang is None:
        return None
    return REPR_GMM_LANGUAGE_ALIASES.get(str(lang), str(lang))


def canonical_repr_gmm_langs(langs: Sequence[str | None] | None) -> list[str]:
    """Return sorted unique language labels in the representation GMM namespace."""
    return sorted(
        {
            canonical
            for lang in (langs or [])
            for canonical in [canonical_repr_gmm_lang(lang)]
            if canonical
        }
    )


def canonical_gmm_task_langs(
    task_langs_by_prompt: Sequence[Sequence[str]] | None,
) -> list[tuple[str, ...]]:
    """Map each prompt's task languages to representation GMM labels."""
    return [
        tuple(canonical_repr_gmm_langs(list(task_langs)))
        for task_langs in (task_langs_by_prompt or [])
    ]


def _validate_gmm_layer_tensors(
    layer_file: Path,
    loaded: LoadedGMMLayer,
) -> None:
    """Validate Gaussian and PCA tensor shapes for one layer artifact."""
    for lang, params in loaded.params.items():
        if params.mean.ndim != 1:
            raise ValueError(
                f"{layer_file} lang={lang}: mean must be 1D, "
                f"got {tuple(params.mean.shape)}"
            )
        if loaded.covariance_type == "diag":
            if params.cov.ndim != 1 or params.cov.shape[0] != params.mean.shape[0]:
                raise ValueError(
                    f"{layer_file} lang={lang}: diag cov must match mean dim, "
                    f"got mean {tuple(params.mean.shape)} cov {tuple(params.cov.shape)}"
                )
        else:
            if params.cov.ndim != 2 or params.cov.shape[0] != params.cov.shape[1]:
                raise ValueError(
                    f"{layer_file} lang={lang}: full cov must be square, "
                    f"got {tuple(params.cov.shape)}"
                )
            if params.cov.shape[0] != params.mean.shape[0]:
                raise ValueError(
                    f"{layer_file} lang={lang}: full cov dim must match mean dim, "
                    f"got mean {tuple(params.mean.shape)} cov {tuple(params.cov.shape)}"
                )
        if params.n_samples <= 0:
            raise ValueError(f"{layer_file} lang={lang}: n_samples must be > 0")

    if loaded.pca is None:
        return
    if loaded.pca.mean.ndim != 1 or loaded.pca.components.ndim != 2:
        raise ValueError(f"{layer_file}: malformed PCA tensors")
    if loaded.pca.components.shape[0] != loaded.pca.k:
        raise ValueError(
            f"{layer_file}: PCA components first dim must equal k, "
            f"got components {tuple(loaded.pca.components.shape)} k={loaded.pca.k}"
        )
    if loaded.pca.components.shape[1] != loaded.pca.mean.shape[0]:
        raise ValueError(
            f"{layer_file}: PCA component width must match PCA mean dim, "
            f"got components {tuple(loaded.pca.components.shape)} "
            f"mean {tuple(loaded.pca.mean.shape)}"
        )


def validate_repr_gmm_dir(
    gmm_dir: str | Path,
    *,
    expected_prompt_langs: Sequence[str | None] | None = None,
    expected_setup_name: str | None = None,
    expected_setup: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate representation GMM artifacts before tracing begins."""
    gmm_path = Path(gmm_dir)
    if not gmm_path.exists():
        raise FileNotFoundError(f"Representation GMM directory not found: {gmm_path}")

    layer_files = sorted(gmm_path.glob("layer_*.pt"))
    if not layer_files:
        raise FileNotFoundError(f"No layer_*.pt files found in {gmm_path}")

    expected_langs = canonical_repr_gmm_langs(expected_prompt_langs)
    shared_langs: set[str] | None = None
    layer_ids: list[int] = []

    for layer_file in layer_files:
        loaded = load_gmm_layer(layer_file)
        file_layer_match = re.search(r"layer_(\d+)\.pt$", layer_file.name)
        if file_layer_match is not None:
            file_layer_idx = int(file_layer_match.group(1))
            if int(loaded.layer_index) != file_layer_idx:
                raise ValueError(
                    f"GMM layer index mismatch for {layer_file}: "
                    f"filename says {file_layer_idx}, payload says {loaded.layer_index}"
                )
            layer_ids.append(file_layer_idx)

        langs = sorted(loaded.params.keys())
        if not langs:
            raise ValueError(f"No languages found in GMM file: {layer_file}")
        if shared_langs is None:
            shared_langs = set(langs)
        elif set(langs) != shared_langs:
            raise ValueError(
                f"Inconsistent language sets across GMM files in {gmm_path}. "
                f"{layer_file.name} has {langs}, expected {sorted(shared_langs)}"
            )

        _validate_gmm_layer_tensors(layer_file, loaded)

    available_langs = sorted(shared_langs or [])
    missing_prompt_langs = sorted(set(expected_langs) - set(available_langs))
    if missing_prompt_langs:
        raise ValueError(
            "Representation GMM support does not cover the prompt language set. "
            f"Missing prompt langs: {missing_prompt_langs}. "
            f"Available GMM langs: {available_langs}. "
            f"GMM dir: {gmm_path}"
        )

    manifest_path = gmm_path / "manifest.json"
    manifest_payload = None
    if expected_setup is not None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Expected GMM manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest_payload = json.load(f)
        if manifest_payload.get("setup_name") != expected_setup_name:
            raise ValueError(
                f"GMM manifest setup mismatch in {manifest_path}: "
                f"expected {expected_setup_name!r}, found {manifest_payload.get('setup_name')!r}"
            )
        manifest_setup = manifest_payload.get("setup") or manifest_payload
        if get_gmm_setup_comparison_view(manifest_setup) != get_gmm_setup_comparison_view(expected_setup):
            raise ValueError(
                f"GMM manifest setup payload mismatch in {manifest_path} for setup {expected_setup_name!r}."
            )

    return {
        "gmm_dir": str(gmm_path),
        "n_layers": len(layer_files),
        "layer_ids": layer_ids,
        "languages": available_langs,
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "setup_name": expected_setup_name,
    }


def resolve_repr_gmm_dir(
    cfg: Mapping[str, Any],
    *,
    prompt_langs: Sequence[str | None],
    revision: str,
) -> tuple[Path, str | None, dict[str, object] | None]:
    """Resolve the GMM directory and optional named-setup metadata."""
    registry = load_gmm_setup_registry(cfg["gmm_setup_registry"])
    if cfg.get("repr_gmm_setup"):
        setup_name = str(cfg["repr_gmm_setup"])
        if setup_name not in registry:
            raise ValueError(
                f"Unknown repr_gmm_setup '{setup_name}'. Available setups: {sorted(registry)}"
            )
        setup = registry[setup_name]
    elif cfg.get("repr_gmm_dir"):
        return Path(str(cfg["repr_gmm_dir"]).format(rev=revision)), None, None
    else:
        setup_name = infer_gmm_setup_name(prompt_langs, registry)
        setup = registry[setup_name]

    if cfg.get("repr_gmm_dir"):
        return Path(str(cfg["repr_gmm_dir"]).format(rev=revision)), setup_name, setup

    gmm_dir = build_gmm_artifact_dir(
        base_dir="logs/gmm",
        setup_name=setup_name,
        model_name=cfg["model_name"],
        revision=revision,
    )
    return gmm_dir, setup_name, setup


################################################################################
# Latent selection
################################################################################

def select_repr_latents(
    layer_latents: torch.Tensor,
    repr_token_agg: str,
    *,
    word_ids: Sequence[int | None] | None = None,
    repr_rollout_k: int = 0,
) -> torch.Tensor:
    """Select token latents for the configured representation aggregation."""
    if repr_rollout_k < 0:
        raise ValueError("repr_rollout_k must be non-negative.")
    if repr_token_agg == "all_tokens":
        if repr_rollout_k > 0:
            raise ValueError(
                "repr_rollout_k cannot be combined with repr_token_agg=all_tokens."
            )
        return layer_latents
    if repr_rollout_k > 0:
        positions = select_rollout_token_positions(
            layer_latents.shape[0],
            repr_token_agg,
            rollout_k=repr_rollout_k,
            rollout_word_cnt=0,
            word_ids=word_ids,
        )
        return layer_latents[positions].contiguous()
    if repr_token_agg == "last_token":
        return layer_latents[-1:].contiguous()
    if repr_token_agg not in {"frac_50", "frac_75"}:
        raise ValueError(f"Unsupported repr_token_agg: {repr_token_agg}")
    pos = select_fractional_token_position(
        layer_latents.shape[0],
        repr_token_agg,
        word_ids=word_ids,
    )
    return layer_latents[pos : pos + 1].contiguous()


def select_repr_word_rollout_latents(
    layer_latents: torch.Tensor,
    repr_token_agg: str,
    *,
    word_ids: Sequence[int | None] | None,
    repr_rollout_k: int,
) -> torch.Tensor:
    """Select word-level latents for a rollout window defined in subtoken space."""
    aggregated_latents, used_word_ids = mean_pool_latents_by_word(
        layer_latents,
        word_ids,
    )
    if not used_word_ids:
        return select_repr_latents(
            aggregated_latents,
            repr_token_agg,
            word_ids=None,
            repr_rollout_k=repr_rollout_k,
        )

    if repr_rollout_k <= 0:
        return select_repr_latents(
            aggregated_latents,
            repr_token_agg,
            word_ids=None,
            repr_rollout_k=0,
        )

    positions = select_rollout_token_positions(
        layer_latents.shape[0],
        repr_token_agg,
        rollout_k=repr_rollout_k,
        rollout_word_cnt=0,
        word_ids=word_ids,
    )
    if not positions:
        return aggregated_latents[-1:].contiguous()

    ordered_word_ids = get_ordered_word_ids(word_ids)
    word_index_by_id = {wid: idx for idx, wid in enumerate(ordered_word_ids)}
    touched_word_ids: list[int] = []
    seen = set()
    for pos in positions:
        wid = word_ids[pos] if word_ids is not None and pos < len(word_ids) else None
        if wid is None:
            continue
        wid_int = int(wid)
        if wid_int in seen:
            continue
        touched_word_ids.append(wid_int)
        seen.add(wid_int)

    if not touched_word_ids:
        return select_repr_latents(
            aggregated_latents,
            repr_token_agg,
            word_ids=None,
            repr_rollout_k=0,
        )

    selected_indices = [
        word_index_by_id[wid]
        for wid in touched_word_ids
        if wid in word_index_by_id
    ]
    if not selected_indices:
        return select_repr_latents(
            aggregated_latents,
            repr_token_agg,
            word_ids=None,
            repr_rollout_k=0,
        )
    return aggregated_latents[selected_indices].contiguous()


################################################################################
# Representation scoring
################################################################################

def compute_repr_langdist_from_latents(
    latents: torch.Tensor,
    layer_indices: Sequence[int],
    gmm_dir: str | Path,
    use_priors: bool = True,
    priors_override: Mapping[str, float] | str | None = None,
) -> dict[str, torch.Tensor]:
    """Compute a language distribution per prompt and layer."""
    if latents.ndim != 3:
        raise ValueError("Latents must be 3D (prompts, layers, hidden_dim).")
    if len(layer_indices) != latents.shape[1]:
        raise ValueError("layer_indices length must match latents.shape[1].")

    gmm_path = Path(gmm_dir)
    lang_order: list[str] | None = None
    posteriors_by_lang: dict[str, list[torch.Tensor]] = {}

    for layer_pos, layer_idx in enumerate(layer_indices):
        loaded = load_gmm_layer(gmm_path / f"layer_{layer_idx}.pt")
        layer_latents = latents[:, layer_pos, :].float()
        layer_posteriors = compute_lang_posteriors(
            layer_latents,
            loaded,
            use_priors=use_priors,
            priors_override=priors_override,
        )

        if lang_order is None:
            lang_order = sorted(layer_posteriors)
            for lang in lang_order:
                posteriors_by_lang[lang] = []
        elif sorted(layer_posteriors) != lang_order:
            raise ValueError("Language keys differ across GMM layers.")

        for lang in lang_order:
            posteriors_by_lang[lang].append(layer_posteriors[lang].cpu())

    return {
        lang: torch.stack(layer_posteriors, dim=1)
        for lang, layer_posteriors in posteriors_by_lang.items()
    }


def compute_repr_langdist_from_sequences(
    sequence_latents_by_prompt: Sequence[torch.Tensor],
    layer_indices: Sequence[int],
    gmm_dir: str | Path,
    repr_priors: str,
    repr_unit: str,
    repr_token_agg: str,
    repr_rollout_k: int,
    word_ids_by_prompt: Sequence[Sequence[int | None] | None] | None,
    *,
    show_progress: bool = True,
) -> tuple[dict[str, torch.Tensor], bool, dict[int, int], str | None]:
    """Score collected sequence latents with the representation GMM."""
    values_by_lang: dict[str, list[float]] = {}
    word_ids_used = False
    pca_k_by_layer: dict[int, int] = {}
    gmm_mode: str | None = None
    priors_override = "uniform" if repr_priors == "uniform" else "empirical"
    layer_iter = enumerate(layer_indices)
    if show_progress:
        layer_iter = tqdm(
            layer_iter,
            total=len(layer_indices),
            desc="Repr/GMM layers",
            unit="layer",
        )

    # Preserve layer-first, prompt-second accumulation for exact output ordering.
    for layer_pos, layer_idx in layer_iter:
        loaded = load_gmm_layer(Path(gmm_dir) / f"layer_{layer_idx}.pt")
        if gmm_mode is None:
            gmm_mode = loaded.mode
        if loaded.pca is not None:
            pca_k_by_layer[int(layer_idx)] = loaded.pca.k
        for prompt_idx, prompt_latents in enumerate(sequence_latents_by_prompt):
            layer_latents = prompt_latents[layer_pos]
            prompt_word_ids = (
                None
                if word_ids_by_prompt is None
                else word_ids_by_prompt[prompt_idx]
            )
            # Subtoken scoring keeps word IDs so fractional anchors can use word ends.
            if repr_unit == "token":
                if repr_rollout_k > 0:
                    layer_latents = select_repr_word_rollout_latents(
                        layer_latents,
                        repr_token_agg,
                        word_ids=prompt_word_ids,
                        repr_rollout_k=repr_rollout_k,
                    )
                    word_ids_used = word_ids_used or bool(
                        get_ordered_word_ids(prompt_word_ids)
                    )
                    prompt_word_ids = None
                else:
                    layer_latents, used = mean_pool_latents_by_word(
                        layer_latents,
                        prompt_word_ids,
                    )
                    word_ids_used = word_ids_used or used
                    prompt_word_ids = None

            scoring_latents = layer_latents
            if repr_token_agg != "all_tokens":
                scoring_latents = select_repr_latents(
                    layer_latents,
                    repr_token_agg,
                    word_ids=prompt_word_ids,
                    repr_rollout_k=repr_rollout_k,
                )

            token_posteriors = compute_lang_posteriors(
                scoring_latents,
                loaded,
                use_priors=True,
                priors_override=priors_override,
            )
            # Average token posteriors into the paper's prompt-level q_lang.
            q_lang = {
                lang: token_posteriors[lang].mean().item()
                for lang in token_posteriors
            }

            for lang, val in q_lang.items():
                values_by_lang.setdefault(lang, []).append(val)

    # Values were appended layer first; restore the (prompts, layers) layout.
    num_prompts = len(sequence_latents_by_prompt)
    langdist = {}
    for lang, values in values_by_lang.items():
        tensor = torch.tensor(values, dtype=torch.float32).view(
            len(layer_indices),
            num_prompts,
        ).T
        langdist[lang] = tensor
    return langdist, word_ids_used, pca_k_by_layer, gmm_mode


def compute_repr_langdist_streamed(
    *,
    model: Any,
    prompts: Sequence[str],
    layer_indices: Sequence[int],
    prompt_token_lengths: Sequence[int],
    trace_batch_size: int,
    gmm_dir: str | Path,
    repr_priors: str,
    repr_unit: str,
    repr_token_agg: str,
    repr_rollout_k: int,
    word_ids_by_prompt: Sequence[Sequence[int | None] | None] | None,
) -> tuple[dict[str, torch.Tensor], bool, dict[int, int], str | None]:
    """Trace sequence latents in batches and score their GMM language distributions."""
    chunks = []
    word_ids_used = False
    pca_k_by_layer: dict[int, int] = {}
    gmm_mode: str | None = None
    for start, end, sequence_latents_batch in stream_sequence_latent_batches(
        model=model,
        prompts=prompts,
        layer_indices=layer_indices,
        prompt_token_lengths=prompt_token_lengths,
        trace_batch_size=trace_batch_size,
        desc="Streaming repr traces",
    ):
        chunk_langdist, chunk_word_ids_used, chunk_pca, chunk_gmm_mode = (
            compute_repr_langdist_from_sequences(
                sequence_latents_by_prompt=sequence_latents_batch,
                layer_indices=layer_indices,
                gmm_dir=gmm_dir,
                repr_priors=repr_priors,
                repr_unit=repr_unit,
                repr_token_agg=repr_token_agg,
                repr_rollout_k=repr_rollout_k,
                word_ids_by_prompt=(
                    None
                    if word_ids_by_prompt is None
                    else word_ids_by_prompt[start:end]
                ),
                show_progress=False,
            )
        )
        chunks.append(
            {
                lang: tensor.detach().cpu()
                for lang, tensor in chunk_langdist.items()
            }
        )
        word_ids_used = word_ids_used or chunk_word_ids_used
        pca_k_by_layer.update(chunk_pca)
        if gmm_mode is None:
            gmm_mode = chunk_gmm_mode
    return concat_langdist_chunks(chunks), word_ids_used, pca_k_by_layer, gmm_mode
