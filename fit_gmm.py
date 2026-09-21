"""Fit per-layer, per-language single-Gaussian parameters for representation-based LLID.

Calibration data is expected under `<data_root>/<lang>/train.csv` for the canonical
PUD split, with `clean.csv` still accepted as a fallback for older/external layouts.
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from tqdm import tqdm

from gmm_setup import (
    DEFAULT_GMM_SETUP_REGISTRY_PATH,
    build_gmm_artifact_dir,
    build_gmm_manifest,
    get_gmm_setup,
)
from latents import (
    collect_position_latents,
    collect_sequence_latents,
    get_transformer_layers,
    mean_pool_latents_by_word,
)
from representation import COVARIANCE_EPS, compute_pca, fit_language_gaussians
from run_eval_utils import format_prompts_for_tokenizer, get_surface_word_ids_for_prompts
from utils import load_nnsight_model, sanitize_model_id, set_seed


DEFAULT_CONFIG_PATH = Path("configs/default.json")
REPR_DEFAULT_KEYS = (
    "repr_priors",
    "repr_pca",
    "repr_pca_variance",
    "repr_cov",
    "repr_unit",
)


################################################################################
# Data loading
################################################################################

def _prompt_jsonl_candidates(data_root: Path) -> list[Path]:
    """Return supported prompt JSONL paths in preference order."""
    parent = data_root.parent
    return [
        parent / "pud_prompts_train.jsonl",
        parent / "ud_prompts_train.jsonl",
    ]


def _record_matches_lang(obj: dict[str, object], lang: str) -> bool:
    """Return whether a prompt record belongs to the requested language."""
    if obj.get("source_lang") == lang or obj.get("lang") == lang:
        return True
    prompt_id = str(obj.get("id", ""))
    return prompt_id.startswith(f"{lang}_pud-train-") or prompt_id.startswith(f"{lang}-train-")


def _load_prompt_records(
    data_root: Path,
    lang: str,
    text_column: str,
    max_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    """Load one language's deterministically sampled calibration prompts."""
    for prompt_jsonl in _prompt_jsonl_candidates(data_root):
        if not prompt_jsonl.exists():
            continue
        records: list[dict[str, object]] = []
        with prompt_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not _record_matches_lang(obj, lang):
                    continue
                text = obj.get("prompt_text") or obj.get("text")
                if text is None:
                    continue
                record = {"text": text}
                if isinstance(obj.get("surface_tokens"), list):
                    record["surface_tokens"] = list(obj["surface_tokens"])
                records.append(record)
        if records:
            sampled_indices = pd.Series(range(len(records))).sample(
                n=min(max_samples, len(records)),
                random_state=seed,
            )
            return [records[int(idx)] for idx in sampled_indices.tolist()]

    csv_path = data_root / lang / "train.csv"
    if not csv_path.exists():
        fallback = data_root / lang / "clean.csv"
        if fallback.exists():
            csv_path = fallback
        else:
            raise FileNotFoundError(f"Missing calibration CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in {csv_path}; "
            f"available columns: {list(df.columns)}"
        )
    df = df.sample(n=min(max_samples, len(df)), random_state=seed)
    prompts = df[text_column].dropna().astype(str).tolist()
    if not prompts:
        raise ValueError(f"No prompts loaded for language '{lang}' from {csv_path}.")
    return [{"text": prompt} for prompt in prompts]


def _load_prompt_records_from_roots(
    data_roots: list[Path],
    lang: str,
    text_column: str,
    max_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    """Load from the first data root containing the requested language."""
    errors: list[str] = []
    for data_root in data_roots:
        try:
            return _load_prompt_records(
                data_root=data_root,
                lang=lang,
                text_column=text_column,
                max_samples=max_samples,
                seed=seed,
            )
        except FileNotFoundError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else "no data roots provided"
    raise FileNotFoundError(f"Missing calibration data for language '{lang}': {detail}")


################################################################################
# CLI and config resolution
################################################################################


def _load_repr_defaults(default_config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    """Load representation defaults shared with evaluation."""
    payload = json.loads(default_config_path.read_text(encoding="utf-8"))
    missing = [key for key in REPR_DEFAULT_KEYS if key not in payload]
    if missing:
        raise KeyError(f"Missing repr default keys in {default_config_path}: {missing}")
    return {key: payload[key] for key in REPR_DEFAULT_KEYS}


def _build_parser(repr_defaults: dict[str, object]) -> argparse.ArgumentParser:
    """Build the GMM-fitting CLI parser."""
    parser = argparse.ArgumentParser(
        "Fit GMM-style Gaussian parameters per layer from calibration prompts."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--setup-name", default=None)
    parser.add_argument("--gmm-setup-registry", default=str(DEFAULT_GMM_SETUP_REGISTRY_PATH))
    parser.add_argument("--data-root", default="data/pud_holdout/pud_langs_train")
    parser.add_argument("--extra-data-root", nargs="*", default=[])
    parser.add_argument("--text-column", default="prompt")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--out-dir", default="logs/gmm")
    parser.add_argument(
        "--repr-priors",
        choices=["uniform", "empirical"],
        default=repr_defaults["repr_priors"],
    )
    parser.add_argument(
        "--repr-pca",
        choices=["none", "layerwise"],
        default=repr_defaults["repr_pca"],
    )
    parser.add_argument(
        "--repr-pca-variance",
        type=float,
        default=repr_defaults["repr_pca_variance"],
    )
    parser.add_argument(
        "--repr-cov",
        choices=["diag", "full"],
        default=repr_defaults["repr_cov"],
    )
    parser.add_argument(
        "--repr-unit",
        choices=["subtoken", "token"],
        default=repr_defaults["repr_unit"],
    )
    parser.add_argument("--trace-batch-size", type=int, default=1)
    return parser


def _get_explicit_cli_keys(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> set[str]:
    """Return explicitly supplied option destinations."""
    explicit_keys: set[str] = set()
    for action in parser._actions:
        if not action.option_strings:
            continue
        if any(arg == option or arg.startswith(f"{option}=") for arg in argv for option in action.option_strings):
            explicit_keys.add(action.dest)
    return explicit_keys


def _resolve_config(
    args: argparse.Namespace,
    explicit_keys: set[str],
) -> tuple[argparse.Namespace, dict[str, object] | None]:
    """Resolve a named setup, rejecting conflicting explicit arguments."""
    setup = None
    if args.setup_name:
        setup = get_gmm_setup(args.setup_name, args.gmm_setup_registry)
        expected_by_key = {
            "languages": setup["languages"],
            "data_root": setup["data_root"],
            "extra_data_root": setup["extra_data_roots"],
            "text_column": setup["text_column"],
            "max_samples": setup["max_samples"],
            "seed": setup["seed"],
            "repr_priors": setup["repr_priors"],
            "repr_pca": setup["repr_pca"],
            "repr_pca_variance": setup["repr_pca_variance"],
            "repr_cov": setup["repr_cov"],
            "repr_unit": setup["repr_unit"],
        }
        for key, expected in expected_by_key.items():
            if key not in explicit_keys:
                continue
            actual = getattr(args, key)
            if key in {"extra_data_root", "languages"}:
                actual = list(actual or [])
            if actual != expected:
                raise ValueError(
                    f"--setup-name {args.setup_name} conflicts with explicit --{key.replace('_', '-')}. "
                    f"Expected {expected!r}, got {actual!r}."
                )

        args.languages = list(setup["languages"])
        args.data_root = str(setup["data_root"])
        args.extra_data_root = list(setup["extra_data_roots"])
        args.text_column = str(setup["text_column"])
        args.max_samples = int(setup["max_samples"])
        args.seed = int(setup["seed"])
        args.repr_priors = str(setup["repr_priors"])
        args.repr_pca = str(setup["repr_pca"])
        args.repr_pca_variance = float(setup["repr_pca_variance"])
        args.repr_cov = str(setup["repr_cov"])
        args.repr_unit = str(setup["repr_unit"])

    if not args.languages:
        raise ValueError("Either --languages or --setup-name is required.")
    return args, setup


################################################################################
# Layer preparation and fitting
################################################################################

def _requires_sequence_latents(args: argparse.Namespace) -> bool:
    """Return whether token aggregation or PCA requires full prompt sequences."""
    return bool(args.repr_unit == "token" or args.repr_pca == "layerwise")


def _resolve_layer_indices(
    model: Any,
    requested_layers: Sequence[int] | None,
) -> list[int]:
    """Resolve requested transformer layers, defaulting to every layer."""
    layers = get_transformer_layers(model)
    layer_indices = list(range(len(layers))) if requested_layers is None else list(requested_layers)
    if not layer_indices:
        raise ValueError("No transformer layers available to collect latents from.")
    return layer_indices


def _fit_and_save_layer(
    *,
    layer_latents_by_lang: dict[str, torch.Tensor],
    layer_idx: int,
    out_root: Path,
    args: argparse.Namespace,
) -> None:
    """Fit and save one layer's PCA and per-language Gaussian parameters."""
    pca = None
    if args.repr_pca == "layerwise":
        pooled = torch.cat(list(layer_latents_by_lang.values()), dim=0)
        pca = compute_pca(pooled, args.repr_pca_variance)
        for lang in layer_latents_by_lang:
            layer_latents_by_lang[lang] = (
                layer_latents_by_lang[lang] - pca.mean
            ) @ pca.components.T

    cov_type = args.repr_cov
    params = fit_language_gaussians(layer_latents_by_lang, covariance_type=cov_type)

    priors = {lang: 1.0 / len(params) for lang in params}
    payload = {
        "pca": None,
        "gaussians": {
            "covariance_type": cov_type,
            "priors": priors if args.repr_priors == "uniform" else None,
            "languages": {
                lang: {
                    "mean": gaussian.mean.detach().cpu(),
                    "cov": gaussian.cov.detach().cpu(),
                    "n_samples": torch.tensor(gaussian.n_samples),
                }
                for lang, gaussian in params.items()
            },
        },
        "meta": {
            "model": args.model_name,
            "revision": args.revision,
            "languages": list(params.keys()),
            "layer": layer_idx,
            "eps": float(COVARIANCE_EPS),
            "repr_unit": args.repr_unit,
        },
    }
    if pca is not None:
        payload["pca"] = {
            "mean": pca.mean.detach().cpu(),
            "components": pca.components.detach().cpu(),
            "explained_var_ratio": pca.explained_var_ratio.detach().cpu(),
            "k": pca.k,
            "var_threshold": pca.var_threshold,
        }

    out_path = out_root / f"layer_{layer_idx}.pt"
    torch.save(payload, out_path)
    print(f"Wrote {out_path}")


def _write_manifest(
    *,
    out_root: Path,
    model_name_safe: str,
    setup: dict[str, object] | None,
    args: argparse.Namespace,
) -> None:
    """Write the resolved fitting configuration alongside layer artifacts."""
    manifest = build_gmm_manifest(
        setup_name=args.setup_name,
        model_name=args.model_name,
        revision=args.revision,
        languages=args.languages,
        data_root=args.data_root,
        extra_data_roots=args.extra_data_root,
        text_column=args.text_column,
        max_samples=args.max_samples,
        seed=args.seed,
        repr_priors=args.repr_priors,
        repr_pca=args.repr_pca,
        repr_pca_variance=args.repr_pca_variance,
        repr_cov=args.repr_cov,
        repr_unit=args.repr_unit,
    )
    manifest["model_name_safe"] = model_name_safe
    manifest["setup"] = setup
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _fit_sequence_layers(
    *,
    model: Any,
    inputs_by_lang: dict[str, dict[str, Any]],
    layer_indices: Sequence[int],
    out_root: Path,
    args: argparse.Namespace,
) -> None:
    """Collect sequence latents one layer at a time and fit each artifact."""
    layer_iter = tqdm(layer_indices, desc="Fitting layers", unit="layer")
    for layer_idx in layer_iter:
        layer_latents_by_lang = {}
        for lang, inputs in tqdm(
            inputs_by_lang.items(),
            desc=f"Collecting layer {layer_idx}",
            unit="lang",
            leave=False,
        ):
            full_latents_list, collected_layer_indices = collect_sequence_latents(
                model=model,
                prompts=inputs["prompts"],
                layer_indices=[layer_idx],
                trace_batch_size=args.trace_batch_size,
                show_progress=False,
            )
            if collected_layer_indices != [layer_idx]:
                raise RuntimeError(
                    f"Collected layer indices changed unexpectedly: "
                    f"{collected_layer_indices} != {[layer_idx]}"
                )
            token_reps = []
            for prompt_idx, prompt_latents in enumerate(full_latents_list):
                layer_latents = prompt_latents[0]
                if args.repr_unit == "token":
                    layer_latents, _ = mean_pool_latents_by_word(
                        layer_latents,
                        inputs["word_ids_list"][prompt_idx],
                    )
                token_reps.append(layer_latents.detach().float().cpu())
            layer_latents_by_lang[lang] = torch.cat(token_reps, dim=0)
            del full_latents_list, token_reps
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _fit_and_save_layer(
            layer_latents_by_lang=layer_latents_by_lang,
            layer_idx=layer_idx,
            out_root=out_root,
            args=args,
        )
        del layer_latents_by_lang
        gc.collect()


def _fit_position_layers(
    *,
    latents_by_lang: dict[str, torch.Tensor],
    layer_indices: Sequence[int],
    out_root: Path,
    args: argparse.Namespace,
) -> None:
    """Fit artifacts from pre-collected position latents."""
    for layer_pos, layer_idx in enumerate(layer_indices):
        layer_latents_by_lang = {}
        for lang, latents in latents_by_lang.items():
            layer_latents_by_lang[lang] = latents[:, layer_pos, :].detach().float().cpu()
        _fit_and_save_layer(
            layer_latents_by_lang=layer_latents_by_lang,
            layer_idx=layer_idx,
            out_root=out_root,
            args=args,
        )


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    parser = _build_parser(_load_repr_defaults())
    args = parser.parse_args()
    explicit_keys = _get_explicit_cli_keys(parser, sys.argv[1:])
    args, setup = _resolve_config(args, explicit_keys)

    set_seed(args.seed)

    model = load_nnsight_model(
        model_name=args.model_name,
        revision=args.revision,
        device=args.device,
        seed=args.seed,
    )

    data_roots = [Path(args.data_root)] + [Path(path) for path in args.extra_data_root]
    layer_indices = _resolve_layer_indices(model, args.layers)
    requires_sequence_latents = _requires_sequence_latents(args)

    # Retain formatted inputs for layer-at-a-time sequence collection.
    inputs_by_lang: dict[str, dict[str, Any]] = {}
    # Position latents have shape (prompts, layers, hidden_dim).
    latents_by_lang: dict[str, torch.Tensor] = {}
    lang_iter = tqdm(args.languages, desc="Loading prompts", unit="lang")
    for lang in lang_iter:
        prompt_records = _load_prompt_records_from_roots(
            data_roots=data_roots,
            lang=lang,
            text_column=args.text_column,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        raw_prompts = [str(record["text"]) for record in prompt_records]
        prompts, prompt_content_start_offsets = format_prompts_for_tokenizer(
            model.tokenizer,
            raw_prompts,
        )

        if requires_sequence_latents:
            surface_tokens_by_prompt = [
                list(record["surface_tokens"])
                if isinstance(record.get("surface_tokens"), list)
                else None
                for record in prompt_records
            ]
            if any(surface_tokens is None for surface_tokens in surface_tokens_by_prompt):
                raise ValueError(f"Missing surface_tokens in prompt records for language '{lang}'.")
            word_ids_list = get_surface_word_ids_for_prompts(
                model.tokenizer,
                prompts,
                surface_tokens_by_prompt=surface_tokens_by_prompt,
                content_start_offsets=prompt_content_start_offsets,
            )
            inputs_by_lang[lang] = {
                "prompts": prompts,
                "word_ids_list": word_ids_list,
            }
        else:
            latents, collected_layer_indices = collect_position_latents(
                model=model,
                prompts=prompts,
                layer_indices=layer_indices,
                position=args.position,
                trace_batch_size=args.trace_batch_size,
                show_progress=False,
            )
            if collected_layer_indices != layer_indices:
                raise RuntimeError(
                    f"Collected layer indices changed unexpectedly: {collected_layer_indices} != {layer_indices}"
                )
            latents_by_lang[lang] = latents

    model_name_safe = sanitize_model_id(args.model_name)
    if args.setup_name is not None:
        out_root = build_gmm_artifact_dir(
            base_dir=args.out_dir,
            setup_name=args.setup_name,
            model_name=args.model_name,
            revision=args.revision,
        )
    else:
        out_root = Path(args.out_dir) / model_name_safe / args.revision
    out_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        out_root=out_root,
        model_name_safe=model_name_safe,
        setup=setup,
        args=args,
    )

    if requires_sequence_latents:
        _fit_sequence_layers(
            model=model,
            inputs_by_lang=inputs_by_lang,
            layer_indices=layer_indices,
            out_root=out_root,
            args=args,
        )
    else:
        _fit_position_layers(
            latents_by_lang=latents_by_lang,
            layer_indices=layer_indices,
            out_root=out_root,
            args=args,
        )

    del model


if __name__ == "__main__":
    main()
