"""Prepare unembedding weights and construct raw or tuned decoding lenses."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from utils import sanitize_model_id


NORM_EPS = 1e-12


################################################################################
# Unembedding
################################################################################

@dataclass
class UnembedInfo:
    """Weights, final normalization, and geometry used for hidden-state scoring."""

    weight: torch.Tensor
    bias: torch.Tensor | None
    normalized_weight: torch.Tensor
    avg_uu: torch.Tensor
    norm_module: torch.nn.Module | None

    @property
    def device(self) -> torch.device:
        return self.weight.device


def _get_final_norm_module(model: Any) -> torch.nn.Module:
    """Locate the model's final normalization layer."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "norm"):
        return model.norm
    if hasattr(model, "final_layer_norm"):
        return model.final_layer_norm
    raise ValueError("Unable to locate final normalization layer on the supplied model.")


def _get_lm_head(model: Any) -> torch.nn.Module:
    """Locate the model's language-model output head."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is None:
            raise ValueError("Model does not expose output embeddings.")
        return head
    raise ValueError("Unable to locate lm_head/output embeddings on the supplied model.")


def prepare_unembed_info(
    model: Any,
    device: torch.device | str | None = None,
) -> UnembedInfo:
    """Prepare raw unembedding weights and row-normalized geometry statistics.

    Final normalization is applied at projection time by ``apply_unembed`` when
    requested. Keeping ``weight`` as the raw LM head avoids applying norm weights
    both in the hidden-state transform and in the unembedding matrix.
    """
    lm_head = _get_lm_head(model)
    norm_module = _get_final_norm_module(model)

    weight = lm_head.weight.detach().clone().float()
    bias = (
        lm_head.bias.detach().clone().float()
        if getattr(lm_head, "bias", None) is not None
        else None
    )

    target_device = torch.device(device) if device is not None else weight.device
    weight = weight.to(target_device)
    if bias is not None:
        bias = bias.to(target_device)

    row_norms = weight.norm(dim=1, keepdim=True).clamp_min(NORM_EPS)
    normalized_weight = weight / row_norms

    vocab_size = weight.shape[0]
    gram = normalized_weight.T @ normalized_weight
    avg_uu = (gram.square().sum() / (vocab_size**2)).sqrt()

    return UnembedInfo(
        weight=weight,
        bias=bias,
        normalized_weight=normalized_weight,
        avg_uu=avg_uu,
        norm_module=norm_module,
    )


def apply_unembed(
    latents: torch.Tensor,
    info: UnembedInfo,
    apply_final_norm: bool = True,
) -> torch.Tensor:
    """Convert hidden states to logits using the unembedding weights."""
    outputs = latents
    if apply_final_norm and info.norm_module is not None:
        original_shape = outputs.shape
        outputs = outputs.reshape(-1, original_shape[-1])
        outputs = info.norm_module(outputs)
        outputs = outputs.reshape(original_shape)
    logits = outputs @ info.weight.T
    if info.bias is not None:
        logits = logits + info.bias
    return logits


################################################################################
# Tuned-lens artifacts
################################################################################

def build_tuned_lens_base_dir(
    model_name_or_path: str,
    revision: str,
    *,
    root: str | Path,
) -> Path:
    """Build the parent directory for one model and revision's tuned-lens runs."""
    return Path(root) / sanitize_model_id(model_name_or_path) / str(revision)


def next_tuned_lens_run_dir(base_dir: str | Path) -> Path:
    """Return the next available timestamped run directory without creating it."""
    base = Path(base_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = base / stamp
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        alternative = base / f"{stamp}_{suffix:02d}"
        if not alternative.exists():
            return alternative
        suffix += 1


def tuned_lens_layer_filename(layer_idx: int) -> str:
    """Return the artifact filename for one translated layer."""
    return f"layer_{int(layer_idx):02d}.pt"


def hash_training_config(config: Mapping[str, Any]) -> str:
    """Return a stable identifier for a tuned-lens training configuration."""
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class TunedLensLayerTranslator:
    """Learned affine hidden-state transformation for one model layer."""

    layer_idx: int
    weight: torch.Tensor
    bias: torch.Tensor
    hidden_size: int
    training_config_hash: str
    validation_kl: float | None = None
    validation_ce: float | None = None

    def to(self, device: torch.device | str) -> "TunedLensLayerTranslator":
        target = torch.device(device)
        return TunedLensLayerTranslator(
            layer_idx=self.layer_idx,
            weight=self.weight.to(target),
            bias=self.bias.to(target),
            hidden_size=self.hidden_size,
            training_config_hash=self.training_config_hash,
            validation_kl=self.validation_kl,
            validation_ce=self.validation_ce,
        )

    def transform_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)


@dataclass
class TunedLensSnapshot:
    """Loaded translators and metadata from one tuned-lens snapshot."""

    snapshot_dir: str
    config: dict[str, Any]
    translators: dict[int, TunedLensLayerTranslator]
    validation_kl_by_layer: dict[int, float]
    validation_ce_by_layer: dict[int, float]
    training_config_hash: str | None


def _is_tuned_lens_snapshot_dir(path: Path) -> bool:
    """Check whether a directory contains a complete-looking tuned-lens snapshot."""
    config_path = path / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    layer_files = list(path.glob("layer_*.pt"))
    if not layer_files:
        return False
    expected_layers = config.get("layers")
    if isinstance(expected_layers, list) and expected_layers:
        return len(layer_files) >= len(expected_layers)
    return True


def _resolve_tuned_lens_snapshot_dir(path: str | Path) -> Path:
    """Accept a snapshot directly or select the latest complete child snapshot."""
    root = Path(path)
    if _is_tuned_lens_snapshot_dir(root):
        return root
    if not root.exists():
        raise FileNotFoundError(f"Tuned lens directory not found: {root}")
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir() and _is_tuned_lens_snapshot_dir(child)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No complete tuned-lens run directories found under {root}. "
            "Expected timestamped subdirectories containing config.json and layer_*.pt files."
        )
    return sorted(candidates, key=lambda candidate: candidate.name)[-1]


def load_tuned_lens_snapshot(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> TunedLensSnapshot:
    """Load one tuned-lens snapshot or the latest complete snapshot under a directory."""
    snapshot_dir = _resolve_tuned_lens_snapshot_dir(path)
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Tuned lens config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    translators: dict[int, TunedLensLayerTranslator] = {}
    validation_kl_by_layer: dict[int, float] = {}
    validation_ce_by_layer: dict[int, float] = {}
    for layer_path in sorted(snapshot_dir.glob("layer_*.pt")):
        payload = torch.load(layer_path, map_location="cpu", weights_only=True)
        layer_idx = int(payload["layer_idx"])
        translator = TunedLensLayerTranslator(
            layer_idx=layer_idx,
            weight=payload["A"].to(torch.float32),
            bias=payload["b"].to(torch.float32),
            hidden_size=int(payload["hidden_size"]),
            training_config_hash=str(
                payload.get("training_config_hash")
                or config.get("training_config_hash")
                or ""
            ),
            validation_kl=(
                None
                if payload.get("validation_kl") is None
                else float(payload["validation_kl"])
            ),
            validation_ce=(
                None
                if payload.get("validation_ce") is None
                else float(payload["validation_ce"])
            ),
        )
        translators[layer_idx] = translator if device is None else translator.to(device)
        if translator.validation_kl is not None:
            validation_kl_by_layer[layer_idx] = translator.validation_kl
        if translator.validation_ce is not None:
            validation_ce_by_layer[layer_idx] = translator.validation_ce

    if not translators:
        raise FileNotFoundError(f"No tuned lens layer_*.pt files found in {snapshot_dir}.")

    if not validation_kl_by_layer and isinstance(config.get("validation_kl_by_layer"), dict):
        validation_kl_by_layer = {
            int(key): float(value)
            for key, value in config["validation_kl_by_layer"].items()
        }
    if not validation_ce_by_layer and isinstance(config.get("validation_ce_by_layer"), dict):
        validation_ce_by_layer = {
            int(key): float(value)
            for key, value in config["validation_ce_by_layer"].items()
        }

    return TunedLensSnapshot(
        snapshot_dir=str(snapshot_dir),
        config=config,
        translators=translators,
        validation_kl_by_layer=validation_kl_by_layer,
        validation_ce_by_layer=validation_ce_by_layer,
        training_config_hash=config.get("training_config_hash"),
    )


################################################################################
# Decoding lenses
################################################################################

class BaseDecodingLens:
    """Shared interface for projecting one or more hidden-state layers to logits."""

    def __init__(self, *, apply_final_norm: bool) -> None:
        self.apply_final_norm = bool(apply_final_norm)

    def project_layer(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def project_all_layers(
        self,
        hidden_states: torch.Tensor,
        layer_indices: Sequence[int],
    ) -> torch.Tensor:
        """Apply the decoding lens to hidden states from multiple layers."""
        if hidden_states.ndim < 2:
            raise ValueError("Expected hidden_states with at least 2 dims (..., layers, hidden_size).")
        if hidden_states.shape[-2] != len(layer_indices):
            raise ValueError(
                "Layer axis mismatch for decoding lens projection: "
                f"latents have {hidden_states.shape[-2]} layers, layer_indices has {len(layer_indices)}."
            )
        logits_by_layer = []
        for layer_pos, layer_idx in enumerate(layer_indices):
            logits = self.project_layer(hidden_states[..., layer_pos, :], int(layer_idx))
            logits_by_layer.append(logits.unsqueeze(-2))
        return torch.cat(logits_by_layer, dim=-2)


class RawLogitLens(BaseDecodingLens):
    """Project hidden states directly through the model's final norm and LM head."""

    def __init__(
        self,
        *,
        unembed_info: UnembedInfo,
        apply_final_norm: bool,
    ) -> None:
        super().__init__(apply_final_norm=apply_final_norm)
        self.unembed_info = unembed_info

    def project_layer(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        _ = layer_idx
        return apply_unembed(
            hidden_states.to(self.unembed_info.device),
            self.unembed_info,
            apply_final_norm=self.apply_final_norm,
        )


class TunedLogitLens(BaseDecodingLens):
    """Apply a learned layer translator before projecting hidden states to logits."""

    def __init__(
        self,
        *,
        unembed_info: UnembedInfo,
        translators: Mapping[int, TunedLensLayerTranslator],
        apply_final_norm: bool,
        final_layer_idx: int,
    ) -> None:
        super().__init__(apply_final_norm=apply_final_norm)
        self.unembed_info = unembed_info
        self.translators = {
            int(layer_idx): translator.to(unembed_info.device)
            for layer_idx, translator in translators.items()
        }
        self.final_layer_idx = int(final_layer_idx)

    def project_layer(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if int(layer_idx) == self.final_layer_idx:
            return apply_unembed(
                hidden_states.to(self.unembed_info.device),
                self.unembed_info,
                apply_final_norm=self.apply_final_norm,
            )
        if int(layer_idx) not in self.translators:
            raise KeyError(f"Missing tuned-lens translator for layer {layer_idx}.")
        translator = self.translators[int(layer_idx)]
        translated = translator.transform_hidden(hidden_states.to(self.unembed_info.device))
        return apply_unembed(
            translated,
            self.unembed_info,
            apply_final_norm=self.apply_final_norm,
        )


def _infer_num_hidden_layers(model: Any) -> int | None:
    """Infer the transformer layer count across supported model layouts."""
    base_model = getattr(model, "model", model)
    config = getattr(base_model, "config", None)
    total_layers = getattr(config, "num_hidden_layers", None)
    if total_layers is None and hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        total_layers = len(base_model.model.layers)
    if total_layers is None and hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
        total_layers = len(base_model.transformer.h)
    return None if total_layers is None else int(total_layers)


def build_decoding_lens(
    model: Any,
    *,
    decoding_lens: str,
    apply_final_norm: bool,
    tuned_lens_dir: str | Path | None = None,
    device: torch.device | str | None = None,
) -> tuple[BaseDecodingLens, dict[str, Any]]:
    """Construct a raw or tuned decoding lens and return it with run metadata."""
    resolved_lens_name = "raw_logitlens" if decoding_lens == "unembed" else decoding_lens
    total_layers = _infer_num_hidden_layers(model)
    unembed_info = prepare_unembed_info(model, device=device)
    if resolved_lens_name == "raw_logitlens":
        return (
            RawLogitLens(unembed_info=unembed_info, apply_final_norm=apply_final_norm),
            {
                "decoding_lens": "raw_logitlens",
                "tuned_lens_dir": None,
                "tuned_lens_training_config_hash": None,
                "tuned_lens_validation_kl_by_layer": {},
                "tuned_lens_validation_ce_by_layer": {},
            },
        )
    if resolved_lens_name != "tuned_lens":
        raise ValueError(f"Unsupported decoding lens: {decoding_lens}")
    if total_layers is None:
        raise ValueError("Unable to infer number of transformer layers for tuned lens.")
    if tuned_lens_dir is None:
        raise ValueError("tuned_lens_dir is required when decoding_lens='tuned_lens'.")
    loaded = load_tuned_lens_snapshot(tuned_lens_dir, device=unembed_info.device)
    return (
        TunedLogitLens(
            unembed_info=unembed_info,
            translators=loaded.translators,
            apply_final_norm=apply_final_norm,
            final_layer_idx=total_layers - 1,
        ),
        {
            "decoding_lens": "tuned_lens",
            "tuned_lens_dir": loaded.snapshot_dir,
            "tuned_lens_training_config_hash": loaded.training_config_hash,
            "tuned_lens_validation_kl_by_layer": loaded.validation_kl_by_layer,
            "tuned_lens_validation_ce_by_layer": loaded.validation_ce_by_layer,
        },
    )
