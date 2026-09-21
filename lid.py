"""Language-identification backends for decode-then-classify evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

PROB_EPS = 1e-12


################################################################################
# GlotLID label normalization
################################################################################

GLOTLID_LABEL_TO_NORMALIZED_LANG = {
    "arb_Arab": "ar",
    "ces_Latn": "cs",
    "eng_Latn": "en",
    "spa_Latn": "es",
    "fra_Latn": "fr",
    "hin_Deva": "hi",
    "ind_Latn": "id",
    "isl_Latn": "is",
    "por_Latn": "pt",
}


def normalize_glotlid_label(label: str) -> str:
    """Map a raw GlotLID label to the evaluation language space."""
    clean_label = label.replace("__label__", "")
    if clean_label in GLOTLID_LABEL_TO_NORMALIZED_LANG:
        return GLOTLID_LABEL_TO_NORMALIZED_LANG[clean_label]

    # Use the first known script variant for the same ISO-639-3 prefix.
    iso3 = clean_label.split("_", 1)[0]
    for glot_label, normalized_lang in GLOTLID_LABEL_TO_NORMALIZED_LANG.items():
        if glot_label.startswith(f"{iso3}_"):
            return normalized_lang

    return clean_label


################################################################################
# Language-ID backends
################################################################################

GLOTLID_MODEL_REPO_ID = "cis-lmu/glotlid"
GLOTLID_MODEL_FILENAME = "model_v3.bin"
GLOTLID_MODEL_REVISION = "74cb50b"


class LangIdBackend:
    """Language-ID backend powered by langid.py."""

    def __init__(self) -> None:
        try:
            import langid  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("langid is required for decode-then-classify mapping.") from exc
        self._langid = langid

    def predict(
        self,
        text: str,
        *,
        languages: Sequence[str] | None = None,
    ) -> tuple[str, dict[str, float]]:
        """Return (label, score_dict) for a given text.

        When `languages is None`, use `langid.classify(text)` directly because
        it already returns the unrestricted argmax label and its confidence.
        When `languages` is provided, we cannot use `classify` because it would
        still choose from all langid languages, so we instead build a filtered
        distribution via `predict_distribution(...)` and take its argmax.
        """
        if languages is None:
            label, score = self._langid.classify(text)
            return label, {label: float(score)}
        dist = self.predict_distribution(text, languages=languages)
        if not dist:
            raise ValueError("No language scores produced by langid.")
        label = max(dist, key=dist.get)
        return label, dist

    def predict_distribution(
        self,
        text: str,
        *,
        languages: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Return a language distribution from `langid.rank`.

        `langid.rank(text)` already returns normalized class probabilities
        (via `LanguageIdentifier.norm_probs`), not logits. So when we restrict
        to a subset of languages, we condition that probability distribution on
        the retained mass by taking `log(prob)` and applying a softmax over the
        kept labels. This is equivalent to L1 renormalization, but makes the
        conditional re-normalization step explicit.
        """
        ranked = self._langid.rank(text)
        if languages is not None:
            allowed = set(languages)
            ranked = [(lang, score) for lang, score in ranked if lang in allowed]
        if not ranked:
            return {}
        prob_tensor = torch.tensor([float(score) for _, score in ranked], dtype=torch.float32)
        # Condition the backend probabilities on the requested language set.
        log_probs = prob_tensor.clamp_min(PROB_EPS).log()
        probs = torch.softmax(log_probs, dim=0)
        return {lang: float(probs[idx].item()) for idx, (lang, _) in enumerate(ranked)}


class GlotLidBackend:
    """GlotLID backend using the published fastText model from Hugging Face."""

    def __init__(
        self,
        *,
        repo_id: str = GLOTLID_MODEL_REPO_ID,
        filename: str = GLOTLID_MODEL_FILENAME,
        revision: str = GLOTLID_MODEL_REVISION,
        cache_dir: str | Path | None = None,
    ) -> None:
        try:
            import fasttext  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("GlotLID requires fasttext-numpy2-wheel.") from exc
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover
            raise ImportError("GlotLID requires huggingface_hub.") from exc

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
        self._model = fasttext.load_model(model_path)
        self._output_matrix = np.asarray(self._model.get_output_matrix(), dtype=np.float32)
        self._labels = list(self._model.get_labels())

    @staticmethod
    def _prepare_text(text: str) -> str:
        """Flatten newlines because fastText sentence vectors do not support them."""
        return text.replace("\n", "  ")

    def predict(
        self,
        text: str,
        *,
        languages: Sequence[str] | None = None,
    ) -> tuple[str, dict[str, float]]:
        dist = self.predict_distribution(text, languages=languages)
        if not dist:
            raise ValueError("No language scores produced by GlotLID.")
        label = max(dist, key=dist.get)
        return label, dist

    def predict_distribution(
        self,
        text: str,
        *,
        languages: Sequence[str] | None = None,
    ) -> dict[str, float]:
        prepared_text = self._prepare_text(text)
        sentence_vector = np.asarray(self._model.get_sentence_vector(prepared_text), dtype=np.float32)

        # Normalize scores over the complete raw GlotLID label space.
        result_vector = np.dot(self._output_matrix, sentence_vector)
        shifted = result_vector - np.max(result_vector)
        exp_scores = np.exp(shifted)
        total = float(np.sum(exp_scores))
        if total <= 0:
            return {}
        probs = exp_scores / total

        # Collapse script-specific labels before conditioning on allowed languages.
        full_distribution: dict[str, float] = {}
        for prob, raw_label in zip(probs.tolist(), self._labels):
            normalized_lang = normalize_glotlid_label(raw_label)
            full_distribution[normalized_lang] = full_distribution.get(normalized_lang, 0.0) + float(prob)

        if languages is None:
            return full_distribution

        allowed = set(languages)
        distribution = {
            normalized_lang: prob
            for normalized_lang, prob in full_distribution.items()
            if normalized_lang in allowed
        }
        if not distribution:
            return {}

        # Condition the collapsed distribution on the requested language set.
        prob_tensor = torch.tensor(list(distribution.values()), dtype=torch.float32)
        log_probs = prob_tensor.clamp_min(PROB_EPS).log()
        conditioned = torch.softmax(log_probs, dim=0)
        conditioned_distribution: dict[str, float] = {}
        for idx, normalized_lang in enumerate(distribution.keys()):
            conditioned_distribution[normalized_lang] = float(conditioned[idx].item())
        return conditioned_distribution


def build_lid_backend(name: str) -> LangIdBackend | GlotLidBackend:
    """Build the requested LID backend."""
    if name == "langid":
        return LangIdBackend()
    if name == "glotlid":
        return GlotLidBackend()
    raise ValueError(f"Unsupported decoding_lid_backend: {name}")
