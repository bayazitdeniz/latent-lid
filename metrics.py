"""Compute metrics from language distributions."""

from collections.abc import Iterable, Mapping, Sequence

import torch


PROB_EPS = 1e-12


################################################################################
# Language-distribution assembly
################################################################################

def concat_langdist_chunks(
    chunks: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Concatenate prompt-batch chunks into one tensor per language."""
    if not chunks:
        raise ValueError("No language distribution chunks to concatenate.")
    langs = sorted({lang for chunk in chunks for lang in chunk})
    out: dict[str, torch.Tensor] = {}
    for lang in langs:
        pieces = []
        fallback_shape = None
        fallback_dtype = None
        for chunk in chunks:
            if lang in chunk:
                tensor = chunk[lang].detach().cpu()
                fallback_shape = tensor.shape[1:]
                fallback_dtype = tensor.dtype
                pieces.append(tensor)
            else:
                if fallback_shape is None:
                    for other in chunk.values():
                        fallback_shape = other.shape[1:]
                        fallback_dtype = other.dtype
                        break
                pieces.append(
                    torch.zeros(
                        (next(iter(chunk.values())).shape[0], *fallback_shape),
                        dtype=fallback_dtype,
                    )
                )
        out[lang] = torch.cat(pieces, dim=0)
    return out


################################################################################
# Language-distribution metrics
################################################################################

def _stack_langdist(
    langdist: Mapping[str, torch.Tensor],
) -> tuple[list[str], torch.Tensor]:
    """Stack per-language tensors along a new language axis.

    Each value normally has shape ``(prompts, layers)`` and contains one
    language's probability mass at every prompt/layer pair.
    The returned tensor therefore has shape ``(prompts, layers, languages)``.
    Inputs shaped ``(layers,)`` are also supported and produce ``(layers, languages)``.

    ``languages[i]`` identifies the language stored at ``stacked[..., i]``.
    """
    if not langdist:
        raise ValueError("langdist cannot be empty.")

    # Convert {language: (prompts, layers)} into (prompts, layers, languages).
    languages = sorted(langdist)
    stacked = torch.stack([langdist[language] for language in languages], dim=-1)
    return languages, stacked


def compute_langdist_metrics(
    langdist: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute dominance and entropy from a language distribution."""
    _, stacked = _stack_langdist(langdist)
    normalization = stacked.sum(dim=-1, keepdim=True).clamp_min(PROB_EPS)
    normalized = stacked / normalization
    dominance = normalized.max(dim=-1).values
    entropy = (-normalized * normalized.clamp_min(PROB_EPS).log()).sum(dim=-1)
    return {"dominance": dominance, "entropy": entropy}


def _langdist_argmax(
    langdist: Mapping[str, torch.Tensor],
) -> tuple[list[str], torch.Tensor]:
    langs, stacked = _stack_langdist(langdist)
    return langs, stacked.argmax(dim=-1)


def compute_langdist_prompt_pivot_mask(
    langdist: Mapping[str, torch.Tensor],
    task_langs_by_prompt: Sequence[Iterable[str]],
) -> torch.Tensor:
    """Return where the dominant language falls outside each prompt's task languages."""
    langs, argmax = _langdist_argmax(langdist)
    if len(task_langs_by_prompt) != int(argmax.shape[0]):
        raise ValueError("task_langs_by_prompt must match number of prompts.")

    pivot_mask = torch.empty_like(argmax, dtype=torch.bool)
    for prompt_idx, task_langs in enumerate(task_langs_by_prompt):
        task_langs = set(task_langs)
        if not task_langs:
            raise ValueError("Each prompt must have at least one task language.")
        task_lang_indices = [
            idx
            for idx, lang in enumerate(langs)
            if lang in task_langs
        ]
        if not task_lang_indices:
            raise ValueError("No task languages found in langdist for a prompt.")

        # Mark layers whose dominant language is one of this prompt's task languages.
        task_lang_mask = torch.zeros_like(argmax[prompt_idx], dtype=torch.bool)
        for idx in task_lang_indices:
            task_lang_mask |= argmax[prompt_idx] == idx
        # A pivot occurs when the dominant language is outside the task languages.
        pivot_mask[prompt_idx] = ~task_lang_mask
    return pivot_mask
