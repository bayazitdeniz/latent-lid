"""Collect and transform model hidden states shared by evaluation and fitting."""

import gc
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from tqdm import tqdm


################################################################################
# Model-layer access
################################################################################

def get_transformer_layers(model: Any) -> Any:
    """Return the transformer block stack for a supported model layout."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unable to locate transformer layers on the supplied model.")


def get_hf_layer_hidden_state(
    hidden_states: Sequence[torch.Tensor] | None,
    layer_idx: int,
) -> torch.Tensor:
    """Return the Hugging Face hidden states for a zero-based transformer layer."""
    hidden_state_idx = int(layer_idx) + 1
    if hidden_states is None or hidden_state_idx >= len(hidden_states):
        raise ValueError(
            f"Could not read hidden state for layer {layer_idx}; "
            f"received {0 if hidden_states is None else len(hidden_states)} hidden-state tensors."
        )
    return hidden_states[hidden_state_idx]


################################################################################
# Saved-trace materialization
################################################################################

def _iter_prompt_batches(prompts: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    for batch_start in range(0, len(prompts), batch_size):
        yield list(prompts[batch_start : batch_start + batch_size])


def _materialize_first_tensor(saved: Any) -> torch.Tensor:
    """Materialize the first tensor from a saved NNsight output."""
    if hasattr(saved, "value"):
        return _materialize_first_tensor(saved.value)
    if isinstance(saved, (list, tuple)):
        for item in saved:
            tensor = _materialize_first_tensor(item)
            if isinstance(tensor, torch.Tensor):
                return tensor
        raise RuntimeError("Saved output contained no tensors.")
    if isinstance(saved, torch.Tensor):
        return saved
    raise RuntimeError(f"Unsupported saved type: {type(saved)}")


def _trim_to_sequence_length(
    tensor: torch.Tensor,
    sequence_length: int | None,
    *,
    padding_side: str = "left",
) -> torch.Tensor:
    """Drop NNsight padding from a saved sequence tensor when its length is known."""
    if sequence_length is None:
        return tensor
    sequence_length = int(sequence_length)
    if sequence_length <= 0:
        raise ValueError(f"sequence length must be positive, got {sequence_length}.")
    if sequence_length > tensor.shape[0]:
        raise IndexError(
            f"sequence length {sequence_length} exceeds saved tensor length {tensor.shape[0]}."
        )
    if sequence_length == tensor.shape[0]:
        return tensor
    if padding_side == "right":
        return tensor[:sequence_length]
    return tensor[-sequence_length:]


def _materialize_layer_sequence(
    saved_output: Any,
    sequence_length: int | None,
    *,
    padding_side: str,
) -> torch.Tensor:
    """Materialize one layer output as an unpadded sequence tensor."""
    tensor = _materialize_first_tensor(saved_output)
    if tensor.dim() == 3:
        tensor = tensor[0]
    elif tensor.dim() != 2:
        raise RuntimeError(f"Unexpected block output shape: {tuple(tensor.shape)}")
    return _trim_to_sequence_length(
        tensor,
        sequence_length,
        padding_side=padding_side,
    )


################################################################################
# Latent collection
################################################################################

@torch.no_grad()
def collect_position_latents(
    model: Any,
    prompts: str | Sequence[str],
    layer_indices: Sequence[int] | None = None,
    position: int = -1,
    *,
    positions_by_prompt: Sequence[int] | None = None,
    sequence_lengths: Sequence[int] | None = None,
    trace_batch_size: int = 1,
    show_progress: bool = True,
    progress_position: int | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Collect one hidden-state position per prompt and transformer layer."""
    if not hasattr(model, "trace"):
        raise TypeError("collect_position_latents expects an nnsight.LanguageModel instance.")

    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    if not prompt_list:
        raise ValueError("prompts cannot be empty.")
    if sequence_lengths is not None and len(sequence_lengths) != len(prompt_list):
        raise ValueError("sequence_lengths must match the number of prompts.")
    if positions_by_prompt is not None and len(positions_by_prompt) != len(prompt_list):
        raise ValueError("positions_by_prompt must match the number of prompts.")
    layers = get_transformer_layers(model)
    resolved_layer_indices = (
        list(range(len(layers))) if layer_indices is None else list(layer_indices)
    )
    if not resolved_layer_indices:
        raise ValueError("No transformer layers available to collect latents from.")
    padding_side = getattr(getattr(model, "tokenizer", None), "padding_side", "left")

    all_prompt_latents = []
    trace_desc = (
        f"Tracing prompts ({len(prompt_list)} prompts, "
        f"{len(resolved_layer_indices)} layers)"
    )
    materialize_desc = f"Materializing latents ({len(prompt_list)} prompts)"

    batch_iter = _iter_prompt_batches(prompt_list, trace_batch_size)
    if show_progress:
        batch_iter = tqdm(
            batch_iter,
            desc=trace_desc,
            unit="batch",
            position=progress_position,
            total=(len(prompt_list) + trace_batch_size - 1) // trace_batch_size,
        )

    for batch in batch_iter:
        batch_start = len(all_prompt_latents)
        per_prompt_saves = []
        with model.trace() as tracer:
            for prompt in batch:
                saved_outputs = []
                with tracer.invoke(prompt):
                    for layer_idx in resolved_layer_indices:
                        saved_outputs.append(layers[layer_idx].output.save())
                per_prompt_saves.append(saved_outputs)

        iterator = (
            tqdm(
                per_prompt_saves,
                desc=materialize_desc,
                unit="prompt",
                position=progress_position,
                leave=False,
            )
            if show_progress and len(batch) > 1
            else per_prompt_saves
        )
        for local_idx, saved_outputs in enumerate(iterator):
            sequence_length = (
                None
                if sequence_lengths is None
                else int(sequence_lengths[batch_start + local_idx])
            )
            layer_latents = []
            for saved_output in saved_outputs:
                tensor = _materialize_layer_sequence(
                    saved_output,
                    sequence_length,
                    padding_side=padding_side,
                )
                effective_length = tensor.shape[0]
                requested_position = (
                    position
                    if positions_by_prompt is None
                    else int(positions_by_prompt[batch_start + local_idx])
                )
                resolved_position = (
                    requested_position
                    if requested_position >= 0
                    else effective_length + requested_position
                )
                if not (0 <= resolved_position < effective_length):
                    raise IndexError(
                        f"Position {requested_position} resolved to {resolved_position}, "
                        f"out of range for length {tensor.shape[0]}."
                    )
                layer_latents.append(tensor[resolved_position])
            if not layer_latents:
                raise RuntimeError(
                    "No layer latents captured; check layer paths and .output shape."
                )
            all_prompt_latents.append(torch.stack(layer_latents, dim=0))
        del per_prompt_saves
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.stack(all_prompt_latents, dim=0), resolved_layer_indices


@torch.no_grad()
def collect_sequence_latents(
    model: Any,
    prompts: str | Sequence[str],
    layer_indices: Sequence[int] | None = None,
    *,
    sequence_lengths: Sequence[int] | None = None,
    trace_batch_size: int = 1,
    show_progress: bool = True,
    progress_position: int | None = None,
    debug_label: str | None = None,
    debug_token_lengths: Sequence[int] | None = None,
) -> tuple[list[torch.Tensor], list[int]]:
    """Collect full-sequence hidden states for each prompt and transformer layer."""
    if not hasattr(model, "trace"):
        raise TypeError("collect_sequence_latents expects an nnsight.LanguageModel instance.")

    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    if not prompt_list:
        raise ValueError("prompts cannot be empty.")
    if sequence_lengths is not None and len(sequence_lengths) != len(prompt_list):
        raise ValueError("sequence_lengths must match the number of prompts.")
    if debug_token_lengths is not None and len(debug_token_lengths) != len(prompt_list):
        raise ValueError("debug_token_lengths must match the number of prompts.")
    layers = get_transformer_layers(model)
    resolved_layer_indices = (
        list(range(len(layers))) if layer_indices is None else list(layer_indices)
    )
    if not resolved_layer_indices:
        raise ValueError("No transformer layers available to collect latents from.")
    padding_side = getattr(getattr(model, "tokenizer", None), "padding_side", "left")

    all_prompt_latents = []
    trace_desc = (
        f"Tracing prompts ({len(prompt_list)} prompts, "
        f"{len(resolved_layer_indices)} layers)"
    )
    materialize_desc = f"Materializing latents ({len(prompt_list)} prompts)"

    batch_iter = _iter_prompt_batches(prompt_list, trace_batch_size)
    if show_progress:
        batch_iter = tqdm(
            batch_iter,
            desc=trace_desc,
            unit="batch",
            position=progress_position,
            total=(len(prompt_list) + trace_batch_size - 1) // trace_batch_size,
        )

    for batch in batch_iter:
        batch_start = len(all_prompt_latents)
        batch_end = batch_start + len(batch)
        if debug_label is not None:
            token_lengths = (
                list(debug_token_lengths[batch_start:batch_end])
                if debug_token_lengths is not None
                else []
            )
            token_detail = ""
            if token_lengths:
                token_detail = (
                    f" token_lengths={token_lengths}"
                    f" max_tokens={max(token_lengths)}"
                    f" min_tokens={min(token_lengths)}"
                )
            print(
                f"[latent_collect] {debug_label}"
                f" batch_start={batch_start}"
                f" batch_end={batch_end}"
                f" batch_size={len(batch)}"
                f" n_layers={len(resolved_layer_indices)}"
                f"{token_detail}",
                flush=True,
            )
        per_prompt_saves = []
        with model.trace() as tracer:
            for prompt in batch:
                saved_outputs = []
                with tracer.invoke(prompt):
                    for layer_idx in resolved_layer_indices:
                        saved_outputs.append(layers[layer_idx].output.save())
                per_prompt_saves.append(saved_outputs)

        iterator = (
            tqdm(
                per_prompt_saves,
                desc=materialize_desc,
                unit="prompt",
                position=progress_position,
                leave=False,
            )
            if show_progress and len(batch) > 1
            else per_prompt_saves
        )
        for local_idx, saved_outputs in enumerate(iterator):
            sequence_length = (
                None
                if sequence_lengths is None
                else int(sequence_lengths[batch_start + local_idx])
            )
            layer_latents = [
                _materialize_layer_sequence(
                    saved_output,
                    sequence_length,
                    padding_side=padding_side,
                )
                for saved_output in saved_outputs
            ]
            if not layer_latents:
                raise RuntimeError(
                    "No layer latents captured; check layer paths and .output shape."
                )
            all_prompt_latents.append(torch.stack(layer_latents, dim=0))
        del per_prompt_saves
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_prompt_latents, resolved_layer_indices


def stream_sequence_latent_batches(
    *,
    model: Any,
    prompts: Sequence[str],
    layer_indices: Sequence[int],
    prompt_token_lengths: Sequence[int],
    trace_batch_size: int,
    desc: str,
) -> Iterator[tuple[int, int, list[torch.Tensor]]]:
    """Yield full-sequence latent batches with their prompt index bounds."""
    if trace_batch_size <= 0:
        raise ValueError("trace_batch_size must be >= 1 for streamed latent collection.")
    total_batches = (len(prompts) + trace_batch_size - 1) // trace_batch_size
    batch_starts = range(0, len(prompts), trace_batch_size)
    batch_starts = tqdm(
        batch_starts,
        total=total_batches,
        desc=desc,
        unit="batch",
    )
    for start in batch_starts:
        end = min(start + trace_batch_size, len(prompts))
        full_latents_batch, _ = collect_sequence_latents(
            model=model,
            prompts=prompts[start:end],
            layer_indices=layer_indices,
            sequence_lengths=prompt_token_lengths[start:end],
            trace_batch_size=trace_batch_size,
            show_progress=False,
        )
        yield start, end, full_latents_batch
        del full_latents_batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


################################################################################
# Latent transformations
################################################################################

def gather_fixed_position_latents(
    full_latents_list: Sequence[torch.Tensor],
    position: int,
) -> torch.Tensor:
    """Stack one sequence position from every prompt and layer."""
    selected = []
    for prompt_latents in full_latents_list:
        layer_selected = []
        for layer_latents in prompt_latents:
            resolved_position = (
                position
                if position >= 0
                else layer_latents.shape[0] + position
            )
            if not (0 <= resolved_position < layer_latents.shape[0]):
                raise IndexError(
                    f"Position {position} resolved to {resolved_position}, "
                    f"out of range for length {layer_latents.shape[0]}."
                )
            layer_selected.append(layer_latents[resolved_position])
        selected.append(torch.stack(layer_selected, dim=0))
    return torch.stack(selected, dim=0)


def mean_pool_latents_by_word(
    latents: torch.Tensor,
    word_ids: Sequence[int | None] | None,
) -> tuple[torch.Tensor, bool]:
    """Mean-pool subtoken latents by word id in first-occurrence order."""
    if word_ids is None:
        return latents, False
    groups: dict[int, list[torch.Tensor]] = {}
    word_order: list[int] = []
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id not in groups:
            groups[word_id] = []
            word_order.append(word_id)
        groups[word_id].append(latents[token_idx])
    if not word_order:
        return latents, False
    averaged = [
        torch.stack(groups[word_id], dim=0).mean(dim=0)
        for word_id in word_order
    ]
    return torch.stack(averaged, dim=0), True
