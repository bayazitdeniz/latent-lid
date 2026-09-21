"""Decode-then-classify evaluation helpers."""

from typing import Sequence

import torch
from tqdm import tqdm

from latents import (
    gather_fixed_position_latents,
    get_hf_layer_hidden_state,
    get_transformer_layers,
    stream_sequence_latent_batches,
)
from lid import LangIdBackend
from lenses import RawLogitLens, TunedLogitLens, apply_unembed
from metrics import concat_langdist_chunks
from run_eval_utils import (
    select_fractional_token_position,
    select_rollout_token_positions,
    select_scoring_token_position,
)


PROB_EPS = 1e-12


class DecodeThenClassifyScorer:
    """Decode from logits/probs and classify language using a LID backend."""

    def __init__(
        self,
        tokenizer,
        lid_backend: LangIdBackend,
        candidate_languages: Sequence[str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.lid_backend = lid_backend
        self.candidate_languages = (
            sorted(set(candidate_languages)) if candidate_languages is not None else None
        )

    def classify_text(self, text: str) -> tuple[str | None, dict[str, float]]:
        """Classify decoded text."""
        return self.lid_backend.predict(text, languages=self.candidate_languages)

    def score_text(self, text: str) -> dict[str, float]:
        """Score decoded text with a language distribution."""
        return self.lid_backend.predict_distribution(text, languages=self.candidate_languages)

    def decode_argmax(self, probs: torch.Tensor) -> str:
        """Decode single token by argmax and return decoded string."""
        token_id = int(probs.argmax(dim=-1).item())
        return self.tokenizer.decode([token_id])

    def decode_topk(self, probs: torch.Tensor, k: int = 5) -> list[str]:
        """Decode top-k tokens and return decoded strings."""
        topk = torch.topk(probs, k=k, dim=-1).indices.squeeze(-1).tolist()
        return [self.tokenizer.decode([int(t)]) for t in topk]

    def decode_topk_with_probs(self, probs: torch.Tensor, k: int = 5) -> list[tuple[str, float]]:
        """Decode top-k tokens and keep their LM probabilities."""
        k = min(int(k), int(probs.shape[-1]))
        top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
        return [
            (self.tokenizer.decode([int(token_id)]), float(token_prob))
            for token_prob, token_id in zip(top_probs.tolist(), top_ids.tolist())
        ]

    def classify_from_probs(
        self,
        probs: torch.Tensor,
        mode: str = "argmax",
        topk: int = 5,
    ) -> tuple[str | None, dict[str, float]]:
        """Decode from probs and classify via the LID backend."""
        if mode == "argmax":
            text = self.decode_argmax(probs)
            return self.classify_text(text)
        if mode == "topk":
            texts = self.decode_topk(probs, k=topk)
            votes: dict[str, float] = {}
            for text in texts:
                label, scores = self.classify_text(text)
                if label is None:
                    continue
                votes[label] = votes.get(label, 0.0) + 1.0
            if not votes:
                return None, {}
            return max(votes, key=votes.get), votes
        if mode == "topk_weighted":
            weighted: dict[str, float] = {}
            for text, token_prob in self.decode_topk_with_probs(probs, k=topk):
                lang_dist = self.score_text(text)
                for lang, lang_prob in lang_dist.items():
                    weighted[lang] = weighted.get(lang, 0.0) + float(token_prob) * float(lang_prob)
            if not weighted:
                return None, {}
            total = sum(weighted.values())
            if total > 0:
                weighted = {lang: val / total for lang, val in weighted.items()}
            return max(weighted, key=weighted.get), weighted
        if mode == "rollout":
            raise NotImplementedError("rollout decoding is not implemented yet.")
        raise ValueError(f"Unsupported decode mode: {mode}")


################################################################################
# Top-p sampling
################################################################################

def sample_top_p_token_id(probs: torch.Tensor, top_p: float) -> int:
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1].")
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= top_p
    keep[0] = True
    filtered_probs = sorted_probs[keep]
    filtered_ids = sorted_ids[keep]
    filtered_probs = filtered_probs / filtered_probs.sum()
    sample_idx = torch.multinomial(filtered_probs, num_samples=1)
    return int(filtered_ids[sample_idx].item())


def sample_top_p_token_ids(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Sample one token id per row from a batched top-p nucleus distribution."""
    if probs.dim() == 1:
        return torch.tensor([sample_top_p_token_id(probs, top_p)], device=probs.device, dtype=torch.long)
    if probs.dim() != 2:
        raise ValueError(f"Expected probs to have shape [batch, vocab], got {tuple(probs.shape)}.")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1].")
    sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= top_p
    keep[:, 0] = True
    filtered_probs = sorted_probs * keep
    filtered_probs = filtered_probs / filtered_probs.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(PROB_EPS)
    sample_idx = torch.multinomial(filtered_probs, num_samples=1)
    return sorted_ids.gather(1, sample_idx).squeeze(1)


################################################################################
# Sampled-rollout caching helpers
################################################################################

def _get_hf_causal_lm(model):
    """Return the underlying HF causal LM for cache-aware rollout sampling."""
    return getattr(model, "_model", model)


def _expand_past_key_values(past_key_values, batch_size: int):
    """Repeat a cached prefix state across the sample batch dimension."""
    if past_key_values is None or batch_size <= 1:
        return past_key_values
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        expanded = _expand_past_key_values(legacy, batch_size)
        cache_type = type(past_key_values)
        if hasattr(cache_type, "from_legacy_cache"):
            return cache_type.from_legacy_cache(expanded)
        return expanded
    if isinstance(past_key_values, torch.Tensor):
        return past_key_values.repeat_interleave(batch_size, dim=0)
    if isinstance(past_key_values, tuple):
        return tuple(_expand_past_key_values(item, batch_size) for item in past_key_values)
    if isinstance(past_key_values, list):
        return [_expand_past_key_values(item, batch_size) for item in past_key_values]
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def _project_hf_cached_layer_logits(
    *,
    step_latents: torch.Tensor,
    layer_idx: int,
    causal_lm,
    decoding_lens,
) -> torch.Tensor:
    """Project HF hidden states, accounting for final-layer hidden-state convention."""
    if isinstance(decoding_lens, (RawLogitLens, TunedLogitLens)):
        n_layers = len(get_transformer_layers(causal_lm))
        if int(layer_idx) == n_layers - 1:
            return apply_unembed(
                step_latents.to(decoding_lens.unembed_info.device),
                decoding_lens.unembed_info,
                apply_final_norm=False,
            )
    return decoding_lens.project_layer(step_latents, int(layer_idx))


def _left_pad_token_id_rows(
    rows: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad variable-length token-id rows and return ids plus attention mask."""
    if not rows:
        raise ValueError("rows must be non-empty.")
    max_len = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for row_idx, row in enumerate(rows):
        row_len = len(row)
        input_ids[row_idx, max_len - row_len :] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[row_idx, max_len - row_len :] = 1
    return input_ids, attention_mask


def _iter_batches(items, batch_size: int):
    """Yield fixed-size slices from a sequence."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _accumulate_sampled_rollout_scores(
    *,
    prompt_indices: Sequence[int],
    layer_pos: int,
    layer_idx: int,
    generated_ids_by_row: Sequence[Sequence[int]],
    rollout_num_samples: int,
    tokenizer,
    scorer,
    score_sums,
    sample_records: list[dict[str, object]],
    observed_langs: set,
):
    """Decode sampled continuations and accumulate per-prompt score sums."""
    for row_idx, generated_ids in enumerate(generated_ids_by_row):
        prompt_idx = int(prompt_indices[row_idx // rollout_num_samples])
        sample_idx = int(row_idx % rollout_num_samples)
        decoded_text = tokenizer.decode(generated_ids) if generated_ids else ""
        scores = scorer.score_text(decoded_text)
        observed_langs.update(scores.keys())
        for lang, score in scores.items():
            score_sums[prompt_idx][layer_pos][lang] = score_sums[prompt_idx][layer_pos].get(lang, 0.0) + float(score)
        sample_records.append(
            {
                "prompt_idx": prompt_idx,
                "layer": int(layer_idx),
                "sample_idx": sample_idx,
                "decode_mode": "rollout_sample",
                "decoded_text": decoded_text,
                "predicted_lang": max(scores, key=scores.get) if scores else None,
            }
        )


################################################################################
# Rollout scoring (teacher-forced-argmax and sampling)
################################################################################

def teacher_forced_argmax_decode_then_classify(
    prompts: Sequence[str],
    full_latents_list,
    layer_indices: Sequence[int],
    token_agg: str,
    rollout_k: int,
    rollout_word_cnt: int,
    model,
    decoding_lens,
    scorer,
    *,
    word_ids_list: Sequence[Sequence[int | None] | None] | None = None,
    show_progress: bool = True,
) -> tuple[dict[str, torch.Tensor], list[list[str]]]:
    """Classify teacher-forced prompt spans using cached full-prompt latents.

    This stays inside the traced prompt sequence, so it reuses `full_latents_list`
    and does not retrace after each predicted token.
    """
    score_grid = [[{} for _ in range(len(layer_indices))] for _ in range(len(prompts))]
    decoded_texts = [["" for _ in range(len(layer_indices))] for _ in range(len(prompts))]
    lang_set = set()
    device = None
    iterator = prompts
    if show_progress:
        iterator = tqdm(prompts, desc="Rollout prompts", unit="prompt")
    for i, _ in enumerate(iterator):
        predicted_token_ids_by_layer = [[] for _ in range(len(layer_indices))]
        prompt_latents = full_latents_list[i]
        prompt_word_ids = None if word_ids_list is None else word_ids_list[i]
        rollout_positions = select_rollout_token_positions(
            prompt_latents.shape[1],
            token_agg,
            rollout_k=rollout_k,
            rollout_word_cnt=rollout_word_cnt,
            word_ids=prompt_word_ids,
        )
        if not rollout_positions:
            continue
        for pos in rollout_positions:
            step_latents = prompt_latents[:, pos, :].unsqueeze(0)
            logits = decoding_lens.project_all_layers(step_latents, layer_indices)
            probs = torch.softmax(logits, dim=-1)[0]
            device = probs.device
            for layer_pos in range(probs.shape[0]):
                token_id = int(probs[layer_pos].argmax(dim=-1).item())
                predicted_token_ids_by_layer[layer_pos].append(token_id)
        for layer_pos, token_ids in enumerate(predicted_token_ids_by_layer):
            text = model.tokenizer.decode(token_ids) if token_ids else ""
            scores = scorer.score_text(text)
            if not scores:
                print(
                    {
                        "event": "empty_lid_scores",
                        "prompt_idx": i,
                        "layer": int(layer_indices[layer_pos]),
                        "decoded_token_ids": token_ids,
                        "decoded_text": text,
                        "decoded_text_repr": repr(text),
                    }
                )
                raise ValueError("decode_then_classify mapping produced empty scores for decoded text.")
            score_grid[i][layer_pos] = scores
            decoded_texts[i][layer_pos] = text
            lang_set.update(scores.keys())
    if not lang_set:
        raise ValueError("No language scores produced by decode_then_classify mapping.")
    if getattr(scorer, "candidate_languages", None):
        langs = sorted(scorer.candidate_languages)
    else:
        langs = sorted(lang_set)
    device = device or torch.device("cpu")
    lang_probs_dec = {
        lang: torch.zeros((len(prompts), len(layer_indices)), device=device)
        for lang in langs
    }
    for i in range(len(prompts)):
        for j in range(len(layer_indices)):
            scores = score_grid[i][j]
            for lang, score in scores.items():
                lang_probs_dec[lang][i, j] = float(score)
    return lang_probs_dec, decoded_texts


def compute_rollout_word_token_budget(
    seq_len: int,
    mode: str,
    *,
    rollout_word_cnt: int,
    word_ids: Sequence[int | None] | None = None,
) -> int:
    """Return the generated-token budget matching the requested prompt words."""
    if rollout_word_cnt < 0:
        raise ValueError("rollout_word_cnt must be non-negative.")
    if rollout_word_cnt == 0:
        return 0
    if word_ids is None:
        raise ValueError("rollout_word_cnt requires word_ids to compute a prompt-word token budget.")

    anchor = select_scoring_token_position(seq_len, mode, word_ids=word_ids)
    future_word_ids: list[int] = []
    seen = set()
    for target_pos in range(anchor + 1, seq_len):
        word_id = word_ids[target_pos]
        if word_id is None or word_id in seen:
            continue
        future_word_ids.append(int(word_id))
        seen.add(int(word_id))
        if len(future_word_ids) >= rollout_word_cnt:
            break
    if not future_word_ids:
        return 1

    selected_word_ids = set(future_word_ids)
    budget = sum(
        1
        for target_pos in range(anchor + 1, seq_len)
        if word_ids[target_pos] is not None and int(word_ids[target_pos]) in selected_word_ids
    )
    return max(1, budget)


def sampled_rollout_decode_then_classify(
    prompts: Sequence[str],
    full_latents_list,
    layer_indices: Sequence[int],
    token_agg: str,
    rollout_k: int,
    rollout_word_cnt: int,
    rollout_top_p: float,
    rollout_num_samples: int,
    model,
    decoding_lens,
    scorer,
    *,
    word_ids_list: Sequence[Sequence[int | None] | None] | None = None,
    show_progress: bool = True,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    """Classify free-running top-p rollouts from cached prompt prefixes.

    Prompts are left-padded into micro-batches, each prompt is expanded across
    `rollout_num_samples`, and sampled LID score distributions are averaged
    uniformly per prompt/layer.
    """
    if rollout_num_samples <= 0:
        raise ValueError("decoding_rollout_num_samples must be positive.")

    tokenizer = model.tokenizer
    causal_lm = _get_hf_causal_lm(model)
    model_input_device = next(causal_lm.parameters()).device
    prompt_batch_size = int(getattr(model, "sampled_rollout_prompt_batch_size", 0) or 0)
    score_sums = [[{} for _ in range(len(layer_indices))] for _ in range(len(prompts))]
    sample_records: list[dict[str, object]] = []
    observed_langs = set()
    device = None

    prompt_infos = []
    for prompt_idx, prompt in enumerate(prompts):
        prompt_word_ids = None if word_ids_list is None else word_ids_list[prompt_idx]
        prompt_latents = full_latents_list[prompt_idx]
        anchor = select_scoring_token_position(prompt_latents.shape[1], token_agg, word_ids=prompt_word_ids)
        target_generated_tokens = (
            compute_rollout_word_token_budget(
                prompt_latents.shape[1],
                token_agg,
                rollout_word_cnt=rollout_word_cnt,
                word_ids=prompt_word_ids,
            )
            if rollout_word_cnt > 0
            else rollout_k + 1
        )
        base_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if anchor >= len(base_ids):
            raise ValueError(
                f"Anchor position {anchor} out of range for prompt tokenization length {len(base_ids)}."
            )
        prompt_infos.append(
            {
                "prompt_idx": prompt_idx,
                "prefix_ids": list(base_ids[: anchor + 1]),
                "target_generated_tokens": int(target_generated_tokens),
            }
        )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must expose a pad or eos token id for padded sampled rollout batching.")

    unique_budgets = {int(info["target_generated_tokens"]) for info in prompt_infos}
    if len(unique_budgets) != 1:
        raise ValueError(
            "sampled rollout prompt batching assumes a shared generation budget; "
            "run one rollout budget setting at a time."
        )
    target_generated_tokens = next(iter(unique_budgets))
    if prompt_batch_size <= 0:
        prompt_batch_size = len(prompt_infos)

    progress = None
    if show_progress:
        progress = tqdm(
            total=len(prompts) * len(layer_indices),
            desc="Sampled rollout prompt-layers",
            unit="prompt-layer",
        )

    for prompt_bucket in _iter_batches(prompt_infos, prompt_batch_size):
        prompt_indices = [int(info["prompt_idx"]) for info in prompt_bucket]
        prompt_count = len(prompt_bucket)
        row_count = prompt_count * rollout_num_samples
        prefix_input_ids, prefix_attention_mask = _left_pad_token_id_rows(
            [info["prefix_ids"] for info in prompt_bucket],
            pad_token_id=pad_token_id,
            device=model_input_device,
        )
        with torch.no_grad():
            prefix_outputs = causal_lm(
                input_ids=prefix_input_ids,
                attention_mask=prefix_attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        for layer_pos, layer_idx in enumerate(layer_indices):
            generated_ids_by_row: list[list[int]] = [[] for _ in range(row_count)]

            step_latents = get_hf_layer_hidden_state(
                prefix_outputs.hidden_states,
                int(layer_idx),
            )[:, -1, :].unsqueeze(1)
            logits = _project_hf_cached_layer_logits(
                step_latents=step_latents,
                layer_idx=int(layer_idx),
                causal_lm=causal_lm,
                decoding_lens=decoding_lens,
            )
            step_probs = torch.softmax(logits[:, 0, :], dim=-1)
            device = step_probs.device
            expanded_step_probs = step_probs.repeat_interleave(rollout_num_samples, dim=0)
            next_token_ids_tensor = sample_top_p_token_ids(expanded_step_probs, rollout_top_p)
            next_token_ids = [int(token_id) for token_id in next_token_ids_tensor.tolist()]
            for row_idx, token_id in enumerate(next_token_ids):
                generated_ids_by_row[row_idx].append(int(token_id))

            if target_generated_tokens > 1:
                current_input_ids = torch.tensor(next_token_ids, dtype=torch.long, device=model_input_device).unsqueeze(1)
                attention_mask = prefix_attention_mask.repeat_interleave(rollout_num_samples, dim=0)
                past_key_values = _expand_past_key_values(prefix_outputs.past_key_values, rollout_num_samples)
                for _ in range(1, target_generated_tokens):
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones((row_count, 1), dtype=attention_mask.dtype, device=attention_mask.device),
                        ],
                        dim=1,
                    )
                    with torch.no_grad():
                        step_outputs = causal_lm(
                            input_ids=current_input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                    past_key_values = step_outputs.past_key_values
                    step_latents = get_hf_layer_hidden_state(
                        step_outputs.hidden_states,
                        int(layer_idx),
                    )[:, -1, :].unsqueeze(1)
                    logits = _project_hf_cached_layer_logits(
                        step_latents=step_latents,
                        layer_idx=int(layer_idx),
                        causal_lm=causal_lm,
                        decoding_lens=decoding_lens,
                    )
                    step_probs = torch.softmax(logits[:, 0, :], dim=-1)
                    next_token_ids_tensor = sample_top_p_token_ids(step_probs, rollout_top_p)
                    current_input_ids = next_token_ids_tensor.to(device=model_input_device, dtype=torch.long).unsqueeze(1)
                    for row_idx, token_id in enumerate(current_input_ids[:, 0].tolist()):
                        generated_ids_by_row[row_idx].append(int(token_id))

            _accumulate_sampled_rollout_scores(
                prompt_indices=prompt_indices,
                layer_pos=layer_pos,
                layer_idx=int(layer_idx),
                generated_ids_by_row=generated_ids_by_row,
                rollout_num_samples=rollout_num_samples,
                tokenizer=tokenizer,
                scorer=scorer,
                score_sums=score_sums,
                sample_records=sample_records,
                observed_langs=observed_langs,
            )
            if progress is not None:
                progress.update(len(prompt_bucket))

    if progress is not None:
        progress.close()

    langs = sorted(getattr(scorer, "candidate_languages", None) or observed_langs)
    if not langs:
        raise ValueError("No language scores produced by sampled rollout mapping.")
    device = device or torch.device("cpu")
    lang_probs_dec = {
        lang: torch.zeros((len(prompts), len(layer_indices)), device=device)
        for lang in langs
    }
    for prompt_idx in range(len(prompts)):
        for layer_pos in range(len(layer_indices)):
            scores = score_sums[prompt_idx][layer_pos]
            for lang, score_sum in scores.items():
                lang_probs_dec[lang][prompt_idx, layer_pos] = float(score_sum) / float(rollout_num_samples)
    return lang_probs_dec, sample_records


################################################################################
# Single-position scoring
################################################################################

def select_decoding_latents(
    full_latents_list,
    mode: str,
    *,
    word_ids_list: Sequence[Sequence[int | None] | None] | None = None,
) -> torch.Tensor:
    """Aggregate sequence latents into one position per prompt and layer."""
    if mode == "last_token":
        return gather_fixed_position_latents(full_latents_list, -1)
    if mode in {"frac_50", "frac_75"}:
        selected = []
        for prompt_idx, prompt_latents in enumerate(full_latents_list):
            layer_selected = []
            prompt_word_ids = None if word_ids_list is None else word_ids_list[prompt_idx]
            for layer_latents in prompt_latents:
                pos = select_fractional_token_position(
                    layer_latents.shape[0],
                    mode,
                    word_ids=prompt_word_ids,
                )
                layer_selected.append(layer_latents[pos])
            selected.append(torch.stack(layer_selected, dim=0))
        return torch.stack(selected, dim=0)
    if mode == "all_mean":
        selected = []
        for prompt_latents in full_latents_list:
            layer_selected = [layer_latents.mean(dim=0) for layer_latents in prompt_latents]
            selected.append(torch.stack(layer_selected, dim=0))
        return torch.stack(selected, dim=0)
    raise ValueError(f"Unsupported token aggregation mode: {mode}")


def score_singlepos_latents(
    latents: torch.Tensor,
    layer_indices: Sequence[int],
    decoding_lens,
    scorer,
    *,
    decode_mode: str,
    topk: int,
    desc: str,
    chunk_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Project latents to vocab in prompt chunks and score decode-then-classify outputs."""
    n_prompts, n_layers = latents.shape[:2]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    score_rows = [[{} for _ in range(n_layers)] for _ in range(n_prompts)]
    lang_set = set()
    pair_iter = tqdm(
        total=n_prompts * n_layers,
        desc=desc,
        unit="prompt-layer",
    )
    for start in range(0, n_prompts, chunk_size):
        end = min(start + chunk_size, n_prompts)
        logits = decoding_lens.project_all_layers(latents[start:end], layer_indices)
        probs = torch.softmax(logits, dim=-1)
        for local_i in range(probs.shape[0]):
            i = start + local_i
            for j in range(probs.shape[1]):
                label, scores = scorer.classify_from_probs(
                    probs[local_i, j],
                    mode=decode_mode,
                    topk=topk,
                )
                if decode_mode == "topk_weighted":
                    score_rows[i][j] = dict(scores)
                    lang_set.update(scores.keys())
                else:
                    score_rows[i][j] = {label: 1.0} if label else {}
                    if label:
                        lang_set.add(label)
                pair_iter.update(1)
        del logits
        del probs
    pair_iter.close()

    if not lang_set:
        raise ValueError("No labels produced by decode_then_classify mapping.")

    if getattr(scorer, "candidate_languages", None):
        langs = sorted(scorer.candidate_languages)
    else:
        langs = sorted(lang_set)
    lang_probs_dec = {
        lang: torch.zeros((n_prompts, n_layers), device=latents.device)
        for lang in langs
    }
    for i in range(n_prompts):
        for j in range(n_layers):
            for lang, val in score_rows[i][j].items():
                if lang in lang_probs_dec:
                    lang_probs_dec[lang][i, j] = float(val)
    return lang_probs_dec


################################################################################
# Streamed entry points
################################################################################

def decode_then_classify_rollout_streamed(
    *,
    prompts: Sequence[str],
    layer_indices: Sequence[int],
    prompt_token_lengths: Sequence[int],
    trace_batch_size: int,
    decode_mode: str,
    token_agg: str,
    rollout_k: int,
    rollout_word_cnt: int,
    rollout_top_p: float,
    rollout_num_samples: int,
    model,
    decoding_lens,
    scorer,
    word_ids_list,
    checkpoint_id: str,
    decoding_lens_name: str,
):
    chunks = []
    decoded_samples_all: list[dict[str, object]] = []
    for start, end, full_latents_batch in stream_sequence_latent_batches(
        model=model,
        prompts=prompts,
        layer_indices=layer_indices,
        prompt_token_lengths=prompt_token_lengths,
        trace_batch_size=trace_batch_size,
        desc=f"Streaming decode traces ({decode_mode})",
    ):
        prompt_batch = prompts[start:end]
        word_ids_batch = None if word_ids_list is None else word_ids_list[start:end]
        if decode_mode == "rollout_argmax":
            chunk_probs, decoded_texts = teacher_forced_argmax_decode_then_classify(
                prompts=prompt_batch,
                full_latents_list=full_latents_batch,
                layer_indices=layer_indices,
                token_agg=token_agg,
                rollout_k=rollout_k,
                rollout_word_cnt=rollout_word_cnt,
                model=model,
                decoding_lens=decoding_lens,
                scorer=scorer,
                word_ids_list=word_ids_batch,
                show_progress=False,
            )
            for local_i in range(len(prompt_batch)):
                global_i = start + local_i
                for layer_pos, layer_idx in enumerate(layer_indices):
                    decoded_samples_all.append(
                        {
                            "checkpoint_id": checkpoint_id,
                            "prompt_idx": global_i,
                            "layer": int(layer_idx),
                            "decode_mode": "rollout_argmax",
                            "decoding_lens": decoding_lens_name,
                            "decoded_text": decoded_texts[local_i][layer_pos],
                            "predicted_lang": (
                                max(chunk_probs, key=lambda lang: float(chunk_probs[lang][local_i, layer_pos].item()))
                                if chunk_probs else None
                            ),
                        }
                    )
        elif decode_mode == "rollout_sample":
            chunk_probs, decoded_samples = sampled_rollout_decode_then_classify(
                prompts=prompt_batch,
                full_latents_list=full_latents_batch,
                layer_indices=layer_indices,
                token_agg=token_agg,
                rollout_k=rollout_k,
                rollout_word_cnt=rollout_word_cnt,
                rollout_top_p=rollout_top_p,
                rollout_num_samples=rollout_num_samples,
                model=model,
                decoding_lens=decoding_lens,
                scorer=scorer,
                word_ids_list=word_ids_batch,
                show_progress=False,
            )
            decoded_samples_all.extend(
                {
                    "checkpoint_id": checkpoint_id,
                    "decoding_lens": decoding_lens_name,
                    **row,
                    "prompt_idx": start + int(row["prompt_idx"]),
                }
                for row in decoded_samples
            )
        else:
            raise ValueError(f"Unsupported streamed rollout decode mode: {decode_mode}")
        chunks.append({lang: tensor.detach().cpu() for lang, tensor in chunk_probs.items()})
    return concat_langdist_chunks(chunks), decoded_samples_all


def decode_then_classify_singlepos_streamed(
    *,
    prompts: Sequence[str],
    layer_indices: Sequence[int],
    prompt_token_lengths: Sequence[int],
    trace_batch_size: int,
    token_agg: str,
    model,
    decoding_lens,
    scorer,
    word_ids_list,
    decode_mode: str,
    topk: int,
    desc: str,
):
    chunks = []
    for start, end, full_latents_batch in stream_sequence_latent_batches(
        model=model,
        prompts=prompts,
        layer_indices=layer_indices,
        prompt_token_lengths=prompt_token_lengths,
        trace_batch_size=trace_batch_size,
        desc=desc,
    ):
        latents_batch = select_decoding_latents(
            full_latents_batch,
            token_agg,
            word_ids_list=None if word_ids_list is None else word_ids_list[start:end],
        )
        chunk_probs = score_singlepos_latents(
            latents=latents_batch,
            layer_indices=layer_indices,
            decoding_lens=decoding_lens,
            scorer=scorer,
            decode_mode=decode_mode,
            topk=topk,
            desc=f"Decode/LID {decode_mode} prompts {start}:{end}",
        )
        chunks.append({lang: tensor.detach().cpu() for lang, tensor in chunk_probs.items()})
    return concat_langdist_chunks(chunks)
