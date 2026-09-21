import unittest

import torch
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from latents import (
    _materialize_first_tensor,
    collect_position_latents,
    collect_sequence_latents,
    get_hf_layer_hidden_state,
    mean_pool_latents_by_word,
)


class HFLayerHiddenStateTests(unittest.TestCase):
    def test_get_hf_layer_hidden_state_skips_embedding_output(self):
        embedding_output = torch.tensor([0.0])
        layer_0_output = torch.tensor([1.0])
        layer_1_output = torch.tensor([2.0])

        hidden = get_hf_layer_hidden_state(
            (embedding_output, layer_0_output, layer_1_output),
            layer_idx=1,
        )

        self.assertTrue(torch.equal(hidden, layer_1_output))


class LatentAggregationTests(unittest.TestCase):
    def test_mean_pool_latents_by_word(self):
        latents = torch.tensor(
            [
                [1.0, 1.0],
                [3.0, 1.0],
                [2.0, 2.0],
                [9.0, 9.0],
            ]
        )
        word_ids = [0, 0, 1, None]
        aggregated, used = mean_pool_latents_by_word(latents, word_ids)
        expected = torch.tensor([[2.0, 1.0], [2.0, 2.0]])
        self.assertTrue(used)
        self.assertTrue(torch.allclose(aggregated, expected))

    def test_mean_pool_latents_without_word_ids(self):
        latents = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        aggregated, used = mean_pool_latents_by_word(latents, None)
        self.assertFalse(used)
        self.assertTrue(torch.allclose(aggregated, latents))


class LatentCollectionTests(unittest.TestCase):
    def test_collect_position_latents_selects_last_token_after_left_padding(self):
        class FakeSaved:
            def __init__(self, value):
                self.value = value

        class FakeOutput:
            def __init__(self, model, layer_idx):
                self.model = model
                self.layer_idx = layer_idx

            def save(self):
                seq_len = self.model.prompt_lengths[self.model.current_prompt]
                max_len = self.model.batch_max_len
                pad_len = max_len - seq_len
                values = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1)
                values = values + (100 * self.layer_idx) + self.model.prompt_offsets[self.model.current_prompt]
                if pad_len > 0:
                    pad = torch.full((1, pad_len, 1), -999.0)
                    values = torch.cat([pad, values], dim=1)
                return FakeSaved(values)

        class FakeLayer:
            def __init__(self, model, layer_idx):
                self.output = FakeOutput(model, layer_idx)

        class FakeInvoke:
            def __init__(self, model, prompt):
                self.model = model
                self.prompt = prompt

            def __enter__(self):
                self.prev_prompt = self.model.current_prompt
                self.model.current_prompt = self.prompt
                return self

            def __exit__(self, exc_type, exc, tb):
                self.model.current_prompt = self.prev_prompt
                return False

        class FakeTracer:
            def __init__(self, model):
                self.model = model

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def invoke(self, prompt):
                return FakeInvoke(self.model, prompt)

        class FakeModel:
            def __init__(self):
                self.prompt_lengths = {
                    "short": 2,
                    "longer": 5,
                }
                self.prompt_offsets = {
                    "short": 10,
                    "longer": 20,
                }
                self.batch_max_len = 5
                self.current_prompt = None
                self.transformer = type("Transformer", (), {})()
                self.transformer.h = [FakeLayer(self, 0), FakeLayer(self, 1)]

            def trace(self):
                return FakeTracer(self)

        model = FakeModel()
        latents, layer_indices = collect_position_latents(
            model=model,
            prompts=["short", "longer"],
            layer_indices=[0, 1],
            position=-1,
            sequence_lengths=[2, 5],
            trace_batch_size=2,
            show_progress=False,
        )

        self.assertEqual(layer_indices, [0, 1])
        self.assertEqual(tuple(latents.shape), (2, 2, 1))
        expected = torch.tensor(
            [
                [[11.0], [111.0]],
                [[24.0], [124.0]],
            ]
        )
        self.assertTrue(torch.equal(latents, expected))

        anchored_latents, anchored_layer_indices = collect_position_latents(
            model=model,
            prompts=["short", "longer"],
            layer_indices=[0, 1],
            position=-1,
            positions_by_prompt=[0, 2],
            sequence_lengths=[2, 5],
            trace_batch_size=2,
            show_progress=False,
        )
        self.assertEqual(anchored_layer_indices, [0, 1])
        anchored_expected = torch.tensor(
            [
                [[10.0], [110.0]],
                [[22.0], [122.0]],
            ]
        )
        self.assertTrue(torch.equal(anchored_latents, anchored_expected))

    def test_collect_sequence_latents_drops_left_padding(self):
        class FakeSaved:
            def __init__(self, value):
                self.value = value

        class FakeOutput:
            def __init__(self, model, layer_idx):
                self.model = model
                self.layer_idx = layer_idx

            def save(self):
                seq_len = self.model.prompt_lengths[self.model.current_prompt]
                max_len = self.model.batch_max_len
                pad_len = max_len - seq_len
                values = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1)
                values = values + (100 * self.layer_idx) + self.model.prompt_offsets[self.model.current_prompt]
                if pad_len > 0:
                    pad = torch.full((1, pad_len, 1), -999.0)
                    values = torch.cat([pad, values], dim=1)
                return FakeSaved(values)

        class FakeLayer:
            def __init__(self, model, layer_idx):
                self.output = FakeOutput(model, layer_idx)

        class FakeInvoke:
            def __init__(self, model, prompt):
                self.model = model
                self.prompt = prompt

            def __enter__(self):
                self.prev_prompt = self.model.current_prompt
                self.model.current_prompt = self.prompt
                return self

            def __exit__(self, exc_type, exc, tb):
                self.model.current_prompt = self.prev_prompt
                return False

        class FakeTracer:
            def __init__(self, model):
                self.model = model

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def invoke(self, prompt):
                return FakeInvoke(self.model, prompt)

        class FakeModel:
            def __init__(self):
                self.prompt_lengths = {
                    "short": 2,
                    "longer": 5,
                }
                self.prompt_offsets = {
                    "short": 10,
                    "longer": 20,
                }
                self.batch_max_len = 5
                self.current_prompt = None
                self.transformer = type("Transformer", (), {})()
                self.transformer.h = [FakeLayer(self, 0), FakeLayer(self, 1)]

            def trace(self):
                return FakeTracer(self)

        model = FakeModel()
        full_latents, layer_indices = collect_sequence_latents(
            model=model,
            prompts=["short", "longer"],
            layer_indices=[0, 1],
            sequence_lengths=[2, 5],
            trace_batch_size=2,
            show_progress=False,
        )

        self.assertEqual(layer_indices, [0, 1])
        self.assertEqual(tuple(full_latents[0].shape), (2, 2, 1))
        self.assertEqual(tuple(full_latents[1].shape), (2, 5, 1))
        expected_short = torch.tensor([[[10.0], [11.0]], [[110.0], [111.0]]])
        expected_long = torch.tensor(
            [
                [[20.0], [21.0], [22.0], [23.0], [24.0]],
                [[120.0], [121.0], [122.0], [123.0], [124.0]],
            ]
        )
        self.assertTrue(torch.equal(full_latents[0], expected_short))
        self.assertTrue(torch.equal(full_latents[1], expected_long))

    def test_collect_sequence_latents_preserves_per_prompt_sequence_lengths(self):
        class FakeSaved:
            def __init__(self, value):
                self.value = value

        class FakeOutput:
            def __init__(self, model, layer_idx):
                self.model = model
                self.layer_idx = layer_idx

            def save(self):
                seq_len = self.model.prompt_lengths[self.model.current_prompt]
                hidden = 3
                tensor = torch.full(
                    (1, seq_len, hidden),
                    fill_value=float(self.layer_idx + seq_len),
                )
                return FakeSaved(tensor)

        class FakeLayer:
            def __init__(self, model, layer_idx):
                self.output = FakeOutput(model, layer_idx)

        class FakeInvoke:
            def __init__(self, model, prompt):
                self.model = model
                self.prompt = prompt

            def __enter__(self):
                self.prev_prompt = self.model.current_prompt
                self.model.current_prompt = self.prompt
                return self

            def __exit__(self, exc_type, exc, tb):
                self.model.current_prompt = self.prev_prompt
                return False

        class FakeTracer:
            def __init__(self, model):
                self.model = model

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def invoke(self, prompt):
                return FakeInvoke(self.model, prompt)

        class FakeModel:
            def __init__(self):
                self.prompt_lengths = {
                    "short": 2,
                    "longer": 5,
                }
                self.current_prompt = None
                self.transformer = type("Transformer", (), {})()
                self.transformer.h = [FakeLayer(self, 0), FakeLayer(self, 1)]

            def trace(self):
                return FakeTracer(self)

        model = FakeModel()
        full_latents, layer_indices = collect_sequence_latents(
            model=model,
            prompts=["short", "longer"],
            layer_indices=[0, 1],
            trace_batch_size=2,
            show_progress=False,
        )

        self.assertEqual(layer_indices, [0, 1])
        self.assertEqual(len(full_latents), 2)
        self.assertEqual(tuple(full_latents[0].shape), (2, 2, 3))
        self.assertEqual(tuple(full_latents[1].shape), (2, 5, 3))
        self.assertNotEqual(full_latents[0].shape[1], full_latents[1].shape[1])


class NNsightBatchPositionRegressionTests(unittest.TestCase):
    MODEL_NAME = "gpt2"

    def _load_local_tiny_model(self):
        try:
            model = LanguageModel(
                self.MODEL_NAME,
                local_files_only=True,
                device_map="cpu",
                dispatch=True,
            )
            tokenizer = model.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as exc:
            self.skipTest(f"{self.MODEL_NAME!r} is not available locally for nnsight regression test: {exc}")
        model.eval()
        return model, tokenizer

    def _batched_short_prompt_layer_output(self, model, prompts):
        layer = model.transformer.h[0]
        with model.trace() as tracer:
            with tracer.invoke(prompts[0]):
                layer.output.save()
            with tracer.invoke(prompts[1]):
                short_saved = layer.output.save()
        tensor = _materialize_first_tensor(short_saved)
        if tensor.dim() == 3:
            tensor = tensor[0]
        return tensor

    def test_batched_position_latents_use_nnsight_padding_coordinates(self):
        model, tokenizer = self._load_local_tiny_model()
        prompts = [
            "The capital of France is Paris, and the capital of Germany is",
            "Hello",
        ]
        sequence_lengths = [
            len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            for prompt in prompts
        ]
        self.assertGreater(sequence_lengths[0], sequence_lengths[1])

        raw_short = self._batched_short_prompt_layer_output(model, prompts)
        expected = raw_short[-1] if tokenizer.padding_side == "left" else raw_short[sequence_lengths[1] - 1]
        batched_latents, batched_layer_indices = collect_position_latents(
            model=model,
            prompts=prompts,
            layer_indices=[0],
            position=-1,
            sequence_lengths=sequence_lengths,
            trace_batch_size=2,
            show_progress=False,
        )

        self.assertEqual(batched_layer_indices, [0])
        self.assertEqual(tuple(batched_latents.shape[:2]), (2, 1))
        self.assertTrue(torch.allclose(batched_latents[1, 0].float(), expected.float(), atol=1e-3, rtol=1e-3))

    def test_batched_sequence_latents_use_nnsight_padding_coordinates(self):
        model, tokenizer = self._load_local_tiny_model()
        prompts = [
            "The capital of France is Paris, and the capital of Germany is",
            "Hello",
        ]
        sequence_lengths = [
            len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            for prompt in prompts
        ]
        self.assertGreater(sequence_lengths[0], sequence_lengths[1])

        raw_short = self._batched_short_prompt_layer_output(model, prompts)
        expected = (
            raw_short[-sequence_lengths[1]:]
            if tokenizer.padding_side == "left"
            else raw_short[: sequence_lengths[1]]
        )
        batched_latents, batched_layer_indices = collect_sequence_latents(
            model=model,
            prompts=prompts,
            layer_indices=[0],
            sequence_lengths=sequence_lengths,
            trace_batch_size=2,
            show_progress=False,
        )

        self.assertEqual(batched_layer_indices, [0])
        self.assertEqual(tuple(batched_latents[1].shape), (1, sequence_lengths[1], raw_short.shape[-1]))
        self.assertTrue(torch.allclose(batched_latents[1][0].float(), expected.float(), atol=1e-3, rtol=1e-3))


@unittest.skipUnless(torch.cuda.is_available(), "Needs CUDA to keep the local equivalence check tractable.")
class NNsightHFEquivalenceTests(unittest.TestCase):
    def test_collect_sequence_latents_matches_hf_hidden_states(self):
        prompt = "The weather is nice today."

        nnsight_model = LanguageModel("meta-llama/Llama-2-7b-hf", device_map="cuda", dispatch=True)
        nnsight_model.eval()
        full_latents, _ = collect_sequence_latents(
            model=nnsight_model,
            prompts=[prompt],
            layer_indices=[0],
            show_progress=False,
        )

        hf_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        hf_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            local_files_only=True,
            use_fast=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
        with torch.no_grad():
            outputs = hf_model(
                **encoded,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )

        nnsight_hidden = full_latents[0][0].detach().float().cpu()
        hf_hidden = outputs.hidden_states[1][0].detach().float().cpu()
        self.assertEqual(tuple(nnsight_hidden.shape), tuple(hf_hidden.shape))
        self.assertTrue(torch.allclose(nnsight_hidden, hf_hidden, atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
