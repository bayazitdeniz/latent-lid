import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from lenses import (
    apply_unembed,
    build_decoding_lens,
    hash_training_config,
    load_tuned_lens_snapshot,
    next_tuned_lens_run_dir,
    prepare_unembed_info,
    tuned_lens_layer_filename,
)


class _ToyModel(torch.nn.Module):
    def __init__(self, num_hidden_layers: int = 2):
        super().__init__()
        self.config = type("Config", (), {"num_hidden_layers": int(num_hidden_layers)})()
        self.lm_head = torch.nn.Linear(2, 3, bias=True)
        self.final_layer_norm = torch.nn.LayerNorm(2)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                    ]
                )
            )
            self.lm_head.bias.zero_()
            self.final_layer_norm.weight.copy_(torch.tensor([2.0, 3.0]))
            self.final_layer_norm.bias.zero_()


class ToyNormHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.final_layer_norm = torch.nn.LayerNorm(2)
        self.lm_head = torch.nn.Linear(2, 3, bias=True)
        with torch.no_grad():
            self.final_layer_norm.weight.copy_(torch.tensor([2.0, 3.0]))
            self.final_layer_norm.bias.zero_()
            self.lm_head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.5],
                        [-0.25, 2.0],
                        [1.0, -1.0],
                    ]
                )
            )
            self.lm_head.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))


class UnembedProjectionTests(unittest.TestCase):
    def test_apply_unembed_with_final_norm_matches_model_norm_then_raw_lm_head(self):
        model = ToyNormHeadModel()
        info = prepare_unembed_info(model, device="cpu")
        hidden = torch.tensor([[1.0, 3.0], [4.0, -2.0]], dtype=torch.float32)

        logits = apply_unembed(hidden, info, apply_final_norm=True)
        expected = model.lm_head(model.final_layer_norm(hidden))

        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))
        old_folded_weight = model.lm_head.weight * model.final_layer_norm.weight.unsqueeze(0)
        old_double_norm_logits = model.final_layer_norm(hidden) @ old_folded_weight.T + model.lm_head.bias
        self.assertFalse(torch.allclose(logits, old_double_norm_logits, atol=1e-6))

    def test_apply_unembed_without_final_norm_uses_raw_lm_head(self):
        model = ToyNormHeadModel()
        info = prepare_unembed_info(model, device="cpu")
        hidden = torch.tensor([[1.0, 3.0], [4.0, -2.0]], dtype=torch.float32)

        logits = apply_unembed(hidden, info, apply_final_norm=False)
        expected = hidden @ model.lm_head.weight.T + model.lm_head.bias

        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))


class TunedLensTests(unittest.TestCase):
    def test_training_config_hash_ignores_key_order(self):
        self.assertEqual(
            hash_training_config({"model": "gpt2", "seed": 42}),
            hash_training_config({"seed": 42, "model": "gpt2"}),
        )

    def test_next_run_dir_adds_suffix_for_timestamp_collision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            with mock.patch("lenses.datetime") as datetime_mock:
                datetime_mock.now.return_value.strftime.return_value = "20260102_030405"
                first = next_tuned_lens_run_dir(base_dir)
                first.mkdir()
                second = next_tuned_lens_run_dir(base_dir)

        self.assertEqual(first.name, "20260102_030405")
        self.assertEqual(second.name, "20260102_030405_01")

    def test_build_decoding_lens_raw_uses_runtime_final_norm_and_raw_lm_head(self):
        model = _ToyModel()
        lens, meta = build_decoding_lens(
            model,
            decoding_lens="raw_logitlens",
            apply_final_norm=True,
            device="cpu",
        )

        hidden = torch.tensor([[[1.0, 3.0]], [[4.0, -2.0]]], dtype=torch.float32)
        logits = lens.project_all_layers(hidden, [0])
        expected = model.lm_head(model.final_layer_norm(hidden[:, 0, :]))

        self.assertTrue(torch.allclose(logits[:, 0, :], expected, atol=1e-6))
        self.assertEqual(meta["decoding_lens"], "raw_logitlens")

    def test_load_tuned_lens_snapshot_reads_config_and_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "training_config_hash": "abc123",
                        "validation_kl_by_layer": {"0": 0.1},
                        "validation_ce_by_layer": {"0": 1.5},
                    }
                ),
                encoding="utf-8",
            )
            torch.save(
                {
                    "A": torch.eye(2),
                    "b": torch.zeros(2),
                    "layer_idx": 0,
                    "hidden_size": 2,
                    "training_config_hash": "abc123",
                    "validation_kl": 0.1,
                    "validation_ce": 1.5,
                },
                root / tuned_lens_layer_filename(0),
            )

            loaded = load_tuned_lens_snapshot(root)

            self.assertEqual(loaded.training_config_hash, "abc123")
            self.assertIn(0, loaded.translators)
            self.assertEqual(loaded.validation_kl_by_layer[0], 0.1)
            self.assertEqual(loaded.validation_ce_by_layer[0], 1.5)

    def test_load_tuned_lens_snapshot_selects_latest_complete_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for run_name, config_hash in (
                ("20260101_000000", "older"),
                ("20260102_000000", "newer"),
            ):
                snapshot_dir = root / run_name
                snapshot_dir.mkdir()
                (snapshot_dir / "config.json").write_text(
                    json.dumps({"layers": [0], "training_config_hash": config_hash}),
                    encoding="utf-8",
                )
                torch.save(
                    {
                        "A": torch.eye(2),
                        "b": torch.zeros(2),
                        "layer_idx": 0,
                        "hidden_size": 2,
                        "training_config_hash": config_hash,
                    },
                    snapshot_dir / tuned_lens_layer_filename(0),
                )
            incomplete_dir = root / "20260103_000000"
            incomplete_dir.mkdir()
            (incomplete_dir / "config.json").write_text(
                json.dumps({"layers": [0]}),
                encoding="utf-8",
            )

            loaded = load_tuned_lens_snapshot(root)

        self.assertEqual(Path(loaded.snapshot_dir).name, "20260102_000000")
        self.assertEqual(loaded.training_config_hash, "newer")

    def test_build_decoding_lens_tuned_uses_layer_specific_translators(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "training_config_hash": "hash1",
                        "validation_kl_by_layer": {"0": 0.1, "1": 0.2},
                        "validation_ce_by_layer": {"0": 1.0, "1": 2.0},
                    }
                ),
                encoding="utf-8",
            )
            torch.save(
                {
                    "A": torch.eye(2),
                    "b": torch.zeros(2),
                    "layer_idx": 0,
                    "hidden_size": 2,
                    "training_config_hash": "hash1",
                    "validation_kl": 0.1,
                    "validation_ce": 1.0,
                },
                root / tuned_lens_layer_filename(0),
            )
            torch.save(
                {
                    "A": 2.0 * torch.eye(2),
                    "b": torch.zeros(2),
                    "layer_idx": 1,
                    "hidden_size": 2,
                    "training_config_hash": "hash1",
                    "validation_kl": 0.2,
                    "validation_ce": 2.0,
                },
                root / tuned_lens_layer_filename(1),
            )

            model = _ToyModel(num_hidden_layers=3)
            lens, meta = build_decoding_lens(
                model,
                decoding_lens="tuned_lens",
                apply_final_norm=False,
                tuned_lens_dir=root,
                device="cpu",
            )

            hidden = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]], dtype=torch.float32)
            logits = lens.project_all_layers(hidden, [0, 1])

            self.assertEqual(tuple(logits.shape), (1, 2, 3))
            self.assertFalse(torch.allclose(logits[:, 0, :], logits[:, 1, :]))
            self.assertEqual(meta["decoding_lens"], "tuned_lens")
            self.assertEqual(meta["tuned_lens_training_config_hash"], "hash1")

    def test_build_decoding_lens_tuned_uses_raw_projection_for_final_layer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "training_config_hash": "hash1",
                        "validation_kl_by_layer": {"0": 0.1},
                        "validation_ce_by_layer": {"0": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            torch.save(
                {
                    "A": 2.0 * torch.eye(2),
                    "b": torch.zeros(2),
                    "layer_idx": 0,
                    "hidden_size": 2,
                    "training_config_hash": "hash1",
                    "validation_kl": 0.1,
                    "validation_ce": 1.0,
                },
                root / tuned_lens_layer_filename(0),
            )

            model = _ToyModel()
            lens, _ = build_decoding_lens(
                model,
                decoding_lens="tuned_lens",
                apply_final_norm=False,
                tuned_lens_dir=root,
                device="cpu",
            )

            hidden = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]], dtype=torch.float32)
            logits = lens.project_all_layers(hidden, [0, 1])
            expected_layer0 = model.lm_head(2.0 * hidden[:, 0, :])
            expected_final = model.lm_head(hidden[:, 1, :])

            self.assertTrue(torch.allclose(logits[:, 0, :], expected_layer0, atol=1e-6))
            self.assertTrue(torch.allclose(logits[:, 1, :], expected_final, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
