import tempfile
import unittest
from pathlib import Path

from datasets import Dataset, DatasetDict
import torch
import torch.nn.functional as F

from fit_tuned_lens import (
    _build_balanced_train_rows,
    _build_padded_token_batch,
    _compute_periodic_steps,
    _rows_contain_text,
    _infer_model_max_length,
    _iter_layer_chunks,
    _load_fineweb_split_rows,
    _masked_kl_and_ce,
    _resolve_fineweb_window_cache_dir,
    _resolve_trainable_layers,
    _select_balanced_rows,
    _window_tokenize_text_rows,
)


class FitTunedLensTests(unittest.TestCase):
    def test_build_balanced_train_rows_uses_rotating_round_robin_order(self):
        rows = [
            {"lang": "en", "text": f"en-{idx}"}
            for idx in range(3)
        ] + [
            {"lang": "fr", "text": f"fr-{idx}"}
            for idx in range(3)
        ]

        balanced, counts, examples_per_lang = _build_balanced_train_rows(
            rows,
            languages=["en", "fr"],
            seed=42,
        )

        self.assertEqual([row["lang"] for row in balanced], ["en", "fr", "fr", "en", "en", "fr"])
        self.assertEqual(counts, {"en": 3, "fr": 3})
        self.assertEqual(examples_per_lang, 3)

    def test_load_tokenized_split_rows_filters_languages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "fineweb_sample"
            dataset = DatasetDict(
                {
                    "train": Dataset.from_list(
                        [
                            {"input_ids": [1, 2, 3], "lang": "en"},
                            {"input_ids": [4, 5, 6], "lang": "fr"},
                            {"input_ids": [7, 8, 9], "lang": "en"},
                        ]
                    ),
                    "validation": Dataset.from_list(
                        [
                            {"input_ids": [10, 11], "lang": "en"},
                            {"input_ids": [12, 13], "lang": "fr"},
                        ]
                    ),
                }
            )
            dataset.save_to_disk(str(dataset_dir))

            rows = _load_fineweb_split_rows(
                dataset_dir=dataset_dir,
                split_name="train",
                languages=["en"],
                max_examples_per_lang=None,
                seed=42,
            )

            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["lang"] == "en" for row in rows))
            self.assertEqual(rows[0]["input_ids"], [1, 2, 3])

    def test_load_tokenized_split_rows_accepts_text_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "fineweb_text_sample"
            dataset = DatasetDict(
                {
                    "train": Dataset.from_list(
                        [
                            {"text": "hello world", "lang": "en"},
                            {"text": "bonjour monde", "lang": "fr"},
                        ]
                    ),
                    "validation": Dataset.from_list(
                        [
                            {"text": "hi there", "lang": "en"},
                            {"text": "salut", "lang": "fr"},
                        ]
                    ),
                }
            )
            dataset.save_to_disk(str(dataset_dir))

            rows = _load_fineweb_split_rows(
                dataset_dir=dataset_dir,
                split_name="train",
                languages=["en", "fr"],
                max_examples_per_lang=None,
                seed=42,
            )

            self.assertEqual(rows[0]["text"], "hello world")
            self.assertTrue(_rows_contain_text(rows))

    def test_build_padded_token_batch_uses_pad_token(self):
        input_ids, attention_mask = _build_padded_token_batch(
            [
                {"input_ids": [11, 12, 13]},
                {"input_ids": [21, 22]},
            ],
            pad_token_id=0,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(input_ids.shape), (2, 3))
        self.assertEqual(tuple(attention_mask.shape), (2, 3))
        self.assertEqual(input_ids.tolist(), [[11, 12, 13], [21, 22, 0]])
        self.assertEqual(attention_mask.tolist(), [[1, 1, 1], [1, 1, 0]])

    def test_infer_model_max_length_prefers_config_positions(self):
        class DummyConfig:
            n_positions = 1024

        class DummyModel:
            config = DummyConfig()

        class DummyTokenizer:
            model_max_length = 2048

        class Wrapped:
            _model = DummyModel()

        self.assertEqual(_infer_model_max_length(Wrapped(), DummyTokenizer()), 1024)

    def test_window_tokenize_text_rows_splits_long_examples(self):
        class DummyTokenizer:
            def __call__(self, text, add_special_tokens=True, truncation=False):
                return {"input_ids": list(range(7))}

        rows = _window_tokenize_text_rows(
            [{"lang": "en", "text": "dummy"}],
            tokenizer=DummyTokenizer(),
            max_length=3,
            desc="test",
        )

        self.assertEqual([row["input_ids"] for row in rows], [[0, 1, 2], [3, 4, 5]])

    def test_window_tokenize_text_rows_keeps_nontrivial_tail_window(self):
        class DummyTokenizer:
            def __call__(self, text, add_special_tokens=True, truncation=False):
                return {"input_ids": list(range(8))}

        rows = _window_tokenize_text_rows(
            [{"lang": "en", "text": "dummy"}],
            tokenizer=DummyTokenizer(),
            max_length=3,
            desc="test",
        )

        self.assertEqual(
            [row["input_ids"] for row in rows],
            [[0, 1, 2], [3, 4, 5], [6, 7]],
        )

    def test_resolve_fineweb_window_cache_dir_includes_model_revision_and_length(self):
        cache_dir = _resolve_fineweb_window_cache_dir(
            dataset_dir=Path("data/fineweb_lens_27lang_55m"),
            model_name="gpt2",
            revision="main",
            max_length=1024,
        )

        self.assertEqual(
            cache_dir,
            Path("data/fineweb_lens_27lang_55m/window_cache/gpt2/main/maxlen_1024"),
        )

    def test_iter_layer_chunks_respects_requested_chunk_size(self):
        chunks = list(_iter_layer_chunks([0, 1, 2, 3, 4], 2))
        self.assertEqual(chunks, [[0, 1], [2, 3], [4]])

    def test_resolve_trainable_tuned_lens_layers_skips_final_layer(self):
        class DummyConfig:
            num_hidden_layers = 4

        class DummyModel:
            config = DummyConfig()

        requested, trainable, final_layer_idx = _resolve_trainable_layers(
            DummyModel(),
            [0, 2, 3],
        )

        self.assertEqual(requested, [0, 2, 3])
        self.assertEqual(trainable, [0, 2])
        self.assertEqual(final_layer_idx, 3)

    def test_resolve_trainable_tuned_lens_layers_rejects_final_only(self):
        class DummyConfig:
            num_hidden_layers = 4

        class DummyModel:
            config = DummyConfig()

        with self.assertRaises(ValueError):
            _resolve_trainable_layers(DummyModel(), [3])

    def test_masked_kl_and_ce_matches_reference_implementation(self):
        student_logits = torch.tensor(
            [
                [
                    [2.0, 0.0, -1.0],
                    [0.5, 1.0, -0.5],
                ]
            ],
            dtype=torch.float32,
        )
        teacher_logits = torch.tensor(
            [
                [
                    [1.5, 0.0, -0.5],
                    [0.0, 1.25, -0.25],
                ]
            ],
            dtype=torch.float32,
        )
        target_ids = torch.tensor([[0, 1]], dtype=torch.long)
        token_mask = torch.tensor([[True, False]])

        loss_kl, loss_ce = _masked_kl_and_ce(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            target_ids=target_ids,
            token_mask=token_mask,
            temperature=1.0,
        )

        flat_mask = token_mask.reshape(-1)
        flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])[flat_mask]
        flat_student = student_logits.reshape(-1, student_logits.shape[-1])[flat_mask]
        flat_targets = target_ids.reshape(-1)[flat_mask]
        teacher_probs = torch.softmax(flat_teacher, dim=-1)
        student_log_probs = torch.log_softmax(flat_student, dim=-1)
        ref_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1).mean()
        ref_ce = F.cross_entropy(flat_student, flat_targets)

        self.assertTrue(torch.allclose(loss_kl, ref_kl, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(loss_ce, ref_ce, atol=1e-6, rtol=1e-6))

    def test_select_balanced_rows_uses_min_available_cap(self):
        rows = [
            {"lang": "en", "input_ids": [1, 2]},
            {"lang": "en", "input_ids": [3, 4]},
            {"lang": "fr", "input_ids": [5, 6]},
        ]
        selected, effective_cap = _select_balanced_rows(
            rows,
            languages=["en", "fr"],
            max_examples_per_lang=2,
            seed=13,
        )

        self.assertEqual(effective_cap, 1)
        self.assertEqual(len(selected), 2)
        self.assertEqual(sorted(row["lang"] for row in selected), ["en", "fr"])

    def test_compute_periodic_steps_includes_final_step(self):
        self.assertEqual(_compute_periodic_steps(total_steps=950, every_steps=400), [400, 800, 950])


if __name__ == "__main__":
    unittest.main()
