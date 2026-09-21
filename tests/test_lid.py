import unittest
from unittest import mock

import torch

from lid import (
    GlotLidBackend,
    LangIdBackend,
    build_lid_backend,
    normalize_glotlid_label,
)


class GlotLidHelperTests(unittest.TestCase):
    def test_normalize_glotlid_label(self):
        self.assertEqual(normalize_glotlid_label("__label__eng_Latn"), "en")
        self.assertEqual(normalize_glotlid_label("arb_Arab"), "ar")
        self.assertEqual(normalize_glotlid_label("__label__und_Latn"), "und_Latn")

    def test_build_lid_backend_langid(self):
        backend = build_lid_backend("langid")
        self.assertEqual(type(backend).__name__, "LangIdBackend")

    def test_build_lid_backend_glotlid(self):
        fake_fasttext = mock.Mock()
        fake_model = mock.Mock()
        fake_model.get_output_matrix.return_value = [[1.0, 0.0], [0.0, 1.0]]
        fake_model.get_labels.return_value = ["__label__eng_Latn", "__label__fra_Latn"]
        fake_fasttext.load_model.return_value = fake_model

        with mock.patch.dict("sys.modules", {"fasttext": fake_fasttext}):
            with mock.patch("huggingface_hub.hf_hub_download", return_value="/tmp/model.bin"):
                backend = build_lid_backend("glotlid")

        self.assertIsInstance(backend, GlotLidBackend)

    def test_glotlid_distribution_is_project_normalized(self):
        backend = GlotLidBackend.__new__(GlotLidBackend)
        backend._labels = ["__label__eng_Latn", "__label__fra_Latn", "__label__spa_Latn"]
        backend._output_matrix = torch.tensor(
            [
                [2.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=torch.float32,
        ).numpy()

        class FakeModel:
            def get_sentence_vector(self, text):
                return [1.0, 0.0]

        backend._model = FakeModel()
        dist = backend.predict_distribution("hello", languages=["en", "fr"])
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=5)
        self.assertEqual(set(dist.keys()), {"en", "fr"})
        self.assertGreater(dist["en"], dist["fr"])

    def test_glotlid_distribution_conditions_on_allowed_normalized_langs(self):
        backend = GlotLidBackend.__new__(GlotLidBackend)
        backend._labels = ["__label__eng_Latn", "__label__fra_Latn", "__label__spa_Latn"]
        backend._output_matrix = torch.tensor(
            [
                [2.0, 0.0],
                [1.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=torch.float32,
        ).numpy()

        class FakeModel:
            def get_sentence_vector(self, text):
                return [1.0, 0.0]

        backend._model = FakeModel()
        dist = backend.predict_distribution("hello", languages=["en", "fr"])
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=5)
        self.assertEqual(set(dist.keys()), {"en", "fr"})
        self.assertAlmostEqual(
            dist["en"],
            float(torch.softmax(torch.log(torch.tensor([0.24472847, 0.09003057])), dim=0)[0]),
            places=5,
        )
        self.assertAlmostEqual(
            dist["fr"],
            float(torch.softmax(torch.log(torch.tensor([0.24472847, 0.09003057])), dim=0)[1]),
            places=5,
        )

    def test_glotlid_distribution_sums_script_variants_before_conditioning(self):
        backend = GlotLidBackend.__new__(GlotLidBackend)
        backend._labels = [
            "__label__arb_Arab",
            "__label__arb_Latn",
            "__label__eng_Latn",
        ]
        backend._output_matrix = torch.tensor(
            [
                [2.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=torch.float32,
        ).numpy()

        class FakeModel:
            def get_sentence_vector(self, text):
                return [1.0, 0.0]

        backend._model = FakeModel()
        dist = backend.predict_distribution("hello", languages=["ar", "en"])
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=5)
        self.assertEqual(set(dist.keys()), {"ar", "en"})
        self.assertGreater(dist["ar"], dist["en"])

    def test_glotlid_rewrites_newlines_before_fasttext(self):
        backend = GlotLidBackend.__new__(GlotLidBackend)
        backend._labels = ["__label__eng_Latn", "__label__fra_Latn"]
        backend._output_matrix = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ).numpy()

        class FakeModel:
            def __init__(self):
                self.last_text = None

            def get_sentence_vector(self, text):
                self.last_text = text
                return [1.0, 0.0]

        backend._model = FakeModel()
        dist = backend.predict_distribution("hello\nworld", languages=["en", "fr"])
        self.assertEqual(backend._model.last_text, "hello  world")
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=5)


class LangIdBackendTests(unittest.TestCase):
    def test_predict_distribution_renormalizes_filtered_rank_probs(self):
        backend = LangIdBackend.__new__(LangIdBackend)

        class FakeLangId:
            def rank(self, text):
                return [("en", 0.6), ("fr", 0.3), ("de", 0.1)]

        backend._langid = FakeLangId()
        dist = backend.predict_distribution("hello", languages=["en", "fr"])
        self.assertAlmostEqual(dist["en"], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(dist["fr"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
