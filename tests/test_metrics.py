import unittest

import torch

from metrics import (
    concat_langdist_chunks,
    compute_langdist_metrics,
    compute_langdist_prompt_pivot_mask,
)


class LangDistMetricsTests(unittest.TestCase):
    def test_concat_langdist_chunks_zero_fills_missing_language(self):
        chunks = [
            {
                "en": torch.tensor([[0.7, 0.6], [0.8, 0.9]]),
                "fr": torch.tensor([[0.3, 0.4], [0.2, 0.1]]),
            },
            {"en": torch.tensor([[1.0, 1.0]])},
        ]

        combined = concat_langdist_chunks(chunks)

        self.assertEqual(list(combined), ["en", "fr"])
        self.assertTrue(
            torch.equal(
                combined["en"],
                torch.tensor([[0.7, 0.6], [0.8, 0.9], [1.0, 1.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                combined["fr"],
                torch.tensor([[0.3, 0.4], [0.2, 0.1], [0.0, 0.0]]),
            )
        )

    def test_langdist_metrics_normalize(self):
        langdist = {
            "en": torch.tensor([[2.0, 1.0]]),
            "fr": torch.tensor([[2.0, 3.0]]),
        }

        metrics = compute_langdist_metrics(langdist)

        self.assertTrue(torch.all(metrics["dominance"] <= 1.0))
        self.assertTrue(torch.all(metrics["entropy"] >= 0.0))

    def test_prompt_pivot_mask(self):
        langdist = {
            "en": torch.tensor([[0.7, 0.2], [0.1, 0.4]]),
            "fr": torch.tensor([[0.3, 0.8], [0.2, 0.1]]),
            "de": torch.tensor([[0.0, 0.0], [0.7, 0.5]]),
        }

        pivot_mask = compute_langdist_prompt_pivot_mask(
            langdist,
            task_langs_by_prompt=[("en", "fr"), ("de",)],
        )

        expected = torch.tensor([[False, False], [False, False]])
        self.assertTrue(torch.equal(pivot_mask, expected))
