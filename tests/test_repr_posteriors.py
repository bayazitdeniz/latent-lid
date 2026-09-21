import unittest
from unittest import mock

import torch

from representation import (
    GaussianParams,
    LoadedGMMLayer,
    compute_lang_posteriors,
    compute_repr_langdist_from_sequences,
)


class ReprPosteriorTests(unittest.TestCase):
    def test_sequence_langdist_reports_word_id_fallback(self):
        loaded = LoadedGMMLayer(
            covariance_type="diag",
            layer_index=0,
            params={
                "en": GaussianParams(
                    mean=torch.tensor([0.0, 0.0]),
                    cov=torch.tensor([1.0, 1.0]),
                    n_samples=10,
                ),
                "fr": GaussianParams(
                    mean=torch.tensor([1.0, 1.0]),
                    cov=torch.tensor([1.0, 1.0]),
                    n_samples=10,
                ),
            },
        )
        sequence_latents = [
            torch.tensor(
                [[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 1.0]]]
            )
        ]

        with mock.patch("representation.load_gmm_layer", return_value=loaded):
            langdist, word_ids_used, _, _ = compute_repr_langdist_from_sequences(
                sequence_latents_by_prompt=sequence_latents,
                layer_indices=[0],
                gmm_dir="unused",
                repr_priors="uniform",
                repr_unit="token",
                repr_token_agg="frac_50",
                repr_rollout_k=1,
                word_ids_by_prompt=[None],
                show_progress=False,
            )

        self.assertFalse(word_ids_used)
        self.assertTrue(
            torch.allclose(
                langdist["en"] + langdist["fr"],
                torch.ones_like(langdist["en"]),
            )
        )

    def test_posteriors_sum_to_one(self):
        params = {
            "en": GaussianParams(mean=torch.tensor([0.0, 0.0]), cov=torch.tensor([1.0, 1.0]), n_samples=10),
            "fr": GaussianParams(mean=torch.tensor([1.0, 1.0]), cov=torch.tensor([1.0, 1.0]), n_samples=10),
        }
        loaded = LoadedGMMLayer(covariance_type="diag", layer_index=0, params=params)
        latents = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        post = compute_lang_posteriors(latents, loaded, use_priors=False)
        totals = post["en"] + post["fr"]
        self.assertTrue(torch.allclose(totals, torch.ones_like(totals), atol=1e-6))

    def test_uniform_priors_override(self):
        params = {
            "en": GaussianParams(mean=torch.tensor([0.0, 0.0]), cov=torch.tensor([1.0, 1.0]), n_samples=100),
            "fr": GaussianParams(mean=torch.tensor([0.0, 0.0]), cov=torch.tensor([1.0, 1.0]), n_samples=1),
        }
        loaded = LoadedGMMLayer(covariance_type="diag", layer_index=0, params=params)
        latents = torch.tensor([[0.0, 0.0]])
        post_empirical = compute_lang_posteriors(latents, loaded, use_priors=True)
        post_uniform = compute_lang_posteriors(latents, loaded, use_priors=True, priors_override="uniform")
        self.assertNotAlmostEqual(post_empirical["en"].item(), post_uniform["en"].item())
        self.assertAlmostEqual(post_uniform["en"].item(), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
