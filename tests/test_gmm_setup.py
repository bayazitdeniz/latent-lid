import unittest

from gmm_setup import get_gmm_setup_comparison_view, infer_gmm_setup_name


class GMMSetupTests(unittest.TestCase):
    def test_infer_setup_name_matches_languages_without_order_or_duplicates(self):
        registry = {
            "two_lang": {"languages": ["en", "fr"]},
            "three_lang": {"languages": ["en", "fr", "tr"]},
        }

        self.assertEqual(
            infer_gmm_setup_name(["fr", "en", "fr"], registry),
            "two_lang",
        )

    def test_infer_setup_name_requires_a_unique_exact_match(self):
        duplicate_registry = {
            "first": {"languages": ["en", "fr"]},
            "second": {"languages": ["fr", "en"]},
        }

        with self.assertRaisesRegex(ValueError, "Multiple GMM setups match"):
            infer_gmm_setup_name(["en", "fr"], duplicate_registry)
        with self.assertRaisesRegex(ValueError, "No GMM setup matches"):
            infer_gmm_setup_name(["en", "tr"], duplicate_registry)

    def test_comparison_view_ignores_manifest_identity_fields(self):
        setup = {
            "languages": ["fr", "en"],
            "data_root": "data/train",
            "extra_data_roots": ["data/extra"],
            "text_column": "prompt",
            "max_samples": 100,
            "seed": 42,
            "repr_priors": "uniform",
            "repr_pca": "layerwise",
            "repr_pca_variance": 0.98,
            "repr_cov": "diag",
            "repr_unit": "token",
        }
        manifest = {
            **setup,
            "languages": ["en", "fr"],
            "schema_version": 1,
            "setup_name": "two_lang",
            "model_name": "org/model",
            "revision": "main",
        }

        self.assertEqual(
            get_gmm_setup_comparison_view(manifest),
            get_gmm_setup_comparison_view(setup),
        )


if __name__ == "__main__":
    unittest.main()
