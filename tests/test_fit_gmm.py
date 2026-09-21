import json
import tempfile
import unittest
from pathlib import Path

from fit_gmm import (
    _build_parser,
    _get_explicit_cli_keys,
    _load_prompt_records,
    _load_prompt_records_from_roots,
    _load_repr_defaults,
    _resolve_config,
)


class FitGmmTests(unittest.TestCase):
    def test_load_prompt_records_prefers_pud_jsonl_surface_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pud_holdout"
            data_root = root / "pud_langs_train"
            data_root.mkdir(parents=True, exist_ok=True)
            prompts_path = root / "pud_prompts_train.jsonl"
            prompts_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "en_pud-train-0",
                                "prompt_text": "hello world",
                                "surface_tokens": ["hello", "world"],
                            }
                        ),
                        json.dumps(
                            {
                                "id": "fr_pud-train-0",
                                "prompt_text": "bonjour monde",
                                "surface_tokens": ["bonjour", "monde"],
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            records = _load_prompt_records(
                data_root=data_root,
                lang="en",
                text_column="prompt",
                max_samples=8,
                seed=42,
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["text"], "hello world")
            self.assertEqual(records[0]["surface_tokens"], ["hello", "world"])

    def test_load_prompt_records_accepts_source_aware_ud_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "ud_holdout"
            data_root = root / "ud_langs_train"
            data_root.mkdir(parents=True, exist_ok=True)
            prompts_path = root / "ud_prompts_train.jsonl"
            prompts_path.write_text(
                json.dumps(
                    {
                        "id": "uk_iu-train-0",
                        "prompt_text": "привіт світ",
                        "source_lang": "uk",
                        "surface_tokens": ["привіт", "світ"],
                    }
                ),
                encoding="utf-8",
            )

            records = _load_prompt_records(
                data_root=data_root,
                lang="uk",
                text_column="prompt",
                max_samples=8,
                seed=42,
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["text"], "привіт світ")
            self.assertEqual(records[0]["surface_tokens"], ["привіт", "світ"])

    def test_load_prompt_records_sampling_order_is_seeded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pud_holdout"
            data_root = root / "pud_langs_train"
            data_root.mkdir(parents=True, exist_ok=True)
            prompts_path = root / "pud_prompts_train.jsonl"
            prompts_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": f"en_pud-train-{idx}",
                            "prompt_text": f"text-{idx}",
                            "surface_tokens": [f"text-{idx}"],
                        }
                    )
                    for idx in range(10)
                ),
                encoding="utf-8",
            )

            records = _load_prompt_records(
                data_root=data_root,
                lang="en",
                text_column="prompt",
                max_samples=4,
                seed=42,
            )

            self.assertEqual(
                [record["text"] for record in records],
                ["text-8", "text-1", "text-5", "text-0"],
            )

    def test_load_prompt_records_from_roots_falls_back_to_ud_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pud_root = Path(tmpdir) / "pud_holdout" / "pud_langs_train"
            ud_root = Path(tmpdir) / "ud_holdout" / "ud_langs_train"
            ud_root.mkdir(parents=True, exist_ok=True)
            (ud_root.parent / "ud_prompts_train.jsonl").write_text(
                json.dumps(
                    {
                        "id": "bg_btb-train-0",
                        "prompt_text": "здравей",
                        "source_lang": "bg",
                        "surface_tokens": ["здравей"],
                    }
                ),
                encoding="utf-8",
            )

            records = _load_prompt_records_from_roots(
                data_roots=[pud_root, ud_root],
                lang="bg",
                text_column="prompt",
                max_samples=8,
                seed=42,
            )

            self.assertEqual(records[0]["text"], "здравей")

    def test_resolve_config_rejects_conflicting_explicit_languages(self):
        args = type(
            "Args",
            (),
            {
                "setup_name": "pud9",
                "gmm_setup_registry": "configs/gmm_setups.json",
                "languages": ["en", "fr"],
                "data_root": "data/pud_holdout/pud_langs_train",
                "extra_data_root": [],
                "text_column": "prompt",
                "max_samples": 100,
                "seed": 42,
                "repr_priors": "uniform",
                "repr_pca": "layerwise",
                "repr_pca_variance": 0.98,
                "repr_cov": "full",
                "repr_unit": "token",
            },
        )()

        with self.assertRaises(ValueError):
            _resolve_config(args, {"languages"})

    def test_resolve_config_detects_both_explicit_cli_forms(self):
        parser = _build_parser(_load_repr_defaults())
        common_args = [
            "--model-name",
            "gpt2",
            "--revision",
            "main",
            "--setup-name",
            "pud9",
        ]

        for seed_args in (["--seed", "7"], ["--seed=7"]):
            with self.subTest(seed_args=seed_args):
                argv = common_args + seed_args
                args = parser.parse_args(argv)
                explicit_keys = _get_explicit_cli_keys(parser, argv)

                self.assertIn("seed", explicit_keys)
                with self.assertRaises(ValueError):
                    _resolve_config(args, explicit_keys)


if __name__ == "__main__":
    unittest.main()
