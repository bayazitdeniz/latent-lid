import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis import (
    build_undetermined_langdist_rows,
    discover_eval_runs,
    load_lang_probs_artifact,
    load_selected_runs,
    load_target_string_menu_dataframe,
    micro_average_langdist_categories,
    micro_average_story_row_values,
    openended_method_label_from_config,
    select_latest_unique_runs,
    summarize_base_instruct_methods,
)


################################################################################
# Evaluation artifact discovery and loading
################################################################################

class AnalysisArtifactTests(unittest.TestCase):
    def test_discover_and_load_selected_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "logs" / "evals" / "exp-1"
            root.mkdir(parents=True)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_name": "demo/model",
                        "revision": "main",
                        "prompts_path": "data/prompts.jsonl",
                        "do_decoding": True,
                        "do_repr": True,
                        "decoding_mapping": "decode_then_classify",
                        "repr_priors": "uniform",
                        "repr_pca": "layerwise",
                        "repr_cov": "full",
                        "repr_unit": "token",
                        "repr_token_agg": "frac_50",
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {"checkpoint_id": "main", "method": "repr_gmm", "layer": 0, "D": 0.5, "H": 0.7},
                ]
            ).to_parquet(root / "metrics.parquet", index=False)
            pd.DataFrame(
                [
                    {
                        "checkpoint_id": "main",
                        "method": "repr_gmm",
                        "prompt_idx": 0,
                        "prompt_id": "en_1",
                        "prompt_lang": "en",
                        "layer": 0,
                        "dominant_lang": "en",
                        "dominance": 0.9,
                        "entropy": 0.1,
                        "pivot": False,
                    }
                ]
            ).to_parquet(root / "prompt_metrics.parquet", index=False)

            manifest = discover_eval_runs(
                log_root=Path(tmpdir) / "logs" / "evals",
                config_filters={"model_name": "demo/model"},
                require_prompt_metrics=True,
            )
            self.assertEqual(len(manifest), 1)

            loaded = load_selected_runs(
                log_root=Path(tmpdir) / "logs" / "evals",
                config_filters={"repr_pca": "layerwise"},
                table="prompt_metrics",
            )
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.iloc[0]["model_name"], "demo/model")

    def test_select_latest_unique_runs(self):
        manifest = pd.DataFrame(
            [
                {
                    "exp_id": "exp-1",
                    "model_name": "m",
                    "revision": "main",
                    "prompts_path": "p.jsonl",
                    "do_decoding": True,
                    "do_repr": False,
                    "decoding_mapping": "decode_then_classify",
                    "repr_priors": None,
                },
                {
                    "exp_id": "exp-2",
                    "model_name": "m",
                    "revision": "main",
                    "prompts_path": "p.jsonl",
                    "do_decoding": True,
                    "do_repr": False,
                    "decoding_mapping": "decode_then_classify",
                    "repr_priors": None,
                },
                {
                    "exp_id": "exp-3",
                    "model_name": "m",
                    "revision": "main",
                    "prompts_path": "p.jsonl",
                    "do_decoding": False,
                    "do_repr": True,
                    "decoding_mapping": "decode_then_classify",
                    "repr_priors": "uniform",
                    "repr_pca": "layerwise",
                    "repr_cov": "full",
                    "repr_unit": "token",
                    "repr_token_agg": "frac_50",
                },
            ]
        )
        selected = select_latest_unique_runs(manifest)
        self.assertEqual(selected["exp_id"].tolist(), ["exp-2", "exp-3"])

    def test_load_lang_probs_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lang_probs_main.pt"
            payload = {"methods": {"repr_gmm": {"langs": ["en"], "probs": torch.ones((1, 1, 1), dtype=torch.float16)}}}
            torch.save(payload, path)
            loaded = load_lang_probs_artifact(path)
            self.assertIn("repr_gmm", loaded["methods"])


################################################################################
# Generic language-distribution summaries
################################################################################

class LangDistSummaryTests(unittest.TestCase):
    def test_build_undetermined_langdist_rows(self):
        artifact = {
            "checkpoint_id": "main",
            "layer_indices": [0, 1],
            "prompt_ids": ["p0"],
            "prompt_langs": ["en"],
            "methods": {
                "repr_gmm": {
                    "langs": ["en", "fr", "de"],
                    "probs": torch.tensor(
                        [[[0.40, 0.38, 0.22], [0.70, 0.20, 0.10]]],
                        dtype=torch.float16,
                    ),
                }
            },
        }
        df = build_undetermined_langdist_rows(
            artifact,
            method="repr_gmm",
            n_eff_threshold=2.5,
            margin_threshold=0.05,
        )
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["assigned_lang"], "undet")
        self.assertTrue(df.iloc[0]["undetermined"])
        self.assertEqual(df.iloc[1]["assigned_lang"], "en")
        self.assertFalse(df.iloc[1]["undetermined"])

    def test_micro_story_averaging_pools_language_values(self):
        category_df = pd.DataFrame(
            [
                {
                    "model": "m",
                    "prompt_id": "p1",
                    "layer_norm": 0.10,
                    "prob_category": "other",
                    "category_lang_count": 2,
                    "category_prob_sum": 0.4,
                    "category_prob_sq_sum": 0.1,
                    "prob_langdist_max": 0.3,
                },
                {
                    "model": "m",
                    "prompt_id": "p2",
                    "layer_norm": 0.20,
                    "prob_category": "other",
                    "category_lang_count": 1,
                    "category_prob_sum": 0.5,
                    "category_prob_sq_sum": 0.25,
                    "prob_langdist_max": 0.5,
                },
                {
                    "model": "m",
                    "prompt_id": "p1",
                    "layer_norm": 0.80,
                    "prob_category": "other",
                    "category_lang_count": 1,
                    "category_prob_sum": 0.9,
                    "category_prob_sq_sum": 0.81,
                    "prob_langdist_max": 0.9,
                },
            ]
        )

        pooled = micro_average_langdist_categories(
            category_df,
            group_cols=("model",),
            layer_min=0.0,
            layer_max=0.5,
            include_max=False,
        ).iloc[0]
        self.assertAlmostEqual(pooled["mean_prob"], 0.3)
        self.assertAlmostEqual(pooled["std_prob"], 0.2)
        self.assertAlmostEqual(pooled["stderr_prob"], 0.2 / np.sqrt(3))
        self.assertEqual(pooled["n_values"], 3)
        self.assertEqual(pooled["n_prompts"], 2)

        maxima = micro_average_story_row_values(
            category_df,
            prob_col="prob_langdist_max",
            group_cols=("model",),
            layer_min=0.0,
            layer_max=0.5,
            include_max=False,
        ).iloc[0]
        self.assertAlmostEqual(maxima["mean_prob"], 0.4)
        self.assertAlmostEqual(maxima["stderr_prob"], 0.1)
        self.assertEqual(maxima["n_values"], 2)


################################################################################
# Open-ended method labels
################################################################################

class OpenEndedMethodLabelTests(unittest.TestCase):
    def test_labels_representation_raw_and_tuned_methods(self):
        cases = [
            ({"do_repr": True}, "repr"),
            (
                {
                    "do_decoding": True,
                    "decoding_lens": "raw_logitlens",
                    "decoding_decode_mode": "rollout_argmax",
                },
                "raw-rmax",
            ),
            (
                {
                    "do_decoding": True,
                    "decoding_lens": "raw_logitlens",
                    "decoding_decode_mode": "rollout_sample",
                },
                "raw-rtopp",
            ),
            (
                {
                    "do_decoding": True,
                    "decoding_lens": "tuned_lens",
                    "decoding_decode_mode": "rollout_argmax",
                },
                "tuned-rmax",
            ),
            (
                {
                    "do_decoding": True,
                    "decoding_lens": "tuned_lens",
                    "decoding_decode_mode": "rollout_sample",
                },
                "tuned-rtopp",
            ),
        ]
        for config, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(openended_method_label_from_config(config), expected)


################################################################################
# Target-string analysis
################################################################################

class TargetStringMenuDataFrameTests(unittest.TestCase):
    def test_load_target_string_menu_dataframe_expands_menu_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_path = Path(tmpdir) / "evals" / "exp-1"
            exp_path.mkdir(parents=True)
            torch.save(
                {
                    "layer_indices": [4, 8],
                    "records": [
                        {
                            "prompt_id": "p0",
                            "prompt_lang": "en",
                            "concept_id": "book",
                            "tgt_lang": "fr",
                            "menu_strings_grouped": [
                                {"text": "bonjour", "langs": ["fr", "es"]},
                                {"text": "hallo", "langs": ["de"]},
                            ],
                            "menu_teacher_forced_sum_logprobs": torch.log(
                                torch.tensor([[0.25, 0.50], [0.125, 0.25]])
                            ),
                            "menu_teacher_forced_token_counts": torch.tensor([2, 1]),
                            "menu_teacher_forced_first_token_logprobs": torch.tensor(
                                [
                                    [math.log(0.50), math.log(0.25)],
                                    [math.log(0.25), math.log(0.125)],
                                ]
                            ),
                            "menu_start_token_logprobs": torch.tensor(
                                [
                                    [math.log(0.40), math.log(0.20)],
                                    [math.log(0.20), math.log(0.10)],
                                ]
                            ),
                        }
                    ],
                },
                exp_path / "target_string_details_main.pt",
            )
            manifest = pd.DataFrame(
                [
                    {
                        "exp_id": "exp-1",
                        "exp_path": str(exp_path),
                        "model_name": "model/a",
                        "display_model_name": "Model A",
                        "revision": "main",
                        "data_source": "translation",
                        "translation_target_lang": "fr",
                    }
                ]
            )

            df = load_target_string_menu_dataframe(manifest, target_langs=["fr"])

        self.assertEqual(len(df), 2)
        self.assertEqual(df["layer"].tolist(), [4, 8])
        self.assertEqual(df["tgt_lang"].tolist(), ["fr", "fr"])
        self.assertEqual(df["menu_group_lang_count"].tolist(), [2, 2])
        self.assertTrue(math.isclose(df.iloc[0]["menu_prob"], 0.25, rel_tol=1e-6))
        self.assertTrue(math.isclose(df.iloc[0]["menu_prob_share"], 0.125, rel_tol=1e-6))
        self.assertTrue(math.isclose(df.iloc[0]["menu_mean_prob"], 0.5, rel_tol=1e-6))
        self.assertTrue(math.isclose(df.iloc[0]["menu_mean_prob_share"], 0.25, rel_tol=1e-6))
        self.assertTrue(math.isclose(df.iloc[0]["menu_first_token_prob"], 0.5, rel_tol=1e-6))
        self.assertTrue(math.isclose(df.iloc[0]["menu_first_token_prob_share"], 0.25, rel_tol=1e-6))
        self.assertTrue(math.isclose(df.iloc[0]["menu_start_token_prob"], 0.4, rel_tol=1e-6))
        self.assertTrue(math.isclose(df.iloc[0]["menu_start_token_prob_share"], 0.2, rel_tol=1e-6))

    def test_load_target_string_menu_dataframe_maps_nonfinite_logprobs_to_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_path = Path(tmpdir) / "evals" / "exp-1"
            exp_path.mkdir(parents=True)
            torch.save(
                {
                    "layer_indices": [0, 1],
                    "records": [
                        {
                            "prompt_id": "p0",
                            "prompt_lang": "en",
                            "concept_id": "book",
                            "tgt_lang": "it",
                            "menu_strings_grouped": [{"text": "ciao", "langs": ["it"]}],
                            "menu_teacher_forced_sum_logprobs": torch.log(
                                torch.tensor([[0.125], [0.0625]])
                            ),
                            "menu_teacher_forced_token_counts": torch.tensor([1]),
                            "menu_teacher_forced_first_token_logprobs": torch.tensor(
                                [[float("-inf")], [float("nan")]]
                            ),
                            "menu_start_token_logprobs": torch.tensor(
                                [[float("-inf")], [float("nan")]]
                            ),
                        }
                    ],
                },
                exp_path / "target_string_details_main.pt",
            )
            manifest = pd.DataFrame(
                [
                    {
                        "exp_id": "exp-1",
                        "exp_path": str(exp_path),
                        "model_name": "model/a",
                        "display_model_name": "Model A",
                        "revision": "main",
                        "data_source": "translation",
                        "translation_target_lang": "it",
                    }
                ]
            )

            df = load_target_string_menu_dataframe(manifest, target_langs=["it"])

        self.assertEqual(len(df), 2)
        self.assertEqual(df["menu_first_token_prob"].tolist(), [0.0, 0.0])
        self.assertEqual(df["menu_first_token_prob_share"].tolist(), [0.0, 0.0])
        self.assertEqual(df["menu_start_token_prob"].tolist(), [0.0, 0.0])
        self.assertEqual(df["menu_start_token_prob_share"].tolist(), [0.0, 0.0])


################################################################################
# Base/instruct summaries
################################################################################

class BaseInstructSummaryTests(unittest.TestCase):
    def test_method_summary_keeps_estimators_and_computes_prompt_level_agreement(self):
        rows = []
        for variant, display_name in (
            ("base", "Llama-3.1-8B"),
            ("instruct", "Llama-3.1-8B-Instruct"),
        ):
            for prompt_idx in (0, 1):
                for method_label in ("repr", "raw-rtopp"):
                    rows.append(
                        {
                            "data_source": "pud21",
                            "display_model_name": display_name,
                            "method_label": method_label,
                            "prompt_idx": prompt_idx,
                            "prompt_id": f"{variant}-{prompt_idx}",
                            "layer": 0,
                            "layer_norm": 0.0,
                            "pivot": float(prompt_idx),
                            "entropy": 1.0 + prompt_idx,
                            "dominant_lang": (
                                "fr"
                                if method_label == "repr" and prompt_idx == 1
                                else "en"
                            ),
                        }
                    )
        summary = summarize_base_instruct_methods(
            pd.DataFrame(rows),
            data_sources=["pud21"],
            families=["Llama-3.1"],
        )

        base = summary[summary["base_instruct_variant"].eq("base")]
        self.assertEqual(len(base), 5)
        self.assertEqual(set(base["method_label"]), {"repr", "raw-rtopp", "agreement_repr_vs_raw-rtopp"})
        agreement = base[base["metric"].eq("agreement")].iloc[0]
        self.assertEqual(agreement["n"], 2)
        self.assertTrue(math.isclose(agreement["value"], 0.5))


if __name__ == "__main__":
    unittest.main()
