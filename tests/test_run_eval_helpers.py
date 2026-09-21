import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from transformers import AutoTokenizer, GPT2TokenizerFast

from gmm_setup import build_gmm_artifact_dir, infer_gmm_setup_name, load_gmm_setup_registry
from latents import gather_fixed_position_latents
from args import (
    build_parser,
    get_explicit_cli_keys,
    infer_task_langs,
    load_config,
    resolve_config,
    resolve_include_prompts_path,
    resolve_pud_prompts_path,
    resolve_synthetic_csv_path,
    resolve_ud_prompts_path,
    validate_pud_prompt_languages,
)
from representation import (
    resolve_repr_gmm_dir,
    select_repr_latents,
    select_repr_word_rollout_latents,
    validate_repr_gmm_dir,
)
from decode_then_classify import (
    _project_hf_cached_layer_logits,
    compute_rollout_word_token_budget,
    sampled_rollout_decode_then_classify,
    sample_top_p_token_id,
    sample_top_p_token_ids,
    score_singlepos_latents,
    select_decoding_latents,
    teacher_forced_argmax_decode_then_classify,
)
from run_eval_utils import (
    append_mean_layer_metric_rows,
    build_langdist_summary_rows,
    build_method_langdist_artifact,
    build_pivot_summary_rows,
    build_prompt_metric_rows,
    compute_surface_token_spans,
    format_prompts_for_tokenizer,
    get_prompt_content_end_token_positions,
    get_surface_word_ids_for_prompts,
    get_word_end_token_positions,
    load_jsonl_prompt_data,
    load_synthetic_prompt_data,
    select_fractional_token_position,
    select_rollout_token_positions,
    should_anchor_to_raw_prompt_content,
)
from target_string import (
    add_menu_mass_summary,
    build_target_string_artifact,
    build_target_string_record_base,
    build_menu_lang_log_masses,
    build_menu_lang_masses,
    build_teacher_forced_candidates,
    get_menu_langs,
    finite_exp,
    normalize_lang_log_masses,
    normalize_lang_masses,
    prepare_start_token_groups,
    run_start_token_scoring,
    run_teacher_forced_scoring,
    score_teacher_forced_one_candidate,
)
from lenses import RawLogitLens, TunedLogitLens, UnembedInfo, apply_unembed

LANGUAGE_SAMPLE_TEXTS = {
    "ar": "الطقس اليوم جميل ومناسب للمشي في الحديقة.",
    "cs": "Dnes je krásné počasí a půjdeme do parku.",
    "en": "The weather is nice today and we will walk to the park.",
    "fr": "Il fait beau aujourd'hui et nous allons au parc.",
    "hi": "आज मौसम अच्छा है और हम पार्क जाएंगे।",
    "id": "Cuaca hari ini cerah dan kami akan berjalan ke taman.",
    "is": "Veðrið er gott í dag og við förum í garðinn.",
    "pt": "Hoje o tempo está bom e vamos caminhar no parque.",
    "es": "Hoy hace buen tiempo y vamos a caminar al parque.",
    "de": "Heute ist das Wetter gut und wir gehen in den Park.",
    "ru": "Сегодня хорошая погода, и мы пойдём в парк.",
    "zh": "今天天气很好，我们去公园散步.",
    "ja": "今日は天気が良くて公園に行きます。",
    "ko": "오늘은 날씨가 좋아서 공원에 갑니다.",
}

ROLLOUT_ALIGNMENT_MODEL_TOKENIZERS = {
    "gpt2": "gpt2",
    "gpt2-xl": "gpt2-xl",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "apertus-8b": "swiss-ai/Apertus-8B-2509",
    "aya-23-8b": "CohereLabs/aya-23-8B",
    "eurollm-9b": "utter-project/EuroLLM-9B",
    "nemo": "mistralai/Mistral-Nemo-Instruct-2407",
}

CHAT_TEMPLATE_AVAILABILITY = {
    "gpt2": ("gpt2", False),
    "gpt2-xl": ("gpt2-xl", False),
    "llama2-7b": ("meta-llama/Llama-2-7b-hf", False),
    "llama-3.1-8b": ("meta-llama/Llama-3.1-8B", False),
    "apertus-8b": ("swiss-ai/Apertus-8B-2509", False),
    "eurollm-9b": ("utter-project/EuroLLM-9B", False),
    "llama-3.1-8b-instruct": ("meta-llama/Llama-3.1-8B-Instruct", True),
    "apertus-8b-instruct": ("swiss-ai/Apertus-8B-Instruct-2509", True),
    "nemo-instruct": ("mistralai/Mistral-Nemo-Instruct-2407", True),
    "aya-23-8b": ("CohereLabs/aya-23-8B", True),
    "eurollm-9b-instruct": ("utter-project/EuroLLM-9B-Instruct", True),
}


class FakeDecodingLens:
    def project_all_layers(self, latents, layer_indices):
        return latents

    def project_layer(self, hidden_states, layer_idx):
        return hidden_states


class RunEvalHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        cls.llama2_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            use_fast=True,
        )
        cls.llama31_instruct_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            use_fast=True,
        )

    def load_model_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def test_load_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "cfg.json"
            cfg_path.write_text(json.dumps({"a": 1}), encoding="utf-8")
            cfg = load_config(str(cfg_path))
            self.assertEqual(cfg["a"], 1)

    def test_parser_accepts_setup_named_pud_sources_not_bare_pud(self):
        defaults = load_config("configs/default.json")
        parser = build_parser(defaults)
        self.assertEqual(parser.parse_args(["--data-source", "pud21"]).data_source, "pud21")
        self.assertEqual(
            parser.parse_args(["--data-source", "translation_to_fr"]).data_source,
            "translation_to_fr",
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["--data-source", "pud"])

    def test_append_mean_layer_metric_rows(self):
        rows = []
        import torch

        append_mean_layer_metric_rows(
            rows,
            checkpoint_id="rev",
            method="m",
            n_prompts=2,
            layer_indices=[0, 1],
            metrics={"D": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["D"], 2.0)
        self.assertEqual(rows[1]["D"], 3.0)

    def test_build_langdist_summary_rows(self):
        import torch

        lang_probs = {
            "en": torch.tensor([[0.8, 0.2], [0.6, 0.1]]),
            "fr": torch.tensor([[0.2, 0.8], [0.4, 0.9]]),
        }
        rows = build_langdist_summary_rows(
            checkpoint_id="rev",
            method="decoding",
            layer_indices=[0, 1],
            langdist=lang_probs,
            n_prompts=2,
            decoding_mapping="decode_then_classify",
            decoding_lid_backend="glotlid",
            decoding_lens="logit_lens",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            list(rows[0].keys()),
            [
                "checkpoint_id",
                "method",
                "decoding_mapping",
                "decoding_lid_backend",
                "decoding_lens",
                "layer",
                "D",
                "H",
                "A",
                "P",
                "n_prompts",
            ],
        )
        self.assertEqual(rows[0]["method"], "decoding")
        self.assertEqual(rows[0]["decoding_mapping"], "decode_then_classify")
        self.assertAlmostEqual(rows[0]["D"], 0.7)
        self.assertAlmostEqual(rows[1]["D"], 0.85)
        self.assertIsNone(rows[0]["P"])

    def test_build_pivot_summary_rows(self):
        import torch

        lang_probs = {
            "en": torch.tensor([[0.8, 0.2], [0.6, 0.1]]),
            "fr": torch.tensor([[0.2, 0.8], [0.4, 0.9]]),
        }
        rows = build_pivot_summary_rows(
            checkpoint_id="rev",
            method="decoding_pivot",
            layer_indices=[0, 1],
            langdist=lang_probs,
            n_prompts=2,
            task_langs_by_prompt=[("en",), ("en",)],
            decoding_mapping="decode_then_classify",
            decoding_lens="logit_lens",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["method"], "decoding_pivot")
        self.assertAlmostEqual(rows[0]["P"], 0.0)
        self.assertAlmostEqual(rows[1]["P"], 1.0)
        self.assertIsNone(rows[0]["D"])

    def test_resolve_config_repr_defaults(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "pud_root": "data/pud_holdout",
            "repr_priors": None,
            "repr_pca": None,
            "repr_pca_variance": None,
            "repr_cov": None,
            "repr_unit": None,
        })
        resolved = resolve_config(cfg, explicit_keys=set())
        self.assertEqual(resolved["prompts_path"], "data/pud_holdout/pud_prompts_test.jsonl")
        self.assertEqual(resolved["data_source"], "pud21")
        self.assertEqual(resolved["repr_gmm_setup"], "pud21")
        self.assertEqual(resolved["repr_priors"], "uniform")
        self.assertEqual(resolved["repr_pca"], "layerwise")
        self.assertEqual(resolved["repr_cov"], "diag")
        self.assertEqual(resolved["repr_token_agg"], "frac_50")
        self.assertEqual(resolved["repr_unit"], "token")

    def test_resolve_config_preserves_explicit_repr(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "pud_root": "data/pud_holdout",
            "repr_priors": "empirical",
        })
        resolved = resolve_config(cfg, explicit_keys={"repr_priors"})
        self.assertEqual(resolved["repr_priors"], "empirical")

    def test_resolve_config_preserves_explicit_repr_token_agg(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "pud_root": "data/pud_holdout",
            "repr_token_agg": "frac_50",
        })
        resolved = resolve_config(cfg, explicit_keys={"repr_token_agg"})
        self.assertEqual(resolved["repr_token_agg"], "frac_50")

    def test_resolve_config_applies_dataset_model_run_defaults(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "CohereLabs/aya-23-8B",
            "data_source": "include_10lang_3domain_cap30",
            "prompts_path": "data/test_prompts.jsonl",
            "max_prompts": 0,
            "max_prompts_per_lang": 64,
            "trace_batch_size": 32,
            "sampled_rollout_prompt_batch_size": 32,
        })
        resolved = resolve_config(cfg, explicit_keys={"data_source", "model_name"})
        self.assertIsNone(resolved["max_prompts"])
        self.assertIsNone(resolved["max_prompts_per_lang"])
        self.assertEqual(resolved["trace_batch_size"], 1)
        self.assertEqual(resolved["sampled_rollout_prompt_batch_size"], 1)

    def test_resolve_config_applies_pud_prompt_cap_default(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "pud21",
            "pud_root": "data/pud_holdout",
            "max_prompts": 999,
            "max_prompts_per_lang": 999,
        })
        resolved = resolve_config(cfg, explicit_keys={"data_source"})
        self.assertIsNone(resolved["max_prompts"])
        self.assertEqual(resolved["max_prompts_per_lang"], 64)

    def test_resolve_config_preserves_explicit_batch_sizes(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "CohereLabs/aya-23-8B",
            "data_source": "include_10lang_3domain_cap30",
            "prompts_path": "data/test_prompts.jsonl",
            "trace_batch_size": 7,
            "sampled_rollout_prompt_batch_size": 3,
        })
        resolved = resolve_config(
            cfg,
            explicit_keys={
                "data_source",
                "model_name",
                "trace_batch_size",
                "sampled_rollout_prompt_batch_size",
            },
        )
        self.assertEqual(resolved["trace_batch_size"], 7)
        self.assertEqual(resolved["sampled_rollout_prompt_batch_size"], 3)

    def test_resolve_config_rejects_combined_decoding_and_repr(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "prompts_path": "data/test_prompts.jsonl",
            "do_decoding": True,
            "do_repr": True,
        })
        with self.assertRaisesRegex(ValueError, "You can only run do_decoding or do_repr"):
            resolve_config(cfg, explicit_keys={"do_decoding", "do_repr"})

    def test_resolve_config_rejects_repr_rollout_with_all_tokens(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "prompts_path": "data/test_prompts.jsonl",
            "do_decoding": False,
            "do_repr": True,
            "repr_token_agg": "all_tokens",
            "repr_rollout_k": 5,
        })
        with self.assertRaisesRegex(ValueError, "repr_token_agg=all_tokens"):
            resolve_config(cfg, explicit_keys={"repr_rollout_k"})

    def test_resolve_config_allows_repr_rollout_with_token_unit(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "prompts_path": "data/test_prompts.jsonl",
            "do_decoding": False,
            "do_repr": True,
            "repr_token_agg": "frac_50",
            "repr_rollout_k": 5,
        })
        resolved = resolve_config(cfg, explicit_keys={"repr_token_agg", "repr_rollout_k"})
        self.assertEqual(resolved["repr_unit"], "token")

    def test_resolve_config_accepts_tuned_lens(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "decoding_lens": "tuned_lens",
        })
        resolved = resolve_config(cfg, explicit_keys={"decoding_lens"})
        self.assertEqual(resolved["decoding_lens"], "tuned_lens")
        self.assertEqual(
            resolved["tuned_lens_dir"],
            "logs/tuned_lens_fineweb27/org_model/{rev}",
        )

    def test_resolve_config_accepts_custom_tuned_lens_root(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "decoding_lens": "tuned_lens",
            "tuned_lens_root": "artifacts/tuned_lenses",
        })
        resolved = resolve_config(
            cfg,
            explicit_keys={"decoding_lens", "tuned_lens_root"},
        )
        self.assertEqual(
            resolved["tuned_lens_dir"],
            "artifacts/tuned_lenses/org_model/{rev}",
        )

    def test_resolve_config_preserves_exact_tuned_lens_dir(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "decoding_lens": "tuned_lens",
            "tuned_lens_dir": "artifacts/exact/{rev}",
        })
        resolved = resolve_config(
            cfg,
            explicit_keys={"decoding_lens", "tuned_lens_dir"},
        )
        self.assertEqual(resolved["tuned_lens_dir"], "artifacts/exact/{rev}")

    def test_resolve_config_does_not_resolve_tuned_lens_dir_for_raw_lens(self):
        cfg = load_config("configs/default.json")
        cfg["model_name"] = "org/model"
        resolved = resolve_config(cfg, explicit_keys=set())
        self.assertIsNone(resolved["tuned_lens_dir"])

    def test_resolve_config_rejects_unknown_decoding_lens(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "decoding_lens": "mystery_lens",
        })
        with self.assertRaises(NotImplementedError):
            resolve_config(cfg, explicit_keys={"decoding_lens"})

    def test_resolve_config_rejects_unsupported_decoding_lid_backend(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "decoding_lid_backend": "mysterylid",
        })
        with self.assertRaises(ValueError):
            resolve_config(cfg, explicit_keys={"decoding_lid_backend"})

    def test_resolve_config_rejects_last_token_word_count_rollout_argmax(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "prompts_path": "data/test_prompts.jsonl",
            "do_decoding": True,
            "decoding_decode_mode": "rollout_argmax",
            "decoding_token_agg": "last_token",
            "decoding_rollout_word_cnt": 1,
        })
        with self.assertRaises(ValueError):
            resolve_config(cfg, explicit_keys={"decoding_decode_mode", "decoding_token_agg", "decoding_rollout_word_cnt"})

    def test_resolve_config_rejects_last_token_word_count_rollout_sample(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "prompts_path": "data/test_prompts.jsonl",
            "do_decoding": True,
            "decoding_decode_mode": "rollout_sample",
            "decoding_token_agg": "last_token",
            "decoding_rollout_word_cnt": 1,
        })
        with self.assertRaises(ValueError):
            resolve_config(cfg, explicit_keys={"decoding_decode_mode", "decoding_token_agg", "decoding_rollout_word_cnt"})

    def test_resolve_config_defaults_synthetic_to_start_token_scoring(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "copy",
            "do_decoding": True,
            "decoding_token_agg": "last_token",
            "repr_token_agg": "last_token",
            "repr_rollout_k": 0,
        })
        resolved = resolve_config(
            cfg,
            explicit_keys={"data_source", "do_decoding", "decoding_token_agg", "repr_token_agg", "repr_rollout_k"},
        )
        self.assertEqual(resolved["decoding_mapping"], "target_string")
        self.assertEqual(resolved["target_string_scoring_mode"], "start_tokens_only")

    def test_resolve_config_rejects_target_string_for_openended_data(self):
        cfg = load_config("configs/default.json")
        cfg.update(
            {
                "model_name": "org/model",
                "data_source": "pud21",
                "prompts_path": "data/test_prompts.jsonl",
                "do_decoding": True,
                "decoding_mapping": "target_string",
            }
        )
        with self.assertRaisesRegex(ValueError, "requires a synthetic data source"):
            resolve_config(cfg, explicit_keys={"decoding_mapping"})

    def test_resolve_config_discards_legacy_decoding_targets_path(self):
        cfg = load_config("configs/default.json")
        cfg["decoding_targets_path"] = "old_targets.jsonl"
        resolved = resolve_config(cfg, explicit_keys=set())
        self.assertNotIn("decoding_targets_path", resolved)

    def test_target_string_scoring_mode_arg_replaces_old_source_arg(self):
        defaults = load_config("configs/default.json")
        parser = build_parser(defaults)
        args = parser.parse_args(["--target-string-scoring-mode", "start_tokens_only"])
        self.assertEqual(args.target_string_scoring_mode, "start_tokens_only")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--target-string-lang-prob-source", "wendler_start_tokens"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--decoding-targets-path", "old_targets.jsonl"])

    def test_lang_mass_normalization_sums_to_one_when_mass_nonzero(self):
        import torch

        records = [
            {
                "menu_strings_grouped": [
                    {"text": "cat", "langs": ["en"]},
                    {"text": "chat", "langs": ["fr"]},
                    {"text": "shared", "langs": ["en", "fr"]},
                ]
            }
        ]
        group_probs = [torch.tensor([[0.2, 0.3, 0.5], [0.0, 0.4, 0.2]])]
        masses, unique_masses, ambiguous_mass, unique_mass = build_menu_lang_masses(
            prompt_target_records=records,
            menu_group_probs=group_probs,
            all_menu_langs=["en", "fr"],
            device="cpu",
        )
        dist = normalize_lang_masses(masses)
        stacked = torch.stack([dist["en"], dist["fr"]], dim=-1)
        self.assertTrue(torch.allclose(stacked.sum(dim=-1), torch.ones(1, 2)))
        self.assertTrue(torch.allclose(masses["en"], torch.tensor([[0.45, 0.1]])))
        self.assertTrue(torch.allclose(masses["fr"], torch.tensor([[0.55, 0.5]])))
        self.assertTrue(torch.allclose(unique_masses["fr"], torch.tensor([[0.3, 0.4]])))
        self.assertTrue(torch.allclose(ambiguous_mass, torch.tensor([[0.5, 0.2]])))
        self.assertTrue(torch.allclose(unique_mass, torch.tensor([[0.5, 0.4]])))

    def test_lang_log_mass_normalization_handles_tiny_start_token_mass(self):
        import torch

        log_masses = {
            "en": torch.tensor([[-120.0, -float("inf")]], dtype=torch.float64),
            "fr": torch.tensor([[-121.0, -float("inf")]], dtype=torch.float64),
        }
        dist = normalize_lang_log_masses(log_masses, empty_policy="uniform")
        stacked = torch.stack([dist["en"], dist["fr"]], dim=-1)

        self.assertTrue(torch.allclose(stacked[:, 0].sum(dim=-1), torch.ones(1, dtype=torch.float64)))
        self.assertTrue(torch.allclose(stacked[:, 1], torch.full((1, 2), 0.5, dtype=torch.float64)))
        self.assertGreater(float(dist["en"][0, 0]), float(dist["fr"][0, 0]))

    def test_menu_lang_log_masses_match_probability_masses_without_underflow(self):
        import torch

        records = [
            {
                "menu_strings_grouped": [
                    {"text": "cat", "langs": ["en"]},
                    {"text": "chat", "langs": ["fr"]},
                    {"text": "shared", "langs": ["en", "fr"]},
                ]
            }
        ]
        group_logprobs = [torch.log(torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64))]
        log_masses, unique_log_masses = build_menu_lang_log_masses(
            prompt_target_records=records,
            menu_group_logprobs=group_logprobs,
            all_menu_langs=["en", "fr"],
            device="cpu",
        )
        dist = normalize_lang_log_masses(log_masses)

        self.assertTrue(torch.allclose(torch.exp(log_masses["en"]), torch.tensor([[0.45]], dtype=torch.float64)))
        self.assertTrue(torch.allclose(torch.exp(log_masses["fr"]), torch.tensor([[0.55]], dtype=torch.float64)))
        self.assertTrue(torch.allclose(torch.exp(unique_log_masses["fr"]), torch.tensor([[0.3]], dtype=torch.float64)))
        self.assertTrue(torch.allclose(dist["en"], torch.tensor([[0.45]], dtype=torch.float64)))
        self.assertTrue(torch.allclose(dist["fr"], torch.tensor([[0.55]], dtype=torch.float64)))

    def test_get_menu_langs(self):
        records = [
            {
                "menu_strings_grouped": [
                    {"text": "cat", "langs": ["en"]},
                    {"text": "chat", "langs": ["fr", "en"]},
                ],
            }
        ]
        self.assertEqual(get_menu_langs(records), ["en", "fr"])

    def test_build_target_string_record_base_contains_shared_fields(self):
        record = {
            "prompt_id": "p1",
            "concept_id": "concept",
            "prompt_lang": "en",
            "tgt_lang": "fr",
            "tgt_text": "chat",
            "menu_strings_grouped": [{"text": "cat", "langs": ["en"]}],
            "menu_strings_unique": [{"text": "cat", "langs": ["en"]}],
            "extra": "unused",
        }
        self.assertEqual(
            build_target_string_record_base(record),
            {
                "prompt_id": "p1",
                "concept_id": "concept",
                "prompt_lang": "en",
                "tgt_lang": "fr",
                "tgt_text": "chat",
                "menu_strings_grouped": [{"text": "cat", "langs": ["en"]}],
                "menu_strings_unique": [{"text": "cat", "langs": ["en"]}],
            },
        )

    def test_build_target_string_artifact(self):
        payload = build_target_string_artifact(checkpoint_id="rev", layer_indices=[0, "2"])
        self.assertEqual(payload, {"checkpoint_id": "rev", "layer_indices": [0, 2], "records": []})

    def test_add_menu_mass_summary(self):
        import torch

        meta = {"target_string_scoring_mode": "start_tokens_only"}
        add_menu_mass_summary(
            meta,
            menu_ambiguous_mass=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            menu_unique_mass=torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
        )
        self.assertAlmostEqual(meta["mean_menu_ambiguous_mass_by_layer"][0], 0.2)
        self.assertAlmostEqual(meta["mean_menu_ambiguous_mass_by_layer"][1], 0.3)
        self.assertAlmostEqual(meta["mean_menu_unique_mass_by_layer"][0], 0.8)
        self.assertAlmostEqual(meta["mean_menu_unique_mass_by_layer"][1], 0.7)

    def test_build_teacher_forced_candidates_preserves_menu_order(self):
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False):
                _ = add_special_tokens
                return {"input_ids": list(range(1, len(text.split()) + 1))}

        records = [
            {
                "prompt_id": "p1",
                "tgt_lang": "fr",
                "tgt_text": "chat",
                "menu_strings_grouped": [
                    {"text": "cat", "langs": ["en"]},
                    {"text": "chat", "langs": ["fr"]},
                ],
            }
        ]
        candidates = build_teacher_forced_candidates(
            model=SimpleNamespace(tokenizer=FakeTokenizer()),
            prompts=["English: cat French:"],
            prompt_target_records=records,
            layer_indices=[0, 2],
        )
        self.assertEqual(
            candidates.sequences,
            ["English: cat French:cat", "English: cat French:chat"],
        )
        self.assertEqual([meta["group_idx"] for meta in candidates.metadata], [0, 1])
        self.assertEqual([bool(meta["is_tgt"]) for meta in candidates.metadata], [False, True])
        self.assertEqual(candidates.sequence_token_lengths, [3, 3])
        self.assertEqual(tuple(candidates.group_sum_logprobs_by_prompt[0].shape), (2, 2))
        self.assertEqual(
            tuple(candidates.group_first_token_logprobs_by_prompt[0].shape),
            (2, 2),
        )
        self.assertEqual(candidates.group_token_counts_by_prompt, [[0, 0]])

    def test_finite_exp_maps_non_finite_entries_to_zero(self):
        import torch

        values = torch.tensor([0.0, -float("inf"), float("nan")])
        actual = finite_exp(values)
        self.assertTrue(torch.allclose(actual[:2], torch.tensor([1.0, 0.0])))
        self.assertEqual(float(actual[2]), 0.0)

    def test_prepare_start_token_groups_returns_groups_and_counts(self):
        records = [
            {
                "prompt_id": "p1",
                "menu_strings_grouped": [
                    {"text": "cat", "langs": ["en"]},
                    {"text": "chat", "langs": ["fr"]},
                ],
                "menu_start_tokens_grouped": [
                    {"text": "cat", "langs": ["en"], "start_token_ids": [1, 2]},
                    {"text": "chat", "langs": ["fr"], "start_token_ids": [3]},
                ],
                "wendler_keep": True,
            }
        ]
        groups, counts = prepare_start_token_groups(records, require_wendler_keep=True)
        self.assertEqual(groups, [[[1, 2], [3]]])
        self.assertEqual(counts, [[2, 1]])

    def test_prepare_start_token_groups_rejects_group_mismatch(self):
        records = [
            {
                "prompt_id": "p1",
                "menu_strings_grouped": [{"text": "cat", "langs": ["en"]}],
                "menu_start_tokens_grouped": [{"text": "chat", "langs": ["en"], "start_token_ids": [1]}],
                "wendler_keep": True,
            }
        ]
        with self.assertRaisesRegex(ValueError, "Start-token artifact group mismatch"):
            prepare_start_token_groups(records, require_wendler_keep=True)

    def test_prepare_start_token_groups_rejects_wendler_keep_false(self):
        records = [
            {
                "prompt_id": "p1",
                "menu_strings_grouped": [{"text": "cat", "langs": ["en"]}],
                "menu_start_tokens_grouped": [{"text": "cat", "langs": ["en"], "start_token_ids": [1]}],
                "wendler_keep": False,
            }
        ]
        with self.assertRaisesRegex(ValueError, "wendler_keep=false"):
            prepare_start_token_groups(records, require_wendler_keep=True)

    def test_start_token_scoring_uses_start_token_keys_without_teacher_forced_outputs(self):
        import torch

        class FakeLens:
            def project_layer(self, hidden, layer_idx):
                _ = hidden, layer_idx
                return torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)

        records = {
            "p1": {
                "prompt_id": "p1",
                "concept_id": "concept",
                "task": "translation",
                "prompt": "English: cat\nFrench:",
                "prompt_lang": "en",
                "tgt_lang": "fr",
                "tgt_text": "chat",
                "menu_strings_grouped": [
                    {"text": "cat", "langs": ["en"]},
                    {"text": "chat", "langs": ["fr"]},
                ],
                "menu_strings_unique": [
                    {"text": "cat", "langs": ["en"]},
                    {"text": "chat", "langs": ["fr"]},
                ],
                "menu_has_ambiguity": False,
                "menu_group_count": 2,
                "menu_unique_count": 2,
                "menu_start_tokens_grouped": [
                    {"text": "cat", "langs": ["en"], "start_token_ids": [0]},
                    {"text": "chat", "langs": ["fr"], "start_token_ids": [2]},
                ],
                "wendler_keep": True,
            }
        }
        with (
            mock.patch(
                "target_string.prepare_unembed_info",
                return_value=SimpleNamespace(device=torch.device("cpu")),
            ),
            mock.patch(
                "target_string.collect_position_latents",
                return_value=(torch.zeros(1, 1, 4), [0]),
            ),
        ):
            result = run_start_token_scoring(
                model=SimpleNamespace(tokenizer=None),
                prompts=["English: cat\nFrench:"],
                prompt_ids=["p1"],
                target_string_records=records,
                layer_indices=[0],
                trace_batch_size=1,
                decoding_lens_obj=FakeLens(),
                require_wendler_keep=True,
                rev="test",
                prompt_token_lengths=[4],
            )

        detail = result.target_string_artifact["records"][0]
        self.assertIn("menu_start_token_logprobs", detail)
        self.assertIn("tgt_first_token_vocab_entropy_bits", detail)
        self.assertIn("lang_mass_start_tokens", detail)
        self.assertIn("lang_dist_start_tokens", detail)
        self.assertNotIn("menu_teacher_forced_sum_logprobs", detail)
        self.assertEqual(result.decoding_meta["target_string_scoring_mode"], "start_tokens_only")
        self.assertIsNotNone(result.tgt_first_token_vocab_entropy_bits)
        log_probs = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=-1)
        expected_entropy_bits = -(torch.exp(log_probs) * log_probs).sum() / torch.log(torch.tensor(2.0))
        self.assertTrue(
            torch.allclose(
                result.tgt_first_token_vocab_entropy_bits,
                expected_entropy_bits.reshape(1, 1),
            )
        )
        self.assertTrue(
            torch.allclose(
                detail["tgt_first_token_vocab_entropy_bits"],
                expected_entropy_bits.reshape(1),
            )
        )
        dist_sum = result.lang_dist_dec["en"] + result.lang_dist_dec["fr"]
        self.assertTrue(torch.allclose(dist_sum, torch.ones_like(dist_sum)))

    def test_start_token_scoring_streams_prompt_latent_batches(self):
        import torch

        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=True):
                _ = add_special_tokens
                return {"input_ids": list(range(len(text.split()) + 1))}

        class FakeLens:
            def project_layer(self, hidden, layer_idx):
                _ = hidden, layer_idx
                return torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)

        records = {}
        for idx, prompt_id in enumerate(["p1", "p2"]):
            records[prompt_id] = {
                "prompt_id": prompt_id,
                "concept_id": f"concept-{idx}",
                "task": "copy",
                "prompt": f"Prompt {idx}:",
                "prompt_lang": "en",
                "tgt_lang": "fr",
                "tgt_text": "chat",
                "menu_strings_grouped": [
                    {"text": "cat", "langs": ["en"]},
                    {"text": "chat", "langs": ["fr"]},
                ],
                "menu_strings_unique": [
                    {"text": "cat", "langs": ["en"]},
                    {"text": "chat", "langs": ["fr"]},
                ],
                "menu_has_ambiguity": False,
                "menu_group_count": 2,
                "menu_unique_count": 2,
                "menu_start_tokens_grouped": [
                    {"text": "cat", "langs": ["en"], "start_token_ids": [0]},
                    {"text": "chat", "langs": ["fr"], "start_token_ids": [2]},
                ],
                "wendler_keep": True,
            }
        collect_calls = []
        collect_positions = []

        def fake_collect(*, prompts, layer_indices, **kwargs):
            collect_calls.append(list(prompts))
            collect_positions.append(kwargs.get("positions_by_prompt"))
            return torch.zeros(len(prompts), len(layer_indices), 4), list(layer_indices)

        with (
            mock.patch(
                "target_string.prepare_unembed_info",
                return_value=SimpleNamespace(device=torch.device("cpu")),
            ),
            mock.patch("target_string.collect_position_latents", side_effect=fake_collect),
        ):
            result = run_start_token_scoring(
                model=SimpleNamespace(tokenizer=FakeTokenizer()),
                prompts=["Prompt 0:", "Prompt 1:"],
                prompt_ids=["p1", "p2"],
                target_string_records=records,
                layer_indices=[0],
                trace_batch_size=1,
                decoding_lens_obj=FakeLens(),
                require_wendler_keep=True,
                rev="test",
                prompt_anchor_positions=[3, 4],
            )

        self.assertEqual(collect_calls, [["Prompt 0:"], ["Prompt 1:"]])
        self.assertEqual(collect_positions, [[3], [4]])
        self.assertEqual(len(result.target_string_artifact["records"]), 2)
        dist_sum = result.lang_dist_dec["en"] + result.lang_dist_dec["fr"]
        self.assertTrue(torch.allclose(dist_sum, torch.ones_like(dist_sum)))

    def test_teacher_forced_scoring_builds_full_target_artifact(self):
        import torch

        class FakeTokenizer:
            token_ids = {
                "Prompt:": [0],
                "cat": [1],
                "chat": [2],
                "Prompt:cat": [0, 1],
                "Prompt:chat": [0, 2],
            }

            def __call__(self, text, add_special_tokens=False):
                _ = add_special_tokens
                return {"input_ids": self.token_ids[text]}

            def convert_ids_to_tokens(self, token_ids):
                return [f"token-{token_id}" for token_id in token_ids]

        class FakeLens:
            def project_layer(self, hidden, layer_idx):
                _ = layer_idx
                logits = torch.tensor(
                    [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]],
                    dtype=torch.float32,
                )
                return logits[: hidden.shape[0]]

        records = {
            "p1": {
                "prompt_id": "p1",
                "concept_id": "concept",
                "task": "translation",
                "prompt": "Prompt:",
                "prompt_lang": "en",
                "tgt_lang": "fr",
                "tgt_text": "chat",
                "menu_strings_grouped": [
                    {"text": "cat", "langs": ["en"]},
                    {"text": "chat", "langs": ["fr"]},
                ],
                "menu_strings_unique": [
                    {"text": "cat", "langs": ["en"]},
                    {"text": "chat", "langs": ["fr"]},
                ],
                "menu_has_ambiguity": False,
                "menu_group_count": 2,
                "menu_unique_count": 2,
            }
        }

        def fake_collect(*, prompts, layer_indices, sequence_lengths, **kwargs):
            _ = kwargs
            max_length = max(sequence_lengths)
            return (
                torch.zeros(len(prompts), len(layer_indices), max_length, 4),
                list(layer_indices),
            )

        with (
            mock.patch(
                "target_string.prepare_unembed_info",
                return_value=SimpleNamespace(device=torch.device("cpu")),
            ),
            mock.patch(
                "target_string.collect_sequence_latents",
                side_effect=fake_collect,
            ),
        ):
            result = run_teacher_forced_scoring(
                model=SimpleNamespace(tokenizer=FakeTokenizer()),
                prompts=["Prompt:"],
                prompt_ids=["p1"],
                target_string_records=records,
                layer_indices=[0],
                trace_batch_size=2,
                decoding_lens_obj=FakeLens(),
                rev="test",
            )

        detail = result.target_string_artifact["records"][0]
        self.assertIn("menu_teacher_forced_sum_logprobs", detail)
        self.assertIn("tgt_step_vocab_entropy_bits", detail)
        self.assertEqual(result.decoding_meta["target_string_scoring_mode"], "multi_token_teacher_forced")
        self.assertEqual(result.tgt_token_count_cpu, [1])
        dist_sum = result.lang_dist_dec["en"] + result.lang_dist_dec["fr"]
        self.assertTrue(torch.allclose(dist_sum, torch.ones_like(dist_sum)))

    # Chunked-softmax Start(w) tests are intentionally disabled with the
    # prototype function. The active Start(w) path uses ordinary full-vocab
    # log_softmax after streaming prompt batches.
    #
    # def test_score_start_token_group_logprobs_matches_log_softmax(self):
    #     import torch
    #
    #     hidden = torch.tensor(
    #         [
    #             [0.2, -0.4, 0.7],
    #             [1.1, 0.3, -0.2],
    #         ],
    #         dtype=torch.float64,
    #     )
    #     weight = torch.tensor(
    #         [
    #             [0.1, 0.2, -0.3],
    #             [-0.4, 0.5, 0.6],
    #             [0.7, -0.1, 0.2],
    #             [0.0, 0.3, -0.8],
    #             [0.9, -0.7, 0.4],
    #         ],
    #         dtype=torch.float64,
    #     )
    #     bias = torch.tensor([0.05, -0.2, 0.1, 0.0, -0.15], dtype=torch.float64)
    #     groups = [[0], [1, 3], [], [2, 4]]
    #     info = SimpleNamespace(weight=weight, bias=bias, device=torch.device("cpu"))
    #
    #     actual = score_start_token_group_logprobs(
    #         hidden_by_layer=hidden,
    #         group_token_ids=groups,
    #         unembed_info=info,
    #         vocab_chunk_size=2,
    #     )
    #
    #     log_probs = torch.log_softmax(hidden @ weight.T + bias, dim=-1)
    #     expected = torch.full((hidden.shape[0], len(groups)), -float("inf"), dtype=torch.float64)
    #     for group_idx, token_ids in enumerate(groups):
    #         if token_ids:
    #             expected[:, group_idx] = torch.logsumexp(log_probs[:, token_ids], dim=-1)
    #
    #     self.assertEqual(actual.dtype, torch.float64)
    #     self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12, equal_nan=True))
    #
    # def test_score_start_token_group_logprobs_without_bias_matches_log_softmax(self):
    #     import torch
    #
    #     hidden = torch.tensor([[0.5, -1.0], [-0.25, 0.75]], dtype=torch.float32)
    #     weight = torch.tensor(
    #         [
    #             [0.2, 0.1],
    #             [-0.3, 0.4],
    #             [0.8, -0.6],
    #             [0.0, 0.5],
    #         ],
    #         dtype=torch.float32,
    #     )
    #     groups = [[0, 2], [3]]
    #     info = SimpleNamespace(weight=weight, bias=None, device=torch.device("cpu"))
    #
    #     actual = score_start_token_group_logprobs(
    #         hidden_by_layer=hidden,
    #         group_token_ids=groups,
    #         unembed_info=info,
    #         vocab_chunk_size=3,
    #     )
    #     log_probs = torch.log_softmax(hidden @ weight.T, dim=-1)
    #     expected = torch.stack(
    #         [
    #             torch.logsumexp(log_probs[:, [0, 2]], dim=-1),
    #             log_probs[:, 3],
    #         ],
    #         dim=-1,
    #     )
    #
    #     self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_resolve_config_synthetic_decoding_only_does_not_require_repr_settings(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "translation",
            "do_decoding": True,
            "do_repr": False,
            "decoding_token_agg": "last_token",
            "repr_token_agg": "frac_50",
            "repr_rollout_k": 3,
        })
        resolved = resolve_config(
            cfg,
            explicit_keys={"data_source", "do_decoding", "do_repr", "decoding_token_agg"},
        )
        self.assertEqual(resolved["decoding_mapping"], "target_string")

    def test_resolve_config_translation_to_inherits_translation_defaults(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "translation_to_fr",
            "do_decoding": True,
            "do_repr": False,
            "decoding_token_agg": "last_token",
            "max_prompts": 0,
            "max_prompts_per_lang": 64,
        })
        resolved = resolve_config(
            cfg,
            explicit_keys={"data_source", "do_decoding", "do_repr", "decoding_token_agg"},
        )
        self.assertIsNone(resolved["max_prompts"])
        self.assertIsNone(resolved["max_prompts_per_lang"])
        self.assertEqual(resolved["decoding_mapping"], "target_string")

    def test_resolve_config_synthetic_repr_only_does_not_require_decoding_settings(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "translation",
            "do_decoding": False,
            "do_repr": True,
            "decoding_token_agg": "frac_50",
            "repr_token_agg": "last_token",
            "repr_rollout_k": 0,
        })
        resolved = resolve_config(
            cfg,
            explicit_keys={"data_source", "do_decoding", "do_repr", "repr_token_agg", "repr_rollout_k"},
        )
        self.assertEqual(resolved["repr_token_agg"], "last_token")

    def test_resolve_config_zero_max_prompts_disables_total_cap(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "prompts_path": "data/test_prompts.jsonl",
            "max_prompts": 0,
            "max_prompts_per_lang": 2,
        })
        resolved = resolve_config(cfg, explicit_keys={"max_prompts", "max_prompts_per_lang"})
        self.assertIsNone(resolved["max_prompts"])
        self.assertEqual(resolved["max_prompts_per_lang"], 2)

    def test_resolve_config_zero_max_prompts_per_lang_disables_lang_cap(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "prompts_path": "data/test_prompts.jsonl",
            "max_prompts": 8,
            "max_prompts_per_lang": 0,
        })
        resolved = resolve_config(cfg, explicit_keys={"max_prompts", "max_prompts_per_lang"})
        self.assertEqual(resolved["max_prompts"], 8)
        self.assertIsNone(resolved["max_prompts_per_lang"])

    def test_resolve_config_zero_zero_prompt_caps_disable_all_caps(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "prompts_path": "data/test_prompts.jsonl",
            "max_prompts": 0,
            "max_prompts_per_lang": 0,
        })
        resolved = resolve_config(cfg, explicit_keys={"max_prompts", "max_prompts_per_lang"})
        self.assertIsNone(resolved["max_prompts"])
        self.assertIsNone(resolved["max_prompts_per_lang"])

    def test_resolve_config_explicit_max_prompts_disables_default_per_lang_cap(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "pud21",
            "pud_root": "data/pud_holdout",
            "max_prompts": 8,
            "max_prompts_per_lang": 64,
        })
        resolved = resolve_config(cfg, explicit_keys={"data_source", "max_prompts"})
        self.assertEqual(resolved["max_prompts"], 8)
        self.assertIsNone(resolved["max_prompts_per_lang"])

    def test_resolve_config_pud_setup_source_sets_matching_gmm_setup(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "pud21",
            "repr_gmm_setup": None,
        })
        resolved = resolve_config(cfg, explicit_keys={"data_source"})
        self.assertEqual(resolved["prompts_path"], "data/pud_holdout/pud_prompts_test.jsonl")
        self.assertEqual(resolved["repr_gmm_setup"], "pud21")

    def test_resolve_config_pud_setup_rejects_conflicting_gmm_setup(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "pud9",
            "repr_gmm_setup": "pud21",
        })
        with self.assertRaisesRegex(ValueError, "requires matching repr_gmm_setup"):
            resolve_config(cfg, explicit_keys={"data_source", "repr_gmm_setup"})

    def test_resolve_config_ud6_source_appends_ud_prompts(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "pud21_ud6",
            "include_nonparallel_ud": False,
        })
        resolved = resolve_config(cfg, explicit_keys={"data_source"})
        self.assertEqual(resolved["repr_gmm_setup"], "pud21_ud6")
        self.assertEqual(resolved["extra_prompts_paths"], ["data/ud_holdout/ud_prompts_test.jsonl"])

    def test_resolve_pud_prompts_path(self):
        self.assertEqual(
            str(resolve_pud_prompts_path("data/pud_holdout", "overlap")),
            "data/pud_holdout/pud_prompts_train.jsonl",
        )
        self.assertEqual(
            str(resolve_pud_prompts_path("data/pud_holdout", "heldout")),
            "data/pud_holdout/pud_prompts_test.jsonl",
        )

    def test_resolve_ud_prompts_path(self):
        self.assertEqual(
            str(resolve_ud_prompts_path("data/ud_holdout", "overlap")),
            "data/ud_holdout/ud_prompts_train.jsonl",
        )
        self.assertEqual(
            str(resolve_ud_prompts_path("data/ud_holdout", "heldout")),
            "data/ud_holdout/ud_prompts_test.jsonl",
        )

    def test_resolve_include_prompts_path(self):
        self.assertEqual(
            str(resolve_include_prompts_path("data", "include_10lang_3domain_cap30", "question_only")),
            "data/include_10lang_3domain_cap30/prompts_question_only.jsonl",
        )
        self.assertEqual(
            str(resolve_include_prompts_path("data", "include_10lang_3domain_all", "minimal_mcq")),
            "data/include_10lang_3domain_all/prompts_minimal_mcq.jsonl",
        )

    def test_resolve_config_include_defaults_prompt_path(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "include_10lang_3domain_cap30",
        })
        resolved = resolve_config(cfg, explicit_keys={"data_source"})
        self.assertEqual(
            resolved["prompts_path"],
            "data/include_10lang_3domain_cap30/prompts_question_only.jsonl",
        )
        self.assertEqual(
            resolved["lid_candidate_langs"],
            ["ar", "en", "es", "fi", "fr", "hi", "id", "pt", "ru", "tr", "zh"],
        )

    def test_resolve_config_include_respects_prompt_style(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "include_10lang_3domain_all",
            "include_prompt_style": "minimal_mcq",
        })
        resolved = resolve_config(cfg, explicit_keys={"data_source", "include_prompt_style"})
        self.assertEqual(
            resolved["prompts_path"],
            "data/include_10lang_3domain_all/prompts_minimal_mcq.jsonl",
        )
        self.assertEqual(
            resolved["lid_candidate_langs"],
            ["ar", "en", "es", "fi", "fr", "hi", "id", "pt", "ru", "tr", "zh"],
        )

    def test_resolve_config_include_repr_uses_english_gmm_setup(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "include_10lang_3domain_all",
            "do_decoding": False,
            "do_repr": True,
        })
        resolved = resolve_config(
            cfg,
            explicit_keys={"data_source", "do_decoding", "do_repr"},
        )
        self.assertEqual(resolved["repr_gmm_setup"], "include_10lang_en")

        cfg["repr_gmm_setup"] = "include_10lang"
        with self.assertRaisesRegex(ValueError, "requires repr_gmm_setup='include_10lang_en'"):
            resolve_config(cfg, explicit_keys={"data_source", "repr_gmm_setup"})

    def test_resolve_config_include_rejects_candidates_without_english(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "data_source": "include_10lang_3domain_cap30",
            "lid_candidate_langs": ["ar", "es", "fi", "fr", "hi", "id", "pt", "ru", "tr", "zh"],
        })
        with self.assertRaisesRegex(ValueError, "must include English"):
            resolve_config(cfg, explicit_keys={"data_source", "lid_candidate_langs"})

    def test_resolve_config_appends_ud_prompts_when_enabled(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "pud_root": "data/pud_holdout",
            "ud_root": "data/ud_holdout",
            "include_nonparallel_ud": True,
        })
        resolved = resolve_config(cfg, explicit_keys={"include_nonparallel_ud"})
        self.assertEqual(resolved["prompts_path"], "data/pud_holdout/pud_prompts_test.jsonl")
        self.assertEqual(resolved["extra_prompts_paths"], ["data/ud_holdout/ud_prompts_test.jsonl"])

    def test_load_gmm_setup_registry(self):
        registry = load_gmm_setup_registry("configs/gmm_setups.json")
        self.assertIn("pud21", registry)
        self.assertEqual(registry["pud21"]["languages"][0], "ar")

    def test_infer_gmm_setup_name(self):
        registry = load_gmm_setup_registry("configs/gmm_setups.json")
        name = infer_gmm_setup_name(["fr", "en", "ar", "hi", "cs", "es", "pt", "is", "id"], registry)
        self.assertEqual(name, "pud9")

    def test_resolve_repr_gmm_dir_prefers_explicit_setup(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "repr_gmm_dir": None,
            "repr_gmm_setup": "pud21",
        })
        path, setup_name, setup = resolve_repr_gmm_dir(
            cfg,
            prompt_langs=["en", "fr"],
            revision="main",
        )
        self.assertEqual(setup_name, "pud21")
        self.assertEqual(setup["languages"][0], "ar")
        self.assertEqual(
            path,
            build_gmm_artifact_dir(
                base_dir="logs/gmm",
                setup_name="pud21",
                model_name="org/model",
                revision="main",
            ),
        )

    def test_resolve_repr_gmm_dir_infers_from_prompts(self):
        cfg = load_config("configs/default.json")
        cfg.update({
            "model_name": "org/model",
            "repr_gmm_dir": None,
            "repr_gmm_setup": None,
        })
        path, setup_name, _ = resolve_repr_gmm_dir(
            cfg,
            prompt_langs=["fr", "en", "ar", "hi", "cs", "es", "pt", "is", "id"],
            revision="main",
        )
        self.assertEqual(setup_name, "pud9")
        self.assertIn("pud9", str(path))

    def test_load_jsonl_prompt_data_uses_source_lang_for_caps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pud_path = Path(tmpdir) / "pud.jsonl"
            ud_path = Path(tmpdir) / "ud.jsonl"
            pud_path.write_text(
                json.dumps(
                    {
                        "id": "en_pud-0",
                        "prompt_text": "hello",
                        "surface_tokens": ["hello"],
                    }
                ),
                encoding="utf-8",
            )
            ud_path.write_text(
                json.dumps(
                    {
                        "id": "uk_iu-test-0",
                        "prompt_text": "привіт",
                        "source_lang": "uk",
                        "surface_tokens": ["привіт"],
                    }
                ),
                encoding="utf-8",
            )

            prompt_data = load_jsonl_prompt_data(
                [pud_path, ud_path],
                max_prompts=None,
                max_prompts_per_lang=1,
            )

            self.assertEqual(prompt_data.prompts, ["hello", "привіт"])
            self.assertEqual(prompt_data.prompt_ids, ["en_pud-0", "uk_iu-test-0"])
            self.assertEqual(prompt_data.prompt_langs, ["en", "uk"])

    def test_infer_task_langs_uses_prompt_task_languages(self):
        langs = infer_task_langs(
            [("ar",), ("fi",), ("ru", "tr"), ("zh",)],
            ["ar", "fi", "ru", "zh"],
        )
        self.assertEqual(langs, ["ar", "fi", "ru", "tr", "zh"])

    def test_validate_pud_prompt_languages_rejects_unexpected_languages(self):
        with self.assertRaisesRegex(ValueError, "unexpected languages"):
            validate_pud_prompt_languages(
                "pud9",
                ["en", "fr", "zh"],
                "configs/gmm_setups.json",
            )

    def test_validate_pud_prompt_languages_accepts_subset(self):
        validate_pud_prompt_languages(
            "pud9",
            ["en", "fr"],
            "configs/gmm_setups.json",
        )

    def test_load_jsonl_prompt_data_filters_before_caps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.jsonl"
            rows = [
                {"id": "fr_pud-0", "prompt_text": "bonjour", "surface_tokens": ["bonjour"]},
                {"id": "de_pud-0", "prompt_text": "hallo", "surface_tokens": ["hallo"]},
                {"id": "en_pud-0", "prompt_text": "hello", "surface_tokens": ["hello"]},
                {"id": "en_pud-1", "prompt_text": "hi", "surface_tokens": ["hi"]},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )

            prompt_data = load_jsonl_prompt_data(
                [path],
                max_prompts=None,
                max_prompts_per_lang=1,
                selected_task_langs={"en"},
            )

            self.assertEqual(prompt_data.prompts, ["hello"])
            self.assertEqual(prompt_data.prompt_ids, ["en_pud-0"])
            self.assertEqual(prompt_data.prompt_langs, ["en"])

    def test_translation_to_resolves_translation_csv_and_filters_target_lang(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "translation.csv"
            path.write_text(
                "\n".join(
                    [
                        "input_lang,target_lang,source_file,source_row,prompt,out_token_str,latent_token_str",
                        "en,fr,src,1,Translate hello,bonjour,hello",
                        "de,fr,src,2,Translate hallo,bonjour,hallo",
                        "en,zh,src,3,Translate book,书,book",
                    ]
                ),
                encoding="utf-8",
            )

            resolved = resolve_synthetic_csv_path(root, "translation_to_fr")
            prompt_data = load_synthetic_prompt_data(
                resolved,
                "translation_to_fr",
                max_prompts=None,
                max_prompts_per_lang=None,
            )

            self.assertEqual(resolved, path)
            self.assertEqual(prompt_data.prompts, ["Translate hello", "Translate hallo"])
            self.assertEqual(prompt_data.prompt_langs, ["en", "de"])
            self.assertEqual(prompt_data.task_langs_by_prompt, [("en", "fr"), ("de", "fr")])
            self.assertEqual(
                prompt_data.target_string_records[prompt_data.prompt_ids[0]]["tgt_lang"],
                "fr",
            )
            self.assertEqual(set(prompt_data.target_string_records), set(prompt_data.prompt_ids))

    def test_target_string_step_scoring_can_return_vocab_entropy(self):
        import math
        import torch

        class FakeLens:
            def project_layer(self, hidden, layer_idx):
                _ = hidden, layer_idx
                return torch.tensor(
                    [
                        [2.0, 0.0, -1.0],
                        [0.0, 1.0, 2.0],
                        [-1.0, 3.0, 0.0],
                    ],
                    dtype=torch.float32,
                )

        logprobs, entropy = score_teacher_forced_one_candidate(
            latents=torch.zeros(1, 3, 4),
            prompt_len=1,
            target_tokens=torch.tensor([2, 1]),
            unembed_info=SimpleNamespace(device=torch.device("cpu")),
            decoding_lens=FakeLens(),
            layer_indices=[0],
            return_vocab_entropy_bits=True,
        )

        expected_logprobs = torch.log_softmax(FakeLens().project_layer(None, 0), dim=-1)[[0, 1], [2, 1]]
        probs = torch.softmax(FakeLens().project_layer(None, 0)[:2], dim=-1)
        expected_entropy = (-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1) / math.log(2)).unsqueeze(0)
        self.assertTrue(torch.allclose(logprobs, expected_logprobs.unsqueeze(0)))
        self.assertTrue(torch.allclose(entropy, expected_entropy))

    def test_config_file_override_arg_is_disabled(self):
        defaults = load_config("configs/default.json")
        parser = build_parser(defaults)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--config", "configs/test.json"])

    def test_config_precedence_default_cli(self):
        defaults = load_config("configs/default.json")
        parser = build_parser(defaults)
        argv = ["--model-name", "cli-model", "--seed", "999"]
        args = parser.parse_args(argv)
        cli_keys = get_explicit_cli_keys(parser, argv)

        cfg = dict(defaults)
        explicit_keys = set()
        for dest in cli_keys:
            cfg[dest] = getattr(args, dest)
            explicit_keys.add(dest)
        resolved = resolve_config(cfg, explicit_keys)

        self.assertEqual(resolved["model_name"], "cli-model")
        self.assertEqual(resolved["prompts_path"], "data/pud_holdout/pud_prompts_test.jsonl")
        self.assertEqual(resolved["seed"], 999)
        self.assertEqual(resolved["lid_candidate_langs"], [])

    def test_explicit_cli_keys_detects_equals_form(self):
        defaults = load_config("configs/default.json")
        parser = build_parser(defaults)
        explicit_keys = get_explicit_cli_keys(
            parser,
            ["--trace-batch-size=7", "--sampled-rollout-prompt-batch-size=3"],
        )
        self.assertIn("trace_batch_size", explicit_keys)
        self.assertIn("sampled_rollout_prompt_batch_size", explicit_keys)

    def test_legacy_eval_langs_cli_alias_uses_lid_candidate_langs(self):
        defaults = load_config("configs/default.json")
        parser = build_parser(defaults)
        argv = ["--eval-langs", "en", "fr"]

        parsed = parser.parse_args(argv)
        explicit_keys = get_explicit_cli_keys(parser, argv)

        self.assertEqual(parsed.lid_candidate_langs, ["en", "fr"])
        self.assertIn("lid_candidate_langs", explicit_keys)

    def test_legacy_eval_langs_config_key_is_migrated(self):
        cfg = load_config("configs/default.json")
        cfg.pop("lid_candidate_langs")
        cfg["eval_langs"] = ["fr", "en", "fr"]

        resolved = resolve_config(cfg, explicit_keys=set())

        self.assertEqual(resolved["lid_candidate_langs"], ["en", "fr"])
        self.assertNotIn("eval_langs", resolved)

    def test_explicit_cli_keys_follow_parser_order(self):
        defaults = load_config("configs/default.json")
        parser = build_parser(defaults)

        explicit_keys = get_explicit_cli_keys(
            parser,
            ["--seed", "999", "--model-name", "cli-model"],
        )

        self.assertEqual(explicit_keys, ["model_name", "seed"])

    def test_gather_fixed_position_latents(self):
        import torch

        full_latents = [
            torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        ]
        selected = gather_fixed_position_latents(full_latents, position=-1)
        self.assertEqual(selected.shape, (1, 2, 4))
        self.assertTrue(torch.allclose(selected[0, 0], full_latents[0][0, -1]))

    def test_select_repr_latents_fractional_modes(self):
        import torch

        layer_latents = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3)

        frac_50 = select_repr_latents(layer_latents, "frac_50")
        frac_75 = select_repr_latents(layer_latents, "frac_75")

        self.assertEqual(frac_50.shape, (1, 3))
        self.assertEqual(frac_75.shape, (1, 3))
        self.assertTrue(torch.allclose(frac_50[0], layer_latents[2]))
        self.assertTrue(torch.allclose(frac_75[0], layer_latents[3]))

    def test_select_repr_latents_fractional_modes_snap_to_word_end(self):
        import torch

        layer_latents = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)
        word_ids = [0, 0, 1, 2, 2, 3]

        frac_75 = select_repr_latents(layer_latents, "frac_75", word_ids=word_ids)

        self.assertTrue(torch.allclose(frac_75[0], layer_latents[4]))

    def test_select_repr_latents_rollout_span(self):
        import torch

        layer_latents = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)

        selected = select_repr_latents(layer_latents, "frac_50", repr_rollout_k=2)

        self.assertEqual(selected.shape, (3, 3))
        self.assertTrue(torch.allclose(selected[0], layer_latents[2]))
        self.assertTrue(torch.allclose(selected[1], layer_latents[3]))
        self.assertTrue(torch.allclose(selected[2], layer_latents[4]))

    def test_select_repr_latents_rollout_rejects_all_tokens(self):
        import torch

        layer_latents = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)

        with self.assertRaisesRegex(ValueError, "repr_rollout_k cannot be combined"):
            select_repr_latents(layer_latents, "all_tokens", repr_rollout_k=1)

    def test_select_repr_word_rollout_latents_uses_touched_words(self):
        import torch

        layer_latents = torch.arange(7 * 2, dtype=torch.float32).reshape(7, 2)
        word_ids = [0, 0, 1, 1, 2, 3, 3]

        selected = select_repr_word_rollout_latents(
            layer_latents,
            "frac_50",
            word_ids=word_ids,
            repr_rollout_k=2,
        )

        word1 = layer_latents[2:4].mean(dim=0)
        word2 = layer_latents[4:5].mean(dim=0)
        word3 = layer_latents[5:7].mean(dim=0)
        self.assertEqual(selected.shape, (3, 2))
        self.assertTrue(torch.allclose(selected[0], word1))
        self.assertTrue(torch.allclose(selected[1], word2))
        self.assertTrue(torch.allclose(selected[2], word3))

    def test_select_decoding_latents_fractional_modes(self):
        import torch

        full_latents = [
            torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
            torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3) + 100.0,
        ]

        frac_50 = select_decoding_latents(full_latents, mode="frac_50")
        frac_75 = select_decoding_latents(full_latents, mode="frac_75")

        self.assertEqual(frac_50.shape, (2, 2, 3))
        self.assertEqual(frac_75.shape, (2, 2, 3))
        self.assertTrue(torch.allclose(frac_50[0, 0], full_latents[0][0, 1]))
        self.assertTrue(torch.allclose(frac_50[1, 0], full_latents[1][0, 2]))
        self.assertTrue(torch.allclose(frac_75[0, 0], full_latents[0][0, 2]))
        self.assertTrue(torch.allclose(frac_75[1, 0], full_latents[1][0, 3]))

    def test_select_decoding_latents_fractional_modes_snap_to_word_end(self):
        import torch

        full_latents = [
            torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3),
        ]

        frac_75 = select_decoding_latents(
            full_latents,
            mode="frac_75",
            word_ids_list=[[0, 0, 1, 2, 2, 3]],
        )

        self.assertTrue(torch.allclose(frac_75[0, 0], full_latents[0][0, 4]))

    def test_select_decoding_latents_last_token(self):
        import torch

        full_latents = [
            torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
        ]

        last_token = select_decoding_latents(full_latents, mode="last_token")

        self.assertTrue(torch.allclose(last_token[0, 0], full_latents[0][0, -1]))

    def test_get_word_end_token_positions(self):
        self.assertEqual(get_word_end_token_positions([0, 0, None, 1, 2, 2]), [1, 3, 5])

    def test_get_word_end_token_positions_ignores_special_token_gaps_without_shifting_words(self):
        self.assertEqual(get_word_end_token_positions([None, 0, 0, 1, 1, None]), [2, 4])

    def test_select_fractional_token_position_snaps_to_word_end(self):
        pos = select_fractional_token_position(6, "frac_75", word_ids=[0, 0, 1, 2, 2, 3])
        self.assertEqual(pos, 4)

    def test_select_fractional_token_position_handles_special_token_nones(self):
        pos = select_fractional_token_position(6, "frac_50", word_ids=[None, 0, 0, 1, 1, None])
        self.assertEqual(pos, 2)

    def test_select_rollout_token_positions_word_count_uses_future_word_targets(self):
        positions = select_rollout_token_positions(
            8,
            "frac_50",
            rollout_k=0,
            rollout_word_cnt=1,
            word_ids=[0, 0, 1, 2, 2, 3, 4, 4],
        )
        self.assertEqual(positions, [4, 5, 6])

    def test_select_rollout_token_positions_frac75_word_count_clips_cleanly_near_prompt_end(self):
        positions = select_rollout_token_positions(
            8,
            "frac_75",
            rollout_k=0,
            rollout_word_cnt=3,
            word_ids=[0, 0, 1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(positions, [5, 6])

    def test_select_rollout_token_positions_frac75_word_count_stays_in_range_with_room_left(self):
        positions = select_rollout_token_positions(
            14,
            "frac_75",
            rollout_k=0,
            rollout_word_cnt=3,
            word_ids=[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        )
        self.assertEqual(positions, [10, 11, 12])

    def test_compute_surface_token_spans_uses_surface_tokens(self):
        spans = compute_surface_token_spans("aux amis.", ["aux", "amis", "."])
        self.assertEqual(spans, [(0, 3), (4, 8), (8, 9)])

    def test_compute_surface_token_spans_respects_content_start_offset(self):
        spans = compute_surface_token_spans("<s>[INST]hello world[/INST]", ["hello", "world"], start_pos=9)
        self.assertEqual(spans, [(9, 14), (15, 20)])

    def test_text_chunk_ids_require_surface_tokens(self):
        with self.assertRaises(ValueError):
            get_surface_word_ids_for_prompts(self.gpt2_tokenizer, ["hello world"])

    def test_text_chunk_id_recovery_uses_surface_tokens(self):
        formatted_prompts, content_offsets = format_prompts_for_tokenizer(
            self.llama2_tokenizer,
            ["aux amis."],
        )
        chunk_ids = get_surface_word_ids_for_prompts(
            self.llama2_tokenizer,
            formatted_prompts,
            surface_tokens_by_prompt=[["aux", "amis", "."]],
            content_start_offsets=content_offsets,
        )[0]
        self.assertEqual(chunk_ids, [None, 0, 1, 1, 2])

    def test_rollout_word_alignment_for_paper_models(self):
        records_by_lang = {}
        with Path("data/pud_holdout/pud_prompts_test.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                lang = str(obj["id"]).split("_", 1)[0]
                if lang in records_by_lang:
                    continue
                records_by_lang[lang] = (obj["id"], obj["prompt_text"], obj["surface_tokens"])

        for model_label, model_name in ROLLOUT_ALIGNMENT_MODEL_TOKENIZERS.items():
            tokenizer = self.load_model_tokenizer(model_name)
            with self.subTest(model=model_label):
                for prompt_id, prompt_text, surface_tokens in records_by_lang.values():
                    formatted_prompts, content_offsets = format_prompts_for_tokenizer(tokenizer, [prompt_text])
                    word_ids = get_surface_word_ids_for_prompts(
                        tokenizer,
                        formatted_prompts,
                        surface_tokens_by_prompt=[surface_tokens],
                        content_start_offsets=content_offsets,
                    )[0]
                    input_ids = tokenizer(formatted_prompts[0], add_special_tokens=True)["input_ids"]
                    self.assertEqual(
                        len(word_ids),
                        len(input_ids),
                        msg=f"{model_label} mismatch on {prompt_id}: len(word_ids)={len(word_ids)} vs len(input_ids)={len(input_ids)}",
                    )

    def test_format_prompts_for_tokenizer_keeps_raw_prompts_without_chat_template(self):
        formatted_prompts, content_offsets = format_prompts_for_tokenizer(
            self.llama2_tokenizer,
            ["hello world"],
        )
        self.assertEqual(formatted_prompts, ["hello world"])
        self.assertEqual(content_offsets, [0])

    def test_format_prompts_for_tokenizer_uses_chat_template_when_available(self):
        formatted_prompts, content_offsets = format_prompts_for_tokenizer(
            self.llama31_instruct_tokenizer,
            ["hello world"],
        )
        self.assertNotEqual(formatted_prompts, ["hello world"])
        self.assertIn("hello world", formatted_prompts[0])
        self.assertGreater(content_offsets[0], 0)

    def test_chat_template_moves_last_token_after_raw_synthetic_prompt(self):
        prompt_text = 'French: "Le chat dort." - Chinese: "'
        formatted_prompts, content_offsets = format_prompts_for_tokenizer(
            self.llama31_instruct_tokenizer,
            [prompt_text],
        )
        formatted_prompt = formatted_prompts[0]
        content_start = content_offsets[0]
        content_end = content_start + len(prompt_text)

        self.assertNotEqual(formatted_prompt, prompt_text)
        self.assertEqual(formatted_prompt[content_start:content_end], prompt_text)
        self.assertGreater(
            len(formatted_prompt),
            content_end,
            msg=(
                "Expected the chat template to append generation-control text "
                "after the raw synthetic prefix."
            ),
        )

        tokenized_raw = self.llama31_instruct_tokenizer(
            prompt_text,
            add_special_tokens=True,
        )["input_ids"]
        tokenized_formatted = self.llama31_instruct_tokenizer(
            formatted_prompt,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        final_token_start, final_token_end = tokenized_formatted["offset_mapping"][-1]
        self.assertGreaterEqual(
            final_token_start,
            content_end,
            msg=(
                "For this instruct tokenizer, position=-1 after chat formatting "
                "is not part of the raw Wendler synthetic prefix."
            ),
        )
        self.assertGreater(len(tokenized_formatted["input_ids"]), len(tokenized_raw))

    def test_content_end_position_selects_raw_synthetic_prompt_inside_chat_template(self):
        prompt_text = 'French: "Le chat dort." - Chinese: "'
        formatted_prompts, content_offsets = format_prompts_for_tokenizer(
            self.llama31_instruct_tokenizer,
            [prompt_text],
        )
        position = get_prompt_content_end_token_positions(
            self.llama31_instruct_tokenizer,
            formatted_prompts,
            [prompt_text],
            content_offsets,
        )[0]
        tokenized_formatted = self.llama31_instruct_tokenizer(
            formatted_prompts[0],
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        content_end = content_offsets[0] + len(prompt_text)
        start, end = tokenized_formatted["offset_mapping"][position]

        self.assertLess(position, len(tokenized_formatted["input_ids"]) - 1)
        self.assertLessEqual(end, content_end)
        self.assertGreater(end, content_offsets[0])
        next_start, next_end = tokenized_formatted["offset_mapping"][position + 1]
        self.assertTrue(
            next_start >= content_end or next_start == next_end,
            msg="The selected anchor should be the final token belonging to the raw prompt.",
        )

    def test_open_ended_pud_include_policy_keeps_generation_position(self):
        self.assertTrue(should_anchor_to_raw_prompt_content("translation_to_zh"))
        self.assertTrue(should_anchor_to_raw_prompt_content("copy"))
        self.assertTrue(should_anchor_to_raw_prompt_content("cloze"))
        for data_source in ["pud21", "pud21_ud6", "include_10lang_3domain_cap30"]:
            with self.subTest(data_source=data_source):
                self.assertFalse(should_anchor_to_raw_prompt_content(data_source))

    def test_raw_content_anchor_would_change_pud_and_include_chat_positions(self):
        examples = {
            "pud21": "The weather is nice today and we will walk to the park.",
            "include_10lang_3domain_cap30": "Question: Quelle option décrit le mieux le passage?\nAnswer:",
        }
        for data_source, prompt_text in examples.items():
            with self.subTest(data_source=data_source):
                formatted_prompts, content_offsets = format_prompts_for_tokenizer(
                    self.llama31_instruct_tokenizer,
                    [prompt_text],
                )
                raw_content_position = get_prompt_content_end_token_positions(
                    self.llama31_instruct_tokenizer,
                    formatted_prompts,
                    [prompt_text],
                    content_offsets,
                )[0]
                tokenized_formatted = self.llama31_instruct_tokenizer(
                    formatted_prompts[0],
                    add_special_tokens=True,
                )
                generation_position = len(tokenized_formatted["input_ids"]) - 1

                self.assertLess(raw_content_position, generation_position)
                self.assertFalse(should_anchor_to_raw_prompt_content(data_source))

    def test_frac50_for_pud_include_uses_raw_prompt_surface_words_under_chat_template(self):
        examples = {
            "pud21": (
                "The weather is nice today and we will walk to the park.",
                ["The", "weather", "is", "nice", "today", "and", "we", "will", "walk", "to", "the", "park", "."],
            ),
            "include_10lang_3domain_cap30": (
                "Question: Quelle option décrit le mieux le passage?\nAnswer:",
                ["Question", ":", "Quelle", "option", "décrit", "le", "mieux", "le", "passage", "?", "Answer", ":"],
            ),
        }
        for data_source, (prompt_text, surface_tokens) in examples.items():
            with self.subTest(data_source=data_source):
                formatted_prompts, content_offsets = format_prompts_for_tokenizer(
                    self.llama31_instruct_tokenizer,
                    [prompt_text],
                )
                word_ids = get_surface_word_ids_for_prompts(
                    self.llama31_instruct_tokenizer,
                    formatted_prompts,
                    surface_tokens_by_prompt=[surface_tokens],
                    content_start_offsets=content_offsets,
                )[0]
                tokenized_formatted = self.llama31_instruct_tokenizer(
                    formatted_prompts[0],
                    add_special_tokens=True,
                    return_offsets_mapping=True,
                )
                position = select_fractional_token_position(
                    len(tokenized_formatted["input_ids"]),
                    "frac_50",
                    word_ids=word_ids,
                )
                start, end = tokenized_formatted["offset_mapping"][position]
                content_start = content_offsets[0]
                content_end = content_start + len(prompt_text)

                self.assertGreater(position, 0)
                self.assertLess(position, len(tokenized_formatted["input_ids"]) - 1)
                self.assertGreaterEqual(start, content_start)
                self.assertLessEqual(end, content_end)
                self.assertFalse(should_anchor_to_raw_prompt_content(data_source))

    def test_format_prompts_for_tokenizer_matches_chat_template_tokenization_without_double_bos(self):
        messages = [{"role": "user", "content": "hello world"}]
        formatted_prompts, _ = format_prompts_for_tokenizer(
            self.llama31_instruct_tokenizer,
            ["hello world"],
        )
        tokenized_formatted = self.llama31_instruct_tokenizer(
            formatted_prompts[0],
            add_special_tokens=True,
        )["input_ids"]
        tokenized_chat_template = self.llama31_instruct_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        self.assertEqual(tokenized_formatted, tokenized_chat_template)

    def test_chat_template_availability_for_paper_models(self):
        for model_label, (model_name, expected_has_chat_template) in CHAT_TEMPLATE_AVAILABILITY.items():
            tokenizer = self.load_model_tokenizer(model_name)
            with self.subTest(model=model_label):
                self.assertEqual(bool(getattr(tokenizer, "chat_template", None)), expected_has_chat_template)

    def test_llama2_tokenizer_prepends_bos_as_none_word_id(self):
        without_specials = self.llama2_tokenizer(
            "hello world",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        with_specials = self.llama2_tokenizer(
            "hello world",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        self.assertEqual(
            self.llama2_tokenizer.convert_ids_to_tokens(with_specials["input_ids"])[:3],
            ["<s>", "\u2581hello", "\u2581world"],
        )
        self.assertEqual(without_specials.word_ids(), [0, 0])
        self.assertEqual(with_specials.word_ids(), [None, 0, 0])
        self.assertEqual(
            get_word_end_token_positions(with_specials.word_ids()),
            [2],
        )

    def test_teacher_forced_argmax_decode_then_classify(self):
        """Test the teacher-forced rollout scorer used by run_eval.py."""
        import torch

        class FakeTokenizer:
            def decode(self, ids):
                if len(ids) == 1:
                    return "a" if ids[0] == 0 else "b"
                return "a b" if ids == [0, 1] else "b a"

        class FakeModel:
            def __init__(self):
                import torch.nn as nn

                self.tokenizer = FakeTokenizer()
                self.lm_head = nn.Linear(2, 2, bias=False)
                self.lm_head.weight.data = torch.eye(2)
                self.norm = nn.Identity()

        class FakeScorer:
            def score_text(self, text):
                return {"en": 0.8, "fr": 0.2} if text == "a b" else {"en": 0.3, "fr": 0.7}

        lang_probs, decoded_texts = teacher_forced_argmax_decode_then_classify(
            prompts=["hello"],
            full_latents_list=[torch.tensor(
                [
                    [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            )],
            layer_indices=[0, 1],
            token_agg="frac_50",
            rollout_k=1,
            rollout_word_cnt=0,
            model=FakeModel(),
            decoding_lens=FakeDecodingLens(),
            scorer=FakeScorer(),
            word_ids_list=[[0, 1, 2, 3]],
            show_progress=False,
        )
        self.assertAlmostEqual(lang_probs["en"][0, 0].item(), 0.8, places=6)
        self.assertAlmostEqual(lang_probs["fr"][0, 0].item(), 0.2, places=6)
        self.assertAlmostEqual(lang_probs["en"][0, 1].item(), 0.3, places=6)
        self.assertAlmostEqual(lang_probs["fr"][0, 1].item(), 0.7, places=6)
        self.assertEqual(decoded_texts[0], ["a b", "b a"])

    def test_teacher_forced_argmax_decode_then_classify_preserves_candidate_language_order(self):
        import torch

        class FakeTokenizer:
            def decode(self, ids):
                return "a"

        class FakeModel:
            def __init__(self):
                import torch.nn as nn

                self.tokenizer = FakeTokenizer()
                self.lm_head = nn.Linear(2, 2, bias=False)
                self.lm_head.weight.data = torch.eye(2)
                self.norm = nn.Identity()

        class FakeScorer:
            candidate_languages = ["fr", "en", "ar"]

            def score_text(self, text):
                return {"en": 0.8, "fr": 0.2}

        lang_probs, _ = teacher_forced_argmax_decode_then_classify(
            prompts=["hello"],
            full_latents_list=[torch.tensor(
                [
                    [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            )],
            layer_indices=[0],
            token_agg="frac_50",
            rollout_k=0,
            rollout_word_cnt=0,
            model=FakeModel(),
            decoding_lens=FakeDecodingLens(),
            scorer=FakeScorer(),
            word_ids_list=[[None, 0, 1]],
            show_progress=False,
        )
        self.assertEqual(list(lang_probs.keys()), ["ar", "en", "fr"])
        self.assertAlmostEqual(lang_probs["ar"][0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(lang_probs["en"][0, 0].item(), 0.8, places=6)
        self.assertAlmostEqual(lang_probs["fr"][0, 0].item(), 0.2, places=6)

    def test_teacher_forced_argmax_decode_then_classify_rejects_empty_score_dicts(self):
        import torch

        class FakeTokenizer:
            def decode(self, ids):
                return ""

        class FakeModel:
            def __init__(self):
                import torch.nn as nn

                self.tokenizer = FakeTokenizer()
                self.lm_head = nn.Linear(2, 2, bias=False)
                self.lm_head.weight.data = torch.eye(2)
                self.norm = nn.Identity()

        class FakeScorer:
            candidate_languages = ["fr", "en"]

            def score_text(self, text):
                return {}

        with self.assertRaisesRegex(ValueError, "empty scores for decoded text"):
            teacher_forced_argmax_decode_then_classify(
                prompts=["hello"],
                full_latents_list=[torch.tensor(
                    [
                        [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
                    ],
                    dtype=torch.float32,
                )],
                layer_indices=[0],
                token_agg="frac_50",
                rollout_k=0,
                rollout_word_cnt=0,
                model=FakeModel(),
                decoding_lens=FakeDecodingLens(),
                scorer=FakeScorer(),
                word_ids_list=[[None, 0, 1]],
                show_progress=False,
            )

    def test_compute_rollout_word_token_budget(self):
        self.assertEqual(
            compute_rollout_word_token_budget(
                7,
                "frac_50",
                rollout_word_cnt=2,
                word_ids=[None, 0, 0, 1, 2, 2, 3],
            ),
            3,
        )
        self.assertEqual(
            compute_rollout_word_token_budget(
                4,
                "frac_75",
                rollout_word_cnt=1,
                word_ids=[None, 0, 1, 1],
            ),
            2,
        )
        with self.assertRaises(ValueError):
            compute_rollout_word_token_budget(
                4,
                "frac_50",
                rollout_word_cnt=1,
                word_ids=None,
            )

    def test_sample_top_p_token_id_restricts_nucleus(self):
        import torch

        probs = torch.tensor([0.60, 0.25, 0.15], dtype=torch.float32)
        self.assertEqual(sample_top_p_token_id(probs, top_p=0.70), 0)

    def test_sample_top_p_token_ids_restrict_nucleus(self):
        import torch

        probs = torch.tensor(
            [
                [0.60, 0.25, 0.15],
                [0.90, 0.05, 0.05],
            ],
            dtype=torch.float32,
        )
        sampled = sample_top_p_token_ids(probs, top_p=0.70)
        self.assertEqual(tuple(sampled.shape), (2,))
        self.assertEqual(int(sampled[0].item()), 0)
        self.assertEqual(int(sampled[1].item()), 0)

    def test_hf_cached_final_layer_raw_lens_uses_lm_head_without_second_norm(self):
        import torch
        import torch.nn as nn

        class FakeCausalLM:
            def __init__(self):
                self.model = SimpleNamespace(layers=[object(), object()])

        norm = nn.LayerNorm(2)
        norm.weight.data = torch.tensor([2.0, 0.5])
        norm.bias.data.zero_()
        info = UnembedInfo(
            weight=torch.eye(2),
            bias=None,
            normalized_weight=torch.eye(2),
            avg_uu=torch.tensor(1.0),
            norm_module=norm,
        )
        lens = RawLogitLens(unembed_info=info, apply_final_norm=True)
        hf_final_hidden = torch.tensor([[[3.0, 1.0]]], dtype=torch.float32)

        logits = _project_hf_cached_layer_logits(
            step_latents=hf_final_hidden,
            layer_idx=1,
            causal_lm=FakeCausalLM(),
            decoding_lens=lens,
        )

        expected_raw_lm_head = apply_unembed(hf_final_hidden, info, apply_final_norm=False)
        double_normed = apply_unembed(hf_final_hidden, info, apply_final_norm=True)
        self.assertTrue(torch.allclose(logits, expected_raw_lm_head))
        self.assertFalse(torch.allclose(logits, double_normed))

    def test_hf_cached_nonfinal_layer_raw_lens_keeps_normal_projection(self):
        import torch
        import torch.nn as nn

        class FakeCausalLM:
            def __init__(self):
                self.model = SimpleNamespace(layers=[object(), object()])

        norm = nn.LayerNorm(2)
        norm.weight.data = torch.tensor([2.0, 0.5])
        norm.bias.data.zero_()
        info = UnembedInfo(
            weight=torch.eye(2),
            bias=None,
            normalized_weight=torch.eye(2),
            avg_uu=torch.tensor(1.0),
            norm_module=norm,
        )
        lens = RawLogitLens(unembed_info=info, apply_final_norm=True)
        nonfinal_hidden = torch.tensor([[[3.0, 1.0]]], dtype=torch.float32)

        logits = _project_hf_cached_layer_logits(
            step_latents=nonfinal_hidden,
            layer_idx=0,
            causal_lm=FakeCausalLM(),
            decoding_lens=lens,
        )

        expected_normal_lens = apply_unembed(nonfinal_hidden, info, apply_final_norm=True)
        raw_lm_head_only = apply_unembed(nonfinal_hidden, info, apply_final_norm=False)
        self.assertTrue(torch.allclose(logits, expected_normal_lens))
        self.assertFalse(torch.allclose(logits, raw_lm_head_only))

    def test_hf_cached_final_layer_tuned_lens_uses_raw_lm_head_without_second_norm(self):
        import torch
        import torch.nn as nn

        class FakeCausalLM:
            def __init__(self):
                self.model = SimpleNamespace(layers=[object(), object()])

        norm = nn.LayerNorm(2)
        norm.weight.data = torch.tensor([2.0, 0.5])
        norm.bias.data.zero_()
        info = UnembedInfo(
            weight=torch.eye(2),
            bias=None,
            normalized_weight=torch.eye(2),
            avg_uu=torch.tensor(1.0),
            norm_module=norm,
        )
        lens = TunedLogitLens(
            unembed_info=info,
            translators={},
            apply_final_norm=True,
            final_layer_idx=1,
        )
        hf_final_hidden = torch.tensor([[[3.0, 1.0]]], dtype=torch.float32)

        logits = _project_hf_cached_layer_logits(
            step_latents=hf_final_hidden,
            layer_idx=1,
            causal_lm=FakeCausalLM(),
            decoding_lens=lens,
        )

        expected_raw_lm_head = apply_unembed(hf_final_hidden, info, apply_final_norm=False)
        double_normed = apply_unembed(hf_final_hidden, info, apply_final_norm=True)
        self.assertTrue(torch.allclose(logits, expected_raw_lm_head))
        self.assertFalse(torch.allclose(logits, double_normed))

    def test_sampled_rollout_decode_then_classify(self):
        import torch

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def __call__(self, prompt, add_special_tokens=True):
                return {"input_ids": [9, 8, 7]}

            def decode(self, ids):
                mapping = {
                    (9, 8): "prefix",
                    (9, 8, 0): "prefix a",
                    (9, 8, 1): "prefix b",
                    (9, 8, 0, 1): "prefix a b",
                    (9, 8, 1, 0): "prefix b a",
                    (0,): "a",
                    (1,): "b",
                    (0, 1): "a b",
                    (1, 0): "b a",
                }
                return mapping[tuple(ids)]

        class FakeModel:
            def __init__(self):
                import torch.nn as nn

                self.tokenizer = FakeTokenizer()
                self.lm_head = nn.Linear(2, 2, bias=False)
                self.lm_head.weight.data = torch.eye(2)
                self.norm = nn.Identity()
                self._model = FakeHFModel()

        class FakeHFModel:
            def __init__(self):
                import torch.nn as nn

                self._param = nn.Parameter(torch.zeros(1))

            def parameters(self):
                yield self._param

            def __call__(
                self,
                input_ids,
                attention_mask=None,
                past_key_values=None,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            ):
                batch_size = input_ids.shape[0]
                hidden = torch.tensor([[[2.0, 1.0]]], dtype=torch.float32).repeat(batch_size, 1, 1)
                cache = ((torch.zeros((batch_size, 1, 1, 1)), torch.zeros((batch_size, 1, 1, 1))),)
                return SimpleNamespace(hidden_states=[torch.zeros_like(hidden), hidden], past_key_values=cache)

        class FakeScorer:
            candidate_languages = ["fr", "en"]

            def score_text(self, text):
                if text == "a b":
                    return {"en": 0.8, "fr": 0.2}
                if text == "b a":
                    return {"en": 0.1, "fr": 0.9}
                raise AssertionError(text)

        with mock.patch(
            "decode_then_classify.sample_top_p_token_ids",
            side_effect=[torch.tensor([0, 1]), torch.tensor([1, 0])],
        ):
            lang_probs, sample_records = sampled_rollout_decode_then_classify(
                prompts=["hello"],
                full_latents_list=[torch.zeros((1, 4, 2), dtype=torch.float32)],
                layer_indices=[0],
                token_agg="frac_50",
                rollout_k=0,
                rollout_word_cnt=1,
                rollout_top_p=0.9,
                rollout_num_samples=2,
                model=FakeModel(),
                decoding_lens=FakeDecodingLens(),
                scorer=FakeScorer(),
                word_ids_list=[[None, 0, 1, 1]],
                show_progress=False,
            )
        self.assertEqual(list(lang_probs.keys()), ["en", "fr"])
        self.assertAlmostEqual(lang_probs["en"][0, 0].item(), 0.45, places=6)
        self.assertAlmostEqual(lang_probs["fr"][0, 0].item(), 0.55, places=6)
        self.assertEqual(len(sample_records), 2)
        self.assertEqual([row["decoded_text"] for row in sample_records], ["a b", "b a"])

    def test_sampled_rollout_decode_then_classify_batches_prompts_for_rollout_k(self):
        import torch

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def __call__(self, prompt, add_special_tokens=True):
                return {"input_ids": [9, 8]}

            def decode(self, ids):
                mapping = {
                    (0,): "a",
                    (1,): "b",
                    (0, 0): "a a",
                    (0, 1): "a b",
                    (1, 0): "b a",
                    (1, 1): "b b",
                }
                return mapping[tuple(ids)]

        class FakeHFModel:
            def __init__(self):
                import torch.nn as nn

                self._param = nn.Parameter(torch.zeros(1))
                self.batch_sizes = []

            def parameters(self):
                yield self._param

            def __call__(
                self,
                input_ids,
                attention_mask=None,
                past_key_values=None,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            ):
                batch_size = input_ids.shape[0]
                self.batch_sizes.append(int(batch_size))
                hidden = torch.tensor([[[2.0, 1.0]]], dtype=torch.float32).repeat(batch_size, 1, 1)
                cache = ((torch.zeros((batch_size, 1, 1, 1)), torch.zeros((batch_size, 1, 1, 1))),)
                return SimpleNamespace(hidden_states=[torch.zeros_like(hidden), hidden], past_key_values=cache)

        class FakeModel:
            def __init__(self):
                import torch.nn as nn

                self.tokenizer = FakeTokenizer()
                self.lm_head = nn.Linear(2, 2, bias=False)
                self.lm_head.weight.data = torch.eye(2)
                self.norm = nn.Identity()
                self._model = FakeHFModel()

        class FakeScorer:
            candidate_languages = ["en"]

            def score_text(self, text):
                return {"en": 1.0}

        fake_model = FakeModel()
        with mock.patch("decode_then_classify.sample_top_p_token_ids", side_effect=[torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1])]):
            sampled_rollout_decode_then_classify(
                prompts=["p1", "p2"],
                full_latents_list=[
                    torch.zeros((1, 2, 2), dtype=torch.float32),
                    torch.zeros((1, 2, 2), dtype=torch.float32),
                ],
                layer_indices=[0],
                token_agg="frac_50",
                rollout_k=1,
                rollout_word_cnt=0,
                rollout_top_p=0.9,
                rollout_num_samples=2,
                model=fake_model,
                decoding_lens=FakeDecodingLens(),
                scorer=FakeScorer(),
                word_ids_list=[[None, 0], [None, 0]],
                show_progress=False,
            )
        self.assertEqual(fake_model._model.batch_sizes, [2, 4])

    def test_score_singlepos_latents_matches_across_chunk_sizes(self):
        import torch

        latents = torch.tensor(
            [
                [[2.0, 1.0], [1.0, 2.0]],
                [[3.0, 0.0], [0.0, 3.0]],
                [[0.5, 2.5], [2.5, 0.5]],
            ],
            dtype=torch.float32,
        )

        class FakeScorer:
            def classify_from_probs(self, probs, mode, topk):
                if int(probs.argmax().item()) == 0:
                    return "en", {"en": 0.8, "fr": 0.2}
                return "fr", {"en": 0.1, "fr": 0.9}

        with mock.patch(
            "decode_then_classify.tqdm",
            side_effect=lambda *args, **kwargs: SimpleNamespace(
                update=lambda n: None,
                close=lambda: None,
            ),
        ):
            small = score_singlepos_latents(
                latents=latents,
                layer_indices=[0, 1],
                decoding_lens=FakeDecodingLens(),
                scorer=FakeScorer(),
                decode_mode="topk_weighted",
                topk=20,
                desc="test",
                chunk_size=1,
            )
            large = score_singlepos_latents(
                latents=latents,
                layer_indices=[0, 1],
                decoding_lens=FakeDecodingLens(),
                scorer=FakeScorer(),
                decode_mode="topk_weighted",
                topk=20,
                desc="test",
                chunk_size=10,
            )

        self.assertEqual(set(small.keys()), set(large.keys()))
        for lang in small:
            self.assertTrue(torch.allclose(small[lang], large[lang]))

    def test_score_singlepos_latents_preserves_candidate_language_order(self):
        import torch

        latents = torch.tensor(
            [
                [[2.0, 1.0]],
            ],
            dtype=torch.float32,
        )

        class FakeScorer:
            candidate_languages = ["fr", "en", "ar"]

            def classify_from_probs(self, probs, mode, topk):
                return "en", {"en": 0.8, "fr": 0.2}

        with mock.patch(
            "decode_then_classify.tqdm",
            side_effect=lambda *args, **kwargs: SimpleNamespace(
                update=lambda n: None,
                close=lambda: None,
            ),
        ):
            lang_probs = score_singlepos_latents(
                latents=latents,
                layer_indices=[0],
                decoding_lens=FakeDecodingLens(),
                scorer=FakeScorer(),
                decode_mode="topk_weighted",
                topk=20,
                desc="test",
                chunk_size=1,
            )

        self.assertEqual(list(lang_probs.keys()), ["ar", "en", "fr"])
        self.assertAlmostEqual(lang_probs["ar"][0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(lang_probs["en"][0, 0].item(), 0.8, places=6)
        self.assertAlmostEqual(lang_probs["fr"][0, 0].item(), 0.2, places=6)

    def test_build_method_langdist_artifact(self):
        import torch

        artifact = build_method_langdist_artifact(
            langdist={
                "en": torch.tensor([[0.9, 0.2]], dtype=torch.float32),
                "fr": torch.tensor([[0.1, 0.8]], dtype=torch.float32),
            },
            decoding_mapping=None,
        )
        self.assertEqual(artifact["langs"], ["en", "fr"])
        self.assertEqual(tuple(artifact["probs"].shape), (1, 2, 2))
        self.assertEqual(artifact["probs"].dtype, torch.float16)

    def test_build_prompt_metric_rows(self):
        import torch

        lang_probs = {
            "en": torch.tensor([[0.8, 0.2], [0.3, 0.1]]),
            "fr": torch.tensor([[0.2, 0.8], [0.7, 0.9]]),
        }
        rows = build_prompt_metric_rows(
            checkpoint_id="main",
            method="repr_gmm",
            layer_indices=[0, 1],
            langdist=lang_probs,
            prompt_ids=["en_1", "fr_1"],
            prompt_langs=["en", "fr"],
            task_langs_by_prompt=[("en",), ("en",)],
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["dominant_lang"], "en")
        self.assertFalse(rows[0]["pivot"])
        self.assertEqual(rows[-1]["dominant_lang"], "fr")
        self.assertTrue(rows[-1]["pivot"])

    def test_validate_repr_gmm_dir_missing_prompt_langs_raises(self):
        import tempfile
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "layer_0.pt"
            payload = {
                "mode": "paper",
                "pca": {
                    "mean": torch.zeros(4),
                    "components": torch.eye(1, 4),
                    "explained_var_ratio": torch.tensor([1.0]),
                    "k": 1,
                    "var_threshold": 0.98,
                },
                "gaussians": {
                    "covariance_type": "diag",
                    "priors": {"en": 0.5, "fr": 0.5},
                    "languages": {
                        "en": {"mean": torch.zeros(1), "cov": torch.ones(1), "n_samples": torch.tensor(10)},
                        "fr": {"mean": torch.ones(1), "cov": torch.ones(1), "n_samples": torch.tensor(10)},
                    },
                },
                "meta": {"layer": 0, "languages": ["en", "fr"]},
            }
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                validate_repr_gmm_dir(tmpdir, expected_prompt_langs=["en", "ar"])

    def test_validate_repr_gmm_dir_checks_manifest_for_named_setup(self):
        import tempfile
        import torch

        registry = load_gmm_setup_registry("configs/gmm_setups.json")
        setup = registry["pud9"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "layer_0.pt"
            payload = {
                "mode": "paper",
                "pca": {
                    "mean": torch.zeros(4),
                    "components": torch.eye(1, 4),
                    "explained_var_ratio": torch.tensor([1.0]),
                    "k": 1,
                    "var_threshold": 0.98,
                },
                "gaussians": {
                    "covariance_type": "diag",
                    "priors": {lang: 1.0 / len(setup["languages"]) for lang in setup["languages"]},
                    "languages": {
                        lang: {"mean": torch.zeros(1), "cov": torch.ones(1), "n_samples": torch.tensor(10)}
                        for lang in setup["languages"]
                    },
                },
                "meta": {"layer": 0, "languages": list(setup["languages"])},
            }
            torch.save(payload, path)
            (Path(tmpdir) / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "setup_name": "pud9",
                        "setup": setup,
                    }
                ),
                encoding="utf-8",
            )
            result = validate_repr_gmm_dir(
                tmpdir,
                expected_prompt_langs=["en", "fr"],
                expected_setup_name="pud9",
                expected_setup=setup,
            )
            self.assertEqual(result["setup_name"], "pud9")


if __name__ == "__main__":
    unittest.main()
