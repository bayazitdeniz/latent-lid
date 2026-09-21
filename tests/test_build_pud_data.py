from types import SimpleNamespace
import unittest

from scripts.data.build_pud_data import _resolve_output_paths, _resolve_split_sizes, _split_texts
from scripts.data.build_ud_data import _filter_reason, _filter_records, _prompt_payload, _split_records


class BuildPudDataTests(unittest.TestCase):
    def test_resolve_split_sizes_uses_explicit_sizes(self):
        args = SimpleNamespace(
            test_size=40,
            train_size=60,
        )
        train_size, test_size = _resolve_split_sizes(args)
        self.assertEqual(train_size, 60)
        self.assertEqual(test_size, 40)

    def test_split_texts_is_deterministic(self):
        texts = [f"row-{idx}" for idx in range(10)]
        train_a, test_a = _split_texts(
            texts,
            train_size=6,
            test_size=4,
            split_seed=42,
            lang="en",
        )
        train_b, test_b = _split_texts(
            texts,
            train_size=6,
            test_size=4,
            split_seed=42,
            lang="en",
        )
        self.assertEqual(train_a, train_b)
        self.assertEqual(test_a, test_b)

    def test_split_texts_reuses_sentence_indices_across_languages(self):
        texts = [f"row-{idx}" for idx in range(10)]
        train_en, test_en = _split_texts(
            texts,
            train_size=6,
            test_size=4,
            split_seed=42,
            lang="en",
        )
        train_fr, test_fr = _split_texts(
            texts,
            train_size=6,
            test_size=4,
            split_seed=42,
            lang="fr",
        )
        self.assertEqual(train_en, train_fr)
        self.assertEqual(test_en, test_fr)

    def test_resolve_output_paths_defaults_to_pud_holdout(self):
        args = SimpleNamespace(
            out_dir="data",
        )
        paths = _resolve_output_paths(args)
        self.assertEqual(str(paths["split_root"]), "data/pud_holdout")
        self.assertEqual(str(paths["out_prompts_test"]), "data/pud_holdout/pud_prompts_test.jsonl")
        self.assertEqual(str(paths["out_gmm_root_train"]), "data/pud_holdout/pud_langs_train")
        self.assertEqual(str(paths["out_prompts_train"]), "data/pud_holdout/pud_prompts_train.jsonl")
        self.assertEqual(str(paths["out_split_meta"]), "data/pud_holdout/pud_split_meta.json")

    def test_ud_holdout_split_records_is_deterministic_for_marathi_policy(self):
        records = [{"text": f"row-{idx}", "surface_tokens": [str(idx)]} for idx in range(10)]
        train_a, test_a = _split_records(records, train_size=6, test_size=4, split_seed=42, lang="mr")
        train_b, test_b = _split_records(records, train_size=6, test_size=4, split_seed=42, lang="mr")
        self.assertEqual(train_a, train_b)
        self.assertEqual(test_a, test_b)

    def test_ud_holdout_prompt_payload_is_source_aware(self):
        payload = _prompt_payload(
            lang="uk",
            cfg={"treebank": "Ukrainian-IU", "file_prefix": "uk_iu"},
            split="train",
            offset=0,
            record={"text": "текст", "surface_tokens": ["текст"], "ud_split": "train"},
        )
        self.assertEqual(payload["id"], "uk_iu-train-0")
        self.assertEqual(payload["source_lang"], "uk")
        self.assertEqual(payload["target_lang"], "uk")
        self.assertEqual(payload["treebank"], "UD_Ukrainian-IU")
        self.assertFalse(payload["is_parallel"])

    def test_ud_holdout_filter_reason_catches_numeric_list_item(self):
        self.assertEqual(_filter_reason({"text": "2.", "surface_tokens": ["2", "."]}), "list_item")

    def test_ud_holdout_filter_reason_catches_fragment_patterns(self):
        self.assertEqual(_filter_reason({"text": "Усе знав.", "surface_tokens": ["Усе", "знав", "."]}), "very_short")
        self.assertEqual(_filter_reason({"text": "А він:", "surface_tokens": ["А", "він", ":"]}), "trailing_colon")
        self.assertEqual(_filter_reason({"text": "Е, е, е.", "surface_tokens": ["Е", ",", "е", ",", "е", "."]}), "repeated_interjection")

    def test_ud_holdout_filter_reason_keeps_longer_punctuation_sentences(self):
        self.assertIsNone(
            _filter_reason(
                {
                    "text": "Він почав писати квапливими неохайними карлючками:",
                    "surface_tokens": ["Він", "почав", "писати", "квапливими", "неохайними", "карлючками", ":"],
                }
            )
        )

    def test_ud_holdout_filter_records_returns_rejected_rows_with_reasons(self):
        kept, counts, filtered_rows = _filter_records(
            [
                {"text": "2.", "surface_tokens": ["2", "."], "ud_split": "test"},
                {"text": "Повне речення тут.", "surface_tokens": ["Повне", "речення", "тут", "."], "ud_split": "train"},
            ]
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(counts, {"list_item": 1})
        self.assertEqual(
            filtered_rows,
            [
                {
                    "text": "2.",
                    "surface_tokens": ["2", "."],
                    "ud_split": "test",
                    "filter_reason": "list_item",
                }
            ],
        )
        self.assertIsNone(
            _filter_reason(
                {
                    "text": "Вас так багато людей сьогодні прийшло...",
                    "surface_tokens": ["Вас", "так", "багато", "людей", "сьогодні", "прийшло", "..."],
                }
            )
        )
        self.assertIsNone(
            _filter_reason(
                {
                    "text": "(Ми вже давно обговорили це питання.)",
                    "surface_tokens": ["(", "Ми", "вже", "давно", "обговорили", "це", "питання", ".", ")"],
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
