import unittest
from pathlib import Path

from data.wendler_2024_data.target_string_artifacts import (
    build_target_string_records,
    concept_id_from_task_row,
    prompt_text_from_task_row,
    write_target_string_artifacts_for_dir,
)
from data.wendler_2024_data.build_translation_pairs_and_filter import cloze_query_prompt
from data.wendler_2024_data.wendler_extend_utils import prompt_line
from run_eval_utils import load_synthetic_prompt_data


class TargetStringArtifactTests(unittest.TestCase):
    def test_prompt_line_omitted_target_leaves_answer_slot_open(self):
        self.assertEqual(
            prompt_line("de", "Buch", "zh"),
            'Deutsch: "Buch" - 中文: "',
        )
        self.assertEqual(
            prompt_line("de", "Buch", "zh", "书"),
            'Deutsch: "Buch" - 中文: "书"',
        )

    def test_prompt_text_repairs_closed_empty_translation_slot(self):
        row = {
            "prompt": 'Deutsch: "Herz" - 中文: "心"\nDeutsch: "Buch" - 中文: ""',
        }
        self.assertEqual(
            prompt_text_from_task_row(row, "translation").splitlines()[-1],
            'Deutsch: "Buch" - 中文: "',
        )

    def test_prompt_text_for_cloze_uses_prefix_before_blank(self):
        row = {
            "blank_prompt_translation_masked": 'يتم إجراء "___" عند محاولة تحقيق شيء ما. إجابة: "محاولة".',
        }
        self.assertEqual(prompt_text_from_task_row(row, "cloze"), 'يتم إجراء "')

    def test_prompt_text_for_cloze_prefers_prebuilt_prompt(self):
        row = {
            "prompt": 'A "___" is used to play sports. Answer: "ball".\nA "___" is often given as a gift. Answer: "',
            "blank_prompt_translation_masked": 'A "___" is often given as a gift. Answer: "flower".',
        }
        self.assertEqual(
            prompt_text_from_task_row(row, "cloze"),
            'A "___" is used to play sports. Answer: "ball".\nA "___" is often given as a gift. Answer: "',
        )

    def test_cloze_query_prompt_handles_full_width_colon(self):
        self.assertEqual(cloze_query_prompt('___」が行われる。答えなさい：「試み」。', "ja"), '___」が行われる。答えなさい: "')

    def test_cloze_query_prompt_falls_back_to_target_prefix(self):
        self.assertEqual(cloze_query_prompt('___」。答え「ボール', "ja", "ボール"), '___」。答え「')

    def test_cloze_query_prompt_handles_japanese_answer_marker(self):
        self.assertEqual(cloze_query_prompt('___」。答え「9」である。', "ja", "ナイン"), '___」。答え「')

    def test_cloze_query_prompt_allows_initial_blank_query(self):
        self.assertEqual(cloze_query_prompt('___」である。', "ja", "八"), "")

    def test_build_target_string_records_groups_surface_forms(self):
        rows = [
            {
                "source_file": "f1.csv",
                "source_row": "0",
                "task": "copy",
                "input_lang": "af",
                "target_lang": "af",
                "prompt": 'Afrikaans: "boek" - Afrikaans: "',
                "out_token_str": "boek",
                "latent_token_str": "book",
                "in_token_str": "boek",
            },
            {
                "source_file": "f2.csv",
                "source_row": "0",
                "task": "copy",
                "input_lang": "nl",
                "target_lang": "nl",
                "prompt": 'Nederlands: "boek" - Nederlands: "',
                "out_token_str": "boek",
                "latent_token_str": "book",
                "in_token_str": "boek",
            },
            {
                "source_file": "f3.csv",
                "source_row": "0",
                "task": "copy",
                "input_lang": "de",
                "target_lang": "de",
                "prompt": 'Deutsch: "Buch" - Deutsch: "',
                "out_token_str": "Buch",
                "latent_token_str": "book",
                "in_token_str": "Buch",
            },
        ]
        records = build_target_string_records(rows, "copy")
        self.assertEqual(len(records), 3)
        record = records[0]
        self.assertEqual(record["concept_id"], "book")
        self.assertTrue(record["menu_has_ambiguity"])
        self.assertEqual(record["menu_group_count"], 2)
        self.assertEqual(
            record["menu_strings_grouped"],
            [
                {"text": "Buch", "langs": ["de"]},
                {"text": "boek", "langs": ["af", "nl"]},
            ],
        )
        self.assertEqual(record["menu_strings_unique"], [{"text": "Buch", "langs": ["de"]}])

    def test_concept_id_for_cloze_prefers_concept_id(self):
        row = {
            "concept_id": "book",
            "target_text": "Buch",
            "latent_token_str": "ignored",
            "in_token_str": "ignored",
        }
        self.assertEqual(concept_id_from_task_row(row, "cloze"), "book")

    def test_load_synthetic_prompt_data_falls_back_without_companion(self):
        csv_path = Path("data/wendler_2024_data/processed/translation.csv")
        companion_path = csv_path.with_name("translation_target_string.jsonl")
        backup_path = csv_path.with_name("translation_target_string.jsonl.bak_test")
        restored = False
        try:
            if companion_path.exists():
                companion_path.rename(backup_path)
                restored = True
            prompt_data = load_synthetic_prompt_data(csv_path, "translation", 2, None)
            self.assertEqual(len(prompt_data.prompts), 2)
            self.assertEqual(len(prompt_data.prompt_ids), 2)
            self.assertEqual(len(prompt_data.target_string_records), 2)
            first_record = prompt_data.target_string_records[prompt_data.prompt_ids[0]]
            self.assertIn("menu_strings_grouped", first_record)
            self.assertEqual(first_record["tgt_lang"], "zh")
            self.assertEqual(prompt_data.prompt_langs[0], "de")
            self.assertEqual(prompt_data.task_langs_by_prompt[0], ("de", "zh"))
            self.assertIsNone(prompt_data.surface_tokens_by_prompt[0])
        finally:
            if backup_path.exists():
                backup_path.rename(companion_path)
            elif restored:
                write_target_string_artifacts_for_dir(csv_path.parent)


if __name__ == "__main__":
    unittest.main()
