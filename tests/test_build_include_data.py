import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data.build_include_data import (
    balance_records,
    get_options,
    make_prompt_id,
    make_row_hash,
    metadata_payload,
    prepare_records,
    prompt_payload,
    render_minimal_mcq,
    render_question_only,
    select_records_by_cell,
    simple_surface_tokens,
    write_jsonl,
)


def _row(lang: str = "en", domain: str = "STEM", idx: int = 0, **overrides):
    row = {
        "dataset_name": "CohereLabs/include-base-44",
        "dataset_variant": "include-base-44",
        "hf_config": "English",
        "source_split": "test",
        "source_row_idx": idx,
        "lang": lang,
        "language": "English",
        "country": "United States",
        "domain": domain,
        "subject": "Physics",
        "level": "Academic",
        "regional_feature": "region neutral",
        "question": f"What is item {idx}?",
        "option_a": "Alpha",
        "option_b": "Beta",
        "option_c": "Gamma",
        "option_d": "Delta",
        "answer": 2,
    }
    row.update(overrides)
    return row


class BuildIncludeEvalSubsetTests(unittest.TestCase):
    def test_get_options_accepts_option_columns(self):
        self.assertEqual(get_options(_row()), ["Alpha", "Beta", "Gamma", "Delta"])

    def test_get_options_accepts_choices_list(self):
        row = _row(choices=["one", "two", "three", "four"])
        for key in ["option_a", "option_b", "option_c", "option_d"]:
            row.pop(key)
        self.assertEqual(get_options(row), ["one", "two", "three", "four"])

    def test_get_options_accepts_choices_json_string(self):
        row = _row(choices=json.dumps(["one", "two", "three", "four"]))
        for key in ["option_a", "option_b", "option_c", "option_d"]:
            row.pop(key)
        self.assertEqual(get_options(row), ["one", "two", "three", "four"])

    def test_get_options_rejects_non_four_option_rows(self):
        row = _row(choices=["one", "two"])
        for key in ["option_a", "option_b", "option_c", "option_d"]:
            row.pop(key)
        with self.assertRaisesRegex(ValueError, "exactly four"):
            get_options(row)

    def test_prompt_rendering(self):
        row = _row()
        self.assertEqual(render_question_only(row), "What is item 0?")
        self.assertEqual(
            render_minimal_mcq(row),
            "What is item 0?\n\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
        )

    def test_stable_hash_and_id_generation(self):
        row = _row(lang="ar", domain="Social Science")
        row["row_hash"] = make_row_hash(row)
        prompt_id = make_prompt_id(row)
        self.assertEqual(prompt_id, make_prompt_id(row))
        self.assertTrue(prompt_id.startswith("include_base_44-ar-social_science-"))

    def test_prompt_payload_is_run_eval_compatible_and_style_specific(self):
        row = _row(lang="ar")
        row["row_hash"] = make_row_hash(row)
        row["id"] = make_prompt_id(row)

        question = prompt_payload(row, "question_only")
        mcq = prompt_payload(row, "minimal_mcq")

        self.assertEqual(question["id"], mcq["id"])
        self.assertEqual(question["prompt_style"], "question_only")
        self.assertEqual(mcq["prompt_style"], "minimal_mcq")
        for payload in [question, mcq]:
            self.assertEqual(payload["lang"], "ar")
            self.assertEqual(payload["source_lang"], "ar")
            self.assertEqual(payload["target_lang"], "ar")
            self.assertIsInstance(payload["surface_tokens"], list)
            self.assertTrue(payload["surface_tokens"])

    def test_simple_surface_tokens_uses_character_tokens_for_unsegmented_scripts(self):
        self.assertEqual(simple_surface_tokens("今天天气 好", lang="zh"), ["今", "天", "天", "气", "好"])
        self.assertEqual(simple_surface_tokens("今日は 天気", lang="ja"), ["今", "日", "は", "天", "気"])
        self.assertEqual(simple_surface_tokens("ภาษา ไทย", lang="th"), ["ภ", "า", "ษ", "า", "ไ", "ท", "ย"])

    def test_simple_surface_tokens_keeps_korean_whitespace_tokens(self):
        self.assertEqual(simple_surface_tokens("오늘 날씨가 좋다", lang="ko"), ["오늘", "날씨가", "좋다"])

    def test_metadata_payload_keeps_include_specific_fields(self):
        row = _row(lang="hi", domain="Arts & Humanities")
        row["row_hash"] = make_row_hash(row)
        row["id"] = make_prompt_id(row)
        meta = metadata_payload(row)

        self.assertEqual(meta["id"], row["id"])
        self.assertEqual(meta["lang"], "hi")
        self.assertEqual(meta["domain"], "Arts & Humanities")
        self.assertEqual(meta["option_c"], "Gamma")
        self.assertEqual(meta["answer"], 2)

    def test_balance_records_uses_min_cell_count_and_is_deterministic(self):
        rows = []
        for lang in ["ar", "es"]:
            for domain in ["STEM", "Social Science"]:
                n = 3 if (lang, domain) != ("es", "STEM") else 2
                for idx in range(n):
                    row = _row(lang=lang, domain=domain, idx=len(rows))
                    row["row_hash"] = make_row_hash(row)
                    row["id"] = make_prompt_id(row)
                    rows.append(row)

        selected_a, stats_a = balance_records(
            rows,
            langs=["ar", "es"],
            domains=["STEM", "Social Science"],
            seed=7,
            max_per_cell=None,
        )
        selected_b, stats_b = balance_records(
            rows,
            langs=["ar", "es"],
            domains=["STEM", "Social Science"],
            seed=7,
            max_per_cell=None,
        )

        self.assertEqual([row["id"] for row in selected_a], [row["id"] for row in selected_b])
        self.assertEqual(stats_a, stats_b)
        self.assertEqual(stats_a["target_per_cell"], 2)
        self.assertEqual(len(selected_a), 8)
        self.assertTrue(all(count == 2 for count in stats_a["selected_by_cell"].values()))

    def test_balance_records_respects_max_per_cell(self):
        rows = []
        for lang in ["ar", "es"]:
            for domain in ["STEM", "Social Science"]:
                for idx in range(3):
                    row = _row(lang=lang, domain=domain, idx=len(rows))
                    row["row_hash"] = make_row_hash(row)
                    row["id"] = make_prompt_id(row)
                    rows.append(row)

        selected, stats = balance_records(
            rows,
            langs=["ar", "es"],
            domains=["STEM", "Social Science"],
            seed=7,
            max_per_cell=1,
        )

        self.assertEqual(stats["target_per_cell"], 1)
        self.assertEqual(len(selected), 4)

    def test_select_records_by_cell_all_keeps_every_cell_row(self):
        rows = []
        for lang in ["ar", "es"]:
            for domain in ["STEM", "Social Science"]:
                n = 3 if (lang, domain) != ("es", "STEM") else 2
                for idx in range(n):
                    row = _row(lang=lang, domain=domain, idx=len(rows))
                    row["row_hash"] = make_row_hash(row)
                    row["id"] = make_prompt_id(row)
                    rows.append(row)

        selected, stats = select_records_by_cell(
            rows,
            langs=["ar", "es"],
            domains=["STEM", "Social Science"],
            seed=7,
            sampling="all",
            max_per_cell=None,
        )

        self.assertIsNone(stats["target_per_cell"])
        self.assertEqual(len(selected), len(rows))
        self.assertEqual(stats["selected_by_cell"]["es::STEM"], 2)
        self.assertEqual(stats["selected_by_cell"]["ar::STEM"], 3)

    def test_select_records_by_cell_cap_keeps_smaller_cells_uncut(self):
        rows = []
        for lang in ["ar", "es"]:
            for domain in ["STEM", "Social Science"]:
                n = 3 if (lang, domain) != ("es", "STEM") else 2
                for idx in range(n):
                    row = _row(lang=lang, domain=domain, idx=len(rows))
                    row["row_hash"] = make_row_hash(row)
                    row["id"] = make_prompt_id(row)
                    rows.append(row)

        selected, stats = select_records_by_cell(
            rows,
            langs=["ar", "es"],
            domains=["STEM", "Social Science"],
            seed=7,
            sampling="cap",
            max_per_cell=2,
        )

        self.assertIsNone(stats["target_per_cell"])
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(count == 2 for count in stats["selected_by_cell"].values()))

    def test_select_records_by_cell_cap_requires_max_per_cell(self):
        row = _row(lang="ar", domain="STEM")
        row["row_hash"] = make_row_hash(row)
        row["id"] = make_prompt_id(row)

        with self.assertRaisesRegex(ValueError, "max-per-cell"):
            select_records_by_cell(
                [row],
                langs=["ar"],
                domains=["STEM"],
                seed=7,
                sampling="cap",
                max_per_cell=None,
            )

    def test_balance_records_rejects_missing_cell(self):
        row = _row(lang="ar", domain="STEM")
        row["row_hash"] = make_row_hash(row)
        row["id"] = make_prompt_id(row)

        with self.assertRaisesRegex(ValueError, "Missing INCLUDE rows"):
            balance_records(
                [row],
                langs=["ar", "es"],
                domains=["STEM"],
                seed=7,
                max_per_cell=None,
            )

    def test_prepare_records_adds_unique_ids(self):
        frame = pd.DataFrame([_row(idx=0), _row(idx=1)])
        records = prepare_records(frame)
        self.assertEqual(len(records), 2)
        self.assertEqual(len({record["id"] for record in records}), 2)
        self.assertTrue(all(record["row_hash"] for record in records))

    def test_write_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            count = write_jsonl(path, [{"a": 1}, {"b": "x"}])
            self.assertEqual(count, 2)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, [{"a": 1}, {"b": "x"}])


if __name__ == "__main__":
    unittest.main()
