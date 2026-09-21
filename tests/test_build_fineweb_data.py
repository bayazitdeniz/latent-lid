import unittest

from scripts.data.build_fineweb_data import _rebalance_split_rows


class BuildFinewebLensCorpusTests(unittest.TestCase):
    def test_rebalance_split_rows_trims_to_smallest_language(self):
        per_language_rows = {
            "en": [
                {"lang": "en", "input_ids": [1, 2]},
                {"lang": "en", "input_ids": [3, 4]},
                {"lang": "en", "input_ids": [5, 6]},
            ],
            "fr": [
                {"lang": "fr", "input_ids": [7, 8]},
                {"lang": "fr", "input_ids": [9, 10]},
            ],
        }
        per_language_stats = [
            {"split": "train", "lang": "en", "chunks": 3, "tokens": 6, "target_chunks": 4, "target_tokens": 8},
            {"split": "train", "lang": "fr", "chunks": 2, "tokens": 4, "target_chunks": 4, "target_tokens": 8},
        ]

        rows, warning = _rebalance_split_rows(per_language_rows, per_language_stats, seq_len=2)

        self.assertEqual(len(rows), 4)
        self.assertIsNotNone(warning)
        self.assertEqual(warning["effective_chunks_per_lang"], 2)
        self.assertEqual(warning["limiting_languages"], ["fr"])
        self.assertEqual(per_language_stats[0]["trimmed_chunks"], 1)
        self.assertEqual(per_language_stats[1]["trimmed_chunks"], 0)


if __name__ == "__main__":
    unittest.main()
