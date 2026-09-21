import unittest

from scripts.data.pud_data_source import (
    PUD_LANGS,
    TREEBANK_MAP,
    UD_EXTRA_LANGS,
    UD_EXTRA_TREEBANKS,
    default_ud_cache_path,
    read_conllu_sentence_records,
)


class PudDataSourceTests(unittest.TestCase):
    def test_default_ud_cache_path(self):
        self.assertEqual(str(default_ud_cache_path("2.17")), "data/ud_cache/ud-treebanks-v2.17.tgz")

    def test_treebank_map_contains_all_pud_languages(self):
        expected = [
            "ar",
            "cs",
            "de",
            "en",
            "es",
            "fi",
            "fr",
            "gl",
            "hi",
            "id",
            "is",
            "it",
            "ja",
            "ko",
            "pl",
            "pt",
            "ru",
            "sv",
            "th",
            "tr",
            "zh",
        ]
        self.assertEqual(sorted(PUD_LANGS), expected)
        for lang in expected:
            self.assertIn(lang, TREEBANK_MAP)

    def test_extra_ud_treebank_map_contains_nonparallel_languages(self):
        expected = ["bg", "fa", "mr", "sr", "uk", "ur"]
        self.assertEqual(sorted(UD_EXTRA_LANGS), expected)
        self.assertEqual(UD_EXTRA_TREEBANKS["uk"]["treebank"], "Ukrainian-IU")
        self.assertEqual(UD_EXTRA_TREEBANKS["uk"]["file_prefix"], "uk_iu")

    def test_read_conllu_sentence_records_uses_surface_multiword_tokens(self):
        conllu = (
            '# text = aux amis.\n'
            '1-2\taux\t_\t_\t_\t_\t_\t_\t_\t_\n'
            '1\tà\t_\t_\t_\t_\t_\t_\t_\t_\n'
            '2\tles\t_\t_\t_\t_\t_\t_\t_\t_\n'
            '3\tamis\t_\t_\t_\t_\t_\t_\t_\t_\n'
            '4\t.\t_\t_\t_\t_\t_\t_\t_\t_\n'
            '\n'
        )
        records = read_conllu_sentence_records(conllu.splitlines(keepends=True))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], "aux amis.")
        self.assertEqual(records[0]["surface_tokens"], ["aux", "amis", "."])


if __name__ == "__main__":
    unittest.main()
