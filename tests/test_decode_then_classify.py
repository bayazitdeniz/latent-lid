import unittest

import torch

from decode_then_classify import DecodeThenClassifyScorer


class DecodeThenClassifyTests(unittest.TestCase):
    def test_classify_from_probs_argmax_topk(self):
        class FakeBackend:
            def predict(self, text, *, languages=None):
                return ("en" if "a" in text else "fr"), {}

            def predict_distribution(self, text, languages=None):
                dist = {"en": 0.8, "fr": 0.2} if "a" in text else {"en": 0.3, "fr": 0.7}
                if languages is None:
                    return dist
                return {lang: dist[lang] for lang in languages if lang in dist}

        class TinyTokenizer:
            def decode(self, ids):
                return "a" if ids[0] == 0 else "b"

        scorer = DecodeThenClassifyScorer(TinyTokenizer(), FakeBackend())
        probs = torch.tensor([0.9, 0.1])
        label, _ = scorer.classify_from_probs(probs, mode="argmax")
        self.assertEqual(label, "en")
        label, votes = scorer.classify_from_probs(probs, mode="topk", topk=2)
        self.assertIn(label, {"en", "fr"})
        self.assertTrue(votes)
        label, weighted = scorer.classify_from_probs(probs, mode="topk_weighted", topk=2)
        self.assertEqual(label, "en")
        self.assertAlmostEqual(sum(weighted.values()), 1.0, places=5)
        self.assertGreater(weighted["en"], weighted["fr"])

    def test_classify_text_passes_candidate_languages_to_backend(self):
        class FakeBackend:
            def __init__(self):
                self.last_languages = None

            def predict(self, text, *, languages=None):
                self.last_languages = list(languages) if languages is not None else None
                return "en", {"en": 1.0}

            def predict_distribution(self, text, languages=None):
                return {"en": 1.0}

        class TinyTokenizer:
            def decode(self, ids):
                return "a"

        backend = FakeBackend()
        scorer = DecodeThenClassifyScorer(TinyTokenizer(), backend, candidate_languages=["fr", "en"])
        label, dist = scorer.classify_text("abc")
        self.assertEqual(label, "en")
        self.assertEqual(dist, {"en": 1.0})
        self.assertEqual(backend.last_languages, ["en", "fr"])


if __name__ == "__main__":
    unittest.main()
