"""
Tests for the vendored Sarvam intent/entity helpers.

Covers ``build_prompt`` + ``IntentEntityResponse`` (main.py) and the
``IndicNormalizer`` / score aggregators (utilities.py). The normalizer is
exercised with its heavyweight backends (Whisper tokenizer + indic-nlp factory)
mocked so no model is downloaded.
"""

import unittest
from unittest.mock import MagicMock


class TestBuildPrompt(unittest.TestCase):
    def test_prompt_carries_input_json_and_template(self):
        from calibrate.stt.sarvam_intent_entity.main import build_prompt, PROMPT_TEMPLATE

        prompt = build_prompt(
            {
                "index": 2,
                "hypothesis": "helo",
                "ground_truth": "hello",
                "context": "greeting",
            }
        )
        self.assertTrue(prompt.startswith(PROMPT_TEMPLATE))
        self.assertIn("**INPUT:**", prompt)
        self.assertIn('"index": 2', prompt)
        self.assertIn('"hypothesis": "helo"', prompt)
        self.assertIn('"ground_truth": "hello"', prompt)
        self.assertIn('"context": "greeting"', prompt)

    def test_context_defaults_to_empty(self):
        from calibrate.stt.sarvam_intent_entity.main import build_prompt

        prompt = build_prompt({"index": 0, "hypothesis": "a", "ground_truth": "b"})
        self.assertIn('"context": ""', prompt)

    def test_response_model_fields_are_bare(self):
        from calibrate.stt.sarvam_intent_entity.main import IntentEntityResponse

        self.assertEqual(
            list(IntentEntityResponse.model_fields.keys()),
            [
                "index",
                "intent_score",
                "intent_explanation",
                "entity_score",
                "ground_truth_entities",
                "preserved_entities",
                "missing_entities",
                "entity_explanation",
            ],
        )


class TestScoreAggregators(unittest.TestCase):
    def test_intent_accuracy(self):
        from calibrate.stt.sarvam_intent_entity.utilities import (
            calculate_intent_accuracy,
        )

        self.assertEqual(calculate_intent_accuracy([1, 0, 1, 0]), 0.5)
        self.assertEqual(calculate_intent_accuracy([]), 0.0)

    def test_entity_metrics(self):
        from calibrate.stt.sarvam_intent_entity.utilities import (
            calculate_entity_metrics,
        )

        m = calculate_entity_metrics([1.0, 0.0])
        self.assertEqual(m["mean"], 0.5)
        self.assertEqual(m["median"], 0.5)
        self.assertEqual(calculate_entity_metrics([]), {"mean": 0.0, "median": 0.0, "std": 0.0})


def _normalizer_with_mocks():
    """IndicNormalizer instance with its model backends mocked (no download)."""
    from calibrate.stt.sarvam_intent_entity.utilities import IndicNormalizer

    norm = object.__new__(IndicNormalizer)  # skip __init__ (model download)
    indic_normalizer = MagicMock()
    indic_normalizer.normalize.side_effect = lambda t: t
    norm.indic_factory = MagicMock()
    norm.indic_factory.get_normalizer.return_value = indic_normalizer
    norm.whisper_tokenizer = MagicMock()
    norm.whisper_tokenizer.normalize.side_effect = lambda t: t
    return norm


class TestIndicNormalizer(unittest.TestCase):
    def test_english_uses_whisper_path_and_strips_punctuation(self):
        norm = _normalizer_with_mocks()
        out = norm.normalize_text("Hello, World!", "english")
        self.assertEqual(out, "hello world")
        norm.whisper_tokenizer.normalize.assert_called_once()
        norm.indic_factory.get_normalizer.assert_not_called()

    def test_indic_lang_uses_indic_normalizer(self):
        norm = _normalizer_with_mocks()
        norm.normalize_text("Ram", "hindi")
        norm.indic_factory.get_normalizer.assert_called_once_with("hi")
        norm.whisper_tokenizer.normalize.assert_not_called()

    def test_empty_and_non_str_passthrough(self):
        norm = _normalizer_with_mocks()
        self.assertEqual(norm.normalize_text("", "english"), "")
        self.assertIsNone(norm.normalize_text(None, "english"))

    def test_normalize_texts_batches(self):
        norm = _normalizer_with_mocks()
        out = norm.normalize_texts(["Hi!", "Bye."], ["english", "english"], n_jobs=1)
        self.assertEqual(out, ["hi", "bye"])

    def test_normalize_texts_empty(self):
        norm = _normalizer_with_mocks()
        self.assertEqual(norm.normalize_texts([], [], n_jobs=1), [])

    def test_normalize_texts_length_mismatch_raises(self):
        norm = _normalizer_with_mocks()
        with self.assertRaises(ValueError):
            norm.normalize_texts(["a"], ["english", "hindi"], n_jobs=1)


if __name__ == "__main__":
    unittest.main()
