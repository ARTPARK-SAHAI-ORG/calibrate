"""
Tests for the intent/entity judge aggregation.

``get_intent_entity_score`` lives in ``calibrate/stt/metrics.py`` (the metric
root). It normalizes reference/prediction via the vendored ``IndicNormalizer``
(mocked here to avoid downloading a model), then delegates to the per-row judge
in ``calibrate/stt/intent_entity.py``, and aggregates with Sarvam's
``calculate_intent_accuracy`` / ``calculate_entity_metrics``.

Run with:
    python -m unittest tests.stt.test_intent_entity -v
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock


def _row(intent, entity):
    return {
        "intent_score": intent,
        "intent_explanation": "because",
        "entity_score": entity,
        "ground_truth_entities": "x",
        "preserved_entities": "x" if entity else "",
        "missing_entities": "" if entity else "x",
        "entity_explanation": "because",
    }


def _identity_normalizer():
    """Mock IndicNormalizer whose normalize_texts returns inputs unchanged."""
    inst = MagicMock()
    inst.normalize_texts.side_effect = lambda texts, langs: list(texts)
    cls = MagicMock(return_value=inst)
    return cls


class TestGetIntentEntityScore(unittest.IsolatedAsyncioTestCase):
    async def test_intent_accuracy_and_entity_mean(self):
        from calibrate.stt import intent_entity as ie
        from calibrate.stt import metrics

        async def fake_judge(reference, prediction, model=None, index=0, context=""):
            mapping = {
                ("a", "a"): _row(1, 1.0),
                ("b", "x"): _row(0, 0.5),
            }
            return mapping[(reference, prediction)]

        with patch.object(metrics, "IndicNormalizer", _identity_normalizer()), \
             patch.object(ie, "intent_entity_judge", AsyncMock(side_effect=fake_judge)):
            result = await metrics.get_intent_entity_score(
                references=["a", "b"],
                predictions=["a", "x"],
            )

        self.assertEqual(result["intent"], 0.5)  # accuracy of [1, 0]
        self.assertEqual(result["entity"], 0.75)  # mean of [1.0, 0.5]
        self.assertEqual(len(result["per_row"]), 2)

    async def test_normalized_text_is_passed_to_judge(self):
        from calibrate.stt import intent_entity as ie
        from calibrate.stt import metrics

        # Normalizer lowercases — the judge must receive the normalized form.
        norm_inst = MagicMock()
        norm_inst.normalize_texts.side_effect = lambda texts, langs: [
            t.lower() for t in texts
        ]
        norm_cls = MagicMock(return_value=norm_inst)

        seen = []

        async def fake_judge(reference, prediction, model=None, index=0, context=""):
            seen.append((reference, prediction))
            return _row(1, 1.0)

        with patch.object(metrics, "IndicNormalizer", norm_cls), \
             patch.object(ie, "intent_entity_judge", AsyncMock(side_effect=fake_judge)):
            await metrics.get_intent_entity_score(
                references=["HELLO"],
                predictions=["Hello"],
            )

        self.assertEqual(seen, [("hello", "hello")])

    async def test_empty_inputs(self):
        from calibrate.stt import intent_entity as ie
        from calibrate.stt import metrics

        with patch.object(metrics, "IndicNormalizer", _identity_normalizer()), \
             patch.object(ie, "intent_entity_judge", AsyncMock()):
            result = await metrics.get_intent_entity_score(references=[], predictions=[])

        self.assertEqual(result["intent"], 0.0)
        self.assertEqual(result["entity"], 0.0)
        self.assertEqual(result["per_row"], [])


if __name__ == "__main__":
    unittest.main()
