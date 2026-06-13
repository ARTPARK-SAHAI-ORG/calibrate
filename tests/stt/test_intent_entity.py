"""
Tests for the intent/entity judge aggregation.

``get_intent_entity_score`` lives in ``calibrate/stt/metrics.py`` (the metric
root); it delegates to the per-row judge in ``calibrate/stt/intent_entity.py``,
which is what these tests patch.

Run with:
    python -m unittest tests.stt.test_intent_entity -v
"""

import unittest
from unittest.mock import patch, AsyncMock


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


class TestGetIntentEntityScore(unittest.IsolatedAsyncioTestCase):
    async def test_aggregates_intent_as_passrate_and_entity_as_mean(self):
        from calibrate.stt import intent_entity as ie
        from calibrate.stt import metrics

        # Return per (reference, prediction): order preserved by gather.
        async def fake_judge(reference, prediction, model=None, index=0, context=""):
            mapping = {
                ("a", "a"): _row(1, 1.0),
                ("b", "x"): _row(0, 0.5),
            }
            return mapping[(reference, prediction)]

        with patch.object(ie, "intent_entity_judge", AsyncMock(side_effect=fake_judge)):
            result = await metrics.get_intent_entity_score(
                references=["a", "b"],
                predictions=["a", "x"],
            )

        self.assertEqual(result["intent"], 0.5)  # one 1, one 0
        self.assertEqual(result["entity"], 0.75)  # mean(1.0, 0.5)
        self.assertEqual(len(result["per_row"]), 2)

    async def test_entity_score_clamped_to_unit_interval(self):
        from calibrate.stt import intent_entity as ie
        from calibrate.stt import metrics

        async def fake_judge(reference, prediction, model=None, index=0, context=""):
            # A misbehaving judge returns out-of-range entity scores.
            return _row(1, 1.7) if reference == "hi" else _row(1, -0.3)

        with patch.object(ie, "intent_entity_judge", AsyncMock(side_effect=fake_judge)):
            result = await metrics.get_intent_entity_score(
                references=["hi", "lo"],
                predictions=["hi", "lo"],
            )

        # 1.7 -> 1.0 and -0.3 -> 0.0; mean = 0.5
        self.assertEqual(result["entity"], 0.5)

    async def test_empty_inputs(self):
        from calibrate.stt import intent_entity as ie
        from calibrate.stt import metrics

        with patch.object(ie, "intent_entity_judge", AsyncMock()):
            result = await metrics.get_intent_entity_score(references=[], predictions=[])

        self.assertEqual(result["intent"], 0.0)
        self.assertEqual(result["entity"], 0.0)
        self.assertEqual(result["per_row"], [])


if __name__ == "__main__":
    unittest.main()
