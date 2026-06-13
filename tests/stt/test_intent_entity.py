"""
Tests for the intent/entity judge aggregation.

``get_intent_entity_score`` lives in ``calibrate/stt/metrics.py`` (the metric
root). It normalizes reference/prediction via the vendored ``IndicNormalizer``
(mocked here to avoid downloading a model), then delegates to the per-row judge
in ``calibrate/stt/sarvam_intent_entity/judge.py``, and aggregates with Sarvam's
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
        from calibrate.stt import sarvam_intent_entity as sie
        from calibrate.stt import metrics

        async def fake_judge(reference, prediction, model=None, index=0, context=""):
            mapping = {
                ("a", "a"): _row(1, 1.0),
                ("b", "x"): _row(0, 0.5),
            }
            return mapping[(reference, prediction)]

        with patch.object(metrics, "IndicNormalizer", _identity_normalizer()), \
             patch.object(sie, "intent_entity_judge", AsyncMock(side_effect=fake_judge)):
            result = await metrics.get_intent_entity_score(
                references=["a", "b"],
                predictions=["a", "x"],
            )

        self.assertEqual(result["intent"], 0.5)  # accuracy of [1, 0]
        self.assertEqual(result["entity"], 0.75)  # mean of [1.0, 0.5]
        self.assertEqual(len(result["per_row"]), 2)

    async def test_normalized_text_is_passed_to_judge(self):
        from calibrate.stt import sarvam_intent_entity as sie
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
             patch.object(sie, "intent_entity_judge", AsyncMock(side_effect=fake_judge)):
            await metrics.get_intent_entity_score(
                references=["HELLO"],
                predictions=["Hello"],
            )

        self.assertEqual(seen, [("hello", "hello")])

    async def test_empty_inputs(self):
        from calibrate.stt import sarvam_intent_entity as sie
        from calibrate.stt import metrics

        with patch.object(metrics, "IndicNormalizer", _identity_normalizer()), \
             patch.object(sie, "intent_entity_judge", AsyncMock()):
            result = await metrics.get_intent_entity_score(references=[], predictions=[])

        self.assertEqual(result["intent"], 0.0)
        self.assertEqual(result["entity"], 0.0)
        self.assertEqual(result["per_row"], [])


class TestIntentEntityJudge(unittest.IsolatedAsyncioTestCase):
    async def test_judge_builds_prompt_and_returns_model_dump(self):
        from calibrate.stt.sarvam_intent_entity import judge as ie

        fake_result = {
            "index": 3,
            "intent_score": 1,
            "intent_explanation": "ok",
            "entity_score": 0.5,
            "ground_truth_entities": "x",
            "preserved_entities": "x",
            "missing_entities": "",
            "entity_explanation": "ok",
        }
        fake_response = MagicMock()
        fake_response.model_dump.return_value = fake_result

        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

        # Bypass the real decorators (backoff/observe) and skip the network.
        inner = ie.intent_entity_judge
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__

        with patch.object(ie, "_build_openrouter_client", return_value=MagicMock()), \
             patch.object(ie.instructor, "apatch", return_value=fake_client):
            result = await inner(
                "hello world", "helo world", model="m", index=3, context="ctx"
            )

        self.assertEqual(result, fake_result)
        # The vendored build_prompt was used: the user message carries the
        # input JSON with the normalized hypothesis/ground_truth.
        _, kwargs = fake_client.chat.completions.create.call_args
        sent = kwargs["messages"][0]["content"]
        self.assertIn('"hypothesis": "helo world"', sent)
        self.assertIn('"ground_truth": "hello world"', sent)
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(kwargs["response_model"], ie.IntentEntityResponse)


if __name__ == "__main__":
    unittest.main()
