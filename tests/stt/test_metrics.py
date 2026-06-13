"""
Tests for calibrate/stt/metrics.py — multi-evaluator judge aggregation.

Run with:
    python -m unittest tests.stt.test_metrics -v
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock


class TestEditMetrics(unittest.TestCase):
    """WER/CER share ``_edit_metric``; ``load`` is mocked to stay pure-unit."""

    def _fake_load(self, recorder):
        """Return a fake ``evaluate`` metric that records the per-row inputs."""
        metric = MagicMock()

        def compute(predictions, references):
            recorder.append((references[0], predictions[0]))
            # toy score: 1.0 when ref/pred differ, else 0.0
            return 0.0 if references[0] == predictions[0] else 1.0

        metric.compute.side_effect = compute
        return metric

    def test_get_wer_score_loads_wer_and_normalizes(self):
        from calibrate.stt import metrics as M

        seen = []
        with patch.object(M, "load", return_value=self._fake_load(seen)) as mock_load:
            result = M.get_wer_score(["Hello World", "foo"], ["hello world", "bar"])

        mock_load.assert_called_once_with("wer")
        # Row 1 ref/pred normalize identically (case-folded) -> 0.0; row 2 differs -> 1.0.
        self.assertEqual(result["per_row"], [0.0, 1.0])
        self.assertEqual(result["score"], 0.5)
        # Normalizer was applied before scoring (case-folded to the same string).
        self.assertEqual(seen[0], ("hello world", "hello world"))

    def test_get_cer_score_loads_cer(self):
        from calibrate.stt import metrics as M

        seen = []
        with patch.object(M, "load", return_value=self._fake_load(seen)) as mock_load:
            result = M.get_cer_score(["abc"], ["abc"])

        mock_load.assert_called_once_with("cer")
        self.assertEqual(result["per_row"], [0.0])
        self.assertEqual(result["score"], 0.0)

    def test_non_string_prediction_becomes_empty(self):
        from calibrate.stt import metrics as M

        seen = []
        with patch.object(M, "load", return_value=self._fake_load(seen)):
            M.get_cer_score(["abc"], [None])

        # None prediction is coerced to "" before scoring.
        self.assertEqual(seen[0][1], "")


class TestSTTGetLLMJudgeScore(unittest.IsolatedAsyncioTestCase):
    async def test_default_evaluator_single_judge(self):
        from calibrate.stt import metrics as stt_metrics

        # Patch stt_llm_judge directly (it has @backoff + @observe decorators
        # so patching text_judge inside it is unreliable).
        # tqdm_asyncio.gather may not preserve input order, so return based on input.
        async def fake_judge(reference, prediction, evaluators=None, fallback_model=None):
            match = reference == prediction
            return {
                "semantic_match": {
                    "match": match,
                    "reasoning": "ok" if match else "mismatch",
                }
            }

        with patch.object(stt_metrics, "stt_llm_judge", AsyncMock(side_effect=fake_judge)):
            result = await stt_metrics.get_llm_judge_score(
                references=["hello", "goodnight"],
                predictions=["hello", "goodbye"],  # first matches, second doesn't
            )

        self.assertEqual(list(result["scores"].keys()), ["semantic_match"])
        self.assertEqual(result["scores"]["semantic_match"]["type"], "binary")
        self.assertEqual(result["scores"]["semantic_match"]["mean"], 0.5)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(len(result["per_row"]), 2)
        # Tally per_row matches: exactly one True and one False
        matches = [row["semantic_match"]["match"] for row in result["per_row"]]
        self.assertEqual(sorted(matches), [False, True])

    async def test_multi_evaluators_per_row_and_aggregate(self):
        from calibrate.stt import metrics as stt_metrics

        custom_evaluators = [
            {
                "name": "semantic_match",
                "system_prompt": "values match",
                "judge_model": "openai/gpt-4.1",
            },
            {
                "name": "completeness",
                "system_prompt": "nothing missing",
                "judge_model": "openai/gpt-4.1",
            },
        ]
        mock_stt_judge = AsyncMock(
            side_effect=[
                {
                    "semantic_match": {"match": True, "reasoning": "ok"},
                    "completeness": {"match": True, "reasoning": "all there"},
                },
                {
                    "semantic_match": {"match": True, "reasoning": "ok"},
                    "completeness": {"match": False, "reasoning": "missing word"},
                },
            ]
        )

        with patch.object(stt_metrics, "stt_llm_judge", mock_stt_judge):
            result = await stt_metrics.get_llm_judge_score(
                references=["hello world", "foo bar"],
                predictions=["hello world", "foo"],
                evaluators=custom_evaluators,
            )

        self.assertEqual(
            set(result["scores"].keys()), {"semantic_match", "completeness"}
        )
        self.assertEqual(result["scores"]["semantic_match"]["mean"], 1.0)
        self.assertEqual(result["scores"]["completeness"]["mean"], 0.5)
        self.assertEqual(result["scores"]["semantic_match"]["type"], "binary")
        # Overall score is mean across evaluators
        self.assertAlmostEqual(result["score"], 0.75)

    async def test_rating_evaluator_aggregates_mean_score(self):
        from calibrate.stt import metrics as stt_metrics

        rating_evaluator = {
            "name": "semantic_accuracy",
            "system_prompt": "rate semantic accuracy",
            "judge_model": "openai/gpt-4.1",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
        }

        async def fake_judge(reference, prediction, evaluators=None, fallback_model=None):
            # Return score based on whether strings match: match=5, mismatch=2
            return {
                "semantic_accuracy": {
                    "reasoning": "ok",
                    "score": 5 if reference == prediction else 2,
                }
            }

        with patch.object(stt_metrics, "stt_llm_judge", AsyncMock(side_effect=fake_judge)):
            result = await stt_metrics.get_llm_judge_score(
                references=["hello", "world", "foo"],
                predictions=["hello", "word", "foo"],  # 2 match, 1 doesn't
                evaluators=[rating_evaluator],
            )

        self.assertEqual(result["scores"]["semantic_accuracy"]["type"], "rating")
        # Two 5s and one 2 → mean = 12/3 = 4.0
        self.assertAlmostEqual(result["scores"]["semantic_accuracy"]["mean"], 4.0)
        self.assertEqual(result["scores"]["semantic_accuracy"]["scale_min"], 1)
        self.assertEqual(result["scores"]["semantic_accuracy"]["scale_max"], 5)

    async def test_custom_evaluators_passed_through(self):
        from calibrate.stt import metrics as stt_metrics

        custom_evaluators = [
            {"name": "x", "system_prompt": "y", "judge_model": "openai/gpt-4.1"}
        ]
        mock_stt_judge = AsyncMock(
            return_value={"x": {"match": True, "reasoning": "ok"}}
        )

        with patch.object(stt_metrics, "stt_llm_judge", mock_stt_judge):
            await stt_metrics.get_llm_judge_score(
                references=["ref"],
                predictions=["pred"],
                evaluators=custom_evaluators,
                fallback_model="custom-model",
            )

        # stt_llm_judge is called positionally for reference/prediction
        call_kwargs = mock_stt_judge.call_args.kwargs
        self.assertEqual(call_kwargs["evaluators"], custom_evaluators)
        self.assertEqual(call_kwargs["fallback_model"], "custom-model")


if __name__ == "__main__":
    unittest.main()
