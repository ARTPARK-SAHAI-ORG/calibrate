"""Unit tests for calibrate/general/metrics.py.

Covers the general (non-conversational) task scoring path:
- _require_evaluators rejects empty / non-list evaluators
- general_judge delegates to general_task_judge with input + output
- get_general_judge_score fans out per row and aggregates binary + rating
  scores into the {scores, score, per_row} shape
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from calibrate.general.metrics import (
    general_judge,
    get_general_judge_score,
    _require_evaluators,
)


BINARY_EV = {
    "name": "faithful",
    "system_prompt": "judge faithfulness",
    "judge_model": "openai/gpt-4.1",
}
RATING_EV = {
    "name": "quality",
    "type": "rating",
    "scale_min": 1,
    "scale_max": 5,
    "system_prompt": "rate quality",
    "judge_model": "openai/gpt-4.1",
}


class TestRequireEvaluators(unittest.TestCase):
    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            _require_evaluators([])

    def test_none_raises(self):
        with self.assertRaises(ValueError):
            _require_evaluators(None)

    def test_duplicate_names_raise(self):
        with self.assertRaises(ValueError):
            _require_evaluators([BINARY_EV, dict(BINARY_EV)])

    def test_valid_list_returned(self):
        out = _require_evaluators([BINARY_EV])
        self.assertEqual(out, [BINARY_EV])


class TestGeneralJudge(unittest.IsolatedAsyncioTestCase):
    async def test_delegates_with_input_and_output(self):
        mock = AsyncMock(return_value={"faithful": {"reasoning": "ok", "match": True}})
        with patch("calibrate.general.metrics.general_task_judge", mock):
            result = await general_judge(
                input_text="the source",
                output="the summary",
                evaluators=[BINARY_EV],
            )
        self.assertEqual(result, {"faithful": {"reasoning": "ok", "match": True}})
        kwargs = mock.call_args.kwargs
        self.assertEqual(kwargs["output"], "the summary")
        self.assertEqual(kwargs["input_text"], "the source")
        self.assertEqual(kwargs["evaluators"], [BINARY_EV])

    async def test_updates_langfuse_trace_when_enabled(self):
        judge = AsyncMock(return_value={"faithful": {"reasoning": "ok", "match": True}})
        lf = MagicMock()
        with patch("calibrate.general.metrics.general_task_judge", judge), patch(
            "calibrate.general.metrics.langfuse_enabled", True
        ), patch("calibrate.general.metrics.langfuse", lf):
            await general_judge(
                input_text="src", output="out", evaluators=[BINARY_EV]
            )
        lf.update_current_trace.assert_called_once()
        call = lf.update_current_trace.call_args.kwargs
        self.assertEqual(call["input"], {"input": "src", "output": "out"})


class TestGetGeneralJudgeScore(unittest.IsolatedAsyncioTestCase):
    async def test_requires_evaluators(self):
        with self.assertRaises(ValueError):
            await get_general_judge_score(["a"], ["b"], evaluators=[])

    async def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            await get_general_judge_score(["a", "b"], ["x"], evaluators=[BINARY_EV])

    async def test_aggregates_binary_scores(self):
        # Two rows: o1 passes, o2 fails → mean 0.5. The result for each row is
        # derived from its own ``output`` arg (not a positional side_effect
        # list) so concurrent completion order cannot scramble the mapping.
        async def fake(_input, output, **kwargs):
            return {"faithful": {"reasoning": output, "match": output == "o1"}}

        with patch("calibrate.general.metrics.general_judge", AsyncMock(side_effect=fake)):
            result = await get_general_judge_score(
                inputs=["i1", "i2"],
                outputs=["o1", "o2"],
                evaluators=[BINARY_EV],
            )
        self.assertEqual(result["scores"]["faithful"]["type"], "binary")
        self.assertAlmostEqual(result["scores"]["faithful"]["mean"], 0.5)
        self.assertAlmostEqual(result["score"], 0.5)
        # per_row preserves input order
        self.assertEqual(
            result["per_row"],
            [
                {"faithful": {"reasoning": "o1", "match": True}},
                {"faithful": {"reasoning": "o2", "match": False}},
            ],
        )

    async def test_aggregates_rating_scores(self):
        scores_by_output = {"o1": 4, "o2": 2}

        async def fake(_input, output, **kwargs):
            return {"quality": {"reasoning": output, "score": scores_by_output[output]}}

        with patch("calibrate.general.metrics.general_judge", AsyncMock(side_effect=fake)):
            result = await get_general_judge_score(
                inputs=["i1", "i2"],
                outputs=["o1", "o2"],
                evaluators=[RATING_EV],
            )
        score = result["scores"]["quality"]
        self.assertEqual(score["type"], "rating")
        self.assertAlmostEqual(score["mean"], 3.0)
        self.assertEqual(score["scale_min"], 1)
        self.assertEqual(score["scale_max"], 5)

    async def test_none_input_passed_through(self):
        mock = AsyncMock(return_value={"faithful": {"reasoning": "ok", "match": True}})
        with patch("calibrate.general.metrics.general_judge", mock):
            await get_general_judge_score(
                inputs=[None],
                outputs=["only-output"],
                evaluators=[BINARY_EV],
            )
        # First positional arg to general_judge is the input — stays None
        args = mock.call_args.args
        self.assertIsNone(args[0])
        self.assertEqual(args[1], "only-output")


if __name__ == "__main__":
    unittest.main()
