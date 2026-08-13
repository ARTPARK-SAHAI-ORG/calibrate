"""
Tests for calibrate_agent/tts/metrics.py — multi-evaluator judge aggregation.

Run with:
    python -m unittest tests.tts.test_metrics -v
"""

import asyncio
import unittest
from unittest.mock import patch, AsyncMock


class TestTTSGetLLMJudgeScore(unittest.IsolatedAsyncioTestCase):
    async def test_default_evaluator_single_judge(self):
        from calibrate_agent.tts import metrics as tts_metrics

        # Patch tts_llm_judge directly (has @backoff + @observe decorators)
        mock_tts_judge = AsyncMock(
            side_effect=[
                {"pronunciation": {"match": True, "reasoning": "clear"}},
                {"pronunciation": {"match": False, "reasoning": "garbled"}},
            ]
        )
        with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
            result = await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
                reference_texts=["hi", "bye"],
            )

        self.assertEqual(list(result["scores"].keys()), ["pronunciation"])
        self.assertEqual(result["scores"]["pronunciation"]["type"], "binary")
        self.assertEqual(result["scores"]["pronunciation"]["mean"], 0.5)
        self.assertEqual(result["score"], 0.5)

    async def test_multi_evaluators_per_row_and_aggregate(self):
        from calibrate_agent.tts import metrics as tts_metrics

        custom_evaluators = [
            {
                "name": "intelligibility",
                "system_prompt": "clear",
                "judge_model": "openai/gpt-4o-audio-preview",
            },
            {
                "name": "pronunciation",
                "system_prompt": "correct",
                "judge_model": "openai/gpt-4o-audio-preview",
            },
        ]
        mock_tts_judge = AsyncMock(
            side_effect=[
                {
                    "intelligibility": {"match": True, "reasoning": "clear"},
                    "pronunciation": {"match": True, "reasoning": "good"},
                },
                {
                    "intelligibility": {"match": True, "reasoning": "clear"},
                    "pronunciation": {"match": False, "reasoning": "mispronounced"},
                },
            ]
        )
        with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
            result = await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
                reference_texts=["hello", "world"],
                evaluators=custom_evaluators,
            )

        self.assertEqual(
            set(result["scores"].keys()), {"intelligibility", "pronunciation"}
        )
        self.assertEqual(result["scores"]["intelligibility"]["mean"], 1.0)
        self.assertEqual(result["scores"]["pronunciation"]["mean"], 0.5)
        self.assertAlmostEqual(result["score"], 0.75)

    async def test_rating_evaluator_aggregates_mean_score(self):
        from calibrate_agent.tts import metrics as tts_metrics

        rating = {
            "name": "naturalness",
            "system_prompt": "rate how natural the speech sounds",
            "judge_model": "openai/gpt-4o-audio-preview",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
        }
        mock_tts_judge = AsyncMock(
            side_effect=[
                {"naturalness": {"score": 5, "reasoning": "very natural"}},
                {"naturalness": {"score": 3, "reasoning": "okay"}},
                {"naturalness": {"score": 4, "reasoning": "good"}},
            ]
        )
        with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
            result = await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav", "/tmp/b.wav", "/tmp/c.wav"],
                reference_texts=["x", "y", "z"],
                evaluators=[rating],
            )

        self.assertEqual(result["scores"]["naturalness"]["type"], "rating")
        # scores (5,3,4) → mean 4.0
        self.assertAlmostEqual(result["scores"]["naturalness"]["mean"], 4.0)
        self.assertEqual(result["scores"]["naturalness"]["scale_min"], 1)
        self.assertEqual(result["scores"]["naturalness"]["scale_max"], 5)

    async def test_custom_evaluators_passed_through(self):
        from calibrate_agent.tts import metrics as tts_metrics

        custom_evaluators = [
            {"name": "x", "system_prompt": "y", "judge_model": "openai/gpt-4o-audio-preview"}
        ]
        mock_tts_judge = AsyncMock(
            return_value={"x": {"match": True, "reasoning": "ok"}}
        )
        with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
            await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav"],
                reference_texts=["text"],
                evaluators=custom_evaluators,
                fallback_model="custom-audio-model",
            )

        call_kwargs = mock_tts_judge.call_args.kwargs
        self.assertEqual(call_kwargs["evaluators"], custom_evaluators)
        self.assertEqual(call_kwargs["fallback_model"], "custom-audio-model")


class TestTTSJudgeOnRow(unittest.IsolatedAsyncioTestCase):
    """``on_row`` fires per row as its judge finishes, without reordering results."""

    EVALUATORS = [
        {"name": "quality", "system_prompt": "q", "judge_model": "openai/gpt-audio"}
    ]

    @staticmethod
    async def _slow_first(audio_path, reference_text, **kwargs):
        # Row 0 finishes last, so completion order differs from dataset order.
        if audio_path.endswith("a.wav"):
            await asyncio.sleep(0.05)
        return {"quality": {"match": True, "reasoning": reference_text}}

    async def test_on_row_called_per_row_and_order_preserved(self):
        from calibrate_agent.tts import metrics as tts_metrics

        seen = []
        with patch.object(
            tts_metrics, "tts_llm_judge", AsyncMock(side_effect=self._slow_first)
        ):
            result = await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
                reference_texts=["hi", "bye"],
                evaluators=self.EVALUATORS,
                on_row=lambda index, row: seen.append((index, row)),
            )

        # Row 1 completes first; each callback carries its own dataset index.
        self.assertEqual([index for index, _ in seen], [1, 0])
        self.assertEqual(seen[0][1]["quality"]["reasoning"], "bye")
        self.assertEqual(seen[1][1]["quality"]["reasoning"], "hi")
        self.assertEqual(
            [row["quality"]["reasoning"] for row in result["per_row"]], ["hi", "bye"]
        )

    async def test_without_on_row_results_unchanged(self):
        from calibrate_agent.tts import metrics as tts_metrics

        with patch.object(
            tts_metrics, "tts_llm_judge", AsyncMock(side_effect=self._slow_first)
        ):
            result = await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
                reference_texts=["hi", "bye"],
                evaluators=self.EVALUATORS,
            )

        self.assertEqual(
            [row["quality"]["reasoning"] for row in result["per_row"]], ["hi", "bye"]
        )
        self.assertEqual(result["scores"]["quality"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
