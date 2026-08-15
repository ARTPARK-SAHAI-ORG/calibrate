"""
Tests for calibrate_agent/tts/metrics.py — multi-evaluator judge aggregation.

Run with:
    python -m unittest tests.tts.test_metrics -v
"""

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


TEMPLATED_EV = {
    "name": "pronunciation",
    "system_prompt": "judge against {{dialect}}",
    "judge_model": "openai/gpt-4o-audio-preview",
}


class TestTTSGetLLMJudgeScoreArguments(unittest.IsolatedAsyncioTestCase):
    async def test_arguments_list_none_regression(self):
        # arguments_list=None: evaluators reach the judge untouched.
        from calibrate_agent.tts import metrics as tts_metrics

        seen = []

        async def fake(evaluators, audio_path, reference_text, fallback_model):
            seen.append(evaluators)
            return {"pronunciation": {"reasoning": "ok", "match": True}}

        with patch.object(tts_metrics, "audio_judge", AsyncMock(side_effect=fake)):
            result = await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
                reference_texts=["hi", "bye"],
                evaluators=[TEMPLATED_EV],
            )

        self.assertAlmostEqual(result["scores"]["pronunciation"]["mean"], 1.0)
        for evaluators in seen:
            self.assertEqual(
                evaluators[0]["system_prompt"], "judge against {{dialect}}"
            )

    async def test_arguments_injected_per_row(self):
        # Per-row args are keyed by evaluator name and reach audio_judge.
        from calibrate_agent.tts import metrics as tts_metrics

        seen_by_audio = {}

        async def fake(evaluators, audio_path, reference_text, fallback_model):
            seen_by_audio[audio_path] = evaluators[0]["system_prompt"]
            return {"pronunciation": {"reasoning": "ok", "match": True}}

        with patch.object(tts_metrics, "audio_judge", AsyncMock(side_effect=fake)):
            await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
                reference_texts=["hi", "bye"],
                evaluators=[TEMPLATED_EV],
                arguments_list=[
                    {"pronunciation": {"dialect": "Indian English"}},
                    {"pronunciation": {"dialect": "British English"}},
                ],
            )

        self.assertEqual(
            seen_by_audio["/tmp/a.wav"], "judge against Indian English"
        )
        self.assertEqual(
            seen_by_audio["/tmp/b.wav"], "judge against British English"
        )

    async def test_arguments_target_only_named_evaluator(self):
        # An evaluator with no entry in the row's args is left unrendered,
        # while a sibling evaluator named in the args is rendered.
        from calibrate_agent.tts import metrics as tts_metrics

        other_ev = {
            "name": "naturalness",
            "system_prompt": "rate against {{dialect}}",
            "judge_model": "openai/gpt-4o-audio-preview",
        }
        seen = {}

        async def fake(evaluators, audio_path, reference_text, fallback_model):
            seen.update({ev["name"]: ev["system_prompt"] for ev in evaluators})
            return {
                "pronunciation": {"reasoning": "ok", "match": True},
                "naturalness": {"reasoning": "ok", "match": True},
            }

        with patch.object(tts_metrics, "audio_judge", AsyncMock(side_effect=fake)):
            await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav"],
                reference_texts=["hi"],
                evaluators=[TEMPLATED_EV, other_ev],
                arguments_list=[{"pronunciation": {"dialect": "Indian English"}}],
            )

        self.assertEqual(seen["pronunciation"], "judge against Indian English")
        self.assertEqual(seen["naturalness"], "rate against {{dialect}}")

    async def test_unknown_evaluator_in_arguments_raises(self):
        # A typo'd / stale evaluator name in a row's args must fail loudly.
        from calibrate_agent.tts import metrics as tts_metrics

        with self.assertRaises(ValueError) as ctx:
            await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav"],
                reference_texts=["hi"],
                evaluators=[TEMPLATED_EV],
                arguments_list=[{"pronounciation": {"dialect": "Indian English"}}],
            )
        self.assertIn("pronounciation", str(ctx.exception))

    async def test_length_mismatch_raises(self):
        from calibrate_agent.tts import metrics as tts_metrics

        with self.assertRaises(ValueError):
            await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
                reference_texts=["hi", "bye"],
                evaluators=[TEMPLATED_EV],
                arguments_list=[{"pronunciation": {"dialect": "Indian English"}}],
            )


if __name__ == "__main__":
    unittest.main()
