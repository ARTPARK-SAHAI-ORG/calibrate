"""
Tests for calibrate/stt/metrics.py — multi-evaluator judge aggregation.

Run with:
    python -m unittest tests.stt.test_metrics -v
"""

import unittest
from unittest.mock import patch, AsyncMock


class TestEditMetrics(unittest.TestCase):
    """WER/CER via jiwer — real computation, no network (jiwer is pure-Python)."""

    def test_get_wer_score_normalizes_case_and_punctuation(self):
        from calibrate.stt import metrics as M

        # Row 1: case-only diff normalizes to identical -> 0.0.
        # Row 2: one of two words wrong -> 0.5.
        result = M.get_wer_score(["Hello, World!", "foo bar"], ["hello world", "foo baz"])

        self.assertEqual(result["per_row"], [0.0, 0.5])

    def test_score_is_dataset_level_not_macro_mean(self):
        from calibrate.stt import metrics as M

        # Row A: 1 error / 2 words. Row B: 0 errors / 4 words.
        # Macro-mean of per-row = (0.5 + 0.0)/2 = 0.25.
        # Dataset-level = total errors / total words = 1/6 ≈ 0.1667.
        result = M.get_wer_score(["hello world", "a b c d"], ["hello word", "a b c d"])

        self.assertEqual(result["per_row"], [0.5, 0.0])
        self.assertAlmostEqual(result["score"], 1 / 6)
        # Confirm it is NOT the naive macro-mean.
        self.assertNotAlmostEqual(result["score"], 0.25)

    def test_get_cer_score_character_level(self):
        from calibrate.stt import metrics as M

        result = M.get_cer_score(["abc"], ["abc"])
        self.assertEqual(result["per_row"], [0.0])
        self.assertEqual(result["score"], 0.0)

        # One char substitution out of three source chars -> 1/3.
        result = M.get_cer_score(["abc"], ["abx"])
        self.assertAlmostEqual(result["score"], 1 / 3)

    def test_non_string_prediction_becomes_empty(self):
        from calibrate.stt import metrics as M

        # None prediction is coerced to "" -> all reference chars are deletions.
        result = M.get_cer_score(["abc"], [None])
        self.assertEqual(result["score"], 1.0)

    def test_nfc_normalization_matches_decomposed_forms(self):
        from calibrate.stt import metrics as M

        # Devanagari QA: precomposed single code point (U+0958) vs decomposed
        # KA + NUKTA (U+0915 U+093C). Different raw code points, same character.
        composed = "क़"
        decomposed = "क़"
        self.assertNotEqual(composed, decomposed)

        # NFC folds them together, so no spurious character-level edit is counted.
        result = M.get_cer_score([composed], [decomposed])
        self.assertEqual(result["score"], 0.0)

        # Also holds for Latin composed vs decomposed accents.
        self.assertEqual(M.get_cer_score(["café"], ["café"])["score"], 0.0)

    def test_empty_input_returns_zero_score(self):
        from calibrate.stt import metrics as M

        result = M.get_wer_score([], [])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["per_row"], [])

    def test_language_aware_indic_normalization_folds_script_variants(self):
        from calibrate.stt import metrics as M

        # Same Hindi word with vs. without a zero-width joiner (ZWJ). NFC and
        # punctuation removal leave the ZWJ in place; the Hindi IndicNormalizer
        # strips it, so the two spellings become identical.
        with_zwj = "क्‍ष"
        without_zwj = "क्ष"
        self.assertNotEqual(with_zwj, without_zwj)

        # english path (no IndicNormalizer) still sees a difference...
        self.assertGreater(
            M.get_cer_score([with_zwj], [without_zwj], language="english")["score"],
            0.0,
        )
        # ...but the Hindi path folds them to zero error.
        self.assertEqual(
            M.get_cer_score([with_zwj], [without_zwj], language="hindi")["score"],
            0.0,
        )

    def test_empty_reference_uses_sentinel_not_inflated_score(self):
        from calibrate.stt import metrics as M

        # A reference that normalizes to nothing becomes the <empty> sentinel:
        # it contributes one ordinary word error, so the score stays at 1.0
        # instead of blowing past 1.0 (the pre-sentinel behaviour).
        result = M.get_wer_score(["", "..."], ["hello", "world"], language="hindi")
        self.assertEqual(result["score"], 1.0)

    def test_unsupported_language_falls_back_gracefully(self):
        from calibrate.stt import metrics as M

        # Urdu (needs an optional dep indic-nlp lacks here) and an unknown
        # language must not crash — they fall back to NFC-only normalization.
        for lang in ("urdu", "klingon"):
            result = M.get_wer_score(
                ["hello world"], ["hello world"], language=lang
            )
            self.assertEqual(result["score"], 0.0)
        self.assertIsNone(M._indic_normalizer("english"))
        self.assertIsNotNone(M._indic_normalizer("hindi"))


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
