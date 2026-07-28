"""
Tests for calibrate_agent/stt/metrics.py — multi-evaluator judge aggregation.

Run with:
    python -m unittest tests.stt.test_metrics -v
"""

import tempfile
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from calibrate_agent.judge_store import JudgeStore


class TestEditMetrics(unittest.TestCase):
    """WER/CER via jiwer — real computation, no network (jiwer is pure-Python)."""

    def test_get_wer_score_normalizes_case_and_punctuation(self):
        from calibrate_agent.stt import metrics as M

        # Row 1: case-only diff normalizes to identical -> 0.0.
        # Row 2: one of two words wrong -> 0.5.
        result = M.get_wer_score(["Hello, World!", "foo bar"], ["hello world", "foo baz"])

        self.assertEqual(result["per_row"], [0.0, 0.5])

    def test_score_is_dataset_level_not_macro_mean(self):
        from calibrate_agent.stt import metrics as M

        # Row A: 1 error / 2 words. Row B: 0 errors / 4 words.
        # Macro-mean of per-row = (0.5 + 0.0)/2 = 0.25.
        # Dataset-level = total errors / total words = 1/6 ≈ 0.1667.
        result = M.get_wer_score(["hello world", "a b c d"], ["hello word", "a b c d"])

        self.assertEqual(result["per_row"], [0.5, 0.0])
        self.assertAlmostEqual(result["score"], 1 / 6)
        # Confirm it is NOT the naive macro-mean.
        self.assertNotAlmostEqual(result["score"], 0.25)

    def test_get_cer_score_character_level(self):
        from calibrate_agent.stt import metrics as M

        result = M.get_cer_score(["abc"], ["abc"])
        self.assertEqual(result["per_row"], [0.0])
        self.assertEqual(result["score"], 0.0)

        # One char substitution out of three source chars -> 1/3.
        result = M.get_cer_score(["abc"], ["abx"])
        self.assertAlmostEqual(result["score"], 1 / 3)

    def test_non_string_prediction_becomes_empty(self):
        from calibrate_agent.stt import metrics as M

        # None prediction is coerced to "" -> all reference chars are deletions.
        result = M.get_cer_score(["abc"], [None])
        self.assertEqual(result["score"], 1.0)

    def test_nfc_normalization_matches_decomposed_forms(self):
        from calibrate_agent.stt import metrics as M

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
        from calibrate_agent.stt import metrics as M

        result = M.get_wer_score([], [])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["per_row"], [])

    def test_language_aware_indic_normalization_folds_script_variants(self):
        from calibrate_agent.stt import metrics as M

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

    def test_empty_reference_pooled_correctly(self):
        from calibrate_agent.stt import metrics as M

        # Empty GT + empty prediction is correct behaviour → contributes nothing
        # to the pooled score (not penalized).
        mixed = M.get_wer_score(["hello world", ""], ["hello world", ""])
        self.assertEqual(mixed["score"], 0.0)

        # Empty GT + hallucinated prediction → the extra words count as
        # insertions and are penalized. One real ref (4 words, 1 sub) plus two
        # inserted junk words → (1 + 2) / 4 = 0.75.
        halluc = M.get_wer_score(
            ["a b c d", ""], ["a b x d", "junk here"]
        )
        self.assertAlmostEqual(halluc["score"], 0.75)

    def test_all_empty_references_guarded(self):
        from calibrate_agent.stt import metrics as M

        # A dataset with no reference words at all would make jiwer return an
        # unbounded count; the guard returns 0.0 instead.
        self.assertEqual(M.get_wer_score(["", "..."], ["", "junk"])["score"], 0.0)
        self.assertEqual(M.get_cer_score([""], ["hello"])["score"], 0.0)

    def test_unsupported_language_falls_back_gracefully(self):
        from calibrate_agent.stt import metrics as M

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
        from calibrate_agent.stt import metrics as stt_metrics

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
        from calibrate_agent.stt import metrics as stt_metrics

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
        from calibrate_agent.stt import metrics as stt_metrics

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
        from calibrate_agent.stt import metrics as stt_metrics

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


class TestSTTGetLLMJudgeScoreWithStore(unittest.IsolatedAsyncioTestCase):
    async def _fake_judge(self, reference, prediction, evaluators=None, fallback_model=None):
        return {
            ev["name"]: {"match": reference == prediction, "reasoning": "ok"}
            for ev in evaluators
        }

    async def test_store_none_matches_pre_store_behavior(self):
        from calibrate_agent.stt import metrics as stt_metrics

        mock = AsyncMock(side_effect=self._fake_judge)
        evaluators = [
            {"name": "semantic_match", "system_prompt": "match", "judge_model": "m"}
        ]
        with patch.object(stt_metrics, "stt_llm_judge", mock):
            result = await stt_metrics.get_llm_judge_score(
                references=["a", "b"],
                predictions=["a", "x"],
                evaluators=evaluators,
                store=None,
            )

        self.assertEqual(mock.call_count, 2)
        self.assertEqual(result["scores"]["semantic_match"]["mean"], 0.5)

    async def test_prepopulated_store_skips_cached_rows(self):
        from calibrate_agent.stt import metrics as stt_metrics

        evaluators = [
            {"name": "semantic_match", "system_prompt": "match", "judge_model": "m"}
        ]
        mock = AsyncMock(side_effect=self._fake_judge)
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            with patch.object(stt_metrics, "stt_llm_judge", mock):
                # First pass grades only row "r1".
                await stt_metrics.get_llm_judge_score(
                    references=["a"],
                    predictions=["a"],
                    evaluators=evaluators,
                    store=store,
                    row_ids=["r1"],
                )
                self.assertEqual(mock.call_count, 1)

                # Second pass includes "r1" (cached) and "r2" (new).
                mock.reset_mock()
                result = await stt_metrics.get_llm_judge_score(
                    references=["a", "b"],
                    predictions=["a", "y"],
                    evaluators=evaluators,
                    store=store,
                    row_ids=["r1", "r2"],
                )

        self.assertEqual(mock.call_count, 1)
        self.assertEqual(mock.call_args.args[0], "b")
        self.assertEqual(mock.call_args.args[1], "y")
        matches = [row["semantic_match"]["match"] for row in result["per_row"]]
        self.assertEqual(matches, [True, False])

    async def test_row_order_preserved_with_partial_cache(self):
        from calibrate_agent.stt import metrics as stt_metrics

        evaluators = [
            {"name": "semantic_match", "system_prompt": "match", "judge_model": "m"}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            with patch.object(
                stt_metrics, "stt_llm_judge", AsyncMock(side_effect=self._fake_judge)
            ):
                # Cache the middle row only.
                await stt_metrics.get_llm_judge_score(
                    references=["b"],
                    predictions=["b"],
                    evaluators=evaluators,
                    store=store,
                    row_ids=["row1"],
                )
                result = await stt_metrics.get_llm_judge_score(
                    references=["a", "b", "c"],
                    predictions=["x", "b", "c"],
                    evaluators=evaluators,
                    store=store,
                    row_ids=["row0", "row1", "row2"],
                )

        matches = [row["semantic_match"]["match"] for row in result["per_row"]]
        self.assertEqual(matches, [False, True, True])

    async def test_changed_system_prompt_invalidates_row(self):
        from calibrate_agent.stt import metrics as stt_metrics

        base_evaluator = {
            "name": "semantic_match",
            "system_prompt": "prompt v1",
            "judge_model": "m",
        }
        mock = AsyncMock(side_effect=self._fake_judge)
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            with patch.object(stt_metrics, "stt_llm_judge", mock):
                await stt_metrics.get_llm_judge_score(
                    references=["a"],
                    predictions=["a"],
                    evaluators=[base_evaluator],
                    store=store,
                    row_ids=["r1"],
                )
                self.assertEqual(mock.call_count, 1)

                # Same row, same prediction, but the prompt text changed.
                changed_evaluator = dict(base_evaluator, system_prompt="prompt v2")
                await stt_metrics.get_llm_judge_score(
                    references=["a"],
                    predictions=["a"],
                    evaluators=[changed_evaluator],
                    store=store,
                    row_ids=["r1"],
                )

        self.assertEqual(mock.call_count, 2)

    async def test_changed_prediction_invalidates_row(self):
        from calibrate_agent.stt import metrics as stt_metrics

        evaluator = {
            "name": "semantic_match",
            "system_prompt": "match",
            "judge_model": "m",
        }
        mock = AsyncMock(side_effect=self._fake_judge)
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            with patch.object(stt_metrics, "stt_llm_judge", mock):
                await stt_metrics.get_llm_judge_score(
                    references=["a"],
                    predictions=["a"],
                    evaluators=[evaluator],
                    store=store,
                    row_ids=["r1"],
                )
                await stt_metrics.get_llm_judge_score(
                    references=["a"],
                    predictions=["different"],
                    evaluators=[evaluator],
                    store=store,
                    row_ids=["r1"],
                )

        self.assertEqual(mock.call_count, 2)

    async def test_adding_second_evaluator_runs_only_the_new_one(self):
        from calibrate_agent.stt import metrics as stt_metrics

        first_evaluator = {
            "name": "semantic_match",
            "system_prompt": "match",
            "judge_model": "m",
        }
        second_evaluator = {
            "name": "completeness",
            "system_prompt": "complete",
            "judge_model": "m",
        }
        mock = AsyncMock(side_effect=self._fake_judge)
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            with patch.object(stt_metrics, "stt_llm_judge", mock):
                await stt_metrics.get_llm_judge_score(
                    references=["a"],
                    predictions=["a"],
                    evaluators=[first_evaluator],
                    store=store,
                    row_ids=["r1"],
                )
                self.assertEqual(mock.call_count, 1)

                mock.reset_mock()
                result = await stt_metrics.get_llm_judge_score(
                    references=["a"],
                    predictions=["a"],
                    evaluators=[first_evaluator, second_evaluator],
                    store=store,
                    row_ids=["r1"],
                )

        # Only the newly-added evaluator triggers a fresh judge call.
        self.assertEqual(mock.call_count, 1)
        called_names = {ev["name"] for ev in mock.call_args.kwargs["evaluators"]}
        self.assertEqual(called_names, {"completeness"})
        self.assertIn("semantic_match", result["per_row"][0])
        self.assertIn("completeness", result["per_row"][0])


def _identity_normalizer():
    inst = MagicMock()
    inst.normalize_texts.side_effect = lambda texts, langs, n_jobs=1: list(texts)
    return inst


def _intent_entity_row(intent=1, entity=1.0):
    return {
        "intent_score": intent,
        "intent_explanation": "because",
        "entity_score": entity,
        "ground_truth_entities": "x",
        "preserved_entities": "x" if entity else "",
        "missing_entities": "" if entity else "x",
        "entity_explanation": "because",
    }


class TestGetIntentEntityScoreWithStore(unittest.IsolatedAsyncioTestCase):
    async def test_store_none_matches_pre_store_behavior(self):
        from calibrate_agent.stt import metrics as stt_metrics
        from calibrate_agent.stt import sarvam_intent_entity as sie

        mock = AsyncMock(return_value=_intent_entity_row(1, 1.0))
        with patch.object(
            stt_metrics, "_get_indic_normalizer", return_value=_identity_normalizer()
        ), patch.object(sie, "intent_entity_judge", mock):
            result = await stt_metrics.get_intent_entity_score(
                references=["a", "b"], predictions=["a", "b"], store=None
            )

        self.assertEqual(mock.call_count, 2)
        self.assertEqual(result["intent"], 1.0)

    async def test_prepopulated_store_skips_cached_rows(self):
        from calibrate_agent.stt import metrics as stt_metrics
        from calibrate_agent.stt import sarvam_intent_entity as sie

        mock = AsyncMock(return_value=_intent_entity_row(1, 1.0))
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            with patch.object(
                stt_metrics,
                "_get_indic_normalizer",
                return_value=_identity_normalizer(),
            ), patch.object(sie, "intent_entity_judge", mock):
                await stt_metrics.get_intent_entity_score(
                    references=["a"],
                    predictions=["a"],
                    store=store,
                    row_ids=["r1"],
                )
                self.assertEqual(mock.call_count, 1)

                mock.reset_mock()
                result = await stt_metrics.get_intent_entity_score(
                    references=["a", "b"],
                    predictions=["a", "b"],
                    store=store,
                    row_ids=["r1", "r2"],
                )

        self.assertEqual(mock.call_count, 1)
        self.assertEqual(len(result["per_row"]), 2)

    async def test_changed_prediction_invalidates_row(self):
        from calibrate_agent.stt import metrics as stt_metrics
        from calibrate_agent.stt import sarvam_intent_entity as sie

        mock = AsyncMock(return_value=_intent_entity_row(1, 1.0))
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            with patch.object(
                stt_metrics,
                "_get_indic_normalizer",
                return_value=_identity_normalizer(),
            ), patch.object(sie, "intent_entity_judge", mock):
                await stt_metrics.get_intent_entity_score(
                    references=["a"],
                    predictions=["a"],
                    store=store,
                    row_ids=["r1"],
                )
                await stt_metrics.get_intent_entity_score(
                    references=["a"],
                    predictions=["changed"],
                    store=store,
                    row_ids=["r1"],
                )

        self.assertEqual(mock.call_count, 2)


class TestGetLLMWerCerScoreWithStore(unittest.IsolatedAsyncioTestCase):
    async def _fake_judge(self, reference, prediction, model=None):
        return {"equivalent": reference == prediction, "reasoning": "ok"}

    async def test_store_none_matches_pre_store_behavior(self):
        from calibrate_agent.stt import metrics as stt_metrics
        from calibrate_agent.stt import sarvam_llm_wer as slw

        mock = AsyncMock(side_effect=self._fake_judge)
        with patch.object(slw, "equivalence_judge", mock):
            result = await stt_metrics.get_llm_wer_cer_score(
                references=["hello world"],
                predictions=["hello word"],
                store=None,
            )

        mock.assert_awaited()
        self.assertIn("llm_wer", result)

    async def test_prepopulated_store_skips_cached_segment_pairs(self):
        from calibrate_agent.stt import metrics as stt_metrics
        from calibrate_agent.stt import sarvam_llm_wer as slw

        mock = AsyncMock(side_effect=self._fake_judge)
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            with patch.object(slw, "equivalence_judge", mock):
                await stt_metrics.get_llm_wer_cer_score(
                    references=["hello world"],
                    predictions=["hello word"],
                    store=store,
                    row_ids=["r1"],
                )
                first_call_count = mock.call_count
                self.assertGreater(first_call_count, 0)

                mock.reset_mock()
                # Same differing segment ("world" vs "word") reappears in a
                # second row; it should not be re-judged.
                result = await stt_metrics.get_llm_wer_cer_score(
                    references=["hello world", "say world"],
                    predictions=["hello word", "say word"],
                    store=store,
                    row_ids=["r1", "r2"],
                )

        self.assertEqual(mock.call_count, 0)
        self.assertEqual(len(result["per_row"]), 2)

    async def test_changed_prediction_reruns_segment_judge(self):
        from calibrate_agent.stt import metrics as stt_metrics
        from calibrate_agent.stt import sarvam_llm_wer as slw

        mock = AsyncMock(side_effect=self._fake_judge)
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            with patch.object(slw, "equivalence_judge", mock):
                await stt_metrics.get_llm_wer_cer_score(
                    references=["hello world"],
                    predictions=["hello word"],
                    store=store,
                    row_ids=["r1"],
                )
                mock.reset_mock()

                # A different mismatched prediction produces a different
                # segment pair, which must be judged (not silently reused).
                await stt_metrics.get_llm_wer_cer_score(
                    references=["hello world"],
                    predictions=["hello wprld"],
                    store=store,
                    row_ids=["r1"],
                )

        self.assertEqual(mock.call_count, 1)


class TestGetSemanticWerScoreWithStore(unittest.IsolatedAsyncioTestCase):
    async def _fake_judge(self, reference, prediction, model=None):
        if reference == prediction:
            return {
                "substitutions": 0, "deletions": 0, "insertions": 0,
                "reference_words": len(reference.split()) or 1,
                "normalized_reference": reference, "normalized_hypothesis": prediction,
                "reasoning": "match",
            }
        return {
            "substitutions": 1, "deletions": 0, "insertions": 0,
            "reference_words": len(reference.split()) or 1,
            "normalized_reference": reference, "normalized_hypothesis": prediction,
            "reasoning": "mismatch",
        }

    async def test_store_none_matches_pre_store_behavior(self):
        from calibrate_agent.stt import metrics as stt_metrics

        mock = AsyncMock(side_effect=self._fake_judge)
        with patch(
            "calibrate_agent.stt.semantic_wer.semantic_wer_judge",
            mock,
        ):
            result = await stt_metrics.get_semantic_wer_score(
                references=["a", "b"], predictions=["a", "c"], store=None
            )

        self.assertEqual(mock.call_count, 2)
        self.assertEqual(len(result["per_row"]), 2)

    async def test_prepopulated_store_skips_cached_rows(self):
        from calibrate_agent.stt import metrics as stt_metrics

        mock = AsyncMock(side_effect=self._fake_judge)
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            with patch(
                "calibrate_agent.stt.semantic_wer.semantic_wer_judge",
                mock,
            ):
                await stt_metrics.get_semantic_wer_score(
                    references=["a"], predictions=["a"], store=store, row_ids=["r1"]
                )
                self.assertEqual(mock.call_count, 1)

                mock.reset_mock()
                result = await stt_metrics.get_semantic_wer_score(
                    references=["a", "b"],
                    predictions=["a", "c"],
                    store=store,
                    row_ids=["r1", "r2"],
                )

        self.assertEqual(mock.call_count, 1)
        self.assertEqual(mock.call_args.args[0], "b")
        # Row order is preserved even though only row "r2" re-ran.
        self.assertEqual(result["per_row"][0]["reasoning"], "match")
        self.assertEqual(result["per_row"][1]["reasoning"], "mismatch")

    async def test_changed_prediction_invalidates_row(self):
        from calibrate_agent.stt import metrics as stt_metrics

        mock = AsyncMock(side_effect=self._fake_judge)
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            with patch(
                "calibrate_agent.stt.semantic_wer.semantic_wer_judge",
                mock,
            ):
                await stt_metrics.get_semantic_wer_score(
                    references=["a"], predictions=["a"], store=store, row_ids=["r1"]
                )
                await stt_metrics.get_semantic_wer_score(
                    references=["a"],
                    predictions=["changed"],
                    store=store,
                    row_ids=["r1"],
                )

        self.assertEqual(mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
