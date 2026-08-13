"""
Tests for calibrate_agent/tts/metrics.py — multi-evaluator judge aggregation.

Run with:
    python -m unittest tests.tts.test_metrics -v
"""

import os
import tempfile
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


def _write_audio(path: str, content: bytes = b"RIFF-fake-audio-bytes") -> None:
    with open(path, "wb") as f:
        f.write(content)


class TestTTSGetLLMJudgeScoreWithStore(unittest.IsolatedAsyncioTestCase):
    """Resumable-checkpoint behavior of get_tts_llm_judge_score."""

    async def test_store_none_behaves_exactly_as_before(self):
        from calibrate_agent.tts import metrics as tts_metrics

        mock_tts_judge = AsyncMock(
            side_effect=[
                {"pronunciation": {"match": True, "reasoning": "clear"}},
                {"pronunciation": {"match": False, "reasoning": "garbled"}},
            ]
        )
        with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
            result = await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/does-not-exist-a.wav", "/tmp/does-not-exist-b.wav"],
                reference_texts=["hi", "bye"],
            )

        self.assertEqual(mock_tts_judge.await_count, 2)
        self.assertEqual(result["scores"]["pronunciation"]["mean"], 0.5)

    async def test_prepopulated_store_only_judges_uncached_rows(self):
        from calibrate_agent.tts import metrics as tts_metrics
        from calibrate_agent.judge_store import JudgeStore, JudgeKey, make_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            audio_a = os.path.join(tmp, "a.wav")
            audio_b = os.path.join(tmp, "b.wav")
            _write_audio(audio_a, b"content-a")
            _write_audio(audio_b, b"content-b")

            store = JudgeStore.load(tmp)
            fp_a = make_fingerprint(
                tts_metrics._audio_content_hash(audio_a),
                "hello",
                tts_metrics.DEFAULT_TTS_EVALUATOR["system_prompt"],
                tts_metrics.DEFAULT_TTS_EVALUATOR["judge_model"],
                "binary",
            )
            key_a = JudgeKey(
                kind="tts_evaluators",
                row_id="row_a",
                fingerprint=fp_a,
                evaluator="pronunciation",
            )
            await store.put(key_a, {"match": True, "reasoning": "cached"})

            mock_tts_judge = AsyncMock(
                return_value={"pronunciation": {"match": False, "reasoning": "fresh"}}
            )
            with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
                result = await tts_metrics.get_tts_llm_judge_score(
                    audio_paths=[audio_a, audio_b],
                    reference_texts=["hello", "world"],
                    store=store,
                    row_ids=["row_a", "row_b"],
                )

            # Only row_b (uncached) triggers a judge call.
            mock_tts_judge.assert_awaited_once_with(
                audio_b,
                "world",
                evaluators=[tts_metrics.DEFAULT_TTS_EVALUATOR],
                fallback_model=tts_metrics.DEFAULT_TTS_JUDGE_MODEL,
            )
            self.assertEqual(
                result["per_row"][0]["pronunciation"],
                {"match": True, "reasoning": "cached"},
            )
            self.assertEqual(
                result["per_row"][1]["pronunciation"],
                {"match": False, "reasoning": "fresh"},
            )

    async def test_results_returned_in_row_order_with_partial_cache(self):
        from calibrate_agent.tts import metrics as tts_metrics
        from calibrate_agent.judge_store import JudgeStore, JudgeKey, make_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            paths = [os.path.join(tmp, f"{i}.wav") for i in range(3)]
            for i, p in enumerate(paths):
                _write_audio(p, f"content-{i}".encode())

            store = JudgeStore.load(tmp)
            # Cache only the middle row.
            fp = make_fingerprint(
                tts_metrics._audio_content_hash(paths[1]),
                "text-1",
                tts_metrics.DEFAULT_TTS_EVALUATOR["system_prompt"],
                tts_metrics.DEFAULT_TTS_EVALUATOR["judge_model"],
                "binary",
            )
            key = JudgeKey(
                kind="tts_evaluators", row_id="1", fingerprint=fp, evaluator="pronunciation"
            )
            await store.put(key, {"match": True, "reasoning": "cached-middle"})

            mock_tts_judge = AsyncMock(
                return_value={"pronunciation": {"match": False, "reasoning": "fresh"}}
            )
            with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
                result = await tts_metrics.get_tts_llm_judge_score(
                    audio_paths=paths,
                    reference_texts=["text-0", "text-1", "text-2"],
                    store=store,
                    row_ids=["0", "1", "2"],
                )

            self.assertEqual(mock_tts_judge.await_count, 2)
            self.assertEqual(
                result["per_row"][1]["pronunciation"]["reasoning"], "cached-middle"
            )
            self.assertEqual(
                result["per_row"][0]["pronunciation"]["reasoning"], "fresh"
            )
            self.assertEqual(
                result["per_row"][2]["pronunciation"]["reasoning"], "fresh"
            )

    async def test_rewriting_audio_bytes_invalidates_cached_row(self):
        """A re-synthesized clip at the same path is re-judged, not reused."""
        from calibrate_agent.tts import metrics as tts_metrics
        from calibrate_agent.judge_store import JudgeStore

        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = tempfile.mkdtemp()
            audio_path = os.path.join(audio_dir, "a.wav")
            _write_audio(audio_path, b"first-take")

            store = JudgeStore.load(tmp)
            mock_tts_judge = AsyncMock(
                return_value={"pronunciation": {"match": True, "reasoning": "r1"}}
            )
            with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
                await tts_metrics.get_tts_llm_judge_score(
                    audio_paths=[audio_path],
                    reference_texts=["hello"],
                    store=store,
                    row_ids=["row_a"],
                )
            self.assertEqual(mock_tts_judge.await_count, 1)

            # Re-synthesize: same path, different bytes.
            _write_audio(audio_path, b"second-take-different-bytes")
            mock_tts_judge.reset_mock()
            mock_tts_judge.return_value = {
                "pronunciation": {"match": False, "reasoning": "r2"}
            }
            with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
                result = await tts_metrics.get_tts_llm_judge_score(
                    audio_paths=[audio_path],
                    reference_texts=["hello"],
                    store=store,
                    row_ids=["row_a"],
                )

            self.assertEqual(mock_tts_judge.await_count, 1)
            self.assertEqual(
                result["per_row"][0]["pronunciation"]["reasoning"], "r2"
            )

    async def test_changed_reference_text_invalidates_row(self):
        from calibrate_agent.tts import metrics as tts_metrics
        from calibrate_agent.judge_store import JudgeStore

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = os.path.join(tmp, "a.wav")
            _write_audio(audio_path, b"same-audio")
            store = JudgeStore.load(tmp)

            mock_tts_judge = AsyncMock(
                return_value={"pronunciation": {"match": True, "reasoning": "r1"}}
            )
            with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
                await tts_metrics.get_tts_llm_judge_score(
                    audio_paths=[audio_path],
                    reference_texts=["hello"],
                    store=store,
                    row_ids=["row_a"],
                )

            mock_tts_judge.reset_mock()
            with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
                await tts_metrics.get_tts_llm_judge_score(
                    audio_paths=[audio_path],
                    reference_texts=["a different sentence"],
                    store=store,
                    row_ids=["row_a"],
                )

            mock_tts_judge.assert_awaited_once()

    async def test_changed_system_prompt_invalidates_row(self):
        from calibrate_agent.tts import metrics as tts_metrics
        from calibrate_agent.judge_store import JudgeStore

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = os.path.join(tmp, "a.wav")
            _write_audio(audio_path, b"same-audio")
            store = JudgeStore.load(tmp)

            evaluator = {
                "name": "pronunciation",
                "system_prompt": "original prompt",
                "judge_model": "openai/gpt-audio",
            }
            mock_tts_judge = AsyncMock(
                return_value={"pronunciation": {"match": True, "reasoning": "r1"}}
            )
            with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
                await tts_metrics.get_tts_llm_judge_score(
                    audio_paths=[audio_path],
                    reference_texts=["hello"],
                    evaluators=[evaluator],
                    store=store,
                    row_ids=["row_a"],
                )

            mock_tts_judge.reset_mock()
            evaluator_edited = dict(evaluator, system_prompt="edited prompt")
            with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
                await tts_metrics.get_tts_llm_judge_score(
                    audio_paths=[audio_path],
                    reference_texts=["hello"],
                    evaluators=[evaluator_edited],
                    store=store,
                    row_ids=["row_a"],
                )

            mock_tts_judge.assert_awaited_once()

    async def test_adding_second_evaluator_only_runs_the_new_one(self):
        from calibrate_agent.tts import metrics as tts_metrics
        from calibrate_agent.judge_store import JudgeStore

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = os.path.join(tmp, "a.wav")
            _write_audio(audio_path, b"same-audio")
            store = JudgeStore.load(tmp)

            ev_first = {
                "name": "pronunciation",
                "system_prompt": "p1",
                "judge_model": "openai/gpt-audio",
            }
            mock_tts_judge = AsyncMock(
                return_value={"pronunciation": {"match": True, "reasoning": "r1"}}
            )
            with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
                await tts_metrics.get_tts_llm_judge_score(
                    audio_paths=[audio_path],
                    reference_texts=["hello"],
                    evaluators=[ev_first],
                    store=store,
                    row_ids=["row_a"],
                )

            ev_second = {
                "name": "clarity",
                "system_prompt": "p2",
                "judge_model": "openai/gpt-audio",
            }
            mock_tts_judge.reset_mock()
            mock_tts_judge.return_value = {"clarity": {"match": False, "reasoning": "r2"}}
            with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
                result = await tts_metrics.get_tts_llm_judge_score(
                    audio_paths=[audio_path],
                    reference_texts=["hello"],
                    evaluators=[ev_first, ev_second],
                    store=store,
                    row_ids=["row_a"],
                )

            # Only the new evaluator ("clarity") triggers a judge call.
            mock_tts_judge.assert_awaited_once_with(
                audio_path,
                "hello",
                evaluators=[ev_second],
                fallback_model=tts_metrics.DEFAULT_TTS_JUDGE_MODEL,
            )
            self.assertEqual(
                result["per_row"][0]["pronunciation"]["reasoning"], "r1"
            )
            self.assertEqual(result["per_row"][0]["clarity"]["reasoning"], "r2")

    async def test_missing_audio_file_does_not_crash_fingerprinting(self):
        from calibrate_agent.tts import metrics as tts_metrics
        from calibrate_agent.judge_store import JudgeStore

        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            mock_tts_judge = AsyncMock(
                side_effect=FileNotFoundError("no such file")
            )
            with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
                with self.assertRaises(FileNotFoundError):
                    await tts_metrics.get_tts_llm_judge_score(
                        audio_paths=["/nonexistent/missing.wav"],
                        reference_texts=["hello"],
                        store=store,
                        row_ids=["row_a"],
                    )
            # Fingerprinting itself didn't raise — the failure is the judge
            # call's own file-not-found error, propagated as-is.
            mock_tts_judge.assert_awaited_once()


class TestAudioContentHash(unittest.TestCase):
    def test_hash_changes_with_content(self):
        from calibrate_agent.tts.metrics import _audio_content_hash

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.wav")
            _write_audio(path, b"one")
            h1 = _audio_content_hash(path)
            _write_audio(path, b"two")
            h2 = _audio_content_hash(path)
            self.assertNotEqual(h1, h2)

    def test_same_content_same_hash(self):
        from calibrate_agent.tts.metrics import _audio_content_hash

        with tempfile.TemporaryDirectory() as tmp:
            path_a = os.path.join(tmp, "a.wav")
            path_b = os.path.join(tmp, "b.wav")
            _write_audio(path_a, b"identical")
            _write_audio(path_b, b"identical")
            self.assertEqual(_audio_content_hash(path_a), _audio_content_hash(path_b))

    def test_missing_file_returns_sentinel_without_raising(self):
        from calibrate_agent.tts.metrics import _audio_content_hash

        result = _audio_content_hash("/definitely/not/a/real/path.wav")
        self.assertIsInstance(result, str)
        self.assertIn("unreadable", result)


if __name__ == "__main__":
    unittest.main()
