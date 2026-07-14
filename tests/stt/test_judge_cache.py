"""
Tests for per-row judge checkpointing / resume.

The cache helpers live in ``calibrate_agent/stt/metrics.py``: ``_cache_key``,
``_load_judge_cache``, and ``_gather_cached`` persist each LLM judge result to a
JSONL file so an interrupted run resumes without re-billing completed calls.
``_score_and_write_results`` wires a per-judge cache file under
``output_dir/judge_cache/``.

Run with:
    python -m unittest tests.stt.test_judge_cache -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


class TestCacheKey(unittest.TestCase):
    def test_stable_and_distinct(self):
        from calibrate_agent.stt.metrics import _cache_key

        self.assertEqual(_cache_key("a", "b"), _cache_key("a", "b"))
        self.assertNotEqual(_cache_key("a", "b"), _cache_key("a", "c"))
        # Field boundaries matter: ("ab","c") must not collide with ("a","bc").
        self.assertNotEqual(_cache_key("ab", "c"), _cache_key("a", "bc"))


class TestLoadJudgeCache(unittest.TestCase):
    def test_missing_file(self):
        from calibrate_agent.stt.metrics import _load_judge_cache

        self.assertEqual(_load_judge_cache(None), {})
        self.assertEqual(_load_judge_cache("/nonexistent/x.jsonl"), {})

    def test_reads_and_tolerates_partial_trailing_line(self):
        from calibrate_agent.stt.metrics import _load_judge_cache

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            path.write_text(
                json.dumps({"key": "k1", "value": {"a": 1}}) + "\n"
                + "\n"  # blank line
                + json.dumps({"key": "k2", "value": 2}) + "\n"
                + '{"key": "k3", "value":'  # truncated crash line
            )
            cache = _load_judge_cache(str(path))
            self.assertEqual(cache, {"k1": {"a": 1}, "k2": 2})


class TestGatherCached(unittest.IsolatedAsyncioTestCase):
    async def test_runs_all_without_cache_path(self):
        from calibrate_agent.stt.metrics import _gather_cached

        calls = []

        async def make(i):
            calls.append(i)
            return i * 10

        out = await _gather_cached(None, ["k0", "k1", "k2"], make, desc="x")
        self.assertEqual(out, [0, 10, 20])
        self.assertEqual(sorted(calls), [0, 1, 2])

    async def test_persists_and_resumes(self):
        from calibrate_agent.stt.metrics import _gather_cached

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cache.jsonl")

            first_calls = []

            async def make_first(i):
                first_calls.append(i)
                return {"v": i}

            out1 = await _gather_cached(path, ["k0", "k1"], make_first, desc="x")
            self.assertEqual(out1, [{"v": 0}, {"v": 1}])
            self.assertEqual(sorted(first_calls), [0, 1])
            self.assertTrue(os.path.exists(path))

            # Second run: same keys are cached, so make must not be called.
            async def make_second(i):
                raise AssertionError("should have hit cache")

            out2 = await _gather_cached(path, ["k0", "k1"], make_second, desc="x")
            self.assertEqual(out2, [{"v": 0}, {"v": 1}])

    async def test_partial_resume_only_missing(self):
        from calibrate_agent.stt.metrics import _gather_cached

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cache.jsonl")
            await _gather_cached(path, ["k0"], lambda i: _const({"v": 0}), desc="x")

            calls = []

            async def make(i):
                calls.append(i)
                return {"v": i}

            out = await _gather_cached(path, ["k0", "k1"], make, desc="x")
            self.assertEqual(out, [{"v": 0}, {"v": 1}])
            self.assertEqual(calls, [1])  # only the uncached index ran

    async def test_duplicate_keys_computed_once(self):
        from calibrate_agent.stt.metrics import _gather_cached

        calls = []

        async def make(i):
            calls.append(i)
            return "r"

        out = await _gather_cached(None, ["k", "k", "k"], make, desc="x")
        self.assertEqual(out, ["r", "r", "r"])
        self.assertEqual(len(calls), 1)  # one representative index only

    async def test_failure_preserves_completed_work(self):
        from calibrate_agent.stt.metrics import _gather_cached, _load_judge_cache

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cache.jsonl")

            async def make(i):
                if i == 1:
                    raise RuntimeError("boom")
                return {"v": i}

            with self.assertRaises(RuntimeError):
                await _gather_cached(path, ["k0", "k1", "k2"], make, desc="x")

            # The successful units were still written; the failed one was not.
            cache = _load_judge_cache(path)
            self.assertIn("k0", cache)
            self.assertIn("k2", cache)
            self.assertNotIn("k1", cache)


async def _const(value):
    return value


def _identity_normalizer():
    inst = MagicMock()
    inst.normalize_texts.side_effect = lambda texts, langs, n_jobs=1: list(texts)
    return inst


class TestLlmWerResume(unittest.IsolatedAsyncioTestCase):
    async def test_second_run_skips_judged_segments(self):
        from calibrate_agent.stt import sarvam_llm_wer as slw
        from calibrate_agent.stt import metrics

        async def judge(reference, prediction, model=None):
            return {"index": 0, "equivalent": True, "reasoning": "r"}

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "llm_wer.jsonl")

            first = AsyncMock(side_effect=judge)
            with patch.object(slw, "equivalence_judge", first):
                r1 = await metrics.get_llm_wer_cer_score(
                    ["doctor ne bola"], ["daktar ne bola"], cache_path=path
                )
            self.assertEqual(first.await_count, 1)

            # Re-run with the same cache: the segment judge must not be called.
            second = AsyncMock(side_effect=judge)
            with patch.object(slw, "equivalence_judge", second):
                r2 = await metrics.get_llm_wer_cer_score(
                    ["doctor ne bola"], ["daktar ne bola"], cache_path=path
                )
            second.assert_not_called()
            self.assertEqual(r1["llm_wer"], r2["llm_wer"])


class TestScoreAndWriteWiring(unittest.IsolatedAsyncioTestCase):
    async def test_cache_paths_passed_under_output_dir(self):
        from calibrate_agent.stt import eval as E

        ie_mock = AsyncMock(
            return_value={
                "intent": 1.0,
                "entity": 1.0,
                "per_row": [
                    {
                        "intent_score": 1,
                        "intent_explanation": "ok",
                        "entity_score": 1.0,
                        "entity_explanation": "ok",
                    }
                ],
            }
        )
        wer_mock = AsyncMock(
            return_value={
                "llm_wer": 0.0,
                "llm_cer": 0.0,
                "per_row": [{"llm_wer": 0.0, "llm_cer": 0.0, "segments": []}],
            }
        )
        judge_mock = AsyncMock(
            return_value={
                "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                "per_row": [{"semantic_match": {"match": True, "reasoning": "ok"}}],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(E, "get_wer_score", return_value={"score": 0.0, "per_row": [0.0]}), \
                 patch.object(E, "get_cer_score", return_value={"score": 0.0, "per_row": [0.0]}), \
                 patch.object(E, "get_intent_entity_score", ie_mock), \
                 patch.object(E, "get_llm_wer_cer_score", wer_mock), \
                 patch.object(E, "get_llm_judge_score", judge_mock):
                await E._score_and_write_results(
                    ids=["a"],
                    gt_transcripts=["x"],
                    pred_transcripts=["x"],
                    output_dir=tmp,
                    evaluator_config_dir=tmp,
                    run_sarvam_judges=True,
                )

            expected = os.path.join(tmp, "judge_cache")
            self.assertEqual(
                ie_mock.call_args.kwargs["cache_path"],
                os.path.join(expected, "intent_entity.jsonl"),
            )
            self.assertEqual(
                wer_mock.call_args.kwargs["cache_path"],
                os.path.join(expected, "llm_wer.jsonl"),
            )
            self.assertEqual(
                judge_mock.call_args.kwargs["cache_path"],
                os.path.join(expected, "llm_judge.jsonl"),
            )


if __name__ == "__main__":
    unittest.main()
