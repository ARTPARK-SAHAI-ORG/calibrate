"""Tests for wiring the pipeline STT engine into the benchmark.

Covers the parts that make ``--engine pipeline`` work end-to-end without a real
pipeline or API keys:
- ``get_stt_language`` full 13-language coverage (+ Sarvam regional variants),
- ``get_ttfs_stats`` percentile aggregation,
- ``run_stt_eval(engine="pipeline")`` routing, TTFS column, concurrency & resume,
- ``_score_and_write_results`` threading TTFS into metrics.json + results.csv.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd


STT_LANGUAGES = [
    "english", "hindi", "kannada", "bengali", "malayalam", "marathi",
    "odia", "punjabi", "tamil", "telugu", "gujarati", "sindhi", "maithili",
]


class TestGetSTTLanguageCoverage(unittest.TestCase):
    def test_all_13_languages_map_distinctly(self):
        from calibrate_agent.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        for lang in STT_LANGUAGES:
            base = get_stt_language(lang, "deepgram-nova-3")
            sarvam = get_stt_language(lang, "sarvam-saaras-v3")
            self.assertIsInstance(base, Language)
            self.assertIsInstance(sarvam, Language)

        # Non-English languages must NOT silently fall back to English.
        self.assertNotEqual(
            get_stt_language("tamil", "deepgram-nova-3"),
            get_stt_language("english", "deepgram-nova-3"),
        )
        self.assertEqual(get_stt_language("tamil", "sarvam-saaras-v3"), Language.TA_IN)
        self.assertEqual(get_stt_language("tamil", "deepgram-nova-3"), Language.TA)

    def test_unknown_language_falls_back_to_english(self):
        from calibrate_agent.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language("klingon", "deepgram-nova-3"), Language.EN)
        self.assertEqual(get_stt_language("klingon", "sarvam-saaras-v3"), Language.EN_IN)


class TestGetTTFSStats(unittest.TestCase):
    def test_percentiles_ignore_none(self):
        from calibrate_agent.stt.metrics import get_ttfs_stats

        stats = get_ttfs_stats([0.2, 0.4, None, 0.6, 0.8, None])
        self.assertAlmostEqual(stats["p50"], 0.5)
        self.assertAlmostEqual(stats["mean"], 0.5)
        self.assertGreaterEqual(stats["p99"], stats["p95"])
        self.assertGreaterEqual(stats["p95"], stats["p50"])

    def test_all_none_returns_none(self):
        from calibrate_agent.stt.metrics import get_ttfs_stats

        self.assertIsNone(get_ttfs_stats([None, None]))
        self.assertIsNone(get_ttfs_stats([]))


def _fake_judge():
    async def judge(refs, preds, evaluators=None, fallback_model=None):
        return {
            "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
            "score": 1.0,
            "per_row": [
                {"semantic_match": {"match": True, "reasoning": "ok"}} for _ in refs
            ],
        }

    return AsyncMock(side_effect=judge)


class TestScoreAndWriteTTFS(unittest.IsolatedAsyncioTestCase):
    async def test_ttfs_in_metrics_and_results(self):
        from calibrate_agent.stt import eval as stt_eval

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(stt_eval, "get_llm_judge_score", _fake_judge()):
                metrics = await stt_eval._score_and_write_results(
                    ids=["a", "b", "c"],
                    gt_transcripts=["hi", "there", "world"],
                    pred_transcripts=["hi", "there", "world"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    ttfs_values=[0.3, 0.5, None],
                )

            self.assertIn("ttfs", metrics)
            self.assertEqual(set(metrics["ttfs"]), {"p50", "p95", "p99", "mean"})
            df = pd.read_csv(out / "results.csv")
            self.assertIn("ttfs", df.columns)

    async def test_shorter_ttfs_list_does_not_drop_rows(self):
        """A ttfs list shorter than ids is padded, not silently truncated via zip."""
        from calibrate_agent.stt import eval as stt_eval

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(stt_eval, "get_llm_judge_score", _fake_judge()):
                metrics = await stt_eval._score_and_write_results(
                    ids=["a", "b", "c"],
                    gt_transcripts=["hi", "there", "world"],
                    pred_transcripts=["hi", "there", "world"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    ttfs_values=[0.3],  # shorter than the 3 ids
                )

            self.assertIn("ttfs", metrics)
            df = pd.read_csv(out / "results.csv")
            self.assertEqual(len(df), 3)  # all rows written, none dropped

    async def test_direct_engine_omits_ttfs(self):
        """All-None TTFS (direct engine) leaves ttfs out entirely."""
        from calibrate_agent.stt import eval as stt_eval

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(stt_eval, "get_llm_judge_score", _fake_judge()):
                metrics = await stt_eval._score_and_write_results(
                    ids=["a", "b"],
                    gt_transcripts=["hi", "there"],
                    pred_transcripts=["hi", "there"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    ttfs_values=[None, None],
                )

            self.assertNotIn("ttfs", metrics)
            df = pd.read_csv(out / "results.csv")
            self.assertNotIn("ttfs", df.columns)


class TestTranscribeAudioPipeline(unittest.IsolatedAsyncioTestCase):
    async def _run(self, provider):
        from calibrate_agent.stt import eval as stt_eval

        pipeline_output = {
            "transcript": " hi ",
            "ttfs": 0.4,
        }
        limiter = AsyncMock()
        with patch.object(stt_eval, "load_audio", return_value=b"\x00\x00"), \
             patch.object(stt_eval, "create_stt_service") as factory, \
             patch.object(stt_eval, "transcribe_via_pipeline",
                          AsyncMock(return_value=pipeline_output)), \
             patch.object(stt_eval.SARVAM_STT_STREAMING_LIMITER, "acquire", limiter):
            result = await stt_eval.transcribe_audio_pipeline.__wrapped__(
                Path("a.wav"), "hi", provider, "hindi", "uid"
            )
        return result, factory, limiter

    async def test_returns_stripped_transcript_and_ttfs(self):
        result, factory, _ = await self._run("deepgram-flux")
        factory.assert_called_once_with("deepgram-flux", "hindi")
        self.assertEqual(result, {"transcript": "hi", "ttfs": 0.4})

    async def test_sarvam_realtime_models_share_the_sarvam_rate_limiter(self):
        for provider in ("sarvam-saaras-v4-realtime", "sarvam-saaras-v3-realtime", "sarvam-saaras-v4"):
            _, _, limiter = await self._run(provider)
            limiter.assert_awaited_once()

    async def test_other_providers_skip_the_sarvam_rate_limiter(self):
        _, _, limiter = await self._run("deepgram-nova-3")
        limiter.assert_not_awaited()


class TestRunSTTEvalDirectRouting(unittest.IsolatedAsyncioTestCase):
    async def test_direct_engine_concurrent_writes_no_ttfs(self):
        """Direct engine routes through the shared concurrent runner: writes all
        rows (respecting max_concurrency) and emits no ttfs column."""
        from calibrate_agent.stt import eval as stt_eval

        async def fake_direct(audio_path, reference, provider, language, uid):
            stem = Path(audio_path).stem
            return {"transcript": f"pred_{stem}"}  # no ttfs key for direct

        gt_data = [{"id": i, "gt": str(i)} for i in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "results.csv"
            with patch.object(
                stt_eval, "transcribe_audio", AsyncMock(side_effect=fake_direct)
            ):
                n = await stt_eval.run_stt_eval(
                    gt_data=gt_data,
                    audio_dir=Path(tmp),
                    provider="deepgram-nova-3",
                    language="english",
                    results_csv_path=csv_path,
                    engine="direct",
                    max_concurrency=3,
                )
            self.assertEqual(n, 3)
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 3)
            self.assertNotIn("ttfs", df.columns)


class TestRunSTTEvalPipelineRouting(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_engine_writes_transcript_and_ttfs(self):
        from calibrate_agent.stt import eval as stt_eval

        async def fake_transcribe(audio_path, reference, provider, language, uid):
            # id is the wav stem; echo a per-id transcript + latency.
            stem = Path(audio_path).stem
            return {
                "transcript": f"pred_{stem}",
                "ttfs": 0.1 * int(stem),
            }

        gt_data = [{"id": 1, "gt": "one"}, {"id": 2, "gt": "two"}]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "results.csv"
            with patch.object(
                stt_eval,
                "transcribe_audio_pipeline",
                AsyncMock(side_effect=fake_transcribe),
            ):
                n = await stt_eval.run_stt_eval(
                    gt_data=gt_data,
                    audio_dir=Path(tmp),
                    provider="deepgram-nova-3",
                    language="tamil",
                    results_csv_path=csv_path,
                    engine="pipeline",
                    max_concurrency=2,
                )

            self.assertEqual(n, 2)
            df = pd.read_csv(csv_path).sort_values("id").reset_index(drop=True)
            self.assertEqual(list(df["pred"]), ["pred_1", "pred_2"])
            self.assertIn("ttfs", df.columns)
            self.assertEqual(len(df), 2)

    async def test_pipeline_engine_resumes_and_skips_processed(self):
        from calibrate_agent.stt import eval as stt_eval

        gt_data = [{"id": 1, "gt": "one"}, {"id": 2, "gt": "two"}]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "results.csv"
            # Pre-seed id=1 as already processed.
            pd.DataFrame(
                [{"id": 1, "gt": "one", "pred": "pred_1", "ttfs": 0.1}]
            ).to_csv(csv_path, index=False)

            calls = []

            async def fake_transcribe(audio_path, reference, provider, language, uid):
                stem = Path(audio_path).stem
                calls.append(stem)
                return {"transcript": f"pred_{stem}", "ttfs": 0.2}

            with patch.object(
                stt_eval,
                "transcribe_audio_pipeline",
                AsyncMock(side_effect=fake_transcribe),
            ):
                await stt_eval.run_stt_eval(
                    gt_data=gt_data,
                    audio_dir=Path(tmp),
                    provider="deepgram-nova-3",
                    language="hindi",
                    results_csv_path=csv_path,
                    engine="pipeline",
                )

            # Only id=2 should have been transcribed (id=1 skipped).
            self.assertEqual(calls, ["2"])
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 2)

    async def test_pipeline_engine_failure_left_for_retry(self):
        """A failing clip is not written (raises no gather-wide cancel)."""
        from calibrate_agent.stt import eval as stt_eval

        gt_data = [{"id": 1, "gt": "one"}, {"id": 2, "gt": "two"}]

        async def fake_transcribe(audio_path, reference, provider, language, uid):
            stem = Path(audio_path).stem
            if stem == "1":
                raise RuntimeError("boom")
            return {"transcript": f"pred_{stem}", "ttfs": 0.2}

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "results.csv"
            with patch.object(
                stt_eval,
                "transcribe_audio_pipeline",
                AsyncMock(side_effect=fake_transcribe),
            ):
                n = await stt_eval.run_stt_eval(
                    gt_data=gt_data,
                    audio_dir=Path(tmp),
                    provider="deepgram-nova-3",
                    language="english",
                    results_csv_path=csv_path,
                    engine="pipeline",
                )

            self.assertEqual(n, 1)  # only id=2 succeeded
            df = pd.read_csv(csv_path)
            self.assertEqual(list(df["id"]), [2])  # id=1 left unwritten


if __name__ == "__main__":
    unittest.main()
