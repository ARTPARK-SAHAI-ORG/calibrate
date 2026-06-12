"""
Tests for calibrate/stt/eval.py — routers, validators, and result writers.

Run with:
    python -m unittest tests.stt.test_eval -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock, MagicMock

import pandas as pd


class TestSTTValidateInputDir(unittest.TestCase):
    def _make_valid_layout(self, base: Path, ids):
        (base / "audios").mkdir()
        pd.DataFrame({"id": ids, "text": [f"text {i}" for i in ids]}).to_csv(
            base / "stt.csv", index=False
        )
        for i in ids:
            (base / "audios" / f"{i}.wav").write_bytes(b"RIFF0000WAVE")

    def test_valid_layout(self):
        from calibrate.stt.eval import validate_stt_input_dir

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_valid_layout(base, ["a", "b"])
            ok, err = validate_stt_input_dir(str(base), "stt.csv")
            self.assertTrue(ok, err)
            self.assertEqual(err, "")

    def test_missing_directory(self):
        from calibrate.stt.eval import validate_stt_input_dir

        ok, err = validate_stt_input_dir("/nonexistent/path/xyz", "stt.csv")
        self.assertFalse(ok)
        self.assertIn("does not exist", err)

    def test_missing_csv(self):
        from calibrate.stt.eval import validate_stt_input_dir

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            ok, err = validate_stt_input_dir(str(base), "stt.csv")
            self.assertFalse(ok)
            self.assertIn("CSV file not found", err)

    def test_missing_audios_dir(self):
        from calibrate.stt.eval import validate_stt_input_dir

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(
                base / "stt.csv", index=False
            )
            ok, err = validate_stt_input_dir(str(base), "stt.csv")
            self.assertFalse(ok)
            self.assertIn("Audios directory not found", err)

    def test_missing_required_columns(self):
        from calibrate.stt.eval import validate_stt_input_dir

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            pd.DataFrame({"foo": ["a"], "bar": ["hi"]}).to_csv(
                base / "stt.csv", index=False
            )
            ok, err = validate_stt_input_dir(str(base), "stt.csv")
            self.assertFalse(ok)
            self.assertIn("missing required column", err)

    def test_missing_audio_files(self):
        from calibrate.stt.eval import validate_stt_input_dir

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            pd.DataFrame({"id": ["a", "b"], "text": ["hi", "yo"]}).to_csv(
                base / "stt.csv", index=False
            )
            (base / "audios" / "a.wav").write_bytes(b"x")
            ok, err = validate_stt_input_dir(str(base), "stt.csv")
            self.assertFalse(ok)
            self.assertIn("Missing audio files", err)
            self.assertIn("b.wav", err)


class TestSTTValidateExistingResultsCSV(unittest.TestCase):
    def test_nonexistent_is_valid(self):
        from calibrate.stt.eval import validate_existing_results_csv

        ok, err = validate_existing_results_csv("/nonexistent.csv")
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_empty_is_valid(self):
        from calibrate.stt.eval import validate_existing_results_csv

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            pd.DataFrame(columns=["id", "gt", "pred"]).to_csv(f.name, index=False)
            path = f.name
        try:
            ok, err = validate_existing_results_csv(path)
            self.assertTrue(ok, err)
        finally:
            os.remove(path)

    def test_valid_columns(self):
        from calibrate.stt.eval import validate_existing_results_csv

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            pd.DataFrame(
                [{"id": "x", "gt": "hi", "pred": "hi"}]
            ).to_csv(f.name, index=False)
            path = f.name
        try:
            ok, err = validate_existing_results_csv(path)
            self.assertTrue(ok, err)
        finally:
            os.remove(path)

    def test_incompatible_structure(self):
        from calibrate.stt.eval import validate_existing_results_csv

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            pd.DataFrame([{"foo": 1, "bar": 2}]).to_csv(f.name, index=False)
            path = f.name
        try:
            ok, err = validate_existing_results_csv(path)
            self.assertFalse(ok)
            self.assertIn("Missing columns", err)
        finally:
            os.remove(path)


class TestSTTValidateEvalOnlyDataset(unittest.TestCase):
    def test_valid(self):
        from calibrate.stt.eval import validate_stt_eval_only_dataset

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                [
                    {"id": "1", "gt": "hi", "pred": "hi"},
                    {"id": "2", "gt": "bye", "pred": "by"},
                ],
                f,
            )
            path = f.name
        try:
            ok, err, rows = validate_stt_eval_only_dataset(path)
            self.assertTrue(ok, err)
            self.assertEqual(len(rows), 2)
        finally:
            os.remove(path)

    def test_missing_file(self):
        from calibrate.stt.eval import validate_stt_eval_only_dataset

        ok, err, rows = validate_stt_eval_only_dataset("/nope.json")
        self.assertFalse(ok)
        self.assertEqual(rows, [])

    def test_not_a_list(self):
        from calibrate.stt.eval import validate_stt_eval_only_dataset

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"id": "1", "gt": "hi", "pred": "hi"}, f)
            path = f.name
        try:
            ok, err, rows = validate_stt_eval_only_dataset(path)
            self.assertFalse(ok)
            self.assertIn("list", err)
        finally:
            os.remove(path)

    def test_missing_fields(self):
        from calibrate.stt.eval import validate_stt_eval_only_dataset

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([{"id": "1", "gt": "hi"}], f)
            path = f.name
        try:
            ok, err, rows = validate_stt_eval_only_dataset(path)
            self.assertFalse(ok)
            self.assertIn("missing required fields", err)
        finally:
            os.remove(path)


class TestTranscribeAudioRouter(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_provider_raises(self):
        from calibrate.stt import eval as stt_eval

        # The router is wrapped in @backoff(max_tries=3), so ValueError
        # would be retried — call ``__wrapped__`` to skip the decorators
        # for unit testing.
        with self.assertRaises(ValueError):
            await stt_eval.transcribe_audio.__wrapped__(
                Path("/tmp/x.wav"),
                "ref",
                "no-such-provider",
                "english",
                "uid",
            )

    async def test_known_provider_routed(self):
        from calibrate.stt import eval as stt_eval

        fake = AsyncMock(return_value={"transcript": "  hello  "})
        with patch.dict(
            "os.environ", {"DEEPGRAM_API_KEY": "x"}
        ), patch.object(stt_eval, "transcribe_deepgram", fake):
            transcript = await stt_eval.transcribe_audio.__wrapped__(
                Path("/tmp/x.wav"),
                "ref",
                "deepgram",
                "english",
                "uid",
            )
        self.assertEqual(transcript, "hello")
        fake.assert_awaited_once()


class TestSTTScoreAndWriteResults(unittest.IsolatedAsyncioTestCase):
    async def test_writes_metrics_and_results(self):
        from calibrate.stt import eval as stt_eval

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None):
            return {
                "scores": {
                    "semantic_match": {"type": "binary", "mean": 1.0}
                },
                "score": 1.0,
                "per_row": [
                    {"semantic_match": {"match": True, "reasoning": "ok"}}
                    for _ in refs
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(
                stt_eval, "get_llm_judge_score", AsyncMock(side_effect=fake_judge)
            ):
                metrics = await stt_eval._score_and_write_results(
                    ids=["1", "2"],
                    gt_transcripts=["hello", "world"],
                    pred_transcripts=["hello", "world"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                )

            self.assertIn("wer", metrics)
            self.assertIn("semantic_match", metrics)
            self.assertTrue((out / "metrics.json").exists())
            self.assertTrue((out / "results.csv").exists())
            df = pd.read_csv(out / "results.csv")
            self.assertTrue(
                set(df.columns)
                >= {
                    "id",
                    "gt",
                    "pred",
                    "wer",
                    "semantic_match",
                    "semantic_match_reasoning",
                }
            )
            self.assertEqual(len(df), 2)

    async def test_rating_evaluator_writes_numeric_score(self):
        from calibrate.stt import eval as stt_eval

        rating = {
            "name": "accuracy",
            "system_prompt": "rate accuracy",
            "judge_model": "openai/gpt-4.1",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
        }

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None):
            return {
                "scores": {
                    "accuracy": {
                        "type": "rating",
                        "mean": 4.0,
                        "scale_min": 1,
                        "scale_max": 5,
                    }
                },
                "score": 4.0,
                "per_row": [
                    {"accuracy": {"score": 4, "reasoning": "ok"}} for _ in refs
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(
                stt_eval, "get_llm_judge_score", AsyncMock(side_effect=fake_judge)
            ):
                await stt_eval._score_and_write_results(
                    ids=["1"],
                    gt_transcripts=["hi"],
                    pred_transcripts=["hi"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    judge_evaluators=[rating],
                )
            df = pd.read_csv(out / "results.csv")
            self.assertEqual(df.iloc[0]["accuracy"], 4)


class TestSTTRunEvalOnly(unittest.IsolatedAsyncioTestCase):
    async def test_runs_evaluator_on_dataset(self):
        from calibrate.stt import eval as stt_eval

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None):
            return {
                "scores": {"semantic_match": {"type": "binary", "mean": 0.5}},
                "score": 0.5,
                "per_row": [
                    {"semantic_match": {"match": True, "reasoning": "ok"}},
                    {"semantic_match": {"match": False, "reasoning": "no"}},
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp) / "ds.json"
            ds_path.write_text(
                json.dumps(
                    [
                        {"id": "1", "gt": "hi", "pred": "hi"},
                        {"id": "2", "gt": "bye", "pred": "by"},
                    ]
                )
            )
            out = Path(tmp) / "out"

            with patch.object(
                stt_eval, "get_llm_judge_score", AsyncMock(side_effect=fake_judge)
            ):
                result = await stt_eval.run_eval_only(
                    dataset_path=str(ds_path),
                    output_dir=str(out),
                )

            self.assertEqual(result["status"], "completed")
            self.assertTrue((out / "metrics.json").exists())
            self.assertTrue((out / "results.csv").exists())

    async def test_invalid_dataset_returns_error(self):
        from calibrate.stt import eval as stt_eval

        result = await stt_eval.run_eval_only(
            dataset_path="/nonexistent.json",
            output_dir=tempfile.mkdtemp(),
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("does not exist", result["error"])


class _FakeSarvamWS:
    """Minimal stand-in for the Sarvam streaming websocket."""

    def __init__(self, messages=None, hang=False):
        self._iter = iter(messages or [])
        self._hang = hang
        self.transcribe = AsyncMock()
        self.flush = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._hang:
            import asyncio

            await asyncio.sleep(3600)
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FakeSarvamConnect:
    def __init__(self, ws):
        self._ws = ws

    async def __aenter__(self):
        return self._ws

    async def __aexit__(self, *exc):
        return False


def _patch_sarvam(stt_eval, ws):
    fake_client = MagicMock()
    fake_client.speech_to_text_streaming.connect = MagicMock(
        return_value=_FakeSarvamConnect(ws)
    )
    return (
        patch.dict("os.environ", {"SARVAM_API_KEY": "sk-fake"}),
        patch.object(stt_eval, "AsyncSarvamAI", return_value=fake_client),
        patch.object(stt_eval, "load_audio", return_value=b"\x00\x00"),
        patch.object(stt_eval, "get_stt_language_code", return_value="hi-IN"),
        patch.object(
            stt_eval.SARVAM_STT_STREAMING_LIMITER, "acquire", AsyncMock()
        ),
    )


class TestTranscribeSarvam(unittest.IsolatedAsyncioTestCase):
    async def test_returns_transcript_on_data_message(self):
        from calibrate.stt import eval as stt_eval

        message = SimpleNamespace(
            type="data",
            data=SimpleNamespace(
                transcript="नमस्ते",
                metrics=SimpleNamespace(processing_latency=0.42),
            ),
        )
        ws = _FakeSarvamWS(messages=[message])
        patches = _patch_sarvam(stt_eval, ws)
        for p in patches:
            p.start()
        try:
            result = await stt_eval.transcribe_sarvam(Path("/tmp/x.wav"), "hindi")
        finally:
            for p in patches:
                p.stop()

        self.assertEqual(result["transcript"], "नमस्ते")
        self.assertEqual(result["ttft"], 0.42)

    async def test_timeout_yields_empty_transcript(self):
        from calibrate.stt import eval as stt_eval

        ws = _FakeSarvamWS(hang=True)
        patches = _patch_sarvam(stt_eval, ws)
        patches = (*patches, patch.object(stt_eval, "SARVAM_STT_RECV_TIMEOUT", 0.01))
        for p in patches:
            p.start()
        try:
            result = await stt_eval.transcribe_sarvam(Path("/tmp/x.wav"), "hindi")
        finally:
            for p in patches:
                p.stop()

        self.assertEqual(result["transcript"], "")
        self.assertIsNone(result["ttft"])

    async def test_error_message_raises(self):
        from calibrate.stt import eval as stt_eval

        message = SimpleNamespace(
            type="error", data=SimpleNamespace(error="boom")
        )
        ws = _FakeSarvamWS(messages=[message])
        patches = _patch_sarvam(stt_eval, ws)
        for p in patches:
            p.start()
        try:
            with self.assertRaises(RuntimeError):
                await stt_eval.transcribe_sarvam(Path("/tmp/x.wav"), "hindi")
        finally:
            for p in patches:
                p.stop()


if __name__ == "__main__":
    unittest.main()
