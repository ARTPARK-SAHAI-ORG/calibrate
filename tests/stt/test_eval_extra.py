"""Extra coverage for stt/eval.py — provider routers and main pathway."""

import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pandas as pd


# --- load_audio -----------------------------------------------------------

class TestLoadAudio(unittest.TestCase):
    def test_load_audio_bytes(self):
        from calibrate.stt import eval as E

        fake_segment = MagicMock()
        fake_segment.set_channels.return_value = fake_segment
        fake_segment.set_frame_rate.return_value = fake_segment
        fake_segment.set_sample_width.return_value = fake_segment
        fake_segment.normalize.return_value = fake_segment
        fake_segment.strip_silence.return_value = fake_segment

        def fake_export(out_io, format):
            out_io.write(b"WAVDATA")

        fake_segment.export = fake_export

        with patch("pydub.AudioSegment.from_file", return_value=fake_segment):
            result = E.load_audio(Path("/tmp/dummy.wav"))
        self.assertEqual(result, b"WAVDATA")

    def test_load_audio_raw_pcm(self):
        from calibrate.stt import eval as E

        fake_segment = MagicMock()
        fake_segment.set_channels.return_value = fake_segment
        fake_segment.set_frame_rate.return_value = fake_segment
        fake_segment.set_sample_width.return_value = fake_segment
        fake_segment.normalize.return_value = fake_segment
        fake_segment.strip_silence.return_value = fake_segment
        fake_segment.raw_data = b"PCMDATA"

        with patch("pydub.AudioSegment.from_file", return_value=fake_segment):
            result = E.load_audio(Path("/tmp/x.wav"), raw_pcm=True)
        self.assertEqual(result, b"PCMDATA")

    def test_load_audio_as_file(self):
        from calibrate.stt import eval as E

        fake_segment = MagicMock()
        fake_segment.set_channels.return_value = fake_segment
        fake_segment.set_frame_rate.return_value = fake_segment
        fake_segment.set_sample_width.return_value = fake_segment
        fake_segment.normalize.return_value = fake_segment
        fake_segment.strip_silence.return_value = fake_segment

        def fake_export(out_io, format):
            out_io.write(b"WAVDATA")

        fake_segment.export = fake_export

        with patch("pydub.AudioSegment.from_file", return_value=fake_segment):
            result = E.load_audio(Path("/tmp/x.wav"), as_file=True)
        self.assertTrue(hasattr(result, "read"))


# --- Provider transcribe_* missing-key paths ------------------------------

class TestProviderAPIKeyMissing(unittest.IsolatedAsyncioTestCase):
    async def test_deepgram_missing_key(self):
        from calibrate.stt.eval import transcribe_deepgram

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_deepgram(Path("/tmp/x.wav"), "english")

    async def test_openai_missing_key(self):
        from calibrate.stt.eval import transcribe_openai

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_openai(Path("/tmp/x.wav"), "english")

    async def test_groq_missing_key(self):
        from calibrate.stt.eval import transcribe_groq

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_groq(Path("/tmp/x.wav"), "english")

    async def test_google_missing_credentials(self):
        from calibrate.stt.eval import transcribe_google

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_google(Path("/tmp/x.wav"), "english")

    async def test_sarvam_missing_key(self):
        from calibrate.stt.eval import transcribe_sarvam

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_sarvam(Path("/tmp/x.wav"), "english")

    async def test_elevenlabs_missing_key(self):
        from calibrate.stt.eval import transcribe_elevenlabs

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_elevenlabs(Path("/tmp/x.wav"), "english")

    async def test_cartesia_missing_key(self):
        from calibrate.stt.eval import transcribe_cartesia

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_cartesia(Path("/tmp/x.wav"), "english")

    async def test_smallest_missing_key(self):
        from calibrate.stt.eval import transcribe_smallest

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_smallest(Path("/tmp/x.wav"), "english")

    async def test_smallest_streaming_missing_key(self):
        from calibrate.stt.eval import transcribe_smallest_streaming

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_smallest_streaming(Path("/tmp/x.wav"), "english")


# --- transcribe_audio router ----------------------------------------------

class TestTranscribeAudioRouter(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_provider_raises(self):
        from calibrate.stt.eval import transcribe_audio

        # Use __wrapped__ to skip backoff retries
        inner = transcribe_audio
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        with self.assertRaises(ValueError):
            await inner(Path("/tmp/x.wav"), "ref", "bogus", "english", "u")

    async def test_routes_to_provider(self):
        from calibrate.stt import eval as E

        fake_fn = AsyncMock(return_value={"transcript": "hello world"})
        inner = E.transcribe_audio
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        with patch.object(E, "transcribe_deepgram", fake_fn):
            result = await inner(Path("/tmp/x.wav"), "ref", "deepgram", "english", "u")
        self.assertEqual(result, "hello world")
        fake_fn.assert_called_once()

    async def test_with_langfuse(self):
        from calibrate.stt import eval as E

        fake_fn = AsyncMock(return_value={"transcript": "x"})
        inner = E.transcribe_audio
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        fake_lf = MagicMock()
        with patch.object(E, "transcribe_deepgram", fake_fn), \
             patch.object(E, "langfuse_enabled", True), \
             patch.object(E, "langfuse", fake_lf), \
             patch.object(E, "create_langfuse_audio_media", return_value=None):
            await inner(Path("/tmp/x.wav"), "ref", "deepgram", "english", "u")
        fake_lf.update_current_trace.assert_called_once()


# --- validate_existing_results_csv ----------------------------------------

class TestValidateExistingResultsCsv(unittest.TestCase):
    def test_nonexistent_returns_ok(self):
        from calibrate.stt.eval import validate_existing_results_csv

        ok, _ = validate_existing_results_csv("/nonexistent/path.csv")
        self.assertTrue(ok)

    def test_empty_is_valid(self):
        from calibrate.stt.eval import validate_existing_results_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "results.csv"
            pd.DataFrame(columns=["id", "gt", "pred"]).to_csv(p, index=False)
            ok, _ = validate_existing_results_csv(str(p))
            self.assertTrue(ok)

    def test_invalid_columns(self):
        from calibrate.stt.eval import validate_existing_results_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "results.csv"
            pd.DataFrame({"foo": [1]}).to_csv(p, index=False)
            ok, err = validate_existing_results_csv(str(p))
            self.assertFalse(ok)
            self.assertIn("Missing columns", err)

    def test_valid_columns(self):
        from calibrate.stt.eval import validate_existing_results_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "results.csv"
            pd.DataFrame({"id": [1], "gt": ["a"], "pred": ["a"]}).to_csv(p, index=False)
            ok, _ = validate_existing_results_csv(str(p))
            self.assertTrue(ok)


# --- validate_stt_eval_only_dataset --------------------------------------

class TestValidateSTTEvalOnlyDataset(unittest.TestCase):
    def test_nonexistent(self):
        from calibrate.stt.eval import validate_stt_eval_only_dataset

        ok, err, _ = validate_stt_eval_only_dataset("/nonexistent.json")
        self.assertFalse(ok)

    def test_invalid_json(self):
        from calibrate.stt.eval import validate_stt_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            p.write_text("{bad")
            ok, err, _ = validate_stt_eval_only_dataset(str(p))
            self.assertFalse(ok)

    def test_not_a_list(self):
        from calibrate.stt.eval import validate_stt_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            p.write_text(json.dumps({"x": 1}))
            ok, _, _ = validate_stt_eval_only_dataset(str(p))
            self.assertFalse(ok)

    def test_row_not_object(self):
        from calibrate.stt.eval import validate_stt_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            p.write_text(json.dumps(["x"]))
            ok, _, _ = validate_stt_eval_only_dataset(str(p))
            self.assertFalse(ok)

    def test_missing_fields(self):
        from calibrate.stt.eval import validate_stt_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            p.write_text(json.dumps([{"id": "a"}]))
            ok, _, _ = validate_stt_eval_only_dataset(str(p))
            self.assertFalse(ok)

    def test_valid(self):
        from calibrate.stt.eval import validate_stt_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            p.write_text(json.dumps([{"id": "a", "gt": "x", "pred": "x"}]))
            ok, err, rows = validate_stt_eval_only_dataset(str(p))
            self.assertTrue(ok)
            self.assertEqual(len(rows), 1)


# --- _score_and_write_results --------------------------------------------

class TestScoreAndWrite(unittest.IsolatedAsyncioTestCase):
    async def test_writes_files(self):
        from calibrate.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(E, "get_wer_score", return_value={"score": 0.1, "per_row": [0.1, 0.1]}), \
                 patch.object(E, "get_cer_score", return_value={"score": 0.2, "per_row": [0.2, 0.2]}), \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                     "per_row": [
                         {"semantic_match": {"match": True, "reasoning": "ok"}},
                         {"semantic_match": {"match": True, "reasoning": "ok"}},
                     ],
                 })):
                result = await E._score_and_write_results(
                    ids=["a", "b"],
                    gt_transcripts=["x", "y"],
                    pred_transcripts=["x", "y"],
                    output_dir=tmp,
                    evaluator_config_dir=tmp,
                )
            self.assertEqual(result["wer"], 0.1)
            self.assertEqual(result["cer"], 0.2)
            self.assertIn("semantic_match", result)
            self.assertTrue((Path(tmp) / "metrics.json").exists())

            import pandas as _pd
            df = _pd.read_csv(Path(tmp) / "results.csv")
            self.assertIn("cer", df.columns)
            self.assertEqual(list(df["cer"]), [0.2, 0.2])

    async def test_rating_evaluator(self):
        from calibrate.stt import eval as E

        rating_ev = {"name": "r", "system_prompt": "x", "judge_model": "m",
                     "type": "rating", "scale_min": 1, "scale_max": 5}

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(E, "get_wer_score", return_value={"score": 0.05, "per_row": [0.05]}), \
             patch.object(E, "get_cer_score", return_value={"score": 0.03, "per_row": [0.03]}), \
             patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                 "scores": {"r": {"type": "rating", "mean": 4.0, "scale_min": 1, "scale_max": 5}},
                 "per_row": [{"r": {"score": 4, "reasoning": "ok"}}],
             })):
            await E._score_and_write_results(
                ids=["a"],
                gt_transcripts=["x"],
                pred_transcripts=["x"],
                output_dir=tmp,
                evaluator_config_dir=tmp,
                judge_evaluators=[rating_ev],
            )


# --- run_eval_only --------------------------------------------------------

class TestRunEvalOnly(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_dataset(self):
        from calibrate.stt.eval import run_eval_only

        with tempfile.TemporaryDirectory() as tmp:
            result = await run_eval_only("/nonexistent.json", tmp)
        self.assertEqual(result["status"], "error")

    async def test_success(self):
        from calibrate.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "data.json"
            ds.write_text(json.dumps([
                {"id": "a", "gt": "x", "pred": "x"},
                {"id": "b", "gt": "y", "pred": None},
            ]))
            out = Path(tmp) / "out"
            with patch.object(E, "get_wer_score", return_value={"score": 0.1, "per_row": [0.1, 0.1]}), \
                 patch.object(E, "get_cer_score", return_value={"score": 0.2, "per_row": [0.2, 0.2]}), \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                     "per_row": [
                         {"semantic_match": {"match": True, "reasoning": "ok"}},
                         {"semantic_match": {"match": True, "reasoning": "ok"}},
                     ],
                 })):
                result = await E.run_eval_only(str(ds), str(out))
        self.assertEqual(result["status"], "completed")


# --- run_stt_eval ---------------------------------------------------------

class TestRunStteval(unittest.IsolatedAsyncioTestCase):
    async def test_processes_new_and_skips_existing(self):
        from calibrate.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "stuff"
            base.mkdir()
            audio_dir = base / "audios"
            audio_dir.mkdir()
            (audio_dir / "a.wav").write_bytes(b"\x00")
            (audio_dir / "b.wav").write_bytes(b"\x00")

            results_csv = base / "results.csv"
            pd.DataFrame([{"id": "a", "gt": "X", "pred": "x"}]).to_csv(
                str(results_csv), index=False
            )

            with patch.object(E, "transcribe_audio", AsyncMock(return_value="hello b")):
                count = await E.run_stt_eval(
                    gt_data=[{"id": "a", "gt": "X"}, {"id": "b", "gt": "Y"}],
                    audio_dir=audio_dir,
                    provider="deepgram",
                    language="english",
                    results_csv_path=str(results_csv),
                )

            self.assertEqual(count, 1)
            df = pd.read_csv(str(results_csv))
            self.assertEqual(len(df), 2)


# --- run_single_provider_eval --------------------------------------------

class TestRunSingleProviderEval(unittest.IsolatedAsyncioTestCase):
    async def test_overwrite_path(self):
        from calibrate.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            (base / "audios" / "a.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a"], "text": ["hello"]}).to_csv(base / "stt.csv", index=False)

            output = Path(tmp) / "out"
            output.mkdir()
            (output / "deepgram").mkdir()
            # Pre-existing results.csv to trigger overwrite path
            (output / "deepgram" / "results.csv").write_text("id,gt,pred\na,hello,hi\n")

            with patch.object(E, "transcribe_audio", AsyncMock(return_value="hello")), \
                 patch.object(E, "get_wer_score", return_value={"score": 0.0, "per_row": [0.0]}), \
                 patch.object(E, "get_cer_score", return_value={"score": 0.0, "per_row": [0.0]}), \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                     "per_row": [{"semantic_match": {"match": True, "reasoning": "ok"}}],
                 })):
                result = await E.run_single_provider_eval(
                    provider="deepgram",
                    language="english",
                    input_dir=str(base),
                    input_file_name="stt.csv",
                    output_dir=str(output),
                    debug=False,
                    debug_count=5,
                    ignore_retry=False,
                    overwrite=True,
                )
            self.assertEqual(result["status"], "completed")

    async def test_existing_invalid_csv_error(self):
        from calibrate.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            (base / "audios" / "a.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a"], "text": ["hello"]}).to_csv(base / "stt.csv", index=False)

            output = Path(tmp) / "out"
            output.mkdir()
            (output / "deepgram").mkdir()
            (output / "deepgram" / "results.csv").write_text("bad,csv\n1,2\n")

            result = await E.run_single_provider_eval(
                provider="deepgram",
                language="english",
                input_dir=str(base),
                input_file_name="stt.csv",
                output_dir=str(output),
                debug=False,
                debug_count=5,
                ignore_retry=False,
                overwrite=False,
            )
            self.assertEqual(result["status"], "error")

    async def test_debug_mode_and_ignore_retry(self):
        from calibrate.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            for i in ["a", "b"]:
                (base / "audios" / f"{i}.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a", "b"], "text": ["hello", "world"]}).to_csv(
                base / "stt.csv", index=False
            )
            output = Path(tmp) / "out"
            output.mkdir()

            with patch.object(E, "transcribe_audio", AsyncMock(return_value="hello")), \
                 patch.object(E, "get_wer_score", return_value={"score": 0.0, "per_row": [0.0]}), \
                 patch.object(E, "get_cer_score", return_value={"score": 0.0, "per_row": [0.0]}), \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                     "per_row": [{"semantic_match": {"match": True, "reasoning": "ok"}}],
                 })):
                result = await E.run_single_provider_eval(
                    provider="deepgram",
                    language="english",
                    input_dir=str(base),
                    input_file_name="stt.csv",
                    output_dir=str(output),
                    debug=True,
                    debug_count=1,
                    ignore_retry=True,
                    overwrite=False,
                )
            self.assertEqual(result["status"], "completed")


# --- main CLI -------------------------------------------------------------

class TestSTTMain(unittest.IsolatedAsyncioTestCase):
    async def test_main_invalid_provider(self):
        from calibrate.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["e.py", "-p", "bogus", "-i", tmp, "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await E.main()

    async def test_main_invalid_input_dir(self):
        from calibrate.stt import eval as E

        argv = ["e.py", "-p", "deepgram", "-i", "/nonexistent", "-o", "/tmp/x"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                await E.main()

    async def test_main_success(self):
        from calibrate.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            (base / "audios" / "a.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(base / "stt.csv", index=False)
            output = Path(tmp) / "out"

            argv = ["e.py", "-p", "deepgram", "-i", str(base), "-o", str(output)]
            fake_result = {"provider": "deepgram", "status": "completed",
                           "metrics": {"wer": 0.1, "semantic_match": {"type": "binary", "mean": 0.9}}}
            with patch.object(sys, "argv", argv), \
                 patch.object(E, "run_single_provider_eval", AsyncMock(return_value=fake_result)):
                await E.main()

    async def test_main_error_status(self):
        from calibrate.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            (base / "audios" / "a.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(base / "stt.csv", index=False)
            output = Path(tmp) / "out"

            argv = ["e.py", "-p", "deepgram", "-i", str(base), "-o", str(output)]
            fake_result = {"provider": "deepgram", "status": "error", "error": "fail"}
            with patch.object(sys, "argv", argv), \
                 patch.object(E, "run_single_provider_eval", AsyncMock(return_value=fake_result)):
                with self.assertRaises(SystemExit):
                    await E.main()


if __name__ == "__main__":
    unittest.main()
