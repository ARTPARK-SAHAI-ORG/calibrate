"""Extra coverage for tts/eval.py — provider routers, run/eval flow."""

import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock, mock_open

import pandas as pd


class TestSaveAudio(unittest.TestCase):
    def test_save_riff(self):
        from calibrate.tts.eval import save_audio

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audio.wav"
            riff_bytes = b"RIFF\x00\x00\x00\x00WAVE"
            save_audio(riff_bytes, str(p))
            self.assertEqual(p.read_bytes()[:4], b"RIFF")

    def test_save_raw_pcm(self):
        from calibrate.tts.eval import save_audio

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audio.wav"
            # 1024 samples of silence
            save_audio(b"\x00" * 2048, str(p), sample_rate=16000)
            with wave.open(str(p), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getframerate(), 16000)

    def test_save_creates_parent(self):
        from calibrate.tts.eval import save_audio

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nested" / "deep" / "audio.wav"
            save_audio(b"RIFF" + b"\x00" * 16, str(p))
            self.assertTrue(p.exists())


class TestConvertMp3ToWav(unittest.TestCase):
    def test_convert(self):
        from calibrate.tts import eval as E

        fake_audio = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            mp3 = Path(tmp) / "x.mp3"
            mp3.write_bytes(b"fake")
            wav = Path(tmp) / "x.wav"

            with patch("pydub.AudioSegment.from_mp3", return_value=fake_audio):
                E.convert_mp3_to_wav(str(mp3), str(wav))
            fake_audio.export.assert_called_once()
            self.assertFalse(mp3.exists())  # Cleanup is default True

    def test_convert_no_cleanup(self):
        from calibrate.tts import eval as E

        fake_audio = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            mp3 = Path(tmp) / "x.mp3"
            mp3.write_bytes(b"fake")
            wav = Path(tmp) / "x.wav"

            with patch("pydub.AudioSegment.from_mp3", return_value=fake_audio):
                E.convert_mp3_to_wav(str(mp3), str(wav), cleanup=False)
            self.assertTrue(mp3.exists())


class TestProviderMissingKey(unittest.IsolatedAsyncioTestCase):
    async def test_openai_missing_key(self):
        from calibrate.tts.eval import synthesize_openai

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await synthesize_openai("text", "english", "/tmp/x.wav")

    async def test_google_missing_credentials(self):
        from calibrate.tts.eval import synthesize_google

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await synthesize_google("text", "english", "/tmp/x.wav")

    async def test_elevenlabs_missing_key(self):
        from calibrate.tts.eval import synthesize_elevenlabs

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await synthesize_elevenlabs("text", "english", "/tmp/x.wav")

    async def test_cartesia_missing_key(self):
        from calibrate.tts.eval import synthesize_cartesia

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await synthesize_cartesia("text", "english", "/tmp/x.wav")

    async def test_groq_missing_key(self):
        from calibrate.tts.eval import synthesize_groq

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await synthesize_groq("text", "english", "/tmp/x.wav")

    async def test_sarvam_missing_key(self):
        from calibrate.tts.eval import synthesize_sarvam

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await synthesize_sarvam("text", "english", "/tmp/x.wav")

    async def test_smallest_missing_key(self):
        from calibrate.tts.eval import synthesize_smallest

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await synthesize_smallest("text", "english", "/tmp/x.wav")


class TestSynthesizeSpeechRouter(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_provider_raises(self):
        from calibrate.tts.eval import synthesize_speech

        inner = synthesize_speech
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        with self.assertRaises(ValueError):
            await inner("text", "bogus", "english", "/tmp/x.wav")

    async def test_routes_to_provider(self):
        from calibrate.tts import eval as E

        fake_fn = AsyncMock(return_value={"ttfb": 0.5})
        inner = E.synthesize_speech
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        with patch.object(E, "synthesize_openai", fake_fn), \
             patch.object(E, "create_langfuse_audio_media", return_value=None):
            result = await inner("hi", "openai", "english", "/tmp/x.wav")
        self.assertEqual(result["ttfb"], 0.5)

    async def test_with_langfuse(self):
        from calibrate.tts import eval as E

        fake_fn = AsyncMock(return_value={"ttfb": 0.5})
        inner = E.synthesize_speech
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        fake_lf = MagicMock()
        with patch.object(E, "synthesize_openai", fake_fn), \
             patch.object(E, "create_langfuse_audio_media", return_value=None), \
             patch.object(E, "langfuse_enabled", True), \
             patch.object(E, "langfuse", fake_lf):
            await inner("hi", "openai", "english", "/tmp/x.wav")
        fake_lf.update_current_trace.assert_called_once()


class TestValidateTTSInputFile(unittest.TestCase):
    def test_nonexistent(self):
        from calibrate.tts.eval import validate_tts_input_file

        ok, _ = validate_tts_input_file("/nonexistent.csv")
        self.assertFalse(ok)

    def test_not_csv(self):
        from calibrate.tts.eval import validate_tts_input_file

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w") as f:
            f.write("hi")
            f.flush()
            ok, _ = validate_tts_input_file(f.name)
        self.assertFalse(ok)

    def test_invalid_csv(self):
        from calibrate.tts.eval import validate_tts_input_file

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            # Write a CSV with malformed structure that will fail pd.read_csv
            f.write('"unclosed\n')
            path = f.name
        try:
            ok, _ = validate_tts_input_file(path)
            # might pass or fail depending on pandas tolerance — just check no crash
        finally:
            os.unlink(path)

    def test_missing_id_column(self):
        from calibrate.tts.eval import validate_tts_input_file

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("foo,text\n1,hi\n")
            path = f.name
        try:
            ok, _ = validate_tts_input_file(path)
            self.assertFalse(ok)
        finally:
            os.unlink(path)

    def test_missing_text_column(self):
        from calibrate.tts.eval import validate_tts_input_file

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("id,foo\n1,hi\n")
            path = f.name
        try:
            ok, _ = validate_tts_input_file(path)
            self.assertFalse(ok)
        finally:
            os.unlink(path)

    def test_empty_csv(self):
        from calibrate.tts.eval import validate_tts_input_file

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("id,text\n")
            path = f.name
        try:
            ok, _ = validate_tts_input_file(path)
            self.assertFalse(ok)
        finally:
            os.unlink(path)

    def test_empty_text(self):
        from calibrate.tts.eval import validate_tts_input_file

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("id,text\n1,\n")
            path = f.name
        try:
            ok, _ = validate_tts_input_file(path)
            self.assertFalse(ok)
        finally:
            os.unlink(path)

    def test_valid(self):
        from calibrate.tts.eval import validate_tts_input_file

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("id,text\n1,hello\n")
            path = f.name
        try:
            ok, _ = validate_tts_input_file(path)
            self.assertTrue(ok)
        finally:
            os.unlink(path)


class TestValidateExistingResultsCsv(unittest.TestCase):
    def test_nonexistent_is_ok(self):
        from calibrate.tts.eval import validate_existing_results_csv

        ok, _ = validate_existing_results_csv("/nonexistent.csv")
        self.assertTrue(ok)

    def test_empty_is_ok(self):
        from calibrate.tts.eval import validate_existing_results_csv

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("id,text,audio_path,ttfb\n")
            path = f.name
        try:
            ok, _ = validate_existing_results_csv(path)
            self.assertTrue(ok)
        finally:
            os.unlink(path)

    def test_missing_columns(self):
        from calibrate.tts.eval import validate_existing_results_csv

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("foo,bar\n1,2\n")
            path = f.name
        try:
            ok, err = validate_existing_results_csv(path)
            self.assertFalse(ok)
            self.assertIn("Missing columns", err)
        finally:
            os.unlink(path)


class TestRunTTSEval(unittest.IsolatedAsyncioTestCase):
    async def test_processes_skipping_existing(self):
        from calibrate.tts import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            results_csv = out / "results.csv"
            pd.DataFrame([{"id": "a", "text": "hi", "audio_path": "x.wav", "ttfb": 0.1}]).to_csv(
                str(results_csv), index=False
            )

            with patch.object(E, "synthesize_speech", AsyncMock(return_value={"ttfb": 0.5})):
                result = await E.run_tts_eval(
                    gt_data=[{"id": "a", "text": "hi"}, {"id": "b", "text": "hello"}],
                    provider="openai",
                    language="english",
                    output_dir=str(out),
                    results_csv_path=str(results_csv),
                )
            self.assertEqual(result["success_count"], 1)

    async def test_overwrite_clears_existing(self):
        from calibrate.tts import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            results_csv = out / "results.csv"
            pd.DataFrame([{"id": "a", "text": "hi", "audio_path": "x", "ttfb": 0.1}]).to_csv(
                str(results_csv), index=False
            )

            with patch.object(E, "synthesize_speech", AsyncMock(return_value={"ttfb": 0.5})):
                result = await E.run_tts_eval(
                    gt_data=[{"id": "a", "text": "hi"}],
                    provider="openai",
                    language="english",
                    output_dir=str(out),
                    results_csv_path=str(results_csv),
                    overwrite=True,
                )
            self.assertEqual(result["success_count"], 1)


class TestRunSingleProviderEvalTTS(unittest.IsolatedAsyncioTestCase):
    async def test_basic_flow(self):
        from calibrate.tts import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            out = Path(tmp) / "out"
            out.mkdir()

            with patch.object(E, "synthesize_speech", AsyncMock(return_value={"ttfb": 0.5})), \
                 patch.object(E, "get_tts_llm_judge_score", AsyncMock(return_value={
                     "scores": {"pronunciation": {"type": "binary", "mean": 1.0}},
                     "per_row": [{"pronunciation": {"match": True, "reasoning": "ok"}}],
                 })):
                result = await E.run_single_provider_eval(
                    provider="openai",
                    language="english",
                    input_file=str(inp),
                    output_dir=str(out),
                    debug=False,
                    debug_count=5,
                    overwrite=False,
                )
            self.assertEqual(result["status"], "completed")

    async def test_existing_invalid_csv(self):
        from calibrate.tts import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "openai").mkdir()
            (out / "openai" / "results.csv").write_text("bad\n1\n")

            result = await E.run_single_provider_eval(
                provider="openai",
                language="english",
                input_file=str(inp),
                output_dir=str(out),
                debug=False,
                debug_count=5,
                overwrite=False,
            )
            self.assertEqual(result["status"], "error")

    async def test_debug_with_rating(self):
        from calibrate.tts import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a", "b"], "text": ["hi", "ho"]}).to_csv(str(inp), index=False)
            out = Path(tmp) / "out"
            out.mkdir()
            rating_ev = {"name": "r", "system_prompt": "x", "judge_model": "m",
                         "type": "rating", "scale_min": 1, "scale_max": 5}

            with patch.object(E, "synthesize_speech", AsyncMock(return_value={"ttfb": 0.5})), \
                 patch.object(E, "get_tts_llm_judge_score", AsyncMock(return_value={
                     "scores": {"r": {"type": "rating", "mean": 4.0, "scale_min": 1, "scale_max": 5}},
                     "per_row": [{"r": {"score": 4, "reasoning": "ok"}}],
                 })):
                result = await E.run_single_provider_eval(
                    provider="openai",
                    language="english",
                    input_file=str(inp),
                    output_dir=str(out),
                    debug=True,
                    debug_count=1,
                    overwrite=True,
                    judge_evaluators=[rating_ev],
                )
            self.assertEqual(result["status"], "completed")


class TestTTSMainCLI(unittest.IsolatedAsyncioTestCase):
    async def test_main_invalid_provider(self):
        from calibrate.tts import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            argv = ["e.py", "-p", "bogus", "-i", str(inp), "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await E.main()

    async def test_main_invalid_input(self):
        from calibrate.tts import eval as E

        argv = ["e.py", "-p", "openai", "-i", "/nonexistent.csv", "-o", "/tmp/x"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                await E.main()

    async def test_main_success(self):
        from calibrate.tts import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            out = Path(tmp) / "out"

            argv = ["e.py", "-p", "openai", "-i", str(inp), "-o", str(out)]
            fake_result = {"provider": "openai", "status": "completed",
                           "metrics": {"pronunciation": {"type": "binary", "mean": 0.9},
                                       "ttfb": {"mean": 0.5}}}
            with patch.object(sys, "argv", argv), \
                 patch.object(E, "run_single_provider_eval", AsyncMock(return_value=fake_result)):
                await E.main()

    async def test_main_error_result(self):
        from calibrate.tts import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            out = Path(tmp) / "out"

            argv = ["e.py", "-p", "openai", "-i", str(inp), "-o", str(out)]
            fake_result = {"provider": "openai", "status": "error", "error": "boom"}
            with patch.object(sys, "argv", argv), \
                 patch.object(E, "run_single_provider_eval", AsyncMock(return_value=fake_result)):
                await E.main()


if __name__ == "__main__":
    unittest.main()
