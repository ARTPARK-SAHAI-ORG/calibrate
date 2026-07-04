"""
Tests for calibrate_agent/tts/eval.py — routers, validators, save_audio, run_tts_eval.

Run with:
    python -m unittest tests.tts.test_eval -v
"""

import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pandas as pd


class TestSaveAudio(unittest.TestCase):
    def test_wav_passthrough(self):
        from calibrate_agent.tts.eval import save_audio

        import io

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 100)
        wav_bytes = buf.getvalue()
        self.assertEqual(wav_bytes[:4], b"RIFF")

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "a.wav")
            save_audio(wav_bytes, out)
            self.assertEqual(Path(out).read_bytes(), wav_bytes)

    def test_raw_pcm_wrapped_in_wav(self):
        from calibrate_agent.tts.eval import save_audio

        pcm = b"\x00\x01" * 200
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "a.wav")
            save_audio(pcm, out, sample_rate=24000)
            with wave.open(out, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), 24000)
                self.assertEqual(wf.readframes(wf.getnframes()), pcm)


class TestTTSValidateInputFile(unittest.TestCase):
    def test_valid(self):
        from calibrate_agent.tts.eval import validate_tts_input_file

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            pd.DataFrame({"id": ["1"], "text": ["hello"]}).to_csv(
                f.name, index=False
            )
            path = f.name
        try:
            ok, err = validate_tts_input_file(path)
            self.assertTrue(ok, err)
        finally:
            os.remove(path)

    def test_missing_file(self):
        from calibrate_agent.tts.eval import validate_tts_input_file

        ok, err = validate_tts_input_file("/nope.csv")
        self.assertFalse(ok)
        self.assertIn("does not exist", err)

    def test_not_csv_extension(self):
        from calibrate_agent.tts.eval import validate_tts_input_file

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("hi")
            path = f.name
        try:
            ok, err = validate_tts_input_file(path)
            self.assertFalse(ok)
            self.assertIn("CSV file", err)
        finally:
            os.remove(path)

    def test_missing_columns(self):
        from calibrate_agent.tts.eval import validate_tts_input_file

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            pd.DataFrame({"foo": ["1"], "bar": ["hi"]}).to_csv(f.name, index=False)
            path = f.name
        try:
            ok, err = validate_tts_input_file(path)
            self.assertFalse(ok)
            self.assertIn("missing required column", err)
        finally:
            os.remove(path)

    def test_empty(self):
        from calibrate_agent.tts.eval import validate_tts_input_file

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            pd.DataFrame(columns=["id", "text"]).to_csv(f.name, index=False)
            path = f.name
        try:
            ok, err = validate_tts_input_file(path)
            self.assertFalse(ok)
            self.assertIn("empty", err)
        finally:
            os.remove(path)

    def test_empty_text_value(self):
        from calibrate_agent.tts.eval import validate_tts_input_file

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            pd.DataFrame({"id": ["1", "2"], "text": ["hello", ""]}).to_csv(
                f.name, index=False
            )
            path = f.name
        try:
            ok, err = validate_tts_input_file(path)
            self.assertFalse(ok)
            self.assertIn("empty text", err)
        finally:
            os.remove(path)


class TestTTSValidateExistingResultsCSV(unittest.TestCase):
    def test_nonexistent_is_valid(self):
        from calibrate_agent.tts.eval import validate_existing_results_csv

        ok, err = validate_existing_results_csv("/nonexistent.csv")
        self.assertTrue(ok)

    def test_valid_columns(self):
        from calibrate_agent.tts.eval import validate_existing_results_csv

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            pd.DataFrame(
                [{"id": "1", "text": "hi", "audio_path": "/tmp/a.wav", "ttfb": 0.3}]
            ).to_csv(f.name, index=False)
            path = f.name
        try:
            ok, err = validate_existing_results_csv(path)
            self.assertTrue(ok, err)
        finally:
            os.remove(path)

    def test_incompatible(self):
        from calibrate_agent.tts.eval import validate_existing_results_csv

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            pd.DataFrame([{"foo": 1, "bar": 2}]).to_csv(f.name, index=False)
            path = f.name
        try:
            ok, err = validate_existing_results_csv(path)
            self.assertFalse(ok)
            self.assertIn("Missing columns", err)
        finally:
            os.remove(path)


class TestSynthesizeSpeechRouter(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_provider_raises(self):
        from calibrate_agent.tts import eval as tts_eval

        with self.assertRaises(ValueError):
            await tts_eval.synthesize_speech.__wrapped__(
                "hello", "no-such-provider", "english", "/tmp/x.wav"
            )

    async def test_known_provider_routed(self):
        from calibrate_agent.tts import eval as tts_eval

        fake = AsyncMock(return_value={"ttfb": 0.42})

        with patch.object(tts_eval, "synthesize_openai", fake), patch.object(
            tts_eval, "create_langfuse_audio_media", lambda p: None
        ):
            result = await tts_eval.synthesize_speech.__wrapped__(
                "hello", "openai", "english", "/tmp/x.wav"
            )
        self.assertEqual(result, {"ttfb": 0.42})
        fake.assert_awaited_once_with("hello", "english", "/tmp/x.wav")


class TestRunTTSEval(unittest.IsolatedAsyncioTestCase):
    async def test_synthesizes_and_writes_csv(self):
        from calibrate_agent.tts import eval as tts_eval

        async def fake_synth(text, provider, language, audio_path):
            Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
            Path(audio_path).write_bytes(b"RIFF" + b"\x00" * 40)
            return {"ttfb": 0.1}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_csv = out / "results.csv"
            with patch.object(
                tts_eval, "synthesize_speech", AsyncMock(side_effect=fake_synth)
            ):
                result = await tts_eval.run_tts_eval(
                    gt_data=[
                        {"id": "1", "text": "hello"},
                        {"id": "2", "text": "world"},
                    ],
                    provider="openai",
                    language="english",
                    output_dir=str(out),
                    results_csv_path=results_csv,
                )

            self.assertEqual(result["success_count"], 2)
            self.assertEqual(len(result["ttfb_values"]), 2)
            df = pd.read_csv(results_csv)
            self.assertEqual(len(df), 2)
            self.assertEqual(set(df.columns), {"id", "text", "audio_path", "ttfb"})
            for p in df["audio_path"]:
                self.assertTrue(Path(p).exists())

    async def test_resume_skips_processed_ids(self):
        from calibrate_agent.tts import eval as tts_eval

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_csv = out / "results.csv"
            # String ids avoid pandas int-coercion mismatch between the CSV
            # and the gt_data dicts when the resume logic compares them.
            pd.DataFrame(
                [{"id": "row_a", "text": "hello", "audio_path": "/x.wav", "ttfb": 0.1}]
            ).to_csv(results_csv, index=False)

            call_count = {"n": 0}

            async def fake_synth(text, provider, language, audio_path):
                call_count["n"] += 1
                Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
                Path(audio_path).write_bytes(b"RIFF")
                return {"ttfb": 0.2}

            with patch.object(
                tts_eval, "synthesize_speech", AsyncMock(side_effect=fake_synth)
            ):
                await tts_eval.run_tts_eval(
                    gt_data=[
                        {"id": "row_a", "text": "hello"},
                        {"id": "row_b", "text": "world"},
                    ],
                    provider="openai",
                    language="english",
                    output_dir=str(out),
                    results_csv_path=results_csv,
                )

            self.assertEqual(call_count["n"], 1)
            df = pd.read_csv(results_csv)
            self.assertEqual(set(df["id"].astype(str)), {"row_a", "row_b"})

    async def test_overwrite_deletes_existing(self):
        from calibrate_agent.tts import eval as tts_eval

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_csv = out / "results.csv"
            pd.DataFrame(
                [{"id": "1", "text": "old", "audio_path": "/x.wav", "ttfb": 0.1}]
            ).to_csv(results_csv, index=False)

            async def fake_synth(text, provider, language, audio_path):
                Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
                Path(audio_path).write_bytes(b"RIFF")
                return {"ttfb": 0.5}

            with patch.object(
                tts_eval, "synthesize_speech", AsyncMock(side_effect=fake_synth)
            ):
                await tts_eval.run_tts_eval(
                    gt_data=[{"id": "1", "text": "new"}],
                    provider="openai",
                    language="english",
                    output_dir=str(out),
                    results_csv_path=results_csv,
                    overwrite=True,
                )

            df = pd.read_csv(results_csv)
            self.assertEqual(df.iloc[0]["text"], "new")
            self.assertEqual(df.iloc[0]["ttfb"], 0.5)


if __name__ == "__main__":
    unittest.main()
