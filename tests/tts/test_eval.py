"""
Tests for calibrate/tts/eval.py — routers, validators, save_audio, run_tts_eval.

Run with:
    python -m unittest tests.tts.test_eval -v
"""

import asyncio
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pandas as pd


class TestSaveAudio(unittest.TestCase):
    def test_wav_passthrough(self):
        from calibrate.tts.eval import save_audio

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
        from calibrate.tts.eval import save_audio

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
        from calibrate.tts.eval import validate_tts_input_file

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
        from calibrate.tts.eval import validate_tts_input_file

        ok, err = validate_tts_input_file("/nope.csv")
        self.assertFalse(ok)
        self.assertIn("does not exist", err)

    def test_not_csv_extension(self):
        from calibrate.tts.eval import validate_tts_input_file

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
        from calibrate.tts.eval import validate_tts_input_file

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
        from calibrate.tts.eval import validate_tts_input_file

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
        from calibrate.tts.eval import validate_tts_input_file

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
        from calibrate.tts.eval import validate_existing_results_csv

        ok, err = validate_existing_results_csv("/nonexistent.csv")
        self.assertTrue(ok)

    def test_valid_columns(self):
        from calibrate.tts.eval import validate_existing_results_csv

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
        from calibrate.tts.eval import validate_existing_results_csv

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
        from calibrate.tts import eval as tts_eval

        with self.assertRaises(ValueError):
            await tts_eval.synthesize_speech.__wrapped__(
                "hello", "no-such-provider", "english", "/tmp/x.wav"
            )

    async def test_known_provider_routed(self):
        from calibrate.tts import eval as tts_eval

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
        from calibrate.tts import eval as tts_eval

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
        from calibrate.tts import eval as tts_eval

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
        from calibrate.tts import eval as tts_eval

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


class TestRunTTSEvalParallel(unittest.IsolatedAsyncioTestCase):
    async def test_concurrency_overlaps(self):
        from calibrate.tts import eval as tts_eval

        row_parallel = 3
        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()
        release = asyncio.Event()
        reached = asyncio.Event()

        async def fake_synth(text, provider, language, audio_path):
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                if in_flight >= row_parallel:
                    reached.set()
            # Hold the row open until enough rows are concurrently in-flight.
            await release.wait()
            async with lock:
                in_flight -= 1
            Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
            Path(audio_path).write_bytes(b"RIFF")
            return {"ttfb": 0.1}

        async def releaser():
            # Once row_parallel rows are simultaneously in-flight, let them go.
            await reached.wait()
            release.set()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_csv = out / "results.csv"
            gt_data = [{"id": f"row_{i}", "text": f"t{i}"} for i in range(6)]
            with patch.object(
                tts_eval, "synthesize_speech", AsyncMock(side_effect=fake_synth)
            ):
                await asyncio.gather(
                    tts_eval.run_tts_eval(
                        gt_data=gt_data,
                        provider="openai",
                        language="english",
                        output_dir=str(out),
                        results_csv_path=results_csv,
                        row_parallel=row_parallel,
                    ),
                    releaser(),
                )

            self.assertGreaterEqual(max_in_flight, row_parallel)

    async def test_output_order_matches_input(self):
        from calibrate.tts import eval as tts_eval

        # Make later ids finish first to force out-of-order completion.
        delays = {"row_a": 0.06, "row_b": 0.02, "row_c": 0.04}

        async def fake_synth(text, provider, language, audio_path):
            _id = Path(audio_path).stem
            await asyncio.sleep(delays[_id])
            Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
            Path(audio_path).write_bytes(b"RIFF")
            return {"ttfb": delays[_id]}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_csv = out / "results.csv"
            gt_data = [
                {"id": "row_a", "text": "a"},
                {"id": "row_b", "text": "b"},
                {"id": "row_c", "text": "c"},
            ]
            with patch.object(
                tts_eval, "synthesize_speech", AsyncMock(side_effect=fake_synth)
            ):
                await tts_eval.run_tts_eval(
                    gt_data=gt_data,
                    provider="openai",
                    language="english",
                    output_dir=str(out),
                    results_csv_path=results_csv,
                    row_parallel=3,
                )

            df = pd.read_csv(results_csv)
            self.assertEqual(
                df["id"].astype(str).tolist(), ["row_a", "row_b", "row_c"]
            )

    async def test_resume_skips_processed_ids_parallel(self):
        from calibrate.tts import eval as tts_eval

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_csv = out / "results.csv"
            pd.DataFrame(
                [{"id": "row_a", "text": "hello", "audio_path": "/x.wav", "ttfb": 0.1}]
            ).to_csv(results_csv, index=False)

            processed = []

            async def fake_synth(text, provider, language, audio_path):
                processed.append(Path(audio_path).stem)
                Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
                Path(audio_path).write_bytes(b"RIFF")
                return {"ttfb": 0.2}

            with patch.object(
                tts_eval, "synthesize_speech", AsyncMock(side_effect=fake_synth)
            ):
                result = await tts_eval.run_tts_eval(
                    gt_data=[
                        {"id": "row_a", "text": "hello"},
                        {"id": "row_b", "text": "world"},
                        {"id": "row_c", "text": "again"},
                    ],
                    provider="openai",
                    language="english",
                    output_dir=str(out),
                    results_csv_path=results_csv,
                    row_parallel=3,
                )

            self.assertNotIn("row_a", processed)
            self.assertEqual(result["success_count"], 2)
            df = pd.read_csv(results_csv)
            self.assertEqual(
                df["id"].astype(str).tolist(), ["row_a", "row_b", "row_c"]
            )

    async def test_overwrite_wipes_and_reprocesses_parallel(self):
        from calibrate.tts import eval as tts_eval

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_csv = out / "results.csv"
            pd.DataFrame(
                [
                    {"id": "row_a", "text": "old", "audio_path": "/x.wav", "ttfb": 0.1},
                    {"id": "row_b", "text": "old", "audio_path": "/y.wav", "ttfb": 0.1},
                ]
            ).to_csv(results_csv, index=False)

            async def fake_synth(text, provider, language, audio_path):
                Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
                Path(audio_path).write_bytes(b"RIFF")
                return {"ttfb": 0.5}

            with patch.object(
                tts_eval, "synthesize_speech", AsyncMock(side_effect=fake_synth)
            ):
                result = await tts_eval.run_tts_eval(
                    gt_data=[
                        {"id": "row_a", "text": "new"},
                        {"id": "row_b", "text": "new"},
                    ],
                    provider="openai",
                    language="english",
                    output_dir=str(out),
                    results_csv_path=results_csv,
                    overwrite=True,
                    row_parallel=2,
                )

            self.assertEqual(result["success_count"], 2)
            df = pd.read_csv(results_csv)
            self.assertEqual(df["id"].astype(str).tolist(), ["row_a", "row_b"])
            self.assertTrue((df["text"] == "new").all())
            self.assertTrue((df["ttfb"] == 0.5).all())


class TestRunTTSEvalRowParallel(unittest.IsolatedAsyncioTestCase):
    """Row-level parallelism in run_tts_eval: bounded concurrency, output
    ordering, resume, and overwrite semantics."""

    async def test_concurrency_actually_overlaps(self):
        from calibrate.tts import eval as tts_eval

        row_parallel = 3
        n_rows = 5
        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()
        release = asyncio.Event()
        reached = asyncio.Event()

        async def fake_synth(text, provider, language, audio_path):
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                if in_flight >= min(row_parallel, n_rows):
                    reached.set()
            # Keep the row open until enough rows are concurrently in-flight.
            await release.wait()
            async with lock:
                in_flight -= 1
            Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
            Path(audio_path).write_bytes(b"RIFF")
            return {"ttfb": 0.1}

        async def releaser():
            await reached.wait()
            release.set()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_csv = out / "results.csv"
            gt_data = [{"id": f"row_{i}", "text": f"t{i}"} for i in range(n_rows)]
            with patch.object(
                tts_eval, "synthesize_speech", AsyncMock(side_effect=fake_synth)
            ):
                await asyncio.gather(
                    tts_eval.run_tts_eval(
                        gt_data=gt_data,
                        provider="openai",
                        language="english",
                        output_dir=str(out),
                        results_csv_path=results_csv,
                        row_parallel=row_parallel,
                    ),
                    releaser(),
                )

            self.assertGreaterEqual(max_in_flight, min(row_parallel, n_rows))

    async def test_concurrency_capped_serialized(self):
        from calibrate.tts import eval as tts_eval

        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        async def fake_synth(text, provider, language, audio_path):
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            # Yield control so a second concurrent row could interleave if the
            # semaphore weren't capping concurrency at 1.
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1
            Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
            Path(audio_path).write_bytes(b"RIFF")
            return {"ttfb": 0.1}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_csv = out / "results.csv"
            gt_data = [{"id": f"row_{i}", "text": f"t{i}"} for i in range(5)]
            with patch.object(
                tts_eval, "synthesize_speech", AsyncMock(side_effect=fake_synth)
            ):
                await tts_eval.run_tts_eval(
                    gt_data=gt_data,
                    provider="openai",
                    language="english",
                    output_dir=str(out),
                    results_csv_path=results_csv,
                    row_parallel=1,
                )

            self.assertEqual(max_in_flight, 1)

    async def test_output_order_preserved_when_later_rows_finish_first(self):
        from calibrate.tts import eval as tts_eval

        # Later rows return first so completion order is the reverse of input.
        delays = {"row_a": 0.06, "row_b": 0.04, "row_c": 0.02}

        async def fake_synth(text, provider, language, audio_path):
            _id = Path(audio_path).stem
            await asyncio.sleep(delays[_id])
            Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
            Path(audio_path).write_bytes(b"RIFF")
            return {"ttfb": delays[_id]}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_csv = out / "results.csv"
            gt_data = [
                {"id": "row_a", "text": "a"},
                {"id": "row_b", "text": "b"},
                {"id": "row_c", "text": "c"},
            ]
            with patch.object(
                tts_eval, "synthesize_speech", AsyncMock(side_effect=fake_synth)
            ):
                await tts_eval.run_tts_eval(
                    gt_data=gt_data,
                    provider="openai",
                    language="english",
                    output_dir=str(out),
                    results_csv_path=results_csv,
                    row_parallel=3,
                )

            df = pd.read_csv(results_csv)
            self.assertEqual(
                df["id"].astype(str).tolist(), ["row_a", "row_b", "row_c"]
            )

    async def test_resume_does_not_recall_processed_id(self):
        from calibrate.tts import eval as tts_eval

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_csv = out / "results.csv"
            pd.DataFrame(
                [{"id": "row_a", "text": "hello", "audio_path": "/x.wav", "ttfb": 0.1}]
            ).to_csv(results_csv, index=False)

            processed = []

            async def fake_synth(text, provider, language, audio_path):
                processed.append(Path(audio_path).stem)
                Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
                Path(audio_path).write_bytes(b"RIFF")
                return {"ttfb": 0.2}

            with patch.object(
                tts_eval, "synthesize_speech", AsyncMock(side_effect=fake_synth)
            ):
                result = await tts_eval.run_tts_eval(
                    gt_data=[
                        {"id": "row_a", "text": "hello"},
                        {"id": "row_b", "text": "world"},
                        {"id": "row_c", "text": "again"},
                    ],
                    provider="openai",
                    language="english",
                    output_dir=str(out),
                    results_csv_path=results_csv,
                    row_parallel=3,
                )

            self.assertNotIn("row_a", processed)
            self.assertEqual(result["success_count"], 2)
            df = pd.read_csv(results_csv)
            self.assertIn("row_a", df["id"].astype(str).tolist())
            self.assertEqual(
                df["id"].astype(str).tolist(), ["row_a", "row_b", "row_c"]
            )

    async def test_overwrite_reprocesses_all_rows(self):
        from calibrate.tts import eval as tts_eval

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_csv = out / "results.csv"
            pd.DataFrame(
                [
                    {"id": "row_a", "text": "old", "audio_path": "/x.wav", "ttfb": 0.1},
                    {"id": "row_b", "text": "old", "audio_path": "/y.wav", "ttfb": 0.1},
                ]
            ).to_csv(results_csv, index=False)

            processed = []

            async def fake_synth(text, provider, language, audio_path):
                processed.append(Path(audio_path).stem)
                Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
                Path(audio_path).write_bytes(b"RIFF")
                return {"ttfb": 0.5}

            with patch.object(
                tts_eval, "synthesize_speech", AsyncMock(side_effect=fake_synth)
            ):
                result = await tts_eval.run_tts_eval(
                    gt_data=[
                        {"id": "row_a", "text": "new"},
                        {"id": "row_b", "text": "new"},
                    ],
                    provider="openai",
                    language="english",
                    output_dir=str(out),
                    results_csv_path=results_csv,
                    overwrite=True,
                    row_parallel=2,
                )

            self.assertEqual(sorted(processed), ["row_a", "row_b"])
            self.assertEqual(result["success_count"], 2)
            df = pd.read_csv(results_csv)
            self.assertEqual(df["id"].astype(str).tolist(), ["row_a", "row_b"])
            self.assertTrue((df["text"] == "new").all())
            self.assertTrue((df["ttfb"] == 0.5).all())


class TestResolveRowParallelTTS(unittest.TestCase):
    """Precedence for resolve_row_parallel('tts', ...): CLI > env > default."""

    def test_cli_value_takes_precedence(self):
        from calibrate.utils import resolve_row_parallel

        with patch.dict(os.environ, {"CALIBRATE_TTS_PARALLEL": "7"}, clear=False):
            self.assertEqual(resolve_row_parallel("tts", 3), 3)

    def test_env_used_when_no_cli(self):
        from calibrate.utils import resolve_row_parallel

        with patch.dict(os.environ, {"CALIBRATE_TTS_PARALLEL": "6"}, clear=False):
            self.assertEqual(resolve_row_parallel("tts", None), 6)

    def test_default_when_nothing_set(self):
        from calibrate.utils import resolve_row_parallel, DEFAULT_ROW_PARALLEL

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CALIBRATE_TTS_PARALLEL", None)
            self.assertEqual(
                resolve_row_parallel("tts", None), DEFAULT_ROW_PARALLEL
            )

    def test_non_positive_and_garbage_fall_back_to_default(self):
        from calibrate.utils import resolve_row_parallel, DEFAULT_ROW_PARALLEL

        # Non-positive CLI values are ignored; with no usable env, fall back.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CALIBRATE_TTS_PARALLEL", None)
            self.assertEqual(resolve_row_parallel("tts", 0), DEFAULT_ROW_PARALLEL)
            self.assertEqual(resolve_row_parallel("tts", -5), DEFAULT_ROW_PARALLEL)

        # Garbage / non-positive env values are ignored too.
        for bad in ("abc", "0", "-3"):
            with patch.dict(
                os.environ, {"CALIBRATE_TTS_PARALLEL": bad}, clear=False
            ):
                self.assertEqual(
                    resolve_row_parallel("tts", None), DEFAULT_ROW_PARALLEL
                )


if __name__ == "__main__":
    unittest.main()
