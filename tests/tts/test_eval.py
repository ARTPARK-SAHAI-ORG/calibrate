"""
Tests for calibrate_agent/tts/eval.py — routers, validators, save_audio, run_tts_eval.

Run with:
    python -m unittest tests.tts.test_eval -v
"""

import asyncio
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

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


class TestSynthesizeGemini(unittest.IsolatedAsyncioTestCase):
    def _chunk(self, pcm: bytes):
        from types import SimpleNamespace

        part = SimpleNamespace(inline_data=SimpleNamespace(data=pcm))
        content = SimpleNamespace(parts=[part])
        return SimpleNamespace(candidates=[SimpleNamespace(content=content)])

    async def test_streams_chunks_and_uses_model_and_voice(self):
        from calibrate_agent.tts import eval as tts_eval

        chunks = [self._chunk(b"\x01\x02" * 50), self._chunk(b"\x03\x04" * 50)]

        async def _stream(*args, **kwargs):
            for c in chunks:
                yield c

        stream_fn = AsyncMock(return_value=_stream())
        client_obj = AsyncMock()
        client_obj.aio.models.generate_content_stream = stream_fn
        saved = {}

        def fake_save(audio_bytes, output_path, sample_rate=24000):
            saved["bytes"] = audio_bytes
            saved["sample_rate"] = sample_rate

        patches = (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-fake"}),
            patch.object(tts_eval.genai, "Client", return_value=client_obj),
            patch.object(tts_eval, "save_audio", side_effect=fake_save),
        )
        for p in patches:
            p.start()
        try:
            result = await tts_eval.synthesize_gemini(
                "hello", "kannada", "/tmp/out.wav"
            )
        finally:
            for p in patches:
                p.stop()

        self.assertIsInstance(result["ttfb"], float)
        # ttfb reflects the FIRST audio chunk; all chunks are concatenated.
        self.assertEqual(saved["bytes"], b"\x01\x02" * 50 + b"\x03\x04" * 50)
        self.assertEqual(saved["sample_rate"], 24000)

        kwargs = stream_fn.await_args.kwargs
        self.assertEqual(kwargs["model"], tts_eval.TTS_PROVIDER_MODELS["gemini"])
        voice = (
            kwargs["config"].speech_config.voice_config.prebuilt_voice_config.voice_name
        )
        self.assertEqual(voice, tts_eval.get_tts_voice("gemini", "kannada"))

    async def test_no_audio_raises_without_writing(self):
        from types import SimpleNamespace
        from calibrate_agent.tts import eval as tts_eval

        # A text-only / blocked response: chunk carries no inline audio.
        empty_chunk = SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=None))]
        )

        async def _stream(*args, **kwargs):
            yield empty_chunk

        client_obj = AsyncMock()
        client_obj.aio.models.generate_content_stream = AsyncMock(
            return_value=_stream()
        )
        save = MagicMock()

        patches = (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-fake"}),
            patch.object(tts_eval.genai, "Client", return_value=client_obj),
            patch.object(tts_eval, "save_audio", save),
        )
        for p in patches:
            p.start()
        try:
            with self.assertRaises(ValueError):
                await tts_eval.synthesize_gemini("hi", "english", "/tmp/out.wav")
        finally:
            for p in patches:
                p.stop()
        save.assert_not_called()

    async def test_missing_key_raises(self):
        from calibrate_agent.tts import eval as tts_eval

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                await tts_eval.synthesize_gemini("hi", "english", "/tmp/out.wav")


def _write_wav(path: str) -> None:
    """Write a tiny valid WAV file so eval-only validation's existence check passes."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)


def _make_run_dir(tmp: str, audio_path: str = "audios/row_1.wav", extra_cols: dict = None) -> str:
    """Create a minimal TTS run dir (results.csv + audios/row_1.wav) under ``tmp``."""
    run_dir = os.path.join(tmp, "openai")
    os.makedirs(os.path.join(run_dir, "audios"))
    _write_wav(os.path.join(run_dir, "audios", "row_1.wav"))
    row = {"id": "row_1", "text": "hello world", "audio_path": audio_path}
    if extra_cols:
        row.update(extra_cols)
    pd.DataFrame([row]).to_csv(os.path.join(run_dir, "results.csv"), index=False)
    return run_dir


class TestTTSValidateEvalOnlyDataset(unittest.TestCase):
    def test_reads_run_directory(self):
        """Pointing at a run directory reads its results.csv with no transform."""
        from calibrate_agent.tts.eval import validate_tts_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(tmp)
            ok, err, rows = validate_tts_eval_only_dataset(run_dir)
            self.assertTrue(ok, err)
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["audio_path"], os.path.join(run_dir, "audios", "row_1.wav")
            )

    def test_extra_columns_ignored_and_cwd_independent(self):
        """A ttfb column is ignored; an audio_path stored relative to a different
        cwd still resolves via the {run_dir}/audios/{basename} fallback."""
        from calibrate_agent.tts.eval import validate_tts_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(
                tmp,
                audio_path="./out/run/openai/audios/row_1.wav",
                extra_cols={"ttfb": 1.23},
            )
            ok, err, rows = validate_tts_eval_only_dataset(run_dir)
            self.assertTrue(ok, err)
            self.assertEqual(
                rows[0]["audio_path"], os.path.join(run_dir, "audios", "row_1.wav")
            )

    def test_path_does_not_exist(self):
        from calibrate_agent.tts.eval import validate_tts_eval_only_dataset

        ok, err, rows = validate_tts_eval_only_dataset("/nope")
        self.assertFalse(ok)
        self.assertEqual(rows, [])
        self.assertIn("does not exist", err)

    def test_rejects_a_file(self):
        from calibrate_agent.tts.eval import validate_tts_eval_only_dataset

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            ok, err, rows = validate_tts_eval_only_dataset(path)
            self.assertFalse(ok)
            self.assertIn("must be a run directory", err)
        finally:
            os.remove(path)

    def test_directory_without_results_csv(self):
        from calibrate_agent.tts.eval import validate_tts_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            ok, err, rows = validate_tts_eval_only_dataset(tmp)
            self.assertFalse(ok)
            self.assertIn("No results.csv", err)

    def test_results_csv_missing_columns(self):
        from calibrate_agent.tts.eval import validate_tts_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame([{"id": "1", "text": "hi"}]).to_csv(
                os.path.join(tmp, "results.csv"), index=False
            )
            ok, err, rows = validate_tts_eval_only_dataset(tmp)
            self.assertFalse(ok)
            self.assertIn("missing required columns", err)

    def test_audio_file_absent(self):
        from calibrate_agent.tts.eval import validate_tts_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(
                [{"id": "1", "text": "hi", "audio_path": "audios/missing.wav"}]
            ).to_csv(os.path.join(tmp, "results.csv"), index=False)
            ok, err, rows = validate_tts_eval_only_dataset(tmp)
            self.assertFalse(ok)
            self.assertIn("audio file does not exist", err)

    def test_empty_results_csv(self):
        from calibrate_agent.tts.eval import validate_tts_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(columns=["id", "text", "audio_path"]).to_csv(
                os.path.join(tmp, "results.csv"), index=False
            )
            ok, err, rows = validate_tts_eval_only_dataset(tmp)
            self.assertFalse(ok)
            self.assertIn("empty", err)


class TestTTSRunEvalOnly(unittest.IsolatedAsyncioTestCase):
    async def test_runs_on_run_directory(self):
        """End-to-end: point run_eval_only at a run dir, no dataset transform."""
        from calibrate_agent.tts import eval as tts_eval

        async def fake_judge(audio_paths, texts, **kwargs):
            return {
                "scores": {"quality": {"type": "binary", "mean": 1.0}},
                "score": 1.0,
                "per_row": [{"quality": {"match": True, "reasoning": "ok"}}],
            }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(tmp, extra_cols={"ttfb": 1.0})
            out = os.path.join(tmp, "eval")

            with patch.object(
                tts_eval, "get_tts_llm_judge_score", AsyncMock(side_effect=fake_judge)
            ):
                result = await tts_eval.run_eval_only(
                    dataset_path=run_dir,
                    output_dir=out,
                    judge_evaluators=[
                        {
                            "name": "quality",
                            "system_prompt": "judge quality",
                            "judge_model": "openai/gpt-4.1",
                            "type": "binary",
                        }
                    ],
                )

            self.assertEqual(result["status"], "completed")
            self.assertIn("quality", result["metrics"])
            self.assertTrue(os.path.exists(os.path.join(out, "metrics.json")))
            df = pd.read_csv(os.path.join(out, "results.csv"))
            # No ttfb column in eval-only results; evaluator columns present.
            self.assertNotIn("ttfb", df.columns)
            self.assertIn("quality", df.columns)
            self.assertIn("quality_reasoning", df.columns)

    async def test_invalid_dataset_returns_error(self):
        from calibrate_agent.tts import eval as tts_eval

        result = await tts_eval.run_eval_only(
            dataset_path="/nonexistent",
            output_dir=tempfile.mkdtemp(),
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("does not exist", result["error"])

    async def test_output_dir_same_as_run_dir_rejected(self):
        """Judging a run into its own dir would clobber its results.csv."""
        from calibrate_agent.tts import eval as tts_eval

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(tmp, extra_cols={"ttfb": 1.0})
            result = await tts_eval.run_eval_only(
                dataset_path=run_dir,
                output_dir=os.path.join(run_dir, ""),  # same dir, trailing slash
            )
            self.assertEqual(result["status"], "error")
            self.assertIn("must differ", result["error"])
            # The original run's results.csv is untouched (ttfb column intact).
            df = pd.read_csv(os.path.join(run_dir, "results.csv"))
            self.assertIn("ttfb", df.columns)


class TestTTSPartialResults(unittest.IsolatedAsyncioTestCase):
    """results.csv holds the rows graded so far while the judge is still running."""

    async def test_partial_results_written_during_run(self):
        from calibrate_agent.tts import metrics as tts_metrics
        from calibrate_agent.tts.eval import _score_and_write_results

        evaluator = {
            "name": "quality",
            "system_prompt": "judge quality",
            "judge_model": "openai/gpt-audio",
            "type": "binary",
        }

        with tempfile.TemporaryDirectory() as out_dir:
            results_path = os.path.join(out_dir, "results.csv")
            seen = {}

            async def fake_judge(audio_path, reference_text, **kwargs):
                if reference_text == "second":
                    # The first row finishes first; capture what is on disk by then.
                    await asyncio.sleep(0.05)
                    seen["partial"] = pd.read_csv(results_path)
                return {
                    "quality": {
                        "match": reference_text == "first",
                        "reasoning": reference_text,
                    }
                }

            with patch.object(
                tts_metrics, "tts_llm_judge", AsyncMock(side_effect=fake_judge)
            ):
                metrics = await _score_and_write_results(
                    ids=["row_a", "row_b"],
                    texts=["first", "second"],
                    audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
                    output_dir=out_dir,
                    evaluator_config_dir=out_dir,
                    judge_evaluators=[evaluator],
                    stream_rows=True,
                )

            partial = seen["partial"]
            self.assertEqual(list(partial["id"]), ["row_a"])
            self.assertEqual(bool(partial.iloc[0]["quality"]), True)
            self.assertEqual(partial.iloc[0]["quality_reasoning"], "first")

            self.assertAlmostEqual(metrics["quality"]["mean"], 0.5)
            df = pd.read_csv(results_path)
            self.assertEqual(list(df["id"]), ["row_a", "row_b"])
            self.assertEqual(
                list(df.columns),
                ["id", "text", "audio_path", "quality", "quality_reasoning"],
            )
            self.assertEqual([bool(v) for v in df["quality"]], [True, False])

    async def test_no_partial_results_without_stream_rows(self):
        """The full run keeps its synthesis results in results.csv and resumes
        from them, so the judge must not overwrite that file mid-run."""
        from calibrate_agent.tts import metrics as tts_metrics
        from calibrate_agent.tts.eval import _score_and_write_results

        evaluator = {
            "name": "quality",
            "system_prompt": "judge quality",
            "judge_model": "openai/gpt-audio",
            "type": "binary",
        }

        with tempfile.TemporaryDirectory() as out_dir:
            results_path = os.path.join(out_dir, "results.csv")
            synthesis_csv = "id,text,audio_path,ttfb\nrow_a,first,/tmp/a.wav,0.2\n"
            with open(results_path, "w") as f:
                f.write(synthesis_csv)
            seen = {}

            async def fake_judge(audio_path, reference_text, **kwargs):
                if reference_text == "second":
                    await asyncio.sleep(0.05)
                    with open(results_path) as f:
                        seen["mid_run"] = f.read()
                return {
                    "quality": {"match": True, "reasoning": reference_text}
                }

            with patch.object(
                tts_metrics, "tts_llm_judge", AsyncMock(side_effect=fake_judge)
            ):
                await _score_and_write_results(
                    ids=["row_a", "row_b"],
                    texts=["first", "second"],
                    audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
                    output_dir=out_dir,
                    evaluator_config_dir=out_dir,
                    judge_evaluators=[evaluator],
                )

            self.assertEqual(seen["mid_run"], synthesis_csv)


if __name__ == "__main__":
    unittest.main()
