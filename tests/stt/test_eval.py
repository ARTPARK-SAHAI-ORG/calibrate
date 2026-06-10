"""
Tests for calibrate/stt/eval.py — routers, validators, and result writers.

Run with:
    python -m unittest tests.stt.test_eval -v
"""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

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


class TestSTTRunSTTEvalParallel(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _gt_data(ids):
        return [{"id": i, "gt": f"gt {i}"} for i in ids]

    async def test_concurrency_overlaps(self):
        """With row_parallel=3 at least 3 rows are transcribed simultaneously."""
        from calibrate.stt import eval as stt_eval

        ids = ["row_a", "row_b", "row_c", "row_d", "row_e"]
        release = asyncio.Event()
        in_flight = 0
        max_in_flight = 0
        reached = asyncio.Event()
        lock = asyncio.Lock()

        async def fake_transcribe(audio_path, reference, provider, language, uid):
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                if in_flight >= 3:
                    reached.set()
            await release.wait()
            async with lock:
                in_flight -= 1
            return f"pred {reference}"

        async def releaser():
            await reached.wait()
            release.set()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.csv"
            with patch.object(
                stt_eval, "transcribe_audio", AsyncMock(side_effect=fake_transcribe)
            ):
                await asyncio.gather(
                    stt_eval.run_stt_eval(
                        gt_data=self._gt_data(ids),
                        audio_dir=Path(tmp),
                        provider="deepgram",
                        language="english",
                        results_csv_path=out,
                        row_parallel=3,
                    ),
                    releaser(),
                )

        self.assertGreaterEqual(max_in_flight, 3)

    async def test_output_order_matches_input(self):
        """CSV rows preserve input order even when later rows finish first."""
        from calibrate.stt import eval as stt_eval

        ids = ["row_a", "row_b", "row_c", "row_d"]

        async def fake_transcribe(audio_path, reference, provider, language, uid):
            # Later ids (lexicographically larger) return faster.
            stem = audio_path.stem
            delay = 0.05 * (len(ids) - ids.index(stem))
            await asyncio.sleep(delay)
            return f"pred-{stem}"

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.csv"
            with patch.object(
                stt_eval, "transcribe_audio", AsyncMock(side_effect=fake_transcribe)
            ):
                count = await stt_eval.run_stt_eval(
                    gt_data=self._gt_data(ids),
                    audio_dir=Path(tmp),
                    provider="deepgram",
                    language="english",
                    results_csv_path=out,
                    row_parallel=4,
                )

            self.assertEqual(count, 4)
            df = pd.read_csv(out)
            self.assertEqual(df["id"].tolist(), ids)
            self.assertEqual(df["pred"].tolist(), [f"pred-{i}" for i in ids])

    async def test_resume_skips_processed_ids(self):
        """Pre-seeded rows are kept, not re-transcribed, and new rows appended."""
        from calibrate.stt import eval as stt_eval

        ids = ["row_a", "row_b", "row_c"]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.csv"
            # Pre-seed first two ids as already processed.
            pd.DataFrame(
                [
                    {"id": "row_a", "gt": "gt row_a", "pred": "old_a"},
                    {"id": "row_b", "gt": "gt row_b", "pred": "old_b"},
                ]
            ).to_csv(out, index=False)

            seen = []

            async def fake_transcribe(audio_path, reference, provider, language, uid):
                seen.append(audio_path.stem)
                return "new_c"

            with patch.object(
                stt_eval, "transcribe_audio", AsyncMock(side_effect=fake_transcribe)
            ):
                count = await stt_eval.run_stt_eval(
                    gt_data=self._gt_data(ids),
                    audio_dir=Path(tmp),
                    provider="deepgram",
                    language="english",
                    results_csv_path=out,
                    row_parallel=4,
                )

            self.assertEqual(seen, ["row_c"])  # only the unprocessed id
            self.assertEqual(count, 1)
            df = pd.read_csv(out)
            self.assertEqual(df["id"].tolist(), ids)
            self.assertEqual(
                df["pred"].tolist(), ["old_a", "old_b", "new_c"]
            )

    async def test_row_parallel_limit_serializes(self):
        """row_parallel=1 forces strictly serial transcription (no overlap)."""
        from calibrate.stt import eval as stt_eval

        ids = ["row_a", "row_b", "row_c"]
        in_flight = 0
        max_in_flight = 0

        async def fake_transcribe(audio_path, reference, provider, language, uid):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return "x"

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.csv"
            with patch.object(
                stt_eval, "transcribe_audio", AsyncMock(side_effect=fake_transcribe)
            ):
                await stt_eval.run_stt_eval(
                    gt_data=self._gt_data(ids),
                    audio_dir=Path(tmp),
                    provider="deepgram",
                    language="english",
                    results_csv_path=out,
                    row_parallel=1,
                )

        self.assertEqual(max_in_flight, 1)

    async def test_env_var_caps_concurrency(self):
        """CALIBRATE_STT_PARALLEL caps concurrency when no row_parallel passed."""
        from calibrate.stt import eval as stt_eval

        ids = ["row_a", "row_b", "row_c", "row_d"]
        in_flight = 0
        max_in_flight = 0

        async def fake_transcribe(audio_path, reference, provider, language, uid):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return "x"

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.csv"
            with patch.dict(
                "os.environ", {"CALIBRATE_STT_PARALLEL": "2"}
            ), patch.object(
                stt_eval, "transcribe_audio", AsyncMock(side_effect=fake_transcribe)
            ):
                await stt_eval.run_stt_eval(
                    gt_data=self._gt_data(ids),
                    audio_dir=Path(tmp),
                    provider="deepgram",
                    language="english",
                    results_csv_path=out,
                )

        self.assertLessEqual(max_in_flight, 2)

    async def test_failure_propagates(self):
        """An exception from transcribe_audio bubbles out of run_stt_eval."""
        from calibrate.stt import eval as stt_eval

        async def fake_transcribe(audio_path, reference, provider, language, uid):
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.csv"
            with patch.object(
                stt_eval, "transcribe_audio", AsyncMock(side_effect=fake_transcribe)
            ):
                with self.assertRaises(RuntimeError):
                    await stt_eval.run_stt_eval(
                        gt_data=self._gt_data(["row_a"]),
                        audio_dir=Path(tmp),
                        provider="deepgram",
                        language="english",
                        results_csv_path=out,
                        row_parallel=2,
                    )


class TestRunSTTEvalRowParallel(unittest.IsolatedAsyncioTestCase):
    """Row-level concurrency behaviour of ``run_stt_eval``."""

    @staticmethod
    def _gt_data(ids):
        # Non-numeric ids on purpose: pandas coerces numeric-looking ids to
        # int on CSV round-trip, which breaks the resume id comparison.
        return [{"id": i, "gt": f"gt {i}"} for i in ids]

    @staticmethod
    def _make_audio(audio_dir: Path, ids):
        for i in ids:
            (audio_dir / f"{i}.wav").write_bytes(b"RIFF0000WAVE")

    async def test_concurrency_actually_overlaps(self):
        """row_parallel=3 over 5 rows reaches a peak in-flight of >= 3."""
        from calibrate.stt import eval as stt_eval

        ids = ["row_a", "row_b", "row_c", "row_d", "row_e"]
        row_parallel = 3
        in_flight = 0
        max_in_flight = 0
        counter_lock = asyncio.Lock()
        release = asyncio.Event()
        reached = asyncio.Event()

        async def fake_transcribe(audio_path, reference, provider, language, uid):
            nonlocal in_flight, max_in_flight
            async with counter_lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                if in_flight >= min(row_parallel, len(ids)):
                    reached.set()
            await release.wait()
            async with counter_lock:
                in_flight -= 1
            return f"pred {reference}"

        async def releaser():
            await reached.wait()
            release.set()

        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            self._make_audio(audio_dir, ids)
            out = audio_dir / "results.csv"
            with patch.object(
                stt_eval, "transcribe_audio", AsyncMock(side_effect=fake_transcribe)
            ):
                await asyncio.gather(
                    stt_eval.run_stt_eval(
                        gt_data=self._gt_data(ids),
                        audio_dir=audio_dir,
                        provider="deepgram",
                        language="english",
                        results_csv_path=out,
                        row_parallel=row_parallel,
                    ),
                    releaser(),
                )

        self.assertGreaterEqual(max_in_flight, min(row_parallel, len(ids)))

    async def test_concurrency_is_capped_at_one(self):
        """row_parallel=1 serializes transcription (peak in-flight never > 1)."""
        from calibrate.stt import eval as stt_eval

        ids = ["row_a", "row_b", "row_c", "row_d"]
        in_flight = 0
        max_in_flight = 0

        async def fake_transcribe(audio_path, reference, provider, language, uid):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return "x"

        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            self._make_audio(audio_dir, ids)
            out = audio_dir / "results.csv"
            with patch.object(
                stt_eval, "transcribe_audio", AsyncMock(side_effect=fake_transcribe)
            ):
                await stt_eval.run_stt_eval(
                    gt_data=self._gt_data(ids),
                    audio_dir=audio_dir,
                    provider="deepgram",
                    language="english",
                    results_csv_path=out,
                    row_parallel=1,
                )

        self.assertEqual(max_in_flight, 1)

    async def test_output_order_preserved_when_later_rows_finish_first(self):
        """Reversed completion order still writes CSV in input order."""
        from calibrate.stt import eval as stt_eval

        ids = ["row_a", "row_b", "row_c", "row_d"]

        async def fake_transcribe(audio_path, reference, provider, language, uid):
            # Earlier ids sleep longer so later ids complete first.
            stem = audio_path.stem
            delay = 0.05 * (len(ids) - ids.index(stem))
            await asyncio.sleep(delay)
            return f"pred-{stem}"

        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            self._make_audio(audio_dir, ids)
            out = audio_dir / "results.csv"
            with patch.object(
                stt_eval, "transcribe_audio", AsyncMock(side_effect=fake_transcribe)
            ):
                count = await stt_eval.run_stt_eval(
                    gt_data=self._gt_data(ids),
                    audio_dir=audio_dir,
                    provider="deepgram",
                    language="english",
                    results_csv_path=out,
                    row_parallel=4,
                )

            self.assertEqual(count, len(ids))
            df = pd.read_csv(out)
            self.assertEqual(df["id"].tolist(), ids)
            self.assertEqual(df["pred"].tolist(), [f"pred-{i}" for i in ids])

    async def test_resume_skips_already_processed_id(self):
        """A pre-seeded id is kept and never re-transcribed."""
        from calibrate.stt import eval as stt_eval

        ids = ["row_a", "row_b", "row_c"]
        seen = []

        async def fake_transcribe(audio_path, reference, provider, language, uid):
            seen.append(audio_path.stem)
            return f"new-{audio_path.stem}"

        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            self._make_audio(audio_dir, ids)
            out = audio_dir / "results.csv"
            # Pre-seed the first id as already processed.
            pd.DataFrame(
                [{"id": "row_a", "gt": "gt row_a", "pred": "old_a"}]
            ).to_csv(out, index=False)

            with patch.object(
                stt_eval, "transcribe_audio", AsyncMock(side_effect=fake_transcribe)
            ):
                await stt_eval.run_stt_eval(
                    gt_data=self._gt_data(ids),
                    audio_dir=audio_dir,
                    provider="deepgram",
                    language="english",
                    results_csv_path=out,
                    row_parallel=4,
                )

            self.assertNotIn("row_a", seen)
            self.assertEqual(sorted(seen), ["row_b", "row_c"])
            df = pd.read_csv(out)
            self.assertIn("row_a", df["id"].tolist())
            self.assertEqual(
                df.loc[df["id"] == "row_a", "pred"].iloc[0], "old_a"
            )


class TestResolveRowParallelPrecedence(unittest.TestCase):
    """Precedence rules of ``resolve_row_parallel`` for the STT component."""

    def test_cli_value_takes_precedence_over_env(self):
        from calibrate.utils import resolve_row_parallel

        with patch.dict("os.environ", {"CALIBRATE_STT_PARALLEL": "7"}):
            self.assertEqual(resolve_row_parallel("stt", 3), 3)

    def test_env_used_when_no_cli_value(self):
        from calibrate.utils import resolve_row_parallel

        with patch.dict("os.environ", {"CALIBRATE_STT_PARALLEL": "6"}):
            self.assertEqual(resolve_row_parallel("stt", None), 6)

    def test_default_when_no_cli_or_env(self):
        from calibrate.utils import resolve_row_parallel, DEFAULT_ROW_PARALLEL

        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("CALIBRATE_STT_PARALLEL", None)
            self.assertEqual(
                resolve_row_parallel("stt", None), DEFAULT_ROW_PARALLEL
            )

    def test_non_positive_and_garbage_fall_back_to_default(self):
        from calibrate.utils import resolve_row_parallel, DEFAULT_ROW_PARALLEL

        # Non-positive CLI values are ignored -> fall through to env/default.
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("CALIBRATE_STT_PARALLEL", None)
            self.assertEqual(
                resolve_row_parallel("stt", 0), DEFAULT_ROW_PARALLEL
            )
            self.assertEqual(
                resolve_row_parallel("stt", -5), DEFAULT_ROW_PARALLEL
            )

        # Garbage / non-positive env values are ignored too.
        for bad in ("abc", "0", "-1"):
            with patch.dict("os.environ", {"CALIBRATE_STT_PARALLEL": bad}):
                self.assertEqual(
                    resolve_row_parallel("stt", None), DEFAULT_ROW_PARALLEL
                )


if __name__ == "__main__":
    unittest.main()
