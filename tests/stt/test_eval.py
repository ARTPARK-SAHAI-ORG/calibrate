"""
Tests for calibrate_agent/stt/eval.py — routers, validators, and result writers.

Run with:
    python -m unittest tests.stt.test_eval -v
"""

import json
import os
import tempfile
import base64
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
        from calibrate_agent.stt.eval import validate_stt_input_dir

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_valid_layout(base, ["a", "b"])
            ok, err = validate_stt_input_dir(str(base), "stt.csv")
            self.assertTrue(ok, err)
            self.assertEqual(err, "")

    def test_missing_directory(self):
        from calibrate_agent.stt.eval import validate_stt_input_dir

        ok, err = validate_stt_input_dir("/nonexistent/path/xyz", "stt.csv")
        self.assertFalse(ok)
        self.assertIn("does not exist", err)

    def test_missing_csv(self):
        from calibrate_agent.stt.eval import validate_stt_input_dir

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            ok, err = validate_stt_input_dir(str(base), "stt.csv")
            self.assertFalse(ok)
            self.assertIn("CSV file not found", err)

    def test_missing_audios_dir(self):
        from calibrate_agent.stt.eval import validate_stt_input_dir

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(
                base / "stt.csv", index=False
            )
            ok, err = validate_stt_input_dir(str(base), "stt.csv")
            self.assertFalse(ok)
            self.assertIn("Audios directory not found", err)

    def test_missing_required_columns(self):
        from calibrate_agent.stt.eval import validate_stt_input_dir

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
        from calibrate_agent.stt.eval import validate_stt_input_dir

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
        from calibrate_agent.stt.eval import validate_existing_results_csv

        ok, err = validate_existing_results_csv("/nonexistent.csv")
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_empty_is_valid(self):
        from calibrate_agent.stt.eval import validate_existing_results_csv

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            pd.DataFrame(columns=["id", "gt", "pred"]).to_csv(f.name, index=False)
            path = f.name
        try:
            ok, err = validate_existing_results_csv(path)
            self.assertTrue(ok, err)
        finally:
            os.remove(path)

    def test_valid_columns(self):
        from calibrate_agent.stt.eval import validate_existing_results_csv

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
        from calibrate_agent.stt.eval import validate_existing_results_csv

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
        from calibrate_agent.stt.eval import validate_stt_eval_only_dataset

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
        from calibrate_agent.stt.eval import validate_stt_eval_only_dataset

        ok, err, rows = validate_stt_eval_only_dataset("/nope.json")
        self.assertFalse(ok)
        self.assertEqual(rows, [])

    def test_not_a_list(self):
        from calibrate_agent.stt.eval import validate_stt_eval_only_dataset

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
        from calibrate_agent.stt.eval import validate_stt_eval_only_dataset

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
        from calibrate_agent.stt import eval as stt_eval

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
        from calibrate_agent.stt import eval as stt_eval

        fake = AsyncMock(return_value={"transcript": "  hello  "})
        with patch.dict(
            "os.environ", {"DEEPGRAM_API_KEY": "x"}
        ), patch.object(stt_eval, "transcribe_deepgram_streaming", fake):
            output = await stt_eval.transcribe_audio.__wrapped__(
                Path("/tmp/x.wav"),
                "ref",
                "deepgram",
                "english",
                "uid",
            )
        self.assertEqual(output["transcript"], "hello")
        fake.assert_awaited_once()


def _fake_intent_entity(intent=1, entity=1.0):
    """Build a fake ``get_intent_entity_score`` returning fixed scores per row."""

    async def _fn(refs, preds, language="english", model=None):
        return {
            "intent": float(intent),
            "entity": float(entity),
            "per_row": [
                {
                    "intent_score": intent,
                    "intent_explanation": "ok",
                    "entity_score": entity,
                    "ground_truth_entities": "NA",
                    "preserved_entities": "NA",
                    "missing_entities": "",
                    "entity_explanation": "ok",
                }
                for _ in refs
            ],
        }

    return AsyncMock(side_effect=_fn)


def _fake_llm_wer(llm_wer=0.05, llm_cer=0.03):
    """Build a fake ``get_llm_wer_cer_score`` returning fixed scores per row."""

    async def _fn(refs, preds, language="english", model=None):
        return {
            "llm_wer": float(llm_wer),
            "llm_cer": float(llm_cer),
            "per_row": [
                {
                    "llm_wer": float(llm_wer),
                    "llm_cer": float(llm_cer),
                    "segments": [],
                }
                for _ in refs
            ],
        }

    return AsyncMock(side_effect=_fn)


def _fake_semantic_wer():
    """Fake ``get_semantic_wer_score`` (part of the default LLM-judge group)."""

    async def _sem(references, predictions, model=None):
        return {
            "semantic_wer": 0.0,
            "per_row": [
                {
                    "semantic_wer": 0.0, "substitutions": 0, "deletions": 0,
                    "insertions": 0, "reference_words": 1,
                    "normalized_reference": "", "normalized_hypothesis": "",
                    "reasoning": "",
                }
                for _ in references
            ],
        }

    return AsyncMock(side_effect=_sem)


class TestSTTScoreAndWriteResults(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from calibrate_agent.stt import eval as stt_eval

        p = patch.object(stt_eval, "get_semantic_wer_score", _fake_semantic_wer())
        p.start()
        self.addCleanup(p.stop)

    async def test_writes_metrics_and_results(self):
        from calibrate_agent.stt import eval as stt_eval

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None):
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
            ), patch.object(
                stt_eval, "get_intent_entity_score", _fake_intent_entity(1, 1.0)
            ), patch.object(
                stt_eval, "get_llm_wer_cer_score", _fake_llm_wer(0.05, 0.03)
            ):
                metrics = await stt_eval._score_and_write_results(
                    ids=["1", "2"],
                    gt_transcripts=["hello", "world"],
                    pred_transcripts=["hello", "world"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    judge_evaluators=[
                        {
                            "name": "semantic_match",
                            "system_prompt": "match",
                            "judge_model": "openai/gpt-4.1",
                        }
                    ],
                    run_llm_judges=True,
                )

            self.assertIn("wer", metrics)
            self.assertIn("semantic_match", metrics)
            # Intent + entity are reported as top-level floats when enabled.
            self.assertEqual(metrics["sarvam_intent_score"], 1.0)
            self.assertEqual(metrics["sarvam_entity_score"], 1.0)
            # LLM-WER/CER likewise.
            self.assertEqual(metrics["sarvam_llm_wer"], 0.05)
            self.assertEqual(metrics["sarvam_llm_cer"], 0.03)
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
                    "sarvam_intent_score",
                    "sarvam_intent_reasoning",
                    "sarvam_entity_score",
                    "sarvam_entity_reasoning",
                    "sarvam_llm_wer",
                    "sarvam_llm_cer",
                    "sarvam_llm_wer_reasoning",
                    "semantic_match",
                    "semantic_match_reasoning",
                }
            )
            self.assertEqual(len(df), 2)

    async def test_intent_entity_present_with_custom_evaluators_when_enabled(self):
        from calibrate_agent.stt import eval as stt_eval

        custom = [
            {
                "name": "completeness",
                "system_prompt": "nothing missing",
                "judge_model": "openai/gpt-4.1",
            }
        ]

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None):
            return {
                "scores": {"completeness": {"type": "binary", "mean": 0.5}},
                "score": 0.5,
                "per_row": [
                    {"completeness": {"match": True, "reasoning": "ok"}} for _ in refs
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(
                stt_eval, "get_llm_judge_score", AsyncMock(side_effect=fake_judge)
            ), patch.object(
                stt_eval, "get_intent_entity_score", _fake_intent_entity(0, 0.25)
            ), patch.object(
                stt_eval, "get_llm_wer_cer_score", _fake_llm_wer(0.4, 0.3)
            ):
                metrics = await stt_eval._score_and_write_results(
                    ids=["a"],
                    gt_transcripts=["hi"],
                    pred_transcripts=["bye"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    judge_evaluators=custom,
                    run_llm_judges=True,
                )

            # With the flag on, intent/entity report alongside a custom evaluator.
            self.assertEqual(metrics["sarvam_intent_score"], 0.0)
            self.assertEqual(metrics["sarvam_entity_score"], 0.25)
            self.assertEqual(metrics["sarvam_llm_wer"], 0.4)
            self.assertEqual(metrics["sarvam_llm_cer"], 0.3)
            self.assertIn("completeness", metrics)
            df = pd.read_csv(out / "results.csv")
            self.assertEqual(df.iloc[0]["sarvam_intent_score"], 0)
            self.assertEqual(df.iloc[0]["sarvam_entity_score"], 0.25)
            self.assertEqual(df.iloc[0]["sarvam_llm_wer"], 0.4)

    async def test_sarvam_judges_run_by_default(self):
        from calibrate_agent.stt import eval as stt_eval

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None):
            return {
                "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
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
            ), patch.object(
                stt_eval, "get_intent_entity_score", _fake_intent_entity(1, 1.0)
            ), patch.object(
                stt_eval, "get_llm_wer_cer_score", _fake_llm_wer(0.05, 0.03)
            ):
                metrics = await stt_eval._score_and_write_results(
                    ids=["1", "2"],
                    gt_transcripts=["hello", "world"],
                    pred_transcripts=["hello", "world"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    judge_evaluators=[
                        {
                            "name": "semantic_match",
                            "system_prompt": "match",
                            "judge_model": "openai/gpt-4.1",
                        }
                    ],
                )

            # Default: the Sarvam judges run and their metrics/columns appear.
            self.assertIn("sarvam_intent_score", metrics)
            self.assertIn("sarvam_entity_score", metrics)
            self.assertIn("sarvam_llm_wer", metrics)
            df = pd.read_csv(out / "results.csv")
            self.assertIn("sarvam_intent_score", df.columns)
            self.assertIn("sarvam_entity_score", df.columns)
            self.assertIn("wer", metrics)
            self.assertIn("semantic_match", metrics)

    async def test_sarvam_judges_skipped_when_disabled(self):
        from calibrate_agent.stt import eval as stt_eval

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None):
            return {
                "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                "score": 1.0,
                "per_row": [
                    {"semantic_match": {"match": True, "reasoning": "ok"}}
                    for _ in refs
                ],
            }

        ie_mock = AsyncMock()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(
                stt_eval, "get_llm_judge_score", AsyncMock(side_effect=fake_judge)
            ), patch.object(stt_eval, "get_intent_entity_score", ie_mock):
                metrics = await stt_eval._score_and_write_results(
                    ids=["1", "2"],
                    gt_transcripts=["hello", "world"],
                    pred_transcripts=["hello", "world"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    judge_evaluators=[
                        {
                            "name": "semantic_match",
                            "system_prompt": "match",
                            "judge_model": "openai/gpt-4.1",
                        }
                    ],
                    run_llm_judges=False,
                )

            # Disabled: the Sarvam judge is never invoked and no sarvam_* keys appear.
            ie_mock.assert_not_awaited()
            self.assertNotIn("sarvam_intent_score", metrics)
            self.assertNotIn("sarvam_entity_score", metrics)
            df = pd.read_csv(out / "results.csv")
            self.assertNotIn("sarvam_intent_score", df.columns)
            self.assertNotIn("sarvam_entity_score", df.columns)
            # WER/CER and the judge column are still written.
            self.assertIn("wer", metrics)
            self.assertIn("semantic_match", metrics)

    async def test_partial_results_written_while_judge_still_running(self):
        import asyncio

        from calibrate_agent.stt import eval as stt_eval
        from calibrate_agent.stt import metrics as stt_metrics

        seen_by_second_row = {}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_path = out / "results.csv"

            async def fake_row_judge(
                reference, prediction, evaluators=None, fallback_model=None
            ):
                if reference == "second":
                    # Wait for the first row's result to land on disk, then
                    # record what the file held mid-run.
                    for _ in range(200):
                        if results_path.exists():
                            seen_by_second_row["csv"] = results_path.read_text()
                            break
                        await asyncio.sleep(0.01)
                return {"semantic_match": {"match": True, "reasoning": reference}}

            with patch.object(
                stt_metrics, "stt_llm_judge", AsyncMock(side_effect=fake_row_judge)
            ), patch.object(
                stt_eval, "get_intent_entity_score", _fake_intent_entity(1, 1.0)
            ), patch.object(
                stt_eval, "get_llm_wer_cer_score", _fake_llm_wer(0.05, 0.03)
            ):
                await stt_eval._score_and_write_results(
                    ids=["row_a", "row_b"],
                    gt_transcripts=["first", "second"],
                    pred_transcripts=["first", "second"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    judge_evaluators=[
                        {
                            "name": "semantic_match",
                            "system_prompt": "match",
                            "judge_model": "openai/gpt-4.1",
                        }
                    ],
                    run_llm_judges=True,
                    stream_rows=True,
                )

            partial = seen_by_second_row.get("csv")
            self.assertIsNotNone(
                partial, "results.csv was not written before the judge finished"
            )
            self.assertIn("row_a", partial)
            self.assertIn("first", partial)

            # The final file replaces the partial one and keeps its full shape.
            df = pd.read_csv(results_path)
            self.assertEqual(len(df), 2)
            self.assertTrue(
                set(df.columns)
                >= {
                    "id",
                    "gt",
                    "pred",
                    "wer",
                    "cer",
                    "semantic_match",
                    "semantic_match_reasoning",
                }
            )
            self.assertEqual(sorted(df["id"]), ["row_a", "row_b"])

    async def test_no_partial_results_without_stream_rows(self):
        """The full evaluation keeps its own transcriptions in results.csv and
        resumes from them, so the judge must not overwrite that file mid-run."""
        import asyncio

        from calibrate_agent.stt import eval as stt_eval
        from calibrate_agent.stt import metrics as stt_metrics

        seen_by_second_row = {}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results_path = out / "results.csv"
            results_path.write_text("id,gt,pred\nrow_a,first,first\n")

            async def fake_row_judge(
                reference, prediction, evaluators=None, fallback_model=None
            ):
                if reference == "second":
                    await asyncio.sleep(0.05)
                    seen_by_second_row["csv"] = results_path.read_text()
                return {"semantic_match": {"match": True, "reasoning": reference}}

            with patch.object(
                stt_metrics, "stt_llm_judge", AsyncMock(side_effect=fake_row_judge)
            ), patch.object(
                stt_eval, "get_intent_entity_score", _fake_intent_entity(1, 1.0)
            ), patch.object(
                stt_eval, "get_llm_wer_cer_score", _fake_llm_wer(0.05, 0.03)
            ):
                await stt_eval._score_and_write_results(
                    ids=["row_a", "row_b"],
                    gt_transcripts=["first", "second"],
                    pred_transcripts=["first", "second"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    judge_evaluators=[
                        {
                            "name": "semantic_match",
                            "system_prompt": "match",
                            "judge_model": "openai/gpt-4.1",
                        }
                    ],
                    run_llm_judges=True,
                )

            self.assertEqual(
                seen_by_second_row.get("csv"),
                "id,gt,pred\nrow_a,first,first\n",
            )

    async def test_llm_judge_failure_still_writes_wer_cer_and_sarvam(self):
        from calibrate_agent.stt import eval as stt_eval

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(
                stt_eval,
                "get_llm_judge_score",
                AsyncMock(side_effect=RuntimeError("judge boom")),
            ), patch.object(
                stt_eval, "get_intent_entity_score", _fake_intent_entity(1, 1.0)
            ), patch.object(
                stt_eval, "get_llm_wer_cer_score", _fake_llm_wer(0.05, 0.03)
            ):
                metrics = await stt_eval._score_and_write_results(
                    ids=["1", "2"],
                    gt_transcripts=["hello", "world"],
                    pred_transcripts=["hello", "world"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    judge_evaluators=[
                        {
                            "name": "semantic_match",
                            "system_prompt": "match",
                            "judge_model": "openai/gpt-4.1",
                        }
                    ],
                )

            # LLM judge failed: WER/CER and the Sarvam judges survive; the
            # evaluator columns/metrics are dropped and nothing crashes.
            self.assertIn("wer", metrics)
            self.assertIn("cer", metrics)
            self.assertIn("sarvam_intent_score", metrics)
            self.assertIn("sarvam_llm_wer", metrics)
            self.assertNotIn("semantic_match", metrics)
            df = pd.read_csv(out / "results.csv")
            self.assertNotIn("semantic_match", df.columns)
            self.assertIn("sarvam_intent_score", df.columns)
            self.assertEqual(len(df), 2)

    async def test_sarvam_failure_still_writes_wer_cer_and_evaluator(self):
        from calibrate_agent.stt import eval as stt_eval

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None):
            return {
                "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
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
            ), patch.object(
                stt_eval,
                "get_intent_entity_score",
                AsyncMock(side_effect=RuntimeError("intent boom")),
            ), patch.object(
                stt_eval, "get_llm_wer_cer_score", _fake_llm_wer(0.05, 0.03)
            ):
                metrics = await stt_eval._score_and_write_results(
                    ids=["1", "2"],
                    gt_transcripts=["hello", "world"],
                    pred_transcripts=["hello", "world"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    judge_evaluators=[
                        {
                            "name": "semantic_match",
                            "system_prompt": "match",
                            "judge_model": "openai/gpt-4.1",
                        }
                    ],
                )

            # One Sarvam judge failed: WER/CER, the evaluator, and the other
            # Sarvam judge (LLM-WER/CER) all survive; only intent/entity drops.
            self.assertIn("wer", metrics)
            self.assertIn("semantic_match", metrics)
            self.assertIn("sarvam_llm_wer", metrics)
            self.assertNotIn("sarvam_intent_score", metrics)
            df = pd.read_csv(out / "results.csv")
            self.assertIn("semantic_match", df.columns)
            self.assertNotIn("sarvam_intent_score", df.columns)
            self.assertIn("sarvam_llm_wer", df.columns)

    async def test_no_llm_judge_when_no_evaluators(self):
        from calibrate_agent.stt import eval as stt_eval

        judge_mock = AsyncMock()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(stt_eval, "get_llm_judge_score", judge_mock):
                metrics = await stt_eval._score_and_write_results(
                    ids=["1", "2"],
                    gt_transcripts=["hello", "world"],
                    pred_transcripts=["hello", "goodbye"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    run_llm_judges=False,
                )

            # No evaluators passed: the LLM judge is never invoked, no evaluator
            # config is written, and only WER/CER are reported.
            judge_mock.assert_not_awaited()
            self.assertIn("wer", metrics)
            self.assertIn("cer", metrics)
            self.assertNotIn("semantic_match", metrics)
            self.assertFalse((out / "config.json").exists())
            df = pd.read_csv(out / "results.csv")
            self.assertEqual(set(df.columns), {"id", "gt", "pred", "wer", "cer"})
            self.assertEqual(len(df), 2)

    async def test_rating_evaluator_writes_numeric_score(self):
        from calibrate_agent.stt import eval as stt_eval

        rating = {
            "name": "accuracy",
            "system_prompt": "rate accuracy",
            "judge_model": "openai/gpt-4.1",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
        }

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None):
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
            ), patch.object(
                stt_eval, "get_intent_entity_score", _fake_intent_entity(1, 1.0)
            ):
                await stt_eval._score_and_write_results(
                    ids=["1"],
                    gt_transcripts=["hi"],
                    pred_transcripts=["hi"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    judge_evaluators=[rating],
                    run_llm_judges=False,
                )
            df = pd.read_csv(out / "results.csv")
            self.assertEqual(df.iloc[0]["accuracy"], 4)


class TestSTTRunEvalOnly(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from calibrate_agent.stt import eval as stt_eval

        p = patch.object(stt_eval, "get_semantic_wer_score", _fake_semantic_wer())
        p.start()
        self.addCleanup(p.stop)

    async def test_runs_evaluator_on_dataset(self):
        from calibrate_agent.stt import eval as stt_eval

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None):
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
            ), patch.object(
                stt_eval, "get_intent_entity_score", _fake_intent_entity(1, 1.0)
            ):
                result = await stt_eval.run_eval_only(
                    dataset_path=str(ds_path),
                    output_dir=str(out),
                    run_llm_judges=False,
                )

            self.assertEqual(result["status"], "completed")
            self.assertTrue((out / "metrics.json").exists())
            self.assertTrue((out / "results.csv").exists())

    async def test_rows_land_in_results_csv_while_the_run_is_going(self):
        """Guards the whole eval-only run, not just the scoring step it calls:
        this fails if ``run_eval_only`` stops asking for row-by-row writing."""
        import asyncio

        from calibrate_agent.stt import eval as stt_eval
        from calibrate_agent.stt import metrics as stt_metrics

        seen = {}

        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp) / "ds.json"
            ds_path.write_text(
                json.dumps(
                    [
                        {"id": "row_a", "gt": "first", "pred": "first"},
                        {"id": "row_b", "gt": "second", "pred": "second"},
                    ]
                )
            )
            out = Path(tmp) / "out"
            results_path = out / "results.csv"

            async def fake_row_judge(
                reference, prediction, evaluators=None, fallback_model=None
            ):
                if reference == "second":
                    for _ in range(200):
                        if results_path.exists():
                            seen["csv"] = results_path.read_text()
                            break
                        await asyncio.sleep(0.01)
                return {"semantic_match": {"match": True, "reasoning": reference}}

            with patch.object(
                stt_metrics, "stt_llm_judge", AsyncMock(side_effect=fake_row_judge)
            ):
                result = await stt_eval.run_eval_only(
                    dataset_path=str(ds_path),
                    output_dir=str(out),
                    judge_evaluators=[
                        {
                            "name": "semantic_match",
                            "system_prompt": "match",
                            "judge_model": "openai/gpt-4.1",
                        }
                    ],
                    run_llm_judges=False,
                )

            self.assertEqual(result["status"], "completed")
            self.assertIn("row_a", seen.get("csv", ""))
            self.assertEqual(sorted(pd.read_csv(results_path)["id"]), ["row_a", "row_b"])

    async def test_invalid_dataset_returns_error(self):
        from calibrate_agent.stt import eval as stt_eval

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
        from calibrate_agent.stt import eval as stt_eval

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
        from calibrate_agent.stt import eval as stt_eval

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
        from calibrate_agent.stt import eval as stt_eval

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


class TestTranscribeOpenAIStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_passes_language_code_to_api(self):
        from calibrate_agent.stt import eval as stt_eval

        done_event = SimpleNamespace(type="transcript.text.done", text="hello world")

        async def _fake_stream():
            yield done_event

        create = AsyncMock(return_value=_fake_stream())
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create = create

        patches = (
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"}),
            patch.object(stt_eval, "AsyncOpenAI", return_value=fake_client),
            patch.object(stt_eval, "load_audio", return_value=SimpleNamespace()),
        )
        for p in patches:
            p.start()
        try:
            result = await stt_eval.transcribe_openai_streaming(
                Path("/tmp/x.wav"), "hindi"
            )
        finally:
            for p in patches:
                p.stop()

        self.assertEqual(result["transcript"], "hello world")
        # OpenAI STT expects an ISO-639-1 code — "hi" for hindi.
        self.assertEqual(create.await_args.kwargs["language"], "hi")
        # The gpt-4o models treat `language` as a soft hint, so we also steer
        # the output language via `prompt` to stop it emitting the wrong script.
        self.assertEqual(
            create.await_args.kwargs["prompt"],
            "Transcribe the audio in Hindi.",
        )


class TestTranscribeGemini(unittest.IsolatedAsyncioTestCase):
    async def test_returns_stripped_transcript_and_uses_model(self):
        from calibrate_agent.stt import eval as stt_eval

        generate = AsyncMock(return_value=SimpleNamespace(text="  hola mundo  "))
        fake_client = MagicMock()
        fake_client.aio.models.generate_content = generate

        patches = (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-fake"}),
            patch.object(stt_eval.genai, "Client", return_value=fake_client),
            patch.object(stt_eval, "load_audio", return_value=b"RIFFfake"),
        )
        for p in patches:
            p.start()
        try:
            result = await stt_eval.transcribe_gemini(Path("/tmp/x.wav"), "hindi")
        finally:
            for p in patches:
                p.stop()

        self.assertEqual(result["transcript"], "hola mundo")
        self.assertEqual(
            generate.await_args.kwargs["model"],
            stt_eval.STT_PROVIDER_MODELS["gemini"],
        )

    async def test_none_text_yields_empty_transcript(self):
        from calibrate_agent.stt import eval as stt_eval

        generate = AsyncMock(return_value=SimpleNamespace(text=None))
        fake_client = MagicMock()
        fake_client.aio.models.generate_content = generate

        patches = (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-fake"}),
            patch.object(stt_eval.genai, "Client", return_value=fake_client),
            patch.object(stt_eval, "load_audio", return_value=b"RIFFfake"),
        )
        for p in patches:
            p.start()
        try:
            result = await stt_eval.transcribe_gemini(Path("/tmp/x.wav"), "english")
        finally:
            for p in patches:
                p.stop()

        self.assertEqual(result["transcript"], "")

    async def test_missing_key_raises(self):
        from calibrate_agent.stt import eval as stt_eval

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                await stt_eval.transcribe_gemini(Path("/tmp/x.wav"), "english")


async def _run_batch(
    stt_eval, fn_name, language, *patches, env=None, audio=b"RIFFfake"
):
    """Call a batch transcribe function with ``load_audio`` and env patched."""
    patches = (
        patch.dict("os.environ", env or {}),
        patch.object(stt_eval, "load_audio", return_value=audio),
        *patches,
    )
    for p in patches:
        p.start()
    try:
        return await getattr(stt_eval, fn_name)(Path("/tmp/x.wav"), language)
    finally:
        for p in patches:
            p.stop()


class TestTranscribeBatchProviders(unittest.IsolatedAsyncioTestCase):
    async def test_deepgram_sends_whole_file_with_model_and_language(self):
        from calibrate_agent.stt import eval as stt_eval

        response = SimpleNamespace(
            results=SimpleNamespace(
                channels=[
                    SimpleNamespace(
                        alternatives=[SimpleNamespace(transcript=" namaste ")]
                    )
                ]
            )
        )
        transcribe_file = AsyncMock(return_value=response)
        fake_client = MagicMock()
        fake_client.listen.v1.media.transcribe_file = transcribe_file

        result = await _run_batch(
            stt_eval,
            "transcribe_deepgram_batch",
            "hindi",
            patch.object(stt_eval, "AsyncDeepgramClient", return_value=fake_client),
            env={"DEEPGRAM_API_KEY": "dg-fake"},
        )

        self.assertEqual(result, {"transcript": "namaste"})
        kwargs = transcribe_file.await_args.kwargs
        self.assertEqual(kwargs["request"], b"RIFFfake")
        self.assertEqual(kwargs["model"], stt_eval.STT_BATCH_PROVIDER_MODELS["deepgram"])
        self.assertEqual(kwargs["language"], "hi")

    async def test_openai_is_not_streamed(self):
        from calibrate_agent.stt import eval as stt_eval

        create = AsyncMock(return_value=" hello world \n")
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create = create

        result = await _run_batch(
            stt_eval,
            "transcribe_openai_batch",
            "hindi",
            patch.object(stt_eval, "AsyncOpenAI", return_value=fake_client),
            env={"OPENAI_API_KEY": "sk-fake"},
        )

        self.assertEqual(result, {"transcript": "hello world"})
        kwargs = create.await_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-transcribe")
        self.assertEqual(kwargs["language"], "hi")
        self.assertEqual(kwargs["prompt"], "Transcribe the audio in Hindi.")
        self.assertNotIn("stream", kwargs)

    async def test_google_recognize_joins_results(self):
        from calibrate_agent.stt import eval as stt_eval

        def _result(text):
            return SimpleNamespace(alternatives=[SimpleNamespace(transcript=text)])

        fake_client = MagicMock()
        fake_client.recognize.return_value = SimpleNamespace(
            results=[_result(" hello "), SimpleNamespace(alternatives=[]), _result("world")]
        )

        result = await _run_batch(
            stt_eval,
            "transcribe_google_batch",
            "sindhi",
            patch.object(stt_eval, "SpeechClient", return_value=fake_client),
            env={
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/creds.json",
                "GOOGLE_CLOUD_PROJECT_ID": "proj",
            },
            audio=b"\x00" * 32000,
        )

        self.assertEqual(result, {"transcript": "hello world"})
        request = fake_client.recognize.call_args.kwargs["request"]
        self.assertTrue(request.content.startswith(b"RIFF"))
        # Sindhi is served by chirp_2 in asia-southeast1.
        self.assertEqual(request.config.model, "chirp_2")
        self.assertEqual(
            request.recognizer,
            "projects/proj/locations/asia-southeast1/recognizers/_",
        )

    async def test_google_splits_clips_longer_than_60_seconds(self):
        import io
        import wave

        from calibrate_agent.stt import eval as stt_eval

        fake_client = MagicMock()
        fake_client.recognize.side_effect = [
            SimpleNamespace(
                results=[SimpleNamespace(alternatives=[SimpleNamespace(transcript=t)])]
            )
            for t in ("first", "second")
        ]

        result = await _run_batch(
            stt_eval,
            "transcribe_google_batch",
            "hindi",
            patch.object(stt_eval, "SpeechClient", return_value=fake_client),
            env={
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/creds.json",
                "GOOGLE_CLOUD_PROJECT_ID": "proj",
            },
            audio=b"\x00" * (61 * 32000),
        )

        self.assertEqual(result, {"transcript": "first second"})
        durations = []
        for call in fake_client.recognize.call_args_list:
            with wave.open(io.BytesIO(call.kwargs["request"].content)) as w:
                durations.append(w.getnframes() / w.getframerate())
        self.assertEqual(durations, [60.0, 1.0])

    async def test_google_empty_audio_is_empty_transcript(self):
        from calibrate_agent.stt import eval as stt_eval

        speech_client = MagicMock()
        result = await _run_batch(
            stt_eval,
            "transcribe_google_batch",
            "hindi",
            patch.object(stt_eval, "SpeechClient", speech_client),
            env={
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/creds.json",
                "GOOGLE_CLOUD_PROJECT_ID": "proj",
            },
            audio=b"",
        )
        self.assertEqual(result, {"transcript": ""})
        speech_client.assert_not_called()

    async def test_google_missing_project_id_raises(self):
        from calibrate_agent.stt import eval as stt_eval

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GOOGLE_CLOUD_PROJECT_ID"):
                await _run_batch(
                    stt_eval,
                    "transcribe_google_batch",
                    "hindi",
                    env={"GOOGLE_APPLICATION_CREDENTIALS": "/tmp/creds.json"},
                )

    async def _gemini(self, language):
        from calibrate_agent.stt import eval as stt_eval

        response = MagicMock()
        response.json.return_value = {"text": "  salam  ", "usage": {"seconds": 1}}
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=response)
        http_client.__aenter__ = AsyncMock(return_value=http_client)
        http_client.__aexit__ = AsyncMock(return_value=False)

        chat_client = MagicMock()
        chat_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=" fallback text "))
                ]
            )
        )
        result = await _run_batch(
            stt_eval,
            "transcribe_gemini_batch",
            language,
            patch.object(stt_eval.httpx, "AsyncClient", return_value=http_client),
            patch.object(stt_eval, "_build_openrouter_client", return_value=chat_client),
            env={"OPENROUTER_API_KEY": "or-fake"},
        )
        return result, http_client.post, chat_client.chat.completions.create, response

    async def test_gemini_uses_transcribe_model_through_openrouter(self):
        from calibrate_agent.stt import eval as stt_eval

        result, post, chat, response = await self._gemini("sindhi")

        self.assertEqual(result, {"transcript": "salam"})
        self.assertEqual(
            post.await_args.args[0],
            "https://openrouter.ai/api/v1/audio/transcriptions",
        )
        kwargs = post.await_args.kwargs
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer or-fake")
        self.assertEqual(
            kwargs["json"],
            {
                "model": f"google/{stt_eval.STT_BATCH_PROVIDER_MODELS['gemini']}",
                "input_audio": {
                    "data": base64.b64encode(b"RIFFfake").decode("utf-8"),
                    "format": "wav",
                },
                "language": "sd",
            },
        )
        response.raise_for_status.assert_called_once()
        chat.assert_not_awaited()

    async def test_gemini_unsupported_languages_use_multimodal_model(self):
        from calibrate_agent.stt import eval as stt_eval

        for language in ("tamil", "maithili"):
            with self.subTest(language=language):
                result, post, chat, _ = await self._gemini(language)
                self.assertEqual(result, {"transcript": "fallback text"})
                post.assert_not_awaited()
                kwargs = chat.await_args.kwargs
                self.assertEqual(
                    kwargs["model"], f"google/{stt_eval.STT_PROVIDER_MODELS['gemini']}"
                )
                content = kwargs["messages"][0]["content"]
                self.assertIn(language, content[0]["text"])
                self.assertEqual(content[1]["input_audio"]["format"], "wav")

    async def test_gemini_flash_sends_audio_as_chat_completion(self):
        from calibrate_agent.stt import eval as stt_eval

        chat_client = MagicMock()
        chat_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=" namaste "))]
            )
        )
        result = await _run_batch(
            stt_eval,
            "transcribe_gemini_flash_batch",
            "hindi",
            patch.object(stt_eval, "_build_openrouter_client", return_value=chat_client),
            env={"OPENROUTER_API_KEY": "or-fake"},
        )

        self.assertEqual(result, {"transcript": "namaste"})
        kwargs = chat_client.chat.completions.create.await_args.kwargs
        self.assertEqual(
            kwargs["model"],
            f"google/{stt_eval.STT_BATCH_PROVIDER_MODELS['gemini-flash']}",
        )
        content = kwargs["messages"][0]["content"]
        self.assertIn("hindi (hi-IN)", content[0]["text"])
        self.assertEqual(
            content[1]["input_audio"],
            {"data": base64.b64encode(b"RIFFfake").decode("utf-8"), "format": "wav"},
        )

    async def test_gemini_flash_empty_content_is_empty_transcript(self):
        from calibrate_agent.stt import eval as stt_eval

        chat_client = MagicMock()
        chat_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
            )
        )
        result = await _run_batch(
            stt_eval,
            "transcribe_gemini_flash_batch",
            "english",
            patch.object(stt_eval, "_build_openrouter_client", return_value=chat_client),
            env={"OPENROUTER_API_KEY": "or-fake"},
        )
        self.assertEqual(result, {"transcript": ""})

    async def _sarvam(self, duration_seconds, job=None):
        import io
        import wave

        from calibrate_agent.stt import eval as stt_eval

        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00" * int(duration_seconds * 32000))

        fake_client = MagicMock()
        fake_client.speech_to_text.transcribe = AsyncMock(
            return_value=SimpleNamespace(transcript=" नमस्ते ")
        )
        fake_client.speech_to_text_job.create_job = AsyncMock(return_value=job)
        result = await _run_batch(
            stt_eval,
            "transcribe_sarvam_batch",
            "hindi",
            patch.object(stt_eval, "AsyncSarvamAI", return_value=fake_client),
            env={"SARVAM_API_KEY": "sk-fake"},
            audio=wav.getvalue(),
        )
        return result, fake_client

    async def test_sarvam_short_clip_uses_rest(self):
        from calibrate_agent.stt import eval as stt_eval

        # A clip of exactly 30 s stays on the REST endpoint.
        result, client = await self._sarvam(30.0)

        self.assertEqual(result, {"transcript": "नमस्ते"})
        kwargs = client.speech_to_text.transcribe.await_args.kwargs
        self.assertEqual(kwargs["model"], stt_eval.STT_BATCH_PROVIDER_MODELS["sarvam"])
        self.assertEqual(kwargs["mode"], "transcribe")
        self.assertEqual(kwargs["language_code"], "hi-IN")
        client.speech_to_text_job.create_job.assert_not_awaited()

    def _sarvam_job(self, state="Completed", transcript="लंबा ऑडियो"):
        job = MagicMock()
        job.job_id = "job-1"
        job.upload_files = AsyncMock()
        job.start = AsyncMock()
        job.wait_until_complete = AsyncMock(
            return_value=SimpleNamespace(job_state=state)
        )

        async def _download(output_dir):
            if transcript is not None:
                with open(os.path.join(output_dir, "audio.wav.json"), "w") as f:
                    json.dump({"transcript": transcript}, f)

        job.download_outputs = AsyncMock(side_effect=_download)
        return job

    async def test_sarvam_long_clip_uses_job_api(self):
        from calibrate_agent.stt import eval as stt_eval

        job = self._sarvam_job()
        result, client = await self._sarvam(31.0, job=job)

        self.assertEqual(result, {"transcript": "लंबा ऑडियो"})
        client.speech_to_text.transcribe.assert_not_awaited()
        self.assertEqual(
            client.speech_to_text_job.create_job.await_args.kwargs,
            {
                "model": stt_eval.STT_BATCH_PROVIDER_MODELS["sarvam"],
                "mode": "transcribe",
                "language_code": "hi-IN",
            },
        )
        uploaded = job.upload_files.await_args.args[0][0]
        self.assertEqual(os.path.basename(uploaded), "audio.wav")
        # The temp directory holding the upload is removed afterwards.
        self.assertFalse(os.path.exists(uploaded))
        job.start.assert_awaited_once()

    async def test_sarvam_failed_job_raises(self):
        with self.assertRaises(RuntimeError):
            await self._sarvam(31.0, job=self._sarvam_job("Failed"))

    async def test_sarvam_job_without_output_raises(self):
        with self.assertRaises(RuntimeError):
            await self._sarvam(31.0, job=self._sarvam_job(transcript=None))

    async def test_sarvam_job_output_without_transcript_is_empty(self):
        job = self._sarvam_job()

        async def _download(output_dir):
            with open(os.path.join(output_dir, "audio.wav.json"), "w") as f:
                json.dump({}, f)

        job.download_outputs = AsyncMock(side_effect=_download)
        result, _ = await self._sarvam(31.0, job=job)
        self.assertEqual(result, {"transcript": ""})

    async def test_cartesia_closes_client(self):
        from calibrate_agent.stt import eval as stt_eval

        fake_client = MagicMock()
        fake_client.stt.transcribe = AsyncMock(
            return_value=SimpleNamespace(text=" hello ")
        )
        fake_client.close = AsyncMock()

        result = await _run_batch(
            stt_eval,
            "transcribe_cartesia_batch",
            "hindi",
            patch.object(stt_eval, "AsyncCartesia", return_value=fake_client),
            env={"CARTESIA_API_KEY": "ck-fake"},
        )

        self.assertEqual(result, {"transcript": "hello"})
        kwargs = fake_client.stt.transcribe.await_args.kwargs
        self.assertEqual(kwargs["model"], stt_eval.STT_BATCH_PROVIDER_MODELS["cartesia"])
        self.assertEqual(kwargs["language"], "hi")
        fake_client.close.assert_awaited_once()

    async def test_elevenlabs_uses_whole_file_model_without_event_tags(self):
        from calibrate_agent.stt import eval as stt_eval

        convert = AsyncMock(return_value=SimpleNamespace(text=" hello "))
        fake_client = MagicMock()
        fake_client.speech_to_text.convert = convert

        result = await _run_batch(
            stt_eval,
            "transcribe_elevenlabs_batch",
            "hindi",
            patch("elevenlabs.AsyncElevenLabs", return_value=fake_client),
            env={"ELEVENLABS_API_KEY": "el-fake"},
        )

        self.assertEqual(result, {"transcript": "hello"})
        kwargs = convert.await_args.kwargs
        self.assertEqual(kwargs["model_id"], "scribe_v2")
        self.assertEqual(kwargs["language_code"], "hin")
        self.assertIs(kwargs["tag_audio_events"], False)

    async def test_smallest_posts_raw_bytes(self):
        from calibrate_agent.stt import eval as stt_eval

        response = MagicMock()
        response.json.return_value = {"status": "success", "transcription": " hi "}
        post = AsyncMock(return_value=response)
        http_client = MagicMock()
        http_client.post = post
        http_client.__aenter__ = AsyncMock(return_value=http_client)
        http_client.__aexit__ = AsyncMock(return_value=False)

        result = await _run_batch(
            stt_eval,
            "transcribe_smallest_batch",
            "hindi",
            patch.object(stt_eval.httpx, "AsyncClient", return_value=http_client),
            env={"SMALLEST_API_KEY": "sm-fake"},
        )

        self.assertEqual(result, {"transcript": "hi"})
        self.assertEqual(post.await_args.args[0], stt_eval.SMALLEST_STT_BATCH_URL)
        kwargs = post.await_args.kwargs
        self.assertEqual(
            kwargs["params"],
            {"model": stt_eval.STT_BATCH_PROVIDER_MODELS["smallest"], "language": "hi"},
        )
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sm-fake")
        self.assertEqual(kwargs["content"], b"RIFFfake")
        response.raise_for_status.assert_called_once()

    async def test_missing_api_key_raises(self):
        from calibrate_agent.stt import eval as stt_eval

        for fn_name in (
            "transcribe_deepgram_batch",
            "transcribe_openai_batch",
            "transcribe_google_batch",
            "transcribe_gemini_batch",
            "transcribe_gemini_flash_batch",
            "transcribe_sarvam_batch",
            "transcribe_cartesia_batch",
            "transcribe_elevenlabs_batch",
            "transcribe_smallest_batch",
        ):
            with self.subTest(fn=fn_name), patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(ValueError):
                    await getattr(stt_eval, fn_name)(Path("/tmp/x.wav"), "english")


class TestTranscribeAudioBatchRouter(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_provider_raises(self):
        from calibrate_agent.stt import eval as stt_eval

        with self.assertRaises(ValueError):
            await stt_eval.transcribe_audio.__wrapped__(
                Path("/tmp/x.wav"), "ref", "no-such-provider", "english", "uid", "batch"
            )

    async def test_routes_to_whole_file_function_and_tags_engine(self):
        from calibrate_agent.stt import eval as stt_eval

        batch_fn = AsyncMock(return_value={"transcript": "  hello  "})
        streaming_fn = AsyncMock(return_value={"transcript": "wrong"})
        fake_lf = MagicMock()
        inner = stt_eval.transcribe_audio
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        with patch.object(stt_eval, "transcribe_deepgram_batch", batch_fn), patch.object(
            stt_eval, "transcribe_deepgram_streaming", streaming_fn
        ), patch.object(stt_eval, "langfuse_enabled", True), patch.object(
            stt_eval, "langfuse", fake_lf
        ), patch.object(stt_eval, "create_langfuse_audio_media", return_value=None):
            output = await inner(
                Path("/tmp/x.wav"), "ref", "deepgram", "english", "uid", "batch"
            )

        self.assertEqual(output, {"transcript": "hello"})
        batch_fn.assert_awaited_once()
        streaming_fn.assert_not_awaited()
        metadata = fake_lf.update_current_trace.call_args.kwargs["metadata"]
        self.assertEqual(metadata["engine"], "batch")

    async def test_groq_shares_the_direct_function(self):
        from calibrate_agent.stt import eval as stt_eval

        groq_fn = AsyncMock(return_value={"transcript": "hi"})
        with patch.object(stt_eval, "transcribe_groq", groq_fn):
            await stt_eval.transcribe_audio.__wrapped__(
                Path("/tmp/x.wav"), "ref", "groq", "english", "uid", "batch"
            )
        groq_fn.assert_awaited_once()


class TestBatchEngineCostModel(unittest.TestCase):
    def test_batch_engine_prices_the_model_it_runs(self):
        from calibrate_agent.stt.eval import _default_stt_model

        self.assertEqual(_default_stt_model("openai", "hindi", "batch"), "gpt-transcribe")
        self.assertEqual(_default_stt_model("sarvam", "hindi", "batch"), "saaras:v4")
        self.assertEqual(_default_stt_model("elevenlabs", "hindi", "batch"), "scribe_v2")
        self.assertEqual(
            _default_stt_model("gemini", "hindi", "batch"), "gemini-3.5-transcribe"
        )
        # Languages the transcription model does not serve run the direct model.
        self.assertEqual(
            _default_stt_model("gemini", "tamil", "batch"),
            _default_stt_model("gemini", "tamil", "direct"),
        )
        self.assertEqual(_default_stt_model("google", "sindhi", "batch"), "chirp_2")

    def test_other_engines_keep_their_models(self):
        from calibrate_agent.stt.eval import _default_stt_model
        from calibrate_agent.utils import STT_PROVIDER_MODELS

        for engine in ("direct", "pipeline"):
            for provider in ("openai", "sarvam", "elevenlabs", "gemini"):
                self.assertEqual(
                    _default_stt_model(provider, "hindi", engine),
                    STT_PROVIDER_MODELS[provider],
                )

    def test_cost_metrics_use_batch_price_and_skip_unpriced_models(self):
        from calibrate_agent.stt.eval import _build_stt_cost_metrics, _default_stt_model

        cost = _build_stt_cost_metrics(
            "openai", [60.0], model=_default_stt_model("openai", "hindi", "batch")
        )
        self.assertEqual(cost["pricing_model"], "gpt-transcribe")
        self.assertAlmostEqual(cost["cost_usd"], 0.0045)

        cost = _build_stt_cost_metrics(
            "elevenlabs", [60.0], model=_default_stt_model("elevenlabs", "hindi", "batch")
        )
        self.assertEqual(cost["pricing_model"], "scribe_v2")
        self.assertAlmostEqual(cost["cost_usd"], 0.22 / 60)

        # Gemini is billed per token, so it has no per-minute price.
        self.assertIsNone(
            _build_stt_cost_metrics(
                "gemini", [60.0], model=_default_stt_model("gemini", "hindi", "batch")
            )
        )


if __name__ == "__main__":
    unittest.main()
