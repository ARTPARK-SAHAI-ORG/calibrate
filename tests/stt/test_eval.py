"""
Tests for calibrate_agent/stt/eval.py — routers, validators, and result writers.

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

    async def _fn(refs, preds, language="english", model=None, **kwargs):
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

    async def _fn(refs, preds, language="english", model=None, **kwargs):
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

    async def _sem(references, predictions, model=None, **kwargs):
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

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None, **kwargs):
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

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None, **kwargs):
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

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None, **kwargs):
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

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None, **kwargs):
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

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None, **kwargs):
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

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None, **kwargs):
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

        async def fake_judge(refs, preds, evaluators=None, fallback_model=None, on_row=None, **kwargs):
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


_RESUME_ROWS = {
    "row_a": ("doctor ne bola", "daktar ne bola"),
    "row_b": ("hello world", "hello word"),
    "row_c": ("doctor ne bola", "daktar ne bola"),
    "row_d": ("good morning", "gud morning"),
}

# Per-reference semantic-WER counts: (substitutions, deletions, insertions,
# reference_words). Deliberately uneven so a pooled score differs from a mean.
_SEMANTIC_COUNTS = {
    "doctor ne bola": (1, 0, 0, 3),
    "hello world": (1, 0, 0, 2),
    "good morning": (2, 1, 0, 2),
}

# Equivalence verdicts per differing word segment.
_EQUIVALENT_SEGMENTS = {
    ("doctor", "daktar"): True,
    ("world", "word"): False,
    ("good", "gud"): True,
}


def _dataset(path, row_ids):
    """Write a ``run_eval_only`` dataset holding the named rows."""
    path.write_text(
        json.dumps(
            [
                {"id": row_id, "gt": _RESUME_ROWS[row_id][0], "pred": _RESUME_ROWS[row_id][1]}
                for row_id in row_ids
            ]
        )
    )


class _StubRowJudges:
    """Patch the four per-row STT judges with counting stubs.

    Every stub answers from a fixed table, so a resumed run and a single run
    over the same rows produce the same numbers.
    """

    def __init__(self, equivalence_judge=None):
        from calibrate_agent.stt import metrics as stt_metrics
        from calibrate_agent.stt import sarvam_intent_entity as sie
        from calibrate_agent.stt import sarvam_llm_wer as slw
        from calibrate_agent.stt import semantic_wer as sw

        async def evaluator(reference, prediction, evaluators=None, fallback_model=None):
            return {
                ev["name"]: {
                    "match": reference == prediction,
                    "reasoning": f"{reference}|{prediction}",
                }
                for ev in evaluators
            }

        async def intent_entity(reference, prediction, model=None, index=0, context=""):
            same = reference == prediction
            return {
                "intent_score": 1 if same else 0,
                "intent_explanation": f"intent {reference}",
                "entity_score": 1.0 if same else 0.5,
                "ground_truth_entities": "NA",
                "preserved_entities": "NA",
                "missing_entities": "",
                "entity_explanation": f"entity {reference}",
            }

        async def semantic(reference, prediction, model=None):
            s, d, i, ref_words = _SEMANTIC_COUNTS[reference]
            return {
                "substitutions": s,
                "deletions": d,
                "insertions": i,
                "reference_words": ref_words,
                "normalized_reference": reference,
                "normalized_hypothesis": prediction,
                "reasoning": f"semantic {reference}",
            }

        async def equivalence(reference, prediction, model=None):
            return {
                "index": 0,
                "equivalent": _EQUIVALENT_SEGMENTS[(reference, prediction)],
                "reasoning": f"equivalence {reference}",
            }

        self.evaluator = AsyncMock(side_effect=evaluator)
        self.intent_entity = AsyncMock(side_effect=intent_entity)
        self.semantic = AsyncMock(side_effect=semantic)
        self.equivalence = AsyncMock(side_effect=equivalence_judge or equivalence)

        normalizer = MagicMock()
        normalizer.normalize_texts.side_effect = (
            lambda texts, langs, n_jobs=1: list(texts)
        )
        self._patches = [
            patch.object(stt_metrics, "stt_llm_judge", self.evaluator),
            patch.object(stt_metrics, "_get_indic_normalizer", return_value=normalizer),
            patch.object(sie, "intent_entity_judge", self.intent_entity),
            patch.object(sw, "semantic_wer_judge", self.semantic),
            patch.object(slw, "equivalence_judge", self.equivalence),
        ]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False

    def reset(self):
        for mock in (self.evaluator, self.intent_entity, self.semantic, self.equivalence):
            mock.reset_mock()

    @property
    def counts(self):
        return {
            "evaluator": self.evaluator.await_count,
            "intent_entity": self.intent_entity.await_count,
            "semantic": self.semantic.await_count,
            "equivalence": self.equivalence.await_count,
        }


_RESUME_EVALUATOR = {
    "name": "semantic_match",
    "system_prompt": "match",
    "judge_model": "openai/gpt-4.1",
}


class TestSTTEvalOnlyResume(unittest.IsolatedAsyncioTestCase):
    """Resuming an eval-only run: pay once per row, land on the same numbers."""

    async def _run(
        self,
        out,
        ds_path,
        evaluators=None,
        overwrite=False,
        run_llm_judges=True,
    ):
        from calibrate_agent.stt import eval as stt_eval

        return await stt_eval.run_eval_only(
            dataset_path=str(ds_path),
            output_dir=str(out),
            judge_evaluators=[evaluators or _RESUME_EVALUATOR],
            run_llm_judges=run_llm_judges,
            overwrite=overwrite,
        )

    def _read(self, out):
        rows = pd.read_csv(out / "results.csv").sort_values("id")
        return json.loads((out / "metrics.json").read_text()), rows

    async def test_resume_pays_only_for_missing_rows_and_matches_a_single_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first_ds = base / "first.json"
            full_ds = base / "full.json"
            _dataset(first_ds, ["row_a", "row_b"])
            _dataset(full_ds, ["row_a", "row_b", "row_c", "row_d"])

            resumed_dir = base / "resumed"
            single_dir = base / "single"

            with _StubRowJudges() as stubs:
                await self._run(resumed_dir, first_ds)
                stubs.reset()
                result = await self._run(resumed_dir, full_ds)

                self.assertEqual(result["status"], "completed")
                # Only row_c and row_d are new. row_c repeats row_a's text, so
                # its word segment already has a verdict and costs nothing; only
                # row_d's ("good", "gud") segment is judged.
                self.assertEqual(
                    stubs.counts,
                    {
                        "evaluator": 2,
                        "intent_entity": 2,
                        "semantic": 2,
                        "equivalence": 1,
                    },
                )

                await self._run(single_dir, full_ds)

            resumed_metrics, resumed_rows = self._read(resumed_dir)
            single_metrics, single_rows = self._read(single_dir)

            self.assertEqual(
                sorted(resumed_rows["id"]), ["row_a", "row_b", "row_c", "row_d"]
            )
            for key in (
                "sarvam_intent_score",
                "sarvam_entity_score",
                "sarvam_llm_wer",
                "sarvam_llm_cer",
                "semantic_wer",
            ):
                self.assertEqual(resumed_metrics[key], single_metrics[key], key)
            self.assertEqual(
                resumed_metrics["semantic_match"], single_metrics["semantic_match"]
            )
            self.assertEqual(resumed_metrics, single_metrics)
            self.assertEqual(
                resumed_rows.to_dict(orient="records"),
                single_rows.to_dict(orient="records"),
            )

    async def test_overwrite_starts_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            _dataset(ds, ["row_a", "row_b"])
            out = base / "out"

            with _StubRowJudges() as stubs:
                await self._run(out, ds)
                (out / "results.csv").write_text("id,gt,pred\nstale,,\n")
                stubs.reset()
                await self._run(out, ds, overwrite=True)

                # Every row is judged again, including the word segments whose
                # verdicts the earlier run had saved.
                self.assertEqual(
                    stubs.counts,
                    {
                        "evaluator": 2,
                        "intent_entity": 2,
                        "semantic": 2,
                        "equivalence": 2,
                    },
                )

            _, rows = self._read(out)
            self.assertEqual(sorted(rows["id"]), ["row_a", "row_b"])

    async def test_changed_evaluator_set_is_judged_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            _dataset(ds, ["row_a", "row_b"])
            out = base / "out"

            with _StubRowJudges() as stubs:
                await self._run(out, ds)
                stubs.reset()
                await self._run(
                    out,
                    ds,
                    evaluators={
                        "name": "accuracy",
                        "system_prompt": "accurate?",
                        "judge_model": "openai/gpt-4.1",
                    },
                )

                # The stored scores belong to a different evaluator, so both
                # rows go back to the judge; the other judges are untouched.
                self.assertEqual(stubs.evaluator.await_count, 2)
                self.assertEqual(stubs.intent_entity.await_count, 0)

            _, rows = self._read(out)
            self.assertIn("accuracy", rows.columns)
            self.assertNotIn("semantic_match", rows.columns)

    async def test_numeric_looking_id_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            ds.write_text(
                json.dumps(
                    [
                        {"id": "1", "gt": "hello world", "pred": "hello word"},
                        {"id": "2", "gt": "good morning", "pred": "gud morning"},
                    ]
                )
            )
            out = base / "out"

            with _StubRowJudges() as stubs:
                await self._run(out, ds)
                stubs.reset()
                await self._run(out, ds)

                self.assertEqual(
                    stubs.counts,
                    {
                        "evaluator": 0,
                        "intent_entity": 0,
                        "semantic": 0,
                        "equivalence": 0,
                    },
                )

            _, rows = self._read(out)
            self.assertEqual(sorted(str(v) for v in rows["id"]), ["1", "2"])

    async def test_duplicate_ids_fail_with_a_clear_message(self):
        from calibrate_agent.stt import eval as stt_eval

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            ds.write_text(
                json.dumps(
                    [
                        {"id": "row_a", "gt": "a", "pred": "a"},
                        {"id": "row_a", "gt": "b", "pred": "b"},
                    ]
                )
            )
            result = await stt_eval.run_eval_only(
                dataset_path=str(ds), output_dir=str(base / "out")
            )

        self.assertEqual(result["status"], "error")
        self.assertIn("Duplicate row id", result["error"])
        self.assertIn("row_a", result["error"])

    async def test_blank_score_cell_is_judged_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            _dataset(ds, ["row_a", "row_b"])
            out = base / "out"

            with _StubRowJudges() as stubs:
                await self._run(out, ds)

                rows = pd.read_csv(out / "results.csv")
                rows["semantic_match"] = rows["semantic_match"].astype(object)
                rows.loc[rows["id"] == "row_a", "semantic_match"] = ""
                rows.to_csv(out / "results.csv", index=False)

                stubs.reset()
                await self._run(out, ds)

                self.assertEqual(stubs.evaluator.await_count, 1)
                self.assertEqual(stubs.intent_entity.await_count, 0)

            _, rows = self._read(out)
            self.assertEqual(sorted(rows["id"]), ["row_a", "row_b"])
            self.assertEqual(list(rows["semantic_match"]), [False, False])

    async def test_row_missing_from_the_dataset_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            both_ds = base / "both.json"
            one_ds = base / "one.json"
            _dataset(both_ds, ["row_a", "row_b"])
            _dataset(one_ds, ["row_a"])
            out = base / "out"

            with _StubRowJudges():
                await self._run(out, both_ds)
                await self._run(out, one_ds)

            _, rows = self._read(out)
            self.assertEqual(list(rows["id"]), ["row_a"])

    async def test_built_in_judge_rows_land_in_results_csv_while_the_run_is_going(self):
        """The intent/entity judge writes each row as it finishes, not at the end."""
        import asyncio

        from calibrate_agent.stt import eval as stt_eval

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            _dataset(ds, ["row_a", "row_b"])
            out = base / "out"
            results_path = out / "results.csv"
            seen = {}

            with _StubRowJudges() as stubs:
                first = stubs.intent_entity.side_effect

                async def wait_for_the_other_row(
                    reference, prediction, model=None, index=0, context=""
                ):
                    if reference == "hello world":
                        for _ in range(200):
                            text = (
                                results_path.read_text()
                                if results_path.exists()
                                else ""
                            )
                            if "sarvam_intent_score" in text:
                                seen["csv"] = text
                                break
                            await asyncio.sleep(0.01)
                    return await first(reference, prediction, model=model, index=index)

                stubs.intent_entity.side_effect = wait_for_the_other_row
                await stt_eval.run_eval_only(
                    dataset_path=str(ds),
                    output_dir=str(out),
                    run_llm_judges=True,
                )

            self.assertIn("doctor ne bola", seen.get("csv", ""))

    async def test_verdicts_survive_a_failed_llm_wer_judge(self):
        """A killed LLM-WER judge keeps the segment verdicts it paid for."""
        from calibrate_agent.stt import eval as stt_eval

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            _dataset(ds, ["row_a", "row_b"])
            out = base / "out"

            async def half_failing(reference, prediction, model=None):
                if reference == "world":
                    raise RuntimeError("judge died")
                return {
                    "index": 0,
                    "equivalent": _EQUIVALENT_SEGMENTS[(reference, prediction)],
                    "reasoning": f"equivalence {reference}",
                }

            with _StubRowJudges(equivalence_judge=half_failing):
                await self._run(out, ds)

            verdicts = json.loads((out / "llm_wer_verdicts.json").read_text())
            self.assertEqual(
                [(v["reference"], v["prediction"]) for v in verdicts],
                [("doctor", "daktar")],
            )

            with _StubRowJudges() as stubs:
                await self._run(out, ds)
                # Only the segment the failed run never got a verdict for.
                self.assertEqual(stubs.equivalence.await_count, 1)
                self.assertEqual(
                    stubs.equivalence.await_args.args[:2], ("world", "word")
                )

    async def test_stored_rows_supply_the_verdicts_without_the_saved_file(self):
        """A judged row's stored segments pay for the same phrase in a new row."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first_ds = base / "first.json"
            full_ds = base / "full.json"
            _dataset(first_ds, ["row_a"])
            # row_c repeats row_a's words, so its one differing segment already
            # has a verdict stored on row_a.
            _dataset(full_ds, ["row_a", "row_c"])
            out = base / "out"

            with _StubRowJudges() as stubs:
                await self._run(out, first_ds)
                (out / "llm_wer_verdicts.json").unlink()
                stubs.reset()
                await self._run(out, full_ds)

                self.assertEqual(stubs.equivalence.await_count, 0)

            _, rows = self._read(out)
            self.assertEqual(list(rows["id"]), ["row_a", "row_c"])
            self.assertEqual(
                json.loads(rows["sarvam_llm_wer_reasoning"].iloc[1]),
                json.loads(rows["sarvam_llm_wer_reasoning"].iloc[0]),
            )

    async def test_a_failing_judge_keeps_the_rows_it_already_scored(self):
        """One judge dying on a new row must not wipe the rows already paid for."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first_ds = base / "first.json"
            full_ds = base / "full.json"
            _dataset(first_ds, ["row_a", "row_b"])
            _dataset(full_ds, ["row_a", "row_b", "row_d"])
            out = base / "out"

            with _StubRowJudges() as stubs:
                await self._run(out, first_ds)
                _, first_rows = self._read(out)
                stored = dict(
                    zip(first_rows["id"], first_rows["semantic_wer"], strict=True)
                )

                stubs.semantic.side_effect = RuntimeError("judge died")
                await self._run(out, full_ds)

            metrics, rows = self._read(out)
            by_id = dict(zip(rows["id"], rows["semantic_wer"], strict=True))
            self.assertEqual(by_id["row_a"], stored["row_a"])
            self.assertEqual(by_id["row_b"], stored["row_b"])
            self.assertTrue(pd.isna(by_id["row_d"]))
            # The judge produced no dataset score this run, so none is reported.
            self.assertNotIn("semantic_wer", metrics)
            # The judges that did run still cover every row.
            self.assertEqual(list(rows["sarvam_intent_score"].isna()), [False] * 3)

    async def test_skipping_the_built_in_judges_keeps_their_stored_scores(self):
        """Resuming with the built-in judges off keeps the scores already stored."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            _dataset(ds, ["row_a", "row_b"])
            out = base / "out"

            with _StubRowJudges() as stubs:
                await self._run(out, ds)
                _, first_rows = self._read(out)
                stubs.reset()
                await self._run(out, ds, run_llm_judges=False)

                self.assertEqual(stubs.counts["semantic"], 0)
                self.assertEqual(stubs.counts["intent_entity"], 0)

            metrics, rows = self._read(out)
            for column in (
                "semantic_wer",
                "sarvam_intent_score",
                "sarvam_entity_score",
                "sarvam_llm_wer",
                "sarvam_llm_cer",
            ):
                self.assertIn(column, rows.columns)
                self.assertEqual(list(rows[column]), list(first_rows[column]))
            # No judge ran, so no dataset-level score is reported for one.
            self.assertNotIn("semantic_wer", metrics)
            self.assertNotIn("sarvam_intent_score", metrics)

    async def test_an_edited_prediction_is_judged_again(self):
        """A row whose text changed cannot keep the verdict on the old text."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            out = base / "out"
            _dataset(ds, ["row_a"])

            with _StubRowJudges() as stubs:
                await self._run(out, ds, run_llm_judges=False)
                _, first_rows = self._read(out)
                self.assertEqual(list(first_rows["semantic_match"]), [False])

                # Same id, corrected prediction.
                ds.write_text(
                    json.dumps(
                        [{"id": "row_a", "gt": "doctor ne bola", "pred": "doctor ne bola"}]
                    )
                )
                stubs.reset()
                await self._run(out, ds, run_llm_judges=False)
                self.assertEqual(stubs.evaluator.await_count, 1)

            _, rows = self._read(out)
            self.assertEqual(list(rows["pred"]), ["doctor ne bola"])
            self.assertEqual(list(rows["semantic_match"]), [True])

    async def test_ids_differing_only_by_type_fail_with_a_clear_message(self):
        from calibrate_agent.stt import eval as stt_eval

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            ds.write_text(
                json.dumps(
                    [
                        {"id": 1, "gt": "a", "pred": "a"},
                        {"id": "1", "gt": "b", "pred": "b"},
                    ]
                )
            )
            result = await stt_eval.run_eval_only(
                dataset_path=str(ds), output_dir=str(base / "out")
            )

        self.assertEqual(result["status"], "error")
        self.assertIn("Duplicate row id", result["error"])

    async def test_an_unusable_id_reports_the_row_instead_of_crashing(self):
        from calibrate_agent.stt import eval as stt_eval

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            ds.write_text(
                json.dumps(
                    [
                        {"id": "row_a", "gt": "a", "pred": "a"},
                        {"id": ["row_b"], "gt": "b", "pred": "b"},
                    ]
                )
            )
            result = await stt_eval.run_eval_only(
                dataset_path=str(ds), output_dir=str(base / "out")
            )

        self.assertEqual(result["status"], "error")
        self.assertIn("Row 1", result["error"])
        self.assertIn("unusable id", result["error"])

    async def test_run_eval_only_asks_for_row_by_row_writing(self):
        from calibrate_agent.stt import eval as stt_eval

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            _dataset(ds, ["row_a"])
            scorer = AsyncMock(return_value={})
            with patch.object(stt_eval, "_score_and_write_results", scorer):
                await stt_eval.run_eval_only(
                    dataset_path=str(ds), output_dir=str(base / "out")
                )

        self.assertIs(scorer.await_args.kwargs.get("stream_rows"), True)


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


if __name__ == "__main__":
    unittest.main()
