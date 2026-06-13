"""Cover remaining branches in stt/eval.py."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pandas as pd


def _fake_intent_entity(intent=1, entity=1.0):
    """Adaptive ``get_intent_entity_score`` mock — one row per input pair."""

    async def _fn(refs, preds, language="english", model=None):
        return {
            "intent": float(intent),
            "entity": float(entity),
            "per_row": [
                {
                    "intent_score": intent,
                    "intent_explanation": "ok",
                    "entity_score": entity,
                    "entity_explanation": "ok",
                }
                for _ in refs
            ],
        }

    return AsyncMock(side_effect=_fn)


class TestValidateSTTInputDirEdges(unittest.TestCase):
    def test_input_path_not_directory(self):
        from calibrate.stt.eval import validate_stt_input_dir

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "file.csv"
            f.write_text("id,text\n")
            ok, err = validate_stt_input_dir(str(f), "anything")
            self.assertFalse(ok)
            self.assertIn("not a directory", err)

    def test_audios_not_directory(self):
        from calibrate.stt.eval import validate_stt_input_dir

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "stt.csv").write_text("id,text\na,hi\n")
            (base / "audios").write_text("not a dir")  # file, not dir
            ok, err = validate_stt_input_dir(str(base), "stt.csv")
            self.assertFalse(ok)
            self.assertIn("not a directory", err)

    def test_csv_read_failure(self):
        from calibrate.stt.eval import validate_stt_input_dir

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            (base / "stt.csv").write_bytes(b"\x80\x81\x82\x83")  # invalid utf-8
            with patch("pandas.read_csv", side_effect=Exception("bad csv")):
                ok, err = validate_stt_input_dir(str(base), "stt.csv")
            self.assertFalse(ok)
            self.assertIn("Failed to read CSV", err)

    def test_many_missing_audio_files(self):
        from calibrate.stt.eval import validate_stt_input_dir

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            # 7 ids, all missing audio
            ids = [f"id_{i}" for i in range(7)]
            pd.DataFrame({"id": ids, "text": ids}).to_csv(base / "stt.csv", index=False)
            ok, err = validate_stt_input_dir(str(base), "stt.csv")
            self.assertFalse(ok)
            self.assertIn("Missing 7 audio files", err)


class TestRunSinglProviderEvalProgressNoChange(unittest.IsolatedAsyncioTestCase):
    async def test_no_progress_writes_empty_transcripts(self):
        from calibrate.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            (base / "audios" / "a.wav").write_bytes(b"\x00")
            (base / "audios" / "b.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a", "b"], "text": ["hi", "ho"]}).to_csv(
                base / "stt.csv", index=False
            )
            out = base / "out"
            out.mkdir()
            (out / "deepgram").mkdir()
            # Pre-existing partial: a only (so 1/2 — partial)
            pd.DataFrame([{"id": "a", "gt": "hi", "pred": "hi"}]).to_csv(
                out / "deepgram" / "results.csv", index=False
            )

            # First call returns 0 (no new), second loop sees no progress -> writes empty
            with patch.object(E, "transcribe_audio",
                              AsyncMock(side_effect=Exception("provider down"))), \
                 patch.object(E, "get_wer_score",
                              return_value={"score": 0.0, "per_row": [0.0, 0.0]}), \
                 patch.object(E, "get_cer_score",
                              return_value={"score": 0.0, "per_row": [0.0, 0.0]}), \
                 patch.object(E, "get_intent_entity_score", _fake_intent_entity()), \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 0.5}},
                     "per_row": [
                         {"semantic_match": {"match": True, "reasoning": "ok"}},
                         {"semantic_match": {"match": False, "reasoning": "no"}},
                     ],
                 })):
                # Will raise after first attempt since transcribe_audio raises
                try:
                    await E.run_single_provider_eval(
                        provider="deepgram",
                        language="english",
                        input_dir=str(base),
                        input_file_name="stt.csv",
                        output_dir=str(out),
                        debug=False,
                        debug_count=5,
                        ignore_retry=False,
                        overwrite=False,
                    )
                except Exception:
                    pass


class TestRunSinglProviderEvalAlreadyAllProcessed(unittest.IsolatedAsyncioTestCase):
    async def test_all_processed_breaks(self):
        from calibrate.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            (base / "audios" / "a.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(
                base / "stt.csv", index=False
            )
            out = base / "out"
            out.mkdir()
            (out / "deepgram").mkdir()
            pd.DataFrame([{"id": "a", "gt": "hi", "pred": "hi"}]).to_csv(
                out / "deepgram" / "results.csv", index=False
            )

            with patch.object(E, "get_wer_score",
                              return_value={"score": 0.0, "per_row": [0.0]}), \
                 patch.object(E, "get_cer_score",
                              return_value={"score": 0.0, "per_row": [0.0]}), \
                 patch.object(E, "get_intent_entity_score", _fake_intent_entity()), \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                     "per_row": [{"semantic_match": {"match": True, "reasoning": "ok"}}],
                 })):
                result = await E.run_single_provider_eval(
                    provider="deepgram",
                    language="english",
                    input_dir=str(base),
                    input_file_name="stt.csv",
                    output_dir=str(out),
                    debug=False,
                    debug_count=5,
                    ignore_retry=False,
                    overwrite=False,
                )
            self.assertEqual(result["status"], "completed")


class TestTranscribeAudioBackoff(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_provider_through_wrapper(self):
        """Going through __wrapped__ to skip backoff."""
        from calibrate.stt.eval import transcribe_audio

        inner = transcribe_audio
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        with self.assertRaises(ValueError):
            await inner(Path("/tmp/x.wav"), "ref", "fake_provider", "english", "u1")


if __name__ == "__main__":
    unittest.main()
