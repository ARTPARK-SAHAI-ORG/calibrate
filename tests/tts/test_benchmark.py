"""
Tests for calibrate_agent/tts/benchmark.py main() — focused on the eval-only
dispatch branch (skip synthesis, run the audio judge on a prior run dir).

Run with:
    python -m pytest tests/tts/test_benchmark.py -v
"""

import io
import os
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pandas as pd


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


class TestTTSBenchmarkEvalOnly(unittest.IsolatedAsyncioTestCase):
    async def _run_main(self, argv_extra):
        from calibrate_agent.tts import benchmark

        argv = ["calibrate-agent", "--eval-only"] + argv_extra
        with patch.object(sys, "argv", argv):
            await benchmark.main()

    async def test_no_dataset_exits(self):
        with self.assertRaises(SystemExit):
            await self._run_main([])

    async def test_success_calls_run_eval_only(self):
        captured = {}

        async def fake_run_eval_only(*, dataset_path, output_dir, judge_evaluators):
            captured["dataset_path"] = dataset_path
            captured["output_dir"] = output_dir
            captured["judge_evaluators"] = judge_evaluators
            return {
                "status": "completed",
                "metrics": {"quality": {"type": "binary", "mean": 0.9}},
                "output_dir": output_dir,
            }

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            with patch(
                "calibrate_agent.tts.benchmark.run_eval_only",
                AsyncMock(side_effect=fake_run_eval_only),
            ):
                await self._run_main(["--dataset", tmp, "-o", out])

        self.assertEqual(captured["dataset_path"], tmp)
        self.assertEqual(captured["output_dir"], out)
        # No config passed → evaluators default to None.
        self.assertIsNone(captured["judge_evaluators"])

    async def test_config_forwards_evaluators(self):
        captured = {}

        async def fake_run_eval_only(*, dataset_path, output_dir, judge_evaluators):
            captured["judge_evaluators"] = judge_evaluators
            return {"status": "completed", "metrics": {}, "output_dir": output_dir}

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text('{"evaluators": [{"name": "q", "system_prompt": "p"}]}')
            with patch(
                "calibrate_agent.tts.benchmark.run_eval_only",
                AsyncMock(side_effect=fake_run_eval_only),
            ):
                await self._run_main(
                    ["--dataset", tmp, "-o", os.path.join(tmp, "out"), "-c", str(cfg)]
                )

        self.assertEqual(captured["judge_evaluators"], [{"name": "q", "system_prompt": "p"}])

    async def test_error_result_exits(self):
        async def fake_run_eval_only(*, dataset_path, output_dir, judge_evaluators):
            return {"status": "error", "error": "boom"}

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "calibrate_agent.tts.benchmark.run_eval_only",
                AsyncMock(side_effect=fake_run_eval_only),
            ):
                with self.assertRaises(SystemExit):
                    await self._run_main(["--dataset", tmp, "-o", os.path.join(tmp, "out")])

    async def test_eval_only_with_yes_runs_cost_gate(self):
        mock_eval = AsyncMock(
            return_value={
                "status": "completed",
                "metrics": {"quality": {"type": "binary", "mean": 0.9}},
                "output_dir": "",
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(tmp)
            out = os.path.join(tmp, "out")
            with patch(
                "calibrate_agent.tts.benchmark.run_eval_only",
                mock_eval,
            ):
                await self._run_main(["--dataset", run_dir, "-o", out, "-y"])

        mock_eval.assert_called_once()
        self.assertEqual(mock_eval.call_args.kwargs["dataset_path"], run_dir)
        self.assertEqual(mock_eval.call_args.kwargs["output_dir"], out)

    async def test_eval_only_cancel_on_decline(self):
        mock_eval = AsyncMock()

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(tmp)
            out = os.path.join(tmp, "out")
            buf = io.StringIO()
            with patch(
                "calibrate_agent.tts.benchmark.confirm_estimated_judge_cost",
                return_value=False,
            ), patch(
                "calibrate_agent.tts.benchmark.run_eval_only",
                mock_eval,
            ), redirect_stdout(buf):
                with self.assertRaises(SystemExit) as ctx:
                    await self._run_main(["--dataset", run_dir, "-o", out])

        self.assertEqual(ctx.exception.code, 0)
        mock_eval.assert_not_called()
        self.assertIn("Judge run cancelled.", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
