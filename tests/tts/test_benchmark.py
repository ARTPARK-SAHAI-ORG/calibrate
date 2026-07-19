"""
Tests for calibrate_agent/tts/benchmark.py main() — focused on the eval-only
dispatch branch (skip synthesis, run the audio judge on a prior run dir).

Run with:
    python -m pytest tests/tts/test_benchmark.py -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock


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

        self.assertEqual(
            captured["judge_evaluators"], [{"name": "q", "system_prompt": "p"}]
        )

    async def test_error_result_exits(self):
        async def fake_run_eval_only(*, dataset_path, output_dir, judge_evaluators):
            return {"status": "error", "error": "boom"}

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "calibrate_agent.tts.benchmark.run_eval_only",
                AsyncMock(side_effect=fake_run_eval_only),
            ):
                with self.assertRaises(SystemExit):
                    await self._run_main(
                        ["--dataset", tmp, "-o", os.path.join(tmp, "out")]
                    )


if __name__ == "__main__":
    unittest.main()
