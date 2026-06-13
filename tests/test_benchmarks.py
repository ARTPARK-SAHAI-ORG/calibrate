"""Tests for benchmark modules — llm, stt, tts."""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pandas as pd


# =============================================================================
# LLM Benchmark
# =============================================================================

class TestLLMBenchmarkRun(unittest.IsolatedAsyncioTestCase):
    async def test_run_basic(self):
        from calibrate.llm import benchmark as B

        fake_results = {"model": "m1", "provider": "openrouter",
                        "metrics": {"passed": 1, "total": 1}, "results": []}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(B, "run_model_tests", AsyncMock(return_value=fake_results)), \
             patch.object(B, "generate_leaderboard"):
            result = await B.run(
                config={"system_prompt": "sp", "tools": [], "test_cases": []},
                models=["m1", "m2"],
                provider="openrouter",
                output_dir=tmp,
            )
        self.assertEqual(result["status"], "completed")
        self.assertIn("m1", result["models"])
        self.assertIn("m2", result["models"])

    async def test_run_leaderboard_error_recorded(self):
        from calibrate.llm import benchmark as B

        fake_results = {"model": "m1", "provider": "openrouter",
                        "metrics": {"passed": 1, "total": 1}, "results": []}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(B, "run_model_tests", AsyncMock(return_value=fake_results)), \
             patch.object(B, "generate_leaderboard", side_effect=RuntimeError("lb fail")):
            result = await B.run(
                config={"system_prompt": "sp", "tools": [], "test_cases": []},
                models=["m1"],
                provider="openrouter",
                output_dir=tmp,
            )
        self.assertIn("leaderboard", result["models"])


class TestLLMBenchmarkMain(unittest.IsolatedAsyncioTestCase):
    async def test_main_basic(self):
        from calibrate.llm import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({
                "system_prompt": "sp", "tools": [], "test_cases": [],
            }))
            argv = ["b.py", "-c", str(cfg), "-m", "m1", "-p", "openrouter",
                    "-o", str(Path(tmp) / "out")]
            fake_results = {"status": "completed", "output_dir": tmp,
                            "leaderboard_dir": tmp,
                            "models": {"m1": {"metrics": {"passed": 1, "total": 1}}}}
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run", AsyncMock(return_value=fake_results)):
                await B.main()

    async def test_main_error_path_exits(self):
        from calibrate.llm import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({
                "system_prompt": "sp", "tools": [], "test_cases": [],
            }))
            argv = ["b.py", "-c", str(cfg), "-m", "m1", "-p", "openrouter",
                    "-o", str(Path(tmp) / "out")]
            fake_results = {"status": "completed", "output_dir": tmp,
                            "leaderboard_dir": tmp,
                            "models": {"m1": {"status": "error", "error": "boom"}}}
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run", AsyncMock(return_value=fake_results)):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_append_mode(self):
        from calibrate.llm import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({
                "system_prompt": "sp", "tools": [], "test_cases": [],
            }))
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()
            (out_dir / "logs").write_text("existing")
            argv = ["b.py", "-c", str(cfg), "-m", "m1", "-p", "openrouter",
                    "-o", str(out_dir)]
            fake_results = {"status": "completed", "output_dir": str(out_dir),
                            "leaderboard_dir": str(out_dir),
                            "models": {"m1": {"metrics": {"passed": 1, "total": 1}}}}
            with patch.object(sys, "argv", argv), \
                 patch.dict(os.environ, {"CALIBRATE_LLM_LOG_APPEND": "1"}), \
                 patch.object(B, "run", AsyncMock(return_value=fake_results)):
                await B.main()


# =============================================================================
# STT Benchmark
# =============================================================================

class TestSTTBenchmarkRun(unittest.IsolatedAsyncioTestCase):
    async def test_run_basic(self):
        from calibrate.stt import benchmark as B

        fake_result = {"provider": "deepgram", "status": "completed",
                       "metrics": {"wer": 0.1,
                                   "semantic_match": {"type": "binary", "mean": 0.9}}}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            (base / "audios" / "a.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(base / "stt.csv", index=False)
            output_dir = str(base / "out")
            with patch.object(B, "run_single_provider_eval",
                              AsyncMock(return_value=fake_result)), \
                 patch.object(B, "generate_leaderboard"):
                result = await B.run(
                    providers=["deepgram", "google"],
                    input_dir=str(base),
                    output_dir=output_dir,
                )
        self.assertEqual(result["status"], "completed")
        self.assertIn("deepgram", result["providers"])

    async def test_run_leaderboard_error(self):
        from calibrate.stt import benchmark as B

        fake_result = {"provider": "deepgram", "status": "completed",
                       "metrics": {"wer": 0.1}}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(B, "run_single_provider_eval",
                          AsyncMock(return_value=fake_result)), \
             patch.object(B, "generate_leaderboard", side_effect=Exception("lb fail")):
            result = await B.run(
                providers=["deepgram"],
                input_dir=tmp,
                output_dir=tmp,
            )
        self.assertIn("leaderboard", result["providers"])


class TestSTTBenchmarkMain(unittest.IsolatedAsyncioTestCase):
    def _make_input_dir(self, tmp: Path):
        (tmp / "audios").mkdir()
        (tmp / "audios" / "a.wav").write_bytes(b"\x00")
        pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(tmp / "stt.csv", index=False)

    async def test_main_invalid_provider(self):
        from calibrate.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_input_dir(base)
            argv = ["b.py", "-p", "bogus", "-i", str(base), "-o", str(base / "out")]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_invalid_input(self):
        from calibrate.stt import benchmark as B

        argv = ["b.py", "-p", "deepgram", "-i", "/nonexistent/missing",
                "-o", "/tmp/x"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                await B.main()

    async def test_main_eval_only_missing_dataset(self):
        from calibrate.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["b.py", "--eval-only", "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_eval_only_success(self):
        from calibrate.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            ds.write_text(json.dumps([{"id": "a", "gt": "hi", "pred": "hi"}]))
            out = base / "out"

            fake_result = {"status": "completed",
                           "metrics": {"wer": 0.1,
                                       "semantic_match": {"type": "binary", "mean": 0.9}}}
            argv = ["b.py", "--eval-only", "--dataset", str(ds), "-o", str(out)]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run_eval_only", AsyncMock(return_value=fake_result)):
                await B.main()

    async def test_main_eval_only_error(self):
        from calibrate.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            ds.write_text("[]")
            out = base / "out"

            fake_result = {"status": "error", "error": "boom"}
            argv = ["b.py", "--eval-only", "--dataset", str(ds), "-o", str(out)]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run_eval_only", AsyncMock(return_value=fake_result)):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_no_provider_no_eval_only(self):
        from calibrate.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["b.py", "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_no_input_dir(self):
        from calibrate.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["b.py", "-p", "deepgram", "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_success_path(self):
        from calibrate.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_input_dir(base)
            out = base / "out"

            fake_run_result = {
                "status": "completed",
                "output_dir": str(out),
                "leaderboard_dir": str(out / "leaderboard"),
                "providers": {
                    "deepgram": {"status": "completed",
                                 "metrics": {"wer": 0.1,
                                             "semantic_match": {"type": "binary", "mean": 0.9}}},
                },
            }
            argv = ["b.py", "-p", "deepgram", "-i", str(base), "-o", str(out)]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run", AsyncMock(return_value=fake_run_result)):
                await B.main()

    async def test_main_error_provider_exits(self):
        from calibrate.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_input_dir(base)
            out = base / "out"

            fake_run_result = {
                "status": "completed",
                "output_dir": str(out),
                "leaderboard_dir": str(out / "leaderboard"),
                "providers": {
                    "deepgram": {"status": "error", "error": "boom"},
                },
            }
            argv = ["b.py", "-p", "deepgram", "-i", str(base), "-o", str(out)]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run", AsyncMock(return_value=fake_run_result)):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_with_config(self):
        from calibrate.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_input_dir(base)
            out = base / "out"
            cfg = base / "cfg.json"
            cfg.write_text(json.dumps({"evaluators": [{"name": "x", "system_prompt": "...", "judge_model": "m"}]}))

            fake_run_result = {
                "status": "completed",
                "output_dir": str(out),
                "leaderboard_dir": str(out / "leaderboard"),
                "providers": {
                    "deepgram": "error: lb",
                },
            }
            argv = ["b.py", "-p", "deepgram", "-i", str(base), "-o", str(out),
                    "-c", str(cfg)]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run", AsyncMock(return_value=fake_run_result)):
                await B.main()


# =============================================================================
# TTS Benchmark
# =============================================================================

class TestTTSBenchmarkRun(unittest.IsolatedAsyncioTestCase):
    async def test_run_basic(self):
        from calibrate.tts import benchmark as B

        fake_result = {"provider": "openai", "status": "completed",
                       "metrics": {"ttfb": {"p50": 0.5, "p95": 0.6, "p99": 0.6, "count": 2},
                                   "pronunciation": {"type": "binary", "mean": 0.9}}}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(B, "run_single_provider_eval", AsyncMock(return_value=fake_result)), \
             patch.object(B, "generate_leaderboard"):
            result = await B.run(
                providers=["openai", "google"],
                input="/tmp/in.csv",
                output_dir=tmp,
            )
        self.assertEqual(result["status"], "completed")

    async def test_run_leaderboard_error(self):
        from calibrate.tts import benchmark as B

        fake_result = {"status": "completed"}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(B, "run_single_provider_eval", AsyncMock(return_value=fake_result)), \
             patch.object(B, "generate_leaderboard", side_effect=Exception("lb fail")):
            result = await B.run(
                providers=["openai"],
                input="/tmp/in.csv",
                output_dir=tmp,
            )
        self.assertIn("leaderboard", result["providers"])


class TestTTSBenchmarkMain(unittest.IsolatedAsyncioTestCase):
    async def test_main_invalid_provider(self):
        from calibrate.tts import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            argv = ["b.py", "-p", "bogus", "-i", str(inp), "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_invalid_input(self):
        from calibrate.tts import benchmark as B

        argv = ["b.py", "-p", "openai", "-i", "/nonexistent.csv", "-o", "/tmp/x"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                await B.main()

    async def test_main_success(self):
        from calibrate.tts import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            out = Path(tmp) / "out"

            fake_run_result = {
                "status": "completed",
                "output_dir": str(out),
                "leaderboard_dir": str(out / "lb"),
                "providers": {
                    "openai": {"status": "completed",
                               "metrics": {
                                   "ttfb": {"p50": 0.5, "p95": 0.6, "p99": 0.6, "count": 2},
                                   "pronunciation": {"type": "binary", "mean": 0.9},
                               }},
                },
            }
            argv = ["b.py", "-p", "openai", "-i", str(inp), "-o", str(out)]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run", AsyncMock(return_value=fake_run_result)):
                await B.main()

    async def test_main_with_config_and_error(self):
        from calibrate.tts import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            out = Path(tmp) / "out"
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text(json.dumps({"evaluators": [{"name": "x", "system_prompt": "x", "judge_model": "m"}]}))

            fake_run_result = {
                "status": "completed",
                "output_dir": str(out),
                "leaderboard_dir": str(out / "lb"),
                "providers": {
                    "openai": {"status": "error", "error": "boom"},
                },
            }
            argv = ["b.py", "-p", "openai", "-i", str(inp), "-o", str(out), "-c", str(cfg)]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run", AsyncMock(return_value=fake_run_result)):
                with self.assertRaises(SystemExit):
                    await B.main()


if __name__ == "__main__":
    unittest.main()
