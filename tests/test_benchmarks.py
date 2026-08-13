"""Tests for benchmark modules — llm, stt, tts."""

import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from os.path import join
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pandas as pd


def _write_judge_cache(provider_dir: Path, *, kind: str = "tts_evaluators") -> None:
    provider_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": kind,
        "row_id": "1",
        "evaluator": None,
        "fingerprint": "fp",
        "result": {"ok": True},
    }
    (provider_dir / "judge_cache.jsonl").write_text(json.dumps(record) + "\n")


# =============================================================================
# LLM Benchmark
# =============================================================================

class TestLLMBenchmarkRun(unittest.IsolatedAsyncioTestCase):
    async def test_run_basic(self):
        from calibrate_agent.llm import benchmark as B

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
        from calibrate_agent.llm import benchmark as B

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
        from calibrate_agent.llm import benchmark as B

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
        from calibrate_agent.llm import benchmark as B

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
        from calibrate_agent.llm import benchmark as B

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
        from calibrate_agent.stt import benchmark as B

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

    async def test_run_forwards_intent_entity_to_provider_eval(self):
        from calibrate_agent.stt import benchmark as B

        fake_result = {"provider": "deepgram", "status": "completed",
                       "metrics": {"wer": 0.1}}
        mock_eval = AsyncMock(return_value=fake_result)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(B, "run_single_provider_eval", mock_eval), \
             patch.object(B, "generate_leaderboard"):
            await B.run(
                providers=["deepgram"],
                input_dir=tmp,
                output_dir=tmp,
                llm_judges=None,
            )
        self.assertIsNone(mock_eval.call_args.kwargs["llm_judges"])

    async def test_run_leaderboard_error(self):
        from calibrate_agent.stt import benchmark as B

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
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_input_dir(base)
            argv = ["b.py", "-p", "bogus", "-i", str(base), "-o", str(base / "out")]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_invalid_input(self):
        from calibrate_agent.stt import benchmark as B

        argv = ["b.py", "-p", "deepgram", "-i", "/nonexistent/missing",
                "-o", "/tmp/x"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                await B.main()

    async def test_main_eval_only_missing_dataset(self):
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["b.py", "--eval-only", "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_eval_only_success(self):
        from calibrate_agent.stt import benchmark as B

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

    async def test_main_eval_only_forwards_language(self):
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            ds.write_text(json.dumps([{"id": "a", "gt": "hi", "pred": "hi"}]))
            out = base / "out"

            fake_result = {"status": "completed", "metrics": {"wer": 0.1}}
            mock_eval = AsyncMock(return_value=fake_result)
            argv = ["b.py", "--eval-only", "--dataset", str(ds),
                    "-o", str(out), "--language", "hindi"]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run_eval_only", mock_eval):
                await B.main()

            self.assertEqual(mock_eval.call_args.kwargs["language"], "hindi")

    async def test_main_eval_only_intent_entity_on_by_default(self):
        from calibrate_agent._cli_args import DEFAULT_STT_LLM_JUDGES
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            ds.write_text(json.dumps([{"id": "a", "gt": "hi", "pred": "hi"}]))
            out = base / "out"

            fake_result = {"status": "completed", "metrics": {"wer": 0.1}}
            mock_eval = AsyncMock(return_value=fake_result)
            argv = ["b.py", "--eval-only", "--dataset", str(ds), "-o", str(out), "-y"]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run_eval_only", mock_eval):
                await B.main()

            self.assertEqual(
                mock_eval.call_args.kwargs["llm_judges"], DEFAULT_STT_LLM_JUDGES
            )

    async def test_main_eval_only_skip_llm_judges_flag(self):
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            ds.write_text(json.dumps([{"id": "a", "gt": "hi", "pred": "hi"}]))
            out = base / "out"

            fake_result = {"status": "completed", "metrics": {"wer": 0.1}}
            mock_eval = AsyncMock(return_value=fake_result)
            argv = ["b.py", "--eval-only", "--dataset", str(ds),
                    "-o", str(out), "--skip-llm-judges", "-y"]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run_eval_only", mock_eval):
                await B.main()

            self.assertEqual(mock_eval.call_args.kwargs["llm_judges"], frozenset())

    async def test_main_eval_only_judges_subset_flag(self):
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            ds.write_text(json.dumps([{"id": "a", "gt": "hi", "pred": "hi"}]))
            out = base / "out"

            fake_result = {"status": "completed", "metrics": {"wer": 0.1}}
            mock_eval = AsyncMock(return_value=fake_result)
            argv = [
                "b.py",
                "--eval-only",
                "--dataset",
                str(ds),
                "-o",
                str(out),
                "--judges",
                "intent,llm_wer",
                "-y",
            ]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run_eval_only", mock_eval):
                await B.main()

            self.assertEqual(
                mock_eval.call_args.kwargs["llm_judges"],
                frozenset({"intent", "llm_wer"}),
            )

    async def test_main_eval_only_error(self):
        from calibrate_agent.stt import benchmark as B

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
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["b.py", "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_no_input_dir(self):
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["b.py", "-p", "deepgram", "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_success_path(self):
        from calibrate_agent.stt import benchmark as B

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

    async def test_main_forwards_intent_entity_flag_to_run(self):
        from calibrate_agent._cli_args import DEFAULT_STT_LLM_JUDGES
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_input_dir(base)
            out = base / "out"

            fake_run_result = {
                "status": "completed",
                "output_dir": str(out),
                "leaderboard_dir": str(out / "leaderboard"),
                "providers": {
                    "deepgram": {"status": "completed", "metrics": {"wer": 0.1}},
                },
            }
            mock_run = AsyncMock(return_value=fake_run_result)

            # Default on.
            argv = ["b.py", "-p", "deepgram", "-i", str(base), "-o", str(out), "-y"]
            with patch.object(sys, "argv", argv), patch.object(B, "run", mock_run):
                await B.main()
            self.assertEqual(
                mock_run.call_args.kwargs["llm_judges"], DEFAULT_STT_LLM_JUDGES
            )

            # Skip flag.
            argv.append("--skip-llm-judges")
            with patch.object(sys, "argv", argv), patch.object(B, "run", mock_run):
                await B.main()
            self.assertEqual(mock_run.call_args.kwargs["llm_judges"], frozenset())

    async def test_main_provider_judges_subset_flag(self):
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_input_dir(base)
            out = base / "out"

            fake_run_result = {
                "status": "completed",
                "output_dir": str(out),
                "leaderboard_dir": str(out / "leaderboard"),
                "providers": {
                    "deepgram": {"status": "completed", "metrics": {"wer": 0.1}},
                },
            }
            mock_run = AsyncMock(return_value=fake_run_result)
            argv = [
                "b.py",
                "-p",
                "deepgram",
                "-i",
                str(base),
                "-o",
                str(out),
                "--judges",
                "intent",
                "-y",
            ]
            with patch.object(sys, "argv", argv), patch.object(B, "run", mock_run):
                await B.main()
            self.assertEqual(
                mock_run.call_args.kwargs["llm_judges"], frozenset({"intent"})
            )

    async def test_main_error_provider_exits(self):
        from calibrate_agent.stt import benchmark as B

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
        from calibrate_agent.stt import benchmark as B

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

    async def test_main_eval_only_cancel_on_decline(self):
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "ds.json"
            ds.write_text(json.dumps([{"id": "a", "gt": "hi", "pred": "hi"}]))
            out = base / "out"

            buf = io.StringIO()
            argv = ["b.py", "--eval-only", "--dataset", str(ds), "-o", str(out)]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run_eval_only", AsyncMock()) as mock_eval, \
                 patch(
                     "calibrate_agent.stt.benchmark.confirm_estimated_judge_cost",
                     return_value=False,
                 ), redirect_stdout(buf):
                with self.assertRaises(SystemExit) as ctx:
                    await B.main()

            self.assertEqual(ctx.exception.code, 0)
            mock_eval.assert_not_called()
            self.assertIn("Judge run cancelled.", buf.getvalue())

    async def test_main_multi_provider_cancel_on_decline(self):
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_input_dir(base)
            out = base / "out"

            buf = io.StringIO()
            argv = ["b.py", "-p", "deepgram", "-i", str(base), "-o", str(out)]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run", AsyncMock()) as mock_run, \
                 patch(
                     "calibrate_agent.stt.benchmark.confirm_estimated_judge_cost",
                     return_value=False,
                 ), redirect_stdout(buf):
                with self.assertRaises(SystemExit) as ctx:
                    await B.main()

            self.assertEqual(ctx.exception.code, 0)
            mock_run.assert_not_called()
            self.assertIn("Judge run cancelled.", buf.getvalue())

    async def test_main_debug_yes_prints_cached_count_note(self):
        from calibrate_agent.stt import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_input_dir(base)
            out = base / "out"
            _write_judge_cache(out / "deepgram", kind="stt")

            fake_run_result = {
                "status": "completed",
                "output_dir": str(out),
                "leaderboard_dir": str(out / "leaderboard"),
                "providers": {
                    "deepgram": {"status": "completed", "metrics": {"wer": 0.1}},
                },
            }
            argv = ["b.py", "-p", "deepgram", "-i", str(base), "-o", str(out), "-d", "-y"]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run", AsyncMock(return_value=fake_run_result)):
                await B.main()

            log_content = Path(join(out, "logs")).read_text()
            self.assertIn("checkpointed", log_content)


# =============================================================================
# TTS Benchmark
# =============================================================================

class TestTTSBenchmarkRun(unittest.IsolatedAsyncioTestCase):
    async def test_run_basic(self):
        from calibrate_agent.tts import benchmark as B

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
        from calibrate_agent.tts import benchmark as B

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
        from calibrate_agent.tts import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            argv = ["b.py", "-p", "bogus", "-i", str(inp), "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await B.main()

    async def test_main_invalid_input(self):
        from calibrate_agent.tts import benchmark as B

        argv = ["b.py", "-p", "openai", "-i", "/nonexistent.csv", "-o", "/tmp/x"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                await B.main()

    async def test_main_success(self):
        from calibrate_agent.tts import benchmark as B

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
        from calibrate_agent.tts import benchmark as B

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

    async def test_main_cancel_on_decline(self):
        from calibrate_agent.tts import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            out = Path(tmp) / "out"

            buf = io.StringIO()
            argv = ["b.py", "-p", "openai", "-i", str(inp), "-o", str(out)]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run", AsyncMock()) as mock_run, \
                 patch(
                     "calibrate_agent.tts.benchmark.confirm_estimated_judge_cost",
                     return_value=False,
                 ), redirect_stdout(buf):
                with self.assertRaises(SystemExit) as ctx:
                    await B.main()

            self.assertEqual(ctx.exception.code, 0)
            mock_run.assert_not_called()
            self.assertIn("Judge run cancelled.", buf.getvalue())

    async def test_main_yes_proceeds_to_run(self):
        from calibrate_agent.tts import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            out = Path(tmp) / "out"

            fake_run_result = {
                "status": "completed",
                "output_dir": str(out),
                "leaderboard_dir": str(out / "lb"),
                "providers": {
                    "openai": {"status": "completed", "metrics": {}},
                },
            }
            mock_run = AsyncMock(return_value=fake_run_result)
            argv = ["b.py", "-p", "openai", "-i", str(inp), "-o", str(out), "-y"]
            with patch.object(sys, "argv", argv), patch.object(B, "run", mock_run):
                await B.main()

            mock_run.assert_called_once()

    async def test_main_yes_prints_cached_count_note(self):
        from calibrate_agent.tts import benchmark as B

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            out = Path(tmp) / "out"
            _write_judge_cache(out / "openai")

            fake_run_result = {
                "status": "completed",
                "output_dir": str(out),
                "leaderboard_dir": str(out / "lb"),
                "providers": {
                    "openai": {"status": "completed", "metrics": {}},
                },
            }
            argv = ["b.py", "-p", "openai", "-i", str(inp), "-o", str(out), "-y"]
            with patch.object(sys, "argv", argv), \
                 patch.object(B, "run", AsyncMock(return_value=fake_run_result)):
                await B.main()

            log_content = Path(join(out, "logs")).read_text()
            self.assertIn("checkpointed", log_content)


if __name__ == "__main__":
    unittest.main()
