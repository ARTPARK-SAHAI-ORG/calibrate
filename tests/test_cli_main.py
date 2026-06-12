"""Tests for calibrate/cli.py — entry point and subcommand dispatching."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


class TestArgsToArgv(unittest.TestCase):
    def test_basic_conversion(self):
        from calibrate.cli import _args_to_argv

        ns = MagicMock()
        ns.__dict__ = {"input_dir": "/tmp/x", "debug": True, "count": 5}
        # use real namespace
        import argparse
        args = argparse.Namespace(input_dir="/tmp/x", debug=True, count=5, none_val=None)
        argv = _args_to_argv(args)
        self.assertIn("--input-dir", argv)
        self.assertIn("/tmp/x", argv)
        self.assertIn("--debug", argv)
        self.assertNotIn("--none-val", argv)

    def test_with_exclude_keys(self):
        from calibrate.cli import _args_to_argv
        import argparse

        args = argparse.Namespace(a="1", b="2")
        argv = _args_to_argv(args, exclude_keys={"a"})
        self.assertNotIn("--a", argv)
        self.assertIn("--b", argv)

    def test_with_flag_mapping(self):
        from calibrate.cli import _args_to_argv
        import argparse

        args = argparse.Namespace(debug_count=5)
        argv = _args_to_argv(args, flag_mapping={"debug_count": "-dc"})
        self.assertIn("-dc", argv)


class TestLoadCliDotenv(unittest.TestCase):
    def test_loads_dotenv_from_current_working_directory(self):
        from calibrate import cli

        with patch.object(cli, "find_dotenv", return_value="/project/src/.env") as find, \
             patch.object(cli, "load_dotenv") as load:
            cli._load_cli_dotenv()

        find.assert_called_once_with(usecwd=True)
        load.assert_called_once_with("/project/src/.env", override=True)


class TestLaunchInkUI(unittest.TestCase):
    def test_no_node(self):
        from calibrate import cli

        with patch("shutil.which", return_value=None):
            with self.assertRaises(SystemExit):
                cli._launch_ink_ui("stt")

    def test_no_bundle(self):
        from calibrate import cli

        with patch("shutil.which", return_value="/usr/bin/node"), \
             patch("pathlib.Path.exists", return_value=False):
            with self.assertRaises(SystemExit):
                cli._launch_ink_ui("stt")

    def test_runs_subprocess(self):
        from calibrate import cli

        fake_result = MagicMock(returncode=0)
        with patch("shutil.which", return_value="/usr/bin/node"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("subprocess.run", return_value=fake_result):
            with self.assertRaises(SystemExit) as ctx:
                cli._launch_ink_ui("stt")
        self.assertEqual(ctx.exception.code, 0)


class TestPrintSampleOutput(unittest.TestCase):
    def test_no_sample(self):
        from calibrate.cli import _print_sample_output

        _print_sample_output({})  # Should not crash

    def test_dict_with_response(self):
        from calibrate.cli import _print_sample_output

        _print_sample_output({"sample_output": {"response": "Hi", "tool_calls": [{"x": 1}]}})

    def test_dict_with_only_response(self):
        from calibrate.cli import _print_sample_output

        _print_sample_output({"sample_output": {"response": "Hi"}})

    def test_dict_with_only_tool_calls(self):
        from calibrate.cli import _print_sample_output

        _print_sample_output({"sample_output": {"response": None, "tool_calls": [{"x": 1}]}})

    def test_non_dict_sample(self):
        from calibrate.cli import _print_sample_output

        _print_sample_output({"sample_output": "some string"})


class TestRunAgentVerify(unittest.TestCase):
    def test_invalid_json_headers(self):
        from calibrate import cli

        with self.assertRaises(SystemExit):
            cli._run_agent_verify("http://x", "{bad json")

    def test_success_with_model(self):
        from calibrate import cli

        fake_result = {"ok": True, "sample_output": {"response": "Hi", "tool_calls": []}}
        with patch("calibrate.connections.TextAgentConnection.verify",
                   AsyncMock(return_value=fake_result)):
            cli._run_agent_verify("http://x", '{"K": "V"}', models=["m1"])

    def test_failure_exits(self):
        from calibrate import cli

        fake_result = {"ok": False, "error": "boom"}
        with patch("calibrate.connections.TextAgentConnection.verify",
                   AsyncMock(return_value=fake_result)):
            with self.assertRaises(SystemExit):
                cli._run_agent_verify("http://x", None)


class TestMainDispatch(unittest.TestCase):
    def _run_with_argv(self, argv):
        from calibrate.cli import main
        with patch.object(sys, "argv", argv):
            main()

    def test_no_component_launches_menu(self):
        from calibrate import cli
        with patch.object(cli, "_launch_ink_ui") as mock:
            mock.side_effect = SystemExit(0)
            with self.assertRaises(SystemExit):
                self._run_with_argv(["calibrate"])

    def test_stt_no_provider_launches_ui(self):
        from calibrate import cli
        with patch.object(cli, "_launch_ink_ui") as mock:
            mock.side_effect = SystemExit(0)
            with self.assertRaises(SystemExit):
                self._run_with_argv(["calibrate", "stt"])

    def test_stt_with_provider_runs_benchmark(self):
        from calibrate import cli
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            import pandas as pd
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(
                base / "stt.csv", index=False
            )
            (base / "audios" / "a.wav").write_bytes(b"\x00")

            with patch("calibrate.stt.benchmark.main", AsyncMock(return_value=None)):
                self._run_with_argv([
                    "calibrate", "stt", "-p", "deepgram",
                    "-i", str(base), "-o", str(base / "out"),
                    "-d", "--ignore_retry", "--overwrite",
                ])

    def test_stt_eval_only_no_dataset_exits(self):
        with self.assertRaises(SystemExit):
            self._run_with_argv(["calibrate", "stt", "--eval-only"])

    def test_stt_eval_only_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "ds.json"
            ds.write_text("[]")
            with patch("calibrate.stt.benchmark.main", AsyncMock(return_value=None)):
                self._run_with_argv([
                    "calibrate", "stt", "--eval-only", "--dataset", str(ds),
                    "-o", tmp,
                ])

    def test_tts_no_provider_launches_ui(self):
        from calibrate import cli
        with patch.object(cli, "_launch_ink_ui") as mock:
            mock.side_effect = SystemExit(0)
            with self.assertRaises(SystemExit):
                self._run_with_argv(["calibrate", "tts"])

    def test_tts_with_provider_runs_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.csv"
            import pandas as pd
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(str(inp), index=False)
            with patch("calibrate.tts.benchmark.main", AsyncMock(return_value=None)):
                self._run_with_argv([
                    "calibrate", "tts", "-p", "openai",
                    "-i", str(inp), "-o", tmp,
                    "-d", "--overwrite",
                ])

    def test_llm_no_config_launches_ui(self):
        from calibrate import cli
        with patch.object(cli, "_launch_ink_ui") as mock:
            mock.side_effect = SystemExit(0)
            with self.assertRaises(SystemExit):
                self._run_with_argv(["calibrate", "llm"])

    def test_llm_verify_no_url_exits(self):
        with self.assertRaises(SystemExit):
            self._run_with_argv(["calibrate", "llm", "--verify"])

    def test_llm_verify_with_url(self):
        from calibrate import cli
        with patch.object(cli, "_run_agent_verify") as mock:
            self._run_with_argv([
                "calibrate", "llm", "--verify",
                "--agent-url", "http://x",
            ])
            mock.assert_called_once()

    def test_llm_eval_only_no_config_exits(self):
        with self.assertRaises(SystemExit):
            self._run_with_argv(["calibrate", "llm", "--eval-only", "--dataset", "/tmp/x.json"])

    def test_llm_eval_only_no_dataset_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text("{}")
            with self.assertRaises(SystemExit):
                self._run_with_argv([
                    "calibrate", "llm", "--eval-only",
                    "-c", str(cfg),
                ])

    def test_llm_eval_only_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text("{}")
            ds = Path(tmp) / "ds.json"
            ds.write_text("[]")
            with patch("calibrate.llm.run_tests.main", AsyncMock(return_value=None)):
                self._run_with_argv([
                    "calibrate", "llm", "--eval-only",
                    "-c", str(cfg), "--dataset", str(ds),
                    "-o", tmp,
                ])

    def test_llm_with_config_no_agent_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({"system_prompt": "sp", "tools": [], "test_cases": []}))
            with patch("calibrate.llm.benchmark.main", AsyncMock(return_value=None)):
                self._run_with_argv([
                    "calibrate", "llm", "-c", str(cfg),
                    "-o", tmp, "-m", "gpt-4.1",
                ])

    def test_llm_with_config_agent_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({
                "agent_url": "http://x", "agent_headers": None,
                "test_cases": [],
            }))
            fake_verify = {"ok": True, "sample_output": {"response": "Hi", "tool_calls": []}}
            with patch("calibrate.connections.TextAgentConnection.verify",
                       AsyncMock(return_value=fake_verify)), \
                 patch("calibrate.llm.tests_leaderboard.generate_leaderboard"), \
                 patch("calibrate.llm.tests.run",
                       AsyncMock(return_value={"m1": {"metrics": {"passed": 1, "total": 1}}})):
                self._run_with_argv([
                    "calibrate", "llm", "-c", str(cfg),
                    "-o", tmp, "-m", "m1", "--skip-verify",
                ])

    def test_llm_with_config_agent_url_no_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({
                "agent_url": "http://x", "test_cases": [],
            }))
            with patch("calibrate.connections.TextAgentConnection.verify",
                       AsyncMock(return_value={"ok": True, "sample_output": {}})), \
                 patch("calibrate.llm.tests.run", AsyncMock(return_value={})):
                self._run_with_argv([
                    "calibrate", "llm", "-c", str(cfg),
                    "-o", tmp,
                ])

    def test_llm_eval_only_debug_forwards_flags(self):
        captured = {}

        def _capture():
            captured["argv"] = list(sys.argv)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text("{}")
            ds = Path(tmp) / "ds.json"
            ds.write_text("[]")
            with patch("calibrate.llm.run_tests.main",
                       AsyncMock(side_effect=_capture)):
                self._run_with_argv([
                    "calibrate", "llm", "--eval-only",
                    "-c", str(cfg), "--dataset", str(ds), "-o", tmp,
                    "-d", "-dc", "2",
                ])
        argv = captured["argv"]
        self.assertIn("-d", argv)
        self.assertIn("-dc", argv)
        self.assertIn("2", argv)

    def test_llm_benchmark_debug_forwards_flags(self):
        captured = {}

        def _capture():
            captured["argv"] = list(sys.argv)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps(
                {"system_prompt": "sp", "tools": [], "test_cases": []}
            ))
            with patch("calibrate.llm.benchmark.main",
                       AsyncMock(side_effect=_capture)):
                self._run_with_argv([
                    "calibrate", "llm", "-c", str(cfg),
                    "-o", tmp, "-m", "gpt-4.1", "-d", "-dc", "2",
                ])
        argv = captured["argv"]
        self.assertIn("-d", argv)
        self.assertIn("-dc", argv)
        self.assertIn("2", argv)

    def test_llm_agent_debug_truncates_test_cases(self):
        captured = {}

        async def fake_run(**kwargs):
            captured["test_cases"] = kwargs.get("test_cases")
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({
                "agent_url": "http://x",
                "test_cases": [
                    {"id": str(i), "history": [], "evaluation": {"type": "tool_call"}}
                    for i in range(4)
                ],
            }))
            with patch("calibrate.llm.tests.run", side_effect=fake_run):
                self._run_with_argv([
                    "calibrate", "llm", "-c", str(cfg),
                    "-o", tmp, "--skip-verify", "-d", "-dc", "1",
                ])
        self.assertEqual(len(captured["test_cases"]), 1)

    def test_simulations_verify(self):
        from calibrate import cli
        with patch.object(cli, "_run_agent_verify") as mock:
            self._run_with_argv([
                "calibrate", "simulations", "--verify",
                "--agent-url", "http://x",
            ])
            mock.assert_called_once()

    def test_simulations_verify_no_url_exits(self):
        with self.assertRaises(SystemExit):
            self._run_with_argv(["calibrate", "simulations", "--verify"])

    def test_simulations_no_type_launches_ui(self):
        from calibrate import cli
        with patch.object(cli, "_launch_ink_ui") as mock:
            mock.side_effect = SystemExit(0)
            with self.assertRaises(SystemExit):
                self._run_with_argv(["calibrate", "simulations"])

    def test_simulations_text_eval_only_no_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text("{}")
            with self.assertRaises(SystemExit):
                self._run_with_argv([
                    "calibrate", "simulations", "--type", "text",
                    "-c", str(cfg), "--eval-only",
                ])

    def test_simulations_text_eval_only_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text("{}")
            ds = Path(tmp) / "ds.json"
            ds.write_text("[]")
            with patch("calibrate.llm.run_simulation.main", AsyncMock(return_value=None)):
                self._run_with_argv([
                    "calibrate", "simulations", "--type", "text",
                    "-c", str(cfg), "--eval-only", "--dataset", str(ds),
                ])

    def test_simulations_text_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text(json.dumps({}))
            with patch("calibrate.llm.run_simulation.main", AsyncMock(return_value=None)):
                self._run_with_argv([
                    "calibrate", "simulations", "--type", "text",
                    "-c", str(cfg), "-o", tmp,
                ])

    def test_simulations_text_with_agent_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text(json.dumps({"agent_url": "http://x"}))
            with patch("calibrate.connections.TextAgentConnection.verify",
                       AsyncMock(return_value={"ok": True, "sample_output": {}})), \
                 patch("calibrate.llm.run_simulation.main", AsyncMock(return_value=None)):
                self._run_with_argv([
                    "calibrate", "simulations", "--type", "text",
                    "-c", str(cfg), "-o", tmp,
                ])

    def test_simulations_text_with_agent_url_verify_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text(json.dumps({"agent_url": "http://x"}))
            with patch("calibrate.connections.TextAgentConnection.verify",
                       AsyncMock(return_value={"ok": False, "error": "boom"})):
                with self.assertRaises(SystemExit):
                    self._run_with_argv([
                        "calibrate", "simulations", "--type", "text",
                        "-c", str(cfg), "-o", tmp,
                    ])

    def test_simulations_leaderboard(self):
        with patch("calibrate.llm.simulation_leaderboard.main") as mock:
            self._run_with_argv([
                "calibrate", "simulations", "leaderboard",
                "-o", "/tmp/out", "-s", "/tmp/save",
            ])
            mock.assert_called_once()

    def test_simulations_voice_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text("{}")
            with patch("calibrate.agent.run_simulation.main", AsyncMock(return_value=None)):
                self._run_with_argv([
                    "calibrate", "simulations", "--type", "voice",
                    "-c", str(cfg), "-o", tmp,
                ])

    def test_status_default(self):
        with patch("calibrate.status.run_status_live",
                   AsyncMock(return_value={"openai": {"status": "pass"}})):
            self._run_with_argv(["calibrate", "status"])

    def test_status_table(self):
        with patch("calibrate.status.run_status_live",
                   AsyncMock(return_value={"openai": {"status": "pass"}})):
            self._run_with_argv(["calibrate", "status", "--table"])

    def test_agent_test_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text("{}")
            with patch("runpy.run_path") as mock:
                self._run_with_argv([
                    "calibrate", "agent", "test",
                    "-c", str(cfg), "-o", tmp,
                ])
                mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
