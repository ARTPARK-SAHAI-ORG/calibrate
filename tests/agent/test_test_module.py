"""Tests for calibrate_agent/agent/test.py — CLI args, parse_bot_config, bot() entry."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


class TestStoreAndGetCliArgs(unittest.TestCase):
    def test_store_and_get(self):
        from calibrate_agent.agent import test as T

        T._store_cli_args(
            argparse.Namespace(config="/tmp/cfg.json", output_dir="/tmp/out")
        )
        self.assertEqual(T.get_cli_arg("config"), "/tmp/cfg.json")
        self.assertEqual(T.get_cli_arg("output_dir"), "/tmp/out")

    def test_get_default(self):
        from calibrate_agent.agent import test as T

        T._store_cli_args(argparse.Namespace())
        self.assertIsNone(T.get_cli_arg("missing"))
        self.assertEqual(T.get_cli_arg("missing", "fallback"), "fallback")

    def test_filters_none(self):
        from calibrate_agent.agent import test as T

        T._store_cli_args(argparse.Namespace(a="set", b=None))
        self.assertEqual(T.get_cli_arg("a"), "set")
        self.assertIsNone(T.get_cli_arg("b"))


class TestParseBotConfig(unittest.TestCase):
    def test_minimal_config(self):
        from calibrate_agent.agent.test import parse_bot_config

        cfg = parse_bot_config({"system_prompt": "be helpful"})
        self.assertEqual(cfg.system_prompt, "be helpful")
        self.assertEqual(cfg.language, "english")
        self.assertEqual(cfg.tools, [])
        self.assertEqual(cfg.stt.provider, "elevenlabs")
        self.assertEqual(cfg.tts.provider, "elevenlabs")
        self.assertEqual(cfg.llm.provider, "openrouter")

    def test_full_config(self):
        from calibrate_agent.agent.test import parse_bot_config

        cfg = parse_bot_config(
            {
                "system_prompt": "sp",
                "language": "hindi",
                "tools": [{"name": "x"}],
                "stt": {"provider": "deepgram", "model": "nova-3"},
                "tts": {"provider": "google", "voice_id": "v1", "model": "m"},
                "llm": {"provider": "openai", "model": "gpt-4", "api_key": "k"},
            }
        )
        self.assertEqual(cfg.language, "hindi")
        self.assertEqual(cfg.stt.provider, "deepgram")
        self.assertEqual(cfg.stt.model, "nova-3")
        self.assertEqual(cfg.tts.voice_id, "v1")
        self.assertEqual(cfg.llm.provider, "openai")

    def test_missing_system_prompt_raises(self):
        from calibrate_agent.agent.test import parse_bot_config

        with self.assertRaises(ValueError):
            parse_bot_config({})


class TestBotEntryPoint(unittest.IsolatedAsyncioTestCase):
    async def test_bot_missing_config_raises(self):
        from calibrate_agent.agent import test as T

        T._store_cli_args(argparse.Namespace())
        with self.assertRaises(RuntimeError):
            await T.bot(MagicMock())

    async def test_bot_runs(self):
        from calibrate_agent.agent import test as T

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(json.dumps({"system_prompt": "sp"}))
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()

            T._store_cli_args(
                argparse.Namespace(
                    config=str(cfg_path),
                    output_dir=str(out_dir),
                )
            )

            # Mock create_transport and run_bot to avoid actual pipecat work
            fake_transport = MagicMock()
            runner_args = MagicMock()
            runner_args.pipeline_idle_timeout_secs = 30
            runner_args.handle_sigint = False

            with (
                patch.object(
                    T, "create_transport", AsyncMock(return_value=fake_transport)
                ),
                patch.object(T, "run_bot", AsyncMock(return_value=None)),
            ):
                await T.bot(runner_args)

    async def test_bot_pre_existing_logs(self):
        from calibrate_agent.agent import test as T

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(json.dumps({"system_prompt": "sp"}))
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()
            (out_dir / "logs").write_text("existing")

            T._store_cli_args(
                argparse.Namespace(
                    config=str(cfg_path),
                    output_dir=str(out_dir),
                )
            )

            with (
                patch.object(
                    T, "create_transport", AsyncMock(return_value=MagicMock())
                ),
                patch.object(T, "run_bot", AsyncMock(return_value=None)),
            ):
                await T.bot(MagicMock())


if __name__ == "__main__":
    unittest.main()
