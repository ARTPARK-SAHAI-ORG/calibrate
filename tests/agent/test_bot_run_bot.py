"""Test agent/bot.py run_bot function with heavy mocking."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


def _make_runner_args():
    args = MagicMock()
    args.pipeline_idle_timeout_secs = 30
    args.handle_sigint = False
    return args


class TestAgentBotRunBot(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_language_raises(self):
        from calibrate.agent import bot as B

        with self.assertRaises(ValueError):
            await B.run_bot(
                transport=MagicMock(),
                runner_args=_make_runner_args(),
                system_prompt="sp",
                tools=[],
                stt_config=B.STTConfig(),
                tts_config=B.TTSConfig(),
                llm_config=B.LLMConfig(),
                language="klingon",
            )

    async def test_basic_run_bot_openrouter(self):
        from calibrate.agent import bot as B

        fake_llm = MagicMock()
        fake_llm.register_function = MagicMock()
        fake_transport = MagicMock()
        fake_transport.event_handler = lambda name: lambda fn: fn
        fake_transport.input.return_value = MagicMock()
        fake_transport.output.return_value = MagicMock()
        fake_runner = MagicMock()
        fake_runner.run = AsyncMock(return_value=None)
        fake_context = MagicMock()
        fake_context.get_messages.return_value = []

        with patch.object(B, "create_stt_service", MagicMock()), \
             patch.object(B, "create_tts_service", MagicMock()), \
             patch.object(B, "OpenAILLMService"), \
             patch.object(B, "OpenRouterLLMService", return_value=fake_llm), \
             patch.object(B, "RTVIProcessor"), \
             patch.object(B, "SileroVADAnalyzer"), \
             patch.object(B, "LocalSmartTurnAnalyzerV3"), \
             patch.object(B, "Pipeline"), \
             patch.object(B, "PipelineTask"), \
             patch.object(B, "PipelineRunner", return_value=fake_runner), \
             patch.object(B, "LLMContext", return_value=fake_context), \
             patch.object(B, "LLMContextAggregatorPair"):
            await B.run_bot(
                transport=fake_transport,
                runner_args=_make_runner_args(),
                system_prompt="sp",
                tools=[],
                stt_config=B.STTConfig(),
                tts_config=B.TTSConfig(),
                llm_config=B.LLMConfig(provider="openrouter"),
                language="english",
            )

    async def test_basic_run_bot_openai(self):
        from calibrate.agent import bot as B

        fake_llm = MagicMock()
        fake_llm.register_function = MagicMock()
        fake_transport = MagicMock()
        fake_transport.event_handler = lambda name: lambda fn: fn
        fake_transport.input.return_value = MagicMock()
        fake_transport.output.return_value = MagicMock()
        fake_runner = MagicMock()
        fake_runner.run = AsyncMock(return_value=None)
        fake_context = MagicMock()
        fake_context.get_messages.return_value = []

        with patch.object(B, "create_stt_service", MagicMock()), \
             patch.object(B, "create_tts_service", MagicMock()), \
             patch.object(B, "OpenAILLMService", return_value=fake_llm), \
             patch.object(B, "OpenRouterLLMService"), \
             patch.object(B, "RTVIProcessor"), \
             patch.object(B, "SileroVADAnalyzer"), \
             patch.object(B, "LocalSmartTurnAnalyzerV3"), \
             patch.object(B, "Pipeline"), \
             patch.object(B, "PipelineTask"), \
             patch.object(B, "PipelineRunner", return_value=fake_runner), \
             patch.object(B, "LLMContext", return_value=fake_context), \
             patch.object(B, "LLMContextAggregatorPair"):
            await B.run_bot(
                transport=fake_transport,
                runner_args=_make_runner_args(),
                system_prompt="sp",
                tools=[],
                stt_config=B.STTConfig(),
                tts_config=B.TTSConfig(),
                llm_config=B.LLMConfig(provider="openai"),
                language="english",
            )


if __name__ == "__main__":
    unittest.main()
