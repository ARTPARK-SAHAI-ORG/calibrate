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
        from calibrate_agent.agent import bot as B

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
        from calibrate_agent.agent import bot as B

        fake_llm = MagicMock()
        fake_llm.register_function = MagicMock()
        fake_transport = MagicMock()
        fake_transport.event_handler = lambda name: lambda fn: fn
        fake_transport.input.return_value = MagicMock()
        fake_transport.output.return_value = MagicMock()
        fake_runner = MagicMock()
        fake_runner.add_workers = AsyncMock()
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
             patch.object(B, "PipelineWorker") as worker_cls, \
             patch.object(B, "WorkerRunner", return_value=fake_runner), \
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
        fake_runner.add_workers.assert_awaited_once_with(worker_cls.return_value)
        fake_runner.run.assert_awaited_once_with()

    async def test_basic_run_bot_openai(self):
        from calibrate_agent.agent import bot as B

        fake_llm = MagicMock()
        fake_llm.register_function = MagicMock()
        fake_transport = MagicMock()
        fake_transport.event_handler = lambda name: lambda fn: fn
        fake_transport.input.return_value = MagicMock()
        fake_transport.output.return_value = MagicMock()
        fake_runner = MagicMock()
        fake_runner.add_workers = AsyncMock()
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
             patch.object(B, "PipelineWorker"), \
             patch.object(B, "WorkerRunner", return_value=fake_runner), \
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


class TestRunBotTurnStrategies(unittest.IsolatedAsyncioTestCase):
    async def _user_turn_strategies(self, decides_turns):
        from calibrate_agent.agent import bot as B

        fake_transport = MagicMock()
        fake_transport.event_handler = lambda name: lambda fn: fn
        fake_runner = MagicMock()
        fake_runner.add_workers = AsyncMock()
        fake_runner.run = AsyncMock(return_value=None)
        fake_context = MagicMock()
        fake_context.get_messages.return_value = []

        with patch.object(B, "create_stt_service", MagicMock()), \
             patch.object(B, "stt_decides_turns", return_value=decides_turns), \
             patch.object(B, "create_tts_service", MagicMock()), \
             patch.object(B, "OpenAILLMService"), \
             patch.object(B, "OpenRouterLLMService", return_value=MagicMock()), \
             patch.object(B, "RTVIProcessor"), \
             patch.object(B, "SileroVADAnalyzer"), \
             patch.object(B, "LocalSmartTurnAnalyzerV3"), \
             patch.object(B, "Pipeline"), \
             patch.object(B, "PipelineWorker"), \
             patch.object(B, "WorkerRunner", return_value=fake_runner), \
             patch.object(B, "LLMContext", return_value=fake_context), \
             patch.object(B, "LLMContextAggregatorPair") as pair:
            await B.run_bot(
                transport=fake_transport,
                runner_args=_make_runner_args(),
                system_prompt="sp",
                tools=[],
                stt_config=B.STTConfig(provider="deepgram-flux"),
                tts_config=B.TTSConfig(),
                llm_config=B.LLMConfig(provider="openrouter"),
                language="english",
            )
        return pair.call_args.kwargs["user_params"].user_turn_strategies

    async def test_provider_that_decides_turns_gets_no_agent_strategies(self):
        self.assertIsNone(await self._user_turn_strategies(True))

    async def test_other_providers_use_smart_turn(self):
        from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy

        strategies = await self._user_turn_strategies(False)
        self.assertTrue(
            any(isinstance(s, TurnAnalyzerUserTurnStopStrategy) for s in strategies.stop)
        )
