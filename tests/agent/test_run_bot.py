"""Test agent/test.py run_bot function with heavy pipecat mocking."""

import tempfile
import unittest
from unittest.mock import patch, AsyncMock, MagicMock


def _make_bot_config(provider="openrouter"):
    from calibrate_agent.agent.test import BotConfig, STTConfig, TTSConfig, LLMConfig

    return BotConfig(
        system_prompt="You are helpful",
        language="english",
        tools=[],
        stt=STTConfig(provider="elevenlabs"),
        tts=TTSConfig(provider="elevenlabs"),
        llm=LLMConfig(provider=provider, model="m1"),
    )


class TestRunBot(unittest.IsolatedAsyncioTestCase):
    async def test_run_bot_basic(self):
        from calibrate_agent.agent import test as T

        fake_llm = MagicMock()
        fake_llm.register_function = MagicMock()
        fake_transport = MagicMock()
        fake_transport.event_handler = lambda name: lambda fn: fn
        fake_transport.input.return_value = MagicMock()
        fake_transport.output.return_value = MagicMock()
        fake_runner = MagicMock()
        fake_runner.run = AsyncMock(return_value=None)

        fake_context = MagicMock()
        fake_context.get_messages.return_value = [
            {"role": "system", "content": "sp"},
            {"role": "user", "content": "Hi"},
        ]

        runner_args = MagicMock()
        runner_args.pipeline_idle_timeout_secs = 30
        runner_args.handle_sigint = False

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(T, "create_stt_service", MagicMock()), \
             patch.object(T, "create_tts_service", MagicMock()), \
             patch.object(T, "OpenAILLMService"), \
             patch.object(T, "OpenRouterLLMService", return_value=fake_llm), \
             patch.object(T, "RTVIProcessor"), \
             patch.object(T, "SileroVADAnalyzer"), \
             patch.object(T, "LocalSmartTurnAnalyzerV3"), \
             patch.object(T, "AudioBufferProcessor"), \
             patch.object(T, "Pipeline"), \
             patch.object(T, "PipelineTask"), \
             patch.object(T, "PipelineRunner", return_value=fake_runner), \
             patch.object(T, "LLMContext", return_value=fake_context), \
             patch.object(T, "LLMContextAggregatorPair"):
            await T.run_bot(_make_bot_config("openrouter"), fake_transport, runner_args, tmp)

    async def test_run_bot_openai(self):
        from calibrate_agent.agent import test as T

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

        runner_args = MagicMock()
        runner_args.pipeline_idle_timeout_secs = 30
        runner_args.handle_sigint = False

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(T, "create_stt_service", MagicMock()), \
             patch.object(T, "create_tts_service", MagicMock()), \
             patch.object(T, "OpenAILLMService", return_value=fake_llm), \
             patch.object(T, "OpenRouterLLMService"), \
             patch.object(T, "RTVIProcessor"), \
             patch.object(T, "SileroVADAnalyzer"), \
             patch.object(T, "LocalSmartTurnAnalyzerV3"), \
             patch.object(T, "AudioBufferProcessor"), \
             patch.object(T, "Pipeline"), \
             patch.object(T, "PipelineTask"), \
             patch.object(T, "PipelineRunner", return_value=fake_runner), \
             patch.object(T, "LLMContext", return_value=fake_context), \
             patch.object(T, "LLMContextAggregatorPair"):
            await T.run_bot(_make_bot_config("openai"), fake_transport, runner_args, tmp)

    async def test_run_bot_unknown_provider(self):
        from calibrate_agent.agent import test as T
        from calibrate_agent.agent.test import BotConfig, STTConfig, TTSConfig, LLMConfig

        # Bypass dataclass validation by setting __dict__
        cfg = BotConfig(
            system_prompt="sp", language="english", tools=[],
            stt=STTConfig(), tts=TTSConfig(), llm=LLMConfig(),
        )
        cfg.llm.provider = "bogus"

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(T, "create_stt_service", MagicMock()), \
             patch.object(T, "create_tts_service", MagicMock()):
            with self.assertRaises(ValueError):
                await T.run_bot(cfg, MagicMock(), MagicMock(), tmp)

    async def test_run_bot_with_webhook_tool(self):
        from calibrate_agent.agent import test as T
        from calibrate_agent.agent.test import BotConfig, STTConfig, TTSConfig, LLMConfig

        cfg = BotConfig(
            system_prompt="sp", language="english",
            tools=[
                {"name": "wh", "description": "wh", "type": "webhook", "webhook": {
                    "url": "http://x", "method": "POST", "headers": [],
                }},
            ],
            stt=STTConfig(), tts=TTSConfig(), llm=LLMConfig(),
        )

        fake_llm = MagicMock()
        registered = {}

        def reg(name, fn):
            registered[name] = fn
        fake_llm.register_function = reg

        fake_transport = MagicMock()
        fake_transport.event_handler = lambda name: lambda fn: fn
        fake_transport.input.return_value = MagicMock()
        fake_transport.output.return_value = MagicMock()
        fake_runner = MagicMock()
        fake_runner.run = AsyncMock(return_value=None)
        fake_context = MagicMock()
        fake_context.get_messages.return_value = []

        runner_args = MagicMock()
        runner_args.pipeline_idle_timeout_secs = 30
        runner_args.handle_sigint = False

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(T, "create_stt_service", MagicMock()), \
             patch.object(T, "create_tts_service", MagicMock()), \
             patch.object(T, "OpenRouterLLMService", return_value=fake_llm), \
             patch.object(T, "RTVIProcessor"), \
             patch.object(T, "SileroVADAnalyzer"), \
             patch.object(T, "LocalSmartTurnAnalyzerV3"), \
             patch.object(T, "AudioBufferProcessor"), \
             patch.object(T, "Pipeline"), \
             patch.object(T, "PipelineTask"), \
             patch.object(T, "PipelineRunner", return_value=fake_runner), \
             patch.object(T, "LLMContext", return_value=fake_context), \
             patch.object(T, "LLMContextAggregatorPair"):
            await T.run_bot(cfg, fake_transport, runner_args, tmp)

        # Exercise registered handlers
        end_call = registered["end_call"]
        params = MagicMock()
        params.result_callback = AsyncMock()
        await end_call(params)

        webhook_fn = registered["wh"]
        params2 = MagicMock()
        params2.function_name = "wh"
        params2.arguments = {"body": {"k": "v"}}
        params2.result_callback = AsyncMock()
        with patch.object(T, "make_webhook_call",
                          AsyncMock(return_value={"status": "success"})):
            await webhook_fn(params2)


if __name__ == "__main__":
    unittest.main()
