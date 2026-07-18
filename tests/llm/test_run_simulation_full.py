"""Test the run_simulation full function flow with heavy mocking."""

import tempfile
import unittest
from unittest.mock import patch, AsyncMock, MagicMock


class TestRunSimulationFullFlow(unittest.IsolatedAsyncioTestCase):
    async def test_basic_flow_openai(self):
        from calibrate_agent.llm import run_simulation as RS

        # Mock pipecat services and runners
        fake_bot_llm = MagicMock()
        fake_bot_llm.register_function = MagicMock()
        fake_user_llm = MagicMock()
        fake_user_llm.register_function = MagicMock()

        # Mock runner so it just completes
        fake_runner = MagicMock()
        fake_runner.run = AsyncMock(return_value=None)

        # Mock the bot context get_messages and access
        fake_bot_context = MagicMock()
        fake_bot_context._messages = [
            {"role": "system", "content": "sp"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Hello"},
        ]
        fake_bot_context.get_messages.return_value = fake_bot_context._messages

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(RS, "OpenAILLMService", side_effect=[fake_bot_llm, fake_user_llm]), \
             patch.object(RS, "OpenRouterLLMService"), \
             patch.object(RS, "PipelineTask"), \
             patch.object(RS, "PipelineRunner", return_value=fake_runner), \
             patch.object(RS, "Pipeline"), \
             patch.object(RS, "LLMContext", return_value=fake_bot_context), \
             patch.object(RS, "LLMContextAggregatorPair"), \
             patch.object(RS, "evaluate_simuation",
                          AsyncMock(return_value={"x": {"reasoning": "ok", "match": True}})):
            result = await RS.run_simulation(
                bot_system_prompt="bp",
                tools=[],
                user_system_prompt="up",
                evaluators=[{"name": "x", "system_prompt": "x", "judge_model": "m"}],
                bot_provider="openai",
                user_provider="openai",
                output_dir=tmp,
                max_turns=2,
            )
        self.assertIn("transcript", result)
        self.assertIn("evaluation_results", result)

    async def test_with_openrouter_provider(self):
        from calibrate_agent.llm import run_simulation as RS

        fake_llm = MagicMock()
        fake_llm.register_function = MagicMock()
        fake_runner = MagicMock()
        fake_runner.run = AsyncMock(return_value=None)
        fake_context = MagicMock()
        fake_context._messages = [{"role": "system", "content": "sp"}]
        fake_context.get_messages.return_value = []

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(RS, "OpenAILLMService"), \
             patch.object(RS, "OpenRouterLLMService", return_value=fake_llm), \
             patch.object(RS, "PipelineTask"), \
             patch.object(RS, "PipelineRunner", return_value=fake_runner), \
             patch.object(RS, "Pipeline"), \
             patch.object(RS, "LLMContext", return_value=fake_context), \
             patch.object(RS, "LLMContextAggregatorPair"), \
             patch.object(RS, "evaluate_simuation",
                          AsyncMock(return_value={"x": {"reasoning": "ok", "match": True}})):
            await RS.run_simulation(
                bot_system_prompt="bp",
                tools=[],
                user_system_prompt="up",
                evaluators=[{"name": "x", "system_prompt": "x", "judge_model": "m"}],
                bot_provider="openrouter",
                user_provider="openrouter",
                output_dir=tmp,
                max_turns=1,
            )

    async def test_pipeline_error_re_raised(self):
        from calibrate_agent.llm import run_simulation as RS

        fake_llm = MagicMock()
        fake_llm.register_function = MagicMock()
        fake_runner = MagicMock()
        fake_runner.run = AsyncMock(side_effect=RuntimeError("pipeline fail"))
        fake_context = MagicMock()
        fake_context._messages = []
        fake_context.get_messages.return_value = []

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(RS, "OpenAILLMService", return_value=fake_llm), \
             patch.object(RS, "PipelineTask"), \
             patch.object(RS, "PipelineRunner", return_value=fake_runner), \
             patch.object(RS, "Pipeline"), \
             patch.object(RS, "LLMContext", return_value=fake_context), \
             patch.object(RS, "LLMContextAggregatorPair"):
            with self.assertRaises(RuntimeError):
                await RS.run_simulation(
                    bot_system_prompt="bp",
                    tools=[],
                    user_system_prompt="up",
                    evaluators=[{"name": "x", "system_prompt": "x", "judge_model": "m"}],
                    output_dir=tmp,
                    max_turns=1,
                )


if __name__ == "__main__":
    unittest.main()
