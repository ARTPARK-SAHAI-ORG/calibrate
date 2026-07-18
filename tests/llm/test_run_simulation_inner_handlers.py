"""Test the inner tool handlers in run_simulation by triggering tool registration paths."""

import tempfile
import unittest
from unittest.mock import patch, AsyncMock, MagicMock


class TestRunSimulationWithTools(unittest.IsolatedAsyncioTestCase):
    async def test_simulation_with_tools_and_webhook(self):
        """run_simulation registers end_call, generic tool, and webhook handlers.

        Calling the registered functions exercises lines 369-437 in the source.
        """
        from calibrate_agent.llm import run_simulation as RS

        bot_llm_registrations = {}

        fake_bot_llm = MagicMock()

        def register(name, fn):
            bot_llm_registrations[name] = fn

        fake_bot_llm.register_function = register
        fake_user_llm = MagicMock()
        fake_user_llm.register_function = MagicMock()

        fake_runner = MagicMock()
        fake_runner.run = AsyncMock(return_value=None)
        fake_context = MagicMock()
        fake_context._messages = []
        fake_context.get_messages.return_value = []

        tools = [
            {"name": "do_stuff", "description": "d", "parameters": []},
            {"name": "wh", "description": "wh", "type": "webhook", "webhook": {
                "url": "http://x", "method": "POST", "headers": [],
            }},
        ]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(RS, "OpenAILLMService",
                          side_effect=[fake_bot_llm, fake_user_llm]), \
             patch.object(RS, "PipelineTask"), \
             patch.object(RS, "PipelineRunner", return_value=fake_runner), \
             patch.object(RS, "Pipeline"), \
             patch.object(RS, "LLMContext", return_value=fake_context), \
             patch.object(RS, "LLMContextAggregatorPair"), \
             patch.object(RS, "evaluate_simuation",
                          AsyncMock(return_value={"x": {"reasoning": "ok", "match": True}})):
            await RS.run_simulation(
                bot_system_prompt="bp",
                tools=tools,
                user_system_prompt="up",
                evaluators=[{"name": "x", "system_prompt": "x", "judge_model": "m"}],
                output_dir=tmp,
                max_turns=1,
            )

        # Verify handlers were registered
        self.assertIn("end_call", bot_llm_registrations)
        self.assertIn("do_stuff", bot_llm_registrations)
        self.assertIn("wh", bot_llm_registrations)

        # Now exercise each registered handler
        end_call_fn = bot_llm_registrations["end_call"]
        generic_fn = bot_llm_registrations["do_stuff"]
        webhook_fn = bot_llm_registrations["wh"]

        params_end = MagicMock()
        params_end.arguments = {"reason": "done"}
        params_end.result_callback = AsyncMock()
        await end_call_fn(params_end)
        params_end.result_callback.assert_called_once()

        params_end_no_reason = MagicMock()
        params_end_no_reason.arguments = None
        params_end_no_reason.result_callback = AsyncMock()
        await end_call_fn(params_end_no_reason)

        params_generic = MagicMock()
        params_generic.function_name = "do_stuff"
        params_generic.arguments = {"x": 1}
        params_generic.result_callback = AsyncMock()
        await generic_fn(params_generic)
        params_generic.result_callback.assert_called_once_with({"status": "received"})

        # Webhook handler
        params_wh = MagicMock()
        params_wh.function_name = "wh"
        params_wh.arguments = {"body": {"k": "v"}}
        params_wh.result_callback = AsyncMock()
        with patch.object(RS, "make_webhook_call",
                          AsyncMock(return_value={"status": "success"})):
            await webhook_fn(params_wh)
        params_wh.result_callback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
