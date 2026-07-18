"""Additional coverage tests for stt/tts/llm metrics modules."""

import unittest
from unittest.mock import patch, AsyncMock


class TestSTTLlmJudge(unittest.IsolatedAsyncioTestCase):
    async def test_stt_llm_judge_default_evaluator_no_langfuse(self):
        from calibrate_agent.stt import metrics as M

        with patch.object(M, "text_judge", AsyncMock(return_value={"semantic_match": {"reasoning": "r", "match": True}})), \
             patch.object(M, "langfuse_enabled", False):
            # Skip backoff retry by calling __wrapped__
            inner = M.stt_llm_judge.__wrapped__ if hasattr(M.stt_llm_judge, "__wrapped__") else M.stt_llm_judge
            result = await inner("ref", "pred")
        self.assertEqual(result["semantic_match"]["match"], True)

    async def test_stt_llm_judge_with_langfuse(self):
        from calibrate_agent.stt import metrics as M
        fake_lf = unittest.mock.MagicMock()

        with patch.object(M, "text_judge", AsyncMock(return_value={"semantic_match": {"reasoning": "r", "match": True}})), \
             patch.object(M, "langfuse_enabled", True), \
             patch.object(M, "langfuse", fake_lf):
            inner = M.stt_llm_judge.__wrapped__ if hasattr(M.stt_llm_judge, "__wrapped__") else M.stt_llm_judge
            await inner("ref", "pred")
        fake_lf.update_current_trace.assert_called_once()


class TestTTSLlmJudge(unittest.IsolatedAsyncioTestCase):
    async def test_tts_llm_judge_default_evaluator(self):
        from calibrate_agent.tts import metrics as M

        with patch.object(M, "audio_judge", AsyncMock(return_value={"pronunciation": {"reasoning": "r", "match": True}})):
            inner = M.tts_llm_judge.__wrapped__ if hasattr(M.tts_llm_judge, "__wrapped__") else M.tts_llm_judge
            result = await inner("/tmp/a.wav", "text")
        self.assertEqual(result["pronunciation"]["match"], True)


class TestLlmMetrics(unittest.IsolatedAsyncioTestCase):
    async def test_test_response_llm_judge(self):
        from calibrate_agent.llm import metrics as M

        with patch.object(M, "text_judge", AsyncMock(return_value={"correctness": {"reasoning": "r", "match": True}})) as mock_tj:
            result = await M.test_response_llm_judge(
                conversation=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}, {"no_content": True}],
                response="Hi there",
                evaluators=[{"name": "correctness", "system_prompt": "...", "judge_model": "x"}],
            )
        self.assertEqual(result["correctness"]["match"], True)
        # Ensure conversation entries with content are included
        user_prompt = mock_tj.call_args.kwargs["user_prompt"]
        self.assertIn("user: Hi", user_prompt)
        self.assertIn("assistant: Hello", user_prompt)
        self.assertIn("Hi there", user_prompt)

    async def test_evaluate_simulation_delegates(self):
        from calibrate_agent.llm import metrics as M

        with patch.object(M, "simulation_judge", AsyncMock(return_value={"x": {"match": True}})) as mock_sj:
            result = await M.evaluate_simuation(
                conversation=[{"role": "user", "content": "Hi"}],
                evaluators=[{"name": "x", "system_prompt": "...", "judge_model": "y"}],
            )
        self.assertEqual(result["x"]["match"], True)
        mock_sj.assert_called_once()


if __name__ == "__main__":
    unittest.main()
