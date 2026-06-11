"""
Unit tests for TextAgentConnection — fake HTTP agent server, no external deps.

Run with:
    python -m pytest tests/test_agent_connection.py -v
or:
    python tests/test_agent_connection.py
"""

import asyncio
import json
import unittest
from unittest.mock import patch, AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helpers to mock httpx responses
# ---------------------------------------------------------------------------

def _make_httpx_response(body: dict, status: int = 200):
    """Return a mock that quacks like an httpx.Response."""
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = body
    mock.raise_for_status = MagicMock()
    if status >= 400:
        import httpx
        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status}",
            request=MagicMock(),
            response=mock,
        )
    return mock


def _patch_httpx(response_body: dict, status: int = 200):
    """Context manager that patches httpx.AsyncClient.post."""
    mock_resp = _make_httpx_response(response_body, status)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return patch("httpx.AsyncClient", return_value=mock_client), mock_client


def _patch_httpx_sequence(outcomes):
    """Patch httpx.AsyncClient.post to yield ``outcomes`` in order.

    Each outcome is either an ``(body, status)`` tuple (a fake response) or an
    Exception instance (raised on that attempt). Lets tests exercise the retry
    loop across multiple attempts.
    """
    side_effects = []
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            side_effects.append(outcome)
        else:
            body, status = outcome
            side_effects.append(_make_httpx_response(body, status))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=side_effects)
    return patch("httpx.AsyncClient", return_value=mock_client), mock_client


# ---------------------------------------------------------------------------
# Tests for TextAgentConnection.call()
# ---------------------------------------------------------------------------

class TestCallTextAgent(unittest.IsolatedAsyncioTestCase):

    async def test_returns_response_text(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        fake_body = {"response": "Hello there!", "tool_calls": []}

        ctx, mock_client = _patch_httpx(fake_body)
        with ctx:
            result = await agent.call([{"role": "user", "content": "Hi"}])

        self.assertEqual(result["response"], "Hello there!")
        self.assertEqual(result["tool_calls"], [])

    async def test_returns_tool_calls(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        fake_body = {
            "response": None,
            "tool_calls": [{"tool": "get_weather", "arguments": {"location": "Mumbai"}}],
        }

        ctx, _ = _patch_httpx(fake_body)
        with ctx:
            result = await agent.call([{"role": "user", "content": "Weather in Mumbai?"}])

        self.assertIsNone(result["response"])
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["tool"], "get_weather")

    async def test_preserves_tool_call_output(self):
        """Optional per-tool-call ``output`` rides through ``call`` verbatim."""
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        fake_body = {
            "response": None,
            "tool_calls": [
                {
                    "tool": "get_weather",
                    "arguments": {"location": "Mumbai"},
                    "output": {"temp": 31, "condition": "humid"},
                }
            ],
        }

        ctx, _ = _patch_httpx(fake_body)
        with ctx:
            result = await agent.call([{"role": "user", "content": "Weather?"}])

        self.assertEqual(
            result["tool_calls"][0]["output"], {"temp": 31, "condition": "humid"}
        )

    async def test_sends_auth_header(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(
            url="http://fake-agent/chat",
            headers={"Authorization": "Bearer sk-test"},
        )

        ctx, mock_client = _patch_httpx({"response": "ok"})
        with ctx:
            await agent.call([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_client.post.call_args.kwargs
        self.assertIn("Authorization", call_kwargs.get("headers", {}))
        self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer sk-test")

    async def test_missing_keys_default_to_none_and_empty(self):
        """Agent response with neither key — should not crash."""
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")

        ctx, _ = _patch_httpx({})  # empty body
        with ctx:
            result = await agent.call([{"role": "user", "content": "Hi"}])

        self.assertIsNone(result["response"])
        self.assertEqual(result["tool_calls"], [])


# ---------------------------------------------------------------------------
# Tests for TextAgentConnection.call() — retry on transient failures
# ---------------------------------------------------------------------------

class TestCallRetry(unittest.IsolatedAsyncioTestCase):

    async def test_retries_then_succeeds_on_502(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx_sequence([
            ({}, 502),
            ({}, 503),
            ({"response": "recovered", "tool_calls": []}, 200),
        ])
        with ctx, patch("asyncio.sleep", AsyncMock()):
            result = await agent.call([{"role": "user", "content": "Hi"}])

        self.assertEqual(result["response"], "recovered")
        self.assertEqual(mock_client.post.await_count, 3)

    async def test_retries_then_succeeds_on_connect_error(self):
        import httpx
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx_sequence([
            httpx.ConnectError("boom"),
            ({"response": "ok", "tool_calls": []}, 200),
        ])
        with ctx, patch("asyncio.sleep", AsyncMock()):
            result = await agent.call([{"role": "user", "content": "Hi"}])

        self.assertEqual(result["response"], "ok")
        self.assertEqual(mock_client.post.await_count, 2)

    async def test_gives_up_after_max_attempts(self):
        from calibrate.connections import TextAgentConnection
        from calibrate.connections import _MAX_ATTEMPTS

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx_sequence([({}, 502)] * _MAX_ATTEMPTS)
        with ctx, patch("asyncio.sleep", AsyncMock()):
            with self.assertRaises(RuntimeError) as cm:
                await agent.call([{"role": "user", "content": "Hi"}])

        self.assertEqual(mock_client.post.await_count, _MAX_ATTEMPTS)
        self.assertIn("502", str(cm.exception))
        self.assertIn(str(_MAX_ATTEMPTS), str(cm.exception))

    async def test_no_retry_on_4xx(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx_sequence([
            ({}, 401),
            ({"response": "should not reach", "tool_calls": []}, 200),
        ])
        with ctx, patch("asyncio.sleep", AsyncMock()):
            with self.assertRaises(RuntimeError) as cm:
                await agent.call([{"role": "user", "content": "Hi"}])

        self.assertEqual(mock_client.post.await_count, 1)
        self.assertIn("401", str(cm.exception))


# ---------------------------------------------------------------------------
# Tests for run_test_external — tool_call evaluation
# ---------------------------------------------------------------------------

class TestRunTestExternalToolCall(unittest.IsolatedAsyncioTestCase):

    async def _run(self, agent_tool_calls, expected_tool_calls):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import run_test_external

        agent = TextAgentConnection(url="http://fake-agent/chat")
        fake_body = {"response": None, "tool_calls": agent_tool_calls}

        evaluation = {
            "type": "tool_call",
            "tool_calls": expected_tool_calls,
        }

        ctx, _ = _patch_httpx(fake_body)
        with ctx:
            return await run_test_external(
                chat_history=[{"role": "user", "content": "What's the weather in Mumbai?"}],
                evaluation=evaluation,
                agent=agent,
            )

    async def test_tool_call_pass_exact_match(self):
        result = await self._run(
            agent_tool_calls=[{"tool": "get_weather", "arguments": {"location": "Mumbai"}}],
            expected_tool_calls=[{"tool": "get_weather", "arguments": {"location": "Mumbai"}}],
        )
        self.assertTrue(result["metrics"]["passed"])

    async def test_tool_call_fail_wrong_tool(self):
        result = await self._run(
            agent_tool_calls=[{"tool": "search_web", "arguments": {"query": "Mumbai weather"}}],
            expected_tool_calls=[{"tool": "get_weather", "arguments": {"location": "Mumbai"}}],
        )
        self.assertFalse(result["metrics"]["passed"])
        self.assertIn("mismatch", result["metrics"]["reasoning"].lower())

    async def test_tool_call_fail_wrong_arguments(self):
        result = await self._run(
            agent_tool_calls=[{"tool": "get_weather", "arguments": {"location": "Delhi"}}],
            expected_tool_calls=[{"tool": "get_weather", "arguments": {"location": "Mumbai"}}],
        )
        self.assertFalse(result["metrics"]["passed"])

    async def test_tool_call_fail_no_tool_calls(self):
        result = await self._run(
            agent_tool_calls=[],
            expected_tool_calls=[{"tool": "get_weather", "arguments": {"location": "Mumbai"}}],
        )
        self.assertFalse(result["metrics"]["passed"])

    async def test_tool_call_pass_no_argument_check(self):
        """If expected tool_call has no 'arguments' key, only tool name is checked."""
        result = await self._run(
            agent_tool_calls=[{"tool": "get_weather", "arguments": {"location": "anywhere"}}],
            expected_tool_calls=[{"tool": "get_weather"}],
        )
        self.assertTrue(result["metrics"]["passed"])


# ---------------------------------------------------------------------------
# Tests for run_test_external — response evaluation
# ---------------------------------------------------------------------------

class TestRunTestExternalResponse(unittest.IsolatedAsyncioTestCase):

    @staticmethod
    def _default_evaluator(name: str = "default") -> dict:
        return {
            "name": name,
            "system_prompt": "Evaluate the response.",
            "judge_model": "openai/gpt-4.1",
        }

    async def _run(self, agent_response, judge_result, evaluators=None):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import run_test_external

        agent = TextAgentConnection(url="http://fake-agent/chat")
        fake_body = {"response": agent_response, "tool_calls": []}

        evaluators = evaluators or [self._default_evaluator()]
        evaluation = {
            "type": "response",
            "criteria": [{"name": ev["name"]} for ev in evaluators],
        }

        mock_judge = AsyncMock(return_value=judge_result)

        ctx, _ = _patch_httpx(fake_body)
        with ctx, patch(
            "calibrate.llm.run_tests.test_response_llm_judge", mock_judge
        ):
            return await run_test_external(
                chat_history=[{"role": "user", "content": "Who are you?"}],
                evaluation=evaluation,
                agent=agent,
                evaluators=evaluators,
            )

    async def test_response_pass_when_judge_says_match(self):
        result = await self._run(
            agent_response="I am a helpful assistant.",
            judge_result={
                "default": {
                    "match": True,
                    "reasoning": "Agent introduced itself clearly",
                }
            },
        )
        self.assertTrue(result["metrics"]["passed"])
        self.assertIn("reasoning", result["metrics"])

    async def test_response_fail_when_judge_says_no_match(self):
        result = await self._run(
            agent_response="The capital of France is Paris.",
            judge_result={
                "default": {
                    "match": False,
                    "reasoning": "Agent did not introduce itself",
                }
            },
        )
        self.assertFalse(result["metrics"]["passed"])

    async def test_response_fail_when_no_response_returned(self):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import run_test_external

        agent = TextAgentConnection(url="http://fake-agent/chat")
        fake_body = {"tool_calls": [{"tool": "something", "arguments": {}}]}

        evaluation = {"type": "response", "criteria": "Agent should greet"}

        ctx, _ = _patch_httpx(fake_body)
        with ctx:
            result = await run_test_external(
                chat_history=[{"role": "user", "content": "Hello"}],
                evaluation=evaluation,
                agent=agent,
            )

        self.assertFalse(result["metrics"]["passed"])
        self.assertIn("tool calls", result["metrics"]["reasoning"].lower())


# ---------------------------------------------------------------------------
# Tests for run_test_external — optional agent-reported metrics
# ---------------------------------------------------------------------------

class TestRunTestExternalMetrics(unittest.IsolatedAsyncioTestCase):

    async def _run(self, fake_body):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import run_test_external

        agent = TextAgentConnection(url="http://fake-agent/chat")
        evaluation = {"type": "tool_call", "tool_calls": []}
        ctx, _ = _patch_httpx(fake_body)
        with ctx:
            return await run_test_external(
                chat_history=[{"role": "user", "content": "hi"}],
                evaluation=evaluation,
                agent=agent,
            )

    async def test_metrics_dict_passed_through_to_output(self):
        result = await self._run(
            {"response": "hi", "tool_calls": [],
             "metrics": {"cost": 0.0021, "prompt_tokens": 1200}}
        )
        self.assertEqual(result["output"]["metrics"]["cost"], 0.0021)

    async def test_no_metrics_key_omitted(self):
        result = await self._run({"response": "hi", "tool_calls": []})
        self.assertNotIn("metrics", result["output"])

    async def test_malformed_metrics_ignored(self):
        result = await self._run(
            {"response": "hi", "tool_calls": [], "metrics": "cheap"}
        )
        self.assertNotIn("metrics", result["output"])

    async def test_agent_reported_cost_lifted_to_output(self):
        result = await self._run(
            {"response": "hi", "tool_calls": [], "metrics": {"cost": 0.05}}
        )
        self.assertEqual(result["output"]["cost"], 0.05)

    async def test_no_cost_when_agent_does_not_report(self):
        result = await self._run({"response": "hi", "tool_calls": []})
        self.assertNotIn("cost", result["output"])

    async def test_malformed_cost_ignored(self):
        result = await self._run(
            {"response": "hi", "tool_calls": [], "metrics": {"cost": "free"}}
        )
        self.assertNotIn("cost", result["output"])

    async def test_agent_cost_feeds_aggregate_mean(self):
        from calibrate.llm.run_tests import _aggregate_cost

        result = await self._run(
            {"response": "hi", "tool_calls": [], "metrics": {"cost": 0.05}}
        )
        self.assertAlmostEqual(_aggregate_cost([result])["mean"], 0.05)

    async def test_agent_reported_latency_used(self):
        result = await self._run(
            {"response": "hi", "tool_calls": [], "metrics": {"latency_ms": 850}}
        )
        self.assertEqual(result["latency_ms"], 850)

    async def test_no_latency_when_agent_does_not_report(self):
        result = await self._run({"response": "hi", "tool_calls": []})
        self.assertNotIn("latency_ms", result)

    async def test_malformed_latency_ignored(self):
        result = await self._run(
            {"response": "hi", "tool_calls": [], "metrics": {"latency_ms": "slow"}}
        )
        self.assertNotIn("latency_ms", result)

    async def test_agent_latency_feeds_aggregate(self):
        from calibrate.llm.run_tests import _aggregate_latency

        result = await self._run(
            {"response": "hi", "tool_calls": [], "metrics": {"latency_ms": 200}}
        )
        agg = _aggregate_latency([result])
        self.assertEqual(agg["mean"], 200)


# ---------------------------------------------------------------------------
# Tests for TextAgentConnection.verify()
# ---------------------------------------------------------------------------

class TestTextAgentConnectionVerify(unittest.IsolatedAsyncioTestCase):

    async def test_verify_passes_valid_response(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, _ = _patch_httpx({"response": "Hi! I'm your assistant.", "tool_calls": []})
        with ctx:
            result = await agent.verify()

        self.assertTrue(result["ok"])
        self.assertEqual(result["sample_output"]["response"], "Hi! I'm your assistant.")
        self.assertEqual(result["sample_output"]["tool_calls"], [])

    async def test_verify_passes_tool_calls_only(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, _ = _patch_httpx({"tool_calls": [{"tool": "fn", "arguments": {}}]})
        with ctx:
            result = await agent.verify()

        self.assertTrue(result["ok"])
        self.assertIsNone(result["sample_output"]["response"])
        self.assertEqual(result["sample_output"]["tool_calls"], [{"tool": "fn", "arguments": {}}])

    async def test_verify_preserves_tool_call_output(self):
        """``verify`` accepts and echoes an optional per-tool-call ``output``."""
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        tool_call = {"tool": "fn", "arguments": {}, "output": {"status": "ok"}}
        ctx, _ = _patch_httpx({"tool_calls": [tool_call]})
        with ctx:
            result = await agent.verify()

        self.assertTrue(result["ok"])
        self.assertEqual(result["sample_output"]["tool_calls"], [tool_call])

    async def test_verify_fails_empty_response(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, _ = _patch_httpx({})
        with ctx:
            result = await agent.verify()

        self.assertFalse(result["ok"])

    async def test_verify_fails_http_error(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, _ = _patch_httpx({}, status=500)
        with ctx:
            result = await agent.verify()

        self.assertFalse(result["ok"])

    async def test_verify_retries_then_succeeds_on_503(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx_sequence([
            ({}, 503),
            ({"response": "up now", "tool_calls": []}, 200),
        ])
        with ctx, patch("asyncio.sleep", AsyncMock()):
            result = await agent.verify()

        self.assertTrue(result["ok"])
        self.assertEqual(mock_client.post.await_count, 2)

    async def test_verify_fails_after_retries_exhausted(self):
        from calibrate.connections import TextAgentConnection
        from calibrate.connections import _MAX_ATTEMPTS

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx_sequence([({}, 502)] * _MAX_ATTEMPTS)
        with ctx, patch("asyncio.sleep", AsyncMock()):
            result = await agent.verify()

        self.assertFalse(result["ok"])
        self.assertEqual(mock_client.post.await_count, _MAX_ATTEMPTS)
        self.assertIn("attempts", result["error"])

    async def test_verify_fails_wrong_tool_call_format(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        # tool_calls items missing 'arguments'
        ctx, _ = _patch_httpx({"tool_calls": [{"tool": "fn"}]})
        with ctx:
            result = await agent.verify()

        self.assertFalse(result["ok"])


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
