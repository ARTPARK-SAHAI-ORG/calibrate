"""Additional tests covering error branches and edge cases in connections.py."""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx


def _mk_mock_client(post_side_effect=None, response=None):
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    if post_side_effect is not None:
        mock_client.post = AsyncMock(side_effect=post_side_effect)
    else:
        mock_client.post = AsyncMock(return_value=response)
    return mock_client


def _mk_resp(body, status=200, text="err body", raw_text=None, json_raises=False):
    m = MagicMock()
    m.status_code = status
    m.text = text
    if json_raises:
        m.json.side_effect = ValueError("not json")
    else:
        m.json.return_value = body
    return m


class TestVerifyErrorPaths(unittest.IsolatedAsyncioTestCase):
    async def test_verify_connect_error(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        mock_client = _mk_mock_client(post_side_effect=httpx.ConnectError("boom"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify()
        self.assertFalse(result["ok"])
        self.assertIn("Could not connect", result["error"])

    async def test_verify_timeout(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        mock_client = _mk_mock_client(post_side_effect=httpx.TimeoutException("t"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify()
        self.assertFalse(result["ok"])
        self.assertIn("timed out", result["error"])

    async def test_verify_generic_exception(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        mock_client = _mk_mock_client(post_side_effect=RuntimeError("nope"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify()
        self.assertFalse(result["ok"])
        self.assertIn("Unexpected error", result["error"])

    async def test_verify_invalid_json(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        resp = _mk_resp({}, json_raises=True)
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify()
        self.assertFalse(result["ok"])
        self.assertIn("not valid JSON", result["error"])

    async def test_verify_non_dict_json(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        resp = _mk_resp(["a", "b"])
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify()
        self.assertFalse(result["ok"])
        self.assertIn("must be a JSON object", result["error"])

    async def test_verify_response_wrong_type(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        resp = _mk_resp({"response": 123})
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify()
        self.assertFalse(result["ok"])
        self.assertIn("must be a string", result["error"])

    async def test_verify_tool_calls_not_list(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        resp = _mk_resp({"tool_calls": {"not": "list"}})
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify()
        self.assertFalse(result["ok"])
        self.assertIn("must be a list", result["error"])

    async def test_verify_tool_call_not_dict(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        resp = _mk_resp({"tool_calls": ["not a dict"]})
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify()
        self.assertFalse(result["ok"])
        self.assertIn("must be an object", result["error"])

    async def test_verify_tool_call_missing_tool(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        resp = _mk_resp({"tool_calls": [{"arguments": {}}]})
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify()
        self.assertFalse(result["ok"])
        self.assertIn('missing required key "tool"', result["error"])

    async def test_verify_tool_call_arguments_not_dict(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        resp = _mk_resp({"tool_calls": [{"tool": "x", "arguments": "no"}]})
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify()
        self.assertFalse(result["ok"])
        self.assertIn("must be an object", result["error"])

    async def test_verify_with_messages_and_model(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x", headers={"X": "Y"})
        resp = _mk_resp({"response": "hi"})
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify(
                messages=[{"role": "user", "content": "go"}],
                model="some-model",
            )
        self.assertTrue(result["ok"])
        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "some-model")
        self.assertEqual(body["messages"][0]["content"], "go")


class TestCallErrorPaths(unittest.IsolatedAsyncioTestCase):
    async def test_call_connect_error(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        mock_client = _mk_mock_client(post_side_effect=httpx.ConnectError("boom"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                await agent.call([{"role": "user", "content": "Hi"}])
        self.assertIn("Could not connect", str(ctx.exception))

    async def test_call_timeout(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        mock_client = _mk_mock_client(post_side_effect=httpx.TimeoutException("t"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                await agent.call([{"role": "user", "content": "Hi"}])
        self.assertIn("timed out", str(ctx.exception))

    async def test_call_generic_exception(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        mock_client = _mk_mock_client(post_side_effect=RuntimeError("nope"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                await agent.call([{"role": "user", "content": "Hi"}])
        self.assertIn("Unexpected error", str(ctx.exception))

    async def test_call_non_200_status(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        resp = _mk_resp({}, status=500, text="server-down")
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                await agent.call([{"role": "user", "content": "Hi"}])
        self.assertIn("HTTP 500", str(ctx.exception))

    async def test_call_retries_http_500_twice_then_succeeds(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        responses = [
            _mk_resp({}, status=500, text="server-down-1"),
            _mk_resp({}, status=500, text="server-down-2"),
            _mk_resp({"response": "ok"}, status=200),
        ]
        mock_client = _mk_mock_client(post_side_effect=responses)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.call([{"role": "user", "content": "Hi"}])
        self.assertEqual(result["response"], "ok")
        self.assertEqual(mock_client.post.await_count, 3)

    async def test_call_stops_after_three_http_500_responses(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        responses = [
            _mk_resp({}, status=500, text="server-down-1"),
            _mk_resp({}, status=500, text="server-down-2"),
            _mk_resp({}, status=500, text="final-server-down"),
        ]
        mock_client = _mk_mock_client(post_side_effect=responses)
        with patch("httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                await agent.call([{"role": "user", "content": "Hi"}])
        self.assertIn("Agent returned HTTP 500: final-server-down", str(ctx.exception))
        self.assertEqual(mock_client.post.await_count, 3)

    async def test_call_does_not_retry_non_500_http_errors(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        resp = _mk_resp({}, status=404, text="missing")
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                await agent.call([{"role": "user", "content": "Hi"}])
        self.assertIn("Agent returned HTTP 404: missing", str(ctx.exception))
        self.assertEqual(mock_client.post.await_count, 1)

    async def test_call_invalid_json(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        resp = _mk_resp({}, json_raises=True, text="not json")
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                await agent.call([{"role": "user", "content": "Hi"}])
        self.assertIn("not valid JSON", str(ctx.exception))

    async def test_call_with_model(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x", headers={"K": "V"})
        resp = _mk_resp({"response": "ok"})
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            await agent.call([{"role": "user", "content": "Hi"}], model="m1")
        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "m1")

    async def test_verify_non_200(self):
        from calibrate.connections import TextAgentConnection
        agent = TextAgentConnection(url="http://x")
        resp = _mk_resp({}, status=404, text="missing")
        mock_client = _mk_mock_client(response=resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify()
        self.assertFalse(result["ok"])
        self.assertIn("HTTP 404", result["error"])


if __name__ == "__main__":
    unittest.main()
