"""Additional tests covering error branches and edge cases in connections.py."""

import asyncio
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


def _mk_ws_cm(enter_side_effect=None):
    """Return a mock that quacks like the async context manager from
    ``websockets.connect(...)``."""
    cm = MagicMock()
    if enter_side_effect is not None:
        cm.__aenter__ = AsyncMock(side_effect=enter_side_effect)
    else:
        cm.__aenter__ = AsyncMock(return_value=MagicMock())
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class TestWebSocketAgentConnectionVerify(unittest.IsolatedAsyncioTestCase):
    async def test_verify_happy_path(self):
        from calibrate.connections import WebSocketAgentConnection

        agent = WebSocketAgentConnection(url="ws://fake-agent/ws")
        connect = MagicMock(return_value=_mk_ws_cm())
        with patch("websockets.connect", connect):
            result = await agent.verify()

        self.assertTrue(result["ok"])
        self.assertIsNone(result["error"])
        connect.assert_called_once()
        self.assertEqual(connect.call_args.kwargs["additional_headers"], None)

    async def test_verify_passes_headers(self):
        from calibrate.connections import WebSocketAgentConnection

        agent = WebSocketAgentConnection(
            url="wss://fake-agent/ws", headers={"Authorization": "Bearer sk-x"}
        )
        connect = MagicMock(return_value=_mk_ws_cm())
        with patch("websockets.connect", connect):
            result = await agent.verify()

        self.assertTrue(result["ok"])
        self.assertEqual(
            connect.call_args.kwargs["additional_headers"],
            {"Authorization": "Bearer sk-x"},
        )

    async def test_verify_connection_error(self):
        from calibrate.connections import WebSocketAgentConnection

        agent = WebSocketAgentConnection(url="ws://fake-agent/ws")
        connect = MagicMock(
            return_value=_mk_ws_cm(
                enter_side_effect=ConnectionRefusedError("refused")
            )
        )
        with patch("websockets.connect", connect):
            result = await agent.verify()

        self.assertFalse(result["ok"])
        self.assertIn("Could not connect", result["error"])
        self.assertIn("refused", result["error"])

    async def test_verify_non_ws_scheme(self):
        from calibrate.connections import WebSocketAgentConnection

        agent = WebSocketAgentConnection(url="http://fake-agent/ws")
        connect = MagicMock(return_value=_mk_ws_cm())
        with patch("websockets.connect", connect):
            result = await agent.verify()

        self.assertFalse(result["ok"])
        self.assertIn("scheme must be", result["error"])
        connect.assert_not_called()

    async def test_verify_timeout(self):
        from calibrate.connections import WebSocketAgentConnection

        agent = WebSocketAgentConnection(url="ws://fake-agent/ws")
        connect = MagicMock(
            return_value=_mk_ws_cm(enter_side_effect=asyncio.TimeoutError())
        )
        with patch("websockets.connect", connect):
            result = await agent.verify()

        self.assertFalse(result["ok"])
        self.assertIn("timed out", result["error"])

    async def test_verify_invalid_uri(self):
        from calibrate.connections import WebSocketAgentConnection

        agent = WebSocketAgentConnection(url="ws://fake-agent/ws")
        connect = MagicMock(side_effect=ValueError("invalid uri"))
        with patch("websockets.connect", connect):
            result = await agent.verify()

        self.assertFalse(result["ok"])
        self.assertIn("Could not connect", result["error"])

    def test_serializer_stored(self):
        from calibrate.connections import WebSocketAgentConnection

        default = WebSocketAgentConnection(url="ws://x")
        self.assertEqual(default.serializer, "protobuf")
        custom = WebSocketAgentConnection(url="ws://x", serializer="raw")
        self.assertEqual(custom.serializer, "raw")


if __name__ == "__main__":
    unittest.main()
