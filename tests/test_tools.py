import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from calibrate_agent.tools import build_tool_specs, fill_missing_tool_results, make_webhook_call
from calibrate_agent.utils import build_tools_schema

WEBHOOK_TOOL = {
    "name": "lookup",
    "description": "Look up an order",
    "type": "webhook",
    "webhook": {
        "url": "http://x/orders",
        "method": "POST",
        "headers": [{"name": "Authorization", "value": "Bearer secret-key-123"}],
        "queryParameters": [
            {"id": "q", "type": "string", "description": "query", "required": True}
        ],
        "body": {
            "description": "Order body",
            "parameters": [
                {"id": "order_id", "type": "string", "description": "id", "required": True},
                {"id": "tags", "type": "array", "description": "tags", "items": {"type": "string"}},
            ],
        },
    },
}
STRUCTURED_TOOL = {
    "name": "plan_next",
    "description": "Plan the next question",
    "parameters": [
        {"id": "question", "type": "string", "description": "q", "required": True},
        {"id": "mood", "type": "string", "description": "m", "enum": ["a", "b"]},
    ],
}


class TestBuildToolSpecs(unittest.TestCase):
    def test_matches_pipecat_schema(self):
        tools = [WEBHOOK_TOOL, STRUCTURED_TOOL]
        specs, webhooks = build_tool_specs(tools)
        schemas, old_webhooks = build_tools_schema(tools)
        self.assertEqual(specs, [s.to_default_dict() for s in schemas])
        self.assertEqual(webhooks, old_webhooks)
        self.assertEqual(
            webhooks["lookup"],
            {
                "url": "http://x/orders",
                "method": "POST",
                "headers": WEBHOOK_TOOL["webhook"]["headers"],
                "timeout": 20,
            },
        )
        self.assertEqual(specs[0]["parameters"]["required"], ["query", "body"])
        self.assertEqual(specs[1]["parameters"]["required"], ["question"])
        self.assertNotIn("plan_next", webhooks)

    def test_missing_url_or_method_raises(self):
        for field in ("url", "method"):
            webhook = {"url": "http://x", "method": "GET"}
            del webhook[field]
            with self.assertRaisesRegex(ValueError, field):
                build_tool_specs([{"name": "t", "description": "d", "type": "webhook", "webhook": webhook}])


class TestFillMissingToolResults(unittest.TestCase):
    def _call(self, name, call_id="c1"):
        return {"role": "assistant", "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": "{}"}}]}

    def test_fills_structured(self):
        out = fill_missing_tool_results([self._call("plan_next")], [STRUCTURED_TOOL])
        self.assertEqual(out[1], {"role": "tool", "content": '{"status": "received"}', "tool_call_id": "c1"})

    def test_keeps_existing(self):
        history = [self._call("plan_next"), {"role": "tool", "tool_call_id": "c1", "content": "real"}]
        self.assertEqual(fill_missing_tool_results(history, [STRUCTURED_TOOL]), history)

    def test_skips_webhook(self):
        history = [self._call("lookup")]
        self.assertEqual(fill_missing_tool_results(history, [WEBHOOK_TOOL]), history)

    def test_old_name_still_works(self):
        from calibrate_agent.llm.run_tests import preprocess_conversation_history

        self.assertIs(preprocess_conversation_history, fill_missing_tool_results)


class TestWebhookLogging(unittest.IsolatedAsyncioTestCase):
    async def test_header_values_not_logged(self):
        response = MagicMock(status=200)
        response.json = AsyncMock(return_value={"ok": True})
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.request = MagicMock(return_value=response)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        from calibrate_agent import tools as T

        with patch("aiohttp.ClientSession", return_value=session), patch.object(T, "logger") as log:
            result = await make_webhook_call(
                {"url": "http://x", "method": "GET", "headers": WEBHOOK_TOOL["webhook"]["headers"]},
                {"query": {"q": "1"}},
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(session.request.call_args.kwargs["headers"], {"Authorization": "Bearer secret-key-123"})
        logged = " ".join(str(c) for c in log.method_calls)
        self.assertIn("Authorization", logged)
        self.assertNotIn("secret-key-123", logged)


if __name__ == "__main__":
    unittest.main()
