import asyncio
import json
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from calibrate_agent import chat as C
from calibrate_agent.chat import agent_reply

WEBHOOK_TOOL = {
    "name": "lookup",
    "description": "Look up an order",
    "type": "webhook",
    "webhook": {"url": "http://x", "method": "GET", "headers": []},
}
STRUCTURED_TOOL = {"name": "plan_next", "description": "Plan", "parameters": []}


def call(name, args="{}", call_id=None):
    return SimpleNamespace(id=call_id or f"id_{name}", function=SimpleNamespace(name=name, arguments=args))


def response(content=None, tool_calls=None, usage=None):
    usage = usage or SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def fake_client(*responses):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=list(responses))
    client.close = AsyncMock()
    return client


async def run(client, tools=(), messages=None, **kwargs):
    kwargs.setdefault("provider", "openai")
    return await agent_reply(
        messages or [{"role": "user", "content": "hi"}],
        "You are a bot",
        list(tools),
        "gpt-x",
        api_key="k",
        client=client,
        **kwargs,
    )


class TestAgentReply(unittest.IsolatedAsyncioTestCase):
    async def test_plain_reply(self):
        client = fake_client(response("Hello"))
        out = await run(client)
        self.assertEqual(out["reply"], "Hello")
        self.assertEqual(out["stopped"], "reply")
        self.assertFalse(out["ended"])
        self.assertEqual(out["tool_calls"], [])
        self.assertEqual(out["new_messages"], [{"role": "assistant", "content": "Hello"}])
        sent = client.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "You are a bot"})
        self.assertEqual(sent["tools"][0]["function"]["name"], "end_call")
        self.assertEqual(out["total_tokens"], 12)

    async def test_structured_tool_then_text(self):
        client = fake_client(response(tool_calls=[call("plan_next", '{"q": 1}')]), response("Done"))
        out = await run(client, [STRUCTURED_TOOL])
        self.assertEqual(out["reply"], "Done")
        self.assertEqual(
            out["tool_calls"],
            [{"id": "id_plan_next", "tool": "plan_next", "arguments": {"q": 1}, "output": {"status": "received"}}],
        )
        self.assertEqual([m["role"] for m in out["new_messages"]], ["assistant", "tool", "assistant"])
        self.assertEqual(out["prompt_tokens"], 20)
        second = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertEqual(second[-1], {"role": "tool", "tool_call_id": "id_plan_next", "content": '{"status": "received"}'})

    async def test_webhook_result_fed_back(self):
        client = fake_client(response(tool_calls=[call("lookup", '{"query": {"a": 1}}')]), response("Found"))
        hook = AsyncMock(return_value={"status": "success", "response": {"order": 7}})
        with patch.object(C, "make_webhook_call", hook):
            out = await run(client, [WEBHOOK_TOOL])
        hook.assert_awaited_once()
        self.assertEqual(hook.call_args.args[0]["url"], "http://x")
        self.assertEqual(hook.call_args.args[1], {"query": {"a": 1}})
        second = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertEqual(json.loads(second[-1]["content"]), {"status": "success", "response": {"order": 7}})
        self.assertEqual(out["reply"], "Found")

    async def test_end_call_stops_after_one_call(self):
        client = fake_client(response("Bye", [call("end_call", '{"reason": "done"}')]), response("never"))
        out = await run(client)
        self.assertEqual(client.chat.completions.create.await_count, 1)
        self.assertTrue(out["ended"])
        self.assertEqual(out["stopped"], "end_call")
        self.assertEqual(out["reply"], "Bye")
        self.assertEqual(out["tool_calls"][0]["arguments"], {"reason": "done"})

    async def test_bad_json_arguments_parse_to_empty(self):
        client = fake_client(response(tool_calls=[call("plan_next", "{not json")]), response("ok"))
        out = await run(client, [STRUCTURED_TOOL])
        self.assertEqual(out["tool_calls"][0]["arguments"], {})

    async def test_cap_reached(self):
        client = fake_client(*[response("thinking", [call("plan_next")]) for _ in range(3)])
        out = await run(client, [STRUCTURED_TOOL], max_tool_rounds=2)
        self.assertEqual(client.chat.completions.create.await_count, 2)
        self.assertEqual(out["stopped"], "max_tool_rounds")
        self.assertEqual(out["reply"], "thinking")
        self.assertFalse(out["ended"])

    async def test_fills_missing_tool_results_in_input(self):
        client = fake_client(response("ok"))
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "plan_next", "arguments": "{}"}}]},
        ]
        await run(client, [STRUCTURED_TOOL], messages=history)
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(sent[-1]["tool_call_id"], "c1")

    async def test_openrouter_cost_from_attribute_and_model_extra(self):
        a = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost=0.5)
        b = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2, model_extra={"cost": 0.25})
        client = fake_client(response(tool_calls=[call("plan_next")], usage=a), response("ok", usage=b))
        out = await run(client, [STRUCTURED_TOOL], provider="openrouter")
        self.assertAlmostEqual(out["cost"], 0.75)
        self.assertEqual(out["total_tokens"], 4)

    async def test_openai_cost_is_none(self):
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost=0.5)
        out = await run(fake_client(response("ok", usage=usage)))
        self.assertIsNone(out["cost"])

    async def test_provider_error_raises(self):
        client = fake_client(RuntimeError("rate limited"))
        with self.assertRaisesRegex(RuntimeError, "rate limited"):
            await run(client)


class _NoEnv(dict):
    def __getitem__(self, key):
        raise AssertionError(f"read os.environ[{key!r}]")

    def get(self, key, default=None):
        raise AssertionError(f"read os.environ.get({key!r})")


class TestClientAndKeys(unittest.IsolatedAsyncioTestCase):
    async def test_environment_is_not_read(self):
        with patch.object(C, "AsyncOpenAI", return_value=fake_client(response("ok"))), patch.object(
            os, "environ", _NoEnv()
        ), patch.object(os, "getenv", side_effect=AssertionError("read os.getenv")):
            out = await agent_reply([], "s", [], "m", "openai", "k")
        self.assertEqual(out["reply"], "ok")

    async def test_missing_key_is_refused_rather_than_read_from_the_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "server-key"}):
            for key in (None, ""):
                with self.assertRaises(ValueError):
                    await agent_reply([], "s", [], "m", "openrouter", key)

    async def test_api_key_and_base_url_reach_client(self):
        made = []

        def factory(**kwargs):
            made.append(kwargs)
            return fake_client(response("ok"))

        with patch.object(C, "AsyncOpenAI", side_effect=factory), patch.dict(
            os.environ, {"OPENAI_API_KEY": "env", "OPENROUTER_API_KEY": "env"}
        ):
            await agent_reply([], "s", [], "m", "openrouter", "given-key")
            await agent_reply([], "s", [], "m", "openai", "other-key")
        self.assertEqual(made[0], {"api_key": "given-key", "base_url": "https://openrouter.ai/api/v1"})
        self.assertEqual(made[1], {"api_key": "other-key", "base_url": None})

    async def test_concurrent_calls_do_not_cross_keys(self):
        seen = {}

        def factory(api_key, base_url):
            client = MagicMock()
            client.close = AsyncMock()

            async def create(**kwargs):
                await asyncio.sleep(0.01)
                return response(f"reply for {api_key}")

            client.chat.completions.create = create
            return client

        with patch.object(C, "AsyncOpenAI", side_effect=factory):
            a, b = await asyncio.gather(
                agent_reply([], "s", [], "m", "openai", "key-a"),
                agent_reply([], "s", [], "m", "openai", "key-b"),
            )
        self.assertEqual(a["reply"], "reply for key-a")
        self.assertEqual(b["reply"], "reply for key-b")


class TestImportIsLight(unittest.TestCase):
    def test_no_pipecat_or_numpy(self):
        code = "import sys, calibrate_agent.chat; print(sorted(m for m in ('pipecat', 'numpy') if m in sys.modules))"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
