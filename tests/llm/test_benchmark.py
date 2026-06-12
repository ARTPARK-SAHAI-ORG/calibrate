"""
Unit tests for agent connection benchmarking.

Covers:
- model param included in request body (no provider)
- model threaded through run_test_external
- verify() sends model for benchmark verify
- output folder naming per model

Run with:
    python -m pytest tests/test_agent_benchmarking.py -v
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Reuse the mock helper from test_agent_connection
# ---------------------------------------------------------------------------

def _make_httpx_response(body: dict, status: int = 200):
    import httpx
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = body
    mock.raise_for_status = MagicMock()
    if status >= 400:
        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status}", request=MagicMock(), response=mock
        )
    return mock


def _patch_httpx(response_body: dict, status: int = 200):
    mock_resp = _make_httpx_response(response_body, status)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return patch("httpx.AsyncClient", return_value=mock_client), mock_client


# ---------------------------------------------------------------------------
# Tests for TextAgentConnection.call() — model in request body (no provider)
# ---------------------------------------------------------------------------

class TestCallTextAgentModelParams(unittest.IsolatedAsyncioTestCase):

    async def test_model_included_in_body(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "ok"})
        with ctx:
            await agent.call(
                [{"role": "user", "content": "Hi"}],
                model="gemma-4-26b-a4b-it",
            )

        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "gemma-4-26b-a4b-it")
        self.assertNotIn("provider", body)
        self.assertIn("messages", body)

    async def test_model_absent_when_not_passed(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "ok"})
        with ctx:
            await agent.call([{"role": "user", "content": "Hi"}])

        body = mock_client.post.call_args.kwargs["json"]
        self.assertNotIn("model", body)
        self.assertNotIn("provider", body)
        self.assertIn("messages", body)

    async def test_model_included_openrouter_format(self):
        """OpenRouter format model string passed as-is (no splitting)."""
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "ok"})
        with ctx:
            await agent.call(
                [{"role": "user", "content": "Hi"}],
                model="google/gemma-4-26b-a4b-it",
            )

        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "google/gemma-4-26b-a4b-it")
        self.assertNotIn("provider", body)


# ---------------------------------------------------------------------------
# Tests for run_test_external — model threaded through
# ---------------------------------------------------------------------------

class TestRunTestExternalModelParams(unittest.IsolatedAsyncioTestCase):

    async def test_model_passed_to_agent_call(self):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import run_test_external

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "Sure, the weather is sunny."})

        evaluation = {"type": "response", "criteria": "Agent answers the question"}
        mock_judge = AsyncMock(
            return_value={"correctness": {"match": True, "reasoning": "ok"}}
        )

        with ctx, patch("calibrate.llm.run_tests.test_response_llm_judge", mock_judge):
            await run_test_external(
                chat_history=[{"role": "user", "content": "What's the weather?"}],
                evaluation=evaluation,
                agent=agent,
                model="gemma-4-26b-a4b-it",
            )

        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "gemma-4-26b-a4b-it")
        self.assertNotIn("provider", body)

    async def test_no_model_param_when_not_passed(self):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import run_test_external

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "hello"})
        mock_judge = AsyncMock(
            return_value={"correctness": {"match": True, "reasoning": "ok"}}
        )

        with ctx, patch("calibrate.llm.run_tests.test_response_llm_judge", mock_judge):
            await run_test_external(
                chat_history=[{"role": "user", "content": "Hi"}],
                evaluation={"type": "response", "criteria": "greet"},
                agent=agent,
            )

        body = mock_client.post.call_args.kwargs["json"]
        self.assertNotIn("model", body)
        self.assertNotIn("provider", body)


# ---------------------------------------------------------------------------
# Tests for TextAgentConnection.verify() — benchmark verify
# ---------------------------------------------------------------------------

class TestVerifyWithModelParams(unittest.IsolatedAsyncioTestCase):

    async def test_verify_includes_model(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "hi"})
        with ctx:
            result = await agent.verify(model="gemma-4-26b-a4b-it")

        self.assertTrue(result["ok"])
        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "gemma-4-26b-a4b-it")
        self.assertNotIn("provider", body)

    async def test_verify_without_model_has_only_messages(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "hi"})
        with ctx:
            result = await agent.verify()

        self.assertTrue(result["ok"])
        body = mock_client.post.call_args.kwargs["json"]
        self.assertNotIn("model", body)
        self.assertNotIn("provider", body)
        self.assertIn("messages", body)

    async def test_verify_passes_even_when_agent_ignores_model(self):
        """Agent that returns valid format regardless of model param should pass."""
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "I am using gemma", "tool_calls": []})
        with ctx:
            result = await agent.verify(model="gemma-4-26b-a4b-it")

        self.assertTrue(result["ok"])

    async def test_verify_openrouter_format_model_passed_as_is(self):
        """OpenRouter format model string is passed as-is, no provider split."""
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "hi"})
        with ctx:
            result = await agent.verify(model="google/gemma-4-26b-a4b-it")

        self.assertTrue(result["ok"])
        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "google/gemma-4-26b-a4b-it")
        self.assertNotIn("provider", body)


# ---------------------------------------------------------------------------
# Tests for _run_single_model folder naming
# ---------------------------------------------------------------------------

class TestFolderNaming(unittest.IsolatedAsyncioTestCase):

    async def _get_folder(self, model: str, agent=True):
        """Run _run_single_model and capture the output dir it creates."""
        import os
        import tempfile
        from calibrate.connections import TextAgentConnection
        from calibrate.llm import _Tests

        fake_agent = TextAgentConnection(url="http://fake-agent/chat") if agent else None
        fake_body = {"response": "hello", "tool_calls": []}
        test_cases = [
            {
                "history": [{"role": "user", "content": "hi"}],
                "evaluation": {"type": "response", "criteria": "greet"},
            }
        ]
        mock_judge = AsyncMock(
            return_value={"correctness": {"match": True, "reasoning": "ok"}}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx, _ = _patch_httpx(fake_body)
            with ctx, patch("calibrate.llm.run_tests.test_response_llm_judge", mock_judge):
                await _Tests._run_single_model(
                    system_prompt="",
                    tools=[],
                    test_cases=test_cases,
                    output_dir=tmpdir,
                    model=model,
                    provider="openrouter",
                    agent=fake_agent,
                )
            created = [
                d for d in os.listdir(tmpdir)
                if os.path.isdir(os.path.join(tmpdir, d))
            ]
            return created[0] if created else None

    async def test_model_name_folder_uses_double_underscore_for_slash(self):
        """Model names with / use __ as separator in folder name."""
        folder = await self._get_folder("google/gemma-4-26b-a4b-it")
        self.assertEqual(folder, "google__gemma-4-26b-a4b-it")

    async def test_plain_model_name_used_directly(self):
        """Plain model name (no slash) used as folder name directly."""
        folder = await self._get_folder("gpt-4o")
        self.assertEqual(folder, "gpt-4o")

    async def test_no_model_saves_directly_to_output_dir(self):
        """Single agent run with no model should save directly to output_dir, no subfolder."""
        folder = await self._get_folder("")
        self.assertIsNone(folder)

    async def test_default_model_string_does_not_leak_into_folder(self):
        """Single agent run with no model saves directly to output_dir — no subfolders created."""
        import os
        import tempfile
        from calibrate.connections import TextAgentConnection
        from calibrate.llm import tests as _tests

        agent = TextAgentConnection(url="http://fake-agent/chat")
        test_cases = [
            {
                "history": [{"role": "user", "content": "hi"}],
                "evaluation": {"type": "response", "criteria": "greet"},
            }
        ]
        mock_judge = AsyncMock(
            return_value={"correctness": {"match": True, "reasoning": "ok"}}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx, _ = _patch_httpx({"response": "hello"})
            with ctx, patch("calibrate.llm.run_tests.test_response_llm_judge", mock_judge):
                await _tests.run(
                    agent=agent,
                    test_cases=test_cases,
                    output_dir=tmpdir,
                    # no model/models passed — simulates plain agent connection run
                )
            subfolders = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
            # No subfolders — results go directly into tmpdir
            self.assertEqual(subfolders, [])
            # results.json written directly to output_dir
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "results.json")))


# ---------------------------------------------------------------------------
# Tests for --debug / --debug_count truncating test cases in benchmark.main
# ---------------------------------------------------------------------------

class TestBenchmarkDebugFlag(unittest.IsolatedAsyncioTestCase):
    async def _run_main(self, argv_extra):
        import json
        import os
        import sys
        import tempfile
        from calibrate.llm import benchmark

        config = {
            "system_prompt": "p",
            "tools": [],
            "test_cases": [{"id": str(i)} for i in range(10)],
        }

        captured = {}

        async def fake_run(*, config, models, provider, output_dir, test_parallel=None):
            captured["config"] = config
            return {
                "status": "completed",
                "output_dir": output_dir,
                "leaderboard_dir": os.path.join(output_dir, "leaderboard"),
                "models": {},
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            with open(cfg_path, "w") as f:
                json.dump(config, f)

            argv = ["calibrate", "-c", cfg_path, "-m", "gpt-4.1", "-o", tmpdir]
            argv.extend(argv_extra)
            with patch.object(sys, "argv", argv), \
                 patch("calibrate.llm.benchmark.run", side_effect=fake_run), \
                 patch("calibrate.llm.benchmark.print_benchmark_summary", return_value=False):
                await benchmark.main()
        return captured["config"]

    async def test_debug_truncates_test_cases(self):
        config = await self._run_main(["-d", "-dc", "3"])
        self.assertEqual(len(config["test_cases"]), 3)

    async def test_debug_default_count(self):
        config = await self._run_main(["-d"])
        self.assertEqual(len(config["test_cases"]), 5)

    async def test_no_debug_keeps_all_test_cases(self):
        config = await self._run_main([])
        self.assertEqual(len(config["test_cases"]), 10)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
