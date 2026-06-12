"""
Integration tests for `calibrate llm` CLI against a fake HTTP agent server.

Requires: pytest-httpserver
    pip install pytest-httpserver

Run with:
    python -m pytest tests/test_cli_integration.py -v

Key design choices:
- Use tool_call evaluation type to avoid needing a real LLM judge.
- The fake server always responds with a valid TOOL_CALL_RESPONSE.
- Verify request uses the default _DEFAULT_VERIFY_MESSAGES from connections.py:
      [{"role": "user", "content": "Hello, are you there?"}]
- Count of requests: single run = 1 verify + 1 test = 2 total.
                     benchmark 2 models = 2 verifies + 2 tests = 4 total.
"""

import json
import os
import sys
import subprocess

import pytest
from pytest_httpserver import HTTPServer


# ---------------------------------------------------------------------------
# Fake responses
# ---------------------------------------------------------------------------

TOOL_CALL_RESPONSE = {
    "response": None,
    "tool_calls": [{"tool": "get_weather", "arguments": {"location": "Mumbai"}}],
}

# BAD_RESPONSE triggers verify failure (missing required keys)
BAD_RESPONSE = {}

# ---------------------------------------------------------------------------
# Test cases that use tool_call evaluation — no LLM judge needed
# ---------------------------------------------------------------------------

TEST_CASES_TOOL_CALL = [
    {
        "history": [{"role": "user", "content": "Get weather in Mumbai"}],
        "evaluation": {
            "type": "tool_call",
            "tool_calls": [{"tool": "get_weather", "arguments": {"location": "Mumbai"}}],
        },
    }
]

# The default verify message sent by calibrate (from calibrate/connections.py)
VERIFY_MESSAGE_CONTENT = "Hello, are you there?"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cli(*args, extra_env=None):
    """Run `calibrate llm ...` as a subprocess and return CompletedProcess."""
    env = {
        **os.environ,
        "OPENAI_API_KEY": "sk-fake",
        "OPENROUTER_API_KEY": "sk-fake",
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "calibrate.cli", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def _write_config(path, agent_url, test_cases=None):
    """Write a JSON config file and return the path."""
    config = {
        "agent_url": agent_url,
        "test_cases": test_cases if test_cases is not None else TEST_CASES_TOOL_CALL,
    }
    with open(path, "w") as f:
        json.dump(config, f)
    return path


def _request_bodies(httpserver: HTTPServer):
    """Return parsed JSON bodies of all requests received by the fake server."""
    bodies = []
    for req, _ in httpserver.log:
        try:
            bodies.append(json.loads(req.data))
        except Exception:
            bodies.append({})
    return bodies


def _is_verify_request(body: dict) -> bool:
    """Return True if this request looks like a verify (default greeting) request."""
    messages = body.get("messages", [])
    if not messages:
        return False
    return any(
        msg.get("content", "") == VERIFY_MESSAGE_CONTENT
        for msg in messages
    )


def _is_test_request(body: dict) -> bool:
    """Return True if this request looks like a real test case request."""
    messages = body.get("messages", [])
    if not messages:
        return False
    return any(
        msg.get("content", "") == "Get weather in Mumbai"
        for msg in messages
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_server(httpserver: HTTPServer):
    """Fake agent that always returns a valid tool-call response."""
    httpserver.expect_request("/chat", method="POST").respond_with_json(TOOL_CALL_RESPONSE)
    return httpserver


@pytest.fixture
def agent_config(tmp_path, agent_server):
    """Write config.json pointing at the fake server; return (cfg_path, out_dir)."""
    cfg_path = tmp_path / "config.json"
    out_dir = str(tmp_path / "out")
    _write_config(str(cfg_path), agent_server.url_for("/chat"))
    return str(cfg_path), out_dir


@pytest.fixture
def bad_agent_server(httpserver: HTTPServer):
    """Fake agent that returns a bad response (causes verify to fail)."""
    httpserver.expect_request("/chat", method="POST").respond_with_json(BAD_RESPONSE)
    return httpserver


@pytest.fixture
def bad_agent_config(tmp_path, bad_agent_server):
    cfg_path = tmp_path / "config.json"
    out_dir = str(tmp_path / "out")
    _write_config(str(cfg_path), bad_agent_server.url_for("/chat"))
    return str(cfg_path), out_dir


# ---------------------------------------------------------------------------
# TestAgentSingleRun
# ---------------------------------------------------------------------------

class TestAgentSingleRun:
    """Single run — no model flag, results saved directly to output_dir."""

    def test_exit_0_on_success(self, agent_config):
        cfg, out = agent_config
        result = run_cli("llm", "-c", cfg, "-o", out)
        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"

    def test_results_json_at_output_root(self, agent_config, tmp_path):
        cfg, out = agent_config
        run_cli("llm", "-c", cfg, "-o", out)
        results_path = os.path.join(out, "results.json")
        assert os.path.exists(results_path), (
            f"results.json not found at {results_path}. "
            f"out dir contents: {os.listdir(out) if os.path.exists(out) else 'dir missing'}"
        )

    def test_no_leaderboard_generated(self, agent_config):
        cfg, out = agent_config
        run_cli("llm", "-c", cfg, "-o", out)
        leaderboard_dir = os.path.join(out, "leaderboard")
        assert not os.path.exists(leaderboard_dir), (
            "leaderboard/ dir should not exist for single runs"
        )

    def test_verify_then_test_requests(self, agent_config, agent_server):
        cfg, out = agent_config
        run_cli("llm", "-c", cfg, "-o", out)
        bodies = _request_bodies(agent_server)
        assert len(bodies) == 2, (
            f"Expected 2 requests (1 verify + 1 test), got {len(bodies)}: {bodies}"
        )
        # First should be verify, second should be the test case
        assert _is_verify_request(bodies[0]), f"First request is not verify: {bodies[0]}"
        assert _is_test_request(bodies[1]), f"Second request is not test: {bodies[1]}"


# ---------------------------------------------------------------------------
# TestAgentBenchmark
# ---------------------------------------------------------------------------

class TestAgentBenchmark:
    """Benchmark mode — two models, per-model subfolders, leaderboard."""

    MODELS = ["gpt-4.1", "gpt-5.1"]

    def _run_benchmark(self, agent_config):
        cfg, out = agent_config
        return run_cli("llm", "-c", cfg, "-o", out, "-m", *self.MODELS), cfg, out

    def test_exit_0_with_two_models(self, agent_config):
        result, _, _ = self._run_benchmark(agent_config)
        assert result.returncode == 0, (
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )

    def test_per_model_subfolders_created(self, agent_config):
        _, _, out = self._run_benchmark(agent_config)
        for model in self.MODELS:
            model_dir = os.path.join(out, model)
            assert os.path.isdir(model_dir), (
                f"Expected subfolder for model '{model}' at {model_dir}. "
                f"out dir: {os.listdir(out) if os.path.exists(out) else 'missing'}"
            )

    def test_leaderboard_csv_generated(self, agent_config):
        _, _, out = self._run_benchmark(agent_config)
        csv_path = os.path.join(out, "leaderboard", "llm_leaderboard.csv")
        assert os.path.exists(csv_path), (
            f"leaderboard CSV not found at {csv_path}"
        )

    def test_each_model_verified_separately(self, agent_config, agent_server):
        _, _, _ = self._run_benchmark(agent_config)
        bodies = _request_bodies(agent_server)
        # With 2 models, there should be 2 verify + 2 test = 4 requests
        assert len(bodies) == 4, (
            f"Expected 4 requests (2 verifies + 2 tests), got {len(bodies)}"
        )
        # Both models should appear as model hints in the request bodies
        model_hints = [b.get("model") for b in bodies if b.get("model")]
        for model in self.MODELS:
            assert model in model_hints, (
                f"Model '{model}' not found in request bodies. Got: {model_hints}"
            )

    def test_per_model_labels_in_stdout(self, agent_config):
        result, _, _ = self._run_benchmark(agent_config)
        # Models run in parallel, so per-test output interleaves; every model's
        # lines must carry its ``[model]`` label to stay attributable.
        for model in self.MODELS:
            assert f"[{model}]" in result.stdout, (
                f"'[{model}]' label not in stdout.\nstdout: {result.stdout}"
            )

    def test_overall_summary_in_stdout(self, agent_config):
        result, _, _ = self._run_benchmark(agent_config)
        assert "Overall Summary" in result.stdout, (
            f"'Overall Summary' not in stdout.\nstdout: {result.stdout}"
        )


# ---------------------------------------------------------------------------
# TestSkipVerify
# ---------------------------------------------------------------------------

class TestSkipVerify:
    """--skip-verify flag skips verification request."""

    def test_skip_verify_flag_skips_verification(self, agent_config, agent_server):
        cfg, out = agent_config
        run_cli("llm", "-c", cfg, "-o", out, "--skip-verify")
        bodies = _request_bodies(agent_server)
        # With skip-verify, only 1 request (the test itself)
        assert len(bodies) == 1, (
            f"Expected 1 request (no verify), got {len(bodies)}: {bodies}"
        )
        assert _is_test_request(bodies[0]), f"Only request is not the test: {bodies[0]}"


# ---------------------------------------------------------------------------
# TestVerifyCommand
# ---------------------------------------------------------------------------

class TestVerifyCommand:
    """--verify flag alone (no -c config): verify connection only."""

    def test_verify_flag_success(self, agent_server):
        result = run_cli(
            "llm", "--verify", "--agent-url", agent_server.url_for("/chat")
        )
        assert result.returncode == 0, (
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        assert "✓" in result.stdout, f"'✓' not in stdout: {result.stdout}"

    def test_verify_flag_failure(self, bad_agent_server):
        result = run_cli(
            "llm", "--verify", "--agent-url", bad_agent_server.url_for("/chat")
        )
        assert result.returncode == 1, (
            f"Expected exit 1, got {result.returncode}\nstdout: {result.stdout}"
        )
        assert "✗" in result.stdout, f"'✗' not in stdout: {result.stdout}"


# ---------------------------------------------------------------------------
# TestVerifyFailureDuringRun
# ---------------------------------------------------------------------------

class TestVerifyFailureDuringRun:
    """When agent returns bad response, verify fails before tests run."""

    def test_bad_agent_response_exits_1(self, bad_agent_config):
        cfg, out = bad_agent_config
        result = run_cli("llm", "-c", cfg, "-o", out)
        assert result.returncode == 1, (
            f"Expected exit 1, got {result.returncode}\nstdout: {result.stdout}"
        )

    def test_error_message_in_stdout(self, bad_agent_config):
        cfg, out = bad_agent_config
        result = run_cli("llm", "-c", cfg, "-o", out)
        # Should show verify failure message
        assert "✗" in result.stdout or "Verification failed" in result.stdout, (
            f"No error message in stdout: {result.stdout}"
        )
