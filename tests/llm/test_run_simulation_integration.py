"""
Integration tests for run_simulation_with_agent, run_single_simulation_task, and main()
in calibrate.llm.run_simulation.

Requires: pytest-httpserver
    pip install pytest-httpserver

Strategy:
  - pytest-httpserver for the external agent HTTP endpoint
  - unittest.mock.patch for openai.AsyncOpenAI (the user simulator)
  - unittest.mock.patch for calibrate.llm.run_simulation.evaluate_simuation
  - tmp_path fixture for output dirs
  - asyncio.run() to call async functions directly
"""

import asyncio
import argparse
import json
import os
import sys

import pytest
from pytest_httpserver import HTTPServer
from unittest.mock import patch, AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Fake HTTP agent response
# ---------------------------------------------------------------------------

FAKE_AGENT_RESPONSE = {"response": "Hello!", "tool_calls": []}

# ---------------------------------------------------------------------------
# Fake evaluate_simuation return value
# ---------------------------------------------------------------------------

FAKE_EVAL_RESULT = {"helpfulness": {"match": True, "reasoning": "looks good"}}

# ---------------------------------------------------------------------------
# Helpers to build mock OpenAI client
# ---------------------------------------------------------------------------


def _make_mock_openai_client(content="I need help with my order."):
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = content
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    MockAsyncOpenAI = MagicMock(return_value=mock_client)
    return MockAsyncOpenAI


# ---------------------------------------------------------------------------
# Minimal simulation config builder
# ---------------------------------------------------------------------------


_HELPFULNESS_EVALUATOR = {
    "name": "helpfulness",
    "system_prompt": "Evaluate whether the agent was helpful.",
    "judge_model": "openai/gpt-5.2",
}


def _make_config(agent_url, max_turns=2, agent_speaks_first=True, num_personas=1, num_scenarios=1):
    personas = [
        {"label": f"p{i+1}", "characteristics": "friendly", "gender": "neutral", "language": "english"}
        for i in range(num_personas)
    ]
    scenarios = [
        {"name": f"s{i+1}", "description": "ask about order"}
        for i in range(num_scenarios)
    ]
    return {
        "agent_url": agent_url,
        "personas": personas,
        "scenarios": scenarios,
        "evaluators": [_HELPFULNESS_EVALUATOR],
        "settings": {"agent_speaks_first": agent_speaks_first, "max_turns": max_turns},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_server(httpserver: HTTPServer):
    """Fake agent that always returns a valid text response."""
    httpserver.expect_request("/chat", method="POST").respond_with_json(FAKE_AGENT_RESPONSE)
    return httpserver


@pytest.fixture
def agent_server_500(httpserver: HTTPServer):
    """Fake agent that always returns 500."""
    httpserver.expect_request("/chat", method="POST").respond_with_data(
        "Internal Server Error", status=500, content_type="text/plain"
    )
    return httpserver


# ---------------------------------------------------------------------------
# TestRunSimulationWithAgent
# ---------------------------------------------------------------------------


class TestRunSimulationWithAgent:
    """6 tests for run_simulation_with_agent."""

    def _run(self, agent_url, max_turns=2, agent_speaks_first=True, user_content="I need help"):
        from calibrate.llm.run_simulation import run_simulation_with_agent
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url=agent_url)
        MockAsyncOpenAI = _make_mock_openai_client(user_content)

        async def _inner():
            with patch("openai.AsyncOpenAI", MockAsyncOpenAI), \
                 patch("calibrate.llm.run_simulation.evaluate_simuation",
                       AsyncMock(return_value=FAKE_EVAL_RESULT)):
                return await run_simulation_with_agent(
                    agent=agent,
                    user_system_prompt="You are a friendly user.",
                    evaluators=[_HELPFULNESS_EVALUATOR],
                    agent_speaks_first=agent_speaks_first,
                    max_turns=max_turns,
                    user_model="gpt-4.1",
                    user_provider="openai",
                )

        return asyncio.run(_inner())

    def test_agent_speaks_first_sends_initial_request(self, agent_server):
        """agent_speaks_first=True: httpserver gets a request before user turn (first body has 'Hi')."""
        self._run(agent_server.url_for("/chat"), agent_speaks_first=True)

        bodies = [json.loads(req.data) for req, _ in agent_server.log]
        assert len(bodies) >= 1, "Expected at least one request to agent"
        first_messages = bodies[0].get("messages", [])
        assert any(
            msg.get("content") == "Hi" for msg in first_messages
        ), f"First request should contain 'Hi' greeting, got: {first_messages}"

    def test_agent_speaks_first_false_no_initial_request(self, agent_server):
        """agent_speaks_first=False: first request has user message, not 'Hi'."""
        self._run(agent_server.url_for("/chat"), agent_speaks_first=False, user_content="I need help")

        bodies = [json.loads(req.data) for req, _ in agent_server.log]
        assert len(bodies) >= 1, "Expected at least one request to agent"
        first_messages = bodies[0].get("messages", [])
        assert not any(
            msg.get("content") == "Hi" for msg in first_messages
        ), f"First request should NOT contain 'Hi' greeting when agent_speaks_first=False, got: {first_messages}"

    def test_transcript_has_correct_roles(self, agent_server):
        """result['transcript'] starts with assistant (agent speaks first) then alternates."""
        result = self._run(agent_server.url_for("/chat"), agent_speaks_first=True, max_turns=2)

        transcript = result["transcript"]
        # Filter out end_reason entries
        messages = [m for m in transcript if m["role"] != "end_reason"]
        assert len(messages) >= 2, f"Expected at least 2 messages, got: {messages}"
        # With agent_speaks_first=True: first is assistant
        assert messages[0]["role"] == "assistant", (
            f"First transcript role should be 'assistant', got: {messages[0]['role']}"
        )
        # Roles should alternate
        for i in range(1, len(messages)):
            expected = "user" if messages[i - 1]["role"] == "assistant" else "assistant"
            assert messages[i]["role"] == expected, (
                f"Role at index {i} should be '{expected}', got '{messages[i]['role']}'"
            )

    def test_max_turns_respected(self, agent_server):
        """max_turns=2: httpserver receives exactly 3 requests (1 initial 'Hi' + 2 turns)."""
        self._run(agent_server.url_for("/chat"), agent_speaks_first=True, max_turns=2)

        num_requests = len(agent_server.log)
        assert num_requests == 3, (
            f"Expected 3 requests (1 initial + 2 turns), got {num_requests}"
        )

    def test_evaluate_called_with_transcript_and_evaluators(self, agent_server):
        """evaluate_simuation mock called once with the transcript and evaluators list."""
        from calibrate.llm.run_simulation import run_simulation_with_agent
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url=agent_server.url_for("/chat"))
        MockAsyncOpenAI = _make_mock_openai_client()
        eval_mock = AsyncMock(return_value=FAKE_EVAL_RESULT)
        evaluators = [_HELPFULNESS_EVALUATOR]

        async def _inner():
            with patch("openai.AsyncOpenAI", MockAsyncOpenAI), \
                 patch("calibrate.llm.run_simulation.evaluate_simuation", eval_mock):
                return await run_simulation_with_agent(
                    agent=agent,
                    user_system_prompt="You are a friendly user.",
                    evaluators=evaluators,
                    agent_speaks_first=True,
                    max_turns=2,
                )

        asyncio.run(_inner())

        eval_mock.assert_called_once()
        call_args = eval_mock.call_args
        transcript_arg = call_args[0][0]
        evaluators_arg = call_args[0][1]
        assert isinstance(transcript_arg, list), "First arg to evaluate_simuation should be a list"
        assert evaluators_arg == evaluators, "Second arg should be evaluators list"

    def test_returns_transcript_and_evaluation_results(self, agent_server):
        """Return dict has 'transcript' and 'evaluation_results' keys with correct content."""
        result = self._run(agent_server.url_for("/chat"), max_turns=2)

        assert "transcript" in result, "Result must have 'transcript' key"
        assert "evaluation_results" in result, "Result must have 'evaluation_results' key"

        eval_results = result["evaluation_results"]
        assert len(eval_results) == 1, f"Expected 1 evaluation result, got {len(eval_results)}"
        item = eval_results[0]
        assert "name" in item, "Evaluation result item must have 'name'"
        assert "value" in item, "Evaluation result item must have 'value'"
        assert "reasoning" in item, "Evaluation result item must have 'reasoning'"
        assert item["name"] == "helpfulness"
        assert item["value"] == 1  # match=True -> int(True) == 1


# ---------------------------------------------------------------------------
# TestRunSingleSimulationTask
# ---------------------------------------------------------------------------


class TestRunSingleSimulationTask:
    """5 tests for run_single_simulation_task."""

    def _run_task(self, agent_url, tmp_path, persona_index=0, scenario_index=0, max_turns=2):
        from calibrate.llm.run_simulation import run_single_simulation_task
        from calibrate.connections import TextAgentConnection

        config = _make_config(agent_url, max_turns=max_turns)
        agent = TextAgentConnection(url=agent_url)
        semaphore = asyncio.Semaphore(1)
        args = argparse.Namespace(model="gpt-4.1", provider="openai")
        persona = config["personas"][persona_index]
        scenario = config["scenarios"][scenario_index]
        MockAsyncOpenAI = _make_mock_openai_client()

        async def _inner():
            with patch("openai.AsyncOpenAI", MockAsyncOpenAI), \
                 patch("calibrate.llm.run_simulation.evaluate_simuation",
                       AsyncMock(return_value=FAKE_EVAL_RESULT)):
                return await run_single_simulation_task(
                    semaphore=semaphore,
                    config=config,
                    persona_index=persona_index,
                    user_persona=persona,
                    scenario_index=scenario_index,
                    scenario=scenario,
                    output_dir=str(tmp_path),
                    args=args,
                    agent=agent,
                )

        return asyncio.run(_inner())

    def test_creates_simulation_output_dir(self, agent_server, tmp_path):
        """simulation_persona_1_scenario_1/ directory is created inside tmp_path."""
        self._run_task(agent_server.url_for("/chat"), tmp_path)
        expected_dir = tmp_path / "simulation_persona_1_scenario_1"
        assert expected_dir.is_dir(), (
            f"Expected directory {expected_dir} to exist. "
            f"Contents: {list(tmp_path.iterdir())}"
        )

    def test_writes_transcript_json(self, agent_server, tmp_path):
        """transcript.json exists and is valid JSON containing a list of messages."""
        self._run_task(agent_server.url_for("/chat"), tmp_path)
        transcript_path = tmp_path / "simulation_persona_1_scenario_1" / "transcript.json"
        assert transcript_path.exists(), f"transcript.json not found at {transcript_path}"
        with open(transcript_path) as f:
            data = json.load(f)
        assert isinstance(data, list), "transcript.json must be a list"
        assert len(data) >= 1, "transcript.json must have at least one message"

    def test_writes_evaluation_results_csv(self, agent_server, tmp_path):
        """evaluation_results.csv exists with name/value/reasoning columns."""
        self._run_task(agent_server.url_for("/chat"), tmp_path)
        csv_path = tmp_path / "simulation_persona_1_scenario_1" / "evaluation_results.csv"
        assert csv_path.exists(), f"evaluation_results.csv not found at {csv_path}"
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert "name" in df.columns, "CSV must have 'name' column"
        assert "value" in df.columns, "CSV must have 'value' column"
        assert "reasoning" in df.columns, "CSV must have 'reasoning' column"

    def test_writes_config_json(self, agent_server, tmp_path):
        """config.json has persona and scenario keys."""
        self._run_task(agent_server.url_for("/chat"), tmp_path)
        config_path = tmp_path / "simulation_persona_1_scenario_1" / "config.json"
        assert config_path.exists(), f"config.json not found at {config_path}"
        with open(config_path) as f:
            data = json.load(f)
        assert "persona" in data, "config.json must have 'persona' key"
        assert "scenario" in data, "config.json must have 'scenario' key"

    def test_exception_propagates(self, agent_server_500, tmp_path):
        """If agent returns 500, the task raises an exception."""
        from calibrate.llm.run_simulation import run_single_simulation_task
        from calibrate.connections import TextAgentConnection

        config = _make_config(agent_server_500.url_for("/chat"), max_turns=2)
        agent = TextAgentConnection(url=agent_server_500.url_for("/chat"))
        semaphore = asyncio.Semaphore(1)
        args = argparse.Namespace(model="gpt-4.1", provider="openai")
        persona = config["personas"][0]
        scenario = config["scenarios"][0]
        MockAsyncOpenAI = _make_mock_openai_client()

        async def _inner():
            with patch("openai.AsyncOpenAI", MockAsyncOpenAI), \
                 patch("calibrate.llm.run_simulation.evaluate_simuation",
                       AsyncMock(return_value=FAKE_EVAL_RESULT)):
                return await run_single_simulation_task(
                    semaphore=semaphore,
                    config=config,
                    persona_index=0,
                    user_persona=persona,
                    scenario_index=0,
                    scenario=scenario,
                    output_dir=str(tmp_path),
                    args=args,
                    agent=agent,
                )

        with pytest.raises(Exception):
            asyncio.run(_inner())


# ---------------------------------------------------------------------------
# TestSimulationMain
# ---------------------------------------------------------------------------


class TestSimulationMain:
    """5 tests for the main() function (called via asyncio.run with mocked dependencies)."""

    def _run_main(self, agent_url, tmp_path, num_personas=1, num_scenarios=1):
        """Call sim_main() with sys.argv patched, mocking openai and evaluate_simuation."""
        from calibrate.llm.run_simulation import main as sim_main

        config = _make_config(
            agent_url,
            max_turns=2,
            num_personas=num_personas,
            num_scenarios=num_scenarios,
        )
        config_path = tmp_path / "sim_config.json"
        output_dir = tmp_path / "out"
        with open(config_path, "w") as f:
            json.dump(config, f)

        MockAsyncOpenAI = _make_mock_openai_client()

        with patch("sys.argv", ["calibrate", "-c", str(config_path), "-o", str(output_dir)]), \
             patch("openai.AsyncOpenAI", MockAsyncOpenAI), \
             patch("calibrate.llm.run_simulation.evaluate_simuation",
                   AsyncMock(return_value=FAKE_EVAL_RESULT)):
            asyncio.run(sim_main())

        return output_dir

    def test_2_personas_2_scenarios_creates_4_dirs(self, agent_server, tmp_path):
        """With 2 personas and 2 scenarios, 4 simulation dirs are created."""
        output_dir = self._run_main(
            agent_server.url_for("/chat"), tmp_path, num_personas=2, num_scenarios=2
        )
        sim_dirs = [
            d for d in output_dir.iterdir()
            if d.is_dir() and d.name.startswith("simulation_")
        ]
        assert len(sim_dirs) == 4, (
            f"Expected 4 simulation dirs, got {len(sim_dirs)}: {[d.name for d in sim_dirs]}"
        )

    def test_results_csv_has_4_rows(self, agent_server, tmp_path):
        """With 2 personas x 2 scenarios, results.csv has 4 data rows."""
        output_dir = self._run_main(
            agent_server.url_for("/chat"), tmp_path, num_personas=2, num_scenarios=2
        )
        import pandas as pd
        results_csv = output_dir / "results.csv"
        assert results_csv.exists(), f"results.csv not found at {results_csv}"
        df = pd.read_csv(results_csv)
        assert len(df) == 4, f"Expected 4 rows in results.csv, got {len(df)}"

    def test_metrics_json_has_correct_means(self, agent_server, tmp_path):
        """metrics.json helpfulness.mean == 1.0 (all passed since FAKE_EVAL_RESULT match=True)."""
        output_dir = self._run_main(
            agent_server.url_for("/chat"), tmp_path, num_personas=2, num_scenarios=2
        )
        metrics_path = output_dir / "metrics.json"
        assert metrics_path.exists(), f"metrics.json not found at {metrics_path}"
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "helpfulness" in metrics, f"'helpfulness' not in metrics: {metrics}"
        assert metrics["helpfulness"]["mean"] == pytest.approx(1.0), (
            f"Expected helpfulness.mean == 1.0, got {metrics['helpfulness']['mean']}"
        )

    def test_exits_0_on_success(self, agent_server, tmp_path):
        """sys.exit is not called (or called with 0) when all tasks succeed."""
        from calibrate.llm.run_simulation import main as sim_main

        config = _make_config(agent_server.url_for("/chat"), max_turns=2)
        config_path = tmp_path / "sim_config.json"
        output_dir = tmp_path / "out"
        with open(config_path, "w") as f:
            json.dump(config, f)

        MockAsyncOpenAI = _make_mock_openai_client()
        exit_calls = []

        def mock_exit(code=0):
            exit_calls.append(code)

        with patch("sys.argv", ["calibrate", "-c", str(config_path), "-o", str(output_dir)]), \
             patch("openai.AsyncOpenAI", MockAsyncOpenAI), \
             patch("calibrate.llm.run_simulation.evaluate_simuation",
                   AsyncMock(return_value=FAKE_EVAL_RESULT)), \
             patch("sys.exit", mock_exit):
            asyncio.run(sim_main())

        # Either no exit was called, or it was called with 0
        for code in exit_calls:
            assert code == 0 or code is None, f"sys.exit called with non-zero code: {code}"

    def test_exits_1_on_failure(self, agent_server_500, tmp_path):
        """If all tasks fail (agent 500), sys.exit(1) is called."""
        from calibrate.llm.run_simulation import main as sim_main

        config = _make_config(agent_server_500.url_for("/chat"), max_turns=2)
        config_path = tmp_path / "sim_config.json"
        output_dir = tmp_path / "out"
        with open(config_path, "w") as f:
            json.dump(config, f)

        MockAsyncOpenAI = _make_mock_openai_client()
        exit_calls = []

        def mock_exit(code=0):
            exit_calls.append(code)

        with patch("sys.argv", ["calibrate", "-c", str(config_path), "-o", str(output_dir)]), \
             patch("openai.AsyncOpenAI", MockAsyncOpenAI), \
             patch("calibrate.llm.run_simulation.evaluate_simuation",
                   AsyncMock(return_value=FAKE_EVAL_RESULT)), \
             patch("sys.exit", mock_exit):
            asyncio.run(sim_main())

        assert 1 in exit_calls, (
            f"Expected sys.exit(1) to be called on failure, got exit calls: {exit_calls}"
        )


# ---------------------------------------------------------------------------
# TestAgentConnectionDetection
# ---------------------------------------------------------------------------


class TestAgentConnectionDetection:
    """2 tests checking that main() passes correct agent= to run_single_simulation_task."""

    def _run_main_capturing_agent(self, config_dict, tmp_path):
        """Run main() and capture the agent argument passed to run_single_simulation_task."""
        from calibrate.llm.run_simulation import main as sim_main

        config_path = tmp_path / "config.json"
        output_dir = tmp_path / "out"
        with open(config_path, "w") as f:
            json.dump(config_dict, f)

        captured_agents = []
        original_task_fn = None

        async def mock_task(*args, **kwargs):
            captured_agents.append(kwargs.get("agent", "NOT_PASSED"))
            # Return a minimal valid result
            return (
                {"name": "simulation_persona_1_scenario_1", "helpfulness": 1.0},
                [{"name": "helpfulness", "value": 1, "reasoning": "ok"}],
            )

        with patch("sys.argv", ["calibrate", "-c", str(config_path), "-o", str(output_dir)]), \
             patch("calibrate.llm.run_simulation.run_single_simulation_task", side_effect=mock_task):
            asyncio.run(sim_main())

        return captured_agents

    def test_config_without_agent_url_passes_none_agent(self, tmp_path):
        """Config with no agent_url means agent=None passed to each task."""
        config = {
            "personas": [{"label": "p1", "characteristics": "friendly", "gender": "neutral", "language": "english"}],
            "scenarios": [{"name": "s1", "description": "ask about order"}],
            "evaluators": [_HELPFULNESS_EVALUATOR],
            "settings": {"agent_speaks_first": True, "max_turns": 2},
            # no agent_url key
        }
        agents = self._run_main_capturing_agent(config, tmp_path)
        assert len(agents) == 1, f"Expected 1 task to be called, got {len(agents)}"
        assert agents[0] is None, (
            f"Expected agent=None when no agent_url, got: {agents[0]}"
        )

    def test_config_with_agent_url_passes_agent_connection(self, tmp_path, httpserver: HTTPServer):
        """Config with agent_url means a TextAgentConnection is passed to each task."""
        from calibrate.connections import TextAgentConnection

        httpserver.expect_request("/chat", method="POST").respond_with_json(FAKE_AGENT_RESPONSE)
        config = {
            "agent_url": httpserver.url_for("/chat"),
            "personas": [{"label": "p1", "characteristics": "friendly", "gender": "neutral", "language": "english"}],
            "scenarios": [{"name": "s1", "description": "ask about order"}],
            "evaluators": [_HELPFULNESS_EVALUATOR],
            "settings": {"agent_speaks_first": True, "max_turns": 2},
        }
        agents = self._run_main_capturing_agent(config, tmp_path)
        assert len(agents) == 1, f"Expected 1 task to be called, got {len(agents)}"
        assert isinstance(agents[0], TextAgentConnection), (
            f"Expected TextAgentConnection, got: {type(agents[0])}"
        )
        assert agents[0].url == httpserver.url_for("/chat"), (
            f"Expected URL {httpserver.url_for('/chat')}, got {agents[0].url}"
        )
