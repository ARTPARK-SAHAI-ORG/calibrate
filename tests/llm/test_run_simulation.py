"""
Unit tests for the ConversationState class in calibrate_agent.llm.run_simulation.

All tests are pure async — no mocks or external dependencies needed.
"""

import asyncio
import pytest

from calibrate_agent.llm.run_simulation import ConversationState


# ---------------------------------------------------------------------------
# 1. record_turn increments turn_count
# ---------------------------------------------------------------------------

def test_record_turn_increments_turn_count():
    state = ConversationState(max_turns=5)
    asyncio.run(state.record_turn())
    assert state.turn_count == 1
    asyncio.run(state.record_turn())
    assert state.turn_count == 2


# ---------------------------------------------------------------------------
# 2. record_turn returns True while under max_turns
# ---------------------------------------------------------------------------

def test_record_turn_returns_true_while_under_max_turns():
    state = ConversationState(max_turns=3)
    result = asyncio.run(state.record_turn())
    assert result is True
    result = asyncio.run(state.record_turn())
    assert result is True


# ---------------------------------------------------------------------------
# 3. record_turn returns False when turn_count >= max_turns
# ---------------------------------------------------------------------------

def test_record_turn_returns_false_at_max_turns():
    state = ConversationState(max_turns=2)
    asyncio.run(state.record_turn())  # turn 1 -> True
    result = asyncio.run(state.record_turn())  # turn 2 -> reaches max, returns False
    assert result is False
    assert state.finished is True


# ---------------------------------------------------------------------------
# 4. record_turn returns False when already finished (after mark_finished)
# ---------------------------------------------------------------------------

def test_record_turn_returns_false_when_already_finished():
    state = ConversationState(max_turns=10)
    asyncio.run(state.mark_finished())
    result = asyncio.run(state.record_turn())
    assert result is False


# ---------------------------------------------------------------------------
# 5. mark_finished returns True on first call
# ---------------------------------------------------------------------------

def test_mark_finished_returns_true_first_call():
    state = ConversationState(max_turns=5)
    result = asyncio.run(state.mark_finished())
    assert result is True


# ---------------------------------------------------------------------------
# 6. mark_finished returns False on second call (idempotent)
# ---------------------------------------------------------------------------

def test_mark_finished_returns_false_second_call():
    state = ConversationState(max_turns=5)
    asyncio.run(state.mark_finished())
    result = asyncio.run(state.mark_finished())
    assert result is False


# ---------------------------------------------------------------------------
# 7. finished flag is True after mark_finished
# ---------------------------------------------------------------------------

def test_finished_flag_true_after_mark_finished():
    state = ConversationState(max_turns=5)
    assert state.finished is False
    asyncio.run(state.mark_finished())
    assert state.finished is True


# ---------------------------------------------------------------------------
# 8. Concurrent record_turn calls from multiple tasks don't exceed max_turns
# ---------------------------------------------------------------------------

def test_concurrent_record_turn_does_not_exceed_max_turns():
    """Fire many concurrent record_turn tasks; the turn count must not exceed max_turns."""

    async def _run():
        max_turns = 5
        state = ConversationState(max_turns=max_turns)
        # Launch 20 concurrent calls
        results = await asyncio.gather(*[state.record_turn() for _ in range(20)])
        # turn_count must not exceed max_turns
        assert state.turn_count <= max_turns, (
            f"turn_count ({state.turn_count}) exceeded max_turns ({max_turns})"
        )
        # Number of True results must equal max_turns - 1 (since the max-th turn returns False)
        true_count = sum(1 for r in results if r is True)
        assert true_count == max_turns - 1, (
            f"Expected {max_turns - 1} True results, got {true_count}"
        )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Evaluator arguments: scenario-level (live) and per-item (eval-only)
# ---------------------------------------------------------------------------

import argparse
import json
import sys
import tempfile
from unittest.mock import patch, AsyncMock, MagicMock

from pytest_httpserver import HTTPServer

from calibrate_agent.llm.run_simulation import (
    run_eval_only_simulation_task,
    run_single_simulation_task,
    validate_simulation_eval_only_dataset,
)

TEMPLATED_EV = {
    "name": "goal_met",
    "system_prompt": "Did the agent handle {{order_id}}?",
    "judge_model": "openai/gpt-4.1",
}

OTHER_EV = {
    "name": "politeness",
    "system_prompt": "Was the agent polite about {{order_id}}?",
    "judge_model": "openai/gpt-4.1",
}


def _mock_openai():
    completion = MagicMock()
    completion.choices[0].message.content = "I need help with my order."
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    return MagicMock(return_value=client)


@pytest.fixture
def agent_server(httpserver: HTTPServer):
    httpserver.expect_request("/chat", method="POST").respond_with_json(
        {"response": "Hello!", "tool_calls": []}
    )
    return httpserver


def _run_live_simulation(agent_url, scenario, evaluators, tmp_path):
    """Run one live simulation; return the evaluators the judge received."""
    from calibrate_agent.connections import TextAgentConnection

    seen = []

    async def fake_judge(_transcript, evs, **_kwargs):
        seen.append(evs)
        return {ev["name"]: {"match": True, "reasoning": "ok"} for ev in evs}

    config = {
        "agent_url": agent_url,
        "personas": [{"characteristics": "friendly", "language": "english"}],
        "scenarios": [scenario],
        "evaluators": evaluators,
        "settings": {"agent_speaks_first": True, "max_turns": 2},
    }

    async def _inner():
        with (
            patch("openai.AsyncOpenAI", _mock_openai()),
            patch(
                "calibrate_agent.llm.run_simulation.evaluate_simuation",
                AsyncMock(side_effect=fake_judge),
            ),
        ):
            await run_single_simulation_task(
                semaphore=asyncio.Semaphore(1),
                config=config,
                persona_index=0,
                user_persona=config["personas"][0],
                scenario_index=0,
                scenario=scenario,
                output_dir=str(tmp_path),
                args=argparse.Namespace(model="gpt-4.1", provider="openai"),
                agent=TextAgentConnection(url=agent_url),
            )

    asyncio.run(_inner())
    assert seen, "judge was never called"
    return seen[0]


def test_scenario_arguments_reach_the_judge(agent_server, tmp_path):
    scenario = {
        "name": "vague inquiry",
        "description": "ask about order",
        "arguments": {"goal_met": {"order_id": "ORD-67890"}},
    }
    evaluators = _run_live_simulation(
        agent_server.url_for("/chat"), scenario, [dict(TEMPLATED_EV)], tmp_path
    )
    assert evaluators[0]["system_prompt"] == "Did the agent handle ORD-67890?"


def test_scenario_without_arguments_leaves_placeholder(agent_server, tmp_path):
    scenario = {"name": "s1", "description": "ask about order"}
    evaluators = _run_live_simulation(
        agent_server.url_for("/chat"), scenario, [dict(TEMPLATED_EV)], tmp_path
    )
    assert evaluators[0]["system_prompt"] == "Did the agent handle {{order_id}}?"


def test_scenario_arguments_target_only_named_evaluator(agent_server, tmp_path):
    scenario = {
        "name": "s1",
        "description": "ask about order",
        "arguments": {"goal_met": {"order_id": "ORD-1"}},
    }
    evaluators = _run_live_simulation(
        agent_server.url_for("/chat"),
        scenario,
        [dict(TEMPLATED_EV), dict(OTHER_EV)],
        tmp_path,
    )
    by_name = {ev["name"]: ev["system_prompt"] for ev in evaluators}
    assert by_name["goal_met"] == "Did the agent handle ORD-1?"
    assert by_name["politeness"] == "Was the agent polite about {{order_id}}?"


def test_scenario_unknown_evaluator_name_raises(agent_server, tmp_path):
    scenario = {
        "name": "s1",
        "description": "ask about order",
        "arguments": {"goal_mett": {"order_id": "ORD-1"}},
    }
    with pytest.raises(ValueError) as excinfo:
        _run_live_simulation(
            agent_server.url_for("/chat"), scenario, [dict(TEMPLATED_EV)], tmp_path
        )
    assert "goal_mett" in str(excinfo.value)


def _run_eval_only_item(item, evaluators):
    """Evaluate one pre-existing transcript; return the evaluators the judge received."""
    seen = []

    async def fake_judge(_transcript, evs, **_kwargs):
        seen.append(evs)
        return {ev["name"]: {"match": True, "reasoning": "ok"} for ev in evs}

    async def _inner(tmp):
        with patch(
            "calibrate_agent.llm.run_simulation.evaluate_simuation",
            AsyncMock(side_effect=fake_judge),
        ):
            await run_eval_only_simulation_task(
                semaphore=asyncio.Semaphore(1),
                item=item,
                item_index=0,
                evaluators=evaluators,
                output_dir=tmp,
            )

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(_inner(tmp))
    assert seen, "judge was never called"
    return seen[0]


def test_eval_only_item_arguments_reach_the_judge():
    evaluators = _run_eval_only_item(
        {
            "conversation_history": [{"role": "user", "content": "hi"}],
            "arguments": {"goal_met": {"order_id": "ORD-42"}},
        },
        [dict(TEMPLATED_EV)],
    )
    assert evaluators[0]["system_prompt"] == "Did the agent handle ORD-42?"


def test_eval_only_item_unknown_evaluator_name_raises():
    with pytest.raises(ValueError) as excinfo:
        _run_eval_only_item(
            {
                "conversation_history": [{"role": "user", "content": "hi"}],
                "arguments": {"goal_mett": {"order_id": "ORD-42"}},
            },
            [dict(TEMPLATED_EV)],
        )
    assert "goal_mett" in str(excinfo.value)


def test_eval_only_dataset_accepts_valid_arguments():
    ok, err = validate_simulation_eval_only_dataset(
        [{"conversation_history": [], "arguments": {"goal_met": {"order_id": "1"}}}]
    )
    assert ok, err


def test_eval_only_dataset_rejects_non_object_arguments():
    ok, err = validate_simulation_eval_only_dataset(
        [{"conversation_history": [], "arguments": "nope"}]
    )
    assert not ok
    assert "Item 0" in err


def test_eval_only_dataset_rejects_non_object_evaluator_arguments():
    ok, err = validate_simulation_eval_only_dataset(
        [{"conversation_history": [], "arguments": {"goal_met": "nope"}}]
    )
    assert not ok
    assert "goal_met" in err


def test_main_rejects_malformed_scenario_arguments(tmp_path, monkeypatch, capsys):
    from calibrate_agent.llm import run_simulation as rs

    config = {
        "agent_url": "http://localhost:1/chat",
        "personas": [{"characteristics": "friendly", "language": "english"}],
        "scenarios": [{"name": "s1", "description": "d", "arguments": "nope"}],
        "evaluators": [dict(TEMPLATED_EV)],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_simulation", "-c", str(config_path), "-o", str(tmp_path / "out")],
    )
    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(rs.main())
    assert excinfo.value.code == 1
    assert "scenario 1" in capsys.readouterr().err


def test_main_rejects_unknown_evaluator_in_scenario_arguments(
    tmp_path, monkeypatch, capsys
):
    # A misspelled evaluator name stops the run during config validation,
    # before any simulation spends a judge call.
    from calibrate_agent.llm import run_simulation as rs

    config = {
        "agent_url": "http://localhost:1/chat",
        "personas": [{"characteristics": "friendly", "language": "english"}],
        "scenarios": [
            {"name": "s1", "description": "d", "arguments": {"typoed": {"a": "b"}}}
        ],
        "evaluators": [dict(TEMPLATED_EV)],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_simulation", "-c", str(config_path), "-o", str(tmp_path / "out")],
    )
    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(rs.main())
    assert excinfo.value.code == 1
    assert "typoed" in capsys.readouterr().err


def test_library_simulation_rejects_unknown_evaluator_in_scenario_arguments(tmp_path):
    # The library entry point stops on a misspelled evaluator name instead of
    # returning a completed run whose simulations all silently failed.
    from calibrate_agent.llm import simulations

    with patch(
        "calibrate_agent.llm.run_simulation.run_single_simulation_task",
        AsyncMock(return_value=({}, [])),
    ):
        with pytest.raises(ValueError) as excinfo:
            asyncio.run(
                simulations.run(
                    personas=[{"characteristics": "friendly", "language": "english"}],
                    scenarios=[
                        {"description": "d", "arguments": {"typoed": {"order_id": "1"}}}
                    ],
                    evaluators=[dict(TEMPLATED_EV)],
                    output_dir=str(tmp_path / "out"),
                )
            )
    assert "typoed" in str(excinfo.value)


def test_eval_only_dataset_rejects_unknown_evaluator_name():
    ok, err = validate_simulation_eval_only_dataset(
        [{"conversation_history": [], "arguments": {"goal_mett": {"order_id": "1"}}}],
        [dict(TEMPLATED_EV)],
    )
    assert not ok
    assert "goal_mett" in err


def test_eval_only_run_prints_failure_reason(tmp_path, capsys):
    from calibrate_agent.llm.run_simulation import run_eval_only_simulations

    with patch(
        "calibrate_agent.llm.run_simulation.evaluate_simuation",
        AsyncMock(side_effect=RuntimeError("judge exploded")),
    ):
        failed = asyncio.run(
            run_eval_only_simulations(
                config={"evaluators": [dict(TEMPLATED_EV)]},
                dataset=[{"conversation_history": [{"role": "user", "content": "hi"}]}],
                output_dir=str(tmp_path / "out"),
            )
        )
    assert failed == 1
    assert "judge exploded" in capsys.readouterr().out


def test_main_leaves_no_output_dir_when_scenario_check_fails(
    tmp_path, monkeypatch, capsys
):
    from calibrate_agent.llm import run_simulation as rs

    config = {
        "agent_url": "http://localhost:1/chat",
        "personas": [{"characteristics": "friendly", "language": "english"}],
        "scenarios": [
            {"name": "s1", "description": "d", "arguments": {"typoed": {"a": "b"}}}
        ],
        "evaluators": [dict(TEMPLATED_EV)],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys, "argv", ["run_simulation", "-c", str(config_path), "-o", str(out_dir)]
    )
    with pytest.raises(SystemExit):
        asyncio.run(rs.main())
    assert not out_dir.exists()
