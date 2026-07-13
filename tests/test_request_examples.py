"""Tests for scripts/request_examples.py — spec examples → SDK/CLI snippets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cli_reference import Option  # noqa: E402
from request_examples import (  # noqa: E402
    command_key,
    learn_more_markdown,
    named_request_examples,
    render_cli_snippet,
    render_python_snippet,
)

_OP_WITH_EXAMPLES = {
    "requestBody": {
        "content": {
            "application/json": {
                "examples": {
                    "build": {
                        "summary": "Agent within Calibrate",
                        "description": "Build inside Calibrate.",
                        "value": {"name": "Support Agent", "type": "agent"},
                    },
                    "connect": {
                        "summary": "Connect external agent",
                        "value": {
                            "name": "My Hosted Agent",
                            "type": "connection",
                            "config": {"agent_url": "https://x.example.com/v1"},
                        },
                    },
                }
            }
        }
    }
}


def test_named_request_examples_extracts_in_order() -> None:
    examples = named_request_examples(_OP_WITH_EXAMPLES)
    assert [e.key for e in examples] == ["build", "connect"]
    assert examples[0].summary == "Agent within Calibrate"
    assert examples[0].description == "Build inside Calibrate."
    # Missing summary falls back to the key; missing description is empty.
    assert examples[1].summary == "Connect external agent"
    assert examples[1].description == ""


def test_named_request_examples_empty_when_absent() -> None:
    assert named_request_examples({}) == []
    assert named_request_examples({"requestBody": {"content": {}}}) == []
    # Entries without a `value` are skipped.
    op = {"requestBody": {"content": {"application/json": {"examples": {"x": {}}}}}}
    assert named_request_examples(op) == []


def test_render_python_snippet_kwargs_and_literals() -> None:
    value = {
        "name": "Support Agent",
        "type": "agent",
        "config": {
            "settings": {"agent_speaks_first": False, "max_assistant_turns": 10},
            "tags": [],
        },
    }
    snippet = render_python_snippet("agents", "create", value)
    assert snippet.startswith("client.agents.create(\n")
    assert '    name="Support Agent",' in snippet
    # JSON literals become Python literals, double-quoted keys, nested indent.
    assert '"agent_speaks_first": False' in snippet
    assert '"max_assistant_turns": 10' in snippet
    assert '"tags": []' in snippet
    assert snippet.endswith(")")


def test_render_python_snippet_non_object_is_positional() -> None:
    assert render_python_snippet("a", "b", "hi") == 'client.a.b("hi")'
    assert render_python_snippet("a", "b", {}) == "client.a.b()"


def test_render_cli_snippet_maps_fields_to_flags() -> None:
    options = [
        Option(flag="--name", type="string"),
        Option(flag="--type", type="type=agent"),
        Option(flag="-c, --config-param", type="type"),
    ]
    value = {
        "name": "Support Agent",
        "type": "connection",
        "config": {"agent_url": "https://x.example.com/v1"},
    }
    snippet = render_cli_snippet("calibrate agents create", options, value)
    assert snippet.startswith("calibrate agents create \\\n")
    assert '--name "Support Agent"' in snippet
    assert '--type "connection"' in snippet
    # `config` resolves to the renamed `--config-param` flag; object → JSON string.
    assert "--config-param '{\"agent_url\":\"https://x.example.com/v1\"}'" in snippet


def test_render_cli_snippet_falls_back_to_field_flag() -> None:
    # No matching option → `--<field>` fallback rather than dropping the field.
    snippet = render_cli_snippet("calibrate x y", [], {"foo": "bar"})
    assert snippet == 'calibrate x y \\\n  --foo "bar"'


def test_command_key_normalizes() -> None:
    assert command_key("agents", "create") == "agents create"
    assert command_key("agent_tests", "run") == "agent-tests run"


def test_learn_more_markdown() -> None:
    note = learn_more_markdown("agents create")
    assert note == (
        "For more details, see [Agent connections](/core-concepts/agent-connections)."
    )
    # No pointer for operations without a configured link.
    assert learn_more_markdown("tests create") is None
