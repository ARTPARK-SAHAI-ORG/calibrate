"""Tests for the spec-examples section in scripts/generate_sdk_docs.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_sdk_docs import _examples_for_route, render_method_page  # noqa: E402
from request_examples import NamedExample  # noqa: E402
from sdk_reference import SdkMethodDoc, SdkRoute  # noqa: E402

_ROUTE = SdkRoute(
    http="POST",
    path="/agents",
    sdk_group="agents",
    sdk_method="create",
    api_group="Agents",
    title="Create agent",
)
_DOC = SdkMethodDoc(
    sdk_group="agents",
    sdk_method="create",
    description="Create an agent.",
    usage_code="client.agents.create(name=\"Support Agent\")\n",
)


def _examples(n: int) -> list[NamedExample]:
    return [
        NamedExample(key=f"k{i}", summary=f"Variant {i}", description="", value={"name": f"a{i}"})
        for i in range(n)
    ]


def test_examples_section_renders_each_variant() -> None:
    page = render_method_page(_ROUTE, _DOC, _examples(2))
    assert "## Examples" in page
    assert "**Variant 0**" in page and "**Variant 1**" in page
    assert page.count("```python") == 3  # 1 Usage + 2 example snippets
    # Ordered: Usage before Examples before the API-endpoint callout.
    assert page.index("## Usage") < page.index("## Examples") < page.index("## API endpoint")


def test_single_example_suppresses_section() -> None:
    # One example is already the Usage block — no point repeating it.
    assert "## Examples" not in render_method_page(_ROUTE, _DOC, _examples(1))
    assert "## Examples" not in render_method_page(_ROUTE, _DOC, [])
    assert "## Examples" not in render_method_page(_ROUTE, _DOC, None)


def test_examples_for_route_reads_op() -> None:
    spec = {
        "paths": {
            "/agents": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "examples": {
                                    "a": {"summary": "A", "value": {"name": "x"}},
                                    "b": {"summary": "B", "value": {"name": "y"}},
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert [e.summary for e in _examples_for_route(_ROUTE, spec)] == ["A", "B"]
    # Unknown route → no examples, no crash.
    missing = SdkRoute("GET", "/nope", "x", "y", "X", "T")
    assert _examples_for_route(missing, spec) == []


def test_examples_section_includes_learn_more_link() -> None:
    from request_examples import NamedExample

    examples = [
        NamedExample(key="build", summary="Build", description="", value={"name": "a"}),
        NamedExample(key="connect", summary="Connect", description="", value={"name": "b"}),
    ]
    page = render_method_page(_ROUTE, _DOC, examples)
    assert "[Agent connections](/core-concepts/agent-connections)" in page
