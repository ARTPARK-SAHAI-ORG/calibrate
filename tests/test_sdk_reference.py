"""Tests for scripts/sdk_reference.py."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from sdk_reference import (  # noqa: E402
    SdkRoute,
    api_group_from_tag,
    build_route_map,
    load_route_map,
    parse_fern_overrides,
    parse_reference_md,
    routes_with_sdk_docs,
)
from sdk_route_samples import SAMPLE_OPENAPI, SAMPLE_OVERRIDES  # noqa: E402

REFERENCE_FIXTURE = dedent(
    """\
    # Reference
    ## Agents
    <details><summary><code>client.agents.<a href="x">list</a>() -> list</code></summary>
    <dl><dd>
    #### 📝 Description
    <dl><dd><dl><dd>
    List all agents for the caller's current org.
    </dd></dl></dd></dl>
    #### 🔌 Usage
    <dl><dd><dl><dd>
    ```python
    from calibrate import Calibrate

    client = Calibrate(
        api_key="<value>",
    )

    client.agents.list()
    ```
    </dd></dl></dd></dl>
    </dd></dl>
    </details>
    """
)

# Minimal shapes that cover every field the merger reads — not a copy of production routes.


@pytest.fixture
def overrides_file(tmp_path: Path) -> Path:
    path = tmp_path / "openapi-overrides.yml"
    path.write_text(yaml.dump(SAMPLE_OVERRIDES), encoding="utf-8")
    return path


def test_api_group_from_tag() -> None:
    assert api_group_from_tag("agents") == "Agents"
    assert api_group_from_tag("agent-tests") == "Agent tests"
    assert api_group_from_tag("") == ""


def test_parse_fern_overrides_reads_yaml(overrides_file: Path) -> None:
    data = parse_fern_overrides(overrides_file)
    assert data["paths"]["/agents"]["get"]["x-fern-sdk-method-name"] == "list"


def test_build_route_map_joins_sdk_names_summary_and_tags() -> None:
    routes = build_route_map(SAMPLE_OVERRIDES, SAMPLE_OPENAPI)
    by_key = {(route.http, route.path): route for route in routes}

    list_agents = by_key[("GET", "/agents")]
    assert list_agents.sdk_group == "agents"
    assert list_agents.sdk_method == "list"
    assert list_agents.title == "List agents"
    assert list_agents.api_group == "Agents"

    run_batch = by_key[("POST", "/agent-tests/run")]
    assert run_batch.sdk_group == "agent_tests"
    assert run_batch.sdk_method == "run_batch"
    assert run_batch.api_group == "Agent tests"

    update_agent = by_key[("PUT", "/agents/{agent_uuid}")]
    assert update_agent.sdk_method == "update"


def test_build_route_map_sorts_paths_and_methods() -> None:
    routes = build_route_map(SAMPLE_OVERRIDES, SAMPLE_OPENAPI)
    assert [route.http for route in routes if route.path == "/agents"] == ["GET", "POST"]
    assert routes[0].path == "/agent-tests/run"
    assert routes[1].path == "/agents"


def test_build_route_map_raises_when_openapi_operation_missing() -> None:
    openapi = {"paths": {"/agents": {"get": {"tags": ["agents"], "summary": "List agents"}}}}
    overrides = {
        "paths": {
            "/agents": {
                "get": {
                    "x-fern-sdk-group-name": "agents",
                    "x-fern-sdk-method-name": "list",
                },
                "post": {
                    "x-fern-sdk-group-name": "agents",
                    "x-fern-sdk-method-name": "create",
                },
            }
        }
    }
    with pytest.raises(ValueError, match="OpenAPI spec missing POST /agents"):
        build_route_map(overrides, openapi)


def test_build_route_map_raises_when_tags_missing() -> None:
    overrides = {
        "paths": {
            "/agents": {
                "get": {
                    "x-fern-sdk-group-name": "agents",
                    "x-fern-sdk-method-name": "list",
                }
            }
        }
    }
    openapi = {"paths": {"/agents": {"get": {"summary": "List agents"}}}}
    with pytest.raises(ValueError, match="missing tags"):
        build_route_map(overrides, openapi)


def test_build_route_map_skips_override_without_sdk_names() -> None:
    overrides = {
        "paths": {
            "/agents": {
                "get": {"x-fern-sdk-group-name": "agents"},
                "post": {
                    "x-fern-sdk-group-name": "agents",
                    "x-fern-sdk-method-name": "create",
                },
            }
        }
    }
    openapi = {
        "paths": {
            "/agents": {
                "post": {"tags": ["agents"], "summary": "Create agent"},
            }
        }
    }
    routes = build_route_map(overrides, openapi)
    assert len(routes) == 1
    assert routes[0].sdk_method == "create"


def test_load_route_map_reads_overrides_file(overrides_file: Path) -> None:
    routes = load_route_map(openapi=SAMPLE_OPENAPI, overrides_path=overrides_file)
    assert any(
        route.http == "POST" and route.path == "/agent-tests/run" for route in routes
    )


def test_parse_reference_md_extracts_method() -> None:
    methods = parse_reference_md(REFERENCE_FIXTURE)
    assert ("agents", "list") in methods
    doc = methods[("agents", "list")]
    assert "List all agents" in doc.description
    assert "client.agents.list()" in doc.usage_code
    assert 'api_key="your_api_key"' in doc.usage_code


def test_routes_with_sdk_docs_pairs_only_when_reference_exists() -> None:
    methods = parse_reference_md(REFERENCE_FIXTURE)
    routes = build_route_map(SAMPLE_OVERRIDES, SAMPLE_OPENAPI)
    paired = routes_with_sdk_docs(routes, methods)
    assert len(paired) == 1
    route, doc = paired[0]
    assert route.sdk_method == "list"
    assert doc.usage_code


def test_routes_with_sdk_docs_omits_unmatched_sdk_methods() -> None:
    routes = [
        SdkRoute(
            http="GET",
            path="/agents",
            sdk_group="agents",
            sdk_method="list",
            api_group="Agents",
            title="List agents",
        ),
        SdkRoute(
            http="POST",
            path="/agents",
            sdk_group="agents",
            sdk_method="create",
            api_group="Agents",
            title="Create agent",
        ),
    ]
    methods = parse_reference_md(REFERENCE_FIXTURE)
    paired = routes_with_sdk_docs(routes, methods)
    assert len(paired) == 1
    assert paired[0][0].sdk_method == "list"


def test_parse_real_reference_file() -> None:
    reference = ROOT.parent / "calibrate-python-sdk" / "reference.md"
    if not reference.is_file():
        pytest.skip("calibrate-python-sdk/reference.md not present locally")
    methods = parse_reference_md(reference.read_text(encoding="utf-8"))
    assert ("agents", "list") in methods
    assert ("agent_tests", "run_batch") in methods
