"""Tests for scripts/sdk_reference.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sdk_reference import (  # noqa: E402
    api_group_from_tag,
    build_route_map,
    load_route_map,
    parse_fern_overrides,
    parse_reference_md,
    routes_with_sdk_docs,
)

FIXTURE = """\
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

FERN_OVERRIDES_FIXTURE = ROOT / "tests" / "fixtures" / "fern_openapi_overrides.yml"
OPENAPI_FIXTURE = ROOT / "docs" / "api-reference" / "openapi.json"


@pytest.fixture
def public_openapi() -> dict:
    if not OPENAPI_FIXTURE.is_file():
        pytest.skip("docs/api-reference/openapi.json not present")
    return json.loads(OPENAPI_FIXTURE.read_text(encoding="utf-8"))


def test_api_group_from_tag() -> None:
    assert api_group_from_tag("agents") == "Agents"
    assert api_group_from_tag("agent-tests") == "Agent tests"


def test_build_route_map_joins_fern_overrides_with_openapi(public_openapi: dict) -> None:
    overrides = parse_fern_overrides(FERN_OVERRIDES_FIXTURE)
    routes = build_route_map(overrides, public_openapi)
    by_key = {(r.http, r.path): r for r in routes}
    list_agents = by_key[("GET", "/agents")]
    assert list_agents.sdk_group == "agents"
    assert list_agents.sdk_method == "list"
    assert list_agents.title == "List agents"
    assert list_agents.api_group == "Agents"
    run_batch = by_key[("POST", "/agent-tests/run")]
    assert run_batch.sdk_method == "run_batch"
    assert run_batch.title == "Run agent tests in batch"
    assert run_batch.api_group == "Agent tests"


def test_parse_reference_md_extracts_method() -> None:
    methods = parse_reference_md(FIXTURE)
    assert ("agents", "list") in methods
    doc = methods[("agents", "list")]
    assert "List all agents" in doc.description
    assert "client.agents.list()" in doc.usage_code
    assert 'api_key="sk_your_api_key"' in doc.usage_code


def test_load_route_map_has_agent_tests_run(public_openapi: dict) -> None:
    routes = load_route_map(
        openapi=public_openapi,
        overrides_path=FERN_OVERRIDES_FIXTURE,
    )
    keys = {(r.http.upper(), r.path) for r in routes}
    assert ("POST", "/agent-tests/agent/{agent_uuid}/run") in keys


def test_routes_with_sdk_docs_pairs_only_when_reference_exists(
    public_openapi: dict,
) -> None:
    methods = parse_reference_md(FIXTURE)
    routes = load_route_map(
        openapi=public_openapi,
        overrides_path=FERN_OVERRIDES_FIXTURE,
    )
    paired = routes_with_sdk_docs(routes, methods)
    assert len(paired) == 1
    route, doc = paired[0]
    assert route.sdk_method == "list"
    assert doc.usage_code


def test_parse_real_reference_file() -> None:
    reference = ROOT.parent / "calibrate-python-sdk" / "reference.md"
    if not reference.is_file():
        pytest.skip("calibrate-python-sdk/reference.md not present locally")
    methods = parse_reference_md(reference.read_text(encoding="utf-8"))
    assert ("agents", "list") in methods
    assert ("agent_tests", "run_batch") in methods
