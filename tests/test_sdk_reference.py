"""Tests for scripts/sdk_reference.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sdk_reference import (  # noqa: E402
    load_route_map,
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


def test_parse_reference_md_extracts_method() -> None:
    methods = parse_reference_md(FIXTURE)
    assert ("agents", "list") in methods
    doc = methods[("agents", "list")]
    assert "List all agents" in doc.description
    assert "client.agents.list()" in doc.usage_code
    assert 'api_key="sk_your_api_key"' in doc.usage_code


def test_load_route_map_has_agent_tests_run() -> None:
    routes = load_route_map()
    keys = {(r.http.upper(), r.path) for r in routes}
    assert ("POST", "/agent-tests/agent/{agent_uuid}/run") in keys


def test_routes_with_sdk_docs_pairs_only_when_reference_exists() -> None:
    methods = parse_reference_md(FIXTURE)
    routes = load_route_map()
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
