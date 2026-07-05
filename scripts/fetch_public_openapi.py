#!/usr/bin/env python3
"""Fetch Calibrate's public OpenAPI spec and normalize it for Mintlify docs.

Downloads from the live backend, then patches the spec so Mintlify's API
playground and auth UI advertise X-API-Key (the public API's real auth path)
and include a production server URL.

Requires ``PUBLIC_API_BASE_URL`` (see ``.env.example``). Loads ``.env`` from
the repo root when present.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "api-reference" / "openapi.json"
API_KEY_SCHEME = "ApiKeyAuth"

# Python SDK snippets keyed by (HTTP method, OpenAPI path). Method names mirror
# fern/openapi-overrides.yml in pense-backend — keep them aligned when the
# public API changes.
PYTHON_SDK_SAMPLES: dict[tuple[str, str], str] = {
    ("GET", "/agents"): """\
from artpark import Calibrate

client = Calibrate(api_key="your_api_key")

agents = client.agents.list()
for agent in agents:
    print(agent.uuid, agent.name)
""",
    ("POST", "/agents/resolve"): """\
from artpark import Calibrate

client = Calibrate(api_key="your_api_key")

result = client.agents.resolve(names=["my-agent", "other-agent"])
print(result.resolved)  # name -> uuid
print(result.not_found)  # names with no match
""",
    ("POST", "/agent-tests/agent/{agent_uuid}/run"): """\
from artpark import Calibrate

client = Calibrate(api_key="your_api_key")

# Omit test_uuids to run every test linked to the agent.
task = client.agent_tests.run(agent_uuid="your-agent-uuid")
print(task.task_id, task.status)

# Or run a subset:
# task = client.agent_tests.run(
#     agent_uuid="your-agent-uuid",
#     test_uuids=["test-uuid-1", "test-uuid-2"],
# )
""",
    ("POST", "/agent-tests/run"): """\
from artpark import Calibrate

client = Calibrate(api_key="your_api_key")

# Omit agent_names to run every agent in your org.
batch = client.agent_tests.run_batch()
for run in batch.runs:
    print(run.agent_name, run.task_id, run.status)
for skip in batch.skipped:
    print(skip.agent_name, skip.reason)
""",
    ("GET", "/agent-tests/run/{task_id}"): """\
from artpark import Calibrate

client = Calibrate(api_key="your_api_key")

status = client.agent_tests.get_run(task_id="your-task-id")
print(status.status, status.passed, status.failed)
if status.results:
    for row in status.results:
        print(row.name, row.passed)
""",
}

# (template, output) pairs whose base-URL placeholders are single-sourced from
# PUBLIC_API_BASE_URL. Keeps every docs page off a hardcoded backend host.
TEMPLATED_PAGES = [
    (
        REPO_ROOT / "docs/templates/api-reference/introduction.mdx",
        REPO_ROOT / "docs/api-reference/introduction.mdx",
    ),
    (
        REPO_ROOT / "docs/templates/reference/api-keys.mdx",
        REPO_ROOT / "docs/reference/api-keys.mdx",
    ),
    (
        REPO_ROOT / "docs/templates/reference/github-actions.mdx",
        REPO_ROOT / "docs/reference/github-actions.mdx",
    ),
    (
        REPO_ROOT / "docs/templates/cli/installation.mdx",
        REPO_ROOT / "docs/cli/installation.mdx",
    ),
    (
        REPO_ROOT / "docs/templates/snippets/public-api-base-url.mdx",
        REPO_ROOT / "docs/snippets/public-api-base-url.mdx",
    ),
]


def _load_dotenv() -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def public_api_base_url() -> str:
    _load_dotenv()
    url = os.getenv("PUBLIC_API_BASE_URL", "").strip().rstrip("/")
    if not url:
        raise SystemExit(
            "PUBLIC_API_BASE_URL is required (set in .env or the environment). "
            "See .env.example."
        )
    return url


def public_openapi_spec_url(base_url: str | None = None) -> str:
    return f"{base_url or public_api_base_url()}/public-api/openapi.json"


def _inject_sdk_code_samples(spec: dict[str, Any]) -> None:
    """Attach Mintlify-readable x-codeSamples for the generated Python SDK."""
    for path, ops in spec.get("paths", {}).items():
        if not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if not isinstance(op, dict):
                continue
            sample = PYTHON_SDK_SAMPLES.get((method.upper(), path))
            if not sample:
                continue
            op["x-codeSamples"] = [
                {
                    "lang": "python",
                    "label": "Python SDK",
                    "source": sample.rstrip("\n") + "\n",
                }
            ]


def _normalize_for_docs(spec: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Ensure servers + API-key auth match what Mintlify and the SDK expect."""
    spec = json.loads(json.dumps(spec))  # deep copy

    spec["servers"] = [
        {
            "url": base_url,
            "description": "Production",
        }
    ]

    components = spec.setdefault("components", {})
    components["securitySchemes"] = {
        API_KEY_SCHEME: {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": (
                "API key. Create one under Workspace settings → API keys "
                "(https://calibrate.artpark.ai/workspace-settings?tab=api-keys). "
                "Prefix: `sk_…`."
            ),
        }
    }

    for ops in spec.get("paths", {}).values():
        if not isinstance(ops, dict):
            continue
        for op in ops.values():
            if not isinstance(op, dict):
                continue
            op["security"] = [{API_KEY_SCHEME: []}]
            params = op.get("parameters")
            if params:
                op["parameters"] = [
                    p
                    for p in params
                    if not (
                        isinstance(p, dict)
                        and p.get("in") == "header"
                        and p.get("name") in ("X-API-Key", "X-Org-UUID")
                    )
                ]
                if not op["parameters"]:
                    op.pop("parameters", None)

    info = spec.setdefault("info", {})
    info.setdefault(
        "description",
        "Programmatic API for CI/automation, authenticated with an API key.",
    )

    _inject_sdk_code_samples(spec)
    return spec


def render_templates(base_url: str) -> list[Path]:
    """Render MDX templates that embed the public API base URL."""
    spec_url = public_openapi_spec_url(base_url)
    written: list[Path] = []
    for template, output in TEMPLATED_PAGES:
        if not template.is_file():
            continue
        text = template.read_text(encoding="utf-8")
        text = text.replace("__PUBLIC_API_BASE_URL__", base_url)
        text = text.replace("__PUBLIC_OPENAPI_SPEC_URL__", spec_url)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        written.append(output)
    return written


def fetch(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    base_url = public_api_base_url()
    url = argv[0] if argv else public_openapi_spec_url(base_url)
    out = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT

    spec = _normalize_for_docs(fetch(url), base_url)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    rendered = render_templates(base_url)
    print(f"Wrote {out} ({len(spec.get('paths', {}))} paths)")
    for path in rendered:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
