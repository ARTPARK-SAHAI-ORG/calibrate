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

INTRO_TEMPLATE = REPO_ROOT / "docs/templates/api-reference/introduction.mdx"
INTRO_OUTPUT = REPO_ROOT / "docs/api-reference/introduction.mdx"

# (template, output) pairs whose base-URL placeholders are single-sourced from
# PUBLIC_API_BASE_URL. Keeps every docs page off a hardcoded backend host.
TEMPLATED_PAGES = [
    (INTRO_TEMPLATE, INTRO_OUTPUT),
    (
        REPO_ROOT / "docs/templates/reference/api-keys.mdx",
        REPO_ROOT / "docs/reference/api-keys.mdx",
    ),
    (
        REPO_ROOT / "docs/templates/reference/github-actions.mdx",
        REPO_ROOT / "docs/reference/github-actions.mdx",
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


def _normalize_for_docs(spec: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Ensure servers + API-key auth match what Mintlify expects."""
    spec = json.loads(json.dumps(spec))  # deep copy

    spec["servers"] = [{"url": base_url, "description": "Production"}]

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

    return spec


def render_templates(base_url: str) -> None:
    spec_url = public_openapi_spec_url(base_url)
    for template, output in TEMPLATED_PAGES:
        text = template.read_text(encoding="utf-8")
        text = text.replace("__PUBLIC_API_BASE_URL__", base_url)
        text = text.replace("__PUBLIC_OPENAPI_SPEC_URL__", spec_url)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


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
    render_templates(base_url)
    print(f"Wrote {out} ({len(spec.get('paths', {}))} paths)")
    for _, output in TEMPLATED_PAGES:
        print(f"Wrote {output}")


if __name__ == "__main__":
    raise SystemExit(main())
