#!/usr/bin/env python3
"""Fetch public OpenAPI spec, generate SDK docs, and render docs templates."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_sdk_docs import generate_sdk_docs, prune_stale_sdk_pages

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "api-reference" / "openapi.json"
API_KEY_SCHEME = "ApiKeyAuth"
DEFAULT_REFERENCE = REPO_ROOT.parent / "calibrate-python-sdk" / "reference.md"

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


def sdk_reference_path() -> Path:
    raw = os.getenv("SDK_REFERENCE_PATH", "").strip()
    if raw:
        return Path(raw)
    if DEFAULT_REFERENCE.is_file():
        return DEFAULT_REFERENCE
    raise SystemExit(
        "SDK_REFERENCE_PATH is required when calibrate-python-sdk/reference.md "
        "is not available locally."
    )


def _normalize_for_docs(spec: dict[str, Any], base_url: str) -> dict[str, Any]:
    spec = json.loads(json.dumps(spec))

    spec["servers"] = [{"url": base_url, "description": "Production"}]

    components = spec.setdefault("components", {})
    components["securitySchemes"] = {
        API_KEY_SCHEME: {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication",
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


def sync_cli_docs() -> None:
    """Generate cloud-CLI docs when ``CLI_DOCS_PATH`` points at a checkout.

    Optional: skipped (with a note) when the env var is unset or the directory
    is missing, so a local ``fetch`` run without the CLI repo still works.
    """
    src = os.getenv("CLI_DOCS_PATH", "").strip()
    if not src:
        print("CLI_DOCS_PATH unset — skipping cloud CLI docs sync")
        return
    src_dir = Path(src)
    if not src_dir.is_dir():
        print(f"CLI_DOCS_PATH not a directory ({src_dir}) — skipping CLI docs sync")
        return
    # SCRIPTS_DIR is already on sys.path (module top), so this import resolves.
    from generate_cli_docs import generate_cli_docs

    generate_cli_docs(src_dir)


def sync_mcp_docs() -> None:
    """Generate MCP docs when ``MCP_DOCS_PATH`` points at a tools directory.

    Optional: skipped (with a note) when the env var is unset or the directory is
    missing, so a local ``fetch`` run without the calibrate-mcp repo still works.
    ``MCP_DOCS_PATH`` points at ``calibrate-mcp/src/mcp-server/tools``; the docs
    read the freshly written OpenAPI spec (``DEFAULT_OUT``) for tool arguments.
    """
    src = os.getenv("MCP_DOCS_PATH", "").strip()
    if not src:
        print("MCP_DOCS_PATH unset — skipping MCP docs sync")
        return
    src_dir = Path(src)
    if not src_dir.is_dir():
        print(f"MCP_DOCS_PATH not a directory ({src_dir}) — skipping MCP docs sync")
        return
    # SCRIPTS_DIR is already on sys.path (module top), so this import resolves.
    from generate_mcp_docs import generate_mcp_docs

    generate_mcp_docs(src_dir)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    base_url = public_api_base_url()
    url = argv[0] if argv else public_openapi_spec_url(base_url)
    out = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT
    reference = sdk_reference_path()

    spec = _normalize_for_docs(fetch(url), base_url)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    render_templates(base_url)
    sdk_slugs = generate_sdk_docs(reference, spec)
    prune_stale_sdk_pages(set(sdk_slugs))

    print(f"Wrote {out} ({len(spec.get('paths', {}))} paths)")
    for _, output in TEMPLATED_PAGES:
        print(f"Wrote {output}")
    print(f"Wrote {len(sdk_slugs)} SDK pages from {reference}")
    sync_cli_docs()
    sync_mcp_docs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
