#!/usr/bin/env python3
"""Fetch Calibrate's public OpenAPI spec and normalize it for Mintlify docs.

Downloads from the live backend, then patches the spec so Mintlify's API
playground and auth UI advertise X-API-Key (the public API's real auth path)
and include a production server URL.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_URL = "https://pense-backend.artpark.ai/public-api/openapi.json"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "docs" / "api-reference" / "openapi.json"
PRODUCTION_SERVER = "https://pense-backend.artpark.ai"
API_KEY_SCHEME = "ApiKeyAuth"


def _normalize_for_docs(spec: dict[str, Any]) -> dict[str, Any]:
    """Ensure servers + API-key auth match what Mintlify expects."""
    spec = json.loads(json.dumps(spec))  # deep copy

    spec["servers"] = [
        {
            "url": PRODUCTION_SERVER,
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
                "Org-scoped API key. Create one in Calibrate under "
                "Settings → API keys. Prefix: `sk_…`."
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
            # Drop FastAPI's optional X-API-Key header param — it's the auth
            # scheme, not a separate optional field.
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
        "Programmatic API for CI/automation, authenticated with an org-scoped API key.",
    )

    return spec


def fetch(url: str = DEFAULT_URL) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    url = argv[0] if argv else DEFAULT_URL
    out = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT

    spec = _normalize_for_docs(fetch(url))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(spec.get('paths', {}))} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
