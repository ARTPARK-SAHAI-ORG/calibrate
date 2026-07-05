"""Tests for scripts/fetch_public_openapi.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_public_openapi import (  # noqa: E402
    _normalize_for_docs,
    public_api_base_url,
    public_openapi_spec_url,
    render_templates,
)

TEST_BASE_URL = "https://api.example.test"


@pytest.fixture(autouse=True)
def api_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLIC_API_BASE_URL", TEST_BASE_URL)


@pytest.fixture
def minimal_spec() -> dict:
    return {
        "openapi": "3.1.0",
        "info": {"title": "Test", "version": "0.0.1"},
        "paths": {
            "/agents": {
                "get": {
                    "tags": ["agents"],
                    "security": [{"HTTPBearer": []}],
                    "parameters": [
                        {"in": "header", "name": "X-API-Key", "schema": {"type": "string"}}
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            },
        },
        "components": {
            "securitySchemes": {
                "HTTPBearer": {"type": "http", "scheme": "bearer"},
            }
        },
    }


def test_public_api_base_url_from_env() -> None:
    assert public_api_base_url() == TEST_BASE_URL
    assert public_openapi_spec_url() == f"{TEST_BASE_URL}/public-api/openapi.json"


def test_public_api_base_url_required(monkeypatch: pytest.MonkeyPatch) -> None:
    import fetch_public_openapi as mod

    monkeypatch.delenv("PUBLIC_API_BASE_URL", raising=False)
    monkeypatch.setattr(mod, "_load_dotenv", lambda: None)
    with pytest.raises(SystemExit):
        public_api_base_url()


def test_normalize_adds_servers_and_api_key_auth(minimal_spec: dict) -> None:
    out = _normalize_for_docs(minimal_spec, TEST_BASE_URL)
    assert out["servers"] == [{"url": TEST_BASE_URL, "description": "Production"}]
    assert set(out["components"]["securitySchemes"]) == {"ApiKeyAuth"}
    assert out["paths"]["/agents"]["get"]["security"] == [{"ApiKeyAuth": []}]
    assert "parameters" not in out["paths"]["/agents"]["get"]


def test_normalize_does_not_mutate_input(minimal_spec: dict) -> None:
    before = json.dumps(minimal_spec)
    _normalize_for_docs(minimal_spec, TEST_BASE_URL)
    assert json.dumps(minimal_spec) == before


def test_render_templates_substitutes_base_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template_a = tmp_path / "intro.mdx"
    template_a.write_text("curl __PUBLIC_API_BASE_URL__/agents\nspec __PUBLIC_OPENAPI_SPEC_URL__\n")
    output_a = tmp_path / "out" / "intro.mdx"

    template_b = tmp_path / "keys.mdx"
    template_b.write_text("base __PUBLIC_API_BASE_URL__\n")
    output_b = tmp_path / "out" / "keys.mdx"

    import fetch_public_openapi as mod

    monkeypatch.setattr(
        mod,
        "TEMPLATED_PAGES",
        [(template_a, output_a), (template_b, output_b)],
    )
    render_templates(TEST_BASE_URL)

    text_a = output_a.read_text()
    assert TEST_BASE_URL in text_a
    assert f"{TEST_BASE_URL}/public-api/openapi.json" in text_a

    assert output_b.read_text() == f"base {TEST_BASE_URL}\n"
