"""Tests for scripts/fetch_public_openapi.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_public_openapi import _normalize_for_docs  # noqa: E402


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


def test_normalize_adds_servers_and_api_key_auth(minimal_spec: dict) -> None:
    out = _normalize_for_docs(minimal_spec)
    assert out["servers"] == [
        {"url": "https://pense-backend.artpark.ai", "description": "Production"}
    ]
    assert set(out["components"]["securitySchemes"]) == {"ApiKeyAuth"}
    assert out["paths"]["/agents"]["get"]["security"] == [{"ApiKeyAuth": []}]
    assert "parameters" not in out["paths"]["/agents"]["get"]


def test_normalize_does_not_mutate_input(minimal_spec: dict) -> None:
    before = json.dumps(minimal_spec)
    _normalize_for_docs(minimal_spec)
    assert json.dumps(minimal_spec) == before
