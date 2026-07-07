"""Tests for scripts/fetch_public_openapi.py and generate_sdk_docs.py."""

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
from generate_sdk_docs import generate_sdk_docs  # noqa: E402

TEST_BASE_URL = "https://api.example.test"
REFERENCE_FIXTURE = ROOT / "tests" / "fixtures" / "reference.md"


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


def test_render_method_page_skips_duplicate_summary_and_signature() -> None:
    from generate_sdk_docs import render_method_page
    from sdk_reference import SdkMethodDoc, SdkRoute

    route = SdkRoute(
        http="POST",
        path="/agent-tests/agent/{agent_uuid}/run",
        sdk_group="agent_tests",
        sdk_method="run",
        api_group="Agent tests",
        title="Run agent tests",
    )
    doc = SdkMethodDoc(
        sdk_group="agent_tests",
        sdk_method="run",
        description="Run tests for an agent as a background job.\n\nExtra detail.",
        usage_code='client.agent_tests.run(agent_uuid="agent_uuid")\n',
    )
    page = render_method_page(route, doc)
    body = page.split("---", 2)[2].split("## Usage")[0]
    assert 'description: "Run tests for an agent as a background job."' in page
    assert "client.agent_tests.run(agent_uuid=" not in body
    assert "Run tests for an agent as a background job." not in body
    assert "Extra detail." in page
    assert "POST /agent-tests/agent/\\{agent_uuid}/run" in page
    assert "## Usage" in page


def test_render_method_page_escapes_quotes_in_frontmatter() -> None:
    from generate_sdk_docs import render_method_page
    from sdk_reference import SdkMethodDoc, SdkRoute

    route = SdkRoute(
        http="PUT",
        path="/agents/{agent_uuid}",
        sdk_group="agents",
        sdk_method="update",
        api_group="Agents",
        title='Update "legacy" agent',
    )
    doc = SdkMethodDoc(
        sdk_group="agents",
        sdk_method="update",
        description='Rename the agent to "production".',
        usage_code="client.agents.update()\n",
    )
    page = render_method_page(route, doc)
    assert 'title: "Update \\"legacy\\" agent"' in page
    assert 'description: "Rename the agent to \\"production\\"."' in page


def test_normalize_strips_x_code_samples(minimal_spec: dict) -> None:
    minimal_spec["paths"]["/agents"]["get"]["x-codeSamples"] = [
        {"lang": "python", "source": "from calibrate import Calibrate\n"}
    ]
    out = _normalize_for_docs(minimal_spec, TEST_BASE_URL)
    assert "x-codeSamples" not in out["paths"]["/agents"]["get"]


def test_render_templates_substitutes_base_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template_a = tmp_path / "intro.mdx"
    template_a.write_text("curl __PUBLIC_API_BASE_URL__/agents\nspec __PUBLIC_OPENAPI_SPEC_URL__\n")
    output_a = tmp_path / "out" / "intro.mdx"

    import fetch_public_openapi as mod

    monkeypatch.setattr(mod, "TEMPLATED_PAGES", [(template_a, output_a)])
    render_templates(TEST_BASE_URL)

    text_a = output_a.read_text()
    assert TEST_BASE_URL in text_a
    assert f"{TEST_BASE_URL}/public-api/openapi.json" in text_a


def test_generate_sdk_docs_writes_pages_and_updates_docs_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import generate_sdk_docs as mod

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    docs_json = docs_dir / "docs.json"
    docs_json.write_text(
        json.dumps(
            {
                "navigation": {
                    "tabs": [
                        {"tab": "CLI", "groups": []},
                        {"tab": "API reference", "groups": []},
                    ]
                },
                "api": {},
            }
        )
        + "\n"
    )
    sdk_root = docs_dir / "sdk"
    overview_template = tmp_path / "overview.template.mdx"
    overview_template.write_text("overview\n")

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "DOCS_JSON", docs_json)
    monkeypatch.setattr(mod, "SDK_ROOT", sdk_root)
    monkeypatch.setattr(mod, "OVERVIEW_TEMPLATE", overview_template)
    monkeypatch.setattr(mod, "OVERVIEW_OUTPUT", sdk_root / "overview.mdx")

    openapi_fixture = ROOT / "docs" / "api-reference" / "openapi.json"
    if not openapi_fixture.is_file():
        pytest.skip("docs/api-reference/openapi.json not present")
    openapi = json.loads(openapi_fixture.read_text(encoding="utf-8"))
    monkeypatch.setenv(
        "FERN_OVERRIDES_PATH",
        str(ROOT / "tests" / "fixtures" / "fern_openapi_overrides.yml"),
    )

    slugs = generate_sdk_docs(REFERENCE_FIXTURE, openapi)
    assert "sdk/overview" in slugs
    assert (sdk_root / "agents" / "list.mdx").is_file()

    mod.prune_stale_sdk_pages(set(slugs))
    assert (sdk_root / "agents" / "list.mdx").is_file()

    docs = json.loads(docs_json.read_text())
    tab_names = [t["tab"] for t in docs["navigation"]["tabs"]]
    assert "Python SDK" in tab_names
    assert docs["api"]["examples"]["languages"] == ["curl", "python"]
    assert docs["api"]["examples"]["autogenerate"] is True
