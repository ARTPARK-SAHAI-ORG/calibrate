"""Tests for scripts/generate_mcp_docs.py (MCP tool page generator)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_mcp_docs import (  # noqa: E402
    Arg,
    _anchor,
    _params_table,
    _schema_type,
    _summary_line,
    _tool_sort_key,
    available_tools_table,
    build_operation_map,
    build_tab,
    generate_mcp_docs,
)
from mcp_reference import parse_tool_ts  # noqa: E402
from mcp_tool_samples import (  # noqa: E402
    CREATE_WIDGET_TS,
    GET_WIDGET_TS,
    LIST_WIDGETS_TS,
    OPENAPI,
    ORPHAN_TS,
    write_samples,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_schema_type_labels() -> None:
    assert _schema_type({"type": "string"}) == "string"
    assert _schema_type({"$ref": "#/components/schemas/WidgetCreate"}) == "WidgetCreate"
    assert _schema_type({"type": "array", "items": {"type": "string"}}) == "array of string"
    assert _schema_type({"anyOf": [{"type": "string"}, {"type": "null"}]}) == "string"
    assert _schema_type({}) == "object"


def test_summary_line_collapses_multiline() -> None:
    assert _summary_line("First line.\n\n- bullet\n- bullet") == "First line."
    assert _summary_line("\n\n  padded  \nmore") == "padded"
    assert _summary_line("") == ""


def test_anchor_matches_mintlify_heading_slug() -> None:
    assert _anchor("Agents") == "agents"
    assert _anchor("Agent tests") == "agent-tests"


def test_tool_sort_key_orders_read_before_write() -> None:
    tools = [parse_tool_ts(t) for t in (CREATE_WIDGET_TS, GET_WIDGET_TS, LIST_WIDGETS_TS)]
    ordered = sorted(tools, key=_tool_sort_key)
    # list -> get -> create (CRUD read-before-write), mirroring Coval
    assert [t.name for t in ordered] == ["list-widgets", "get-widget", "create-widget"]


def test_params_table_marks_required_and_escapes_pipe() -> None:
    table = _params_table(
        [
            Arg(name="name", type="string", required=True, description="A | B"),
            Arg(name="color", type="string", required=False, description=""),
        ]
    )
    assert "| Parameter | Type | Required | Description |" in table
    assert "| `name` | string | Yes | A \\| B |" in table
    assert "| `color` | string | No | — |" in table


def test_params_table_empty_when_no_args() -> None:
    assert _params_table([]) == ""


def test_available_tools_table_links_to_anchors() -> None:
    tools = [parse_tool_ts(GET_WIDGET_TS), parse_tool_ts(CREATE_WIDGET_TS)]
    table = available_tools_table([("Widgets", tools)])
    assert "| Category | Tools |" in table
    assert "[Widgets](/mcp/tools#widgets)" in table
    assert "`get-widget`, `create-widget`" in table


# --------------------------------------------------------------------------- #
# build_operation_map
# --------------------------------------------------------------------------- #
def test_operation_map_keys_and_args() -> None:
    ops = build_operation_map(OPENAPI)
    assert "createwidgetwidgetspostop" in ops
    create = ops["createwidgetwidgetspostop"]
    assert create.tag == "widgets"
    assert create.api_page == "POST /widgets"
    arg_names = {a.name: a for a in create.args}
    assert arg_names["name"].required is True
    assert arg_names["color"].required is False
    assert arg_names["color"].type == "string"


def test_operation_map_drops_header_params() -> None:
    ops = build_operation_map(OPENAPI)
    get = ops["getwidgetwidgetswidgetuuidgetop"]
    assert [a.name for a in get.args] == ["widget_uuid"]  # X-API-Key dropped


def test_operation_map_unwraps_optional_body_anyof() -> None:
    ops = build_operation_map(OPENAPI)
    run = ops["rungadgetsgadgetsrunpostop"]
    assert [a.name for a in run.args] == ["gadget_names"]
    assert run.args[0].required is False
    assert run.args[0].type == "array of string"


# --------------------------------------------------------------------------- #
# build_tab
# --------------------------------------------------------------------------- #
def test_build_tab_shape() -> None:
    tab = build_tab(["mcp/overview", "mcp/installation"], ["mcp/tools"])
    assert tab["tab"] == "MCP"
    assert tab["groups"][0] == {
        "group": "Getting started",
        "pages": ["mcp/overview", "mcp/installation"],
    }
    assert tab["groups"][1] == {"group": "Tools reference", "pages": ["mcp/tools"]}


def test_build_tab_omits_empty_reference_group() -> None:
    tab = build_tab(["mcp/overview"], [])
    assert len(tab["groups"]) == 1


# --------------------------------------------------------------------------- #
# end-to-end generate_mcp_docs
# --------------------------------------------------------------------------- #
def _docs_json(tmp_path: Path) -> Path:
    docs_json = tmp_path / "docs.json"
    docs_json.write_text(
        json.dumps(
            {
                "navigation": {
                    "tabs": [
                        {"tab": "Home"},
                        {"tab": "Python SDK"},
                        {"tab": "Integrations"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return docs_json


def _template_dir(tmp_path: Path) -> Path:
    tdir = tmp_path / "templates"
    tdir.mkdir(exist_ok=True)
    # overview carries the injection token; installation is a plain page.
    (tdir / "overview.mdx").write_text(
        "# MCP overview\n\n## Available tools\n\n{/* AVAILABLE_TOOLS */}\n",
        encoding="utf-8",
    )
    (tdir / "installation.mdx").write_text("# Install\n", encoding="utf-8")
    return tdir


def _run(tmp_path: Path, samples=None):
    tools_dir = write_samples(tmp_path / "tools", samples)
    docs_json = _docs_json(tmp_path)
    output_root = tmp_path / "docs" / "mcp"
    generate_mcp_docs(
        tools_dir,
        docs_json=docs_json,
        output_root=output_root,
        template_dir=_template_dir(tmp_path),
        openapi=OPENAPI,
    )
    return output_root, docs_json


def test_generate_writes_overview_installation_and_tools(tmp_path: Path) -> None:
    output_root, _ = _run(tmp_path)
    assert (output_root / "overview.mdx").is_file()
    assert (output_root / "installation.mdx").is_file()
    assert (output_root / "tools.mdx").is_file()
    # no per-resource pages in the Coval-style structure
    assert not (output_root / "widgets.mdx").exists()


def test_overview_gets_available_tools_injected(tmp_path: Path) -> None:
    output_root, _ = _run(tmp_path)
    overview = (output_root / "overview.mdx").read_text(encoding="utf-8")
    assert "{/* AVAILABLE_TOOLS */}" not in overview  # token replaced
    assert "| Category | Tools |" in overview
    assert "[Widgets](/mcp/tools#widgets)" in overview
    assert "[Gadgets](/mcp/tools#gadgets)" in overview


def test_tools_page_content(tmp_path: Path) -> None:
    output_root, _ = _run(tmp_path)
    tools = (output_root / "tools.mdx").read_text(encoding="utf-8")
    assert 'title: "Tools"' in tools
    # category headings (## ) and per-tool sections (### )
    assert "## Widgets" in tools
    assert "### create-widget" in tools
    # read-before-write ordering within a category
    assert tools.index("### list-widgets") < tools.index("### get-widget")
    assert tools.index("### get-widget") < tools.index("### create-widget")
    # scope/access + parameters table + escaped cross-link
    assert "**Scope:** `write` · **Access:** Write" in tools
    assert "| `name` | string | Yes | Widget name. |" in tools
    assert "GET /widgets/&#123;widget_uuid&#125;" in tools
    # no-arg tool omits the Parameters block
    list_section = tools[tools.index("### list-widgets"):tools.index("### get-widget")]
    assert "**Parameters**" not in list_section


def test_generate_inserts_tab_after_python_sdk(tmp_path: Path) -> None:
    _, docs_json = _run(tmp_path)
    names = [t["tab"] for t in json.loads(docs_json.read_text())["navigation"]["tabs"]]
    assert names == ["Home", "Python SDK", "MCP", "Integrations"]


def test_generate_is_idempotent(tmp_path: Path) -> None:
    output_root, docs_json = _run(tmp_path)
    first_tools = (output_root / "tools.mdx").read_text()
    first_json = docs_json.read_text()
    generate_mcp_docs(
        write_samples(tmp_path / "tools"),
        docs_json=docs_json,
        output_root=output_root,
        template_dir=_template_dir(tmp_path),
        openapi=OPENAPI,
    )
    assert (output_root / "tools.mdx").read_text() == first_tools
    names = [t["tab"] for t in json.loads(docs_json.read_text())["navigation"]["tabs"]]
    assert names.count("MCP") == 1
    assert docs_json.read_text() == first_json


def test_generate_prunes_stale_pages(tmp_path: Path) -> None:
    output_root, docs_json = _run(tmp_path)
    stale = output_root / "obsolete.mdx"
    stale.write_text("old", encoding="utf-8")
    generate_mcp_docs(
        write_samples(tmp_path / "tools"),
        docs_json=docs_json,
        output_root=output_root,
        template_dir=_template_dir(tmp_path),
        openapi=OPENAPI,
    )
    assert not stale.exists()


def test_unmatched_tool_aborts_before_writing(tmp_path: Path) -> None:
    docs_json = _docs_json(tmp_path)
    output_root = tmp_path / "docs" / "mcp"
    with pytest.raises(SystemExit, match="matched no OpenAPI operation"):
        generate_mcp_docs(
            write_samples(tmp_path / "tools", {"orphan.ts": ORPHAN_TS}),
            docs_json=docs_json,
            output_root=output_root,
            template_dir=_template_dir(tmp_path),
            openapi=OPENAPI,
        )
    assert not output_root.exists()
    assert "MCP" not in docs_json.read_text()
