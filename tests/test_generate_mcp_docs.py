"""Tests for scripts/generate_mcp_docs.py (MCP tool page generator)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_mcp_docs import (
    Arg,
    Operation,
    _anchor,
    _examples_block,
    _params_table,
    _schema_type,
    _summary_line,
    _tool_sort_key,
    available_tools_table,
    build_operation_map,
    build_tab,
    generate_mcp_docs,
)
from mcp_reference import McpTool, parse_tool_ts
from request_examples import NamedExample
from mcp_tool_samples import (
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
    assert (
        _schema_type({"type": "array", "items": {"type": "string"}})
        == "array of string"
    )
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
    tools = [
        parse_tool_ts(t) for t in (CREATE_WIDGET_TS, GET_WIDGET_TS, LIST_WIDGETS_TS)
    ]
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


def test_available_tools_table_links_and_columns() -> None:
    tools = [parse_tool_ts(GET_WIDGET_TS), parse_tool_ts(CREATE_WIDGET_TS)]
    table = available_tools_table([("Widgets", "Manage widgets", tools)])
    # count lead-in + three-column table (Category | Tools | Purpose)
    assert "exposes 2 tools across 1 categories:" in table
    assert "| Category | Tools | Purpose |" in table
    assert "[Widgets](/mcp/tools#widgets)" in table
    assert "`get-widget`, `create-widget`" in table
    assert "Manage widgets" in table


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


def test_operation_map_extracts_named_examples() -> None:
    create = build_operation_map(OPENAPI)["createwidgetwidgetspostop"]
    # value-less "no_value" entry is dropped; only the two real variants survive
    assert [e.key for e in create.examples] == ["basic_widget", "colored_widget"]
    assert create.examples[0].summary == "Basic widget"
    assert create.examples[0].value == {"name": "sprocket"}


# --------------------------------------------------------------------------- #
# _examples_block
# --------------------------------------------------------------------------- #
def _op_with_examples(tag: str, examples: list[NamedExample]) -> Operation:
    return Operation(
        path="/widgets", method="post", summary="", tag=tag, args=[], examples=examples
    )


def _tool(name: str) -> McpTool:
    return McpTool(name=name, scopes=["write"])


def test_examples_block_renders_labelled_tools_call_payloads() -> None:
    op = _op_with_examples(
        "widgets",
        [
            NamedExample("basic", "Basic widget", "A minimal widget.", {"name": "s"}),
            NamedExample(
                "colored", "Colored widget", "", {"name": "s", "color": "red"}
            ),
        ],
    )
    text = "\n".join(_examples_block(_tool("create-widget"), op))
    assert "**Examples**" in text
    assert "**Basic widget**" in text
    assert "A minimal widget." in text  # description rendered
    assert "**Colored widget**" in text
    # each variant is a JSON tools/call payload naming the tool + its arguments
    assert '"name": "create-widget"' in text
    assert '"arguments": {' in text
    assert '"color": "red"' in text
    assert text.count("```json") == 2
    # no production learn-more link for the synthetic "widgets create" key
    assert "For more details" not in text


def test_examples_block_renders_single_variant() -> None:
    # No Usage block on the MCP page, so even one example gets a payload.
    op = _op_with_examples("widgets", [NamedExample("only", "Only", "", {"name": "s"})])
    text = "\n".join(_examples_block(_tool("create-widget"), op))
    assert "**Examples**" in text
    assert '"name": "create-widget"' in text
    assert text.count("```json") == 1


def test_examples_block_omitted_when_no_examples() -> None:
    assert (
        _examples_block(_tool("create-widget"), _op_with_examples("widgets", [])) == []
    )


def test_examples_block_adds_agent_connection_learn_more() -> None:
    # create-agent (tag "agents", verb "create") -> the production learn-more link
    op = _op_with_examples(
        "agents",
        [
            NamedExample("build", "Build in Calibrate", "", {"name": "a"}),
            NamedExample(
                "connect", "Connect external", "", {"name": "a", "config": {}}
            ),
        ],
    )
    text = "\n".join(_examples_block(_tool("create-agent"), op))
    assert "[Agent connections](/core-concepts/agent-connections)" in text


# --------------------------------------------------------------------------- #
# build_tab
# --------------------------------------------------------------------------- #
def test_build_tab_single_group_in_coval_order() -> None:
    # A set of written pages (order irrelevant) becomes one group in NAV_ORDER.
    tab = build_tab(
        {
            "mcp/troubleshooting",
            "mcp/overview",
            "mcp/tools",
            "mcp/installation",
            "mcp/beginners-guide",
        }
    )
    assert tab["tab"] == "MCP"
    assert len(tab["groups"]) == 1
    assert tab["groups"][0]["group"] == "MCP server"
    assert tab["groups"][0]["pages"] == [
        "mcp/overview",
        "mcp/installation",
        "mcp/tools",
        "mcp/beginners-guide",
        "mcp/troubleshooting",
    ]


def test_build_tab_skips_missing_tools_page() -> None:
    tab = build_tab({"mcp/overview", "mcp/installation"})
    assert tab["groups"][0]["pages"] == ["mcp/overview", "mcp/installation"]


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
    # overview carries the injection token; the rest are plain pages. All four
    # hand-written guide templates must exist or the generator aborts.
    (tdir / "overview.mdx").write_text(
        "# MCP overview\n\n## Available tools\n\n{/* AVAILABLE_TOOLS */}\n",
        encoding="utf-8",
    )
    (tdir / "installation.mdx").write_text("# Install\n", encoding="utf-8")
    (tdir / "beginners-guide.mdx").write_text("# Beginner's guide\n", encoding="utf-8")
    (tdir / "troubleshooting.mdx").write_text("# Troubleshooting\n", encoding="utf-8")
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


def test_generate_writes_all_five_pages(tmp_path: Path) -> None:
    output_root, _ = _run(tmp_path)
    for name in (
        "overview",
        "installation",
        "tools",
        "beginners-guide",
        "troubleshooting",
    ):
        assert (output_root / f"{name}.mdx").is_file()
    # no per-resource pages in the Coval-style structure
    assert not (output_root / "widgets.mdx").exists()


def test_generate_single_group_sidebar_order(tmp_path: Path) -> None:
    _, docs_json = _run(tmp_path)
    tabs = json.loads(docs_json.read_text())["navigation"]["tabs"]
    mcp = next(t for t in tabs if t["tab"] == "MCP")
    assert len(mcp["groups"]) == 1
    assert mcp["groups"][0]["group"] == "MCP server"
    assert mcp["groups"][0]["pages"] == [
        "mcp/overview",
        "mcp/installation",
        "mcp/tools",
        "mcp/beginners-guide",
        "mcp/troubleshooting",
    ]


def test_overview_gets_available_tools_injected(tmp_path: Path) -> None:
    output_root, _ = _run(tmp_path)
    overview = (output_root / "overview.mdx").read_text(encoding="utf-8")
    assert "{/* AVAILABLE_TOOLS */}" not in overview  # token replaced
    assert "| Category | Tools | Purpose |" in overview
    assert "across 2 categories:" in overview  # count lead-in
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
    list_section = tools[
        tools.index("### list-widgets") : tools.index("### get-widget")
    ]
    assert "**Parameters**" not in list_section
    # multi-variant create-widget grows an Examples block of tools/call payloads
    create_section = tools[tools.index("### create-widget") :]
    assert "**Examples**" in create_section
    assert "**Basic widget**" in create_section
    assert '"name": "create-widget"' in create_section
    assert create_section.count("```json") == 2


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
