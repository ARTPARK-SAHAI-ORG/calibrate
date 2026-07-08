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
    _args_table,
    _schema_type,
    _summary_line,
    build_operation_map,
    build_tab,
    generate_mcp_docs,
    patch_docs_json,
)
from mcp_reference import parse_tool_ts  # noqa: E402
from mcp_tool_samples import (  # noqa: E402
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


def test_args_table_marks_required_and_escapes_pipe() -> None:
    table = _args_table(
        [
            Arg(name="name", type="string", required=True, description="A | B"),
            Arg(name="color", type="string", required=False, description=""),
        ]
    )
    assert "| `name` | string | Yes | A \\| B |" in table
    assert "| `color` | string | No | — |" in table


def test_args_table_empty_when_no_args() -> None:
    assert _args_table([]) == ""


# --------------------------------------------------------------------------- #
# build_operation_map
# --------------------------------------------------------------------------- #
def test_operation_map_keys_and_args() -> None:
    ops = build_operation_map(OPENAPI)
    # keyed by operation_key(operationId)
    assert "createwidgetwidgetspostop" in ops
    assert "rungadgetsgadgetsrunpostop" in ops

    create = ops["createwidgetwidgetspostop"]
    assert create.tag == "widgets"
    assert create.api_page == "POST /widgets"
    arg_names = {a.name: a for a in create.args}
    assert arg_names["name"].required is True
    # color is anyOf [string, null] -> optional, typed string
    assert arg_names["color"].required is False
    assert arg_names["color"].type == "string"


def test_operation_map_drops_header_params() -> None:
    ops = build_operation_map(OPENAPI)
    get = ops["getwidgetwidgetswidgetuuidgetop"]
    names = [a.name for a in get.args]
    assert names == ["widget_uuid"]  # X-API-Key header dropped


def test_operation_map_unwraps_optional_body_anyof() -> None:
    ops = build_operation_map(OPENAPI)
    run = ops["rungadgetsgadgetsrunpostop"]
    # body is anyOf [GadgetBatch, null]; its properties still surface
    names = [a.name for a in run.args]
    assert names == ["gadget_names"]
    assert run.args[0].required is False
    assert run.args[0].type == "array of string"


# --------------------------------------------------------------------------- #
# build_tab
# --------------------------------------------------------------------------- #
def test_build_tab_shape() -> None:
    tab = build_tab(["mcp/overview"], ["mcp/widgets", "mcp/gadgets"])
    assert tab["tab"] == "MCP"
    assert tab["groups"][0] == {"group": "Getting started", "pages": ["mcp/overview"]}
    assert tab["groups"][1] == {
        "group": "Tools",
        "pages": ["mcp/widgets", "mcp/gadgets"],
    }


def test_build_tab_omits_empty_tools_group() -> None:
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
    (tdir / "overview.mdx").write_text("# MCP overview\n", encoding="utf-8")
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


def test_generate_writes_one_page_per_tag(tmp_path: Path) -> None:
    output_root, _ = _run(tmp_path)
    assert (output_root / "overview.mdx").is_file()
    assert (output_root / "widgets.mdx").is_file()
    assert (output_root / "gadgets.mdx").is_file()


def test_generate_page_content(tmp_path: Path) -> None:
    output_root, _ = _run(tmp_path)
    widgets = (output_root / "widgets.mdx").read_text(encoding="utf-8")
    # frontmatter + banner
    assert 'title: "Widgets"' in widgets
    assert "do not edit directly" in widgets
    # one section per tool, sorted by name
    assert widgets.index("## create-widget") < widgets.index("## get-widget")
    assert widgets.index("## get-widget") < widgets.index("## list-widgets")
    # scope + access line
    assert "**Scope:** `write` · **Access:** Write" in widgets
    assert "**Access:** Read-only" in widgets
    # arg table with required marker; multi-line desc collapsed to first line
    assert "| `name` | string | Yes | Widget name. |" in widgets
    # API cross-link, with MDX-escaped path braces
    assert "GET /widgets/&#123;widget_uuid&#125;" in widgets
    # no-arg tool omits the Arguments block
    list_section = widgets[widgets.index("## list-widgets"):]
    assert "**Arguments**" not in list_section


def test_generate_inserts_tab_after_python_sdk(tmp_path: Path) -> None:
    _, docs_json = _run(tmp_path)
    tabs = json.loads(docs_json.read_text())["navigation"]["tabs"]
    names = [t["tab"] for t in tabs]
    assert names == ["Home", "Python SDK", "MCP", "Integrations"]


def test_generate_is_idempotent(tmp_path: Path) -> None:
    output_root, docs_json = _run(tmp_path)
    first = (output_root / "widgets.mdx").read_text()
    tabs_first = docs_json.read_text()
    # regenerate over the same tree
    generate_mcp_docs(
        write_samples(tmp_path / "tools"),
        docs_json=docs_json,
        output_root=output_root,
        template_dir=_template_dir(tmp_path),
        openapi=OPENAPI,
    )
    assert (output_root / "widgets.mdx").read_text() == first
    names = [t["tab"] for t in json.loads(docs_json.read_text())["navigation"]["tabs"]]
    assert names.count("MCP") == 1  # no duplicate tab
    assert docs_json.read_text() == tabs_first


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
    # A tool whose operation is missing from the spec must abort the run and
    # leave the docs untouched (no pages, no docs.json edit).
    samples = {"orphan.ts": ORPHAN_TS}
    docs_json = _docs_json(tmp_path)
    output_root = tmp_path / "docs" / "mcp"
    with pytest.raises(SystemExit, match="matched no OpenAPI operation"):
        generate_mcp_docs(
            write_samples(tmp_path / "tools", samples),
            docs_json=docs_json,
            output_root=output_root,
            template_dir=_template_dir(tmp_path),
            openapi=OPENAPI,
        )
    assert not output_root.exists()
    assert "MCP" not in docs_json.read_text()
