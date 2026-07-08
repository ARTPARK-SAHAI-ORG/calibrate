"""Tests for scripts/mcp_reference.py (Speakeasy MCP tool parser)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp_reference import (  # noqa: E402
    McpTool,
    operation_key,
    parse_tool_ts,
    parse_tools_dir,
)
from mcp_tool_samples import (  # noqa: E402
    CREATE_WIDGET_TS,
    GET_WIDGET_TS,
    HELPER_TS,
    LIST_WIDGETS_TS,
    write_samples,
)


def test_operation_key_matches_speakeasy_model_stem() -> None:
    # Speakeasy strips non-alphanumerics from the operationId and appends "op".
    assert (
        operation_key("resolve_agent_names_agents_resolve_post")
        == "resolveagentnamesagentsresolvepostop"
    )
    assert (
        operation_key("get_agent_endpoint_agents__agent_uuid__get")
        == "getagentendpointagentsagentuuidgetop"
    )


def test_parse_read_only_tool_with_path_param() -> None:
    tool = parse_tool_ts(GET_WIDGET_TS)
    assert isinstance(tool, McpTool)
    assert tool.name == "get-widget"
    assert tool.scopes == ["read"]
    assert tool.read_only is True
    assert tool.destructive is False
    assert tool.operation_ref == "getwidgetwidgetswidgetuuidgetop"
    assert tool.read_write_label == "Read-only"


def test_parse_write_tool_multiline_description() -> None:
    tool = parse_tool_ts(CREATE_WIDGET_TS)
    assert tool.name == "create-widget"
    assert tool.scopes == ["write"]
    assert tool.read_only is False
    assert tool.read_write_label == "Write"
    # The template literal is captured whole (newlines preserved) and stripped.
    assert tool.description.startswith("Create a new widget")
    assert "\n" in tool.description
    assert tool.operation_ref == "createwidgetwidgetspostop"


def test_parse_tool_with_no_args_still_has_operation_ref() -> None:
    tool = parse_tool_ts(LIST_WIDGETS_TS)
    assert tool.name == "list-widgets"
    assert tool.operation_ref == "listwidgetswidgetsgetop"


def test_helper_file_without_name_is_not_a_tool() -> None:
    assert parse_tool_ts(HELPER_TS) is None


def test_parse_tools_dir_skips_non_tools_and_sorts(tmp_path: Path) -> None:
    tools_dir = write_samples(tmp_path / "tools")
    tools = parse_tools_dir(tools_dir)
    names = [t.name for t in tools]
    # helpers.ts is skipped; the four real tools remain, sorted by file name
    # (gadgetsRun, widgetsCreate, widgetsGet, widgetsList).
    assert names == ["run-gadgets", "create-widget", "get-widget", "list-widgets"]
