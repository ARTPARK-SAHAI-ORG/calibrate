#!/usr/bin/env python3
"""Parse Speakeasy-generated MCP tool definitions (``src/mcp-server/tools/*.ts``).

Speakeasy emits one ``.ts`` per MCP tool with a fixed, machine-generated shape::

    import { agentsResolve } from "../../funcs/agentsResolve.js";
    import { ResolveAgentNamesAgentsResolvePostRequest$zodSchema }
      from "../../models/resolveagentnamesagentsresolvepostop.js";
    import { formatResult, ToolDefinition } from "../tools.js";

    const args = {
      request: ResolveAgentNamesAgentsResolvePostRequest$zodSchema,
    };

    export const tool$agentsResolve: ToolDefinition<typeof args> = {
      name: "resolve-agent-names",
      description:
        `Map human-friendly agent names to UUIDs ...
    `,
      scopes: ["read"],
      annotations: {
        "title": "Resolve agent names to UUIDs",
        "destructiveHint": false,
        "readOnlyHint": true,
        ...
      },
      ...
    };

This module turns that text into an :class:`McpTool`. It performs no I/O beyond
reading the tool files. The tool's ``models/<stem>.js`` import is the link back
to the OpenAPI operation: ``<stem>`` equals the operation's ``operationId`` with
every non-alphanumeric character stripped, plus a trailing ``op`` — see
:func:`operation_key`. The generator uses that to pull each tool's arguments and
API cross-link from the OpenAPI spec (single-sourced, like the SDK/CLI docs).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ``export const tool$foo`` / ``models/<stem>.js`` — the only two hooks we need.
_MODEL_IMPORT = re.compile(r'from\s+"\.\./\.\./models/([a-z0-9]+)\.js"')
_NAME = re.compile(r'^\s*name:\s*"([^"]+)"', re.MULTILINE)
_SCOPES = re.compile(r"^\s*scopes:\s*\[([^\]]*)\]", re.MULTILINE)
_READ_ONLY = re.compile(r'"?readOnlyHint"?:\s*(true|false)')
_DESTRUCTIVE = re.compile(r'"?destructiveHint"?:\s*(true|false)')
# ``description:`` followed by a backtick template literal (possibly on the next
# line). Non-greedy up to the closing backtick; template literals never contain a
# bare backtick in this generated code.
_DESCRIPTION = re.compile(r"description:\s*`(.*?)`", re.DOTALL)


@dataclass
class McpTool:
    name: str                       # kebab tool name, e.g. "resolve-agent-names"
    description: str = ""
    scopes: list[str] = field(default_factory=list)
    read_only: bool = False
    destructive: bool = False
    operation_ref: str = ""         # models/<stem> — links to an OpenAPI op

    @property
    def read_write_label(self) -> str:
        return "Read-only" if self.read_only else "Write"


def operation_key(operation_id: str) -> str:
    """Normalize an OpenAPI ``operationId`` to the tool's model-import stem.

    Speakeasy names the request model file after the operation id with all
    non-alphanumerics removed and an ``op`` suffix, e.g.
    ``resolve_agent_names_agents_resolve_post`` ->
    ``resolveagentnamesagentsresolvepostop``. Applying the same transform to an
    OpenAPI op's id lets the generator match tools to operations deterministically.
    """
    return re.sub(r"[^a-z0-9]", "", operation_id.lower()) + "op"


def _parse_scopes(raw: str) -> list[str]:
    return [s.strip().strip('"').strip("'") for s in raw.split(",") if s.strip()]


def parse_tool_ts(text: str) -> McpTool | None:
    """Parse one Speakeasy tool ``.ts`` file into an :class:`McpTool` (or ``None``).

    Returns ``None`` when the file has no ``name:`` field — i.e. it is not a tool
    definition (shared helpers, the registry, etc.).
    """
    text = text.replace("\r\n", "\n")
    name_match = _NAME.search(text)
    if not name_match:
        return None

    model_match = _MODEL_IMPORT.search(text)
    desc_match = _DESCRIPTION.search(text)
    scopes_match = _SCOPES.search(text)
    read_only_match = _READ_ONLY.search(text)
    destructive_match = _DESTRUCTIVE.search(text)

    return McpTool(
        name=name_match.group(1),
        description=desc_match.group(1).strip() if desc_match else "",
        scopes=_parse_scopes(scopes_match.group(1)) if scopes_match else [],
        read_only=bool(read_only_match) and read_only_match.group(1) == "true",
        destructive=bool(destructive_match) and destructive_match.group(1) == "true",
        operation_ref=model_match.group(1) if model_match else "",
    )


def parse_tools_dir(tools_dir: Path) -> list[McpTool]:
    """Parse every ``*.ts`` tool file under ``tools_dir``, sorted by file name."""
    tools: list[McpTool] = []
    for path in sorted(Path(tools_dir).glob("*.ts")):
        tool = parse_tool_ts(path.read_text(encoding="utf-8"))
        if tool is not None:
            tools.append(tool)
    return tools
