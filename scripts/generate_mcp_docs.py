#!/usr/bin/env python3
"""Generate Mintlify MDX pages for the Speakeasy-generated MCP server.

Reads the tool definitions the ``calibrate-mcp`` repo ships under
``src/mcp-server/tools/*.ts`` and produces one page per API resource (``agents``,
``agent-tests``, ``tests``), each tool rendered as a ``##`` section — description,
a scopes / read-only line, an arguments table, and a cross-link to the matching
operation in the API reference tab.

Like the SDK and CLI generators, the structure is single-sourced, not
hand-maintained: each tool links to an OpenAPI operation via its request-model
import (see ``mcp_reference.operation_key``), and the operation's tag drives the
resource grouping (``api_group_from_tag`` — shared with the SDK/CLI generators,
so ``agent-tests`` -> "Agent tests" everywhere) while its parameters and request
body drive the arguments table. The Getting-started overview is a hand-written
template under ``docs/templates/mcp/``.

Run by the sync workflow via ``fetch_public_openapi.py`` when ``MCP_DOCS_PATH``
points at a ``calibrate-mcp`` checkout. Edit the tool definitions (in
``calibrate-mcp``) or the overview template — never the generated pages.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docs_mdx import escape_mdx_prose, frontmatter_value, table_cell  # noqa: E402
from docs_nav import insert_tab  # noqa: E402
from mcp_reference import McpTool, operation_key, parse_tools_dir  # noqa: E402
from sdk_reference import api_group_from_tag  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
DOCS_JSON = DOCS_ROOT / "docs.json"
DEFAULT_OPENAPI = DOCS_ROOT / "api-reference" / "openapi.json"
OUTPUT_ROOT = DOCS_ROOT / "mcp"
TEMPLATE_DIR = REPO_ROOT / "docs" / "templates" / "mcp"

TAB_NAME = "MCP"
# Place the MCP tab after "Python SDK" when present, else the cloud CLI, else Home.
INSERT_AFTER = ("Python SDK", "CLI", "Home")
OVERVIEW_SLUG = "mcp/overview"

# Hand-written conceptual pages copied verbatim into the Getting-started group.
GETTING_STARTED = ["overview"]

GENERATED_BANNER = (
    "{/* Generated from calibrate-mcp/src/mcp-server/tools by "
    "scripts/generate_mcp_docs.py — do not edit directly. */}"
)


@dataclass(frozen=True)
class Operation:
    path: str
    method: str
    summary: str
    tag: str
    args: list["Arg"]

    @property
    def api_page(self) -> str:
        return f"{self.method.upper()} {self.path}"


@dataclass(frozen=True)
class Arg:
    name: str
    type: str
    required: bool
    description: str


def _schema_type(schema: dict[str, Any]) -> str:
    """A short human type label for an OpenAPI (sub)schema."""
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    for combiner in ("anyOf", "oneOf"):
        options = [s for s in schema.get(combiner, []) if s.get("type") != "null"]
        if options:
            return _schema_type(options[0])
    t = schema.get("type")
    if t == "array":
        item = schema.get("items") or {}
        inner = _schema_type(item) if item else ""
        return f"array of {inner}" if inner else "array"
    return t or "object"


def _is_optional_schema(schema: dict[str, Any]) -> bool:
    """True when a property schema explicitly allows ``null`` (anyOf [..., null])."""
    for combiner in ("anyOf", "oneOf"):
        if any(s.get("type") == "null" for s in schema.get(combiner, [])):
            return True
    return False


def _resolve(schema: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    """Resolve a schema to its effective object shape.

    Unwraps a top-level ``$ref`` and drops the ``null`` arm of an
    ``anyOf``/``oneOf`` (how an optional request body is expressed), so the
    caller sees the underlying object's ``properties``/``required``.
    """
    seen: set[str] = set()
    while isinstance(schema, dict):
        ref = schema.get("$ref")
        if ref:
            name = ref.rsplit("/", 1)[-1]
            if name in seen:
                return {}
            seen.add(name)
            schema = components.get(name, {})
            continue
        for combiner in ("anyOf", "oneOf"):
            options = [s for s in schema.get(combiner, []) if s.get("type") != "null"]
            if options:
                schema = options[0]
                break
        else:
            break
    return schema


def _body_args(request_body: dict[str, Any], components: dict[str, Any]) -> list[Arg]:
    content = (request_body or {}).get("content", {})
    media = content.get("application/json") or next(iter(content.values()), {})
    schema = _resolve(media.get("schema", {}), components)
    required = set(schema.get("required", []))
    args: list[Arg] = []
    for name, prop in (schema.get("properties") or {}).items():
        args.append(
            Arg(
                name=name,
                type=_schema_type(prop),
                required=name in required and not _is_optional_schema(prop),
                description=(prop.get("description") or "").strip(),
            )
        )
    return args


def _param_args(parameters: list[dict[str, Any]]) -> list[Arg]:
    args: list[Arg] = []
    for p in parameters or []:
        if not isinstance(p, dict) or p.get("in") == "header":
            continue
        schema = p.get("schema") or {}
        args.append(
            Arg(
                name=p.get("name", ""),
                type=_schema_type(schema),
                required=bool(p.get("required")),
                description=(p.get("description") or schema.get("description") or "").strip(),
            )
        )
    return args


def build_operation_map(openapi: dict[str, Any]) -> dict[str, Operation]:
    """Map ``operation_key(operationId)`` -> :class:`Operation` for every op."""
    components = openapi.get("components", {}).get("schemas", {})
    ops: dict[str, Operation] = {}
    for path, methods in openapi.get("paths", {}).items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if not isinstance(op, dict) or "operationId" not in op:
                continue
            tags = op.get("tags") or []
            args = _param_args(op.get("parameters", [])) + _body_args(
                op.get("requestBody", {}), components
            )
            ops[operation_key(op["operationId"])] = Operation(
                path=path,
                method=method,
                summary=(op.get("summary") or "").strip(),
                tag=tags[0] if tags else "",
                args=args,
            )
    return ops


def _summary_line(text: str) -> str:
    """First non-blank line of a (possibly multi-line) description.

    OpenAPI property descriptions can be long multi-paragraph markdown with code
    fences (e.g. the agent ``config`` object). A markdown table cell can hold
    none of that, so the table shows the opening line and the API-reference
    cross-link carries the full schema.
    """
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _args_table(args: list[Arg]) -> str:
    if not args:
        return ""
    lines = [
        "| Argument | Type | Required | Description |",
        "| --- | --- | --- | --- |",
    ]
    for a in args:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{a.name}`",
                    a.type or "—",
                    "Yes" if a.required else "No",
                    table_cell(_summary_line(a.description)) or "—",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _tool_section(tool: McpTool, op: Operation) -> list[str]:
    lines = [f"## {tool.name}", ""]
    if tool.description:
        lines += [escape_mdx_prose(tool.description), ""]

    meta: list[str] = []
    if tool.scopes:
        meta.append("**Scope:** " + ", ".join(f"`{s}`" for s in tool.scopes))
    meta.append("**Access:** " + ("Read-only" if tool.read_only else "Write"))
    lines += [" · ".join(meta), ""]

    table = _args_table(op.args)
    if table:
        lines += ["**Arguments**", "", table, ""]

    api_page = escape_mdx_prose(op.api_page)
    lines += [
        f"See **{api_page}** in the "
        f"[API reference](/api-reference/introduction) tab for the full schema.",
        "",
    ]
    return lines


def _frontmatter(title: str, description: str) -> list[str]:
    return [
        "---",
        f"title: {frontmatter_value(title)}",
        f"description: {frontmatter_value(description)}",
        "---",
        "",
        GENERATED_BANNER,
        "",
    ]


def render_resource_page(title: str, tools: list[tuple[McpTool, Operation]]) -> str:
    description = f"MCP tools for {title.lower()}."
    lines = _frontmatter(title, description)
    for tool, op in tools:
        lines += _tool_section(tool, op)
    return "\n".join(lines).rstrip("\n") + "\n"


def _copy_getting_started(output_root: Path, template_dir: Path) -> list[str]:
    output_root.mkdir(parents=True, exist_ok=True)
    page_ids: list[str] = []
    for name in GETTING_STARTED:
        template = template_dir / f"{name}.mdx"
        if not template.is_file():
            raise SystemExit(f"Getting-started template missing: {template}")
        (output_root / f"{name}.mdx").write_text(
            template.read_text(encoding="utf-8"), encoding="utf-8"
        )
        page_ids.append(f"mcp/{name}")
    return page_ids


def _prune_stale(output_root: Path, keep_page_ids: set[str]) -> list[Path]:
    docs_root = output_root.parent
    removed: list[Path] = []
    for mdx in output_root.rglob("*.mdx"):
        page_id = str(mdx.relative_to(docs_root).with_suffix("")).replace("\\", "/")
        if page_id not in keep_page_ids:
            mdx.unlink()
            removed.append(mdx)
    return removed


def build_tab(getting_started: list[str], resources: list[str]) -> dict:
    groups: list[dict] = [{"group": "Getting started", "pages": getting_started}]
    if resources:
        groups.append({"group": "Tools", "pages": resources})
    return {"tab": TAB_NAME, "groups": groups}


def patch_docs_json(docs_json: Path, tab: dict) -> None:
    data = json.loads(docs_json.read_text(encoding="utf-8"))
    data["navigation"]["tabs"] = insert_tab(
        data["navigation"]["tabs"], tab, after=INSERT_AFTER
    )
    docs_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _fail_unmatched_tools(tools: list[McpTool]) -> None:
    """Abort when a tool matched no OpenAPI operation — args/cross-link would break.

    This runs unattended on a schedule, so instead of silently shipping a tool
    page with no arguments or API link, emit a GitHub Actions error annotation and
    raise — failing the sync so a human resolves the mismatch (a renamed operation
    or a stale OpenAPI spec) before anything is published.
    """
    names = ", ".join(t.name for t in tools)
    print(f"::error title=MCP docs::MCP tool(s) matched no OpenAPI operation: {names}")
    raise SystemExit(
        f"MCP docs generation aborted: {len(tools)} tool(s) matched no OpenAPI "
        f"operation: {names}. The OpenAPI spec may be stale, or an operationId "
        f"was renamed. Refresh docs/api-reference/openapi.json and re-run."
    )


def generate_mcp_docs(
    tools_dir: Path,
    *,
    docs_json: Path = DOCS_JSON,
    output_root: Path = OUTPUT_ROOT,
    template_dir: Path = TEMPLATE_DIR,
    openapi_path: Path = DEFAULT_OPENAPI,
    openapi: dict[str, Any] | None = None,
) -> None:
    tools_dir = Path(tools_dir)
    if not tools_dir.is_dir():
        raise SystemExit(f"MCP tools directory not found: {tools_dir}")

    spec = openapi
    if spec is None:
        if not openapi_path.is_file():
            raise SystemExit(f"OpenAPI spec not found: {openapi_path}")
        spec = json.loads(openapi_path.read_text(encoding="utf-8"))

    op_map = build_operation_map(spec)
    tools = parse_tools_dir(tools_dir)

    # Every tool must resolve to an operation, else its args/cross-link are empty.
    # Fail loudly BEFORE writing anything so the docs are left untouched.
    unmatched = [t for t in tools if t.operation_ref not in op_map]
    if unmatched:
        _fail_unmatched_tools(unmatched)

    by_tag: dict[str, list[tuple[McpTool, Operation]]] = defaultdict(list)
    for tool in tools:
        op = op_map[tool.operation_ref]
        by_tag[op.tag].append((tool, op))

    docs_root = output_root.parent
    resource_pages: list[str] = []
    for tag in sorted(by_tag, key=api_group_from_tag):
        title = api_group_from_tag(tag)
        pairs = sorted(by_tag[tag], key=lambda pair: pair[0].name)
        slug = f"mcp/{tag}"
        out_path = docs_root / f"{slug}.mdx"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_resource_page(title, pairs), encoding="utf-8")
        resource_pages.append(slug)

    getting_started = _copy_getting_started(output_root, template_dir)
    tab = build_tab(getting_started, resource_pages)
    removed = _prune_stale(output_root, {*getting_started, *resource_pages})
    patch_docs_json(docs_json, tab)

    print(f"Generated {len(resource_pages)} MCP resource page(s) under {output_root}")
    if removed:
        print(f"Pruned {len(removed)} stale page(s)")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        raise SystemExit("usage: generate_mcp_docs.py <calibrate-mcp/src/mcp-server/tools>")
    generate_mcp_docs(Path(argv[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
