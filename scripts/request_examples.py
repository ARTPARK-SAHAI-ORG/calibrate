"""Render a route's *named* request-body examples as SDK / CLI snippets.

The public OpenAPI spec carries one or more named request examples per operation
(the backend declares them via FastAPI ``Body(openapi_examples=…)``). Fern and
Speakeasy each feature only the *first* one, so an operation with genuinely
distinct variants — e.g. building an agent inside Calibrate vs. connecting an
external HTTP agent — loses that distinction on the generated SDK and CLI pages.

These helpers re-render *every* named example straight from the same spec, so the
SDK/CLI pages can show all variants with the OpenAPI spec as the single source of
truth (no example body is hand-copied into the docs repo). They are intentionally
generic: any operation that grows a second named example gets the treatment for
free.

Pure functions, no I/O — the generators own reading the spec and writing pages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NamedExample:
    """One entry of an operation's request-body ``examples`` map."""

    key: str          # the map key, e.g. "openai_compatible_connection"
    summary: str      # human label; falls back to the key
    description: str  # optional prose shown above the snippet
    value: Any        # the example request body


def named_request_examples(op: dict[str, Any]) -> list[NamedExample]:
    """Extract the JSON request body's named ``examples`` from an operation.

    Returns them in spec order. Empty when the op has no request body, no
    ``examples`` map, or only entries missing a ``value`` — callers use the
    count to decide whether a multi-variant section is worth rendering.
    """
    content = (op.get("requestBody") or {}).get("content") or {}
    body = content.get("application/json") or {}
    examples = body.get("examples") or {}
    result: list[NamedExample] = []
    for key, entry in examples.items():
        if not isinstance(entry, dict) or "value" not in entry:
            continue
        result.append(
            NamedExample(
                key=key,
                summary=(entry.get("summary") or key).strip(),
                description=(entry.get("description") or "").strip(),
                value=entry["value"],
            )
        )
    return result


def _py_literal(value: Any, indent: int = 0) -> str:
    """Format a JSON-decoded value as a Python literal (double-quoted, indented).

    Mirrors the style of Fern's own usage snippets (``True``/``None``,
    double-quoted keys) so the generated Examples section reads identically to
    the Usage block above it.
    """
    pad = "    " * indent
    child = "    " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = list(value.items())
        lines = ["{"]
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            lines.append(f"{child}{json.dumps(k)}: {_py_literal(v, indent + 1)}{comma}")
        lines.append(pad + "}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return "[]"
        lines = ["["]
        for i, item in enumerate(value):
            comma = "," if i < len(value) - 1 else ""
            lines.append(f"{child}{_py_literal(item, indent + 1)}{comma}")
        lines.append(pad + "]")
        return "\n".join(lines)
    # bool must precede the numeric fallthrough — bool is a subclass of int.
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return json.dumps(value)  # str -> double-quoted; int/float -> bare


def render_python_snippet(sdk_group: str, sdk_method: str, value: Any) -> str:
    """Render one example body as a ``client.<group>.<method>(…)`` call.

    A top-level object becomes keyword arguments (one per line); anything else is
    passed as a single positional argument.
    """
    call = f"client.{sdk_group}.{sdk_method}"
    if not isinstance(value, dict):
        return f"{call}({_py_literal(value)})"
    if not value:
        return f"{call}()"
    lines = [f"{call}("]
    for key, val in value.items():
        lines.append(f"    {key}={_py_literal(val, 1)},")
    lines.append(")")
    return "\n".join(lines)


def _cli_value(value: Any) -> str:
    """Format one flag value for a shell command.

    Objects/arrays are emitted as single-quoted compact JSON (the shape the
    Speakeasy CLI expects for a JSON-string flag); scalars are shell-quoted.
    """
    if isinstance(value, (dict, list)):
        return "'" + json.dumps(value, separators=(",", ":")) + "'"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return '""'
    if isinstance(value, str):
        return json.dumps(value)  # double-quoted, escaped
    return json.dumps(value)


def _flag_for(field: str, options: list[Any]) -> str:
    """Resolve a body field name to its CLI flag using the command's options.

    Prefers an exact ``--<field>`` long flag, then a long flag that starts with
    or contains the field name (Speakeasy renames e.g. ``config`` to
    ``--config-param``). Falls back to ``--<field>`` when nothing matches.
    """
    def long_flags(opt: Any) -> list[str]:
        return [
            token.strip()
            for token in opt.flag.split(",")
            if token.strip().startswith("--")
        ]

    exact = f"--{field}"
    for opt in options:
        if exact in long_flags(opt):
            return exact
    for opt in options:
        for lf in long_flags(opt):
            if lf[2:].startswith(field):
                return lf
    for opt in options:
        for lf in long_flags(opt):
            if field in lf[2:]:
                return lf
    return exact


def render_cli_snippet(command: str, options: list[Any], value: Any) -> str:
    """Render one example body as a ``calibrate …`` invocation.

    ``command`` is the full command (``calibrate agents create``); ``options``
    are the command's parsed flags (used to map body fields to flag names).
    Non-object bodies are skipped by the caller, so this expects a dict.
    """
    parts = [command]
    for field, val in value.items():
        parts.append(f"{_flag_for(field, options)} {_cli_value(val)}")
    return " \\\n  ".join(parts)
