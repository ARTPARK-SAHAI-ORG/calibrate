"""Shared MDX-escaping helpers for the doc generators.

Mintlify pages are MDX, so ``<``/``>``/``{``/``}`` in prose and table cells must
be escaped or they parse as JSX. Both the CLI generator (``generate_cli_docs.py``)
and the MCP generator (``generate_mcp_docs.py``) render tables and frontmatter
from upstream text, so the escaping lives here once rather than drifting between
the two.
"""

from __future__ import annotations

import re


def escape_mdx_prose(text: str) -> str:
    """Escape MDX-significant characters in prose, leaving inline code spans alone."""
    out: list[str] = []
    for i, part in enumerate(re.split(r"(`[^`]*`)", text)):
        if i % 2 == 1:
            out.append(part)
            continue
        part = (
            part.replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("{", "&#123;")
            .replace("}", "&#125;")
        )
        out.append(part)
    return "".join(out)


def table_cell(text: str) -> str:
    """Escape a value for use inside a markdown table cell."""
    return escape_mdx_prose(text).replace("|", "\\|").strip()


def frontmatter_value(text: str) -> str:
    """Quote a value for a YAML frontmatter field."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
