"""Generate Mintlify SDK reference pages from Fern reference.md."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from docs_nav import insert_tab
from request_examples import (
    NamedExample,
    command_key,
    learn_more_markdown,
    named_request_examples,
    render_python_snippet,
)
from sdk_reference import (
    SdkMethodDoc,
    SdkRoute,
    load_route_map,
    parse_reference_file,
    routes_with_sdk_docs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SDK_TAB_NAME = "Python SDK"
DOCS_JSON = REPO_ROOT / "docs" / "docs.json"
SDK_ROOT = REPO_ROOT / "docs" / "sdk"
OVERVIEW_TEMPLATE = REPO_ROOT / "docs" / "templates" / "sdk" / "overview.mdx"
OVERVIEW_OUTPUT = SDK_ROOT / "overview.mdx"


def _frontmatter_value(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    return text


def _mdx_escape(text: str) -> str:
    return text.replace("{", "\\{")


def _split_description(description: str) -> tuple[str, str]:
    text = description.strip()
    if not text:
        return "", ""
    lines = text.splitlines()
    summary = lines[0].strip()
    body = "\n".join(lines[1:]).strip()
    return summary, body


def _render_examples_section(
    route: SdkRoute, examples: list[NamedExample]
) -> list[str]:
    """Render an `## Examples` block, one labelled snippet per named variant.

    Only emitted when the operation has ≥2 named request examples — a single
    example is already the Usage block, so repeating it adds nothing. Keeps every
    distinct request shape (build-in-Calibrate vs. connect-external) visible on
    the SDK page instead of only Fern's first-example Usage snippet.
    """
    if len(examples) < 2:
        return []
    parts = ["## Examples\n\n"]
    for example in examples:
        parts.append(f"**{_mdx_escape(example.summary)}**\n\n")
        if example.description:
            parts.append(f"{_mdx_escape(example.description)}\n\n")
        snippet = render_python_snippet(
            route.sdk_group, route.sdk_method, example.value
        )
        parts.append(f"```python\n{snippet}\n```\n\n")
    note = learn_more_markdown(command_key(route.sdk_group, route.sdk_method))
    if note:
        parts.append(f"{note}\n\n")
    return parts


def render_method_page(
    route: SdkRoute,
    doc: SdkMethodDoc,
    examples: list[NamedExample] | None = None,
) -> str:
    api_page = _mdx_escape(route.mintlify_api_page)
    api_callout = (
        f"See **{api_page}** in the "
        f"[API reference](/api-reference/introduction) tab."
    )
    summary, body = _split_description(doc.description)
    if not summary:
        summary = f"client.{route.sdk_group}.{route.sdk_method}()"
    summary_escaped = _mdx_escape(summary)
    body_escaped = _mdx_escape(body) if body else ""
    title_escaped = _frontmatter_value(route.title)
    description_escaped = _frontmatter_value(summary_escaped[:160])
    parts = [
        "---\n",
        f'title: "{title_escaped}"\n',
        f'description: "{description_escaped}"\n',
        "---\n\n",
        "{/* Generated from Fern reference.md — do not edit directly. */}\n\n",
    ]
    if body_escaped:
        parts.append(f"{body_escaped}\n\n")
    parts.extend(
        [
            "## Usage\n\n",
            f"```python\n{doc.usage_code.rstrip()}\n```\n\n",
        ]
    )
    parts.extend(_render_examples_section(route, examples or []))
    parts.extend(
        [
            "## API endpoint\n\n",
            f"{api_callout}\n",
        ]
    )
    return "".join(parts)


def _examples_for_route(
    route: SdkRoute, openapi: dict[str, Any]
) -> list[NamedExample]:
    op = openapi.get("paths", {}).get(route.path, {}).get(route.http.lower(), {})
    return named_request_examples(op) if isinstance(op, dict) else []


def write_sdk_pages(
    paired: list[tuple[SdkRoute, SdkMethodDoc]],
    openapi: dict[str, Any],
) -> list[str]:
    written: list[str] = []
    for route, doc in paired:
        out = SDK_ROOT / route.sdk_group.replace("_", "-") / f"{route.sdk_method}.mdx"
        out.parent.mkdir(parents=True, exist_ok=True)
        examples = _examples_for_route(route, openapi)
        out.write_text(
            render_method_page(route, doc, examples), encoding="utf-8"
        )
        written.append(route.doc_slug)
    return written


def copy_overview() -> None:
    OVERVIEW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OVERVIEW_OUTPUT.write_text(OVERVIEW_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")


def _api_reference_groups(routes: list[SdkRoute]) -> list[dict[str, Any]]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for route in routes:
        page = route.mintlify_api_page
        if page not in by_group[route.api_group]:
            by_group[route.api_group].append(page)
    groups: list[dict[str, Any]] = [
        {"group": "Getting started", "pages": ["api-reference/introduction"]},
    ]
    for group_name in sorted(by_group):
        groups.append(
            {
                "group": group_name,
                "openapi": "api-reference/openapi.json",
                "pages": by_group[group_name],
            }
        )
    return groups


def _sdk_nav_groups(paired: list[tuple[SdkRoute, SdkMethodDoc]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for route, _ in paired:
        slug = route.doc_slug
        label = route.api_group
        if slug not in by_group[label]:
            by_group[label].append(slug)
    groups: list[dict[str, Any]] = [
        {"group": "Getting started", "pages": ["sdk/overview"]},
    ]
    for group_name in sorted(by_group):
        groups.append({"group": group_name, "pages": by_group[group_name]})
    return groups


def update_docs_json(
    routes: list[SdkRoute],
    paired: list[tuple[SdkRoute, SdkMethodDoc]],
) -> None:
    docs = json.loads(DOCS_JSON.read_text(encoding="utf-8"))
    tabs = docs["navigation"]["tabs"]

    api_tab = next(t for t in tabs if t.get("tab") == "API reference")
    # Only explicit groups — tab-level openapi would also auto-generate tag groups
    # (agents, tests, …) and duplicate our curated Agents/Tests/Agent tests sections.
    api_tab.pop("openapi", None)
    api_tab["groups"] = _api_reference_groups(routes)

    sdk_tab = {
        "tab": SDK_TAB_NAME,
        "groups": _sdk_nav_groups(paired),
    }
    # Place the SDK tab after the cloud "CLI" tab when it exists, else after
    # "Home" (the CLI tab is inserted later in the run). Drop any prior SDK tab
    # (including the legacy "SDK" name) so re-runs replace rather than duplicate.
    docs["navigation"]["tabs"] = insert_tab(
        tabs, sdk_tab, after=("CLI", "Home"), remove={SDK_TAB_NAME, "SDK"}
    )

    examples = docs.setdefault("api", {}).setdefault("examples", {})
    examples["languages"] = ["curl", "python"]
    examples["defaults"] = "required"
    examples["autogenerate"] = True

    DOCS_JSON.write_text(json.dumps(docs, indent=2) + "\n", encoding="utf-8")


def generate_sdk_docs(reference_path: Path, openapi: dict[str, Any]) -> list[str]:
    routes = load_route_map(openapi=openapi)
    methods = parse_reference_file(reference_path)
    paired = routes_with_sdk_docs(routes, methods)
    copy_overview()
    written = write_sdk_pages(paired, openapi)
    update_docs_json(routes, paired)
    return ["sdk/overview", *written]


def prune_stale_sdk_pages(active_slugs: set[str]) -> None:
    if not SDK_ROOT.is_dir():
        return
    for path in SDK_ROOT.rglob("*.mdx"):
        rel_slug = path.relative_to(REPO_ROOT / "docs").with_suffix("").as_posix()
        if rel_slug == "sdk/overview":
            continue
        if rel_slug not in active_slugs:
            path.unlink()
    for directory in sorted(SDK_ROOT.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


if __name__ == "__main__":
    import sys

    ref = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if ref is None:
        raise SystemExit("usage: generate_sdk_docs.py <path/to/reference.md>")
    from sdk_reference import DEFAULT_OPENAPI

    spec = json.loads(DEFAULT_OPENAPI.read_text(encoding="utf-8"))
    slugs = generate_sdk_docs(ref, spec)
    prune_stale_sdk_pages(set(slugs))
    print(f"Wrote {len(slugs)} SDK pages")
