#!/usr/bin/env python3
"""Generate Mintlify MDX pages for the cloud ``calibrate`` CLI.

Reads the Cobra-generated markdown that ``calibrate-cli`` ships under its
``docs/`` folder and produces one **Resources** page per command (``agents``,
``agent-tests``, ``auth``, …), each subcommand a ``##`` section with usage, a
clean ``Option | Type | Default | Description`` table, and examples. Hand-written
conceptual pages (overview, agent mode) live under **Getting started**.

The sidebar is derived from the CLI source itself — no hand-maintained map
(mirroring how the SDK docs derive structure from the backend OpenAPI/Fern
overrides): resource titles come from ``api_group_from_tag`` (shared with the SDK
generator, ``agent-tests`` -> "Agent tests"); resource and subcommand order
follow each command's Cobra "SEE ALSO" list.

Pages are kept minimal (Coval-style): no per-page global-flags callout and no
noise rows in option tables — global flags are documented once under Getting
started, and placeholder-only flags are dropped.

Run by the sync workflow via ``fetch_public_openapi.py`` when ``CLI_DOCS_PATH``
is set. Edit the source markdown (in ``calibrate-cli``) or the Getting-started
templates — never the generated pages.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli_reference import (  # noqa: E402
    CliCommand,
    Option,
    last_segment,
    parse_cli_doc,
    parse_options,
    resource_of,
    resource_slug,
    strip_md_suffix,
)
from docs_mdx import escape_mdx_prose as _escape_mdx_prose  # noqa: E402
from docs_mdx import frontmatter_value as _frontmatter_value  # noqa: E402
from docs_mdx import table_cell as _table_cell  # noqa: E402
from docs_nav import insert_tab  # noqa: E402
from request_examples import (  # noqa: E402
    NamedExample,
    learn_more_markdown,
    named_request_examples,
    render_cli_snippet,
)
from sdk_reference import api_group_from_tag, load_route_map  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
DOCS_JSON = DOCS_ROOT / "docs.json"
DEFAULT_OPENAPI = DOCS_ROOT / "api-reference" / "openapi.json"
OUTPUT_ROOT = DOCS_ROOT / "cli" / "calibrate"
TEMPLATE_DIR = REPO_ROOT / "docs" / "templates" / "cli" / "calibrate"

TAB_NAME = "CLI"
INSERT_AFTER_TAB = ("API reference", "Home")
ROOT_FILE = "calibrate.md"

# Hand-written conceptual pages copied verbatim into the Getting-started group,
# in sidebar order. Each is a template under TEMPLATE_DIR.
GETTING_STARTED = ["overview", "agent-mode"]

GENERATED_BANNER = (
    "{/* Generated from calibrate-cli/docs by scripts/generate_cli_docs.py — "
    "do not edit directly. */}"
)
# Token in the overview template replaced with the auto-generated global-flags
# table (the root command's persistent flags). If the root command is missing it
# collapses to an empty MDX comment.
GLOBAL_FLAGS_TOKEN = "{/* GLOBAL_FLAGS */}"

# Cobra emits placeholder descriptions for auto-generated auth/body flags
# (e.g. ``--x-api-key string   string value``). They add nothing, so a
# non-required flag whose description is one of these is dropped from tables.
PLACEHOLDER_DESCRIPTIONS = {"", "string value", "string"}

# Universal flag dropped from every table.
DROPPED_FLAGS = {"-h, --help"}

# The generic "send the whole request body as JSON" escape hatches. They're
# redundant noise when a command has real per-field flags, but they're the ONLY
# way to pass input when it doesn't (e.g. run-batch's `agent_names`), so they're
# dropped only when per-field flags exist.
BODY_FLAGS = {"--body", "-b, --body-param"}


def _keep_option(o: Option) -> bool:
    if o.flag in DROPPED_FLAGS:
        return False
    if o.required:
        return True
    return o.description.strip().lower() not in PLACEHOLDER_DESCRIPTIONS


def _options_table(options: list[Option]) -> str:
    """Render an options table (or ""); columns adapt to the rows.

    ``Option`` and ``Description`` always show; ``Type`` and ``Default`` are
    included only when at least one row populates them, so an all-empty column is
    never shown. Dropped/placeholder flags are filtered out first.
    """
    rows = [o for o in options if _keep_option(o)]
    # Drop the raw-body escape hatches only when real per-field flags exist.
    per_field = [o for o in rows if o.flag not in BODY_FLAGS]
    if per_field:
        rows = per_field
    if not rows:
        return ""
    show_type = any(o.type for o in rows)
    show_default = any(o.default for o in rows)

    headers = ["Option"] + (["Type"] if show_type else []) + (
        ["Default"] if show_default else []
    ) + ["Description"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for o in rows:
        desc = _table_cell(o.description)
        if o.required:
            desc = ("**Required.** " + desc).strip() if desc else "**Required.**"
        cells = [f"`{o.flag}`"]
        if show_type:
            cells.append(o.type or "—")
        if show_default:
            cells.append(f"`{o.default}`" if o.default else "—")
        cells.append(desc or "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _synopsis_lines(cmd: CliCommand) -> list[str]:
    """The long description as a paragraph — only when it adds info over ``short``."""
    synopsis = cmd.synopsis.strip()
    if synopsis and synopsis != cmd.short:
        return [_escape_mdx_prose(synopsis), ""]
    return []


def _command_key(command: str) -> str:
    """Normalize a command path for matching a CLI command to an SDK route.

    ``calibrate agents create`` -> ``agents create``; hyphens and underscores are
    unified (Fern uses ``agent_tests``, Cobra ``agent-tests``) so the two naming
    conventions — kept in lockstep by the backend overlays — compare equal.
    """
    parts = command.split()
    tail = parts[1:] if parts and parts[0] == "calibrate" else parts
    return " ".join(tail).replace("_", "-").lower()


def _examples_by_command(openapi: dict) -> dict[str, list[NamedExample]]:
    """Map each CLI command to its operation's ≥2 named request examples.

    Reuses the Fern route map (path + SDK group/method) to bridge a CLI command
    to its spec operation, since the SDK and CLI naming overlays are kept 1:1.
    Best-effort: if the Fern overrides aren't available (they live in the backend
    repo, present during the sync workflow), returns ``{}`` and the CLI pages
    fall back to Cobra's own Examples block.
    """
    try:
        routes = load_route_map(openapi=openapi)
    except SystemExit:
        return {}
    result: dict[str, list[NamedExample]] = {}
    for route in routes:
        op = openapi.get("paths", {}).get(route.path, {}).get(route.http.lower(), {})
        examples = named_request_examples(op) if isinstance(op, dict) else []
        if len(examples) >= 2:
            key = f"{route.sdk_group} {route.sdk_method}".replace("_", "-").lower()
            result[key] = examples
    return result


def _spec_examples_block(cmd: CliCommand, examples: list[NamedExample]) -> list[str]:
    """Render one labelled `bash` invocation per named request variant."""
    options = parse_options(cmd.options)
    lines = ["**Examples**", ""]
    for example in examples:
        lines += [
            f"_{example.summary}_",
            "",
            "```bash",
            render_cli_snippet(cmd.command, options, example.value),
            "```",
            "",
        ]
    note = learn_more_markdown(_command_key(cmd.command))
    if note:
        lines += [note, ""]
    return lines


def _command_body(
    cmd: CliCommand,
    spec_examples: list[NamedExample] | None = None,
) -> list[str]:
    """Usage block + options table + examples for a single command.

    When the spec supplies ≥2 named request examples for this command they
    replace Cobra's single (often empty) Examples block, so distinct request
    shapes stay visible on the CLI page.
    """
    lines: list[str] = []
    if cmd.usage:
        lines += ["```bash", cmd.usage.strip(), "```", ""]
    table = _options_table(parse_options(cmd.options))
    if table:
        lines += ["**Options**", "", table, ""]
    if spec_examples:
        lines += _spec_examples_block(cmd, spec_examples)
    elif cmd.examples:
        lines += ["**Examples**", "", "```bash", cmd.examples.strip(), "```", ""]
    return lines


def _subcommand_section(
    cmd: CliCommand,
    examples_by_command: dict[str, list[NamedExample]] | None = None,
) -> list[str]:
    heading = cmd.short.strip() or cmd.subcommand
    spec_examples = (examples_by_command or {}).get(_command_key(cmd.command))
    return [
        f"## {heading}",
        "",
        *_synopsis_lines(cmd),
        *_command_body(cmd, spec_examples),
    ]


def _frontmatter(title: str, description: str) -> list[str]:
    return [
        "---",
        f"title: {_frontmatter_value(title)}",
        f"description: {_frontmatter_value(description)}",
        "---",
        "",
        GENERATED_BANNER,
        "",
    ]


def render_resource_page(
    title: str,
    parent: CliCommand | None,
    subcommands: list[CliCommand],
    examples_by_command: dict[str, list[NamedExample]] | None = None,
) -> str:
    """A resource page — straight into one ## section per subcommand (no preamble)."""
    lines = _frontmatter(title, parent.short if parent else "")
    for cmd in subcommands:
        lines += _subcommand_section(cmd, examples_by_command)
    return "\n".join(lines).rstrip("\n") + "\n"


def render_leaf_page(
    title: str,
    cmd: CliCommand,
    examples_by_command: dict[str, list[NamedExample]] | None = None,
) -> str:
    """A standalone command (no subcommands) rendered as a single page."""
    spec_examples = (examples_by_command or {}).get(_command_key(cmd.command))
    lines = (
        _frontmatter(title, cmd.short)
        + _synopsis_lines(cmd)
        + _command_body(cmd, spec_examples)
    )
    return "\n".join(lines).rstrip("\n") + "\n"


def resource_title(resource: str) -> str:
    """Sidebar/page title for a resource, e.g. ``agent-tests`` -> "Agent tests".

    Uses the same tag→label transform the SDK generator applies to OpenAPI tags,
    so both docs sets label the same resource identically.
    """
    return api_group_from_tag(resource)


def resource_order(root_cmd: CliCommand | None) -> list[str]:
    """Resource names in the order the root command's SEE ALSO lists them."""
    if root_cmd is None:
        return []
    order: list[str] = []
    for ref in root_cmd.see_also:
        res = resource_of(ref.target)
        if res and res not in order:
            order.append(res)
    return order


def subcommand_order(parent: CliCommand | None) -> list[str]:
    """Subcommand segments in the order the parent command's SEE ALSO lists them."""
    if parent is None:
        return []
    order: list[str] = []
    for ref in parent.see_also:
        segments = strip_md_suffix(ref.target).split("_")
        if len(segments) >= 3:
            order.append(segments[-1])
    return order


def api_tags(openapi_path: Path) -> set[str] | None:
    """Set of OpenAPI operation tags, or ``None`` if the spec isn't available.

    Used to keep only API-backed command groups in the docs — a CLI command
    whose name is an OpenAPI tag (``agents``, ``agent-tests``) is a real API
    resource; local/utility commands (``auth``, ``configure``, ``version``, …)
    are not, and are covered under Getting started instead.
    """
    if not openapi_path.is_file():
        return None
    spec = json.loads(openapi_path.read_text(encoding="utf-8"))
    tags: set[str] = set()
    for ops in spec.get("paths", {}).values():
        if not isinstance(ops, dict):
            continue
        for op in ops.values():
            if isinstance(op, dict):
                tags.update(op.get("tags") or [])
    return tags


def _discover(src_dir: Path) -> tuple[CliCommand | None, dict[str, dict]]:
    """Parse every source file into the root command + per-resource groupings."""
    root_cmd: CliCommand | None = None
    resources: dict[str, dict] = {}
    for md_path in sorted(src_dir.glob("*.md")):
        cmd = parse_cli_doc(md_path.read_text(encoding="utf-8"))
        if md_path.name == ROOT_FILE:
            root_cmd = cmd
            continue
        res = resource_of(md_path.name)
        if res is None:
            continue
        entry = resources.setdefault(res, {"parent": None, "subs": {}})
        if len(strip_md_suffix(md_path.name).split("_")) == 2:
            entry["parent"] = cmd
        else:
            entry["subs"][last_segment(md_path.name)] = cmd
    return root_cmd, resources


def _ordered_subs(subs: dict[str, CliCommand], order: list[str]) -> list[CliCommand]:
    ordered = [subs[k] for k in order if k in subs]
    ordered += [subs[k] for k in sorted(subs) if k not in order]
    return ordered


def _ordered_resources(resources: dict[str, dict], order: list[str]) -> list[str]:
    ordered = [r for r in order if r in resources]
    ordered += [r for r in sorted(resources) if r not in order]
    return ordered


def _copy_getting_started(
    output_root: Path, template_dir: Path, substitutions: dict[str, str] | None = None
) -> list[str]:
    """Copy each hand-written Getting-started template; return page ids in order.

    ``substitutions`` replaces tokens (e.g. the global-flags placeholder) with
    generated content before writing.
    """
    substitutions = substitutions or {}
    output_root.mkdir(parents=True, exist_ok=True)
    page_ids: list[str] = []
    for name in GETTING_STARTED:
        template = template_dir / f"{name}.mdx"
        if not template.is_file():
            raise SystemExit(f"Getting-started template missing: {template}")
        text = template.read_text(encoding="utf-8")
        for token, value in substitutions.items():
            text = text.replace(token, value)
        (output_root / f"{name}.mdx").write_text(text, encoding="utf-8")
        page_ids.append(f"cli/calibrate/{name}")
    return page_ids


def global_flags_table(root_cmd: CliCommand | None) -> str:
    """The persistent-flags table for the overview, from the root command.

    The root ``calibrate`` command's own options are the global flags every
    subcommand inherits, so this stays complete and in sync automatically.
    """
    if root_cmd is None:
        return ""
    return _options_table(parse_options(root_cmd.options))


def _prune_stale(output_root: Path, keep_page_ids: set[str]) -> list[Path]:
    docs_root = output_root.parents[1]
    removed: list[Path] = []
    for mdx in output_root.rglob("*.mdx"):
        page_id = str(mdx.relative_to(docs_root).with_suffix("")).replace("\\", "/")
        if page_id not in keep_page_ids:
            mdx.unlink()
            removed.append(mdx)
    return removed


def patch_docs_json(
    docs_json: Path, tab: dict, insert_after_tab: str | tuple[str, ...]
) -> None:
    data = json.loads(docs_json.read_text(encoding="utf-8"))
    data["navigation"]["tabs"] = insert_tab(
        data["navigation"]["tabs"], tab, after=insert_after_tab
    )
    docs_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_tab(getting_started: list[str], resources: list[str]) -> dict:
    """Assemble the docs.json tab: Getting started, then Guides."""
    groups: list[dict] = [{"group": "Getting started", "pages": getting_started}]
    if resources:
        groups.append({"group": "Guides", "pages": resources})
    return {"tab": TAB_NAME, "groups": groups}


def _match_key(name: str) -> str:
    """Normalize a tag/resource name for forgiving comparison.

    Case-, hyphen-, underscore- and space-insensitive, so a backend tag rename
    like ``agents`` -> ``Agents`` does not silently drop the matching page.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _fail_missing_pages(tags: list[str]) -> None:
    """Abort when an API tag matched no CLI command — a likely dropped/renamed page.

    This runs unattended on a schedule, so instead of silently shipping docs with
    a page missing, emit a GitHub Actions error annotation and raise — failing the
    sync so a human resolves the mismatch before anything is published.
    """
    joined = ", ".join(tags)
    print(f"::error title=CLI docs::API tag(s) with no CLI page: {joined}")
    raise SystemExit(
        f"CLI docs generation aborted: {len(tags)} API tag(s) matched no CLI "
        f"command — no page would be generated for: {joined}. A backend tag "
        f"rename/removal may have dropped a page, or the CLI lacks a command for "
        f"this API resource. Fix the mismatch (or the tag) and re-run."
    )


def generate_cli_docs(
    src_dir: Path,
    *,
    docs_json: Path = DOCS_JSON,
    output_root: Path = OUTPUT_ROOT,
    template_dir: Path = TEMPLATE_DIR,
    openapi_path: Path = DEFAULT_OPENAPI,
    include_tags: set[str] | None = None,
) -> None:
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise SystemExit(f"CLI docs source directory not found: {src_dir}")

    docs_root = output_root.parents[1]
    root_cmd, resources = _discover(src_dir)

    # Only document API-backed command groups. ``include_tags`` overrides the
    # lookup (used by tests); otherwise derive from the OpenAPI spec. When the
    # spec is unavailable, include everything rather than silently dropping all.
    # Match forgivingly (case/hyphen-insensitive) so a tag rename doesn't drop a
    # page, and track which tags matched so unmatched ones can be flagged loudly.
    tags = include_tags if include_tags is not None else api_tags(openapi_path)
    tag_by_key = {_match_key(t): t for t in tags} if tags is not None else None

    # Spec-derived request examples, keyed by CLI command. Best-effort: a missing
    # or unreadable spec just means Cobra's own Examples blocks are used.
    examples_by_command: dict[str, list[NamedExample]] = {}
    try:
        spec = json.loads(Path(openapi_path).read_text(encoding="utf-8"))
        examples_by_command = _examples_by_command(spec)
    except (OSError, ValueError):
        pass

    # Decide what to document first, WITHOUT writing anything.
    selected: list[str] = []
    matched_tags: set[str] = set()
    skipped: list[str] = []
    for res in _ordered_resources(resources, resource_order(root_cmd)):
        if tag_by_key is not None:
            tag = tag_by_key.get(_match_key(res))
            if tag is None:
                skipped.append(res)
                continue
            matched_tags.add(tag)
        selected.append(res)

    # Fail loudly BEFORE touching any file: an API tag with no matching CLI
    # command means a page would silently vanish (e.g. a backend rename). Aborting
    # leaves the docs untouched rather than publishing an incomplete set.
    if tags is not None:
        unmatched = sorted(set(tags) - matched_tags)
        if unmatched:
            _fail_missing_pages(unmatched)

    resource_pages: list[str] = []
    for res in selected:
        entry = resources[res]
        title = resource_title(res)
        subs = _ordered_subs(entry["subs"], subcommand_order(entry["parent"]))
        if subs:
            content = render_resource_page(
                title, entry["parent"], subs, examples_by_command
            )
        elif entry["parent"] is not None:
            content = render_leaf_page(
                title, entry["parent"], examples_by_command
            )
        else:
            continue
        slug = resource_slug(res)
        out_path = docs_root / f"{slug}.mdx"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        resource_pages.append(slug)

    getting_started = _copy_getting_started(
        output_root,
        template_dir,
        {GLOBAL_FLAGS_TOKEN: global_flags_table(root_cmd)},
    )
    tab = build_tab(getting_started, resource_pages)
    page_ids = {*getting_started, *resource_pages}
    removed = _prune_stale(output_root, page_ids)
    patch_docs_json(docs_json, tab, INSERT_AFTER_TAB)

    print(f"Generated {len(resource_pages)} CLI resource page(s) under {output_root}")
    if skipped:
        print(f"Skipped {len(skipped)} non-API command(s): {', '.join(sorted(skipped))}")
    if removed:
        print(f"Pruned {len(removed)} stale page(s)")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        raise SystemExit("usage: generate_cli_docs.py <calibrate-cli-docs-dir>")
    generate_cli_docs(Path(argv[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
