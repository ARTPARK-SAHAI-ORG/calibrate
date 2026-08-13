"""Tests for scripts/generate_cli_docs.py (per-resource page generator)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_cli_docs import (  # noqa: E402
    _escape_mdx_prose,
    _options_table,
    _subcommand_section,
    _table_cell,
    api_tags,
    build_tab,
    generate_cli_docs,
    global_flags_table,
    patch_docs_json,
    render_leaf_page,
    render_resource_page,
    resource_order,
    resource_title,
    subcommand_order,
)
from cli_reference import Option, parse_cli_doc  # noqa: E402
from cli_doc_samples import API_TAGS, SAMPLES, write_samples  # noqa: E402


def _sample(name: str):
    """Parse a synthetic sample by filename."""
    return parse_cli_doc(SAMPLES[name])


# --------------------------------------------------------------------------- #
# _options_table
# --------------------------------------------------------------------------- #
def test_options_table_drops_help_and_body_marks_required() -> None:
    table = _options_table(
        [
            Option(flag="-h, --help", description="help for x"),
            Option(flag="--body", type="string", description="Request body as JSON"),
            Option(
                flag="-a, --agent-uuid",
                type="string",
                required=True,
                description="The agent to test",
            ),
        ]
    )
    # help is dropped; --body is dropped because a per-field flag exists
    assert "--help" not in table
    assert "--body" not in table
    # required flag keeps its marker and is backticked
    assert "**Required.**" in table
    assert "`-a, --agent-uuid`" in table


def test_options_table_keeps_body_when_it_is_the_only_input() -> None:
    # run-batch has no per-field flags — --body is the only way to pass input
    table = _options_table(
        [
            Option(flag="--body", type="string", description="Request body as JSON"),
            Option(flag="-b, --body-param", type="string", description="JSON object"),
        ]
    )
    assert "`--body`" in table


def test_options_table_omits_default_column_when_all_empty() -> None:
    table = _options_table(
        [
            Option(flag="-a, --agent-uuid", type="string", required=True, description="x"),
            Option(flag="-t, --test-uuids", type="string", description="y"),
        ]
    )
    assert "Default" not in table
    assert "| Option | Type | Description |" in table


def test_options_table_keeps_default_column_when_present() -> None:
    table = _options_table(
        [Option(flag="-t, --type", type="string", default="agent", description="y")]
    )
    assert "| Option | Type | Default | Description |" in table
    assert "`agent`" in table


def test_options_table_empty_when_only_help() -> None:
    assert _options_table([Option(flag="-h, --help", description="help")]) == ""


def test_options_table_drops_placeholder_only_flags() -> None:
    # Cobra emits "--x-api-key string   string value" — pure noise; dropped.
    table = _options_table(
        [
            Option(flag="--x-api-key", type="string", description="string value"),
            Option(flag="--x-org-uuid", type="string", description="string value"),
            Option(flag="-n, --names", type="stringArray", description="Names to resolve"),
        ]
    )
    assert "--x-api-key" not in table
    assert "--x-org-uuid" not in table
    assert "`-n, --names`" in table


def test_options_table_keeps_required_flag_even_without_description() -> None:
    # a required flag with no description ("[required]") must survive the filter
    table = _options_table([Option(flag="-n, --names", type="stringArray", required=True)])
    assert "`-n, --names`" in table
    assert "**Required.**" in table


def test_options_table_renders_default_in_backticks() -> None:
    table = _options_table(
        [
            Option(
                flag="-o, --output-format",
                type="string",
                default="pretty",
                description="Output format",
            )
        ]
    )
    assert "`pretty`" in table


# --------------------------------------------------------------------------- #
# _table_cell
# --------------------------------------------------------------------------- #
def test_table_cell_escapes_pipe_and_angle_brackets() -> None:
    cell = _table_cell("Filter (e.g. '.items[] | .id')")
    assert "\\|" in cell
    assert "|" not in cell.replace("\\|", "")
    assert _table_cell("<x>") == "&lt;x&gt;"


# --------------------------------------------------------------------------- #
# _escape_mdx_prose
# --------------------------------------------------------------------------- #
def test_escape_mdx_prose_escapes_outside_code_spans() -> None:
    assert _escape_mdx_prose("<id>") == "&lt;id&gt;"
    # inside backticks stays verbatim
    assert _escape_mdx_prose("`<id>`") == "`<id>`"
    assert _escape_mdx_prose("{x}") == "&#123;x&#125;"


# --------------------------------------------------------------------------- #
# _subcommand_section
# --------------------------------------------------------------------------- #
def test_subcommand_section_structure() -> None:
    # widgets create has real options, so it renders an Options table
    cmd = _sample("calibrate_widgets_create.md")
    text = "\n".join(_subcommand_section(cmd))
    assert text.startswith("## Create a widget")
    assert "```bash" in text
    assert "calibrate widgets create" in text
    assert "**Options**" in text
    assert "| Option | Type" in text
    assert "**Examples**" in text


def test_subcommand_section_omits_options_when_all_placeholders() -> None:
    # widgets list only carries placeholder x-* auth flags -> no Options table
    cmd = _sample("calibrate_widgets_list.md")
    text = "\n".join(_subcommand_section(cmd))
    assert text.startswith("## List widgets")
    assert "**Options**" not in text
    assert "--x-api-key" not in text


# --------------------------------------------------------------------------- #
# render_resource_page
# --------------------------------------------------------------------------- #
def test_render_resource_page() -> None:
    parent = _sample("calibrate_widgets.md")
    list_cmd = _sample("calibrate_widgets_list.md")
    create_cmd = _sample("calibrate_widgets_create.md")

    page = render_resource_page("Widgets", parent, [list_cmd, create_cmd])
    assert 'title: "Widgets"' in page
    assert "do not edit directly" in page
    assert "## List widgets" in page
    assert "## Create a widget" in page
    # minimal: no per-page global-flags callout, and it goes straight into the
    # first section with no preamble prose after the banner
    assert "<Note>" not in page
    banner_end = page.index("*/}") + len("*/}")
    assert page[banner_end:].lstrip().startswith("## List widgets")


# --------------------------------------------------------------------------- #
# render_leaf_page
# --------------------------------------------------------------------------- #
def test_render_leaf_page() -> None:
    cmd = _sample("calibrate_ping.md")
    page = render_leaf_page("Ping", cmd)
    assert 'title: "Ping"' in page
    assert "```bash" in page
    # a leaf page has no subcommand sections and no global-flags callout
    assert "## " not in page
    assert "<Note>" not in page


# --------------------------------------------------------------------------- #
# global_flags_table
# --------------------------------------------------------------------------- #
def test_global_flags_table_is_complete_from_root_command() -> None:
    root = _sample("calibrate.md")
    table = global_flags_table(root)
    # the full persistent-flag set — including ones a curated table would miss
    for flag in ("--no-interactive", "--agent-mode", "--output-format", "--jq"):
        assert flag in table
    assert "| Option | Type | Default | Description |" in table


def test_global_flags_table_empty_without_root() -> None:
    assert global_flags_table(None) == ""


# --------------------------------------------------------------------------- #
# derivation helpers (replace the old hand-maintained map)
# --------------------------------------------------------------------------- #
def test_resource_title_matches_sdk_tag_transform() -> None:
    assert resource_title("agent-tests") == "Agent tests"
    assert resource_title("agents") == "Agents"
    assert resource_title("auth") == "Auth"


def test_resource_order_follows_root_see_also() -> None:
    root = _sample("calibrate.md")
    order = resource_order(root)
    assert order == ["widgets", "ping", "secrets"]
    assert resource_order(None) == []


def test_subcommand_order_follows_parent_see_also() -> None:
    parent = _sample("calibrate_widgets.md")
    order = subcommand_order(parent)
    assert order == ["list", "create"]
    # the back-link to the root command is not treated as a subcommand
    assert "calibrate" not in order
    assert subcommand_order(None) == []


# --------------------------------------------------------------------------- #
# build_tab
# --------------------------------------------------------------------------- #
def test_build_tab_orders_getting_started_then_resources() -> None:
    tab = build_tab(
        getting_started=["cli/overview", "cli/agent-mode"],
        resources=["cli/agents", "cli/version"],
    )
    assert tab["tab"] == "CLI"
    assert [g["group"] for g in tab["groups"]] == ["Getting started", "Guides"]
    groups = {g["group"]: g["pages"] for g in tab["groups"]}
    assert groups["Getting started"] == ["cli/overview", "cli/agent-mode"]
    assert groups["Guides"] == ["cli/agents", "cli/version"]


def test_build_tab_omits_empty_resources_group() -> None:
    tab = build_tab(getting_started=["cli/overview"], resources=[])
    assert [g["group"] for g in tab["groups"]] == ["Getting started"]


# --------------------------------------------------------------------------- #
# api_tags (API-resource filter)
# --------------------------------------------------------------------------- #
def test_api_tags_reads_operation_tags(tmp_path: Path) -> None:
    spec = {
        "paths": {
            "/agents": {"get": {"tags": ["agents"]}},
            "/agent-tests/run": {"post": {"tags": ["agent-tests"]}},
        }
    }
    spec_file = tmp_path / "openapi.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    assert api_tags(spec_file) == {"agents", "agent-tests"}


def test_api_tags_none_when_spec_missing(tmp_path: Path) -> None:
    assert api_tags(tmp_path / "nope.json") is None


def test_generate_includes_all_when_tags_unknown(tmp_path: Path) -> None:
    # include_tags=None + missing spec -> include every resource (no silent drop)
    output_root, docs_json, template_dir, src = _setup_e2e(tmp_path)
    generate_cli_docs(
        src,
        docs_json=docs_json,
        output_root=output_root,
        template_dir=template_dir,
        openapi_path=tmp_path / "missing-openapi.json",
    )
    # no tags known -> even the non-API 'secrets' command is documented
    assert (output_root / "secrets.mdx").exists()
    assert (output_root / "widgets.mdx").exists()


# --------------------------------------------------------------------------- #
# patch_docs_json
# --------------------------------------------------------------------------- #
def test_patch_docs_json_inserts_after_and_is_idempotent(tmp_path: Path) -> None:
    docs_json = tmp_path / "docs.json"
    docs_json.write_text(
        json.dumps({"navigation": {"tabs": [{"tab": "Home"}, {"tab": "API reference"}]}}),
        encoding="utf-8",
    )
    tab = {"tab": "CLI", "groups": []}

    patch_docs_json(docs_json, tab, insert_after_tab="Home")
    tabs = json.loads(docs_json.read_text())["navigation"]["tabs"]
    assert [t["tab"] for t in tabs] == ["Home", "CLI", "API reference"]

    # idempotent — running again does not duplicate the tab
    patch_docs_json(docs_json, tab, insert_after_tab="Home")
    tabs = json.loads(docs_json.read_text())["navigation"]["tabs"]
    assert [t["tab"] for t in tabs].count("CLI") == 1
    assert [t["tab"] for t in tabs] == ["Home", "CLI", "API reference"]


# --------------------------------------------------------------------------- #
# generate_cli_docs (end-to-end)
# --------------------------------------------------------------------------- #
def _setup_e2e(tmp_path: Path):
    output_root = tmp_path / "docs" / "cli"
    output_root.mkdir(parents=True)
    docs_json = tmp_path / "docs" / "docs.json"
    docs_json.write_text(
        json.dumps({"navigation": {"tabs": [{"tab": "Home"}, {"tab": "API reference"}]}}),
        encoding="utf-8",
    )
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "overview.mdx").write_text("# Overview\n", encoding="utf-8")
    (template_dir / "agent-mode.mdx").write_text("# Agent mode\n", encoding="utf-8")
    src = write_samples(tmp_path / "src")
    return output_root, docs_json, template_dir, src


def _run_e2e(output_root, docs_json, template_dir, src) -> None:
    generate_cli_docs(
        src,
        docs_json=docs_json,
        output_root=output_root,
        template_dir=template_dir,
        include_tags=API_TAGS,
    )


def test_generate_cli_docs_end_to_end(tmp_path: Path) -> None:
    output_root, docs_json, template_dir, src = _setup_e2e(tmp_path)
    # output_root.parent must be the tmp docs dir so pages land under tmp
    assert output_root.parent == tmp_path / "docs"

    _run_e2e(output_root, docs_json, template_dir, src)

    widgets = output_root / "widgets.mdx"
    assert widgets.exists()
    assert "## List widgets" in widgets.read_text()

    # a single-command resource renders as one leaf page
    assert (output_root / "ping.mdx").exists()

    # non-API commands are not documented as resource pages
    assert not (output_root / "secrets.mdx").exists()

    # hand-written Getting-started pages are copied in
    assert (output_root / "overview.mdx").exists()
    assert (output_root / "agent-mode.mdx").exists()

    # no nested per-subcommand files
    assert not (output_root / "widgets" / "list.mdx").exists()

    tabs = json.loads(docs_json.read_text())["navigation"]["tabs"]
    # the cloud CLI tab is named "CLI", inserted after "API reference"
    assert [t["tab"] for t in tabs] == ["Home", "API reference", "CLI"]
    cli_tab = next(t for t in tabs if t["tab"] == "CLI")
    groups = {g["group"]: g["pages"] for g in cli_tab["groups"]}
    assert groups["Getting started"] == ["cli/overview", "cli/agent-mode"]
    # only API-backed resources appear under Guides
    assert set(groups["Guides"]) == {"cli/widgets", "cli/ping"}

    # running a second time still yields exactly one tab
    _run_e2e(output_root, docs_json, template_dir, src)
    tabs = json.loads(docs_json.read_text())["navigation"]["tabs"]
    assert [t["tab"] for t in tabs].count("CLI") == 1


def test_generate_cli_docs_prunes_stale_pages(tmp_path: Path) -> None:
    output_root, docs_json, template_dir, src = _setup_e2e(tmp_path)
    _run_e2e(output_root, docs_json, template_dir, src)

    stale = output_root / "stale.mdx"
    stale.write_text("stale\n", encoding="utf-8")

    _run_e2e(output_root, docs_json, template_dir, src)

    assert not stale.exists()
    assert (output_root / "widgets.mdx").exists()
    assert (output_root / "overview.mdx").exists()


def test_generate_forgiving_tag_match(tmp_path: Path) -> None:
    # a recased/re-spaced backend tag must still match its CLI command
    output_root, docs_json, template_dir, src = _setup_e2e(tmp_path)
    generate_cli_docs(
        src,
        docs_json=docs_json,
        output_root=output_root,
        template_dir=template_dir,
        include_tags={"Widgets", "PING"},
    )
    assert (output_root / "widgets.mdx").exists()
    assert (output_root / "ping.mdx").exists()


def test_generate_fails_hard_on_unmatched_tag(tmp_path: Path, capsys) -> None:
    # an API tag with no matching CLI command must abort, not vanish silently
    output_root, docs_json, template_dir, src = _setup_e2e(tmp_path)
    with pytest.raises(SystemExit) as exc:
        generate_cli_docs(
            src,
            docs_json=docs_json,
            output_root=output_root,
            template_dir=template_dir,
            include_tags={"widgets", "members"},
        )
    assert "members" in str(exc.value)
    assert "::error" in capsys.readouterr().out  # GitHub Actions annotation
    # nothing is written on failure — the docs are left untouched
    assert not (output_root / "widgets.mdx").exists()
    assert not (output_root / "overview.mdx").exists()


# --------------------------------------------------------------------------- #
# spec-derived request examples (connect-vs-create)
# --------------------------------------------------------------------------- #
def _spec_two_examples() -> dict:
    return {
        "paths": {
            "/agents": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "examples": {
                                    "build": {
                                        "summary": "Agent within Calibrate",
                                        "value": {"name": "Support Agent", "type": "agent"},
                                    },
                                    "connect": {
                                        "summary": "Connect external agent",
                                        "value": {
                                            "name": "My Hosted Agent",
                                            "type": "connection",
                                            "config": {"agent_url": "https://x.example.com/v1"},
                                        },
                                    },
                                }
                            }
                        }
                    },
                }
            }
        }
    }


def test_examples_by_command_maps_via_route_map() -> None:
    overrides = {
        "paths": {"/agents": {"post": {
            "x-fern-sdk-group-name": "agents",
            "x-fern-sdk-method-name": "create",
        }}}
    }
    from sdk_reference import build_route_map

    spec = _spec_two_examples()
    spec["paths"]["/agents"]["post"]["tags"] = ["agents"]
    # Route map bridges the CLI command to the op; key is the normalized command.
    routes = build_route_map(overrides, spec)
    assert routes  # sanity: the override resolves

    # _examples_by_command loads overrides from disk; exercise the pure mapping
    # by monkeypatching load_route_map to the in-memory route map.
    import generate_cli_docs as gcd

    orig = gcd.load_route_map
    gcd.load_route_map = lambda openapi=None: routes
    try:
        mapping = gcd._examples_by_command(spec)
    finally:
        gcd.load_route_map = orig
    assert set(mapping) == {"agents create"}
    assert [e.summary for e in mapping["agents create"]] == [
        "Agent within Calibrate",
        "Connect external agent",
    ]


def test_subcommand_section_injects_spec_examples() -> None:
    from cli_reference import CliCommand
    from generate_cli_docs import _subcommand_section
    from request_examples import named_request_examples

    cmd = CliCommand(
        command="calibrate agents create",
        short="Create agent",
        options="  --name string\n  --type type\n  -c, --config-param type",
        examples="calibrate agents create",  # Cobra's minimal block
    )
    examples = named_request_examples(_spec_two_examples()["paths"]["/agents"]["post"])
    section = "\n".join(_subcommand_section(cmd, {"agents create": examples}))
    assert "_Agent within Calibrate_" in section
    assert "_Connect external agent_" in section
    assert "--config-param '{\"agent_url\":\"https://x.example.com/v1\"}'" in section
    # Spec examples REPLACE Cobra's minimal block, not append to it.
    assert section.count("**Examples**") == 1


def test_subcommand_section_without_spec_examples_uses_cobra() -> None:
    from cli_reference import CliCommand
    from generate_cli_docs import _subcommand_section

    cmd = CliCommand(
        command="calibrate agents create",
        short="Create agent",
        examples="calibrate agents create --name X",
    )
    section = "\n".join(_subcommand_section(cmd, {}))
    assert "calibrate agents create --name X" in section


def test_subcommand_section_includes_learn_more_link() -> None:
    from cli_reference import CliCommand
    from generate_cli_docs import _subcommand_section
    from request_examples import named_request_examples

    cmd = CliCommand(
        command="calibrate agents create",
        short="Create agent",
        options="  --name string\n  -c, --config-param type",
    )
    examples = named_request_examples(_spec_two_examples()["paths"]["/agents"]["post"])
    section = "\n".join(_subcommand_section(cmd, {"agents create": examples}))
    assert "[Agent connections](/core-concepts/agent-connections)" in section
