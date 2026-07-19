"""Tests for scripts/cli_reference.py.

Inputs are synthetic Cobra docs from ``cli_doc_samples`` — not a snapshot of the
real CLI (which drifts and is validated by the sync workflow instead).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli_reference import (
    OVERVIEW_SLUG,
    CliCommand,
    Option,
    SeeAlso,
    command_from_filename,
    filename_to_slug,
    last_segment,
    parse_cli_doc,
    parse_options,
    resource_of,
    resource_slug,
)
from cli_doc_samples import ROOT_CMD, WIDGETS, WIDGETS_CREATE


# --------------------------------------------------------------------------- #
# filename helpers
# --------------------------------------------------------------------------- #
def test_command_from_filename_hyphenated_segment() -> None:
    assert (
        command_from_filename("calibrate_agent-tests_run.md")
        == "calibrate agent-tests run"
    )


def test_command_from_filename_simple() -> None:
    assert command_from_filename("calibrate_agents_list.md") == "calibrate agents list"


def test_filename_to_slug_leaf() -> None:
    assert filename_to_slug("calibrate_agents_list.md") == "cli/calibrate/agents/list"


def test_filename_to_slug_hyphenated_segment() -> None:
    assert (
        filename_to_slug("calibrate_agent-tests_run.md")
        == "cli/calibrate/agent-tests/run"
    )


def test_filename_to_slug_root_is_overview() -> None:
    assert filename_to_slug("calibrate.md") == OVERVIEW_SLUG
    assert OVERVIEW_SLUG == "cli/calibrate/overview"


def test_resource_of() -> None:
    assert resource_of("calibrate_agents_list.md") == "agents"
    assert resource_of("calibrate_agents.md") == "agents"
    assert resource_of("calibrate_agent-tests_run.md") == "agent-tests"
    assert resource_of("calibrate.md") is None


def test_last_segment() -> None:
    assert last_segment("calibrate_agents_list.md") == "list"
    assert last_segment("calibrate_agent-tests_run-batch.md") == "run-batch"
    assert last_segment("calibrate_configure.md") == "configure"


def test_resource_slug() -> None:
    assert resource_slug("agents") == "cli/calibrate/agents"


# --------------------------------------------------------------------------- #
# parse_cli_doc
# --------------------------------------------------------------------------- #
def test_parse_command_all_sections() -> None:
    cmd = parse_cli_doc(WIDGETS_CREATE)

    assert cmd.command == "calibrate widgets create"
    assert cmd.short == "Create a widget"
    assert cmd.synopsis.startswith("Create a new widget")
    assert "calibrate widgets create [flags]" in cmd.usage
    assert "calibrate widgets create --name" in cmd.examples
    assert "--widget-id" in cmd.options
    assert "--agent-mode" in cmd.inherited_options

    (entry,) = [s for s in cmd.see_also if s.target == "calibrate_widgets.md"]
    assert isinstance(entry, SeeAlso)
    assert entry.label == "calibrate widgets"
    assert entry.description == "Operations for widgets"


def test_parse_root_lists_resources_in_see_also() -> None:
    cmd = parse_cli_doc(ROOT_CMD)

    assert cmd.command == "calibrate"
    assert cmd.examples == ""
    assert [s.target for s in cmd.see_also] == [
        "calibrate_widgets.md",
        "calibrate_ping.md",
        "calibrate_secrets.md",
    ]
    assert cmd.subcommand == "calibrate"


def test_parse_falls_back_to_short_synopsis() -> None:
    # WIDGETS parent's synopsis equals its short line
    cmd = parse_cli_doc(WIDGETS)
    assert cmd.short == "Operations for widgets"
    assert cmd.synopsis == "Operations for widgets"


def test_parse_minimal_synthetic_doc() -> None:
    text = (
        "## calibrate demo\n\nDo a demo\n\n"
        "### Synopsis\n\nLonger demo prose here.\n\n"
        "```\ncalibrate demo [flags]\n```\n"
    )
    cmd = parse_cli_doc(text)

    assert cmd.command == "calibrate demo"
    assert cmd.short == "Do a demo"
    assert cmd.synopsis == "Longer demo prose here."
    assert cmd.usage == "calibrate demo [flags]"
    assert cmd.examples == ""
    assert cmd.options == ""
    assert cmd.inherited_options == ""
    assert cmd.see_also == []


def test_parse_cli_doc_requires_title() -> None:
    with pytest.raises(ValueError):
        parse_cli_doc("no title here\njust prose\n")


def test_subcommand_property() -> None:
    assert (
        CliCommand(command="calibrate agents list", short="x").subcommand
        == "agents list"
    )


# --------------------------------------------------------------------------- #
# parse_options
# --------------------------------------------------------------------------- #
def _by_flag(options: list[Option], flag: str) -> Option:
    matches = [o for o in options if o.flag == flag]
    assert len(matches) == 1, f"expected exactly one {flag!r}, got {matches}"
    return matches[0]


def test_parse_options_covers_every_variation() -> None:
    options = parse_options(parse_cli_doc(WIDGETS_CREATE).options)

    # required flag: [required] suffix stripped, description kept
    required = _by_flag(options, "-a, --widget-id")
    assert required.required is True
    assert required.type == "string"
    assert required.description == "The widget to update. Must exist."

    # bool flag: no type token
    boolean = _by_flag(options, "--enabled")
    assert boolean.type == ""
    assert boolean.description == "Enable the widget on create"

    # custom (non-standard) type placeholder is taken verbatim
    custom = _by_flag(options, "-c, --config-param")
    assert custom.type == "type=full"
    assert custom.description == "Behavioral config. The keys depend on type."

    # typed flag with a default
    tier = _by_flag(options, "-t, --tier")
    assert tier.type == "tier"
    assert tier.default == "basic"
    assert tier.description == "Tier level"

    # placeholder junk flags still parse (the generator drops them later)
    assert _by_flag(options, "--x-api-key").description == "string value"


def test_parse_options_default_with_spaces() -> None:
    line = '      --color string           Control colored output (default "auto")\n'
    (opt,) = parse_options(line)
    assert opt.default == "auto"


def test_parse_options_jq_pipe_in_description() -> None:
    options = parse_options(parse_cli_doc(ROOT_CMD).options)
    jq = _by_flag(options, "-q, --jq")
    assert "|" in jq.description  # a raw pipe survives parsing


def test_parse_options_empty_inputs() -> None:
    assert parse_options("") == []
    assert parse_options("\n   \n\n") == []
