#!/usr/bin/env python3
"""Parse Cobra-generated CLI reference markdown (``calibrate-cli/docs/*.md``).

Speakeasy/Cobra emits one ``.md`` per command with a fixed shape::

    ## calibrate agents list

    List Agents

    ### Synopsis

    <prose...>

    ```
    calibrate agents list [flags]
    ```

    ### Examples

    ```
      calibrate agents list
    ```

    ### Options

    ```
      -h, --help ...
    ```

    ### Options inherited from parent commands

    ```
      ...
    ```

    ### SEE ALSO

    * [calibrate agents](calibrate_agents.md)  - Operations for agents

This module turns that text into a :class:`CliCommand` and provides the
filename<->slug helpers the generator (``generate_cli_docs.py``) shares with
these tests. It performs no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Page-path prefix every generated cloud-CLI page lives under, e.g.
# ``cli/calibrate/agents/list``. The root command maps to the hand-written
# overview page.
SLUG_PREFIX = "cli/calibrate"
OVERVIEW_SLUG = f"{SLUG_PREFIX}/overview"
ROOT_FILENAME = "calibrate.md"


@dataclass
class SeeAlso:
    label: str  # "calibrate agents"
    target: str  # "calibrate_agents.md"
    description: str  # "Operations for agents"


@dataclass
class Option:
    flag: str  # "-o, --output-format" / "--dry-run"
    type: str = ""  # "string" / "stringArray" / "" for bool
    default: str = ""  # value inside (default "...")
    description: str = ""
    required: bool = False


@dataclass
class CliCommand:
    command: str  # "calibrate agents list"
    short: str  # one-line description
    synopsis: str = ""  # long description prose
    usage: str = ""  # contents of the usage code block
    examples: str = ""  # contents of the examples code block
    options: str = ""  # contents of the options code block
    inherited_options: str = ""  # inherited-options code block
    see_also: list[SeeAlso] = field(default_factory=list)

    @property
    def subcommand(self) -> str:
        """Command path without the leading ``calibrate `` (e.g. ``agents list``)."""
        parts = self.command.split()
        return " ".join(parts[1:]) if len(parts) > 1 else self.command


def strip_md_suffix(filename: str) -> str:
    return filename[:-3] if filename.endswith(".md") else filename


def command_from_filename(filename: str) -> str:
    """``calibrate_agent-tests_run.md`` -> ``calibrate agent-tests run``.

    Underscores separate command-path segments; hyphens live inside a segment.
    """
    return strip_md_suffix(filename).replace("_", " ")


def filename_to_slug(filename: str) -> str:
    """``calibrate_agents_list.md`` -> ``cli/calibrate/agents/list``.

    The root command (``calibrate.md``) maps to the overview page.
    """
    stem = strip_md_suffix(filename)
    segments = stem.split("_")
    if len(segments) <= 1:
        return OVERVIEW_SLUG
    return f"{SLUG_PREFIX}/" + "/".join(segments[1:])


def resource_of(filename: str) -> str | None:
    """Top-level command group a file belongs to.

    ``calibrate_agents_list.md`` and ``calibrate_agents.md`` -> ``agents``;
    the root ``calibrate.md`` -> ``None``.
    """
    segments = strip_md_suffix(filename).split("_")
    return segments[1] if len(segments) >= 2 else None


def last_segment(filename: str) -> str:
    """Trailing command segment: ``calibrate_agents_list.md`` -> ``list``."""
    return strip_md_suffix(filename).split("_")[-1]


def resource_slug(resource: str) -> str:
    return f"{SLUG_PREFIX}/{resource}"


def parse_options(block: str) -> list[Option]:
    """Parse a Cobra ``### Options`` code block into structured :class:`Option`s.

    Handles the aligned ``  -o, --flag TYPE   description (default "x")`` layout,
    boolean flags (no type token), and ``[required]`` markers.
    """
    options: list[Option] = []
    for raw in block.splitlines():
        if not raw.strip():
            continue
        m = re.match(r"^\s*((?:-\w, )?--[\w.-]+)(.*)$", raw)
        if not m:
            continue
        flag = m.group(1).strip()
        rest = m.group(2)

        # Cobra prints an optional type token one space after the flag, then 2+
        # spaces before the description. The token is the flag's value
        # placeholder — a standard type (``string``) or a custom one
        # (``type=agent``, an enum name); boolean flags have none. Take whatever
        # sits in that slot rather than allow-listing type names.
        type_ = ""
        tm = re.match(r"^\s(\S+)\s{2,}", rest)
        if tm:
            type_ = tm.group(1)
            desc = rest[tm.end() :].strip()
        else:
            desc = rest.strip()

        default = ""
        dm = re.search(r"\(default (.*?)\)\s*$", desc)
        if dm:
            default = dm.group(1).strip().strip('"')
            desc = desc[: dm.start()].strip()

        required = False
        if desc.endswith("[required]"):
            required = True
            desc = desc[: -len("[required]")].strip()

        options.append(
            Option(
                flag=flag,
                type=type_,
                default=default,
                description=desc,
                required=required,
            )
        )
    return options


def _extract_code_block(text: str) -> str:
    """Return the contents of the first fenced code block in ``text`` (or "")."""
    match = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    return match.group(1).rstrip("\n") if match else ""


def _split_sections(body: str) -> tuple[str, dict[str, str]]:
    """Split on ``### `` headers.

    Returns ``(preamble, sections)`` where ``preamble`` is everything before the
    first ``### `` header and ``sections`` maps header text -> section body.
    """
    parts = re.split(r"^### (.+)$", body, flags=re.MULTILINE)
    preamble = parts[0]
    sections: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        sections[header] = content
    return preamble, sections


def _parse_see_also(section: str) -> list[SeeAlso]:
    entries: list[SeeAlso] = []
    pattern = re.compile(r"\*\s+\[([^\]]+)\]\(([^)]+)\)\s*-?\s*(.*)")
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("*"):
            continue
        m = pattern.match(line)
        if not m:
            continue
        label, target, desc = m.group(1), m.group(2), m.group(3).strip()
        entries.append(
            SeeAlso(label=label.strip(), target=target.strip(), description=desc)
        )
    return entries


def parse_cli_doc(text: str) -> CliCommand:
    """Parse one Cobra command markdown file into a :class:`CliCommand`."""
    text = text.replace("\r\n", "\n")

    title_match = re.search(r"^##\s+(.+)$", text, re.MULTILINE)
    if not title_match:
        raise ValueError("CLI doc missing '## <command>' title")
    command = title_match.group(1).strip()

    body = text[title_match.end() :]
    preamble, sections = _split_sections(body)

    short = " ".join(
        line.strip() for line in preamble.strip().splitlines() if line.strip()
    )

    synopsis_section = sections.get("Synopsis", "")
    usage = _extract_code_block(synopsis_section)
    # Synopsis prose is everything before its (usage) code fence.
    synopsis_prose = re.split(r"```", synopsis_section, maxsplit=1)[0].strip()
    # Fall back to the short description when there's no distinct long form.
    synopsis = synopsis_prose or short

    return CliCommand(
        command=command,
        short=short,
        synopsis=synopsis,
        usage=usage,
        examples=_extract_code_block(sections.get("Examples", "")),
        options=_extract_code_block(sections.get("Options", "")),
        inherited_options=_extract_code_block(
            sections.get("Options inherited from parent commands", "")
        ),
        see_also=_parse_see_also(sections.get("SEE ALSO", "")),
    )
