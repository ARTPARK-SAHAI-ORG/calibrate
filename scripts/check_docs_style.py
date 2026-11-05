#!/usr/bin/env python3
"""Check docs pages against the mechanical rules in DOCS_STYLE.md.

Only the rules a regular expression settles with no false positives live here:
em-dashes and a list of banned phrases. Everything that needs the sentence
around it read (sentence case headings, showing a shape instead of describing
it, naming the setting value beside a shape, plain words, one idea per
sentence) belongs to the Claude review, which reads the same DOCS_STYLE.md.

Usage:
    python scripts/check_docs_style.py --staged          # staged changes
    python scripts/check_docs_style.py --base origin/main  # a branch's changes
    python scripts/check_docs_style.py docs/cli/text-to-text.mdx  # whole files

With --staged or --base only the lines the change adds are reported, so an
existing page is never flagged for wording someone else wrote. Lines inside
fenced code blocks and the frontmatter block are skipped.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Pages generated from the backend spec, the cloud CLI, or the MCP server. Their
# wording comes from those sources, so a regenerate must never fail this check.
GENERATED_PREFIXES = (
    "docs/api-reference/",
    "docs/sdk/",
    "docs/mcp/",
    "docs/cli/calibrate/",
)
GENERATED_FILES = (
    "docs/reference/api-keys.mdx",
    "docs/reference/github-actions.mdx",
)

# Phrases with no reading other than the one DOCS_STYLE.md bans. A phrase that
# also has an ordinary use ("the cost per call", "the page opens on its own
# page") is left to the Claude review, which can read the sentence around it. A
# blocking check that fires on ordinary writing gets bypassed, and then it
# enforces nothing.
BANNED_PHRASES = (
    "one instruction per call",
    "plain instruction",
    "under the hood",
    "out of the box",
    "as requested",
    "previously",
)

_CODE_SPAN = re.compile(r"`[^`]*`")


def is_checked(path: str) -> bool:
    """True for a hand-written docs page."""
    return (
        path.startswith("docs/")
        and path.endswith(".mdx")
        and not path.startswith(GENERATED_PREFIXES)
        and path not in GENERATED_FILES
    )


def prose_lines(text: str) -> dict[int, str]:
    """Return the 1-indexed lines that carry prose.

    Fenced code blocks and the frontmatter block are left out: a request body or
    a shell command is not writing.
    """
    out: dict[int, str] = {}
    in_fence = False
    in_frontmatter = False
    for n, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if n == 1 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out[n] = line
    return out


def line_violations(line: str) -> list[tuple[str, str]]:
    """Return (rule, detail) for every rule the line breaks."""
    found: list[tuple[str, str]] = []
    bare = _CODE_SPAN.sub(" ", line)

    if "—" in line:
        found.append(("em-dash", "use a comma, a period, or a colon"))

    lowered = bare.lower()
    for phrase in BANNED_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            found.append(("banned phrase", f'"{phrase}"'))

    return found


def check_file(path: Path, only_lines: set[int] | None) -> list[str]:
    """Return one message per violation in ``path``."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    messages = []
    for n, line in prose_lines(text).items():
        if only_lines is not None and n not in only_lines:
            continue
        for rule, detail in line_violations(line):
            messages.append(f"{path}:{n}: {rule}: {detail}")
    return messages


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def added_lines(diff: str) -> dict[str, set[int]]:
    """Map each file in a unified diff to the line numbers it adds."""
    out: dict[str, set[int]] = {}
    path = None
    lineno = 0
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            out.setdefault(path, set())
        elif line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            lineno = int(match.group(1)) if match else 0
        elif line.startswith("+") and not line.startswith("+++"):
            if path:
                out[path].add(lineno)
            lineno += 1
        elif not line.startswith("-") and not line.startswith("\\"):
            lineno += 1
    return out


def from_git(staged: bool, base: str | None) -> dict[str, set[int]]:
    args = ["diff", "--unified=0", "--diff-filter=ACMR"]
    if staged:
        args.append("--cached")
    elif base:
        args.append(f"{base}...HEAD")
    return added_lines(_git(*args))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="docs pages to check in full")
    parser.add_argument("--staged", action="store_true", help="check staged changes")
    parser.add_argument("--base", help="check what this branch adds on top of BASE")
    args = parser.parse_args(argv)

    if args.staged or args.base:
        changed = from_git(args.staged, args.base)
        targets = {p: lines for p, lines in changed.items() if is_checked(p) and lines}
    else:
        targets = {p: None for p in args.paths if is_checked(p)}

    messages: list[str] = []
    for path, lines in sorted(targets.items()):
        messages.extend(check_file(Path(path), lines))

    if not messages:
        return 0

    print("✗ Docs style check failed. See DOCS_STYLE.md for each rule.\n")
    for message in messages:
        print(f"  {message}")
    print(f"\n{len(messages)} violation(s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
