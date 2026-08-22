#!/usr/bin/env python
"""Local stand-in for Codecov's `patch` check: are the lines this branch adds
covered by tests?

Codecov's patch target is `auto`, i.e. the base commit's coverage, so there is
no fixed number to hardcode. This uses the run's own overall coverage as the
local stand-in for the base: main is nearly all of the code, so its coverage
and this run's overall coverage sit within a point of each other.

Run with --no-tests to reuse an existing coverage.xml instead of re-running the
suite.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE_XML = ROOT / "coverage.xml"
MEASURED_DIR = "calibrate_agent"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def merge_base() -> tuple[str, str] | None:
    """The branch to compare against and the commit it forked from."""
    for ref in ("origin/main", "main"):
        found = _git("merge-base", ref, "HEAD")
        if found.returncode == 0:
            return ref, found.stdout.strip()
    return None


def measured_files_changed(base: str) -> bool:
    """Has this branch touched the code coverage measures?"""
    return bool(_git("diff", "--name-only", base, "--", MEASURED_DIR).stdout.strip())


def overall_line_rate(coverage_xml: Path) -> float:
    """Percentage of measured lines the whole run covered."""
    rate = ET.parse(coverage_xml).getroot().get("line-rate")
    if rate is None:
        raise ValueError(f"{coverage_xml} has no line-rate — is it a coverage report?")
    return float(rate) * 100


def main(argv: list[str]) -> int:
    found = merge_base()
    if found is None:
        print("No main branch to compare against — skipping the patch coverage check.")
        return 0
    ref, base = found

    if not measured_files_changed(base):
        print(f"No {MEASURED_DIR}/ changes — skipping the patch coverage check.")
        return 0

    if "--no-tests" not in argv:
        tests = subprocess.run(
            [
                "uv", "run", "--extra", "dev", "pytest", "tests/", "-q",
                "--cov=calibrate_agent", "--cov-report=xml", "--cov-report=term",
            ],
            cwd=ROOT,
            check=False,
        )
        if tests.returncode != 0:
            return tests.returncode

    if not COVERAGE_XML.exists():
        print(f"No {COVERAGE_XML.name} — run this without --no-tests.", file=sys.stderr)
        return 1

    target = overall_line_rate(COVERAGE_XML)
    print(f"Target (this run's overall coverage): {target:.2f}%")

    return subprocess.run(
        [
            "uv", "run", "--extra", "dev", "diff-cover", str(COVERAGE_XML),
            f"--compare-branch={ref}", f"--fail-under={target:.2f}",
        ],
        cwd=ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
