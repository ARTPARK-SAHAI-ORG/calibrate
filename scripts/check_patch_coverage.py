#!/usr/bin/env python
"""Local stand-in for Codecov's `patch` check: are the lines this branch adds
covered by tests?

Codecov's patch target is `auto`, i.e. the base commit's coverage, so there is
no fixed number to hardcode. This uses the run's own overall coverage as the
local stand-in for the base: main is nearly all of the code, so its coverage
and this run's overall coverage sit within a point of each other.
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COVERAGE_XML = ROOT / "coverage.xml"


def compare_branch():
    for ref in ("origin/main", "main"):
        if subprocess.run(
            ["git", "merge-base", ref, "HEAD"], cwd=ROOT, capture_output=True
        ).returncode == 0:
            return ref
    return None


def main():
    run_tests = "--no-tests" not in sys.argv
    if run_tests:
        tests = subprocess.run(
            [
                "uv", "run", "--extra", "dev", "pytest", "tests/", "-q",
                "--cov=calibrate_agent", "--cov-report=xml", "--cov-report=term",
            ],
            cwd=ROOT,
        )
        if tests.returncode != 0:
            return tests.returncode

    if not COVERAGE_XML.exists():
        print("No coverage.xml — run this without --no-tests.", file=sys.stderr)
        return 1

    base = compare_branch()
    if base is None:
        print("No main branch to compare against — skipping patch coverage.")
        return 0

    overall = float(ET.parse(COVERAGE_XML).getroot().get("line-rate", 0)) * 100
    print(f"Target (this run's overall coverage): {overall:.2f}%")

    return subprocess.run(
        [
            "uv", "run", "--extra", "dev", "diff-cover", str(COVERAGE_XML),
            f"--compare-branch={base}", f"--fail-under={overall:.2f}",
        ],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
