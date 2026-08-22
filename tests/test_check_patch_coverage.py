"""Tests for scripts/check_patch_coverage.py (the local Codecov patch check)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_patch_coverage as cpc  # noqa: E402


def _write_report(tmp_path: Path, line_rate: str | None) -> Path:
    attr = "" if line_rate is None else f' line-rate="{line_rate}"'
    path = tmp_path / "coverage.xml"
    path.write_text(f"<coverage{attr}><packages/></coverage>")
    return path


class TestOverallLineRate:
    def test_converts_the_rate_to_a_percentage(self, tmp_path: Path) -> None:
        assert cpc.overall_line_rate(_write_report(tmp_path, "0.8861")) == pytest.approx(
            88.61
        )

    def test_full_coverage_is_a_hundred(self, tmp_path: Path) -> None:
        assert cpc.overall_line_rate(_write_report(tmp_path, "1")) == 100.0

    def test_a_report_without_the_rate_is_an_error(self, tmp_path: Path) -> None:
        # A silent 0% target would pass every branch while checking nothing.
        with pytest.raises(ValueError, match="line-rate"):
            cpc.overall_line_rate(_write_report(tmp_path, None))


class TestMergeBase:
    def test_prefers_the_remote_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cpc, "_git", lambda *a: subprocess.CompletedProcess(a, 0, "sha1\n", "")
        )
        assert cpc.merge_base() == ("origin/main", "sha1")

    def test_falls_back_to_the_local_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_git(*args: str) -> subprocess.CompletedProcess:
            if "origin/main" in args:
                return subprocess.CompletedProcess(args, 128, "", "no such ref")
            return subprocess.CompletedProcess(args, 0, "sha2\n", "")

        monkeypatch.setattr(cpc, "_git", fake_git)
        assert cpc.merge_base() == ("main", "sha2")

    def test_no_main_at_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cpc, "_git", lambda *a: subprocess.CompletedProcess(a, 128, "", "")
        )
        assert cpc.merge_base() is None


class TestMeasuredFilesChanged:
    def test_true_when_the_diff_names_a_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cpc,
            "_git",
            lambda *a: subprocess.CompletedProcess(a, 0, "calibrate_agent/utils.py\n", ""),
        )
        assert cpc.measured_files_changed("sha") is True

    def test_false_when_the_diff_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cpc, "_git", lambda *a: subprocess.CompletedProcess(a, 0, "\n", "")
        )
        assert cpc.measured_files_changed("sha") is False

    def test_scopes_the_diff_to_the_measured_directory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[str, ...]] = []

        def fake_git(*args: str) -> subprocess.CompletedProcess:
            seen.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(cpc, "_git", fake_git)
        cpc.measured_files_changed("base_sha")
        assert seen == [("diff", "--name-only", "base_sha", "--", "calibrate_agent")]


class TestMain:
    def test_skips_when_there_is_no_main_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cpc, "merge_base", lambda: None)
        monkeypatch.setattr(
            cpc.subprocess, "run", lambda *a, **k: pytest.fail("nothing should run")
        )
        assert cpc.main([]) == 0

    def test_skips_when_no_measured_file_changed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cpc, "merge_base", lambda: ("origin/main", "sha"))
        monkeypatch.setattr(cpc, "measured_files_changed", lambda base: False)
        monkeypatch.setattr(
            cpc.subprocess, "run", lambda *a, **k: pytest.fail("nothing should run")
        )
        assert cpc.main([]) == 0

    def test_a_failing_suite_stops_before_the_coverage_check(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(cpc, "merge_base", lambda: ("origin/main", "sha"))
        monkeypatch.setattr(cpc, "measured_files_changed", lambda base: True)
        # A usable report, so only the early exit can stop the coverage check.
        monkeypatch.setattr(cpc, "COVERAGE_XML", _write_report(tmp_path, "0.9"))
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 1)

        monkeypatch.setattr(cpc.subprocess, "run", fake_run)
        assert cpc.main([]) == 1
        assert len(calls) == 1
        assert "pytest" in calls[0]

    def test_missing_report_with_no_tests(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(cpc, "merge_base", lambda: ("origin/main", "sha"))
        monkeypatch.setattr(cpc, "measured_files_changed", lambda base: True)
        monkeypatch.setattr(cpc, "COVERAGE_XML", tmp_path / "coverage.xml")
        assert cpc.main(["--no-tests"]) == 1

    def test_passes_the_overall_coverage_as_the_target(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(cpc, "merge_base", lambda: ("origin/main", "sha"))
        monkeypatch.setattr(cpc, "measured_files_changed", lambda base: True)
        monkeypatch.setattr(cpc, "COVERAGE_XML", _write_report(tmp_path, "0.8861"))
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(cpc.subprocess, "run", fake_run)
        assert cpc.main(["--no-tests"]) == 0
        assert calls[0][4] == "diff-cover"
        assert "--compare-branch=origin/main" in calls[0]
        assert "--fail-under=88.61" in calls[0]

    def test_reports_the_coverage_check_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(cpc, "merge_base", lambda: ("origin/main", "sha"))
        monkeypatch.setattr(cpc, "measured_files_changed", lambda base: True)
        monkeypatch.setattr(cpc, "COVERAGE_XML", _write_report(tmp_path, "0.9"))
        monkeypatch.setattr(
            cpc.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 1)
        )
        assert cpc.main(["--no-tests"]) == 1
