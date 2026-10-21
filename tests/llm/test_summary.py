"""Tests for calibrate_agent.llm.summary and the places that use it."""

import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from calibrate_agent.llm.summary import (
    NO_ANSWER,
    aggregate_criteria,
    format_pass_rate,
    format_share,
    mean_interval,
    pass_rate_from_metrics,
    pass_rate_summary,
    wilson_interval,
)


def _row(passed, error=False):
    row = {"metrics": {"passed": passed}}
    if error:
        row["error"] = True
    return row


class TestWilsonInterval(unittest.TestCase):
    def test_worked_examples(self):
        ci = wilson_interval(15, 20)
        self.assertEqual((round(ci["low"], 1), round(ci["high"], 1)), (53.1, 88.8))
        ci = wilson_interval(19, 20)
        self.assertEqual((round(ci["low"], 1), round(ci["high"], 1)), (76.4, 99.1))
        self.assertEqual(round(wilson_interval(20, 20)["high"], 1), 100.0)

    def test_stays_inside_0_to_100(self):
        self.assertEqual(wilson_interval(0, 20)["low"], 0.0)
        self.assertEqual(wilson_interval(19, 19)["high"], 100.0)

    def test_no_cases(self):
        self.assertIsNone(wilson_interval(0, 0))
        self.assertIsNone(wilson_interval(0, -1))


class TestMeanInterval(unittest.TestCase):
    def test_interval_around_mean(self):
        ci = mean_interval([2, 3, 4, 3], 1, 5)
        self.assertAlmostEqual((ci["low"] + ci["high"]) / 2, 3.0)
        self.assertLess(ci["low"], 3.0)

    def test_clamped_to_scale(self):
        self.assertEqual(mean_interval([1, 5], 1, 5), {"low": 1, "high": 5})

    def test_too_few_scores(self):
        self.assertIsNone(mean_interval([4], 1, 5))
        self.assertIsNone(mean_interval([], 1, 5))

    def test_identical_scores(self):
        self.assertIsNone(mean_interval([3, 3, 3], 1, 5))


class TestPassRate(unittest.TestCase):
    def test_leaves_out_unanswered_rows(self):
        rows = [_row(True), _row(True), _row(False), _row(False, error=True)]
        summary = pass_rate_summary(rows)
        self.assertAlmostEqual(summary["pass_rate"], 2 / 3 * 100)
        self.assertEqual(summary["pass_rate_ci"], wilson_interval(2, 3))

    def test_nothing_answered(self):
        summary = pass_rate_summary([_row(False, error=True)])
        self.assertEqual(summary, {"pass_rate": None, "pass_rate_ci": None})

    def test_from_metrics_matches_rows(self):
        metrics = {"passed": 2, "total": 4, "errored": 1}
        rows = [_row(True), _row(True), _row(False), _row(False, error=True)]
        self.assertEqual(pass_rate_from_metrics(metrics), pass_rate_summary(rows))

    def test_from_metrics_without_errored(self):
        self.assertEqual(pass_rate_from_metrics({"passed": 3, "total": 4})["pass_rate"], 75.0)

    def test_format(self):
        self.assertEqual(
            format_pass_rate({"passed": 2, "total": 4, "errored": 1}), "2/3 (66.7%)"
        )
        self.assertEqual(format_pass_rate({}), NO_ANSWER)
        self.assertEqual(
            format_pass_rate({"passed": 0, "total": 2, "errored": 2}), NO_ANSWER
        )
        self.assertEqual(format_share(1, 4), "1/4 (25.0%)")


def _case(criteria, judge_results, type_="response"):
    return {
        "metrics": {"passed": True, "judge_results": judge_results},
        "test_case": {"evaluation": {"type": type_, "criteria": criteria}},
    }


class TestAggregateCriteria(unittest.TestCase):
    def test_binary_and_rating_carry_ci(self):
        registry = {
            "acc": {"name": "acc", "type": "binary", "id": "ev-1"},
            "flu": {"name": "flu", "type": "rating", "scale_min": 1, "scale_max": 5},
        }
        refs = [{"name": "acc"}, {"name": "flu"}]
        results = [
            _case(refs, {"acc": {"match": True}, "flu": {"score": 4}}),
            _case(refs, {"acc": {"match": False}, "flu": {"score": 2}}),
        ]
        agg = aggregate_criteria(results, registry)
        self.assertEqual(agg["acc"]["ci"], wilson_interval(1, 2))
        self.assertEqual(agg["acc"]["evaluator_id"], "ev-1")
        self.assertEqual(agg["flu"]["ci"], mean_interval([4, 2], 1, 5))
        self.assertNotIn("evaluator_id", agg["flu"])

    def test_single_rating_has_no_ci(self):
        registry = {"flu": {"name": "flu", "type": "rating", "scale_min": 1, "scale_max": 5}}
        agg = aggregate_criteria([_case([{"name": "flu"}], {"flu": {"score": 3}})], registry)
        self.assertIsNone(agg["flu"]["ci"])

    def test_string_criteria_use_default_evaluator(self):
        registry = {"correctness": {"name": "correctness", "type": "binary"}}
        agg = aggregate_criteria(
            [_case("be nice", {"correctness": {"match": True}})], registry
        )
        self.assertEqual(agg["correctness"]["passed"], 1)


class TestImportIsLight(unittest.TestCase):
    def test_no_heavy_modules(self):
        code = (
            "import sys, calibrate_agent.llm.summary;"
            "print([m for m in ('pandas','numpy','pipecat','instructor') if m in sys.modules])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(out.stdout.strip(), "[]")


class TestWriteOutputs(unittest.TestCase):
    def test_metrics_json_has_pass_rate_and_ci(self):
        from calibrate_agent.llm.run_tests import _write_test_results_outputs

        rows = [_row(True), _row(False), _row(False, error=True)]
        with tempfile.TemporaryDirectory() as tmp:
            _write_test_results_outputs(rows, tmp, {})
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())
        self.assertEqual(metrics["pass_rate"], 50.0)
        self.assertEqual(metrics["pass_rate_ci"], wilson_interval(1, 2))

    def test_finish_test_run_logs_each_outcome(self):
        from calibrate_agent.llm.run_tests import finish_test_run

        cases = [
            ([_row(True)], ["[m] 🎉 All tests passed!"]),
            ([_row(False)], ["[m] ❌ All tests failed!"]),
            ([_row(False, error=True)], [f"[m] {NO_ANSWER}"]),
            ([], [f"[m] {NO_ANSWER}"]),
            (
                [_row(True), _row(False), _row(False), _row(False, error=True)],
                [
                    "[m] ✅ Total Passed: 1/3 (33.3%)",
                    "[m] ❌ Total Failed: 2/3 (66.7%)",
                ],
            ),
        ]
        for rows, expected in cases:
            logged = []
            with tempfile.TemporaryDirectory() as tmp:
                finish_test_run(rows, tmp, {}, log=logged.append, prefix="[m] ")
            self.assertEqual(logged, expected)

    def test_exit_if_run_failed_skips_non_dict_entries(self):
        from calibrate_agent.llm.run_tests import exit_if_run_failed

        exit_if_run_failed(["error: leaderboard failed", {"metrics": {"total": 1}}])


class TestLeaderboard(unittest.TestCase):
    def _write(self, base, name, metrics):
        d = base / name
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps(metrics))

    def test_likely_range_columns(self):
        from calibrate_agent.llm.tests_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write(base, "m1", {
                "total": 21, "passed": 15, "errored": 1,
                "criteria": {
                    "acc": {"type": "binary", "pass_rate": 75.0,
                            "ci": {"low": 53.1, "high": 88.8}},
                    "flu": {"type": "rating", "mean": 3.5,
                            "ci": {"low": 3.0, "high": 4.0}},
                },
            })
            self._write(base, "m2", {"total": 2, "passed": 1})
            generate_leaderboard(str(base), str(base / "leaderboard"))
            df = pd.read_csv(base / "leaderboard" / "llm_leaderboard.csv")

        self.assertEqual(
            list(df.columns[:7]),
            ["model", "passed", "total", "answered",
             "pass_rate", "pass_rate_low", "pass_rate_high"],
        )
        self.assertEqual(
            [c for c in df.columns if c.startswith(("acc", "flu"))],
            ["acc", "acc_low", "acc_high", "flu", "flu_low", "flu_high"],
        )
        m1 = df[df["model"] == "m1"].iloc[0]
        self.assertEqual(m1["answered"], 20)
        self.assertEqual(m1["passed"] / m1["answered"] * 100, m1["pass_rate"])
        ci = wilson_interval(15, 20)
        self.assertEqual(m1["pass_rate"], 75.0)
        self.assertAlmostEqual(m1["pass_rate_low"], ci["low"])
        self.assertAlmostEqual(m1["pass_rate_high"], ci["high"])
        self.assertEqual((m1["acc_low"], m1["acc_high"]), (53.1, 88.8))
        self.assertEqual((m1["flu"], m1["flu_low"]), (3.5, 3.0))
        m2 = df[df["model"] == "m2"].iloc[0]
        self.assertTrue(pd.isna(m2["acc"]) and pd.isna(m2["acc_low"]))

    def test_write_leaderboard_records_failure(self):
        from unittest.mock import patch

        from calibrate_agent.llm.tests_leaderboard import write_leaderboard

        results: dict = {}
        with patch(
            "calibrate_agent.llm.tests_leaderboard.generate_leaderboard",
            side_effect=RuntimeError("boom"),
        ):
            save_dir = write_leaderboard("/tmp/x", results)
        self.assertEqual(save_dir, str(Path("/tmp/x") / "leaderboard"))
        self.assertEqual(results["leaderboard"], "error: boom")


@unittest.skipIf(os.geteuid() == 0, "root can write to a read-only folder")
class TestLeaderboardNotWritable(unittest.TestCase):
    """A leaderboard that cannot be written fails the command in both benchmark paths."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        model_dir = self.out / "m1"
        model_dir.mkdir()
        (model_dir / "metrics.json").write_text(
            json.dumps({"total": 1, "passed": 1, "errored": 0})
        )
        self.lb = self.out / "leaderboard"
        self.lb.mkdir()
        self.lb.chmod(stat.S_IRUSR | stat.S_IXUSR)

    def tearDown(self):
        self.lb.chmod(stat.S_IRWXU)
        self._tmp.cleanup()

    def _assert_failed(self, out: str, exit_cm):
        self.assertEqual(exit_cm.exception.code, 1)
        self.assertIn("Leaderboard not saved: error:", out)
        self.assertIn("Permission denied", out)
        self.assertNotIn("Leaderboard saved", out)

    def test_benchmark_main_exits_non_zero(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from calibrate_agent.llm import benchmark

        cfg = self.out / "config.json"
        cfg.write_text(json.dumps({"test_cases": []}))
        argv = ["calibrate_agent", "-c", str(cfg), "-m", "m1", "-o", str(self.out)]
        buf = io.StringIO()
        with patch.object(sys, "argv", argv), \
             patch.object(benchmark, "run_model_tests",
                          AsyncMock(return_value={"metrics": {"total": 1, "passed": 1}})), \
             contextlib.redirect_stdout(buf), \
             self.assertRaises(SystemExit) as cm:
            asyncio.run(benchmark.main())
        self._assert_failed(buf.getvalue(), cm)

    def test_cli_agent_benchmark_exits_non_zero(self):
        from unittest.mock import AsyncMock, patch

        from calibrate_agent.cli import main

        cfg = self.out / "config.json"
        cfg.write_text(json.dumps({"agent_url": "http://x", "test_cases": []}))
        argv = ["calibrate_agent", "llm", "-c", str(cfg), "-o", str(self.out),
                "-m", "m1", "--skip-verify"]
        buf = io.StringIO()
        with patch.object(sys, "argv", argv), \
             patch("calibrate_agent.llm.tests.run",
                   AsyncMock(return_value={"m1": {"metrics": {"total": 1, "passed": 1}}})), \
             contextlib.redirect_stdout(buf), \
             self.assertRaises(SystemExit) as cm:
            main()
        self._assert_failed(buf.getvalue(), cm)


class TestAggregateJudgeScores(unittest.TestCase):
    def test_binary_and_rating_means(self):
        from calibrate_agent.judges import aggregate_judge_scores

        evaluators = [
            {"name": "ok", "type": "binary"},
            {"name": "flu", "type": "rating", "scale_min": 1, "scale_max": 5},
        ]
        rows = [
            {"ok": {"match": True}, "flu": {"score": 4}},
            {"ok": {"match": False}, "flu": {"score": 2}},
        ]
        out = aggregate_judge_scores(evaluators, rows)
        self.assertEqual(out["scores"]["ok"], {"type": "binary", "mean": 0.5})
        self.assertEqual(
            out["scores"]["flu"],
            {"type": "rating", "mean": 3.0, "scale_min": 1, "scale_max": 5},
        )
        self.assertEqual(out["score"], 1.75)


if __name__ == "__main__":
    unittest.main()
