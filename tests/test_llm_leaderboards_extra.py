"""Cover branches in llm simulation_leaderboard.py and tests_leaderboard.py."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _write(base: Path, name: str, metrics):
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    if metrics is not None:
        (d / "metrics.json").write_text(metrics if isinstance(metrics, str) else json.dumps(metrics))


class TestSimulationLeaderboard(unittest.TestCase):
    def test_missing_base_dir_raises(self):
        from calibrate.llm.simulation_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                generate_leaderboard(str(Path(tmp) / "missing"), str(Path(tmp) / "save"))

    def test_no_model_dirs(self):
        from calibrate.llm.simulation_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            generate_leaderboard(tmp, str(Path(tmp) / "lb"))

    def test_missing_metrics_json_skipped(self):
        from calibrate.llm.simulation_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "m1").mkdir()  # no metrics.json
            generate_leaderboard(tmp, str(base / "lb"))

    def test_invalid_json_skipped(self):
        from calibrate.llm.simulation_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write(base, "m1", "{not json")
            generate_leaderboard(tmp, str(base / "lb"))

    def test_binary_rating_legacy_metrics(self):
        from calibrate.llm.simulation_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write(base, "m1", {
                "task_complete": {"type": "binary", "mean": 0.75},
                "helpfulness": {"type": "rating", "mean": 4.0, "scale_min": 1, "scale_max": 5},
                "no_scale": {"type": "rating", "mean": 0, "scale_min": 1, "scale_max": 1},
                "legacy": 0.8,
            })
            generate_leaderboard(tmp, str(base / "lb"))
            csv = base / "lb" / "simulation_leaderboard.csv"
            self.assertTrue(csv.exists())

    def test_main_cli(self):
        from calibrate.llm import simulation_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write(base, "m1", {"x": {"type": "binary", "mean": 0.5}})
            argv = ["leaderboard.py", "-o", tmp, "-s", str(base / "lb")]
            with patch.object(sys, "argv", argv):
                simulation_leaderboard.main()


class TestTestsLeaderboard(unittest.TestCase):
    def test_missing_base_dir_raises(self):
        from calibrate.llm.tests_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                generate_leaderboard(str(Path(tmp) / "missing"), str(Path(tmp) / "save"))

    def test_no_model_dirs(self):
        from calibrate.llm.tests_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            generate_leaderboard(tmp, str(Path(tmp) / "lb"))

    def test_missing_and_invalid(self):
        from calibrate.llm.tests_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "m1").mkdir()
            _write(base, "m2", "{bad json")
            generate_leaderboard(tmp, str(base / "lb"))

    def test_full_leaderboard(self):
        from calibrate.llm.tests_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write(base, "m1", {
                "passed": 2, "total": 3,
                "criteria": {
                    "accuracy": {"type": "binary", "pass_rate": 0.66},
                    "tone": {"type": "rating", "mean": 4.0},
                },
            })
            _write(base, "m2", {
                "passed": 0, "total": 0,  # tests _to_percent total<=0 branch
                "criteria": {"missing_one": {"type": "binary", "pass_rate": 0.0}},
            })
            generate_leaderboard(tmp, str(base / "lb"))
            self.assertTrue((base / "lb" / "llm_leaderboard.csv").exists())

    def test_main_cli(self):
        from calibrate.llm import tests_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write(base, "m1", {"passed": 1, "total": 1, "criteria": {}})
            argv = ["leaderboard.py", "-o", tmp, "-s", str(base / "lb")]
            with patch.object(sys, "argv", argv):
                tests_leaderboard.main()


if __name__ == "__main__":
    unittest.main()
