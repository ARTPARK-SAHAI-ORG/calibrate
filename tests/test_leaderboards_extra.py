"""Cover gaps in stt/tts leaderboard modules and llm leaderboard modules."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import openpyxl  # noqa


def _write_provider(base: Path, name: str, metrics=None, results=None):
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    if metrics is not None:
        (d / "metrics.json").write_text(json.dumps(metrics))
    if results is not None:
        pd.DataFrame(results).to_csv(d / "results.csv", index=False)


class TestSTTLeaderboardExtra(unittest.TestCase):
    def test_output_dir_missing_raises(self):
        from calibrate.stt.leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp) / "save"
            with self.assertRaises(FileNotFoundError):
                generate_leaderboard(str(Path(tmp) / "missing"), str(save_dir))

    def test_no_provider_folders(self):
        from calibrate.stt.leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            result = generate_leaderboard(tmp)
            self.assertTrue(Path(result).exists())

    def test_missing_metrics_json_warning(self):
        from calibrate.stt.leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d = base / "p"
            d.mkdir()
            # no metrics.json, no results.csv
            generate_leaderboard(tmp)

    def test_legacy_metrics_format(self):
        from calibrate.stt.leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(
                base,
                "p1",
                metrics=[
                    {"metric_name": "wer", "mean": 0.1},
                    {"some_other": 0.5},
                ],
            )
            _write_provider(
                base,
                "p2",
                metrics={"metric_name": "wer", "mean": 0.2},  # dict form
            )
            generate_leaderboard(tmp)

    def test_unique_sheet_name_collision(self):
        from calibrate.stt.leaderboard import _unique_sheet_name

        existing = set()
        a = _unique_sheet_name("a" * 40, existing)
        b = _unique_sheet_name("a" * 40, existing)
        self.assertNotEqual(a, b)

    def test_main_cli(self):
        from calibrate.stt import leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "p1", metrics={"wer": 0.1})
            argv = ["leaderboard.py", "-o", tmp, "-s", str(base / "lb")]
            with patch.object(sys, "argv", argv):
                leaderboard.main()
            self.assertTrue((base / "lb" / "stt_leaderboard.xlsx").exists())


class TestTTSLeaderboardExtra(unittest.TestCase):
    def test_output_dir_missing_raises(self):
        from calibrate.tts.leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp) / "save"
            with self.assertRaises(FileNotFoundError):
                generate_leaderboard(str(Path(tmp) / "missing"), str(save_dir))

    def test_no_provider_folders(self):
        from calibrate.tts.leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            result = generate_leaderboard(tmp)
            self.assertTrue(Path(result).exists())

    def test_missing_files(self):
        from calibrate.tts.leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "p").mkdir()
            generate_leaderboard(tmp)

    def test_legacy_metrics_format(self):
        from calibrate.tts.leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(
                base,
                "p1",
                metrics=[
                    {"metric_name": "ttfb", "mean": 1.2},
                    {"random": 0.5},
                ],
            )
            _write_provider(base, "p2", metrics={"metric_name": "ttfb", "mean": 1.3})
            generate_leaderboard(tmp)

    def test_main_cli(self):
        from calibrate.tts import leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "p1", metrics={"ttfb": {"mean": 1.0}})
            argv = ["leaderboard.py", "-o", tmp]
            with patch.object(sys, "argv", argv):
                leaderboard.main()


class TestSimulationLeaderboardExtra(unittest.TestCase):
    def test_no_runs(self):
        from calibrate.llm.simulation_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            generate_leaderboard(tmp, str(Path(tmp) / "lb"))


class TestTestsLeaderboardExtra(unittest.TestCase):
    def test_no_runs(self):
        from calibrate.llm.tests_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            generate_leaderboard(tmp, str(Path(tmp) / "lb"))


if __name__ == "__main__":
    unittest.main()
