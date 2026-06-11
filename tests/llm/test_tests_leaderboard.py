"""
Tests for calibrate/llm/tests_leaderboard.py.

Covers:
- Multi-model leaderboard with per-criterion columns
- Backward compatibility: old-style metrics.json (no "criteria" key)
- Union of criteria across models (missing criteria → NaN column value)
- CSV output produced

Run with:
    python -m pytest tests/test_tests_leaderboard.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from calibrate.llm.tests_leaderboard import generate_leaderboard


def _write_model(base: Path, model_name: str, metrics: dict) -> None:
    model_dir = base / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.json").write_text(json.dumps(metrics))


class TestNumericOrNone(unittest.TestCase):
    def test_numbers_pass_through(self):
        from calibrate.llm._metrics_utils import _numeric_or_none

        self.assertEqual(_numeric_or_none(0.0), 0.0)
        self.assertEqual(_numeric_or_none(3), 3)

    def test_non_numbers_and_bools_become_none(self):
        from calibrate.llm._metrics_utils import _numeric_or_none

        self.assertIsNone(_numeric_or_none(None))
        self.assertIsNone(_numeric_or_none("0.02"))
        self.assertIsNone(_numeric_or_none(True))


class TestLeaderboardMultiCriteria(unittest.TestCase):

    def test_per_criterion_columns_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_model(base, "model-a", {
                "total": 4, "passed": 3,
                "criteria": {
                    "accuracy": {"passed": 2, "total": 2, "pass_rate": 100.0},
                    "tone": {"passed": 1, "total": 2, "pass_rate": 50.0},
                },
            })
            _write_model(base, "model-b", {
                "total": 4, "passed": 4,
                "criteria": {
                    "accuracy": {"passed": 2, "total": 2, "pass_rate": 100.0},
                    "tone": {"passed": 2, "total": 2, "pass_rate": 100.0},
                },
            })

            save_dir = base / "leaderboard"
            generate_leaderboard(str(base), str(save_dir))

            df = pd.read_csv(save_dir / "llm_leaderboard.csv")
            self.assertIn("accuracy", df.columns)
            self.assertIn("tone", df.columns)
            self.assertIn("pass_rate", df.columns)

            # Row order is sorted by model name
            self.assertEqual(list(df["model"]), ["model-a", "model-b"])
            self.assertEqual(
                df.loc[df["model"] == "model-a", "tone"].iloc[0], 50.0
            )
            self.assertEqual(
                df.loc[df["model"] == "model-b", "tone"].iloc[0], 100.0
            )

    def test_union_of_criteria_across_models(self):
        """Model A has criterion X but not Y; model B has Y but not X."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_model(base, "model-a", {
                "total": 1, "passed": 1,
                "criteria": {
                    "accuracy": {"passed": 1, "total": 1, "pass_rate": 100.0},
                },
            })
            _write_model(base, "model-b", {
                "total": 1, "passed": 0,
                "criteria": {
                    "tone": {"passed": 0, "total": 1, "pass_rate": 0.0},
                },
            })

            save_dir = base / "leaderboard"
            generate_leaderboard(str(base), str(save_dir))

            df = pd.read_csv(save_dir / "llm_leaderboard.csv")
            self.assertIn("accuracy", df.columns)
            self.assertIn("tone", df.columns)
            # Missing criterion → NaN
            self.assertTrue(
                pd.isna(df.loc[df["model"] == "model-a", "tone"].iloc[0])
            )
            self.assertTrue(
                pd.isna(df.loc[df["model"] == "model-b", "accuracy"].iloc[0])
            )

    def test_backward_compat_no_criteria_key(self):
        """Old-style metrics.json (just total/passed) should still produce a flat leaderboard."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_model(base, "old-a", {"total": 4, "passed": 3})
            _write_model(base, "old-b", {"total": 4, "passed": 4})

            save_dir = base / "leaderboard"
            generate_leaderboard(str(base), str(save_dir))

            df = pd.read_csv(save_dir / "llm_leaderboard.csv")
            self.assertIn("pass_rate", df.columns)
            self.assertIn("passed", df.columns)
            self.assertIn("total", df.columns)
            # No criterion columns (latency_ms and cost are standard columns, empty here)
            expected_cols = {
                "model", "passed", "total", "pass_rate", "latency_ms", "cost",
            }
            self.assertEqual(set(df.columns), expected_cols)
            # Cost is absent in old-style metrics → empty column
            self.assertTrue(df["cost"].isna().all())

    def test_cost_column_shows_mean_cost(self):
        """The leaderboard surfaces the per-model mean cost, None when absent."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_model(base, "model-a", {
                "total": 2, "passed": 2,
                "cost": {"mean": 0.03, "min": 0.02, "max": 0.04, "count": 2},
            })
            _write_model(base, "model-b", {"total": 2, "passed": 1})

            save_dir = base / "leaderboard"
            generate_leaderboard(str(base), str(save_dir))

            df = pd.read_csv(save_dir / "llm_leaderboard.csv")
            self.assertIn("cost", df.columns)
            by_model = dict(zip(df["model"], df["cost"]))
            self.assertAlmostEqual(by_model["model-a"], 0.03)
            self.assertTrue(pd.isna(by_model["model-b"]))

    def test_skip_leaderboard_folder(self):
        """The existing 'leaderboard' subdirectory inside output_dir should be ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_model(base, "model-a", {"total": 1, "passed": 1})
            # Pre-existing leaderboard folder should be ignored, not processed as a model
            (base / "leaderboard").mkdir()
            (base / "leaderboard" / "metrics.json").write_text(
                json.dumps({"total": 99, "passed": 99})
            )

            save_dir = base / "leaderboard_out"
            generate_leaderboard(str(base), str(save_dir))

            df = pd.read_csv(save_dir / "llm_leaderboard.csv")
            self.assertEqual(list(df["model"]), ["model-a"])

    def test_rating_criterion_shows_mean_score_in_column(self):
        """When a criterion is rating-type, the leaderboard column shows the
        mean score (raw), not a pass_rate percentage."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_model(base, "model-a", {
                "total": 3, "passed": 3,
                "criteria": {
                    "accuracy": {
                        "type": "binary",
                        "passed": 3, "total": 3, "pass_rate": 100.0,
                    },
                    "fluency": {
                        "type": "rating",
                        "mean": 4.2, "min": 3, "max": 5, "count": 3,
                        "scale_min": 1, "scale_max": 5,
                    },
                },
            })
            _write_model(base, "model-b", {
                "total": 3, "passed": 2,
                "criteria": {
                    "accuracy": {
                        "type": "binary",
                        "passed": 2, "total": 3, "pass_rate": 66.67,
                    },
                    "fluency": {
                        "type": "rating",
                        "mean": 3.0, "min": 2, "max": 4, "count": 3,
                        "scale_min": 1, "scale_max": 5,
                    },
                },
            })

            save_dir = base / "leaderboard"
            generate_leaderboard(str(base), str(save_dir))

            df = pd.read_csv(save_dir / "llm_leaderboard.csv")
            # fluency column holds raw mean, not a percentage
            self.assertEqual(
                df.loc[df["model"] == "model-a", "fluency"].iloc[0], 4.2
            )
            self.assertEqual(
                df.loc[df["model"] == "model-b", "fluency"].iloc[0], 3.0
            )
            # accuracy (binary) shows pass_rate
            self.assertEqual(
                df.loc[df["model"] == "model-a", "accuracy"].iloc[0], 100.0
            )

    def test_latency_column(self):
        """latency_ms shows the mean; absent → None/NaN."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_model(base, "model-a", {
                "total": 2, "passed": 2,
                "latency_ms": {"mean": 150, "min": 100, "max": 200, "count": 2},
            })
            # model-b has no latency (e.g. eval-only)
            _write_model(base, "model-b", {"total": 2, "passed": 1})

            save_dir = base / "leaderboard"
            generate_leaderboard(str(base), str(save_dir))

            df = pd.read_csv(save_dir / "llm_leaderboard.csv")
            self.assertIn("latency_ms", df.columns)
            self.assertEqual(
                df.loc[df["model"] == "model-a", "latency_ms"].iloc[0], 150
            )
            self.assertTrue(
                pd.isna(df.loc[df["model"] == "model-b", "latency_ms"].iloc[0])
            )

    def test_empty_output_dir(self):
        """No model subfolders — should not crash, nothing written."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            save_dir = base / "leaderboard"
            # Should print and return, not crash
            generate_leaderboard(str(base), str(save_dir))
            # Leaderboard dir was created but CSV not written
            self.assertTrue(save_dir.exists())
            self.assertFalse((save_dir / "llm_leaderboard.csv").exists())


if __name__ == "__main__":
    unittest.main()
