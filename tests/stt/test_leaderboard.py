"""
Tests for calibrate/stt/leaderboard.py.

Covers:
- Dynamic metric discovery from metrics.json (no hardcoded metric list)
- Excel workbook is produced with summary sheet + per-provider sheets
- Handles single-evaluator and multi-evaluator metrics.json
- Skips the `leaderboard` subdir inside output_dir

Run with:
    python -m unittest tests.stt.test_leaderboard -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import openpyxl  # noqa: F401 — ensures xlsx reading works

from calibrate.stt.leaderboard import generate_leaderboard as generate_stt_leaderboard


def _write_provider(
    base: Path,
    provider: str,
    metrics: dict,
    results_rows: list[dict] | None = None,
) -> None:
    provider_dir = base / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    (provider_dir / "metrics.json").write_text(json.dumps(metrics))
    if results_rows is not None:
        pd.DataFrame(results_rows).to_csv(
            provider_dir / "results.csv", index=False
        )


class TestSTTLeaderboard(unittest.TestCase):

    def test_default_single_evaluator_produces_score_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "deepgram", {
                "wer": 0.1,
                "semantic_match": {"type": "binary", "mean": 0.85},
            }, results_rows=[
                {"id": 1, "gt": "hello", "pred": "hello", "semantic_match": True},
            ])
            _write_provider(base, "google", {
                "wer": 0.2,
                "semantic_match": {"type": "binary", "mean": 0.75},
            }, results_rows=[
                {"id": 1, "gt": "hello", "pred": "hallo", "semantic_match": False},
            ])

            save_dir = base / "leaderboard"
            generate_stt_leaderboard(str(base), str(save_dir))

            # Excel workbook exists with summary sheet
            xlsx = save_dir / "stt_leaderboard.xlsx"
            self.assertTrue(xlsx.exists())
            summary = pd.read_excel(xlsx, sheet_name="summary")
            self.assertIn("wer", summary.columns)
            self.assertIn("semantic_match", summary.columns)
            self.assertEqual(set(summary["run"]), {"deepgram", "google"})

    def test_custom_criterion_metrics_surface_dynamically(self):
        """A provider with custom criterion `semantic_match` should produce a
        column in the summary."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "provider-a", {
                "wer": 0.05,
                "semantic_match": {"type": "binary", "mean": 0.9},
                "completeness": {"type": "binary", "mean": 0.7},
            })

            save_dir = base / "leaderboard"
            generate_stt_leaderboard(str(base), str(save_dir))

            xlsx = save_dir / "stt_leaderboard.xlsx"
            summary = pd.read_excel(xlsx, sheet_name="summary")
            self.assertIn("semantic_match", summary.columns)
            self.assertIn("completeness", summary.columns)

    def test_skips_existing_leaderboard_folder(self):
        """A pre-existing `leaderboard` subdir under output_dir must not be
        treated as a provider (hardcoded skip in the leaderboard code)."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "provider-x", {
                "wer": 0.1,
                "semantic_match": {"type": "binary", "mean": 1.0},
            }, results_rows=[
                {"id": 1, "gt": "hi", "pred": "hi", "semantic_match": True},
            ])
            # Pre-existing leaderboard directory — must be skipped
            (base / "leaderboard").mkdir()
            (base / "leaderboard" / "metrics.json").write_text(
                json.dumps({"wer": 999.0})
            )

            # Save inside the default location (base/leaderboard)
            generate_stt_leaderboard(str(base))

            xlsx = base / "leaderboard" / "stt_leaderboard.xlsx"
            summary = pd.read_excel(xlsx, sheet_name="summary")
            self.assertEqual(list(summary["run"]), ["provider-x"])


if __name__ == "__main__":
    unittest.main()
