"""
Tests for calibrate/tts/leaderboard.py.

Run with:
    python -m unittest tests.tts.test_leaderboard -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import openpyxl  # noqa: F401

from calibrate.tts.leaderboard import generate_leaderboard as generate_tts_leaderboard


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


class TestTTSLeaderboard(unittest.TestCase):

    def test_default_single_evaluator_and_ttfb(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "openai", {
                "pronunciation": {"type": "binary", "mean": 0.95},
                "ttfb": {"p50": 0.4, "p95": 0.45, "p99": 0.46, "count": 2},
            }, results_rows=[
                {"id": 1, "text": "hi", "pronunciation": True, "ttfb": 0.4},
            ])
            _write_provider(base, "elevenlabs", {
                "pronunciation": {"type": "binary", "mean": 0.8},
                "ttfb": {"p50": 0.3, "p95": 0.31, "p99": 0.31, "count": 2},
            }, results_rows=[
                {"id": 1, "text": "hi", "pronunciation": True, "ttfb": 0.3},
            ])

            save_dir = base / "leaderboard"
            generate_tts_leaderboard(str(base), str(save_dir))

            xlsx = save_dir / "tts_leaderboard.xlsx"
            self.assertTrue(xlsx.exists())
            summary = pd.read_excel(xlsx, sheet_name="summary")
            self.assertIn("pronunciation", summary.columns)
            self.assertIn("ttfb_p50", summary.columns)
            self.assertIn("ttfb_p95", summary.columns)
            self.assertIn("ttfb_p99", summary.columns)

    def test_multi_criterion_metrics_surface_dynamically(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "provider-a", {
                "intelligibility": {"type": "binary", "mean": 0.9},
                "pronunciation": {"type": "binary", "mean": 0.85},
                "ttfb": {"p50": 0.4, "p95": 0.45, "p99": 0.46, "count": 2},
            })

            save_dir = base / "leaderboard"
            generate_tts_leaderboard(str(base), str(save_dir))

            xlsx = save_dir / "tts_leaderboard.xlsx"
            summary = pd.read_excel(xlsx, sheet_name="summary")
            self.assertIn("intelligibility", summary.columns)
            self.assertIn("pronunciation", summary.columns)
            self.assertIn("ttfb_p50", summary.columns)


if __name__ == "__main__":
    unittest.main()
