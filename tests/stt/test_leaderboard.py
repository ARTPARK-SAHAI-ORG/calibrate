"""
Tests for calibrate_agent/stt/leaderboard.py.

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

from calibrate_agent.stt.leaderboard import generate_leaderboard as generate_stt_leaderboard


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

    def test_cost_metrics_surface_as_scalar_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "openai", {
                "wer": 0.1,
                "semantic_match": {"type": "binary", "mean": 0.9},
                "cost": {
                    "audio_minutes": 2.0,
                    "cost_per_minute_currency": 0.006,
                    "cost_usd": 0.012,
                },
            })

            save_dir = base / "leaderboard"
            generate_stt_leaderboard(str(base), str(save_dir))

            summary = pd.read_excel(save_dir / "stt_leaderboard.xlsx", sheet_name="summary")
            self.assertIn("cost_usd", summary.columns)
            self.assertNotIn("cost_per_minute_currency", summary.columns)
            self.assertNotIn("audio_minutes", summary.columns)
            self.assertEqual(float(summary.iloc[0]["cost_usd"]), 0.012)

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

    def test_regenerate_archives_previous_summary_to_past_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(
                base,
                "deepgram",
                {"wer": 0.1, "sarvam_intent_score": 0.9},
                results_rows=[{"id": 1, "gt": "a", "pred": "a"}],
            )
            save_dir = base / "leaderboard"
            generate_stt_leaderboard(str(base), str(save_dir))

            # Update metrics and regenerate — prior summary should land on past_runs.
            _write_provider(
                base,
                "deepgram",
                {"wer": 0.2, "sarvam_llm_wer": 0.15},
                results_rows=[{"id": 1, "gt": "a", "pred": "a"}],
            )
            generate_stt_leaderboard(str(base), str(save_dir))

            xlsx = save_dir / "stt_leaderboard.xlsx"
            summary = pd.read_excel(xlsx, sheet_name="summary")
            past = pd.read_excel(xlsx, sheet_name="past_runs")
            self.assertIn("sarvam_llm_wer", summary.columns)
            self.assertIn("archived_at", past.columns)
            self.assertIn("sarvam_intent_score", past.columns)
            self.assertAlmostEqual(float(past["sarvam_intent_score"].iloc[0]), 0.9)

    def test_past_runs_missing_summary_keeps_existing_past(self):
        from calibrate_agent.stt import leaderboard as LB

        with tempfile.TemporaryDirectory() as tmp:
            xlsx = Path(tmp) / "stt_leaderboard.xlsx"
            existing = pd.DataFrame(
                [{"archived_at": "2020-01-01T00:00:00Z", "run": "old", "wer": 0.3}]
            )
            with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
                existing.to_excel(writer, sheet_name="past_runs", index=False)
                # Intentionally no summary sheet.

            out = LB._load_and_extend_past_runs(xlsx, pd.DataFrame([{"run": "new"}]))
            self.assertEqual(list(out["run"]), ["old"])
            self.assertAlmostEqual(float(out["wer"].iloc[0]), 0.3)

    def test_past_runs_empty_summary_keeps_existing_past(self):
        from calibrate_agent.stt import leaderboard as LB

        with tempfile.TemporaryDirectory() as tmp:
            xlsx = Path(tmp) / "stt_leaderboard.xlsx"
            existing = pd.DataFrame(
                [{"archived_at": "2020-01-01T00:00:00Z", "run": "old", "wer": 0.3}]
            )
            with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
                existing.to_excel(writer, sheet_name="past_runs", index=False)
                pd.DataFrame().to_excel(writer, sheet_name="summary", index=False)

            out = LB._load_and_extend_past_runs(xlsx, pd.DataFrame([{"run": "new"}]))
            self.assertEqual(list(out["run"]), ["old"])

    def test_write_workbook_includes_stub_for_empty_provider_results(self):
        from calibrate_agent.stt import leaderboard as LB

        with tempfile.TemporaryDirectory() as tmp:
            xlsx = Path(tmp) / "stt_leaderboard.xlsx"
            LB._write_leaderboard_workbook(
                pd.DataFrame([{"run": "deepgram", "wer": 0.1}]),
                {"deepgram": pd.DataFrame()},
                xlsx,
                past_runs_df=pd.DataFrame(
                    [{"archived_at": "t", "run": "deepgram", "wer": 0.2}]
                ),
            )
            stub = pd.read_excel(xlsx, sheet_name="deepgram")
            self.assertIn("No results.csv found", stub["info"].iloc[0])
            past = pd.read_excel(xlsx, sheet_name="past_runs")
            self.assertEqual(list(past["run"]), ["deepgram"])

    def test_unique_sheet_name_avoids_collisions(self):
        from calibrate_agent.stt.leaderboard import _unique_sheet_name

        existing = {"deepgram"}
        self.assertEqual(_unique_sheet_name("deepgram", existing), "deepgram_1")
        self.assertEqual(_unique_sheet_name("a[]:*?/\\b", set()), "a_______b")
        self.assertEqual(_unique_sheet_name("   ", set()), "run")

    def test_past_runs_generic_exception_on_past_sheet_yields_empty_past(self):
        from unittest.mock import patch

        from calibrate_agent.stt import leaderboard as LB

        with tempfile.TemporaryDirectory() as tmp:
            xlsx = Path(tmp) / "stt_leaderboard.xlsx"
            with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
                pd.DataFrame([{"run": "deepgram", "wer": 0.1}]).to_excel(
                    writer, sheet_name="summary", index=False
                )

            real_read = pd.read_excel

            def fake_read_excel(*args, **kwargs):
                sheet = kwargs.get("sheet_name")
                if sheet == "past_runs":
                    raise OSError("corrupt sheet")
                return real_read(*args, **kwargs)

            with patch.object(LB.pd, "read_excel", side_effect=fake_read_excel):
                out = LB._load_and_extend_past_runs(
                    xlsx, pd.DataFrame([{"run": "new"}])
                )
            self.assertIn("archived_at", out.columns)
            self.assertEqual(list(out["run"]), ["deepgram"])

    def test_leaderboard_main_cli(self):
        import sys
        from unittest.mock import patch

        from calibrate_agent.stt import leaderboard as LB

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "deepgram", {"wer": 0.1})
            save = base / "lb"
            with patch.object(
                sys,
                "argv",
                ["leaderboard.py", "-o", str(base), "-s", str(save)],
            ):
                LB.main()
            self.assertTrue((save / "stt_leaderboard.xlsx").exists())


if __name__ == "__main__":
    unittest.main()
