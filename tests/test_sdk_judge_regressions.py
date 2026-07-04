"""
Regression tests for SDK-path judge behavior.

Run with:
    python -m pytest tests/test_sdk_judge_regressions.py -v
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Bug 1: SDK simulation metrics.json must carry `type`
# ---------------------------------------------------------------------------


class TestSDKSimulationMetricsCarryType(unittest.IsolatedAsyncioTestCase):

    async def test_rating_criterion_aggregate_gets_type_field(self):
        """After `calibrate_agent.llm.simulations.run(...)` completes, the saved
        metrics.json should include `type: "rating"` for each rating
        criterion so the leaderboard doesn't multiply its mean by 100."""
        from calibrate_agent.llm import simulations

        # Fake `run_single_simulation_task` that returns the shape the SDK expects:
        # (simulation_metrics_row, evaluation_results)
        # Each evaluation_result carries {name, type, value, reasoning}.
        async def fake_task(
            semaphore, config, persona_index, user_persona, scenario_index,
            scenario, output_dir, args, agent=None,
        ):
            async with semaphore:
                sim_name = f"sim_{persona_index}_{scenario_index}"
                eval_results = [
                    {"name": "tool_usage", "type": "binary", "value": 1.0, "reasoning": "ok"},
                    {"name": "fluency", "type": "rating", "value": 4.0, "reasoning": "good"},
                ]
                sim_metrics = {"name": sim_name, "tool_usage": 1.0, "fluency": 4.0}
                return sim_metrics, eval_results

        with tempfile.TemporaryDirectory() as tmp, \
             patch(
                 "calibrate_agent.llm.run_simulation.run_single_simulation_task",
                 side_effect=fake_task,
             ):
            await simulations.run(
                system_prompt="...",
                tools=[],
                personas=[{"characteristics": "friendly", "language": "english"}],
                scenarios=[{"description": "basic inquiry"}],
                evaluators=[
                    {
                        "name": "tool_usage",
                        "system_prompt": "calls tool",
                        "judge_model": "openai/gpt-5.2",
                    },
                    {
                        "name": "fluency",
                        "system_prompt": "rate fluency",
                        "judge_model": "openai/gpt-5.2",
                        "type": "rating",
                        "scale_min": 1,
                        "scale_max": 5,
                    },
                ],
                output_dir=tmp,
                model="gpt-4.1",
                provider="openai",
            )

            # Find the model subfolder that was created (single model path uses _flat_output=True)
            with open(os.path.join(tmp, "metrics.json")) as f:
                metrics = json.load(f)

        self.assertIn("tool_usage", metrics)
        self.assertIn("fluency", metrics)
        self.assertEqual(metrics["tool_usage"]["type"], "binary")
        self.assertEqual(metrics["fluency"]["type"], "rating")
        # Raw means are preserved (no 100x scaling at this stage)
        self.assertAlmostEqual(metrics["fluency"]["mean"], 4.0)
        self.assertAlmostEqual(metrics["tool_usage"]["mean"], 1.0)


# ---------------------------------------------------------------------------
# Bug 3: simulation_leaderboard `overall` must be unit-consistent across
# mixed binary/rating criteria
# ---------------------------------------------------------------------------


class TestSimulationLeaderboardOverallUnitConsistent(unittest.TestCase):

    def test_overall_uses_normalized_values_for_rating(self):
        """Binary % and rating raw means can't be averaged directly.
        The overall column should be computed from normalized (0-100) values:
        for rating, (mean - scale_min) / (scale_max - scale_min) * 100.
        """
        import pathlib
        from calibrate_agent.llm.simulation_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            # Model A: binary 80%, rating mean 4/5 → normalized 75%.
            # Expected overall (normalized mean) = (80 + 75) / 2 = 77.5
            (base / "model-a").mkdir()
            (base / "model-a" / "metrics.json").write_text(
                json.dumps({
                    "accuracy": {"type": "binary", "mean": 0.8},
                    "fluency": {
                        "type": "rating",
                        "mean": 4.0,
                        "scale_min": 1,
                        "scale_max": 5,
                    },
                })
            )
            # Model B: binary 100%, rating mean 1/5 → normalized 0%.
            # Expected overall = (100 + 0) / 2 = 50
            (base / "model-b").mkdir()
            (base / "model-b" / "metrics.json").write_text(
                json.dumps({
                    "accuracy": {"type": "binary", "mean": 1.0},
                    "fluency": {
                        "type": "rating",
                        "mean": 1.0,
                        "scale_min": 1,
                        "scale_max": 5,
                    },
                })
            )

            save_dir = base / "leaderboard"
            generate_leaderboard(str(base), str(save_dir))

            import pandas as pd
            df = pd.read_csv(save_dir / "simulation_leaderboard.csv")

            row_a = df[df["model"] == "model-a"].iloc[0]
            row_b = df[df["model"] == "model-b"].iloc[0]

            # Display values stay on their own scale (binary %, rating raw mean)
            self.assertAlmostEqual(row_a["accuracy"], 80.0)
            self.assertAlmostEqual(row_a["fluency"], 4.0)
            self.assertAlmostEqual(row_b["accuracy"], 100.0)
            self.assertAlmostEqual(row_b["fluency"], 1.0)

            # Overall is computed from normalized 0-100 values, so it's
            # unit-consistent. Adding the rating criterion can no longer
            # arbitrarily reorder models via unit-mixing.
            self.assertAlmostEqual(row_a["overall"], 77.5)
            self.assertAlmostEqual(row_b["overall"], 50.0)


# ---------------------------------------------------------------------------
# Bug 5: simulation metrics.json must persist scale_min/scale_max for rating
# ---------------------------------------------------------------------------


class TestSimulationMetricsPersistScaleBounds(unittest.IsolatedAsyncioTestCase):

    async def test_sdk_simulation_writes_scale_for_rating_criterion(self):
        """The simulation leaderboard normalizes rating means via
        (mean - scale_min) / (scale_max - scale_min) * 100. Without the
        bounds in metrics.json it falls back to 0..1, treating a 1-5 mean
        of 4.0 as 400% in the overall column. The SDK simulation writer
        must persist scale_min/scale_max for rating criteria.
        """
        from calibrate_agent.llm import simulations

        async def fake_task(
            semaphore, config, persona_index, user_persona, scenario_index,
            scenario, output_dir, args, agent=None,
        ):
            async with semaphore:
                eval_results = [
                    {
                        "name": "fluency",
                        "type": "rating",
                        "value": 4.0,
                        "reasoning": "good",
                        "scale_min": 1,
                        "scale_max": 5,
                    },
                ]
                sim_metrics = {"name": f"sim_{persona_index}_{scenario_index}", "fluency": 4.0}
                return sim_metrics, eval_results

        with tempfile.TemporaryDirectory() as tmp, \
             patch(
                 "calibrate_agent.llm.run_simulation.run_single_simulation_task",
                 side_effect=fake_task,
             ):
            await simulations.run(
                system_prompt="...",
                tools=[],
                personas=[{"characteristics": "x", "language": "english"}],
                scenarios=[{"description": "x"}],
                evaluators=[{
                    "name": "fluency",
                    "system_prompt": "rate fluency",
                    "judge_model": "openai/gpt-5.2",
                    "type": "rating",
                    "scale_min": 1,
                    "scale_max": 5,
                }],
                output_dir=tmp,
                model="gpt-4.1",
                provider="openai",
            )

            with open(os.path.join(tmp, "metrics.json")) as f:
                metrics = json.load(f)

        self.assertEqual(metrics["fluency"]["type"], "rating")
        self.assertEqual(metrics["fluency"]["scale_min"], 1)
        self.assertEqual(metrics["fluency"]["scale_max"], 5)
        self.assertAlmostEqual(metrics["fluency"]["mean"], 4.0)


class TestSimulationLeaderboardNormalizesWithPersistedBounds(unittest.TestCase):
    """End-to-end: a metrics.json written by the SDK with scale bounds
    should produce the right normalized overall in the leaderboard."""

    def test_overall_uses_persisted_bounds(self):
        import pathlib
        import pandas as pd
        from calibrate_agent.llm.simulation_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            (base / "model-a").mkdir()
            (base / "model-a" / "metrics.json").write_text(json.dumps({
                "fluency": {
                    "type": "rating",
                    "mean": 4.0,
                    "scale_min": 1,
                    "scale_max": 5,
                },
            }))

            save_dir = base / "leaderboard"
            generate_leaderboard(str(base), str(save_dir))

            df = pd.read_csv(save_dir / "simulation_leaderboard.csv")
            row = df[df["model"] == "model-a"].iloc[0]
            # Rating display column shows raw mean
            self.assertAlmostEqual(row["fluency"], 4.0)
            # Overall normalized: (4-1)/(5-1)*100 = 75.0
            # Without scale bounds the fallback would give 400.
            self.assertAlmostEqual(row["overall"], 75.0)


if __name__ == "__main__":
    unittest.main()
