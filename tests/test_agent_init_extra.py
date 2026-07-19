"""Cover calibrate_agent/agent/__init__.py public API via mocks."""

import tempfile
import unittest
from unittest.mock import patch, AsyncMock


class TestConfigDataclasses(unittest.TestCase):
    def test_stt_config_to_dict(self):
        from calibrate_agent.agent import STTConfig

        self.assertEqual(
            STTConfig(provider="deepgram").to_dict(), {"provider": "deepgram"}
        )

    def test_tts_config_with_voice(self):
        from calibrate_agent.agent import TTSConfig

        self.assertEqual(
            TTSConfig(provider="x", voice_id="v1").to_dict(),
            {"provider": "x", "voice_id": "v1"},
        )

    def test_tts_config_no_voice(self):
        from calibrate_agent.agent import TTSConfig

        self.assertEqual(TTSConfig(provider="x").to_dict(), {"provider": "x"})

    def test_llm_config(self):
        from calibrate_agent.agent import LLMConfig

        self.assertEqual(
            LLMConfig(provider="openrouter", model="m").to_dict(),
            {"provider": "openrouter", "model": "m"},
        )


class TestSimulationRun(unittest.IsolatedAsyncioTestCase):
    async def test_simulation_run_happy_path(self):
        from calibrate_agent.agent import simulation, STTConfig, TTSConfig, LLMConfig

        fake_result = (
            {"id": "p0_s0", "elapsed": 1.0},
            [
                {
                    "name": "criterion_a",
                    "value": 1.0,
                    "type": "binary",
                    "evaluator_id": "ev1",
                },
                {
                    "name": "rating_b",
                    "value": 4,
                    "type": "rating",
                    "scale_min": 1,
                    "scale_max": 5,
                },
            ],
            {"score": 0.85},
        )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "calibrate_agent.agent.run_simulation.run_single_simulation_task",
                AsyncMock(return_value=fake_result),
            ),
        ):
            result = await simulation.run(
                system_prompt="sp",
                tools=[],
                personas=[
                    {"characteristics": "p", "gender": "male", "language": "english"}
                ],
                scenarios=[{"description": "s"}],
                evaluators=[
                    {"name": "criterion_a", "system_prompt": "x", "judge_model": "m"}
                ],
                output_dir=tmp,
                stt=STTConfig(),
                tts=TTSConfig(),
                llm=LLMConfig(),
            )

        self.assertEqual(result["status"], "completed")
        self.assertIn("criterion_a", result["metrics"])
        self.assertIn("rating_b", result["metrics"])
        self.assertIn("stt_llm_judge", result["metrics"])
        self.assertEqual(result["metrics"]["rating_b"]["scale_min"], 1)

    async def test_simulation_run_failure_aggregates(self):
        from calibrate_agent.agent import simulation

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "calibrate_agent.agent.run_simulation.run_single_simulation_task",
                AsyncMock(side_effect=RuntimeError("simfail")),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await simulation.run(
                    system_prompt="sp",
                    tools=[],
                    personas=[
                        {
                            "characteristics": "p",
                            "gender": "male",
                            "language": "english",
                        }
                    ],
                    scenarios=[{"description": "s"}],
                    evaluators=[
                        {"name": "c", "system_prompt": "x", "judge_model": "m"}
                    ],
                    output_dir=tmp,
                )
        self.assertIn("simulation(s) failed", str(ctx.exception))

    async def test_simulation_run_handles_none_result(self):
        from calibrate_agent.agent import simulation

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "calibrate_agent.agent.run_simulation.run_single_simulation_task",
                AsyncMock(return_value=None),
            ),
        ):
            result = await simulation.run(
                system_prompt="sp",
                tools=[],
                personas=[
                    {"characteristics": "p", "gender": "male", "language": "english"}
                ],
                scenarios=[{"description": "s"}],
                evaluators=[{"name": "c", "system_prompt": "x", "judge_model": "m"}],
                output_dir=tmp,
            )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"], {})

    async def test_simulation_run_single_delegates(self):
        from calibrate_agent.agent import simulation

        fake_inner = AsyncMock(return_value={"status": "completed"})
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("calibrate_agent.agent.run_simulation.run_simulation", fake_inner),
        ):
            result = await simulation.run_single(
                system_prompt="sp",
                language="english",
                gender="female",
                evaluators=[{"name": "c", "system_prompt": "x", "judge_model": "m"}],
                output_dir=tmp,
            )
        self.assertEqual(result["status"], "completed")
        fake_inner.assert_called_once()


class TestCalibrateInit(unittest.TestCase):
    def test_version_fallback(self):
        # The fallback "0.0.0-dev" branch only triggers if the installed
        # package is not findable. Force PackageNotFoundError to exercise the branch.
        import importlib
        from importlib.metadata import PackageNotFoundError

        with patch(
            "importlib.metadata.version",
            side_effect=PackageNotFoundError("calibrate-agent"),
        ):
            import calibrate_agent

            importlib.reload(calibrate_agent)
            self.assertEqual(calibrate_agent.__version__, "0.0.0-dev")

        # Restore module state
        importlib.reload(calibrate_agent)

    def test_submodules_are_lazy(self):
        # Importing calibrate_agent must NOT eagerly import the heavy submodules,
        # so a broken optional provider SDK (e.g. deepgram) inside
        # calibrate_agent.stt can't break `import calibrate_agent` itself.
        import sys
        import importlib

        for name in (
            "calibrate_agent",
            "calibrate_agent.stt",
            "calibrate_agent.tts",
            "calibrate_agent.llm",
            "calibrate_agent.agent",
        ):
            sys.modules.pop(name, None)

        import calibrate_agent

        for name in (
            "calibrate_agent.stt",
            "calibrate_agent.tts",
            "calibrate_agent.llm",
            "calibrate_agent.agent",
        ):
            self.assertNotIn(name, sys.modules)

        # Submodules remain accessible as attributes (imported on demand).
        self.assertIs(
            calibrate_agent.stt, importlib.import_module("calibrate_agent.stt")
        )
        self.assertIn("stt", dir(calibrate_agent))

    def test_unknown_attribute_raises(self):
        import calibrate_agent

        with self.assertRaises(AttributeError):
            calibrate_agent.does_not_exist  # noqa: B018


if __name__ == "__main__":
    unittest.main()
