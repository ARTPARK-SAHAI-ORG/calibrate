"""Cover llm/__init__.py public API entry points via heavy mocking."""

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


class TestLLMTestsRun(unittest.IsolatedAsyncioTestCase):
    async def test_run_single_model_default(self):
        from calibrate.llm import tests

        fake_test_result = {
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": True, "reasoning": "ok", "judge_results": {}},
        }

        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_tests.run_test", AsyncMock(return_value=fake_test_result)):
            result = await tests.run(
                test_cases=[{
                    "id": "tc1",
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "respond hello"},
                }],
                system_prompt="sp",
                tools=[],
                output_dir=tmp,
                model="gpt-4.1",
                provider="openrouter",
            )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"]["passed"], 1)

    async def test_run_multi_model_parallel(self):
        from calibrate.llm import tests

        fake_test_result = {
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": True, "judge_results": {}},
        }

        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_tests.run_test", AsyncMock(return_value=fake_test_result)):
            result = await tests.run(
                test_cases=[{
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "respond hello"},
                }],
                output_dir=tmp,
                models=["m1", "m2"],
                provider="openrouter",
            )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(set(result["models"].keys()), {"m1", "m2"})

    async def _capture_run_semaphore(self, *, max_parallel=None, env=None, agent=False):
        """Run ``tests.run`` (multi-model) and capture the model-level Semaphore size."""
        import asyncio
        import calibrate.llm as llm_pkg
        from calibrate.llm import tests

        # The model-level semaphore is created first in tests.run, before any
        # per-model test-case-level semaphore, so values[0] is the model-level size.
        values = []
        real_semaphore = asyncio.Semaphore

        def fake_semaphore(value):
            values.append(value)
            return real_semaphore(value)

        fake_test_result = {
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": True, "judge_results": {}},
        }
        env = env or {}
        fake_agent = MagicMock() if agent else None
        target = "run_test_external" if agent else "run_test"
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict("os.environ", env, clear=False), \
             patch(f"calibrate.llm.run_tests.{target}",
                   AsyncMock(return_value=fake_test_result)), \
             patch.object(llm_pkg.asyncio, "Semaphore", side_effect=fake_semaphore):
            if "CALIBRATE_LLM_BENCHMARK_PARALLEL" not in env:
                os.environ.pop("CALIBRATE_LLM_BENCHMARK_PARALLEL", None)
            await tests.run(
                test_cases=[{
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "x"},
                }],
                output_dir=tmp,
                models=["m1", "m2"],
                max_parallel=max_parallel,
                agent=fake_agent,
            )
        return values[0]

    async def test_run_model_parallel_env_var(self):
        value = await self._capture_run_semaphore(
            env={"CALIBRATE_LLM_BENCHMARK_PARALLEL": "5"}
        )
        self.assertEqual(value, 5)

    async def test_run_model_parallel_arg_overrides_env(self):
        value = await self._capture_run_semaphore(
            max_parallel=8, env={"CALIBRATE_LLM_BENCHMARK_PARALLEL": "5"}
        )
        self.assertEqual(value, 8)

    async def test_run_model_parallel_zero_env_default(self):
        value = await self._capture_run_semaphore(
            env={"CALIBRATE_LLM_BENCHMARK_PARALLEL": "0"}
        )
        self.assertEqual(value, 2)

    async def test_run_model_parallel_agent_path_uses_resolver(self):
        value = await self._capture_run_semaphore(
            env={"CALIBRATE_LLM_BENCHMARK_PARALLEL": "6"}, agent=True
        )
        self.assertEqual(value, 6)

    async def test_run_multi_model_with_exception(self):
        from calibrate.llm import tests

        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_tests.run_test", AsyncMock(side_effect=RuntimeError("nope"))):
            result = await tests.run(
                test_cases=[{
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "x"},
                }],
                output_dir=tmp,
                models=["m1"],
            )
        self.assertEqual(result["models"]["m1"]["status"], "error")

    async def test_run_with_agent(self):
        from calibrate.llm import tests

        fake_test_result = {
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": True, "judge_results": {}},
        }
        fake_agent = MagicMock()

        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_tests.run_test_external", AsyncMock(return_value=fake_test_result)):
            result = await tests.run(
                test_cases=[{
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "respond"},
                }],
                output_dir=tmp,
                agent=fake_agent,
            )
        self.assertEqual(result["status"], "completed")

    async def test_run_agent_benchmark(self):
        from calibrate.llm import tests

        fake_test_result = {
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": True, "judge_results": {}},
        }
        fake_agent = MagicMock()

        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_tests.run_test_external", AsyncMock(return_value=fake_test_result)):
            result = await tests.run(
                test_cases=[{
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "x"},
                }],
                output_dir=tmp,
                agent=fake_agent,
                models=["m1", "m2"],
            )
        self.assertEqual(set(result.keys()), {"m1", "m2"})

    async def test_run_single(self):
        from calibrate.llm import tests

        fake_test_result = {
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": True, "judge_results": {}},
        }

        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_tests.run_test", AsyncMock(return_value=fake_test_result)):
            result = await tests.run_single(
                test_cases=[{
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "x"},
                }],
                output_dir=tmp,
                model="m1",
                run_name="run1",
            )
        self.assertEqual(result["status"], "completed")

    async def test_run_test_single(self):
        from calibrate.llm import tests

        fake_test_result = {
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": True, "judge_results": {}},
        }
        with patch("calibrate.llm.run_tests.run_test", AsyncMock(return_value=fake_test_result)):
            result = await tests.run_test(
                chat_history=[{"role": "user", "content": "hi"}],
                evaluation={"type": "tool_call", "tool_calls": []},
                system_prompt="sp",
                model="m1",
                provider="openrouter",
            )
        self.assertTrue(result["metrics"]["passed"])

    async def test_run_inference(self):
        from calibrate.llm import tests

        fake_inf_result = {"response": "Hi", "tool_calls": [], "captured_errors": []}
        with patch("calibrate.llm.run_tests.run_inference", AsyncMock(return_value=fake_inf_result)):
            result = await tests.run_inference(
                chat_history=[{"role": "user", "content": "hi"}],
                system_prompt="sp",
                model="m1",
                provider="openrouter",
            )
        self.assertEqual(result["response"], "Hi")

    def test_leaderboard_delegates(self):
        from calibrate.llm import tests as T

        with patch("calibrate.llm.tests_leaderboard.generate_leaderboard") as gl:
            T.leaderboard("/tmp/x", "/tmp/y")
            gl.assert_called_once_with(output_dir="/tmp/x", save_dir="/tmp/y")


class TestLLMSimulationsRun(unittest.IsolatedAsyncioTestCase):
    async def test_run_single_model(self):
        from calibrate.llm import simulations

        fake_result = (
            {"persona_index": 0, "scenario_index": 0, "elapsed": 1.0},
            [{"name": "task_complete", "value": 1.0, "type": "binary"}],
        )
        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_simulation.run_single_simulation_task",
                   AsyncMock(return_value=fake_result)):
            result = await simulations.run(
                personas=[{"characteristics": "p", "gender": "male", "language": "english"}],
                scenarios=[{"description": "s"}],
                evaluators=[{"name": "task_complete", "system_prompt": "x", "judge_model": "m"}],
                output_dir=tmp,
                model="m1",
            )
        self.assertEqual(result["status"], "completed")
        self.assertIn("task_complete", result["metrics"])

    async def test_run_multi_model(self):
        from calibrate.llm import simulations

        fake_result = (
            {"persona_index": 0, "scenario_index": 0, "elapsed": 1.0},
            [{"name": "task_complete", "value": 1.0, "type": "binary"}],
        )
        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_simulation.run_single_simulation_task",
                   AsyncMock(return_value=fake_result)):
            result = await simulations.run(
                personas=[{"characteristics": "p", "gender": "male", "language": "english"}],
                scenarios=[{"description": "s"}],
                evaluators=[{"name": "task_complete", "system_prompt": "x", "judge_model": "m"}],
                output_dir=tmp,
                models=["m1", "m2"],
            )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(set(result["models"].keys()), {"m1", "m2"})

    async def test_run_multi_model_with_exception(self):
        from calibrate.llm import simulations

        # Force a require_simulation_evaluators failure at the multi-model level
        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.judges.require_simulation_evaluators", side_effect=RuntimeError("fail")):
            result = await simulations.run(
                personas=[{"characteristics": "p", "gender": "male", "language": "english"}],
                scenarios=[{"description": "s"}],
                evaluators=[{"name": "x", "system_prompt": "x", "judge_model": "m"}],
                output_dir=tmp,
                models=["m1"],
            )
        self.assertEqual(result["models"]["m1"]["status"], "error")

    async def test_run_with_agent(self):
        from calibrate.llm import simulations

        fake_result = (
            {"persona_index": 0, "scenario_index": 0, "elapsed": 1.0},
            [{"name": "x", "value": 1.0, "type": "binary", "evaluator_id": "ev1",
              "scale_min": 1, "scale_max": 5}],
        )
        fake_agent = MagicMock()

        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_simulation.run_single_simulation_task",
                   AsyncMock(return_value=fake_result)):
            result = await simulations.run(
                personas=[{"characteristics": "p", "gender": "male", "language": "english"}],
                scenarios=[{"description": "s"}],
                evaluators=[{"name": "x", "system_prompt": "x", "judge_model": "m"}],
                output_dir=tmp,
                agent=fake_agent,
            )
        self.assertEqual(result["status"], "completed")

    async def test_run_single(self):
        from calibrate.llm import simulations

        fake_result = (
            {"persona_index": 0, "scenario_index": 0, "elapsed": 1.0},
            [{"name": "x", "value": 1, "type": "binary"}],
        )
        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_simulation.run_single_simulation_task",
                   AsyncMock(return_value=fake_result)):
            result = await simulations.run_single(
                personas=[{"characteristics": "p", "gender": "male", "language": "english"}],
                scenarios=[{"description": "s"}],
                evaluators=[{"name": "x", "system_prompt": "x", "judge_model": "m"}],
                output_dir=tmp,
                model="m1",
            )
        self.assertEqual(result["status"], "completed")

    async def test_run_simulation_delegates(self):
        from calibrate.llm import simulations

        with patch("calibrate.llm.run_simulation.run_simulation",
                   AsyncMock(return_value={"status": "ok"})) as mocked:
            result = await simulations.run_simulation(
                bot_system_prompt="bp",
                tools=[],
                user_system_prompt="up",
                evaluators=[{"name": "x", "system_prompt": "x", "judge_model": "m"}],
            )
        self.assertEqual(result["status"], "ok")
        mocked.assert_called_once()

    def test_leaderboard_delegates(self):
        from calibrate.llm import simulations as S

        with patch("calibrate.llm.simulation_leaderboard.generate_leaderboard") as gl:
            S.leaderboard("/tmp/x", "/tmp/y")
            gl.assert_called_once_with(output_dir="/tmp/x", save_dir="/tmp/y")


if __name__ == "__main__":
    unittest.main()
