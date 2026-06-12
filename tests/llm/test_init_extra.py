"""Cover llm/__init__.py public API entry points via heavy mocking."""

import asyncio
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

    async def test_run_with_agent_resumes_completed_test_cases(self):
        from calibrate.llm import tests

        fake_test_result = {
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": True, "judge_results": {}},
        }
        fake_agent = MagicMock()
        test_cases = [
            {
                "id": "tc1",
                "history": [{"role": "user", "content": "a"}],
                "evaluation": {"type": "response", "criteria": "x"},
            },
            {
                "id": "tc2",
                "history": [{"role": "user", "content": "b"}],
                "evaluation": {"type": "response", "criteria": "y"},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            # Pre-seed a prior run where tc1 already completed.
            prior = [{
                "test_case_id": "tc1",
                "output": {"response": "done", "tool_calls": []},
                "metrics": {"passed": True, "judge_results": {}},
                "test_case": test_cases[0],
            }]
            with open(os.path.join(tmp, "results.json"), "w") as f:
                json.dump(prior, f)

            mock_external = AsyncMock(return_value=fake_test_result)
            with patch(
                "calibrate.llm.run_tests.run_test_external", mock_external
            ):
                result = await tests.run(
                    test_cases=test_cases,
                    output_dir=tmp,
                    agent=fake_agent,
                )

        self.assertEqual(result["status"], "completed")
        # Only tc2 should have been (re-)run; tc1 was resumed from disk.
        self.assertEqual(mock_external.await_count, 1)
        self.assertEqual(result["metrics"]["total"], 2)

    async def test_run_with_agent_overwrite_reruns_everything(self):
        from calibrate.llm import tests

        fake_test_result = {
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": True, "judge_results": {}},
        }
        fake_agent = MagicMock()
        test_cases = [
            {
                "id": "tc1",
                "history": [{"role": "user", "content": "a"}],
                "evaluation": {"type": "response", "criteria": "x"},
            },
            {
                "id": "tc2",
                "history": [{"role": "user", "content": "b"}],
                "evaluation": {"type": "response", "criteria": "y"},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "results.json"), "w") as f:
                json.dump(
                    [{"test_case_id": "tc1", "metrics": {"passed": True}}], f
                )

            mock_external = AsyncMock(return_value=fake_test_result)
            with patch(
                "calibrate.llm.run_tests.run_test_external", mock_external
            ):
                await tests.run(
                    test_cases=test_cases,
                    output_dir=tmp,
                    agent=fake_agent,
                    overwrite=True,
                )

        # overwrite=True ignores the prior results — both cases re-run.
        self.assertEqual(mock_external.await_count, 2)

    async def test_run_raises_on_duplicate_ids_before_running(self):
        from calibrate.llm import tests

        fake_agent = MagicMock()
        test_cases = [
            {"id": "dup", "history": [{"role": "user", "content": "a"}],
             "evaluation": {"type": "response", "criteria": "x"}},
            {"id": "dup", "history": [{"role": "user", "content": "b"}],
             "evaluation": {"type": "response", "criteria": "y"}},
        ]
        mock_external = AsyncMock(return_value={
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": True, "judge_results": {}},
        })
        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_tests.run_test_external", mock_external):
            with self.assertRaises(ValueError):
                await tests.run(
                    test_cases=test_cases, output_dir=tmp, agent=fake_agent,
                )
        # Fail-fast: no test cases were dispatched.
        self.assertEqual(mock_external.await_count, 0)

    async def test_run_without_ids_does_not_resume(self):
        from calibrate.llm import tests

        async def fresh_result(*args, **kwargs):
            return {
                "output": {"response": "Hi", "tool_calls": []},
                "metrics": {"passed": True, "judge_results": {}},
            }
        fake_agent = MagicMock()

        # No ids on the cases — there's no stable key to resume by, so a re-run
        # of the same dataset must re-evaluate everything (no false skips).
        def make_cases():
            return [
                {"history": [{"role": "user", "content": "a"}],
                 "evaluation": {"type": "response", "criteria": "x"}},
                {"history": [{"role": "user", "content": "b"}],
                 "evaluation": {"type": "response", "criteria": "y"}},
            ]

        with tempfile.TemporaryDirectory() as tmp:
            mock_external = AsyncMock(side_effect=fresh_result)
            with patch("calibrate.llm.run_tests.run_test_external", mock_external):
                await tests.run(
                    test_cases=make_cases(), output_dir=tmp, agent=fake_agent,
                )
                mock_external.reset_mock()
                await tests.run(
                    test_cases=make_cases(), output_dir=tmp, agent=fake_agent,
                )

        # Second run re-ran both cases since they carry no resumable id.
        self.assertEqual(mock_external.await_count, 2)

    async def test_run_with_agent_aggregates_cost_latency_and_tokens(self):
        from calibrate.llm import tests

        fake_test_result = {
            "output": {
                "response": "Hi",
                "tool_calls": [],
                "metrics": {"cost": 0.002768, "latency_ms": 1245.8, "total_tokens": 4387},
                "cost": 0.002768,
                "total_tokens": 4387,
            },
            "metrics": {"passed": True, "judge_results": {}},
            "latency_ms": 1245.8,
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
            with open(os.path.join(result["output_dir"], "metrics.json")) as f:
                written = json.load(f)

        self.assertEqual(result["status"], "completed")
        self.assertIn("cost", result["metrics"])
        self.assertEqual(result["metrics"]["cost"]["count"], 1)
        self.assertAlmostEqual(result["metrics"]["cost"]["mean"], 0.002768)
        self.assertIn("latency_ms", result["metrics"])
        self.assertEqual(result["metrics"]["latency_ms"]["count"], 1)
        self.assertEqual(result["metrics"]["latency_ms"]["mean"], 1246)
        self.assertIn("total_tokens", result["metrics"])
        self.assertEqual(result["metrics"]["total_tokens"]["count"], 1)
        self.assertEqual(result["metrics"]["total_tokens"]["mean"], 4387)
        self.assertIn("cost", written)
        self.assertIn("latency_ms", written)
        self.assertIn("total_tokens", written)

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

    async def test_run_agent_benchmark_failed_case_labeled(self):
        from calibrate.llm import tests

        # A failing case in the agent-benchmark path exercises the labeled
        # "[model] ❌ ... failed" log branch (the passing branch is covered above).
        fake_test_result = {
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": False, "judge_results": {}},
        }
        fake_agent = MagicMock()

        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_tests.run_test_external",
                   AsyncMock(return_value=fake_test_result)):
            result = await tests.run(
                test_cases=[{
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "x"},
                }],
                output_dir=tmp,
                agent=fake_agent,
                models=["m1"],
            )
        self.assertEqual(set(result.keys()), {"m1"})
        self.assertEqual(result["m1"]["metrics"]["passed"], 0)

    async def test_run_agent_benchmark_one_model_failure_isolated(self):
        from calibrate.llm import tests

        # One model's hard failure must not cancel the others: it's recorded as
        # an error entry while the healthy model still completes.
        fake_ok = {
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": True, "judge_results": {}},
        }

        async def flaky_external(*args, **kwargs):
            if kwargs.get("model") == "bad":
                raise RuntimeError("agent timed out")
            return fake_ok

        fake_agent = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_tests.run_test_external",
                   AsyncMock(side_effect=flaky_external)):
            result = await tests.run(
                test_cases=[{
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "x"},
                }],
                output_dir=tmp,
                agent=fake_agent,
                models=["good", "bad"],
            )
        self.assertEqual(set(result.keys()), {"good", "bad"})
        self.assertEqual(result["bad"]["status"], "error")
        self.assertIn("agent timed out", result["bad"]["error"])
        # The healthy model still produced real metrics.
        self.assertEqual(result["good"]["metrics"]["passed"], 1)

    async def test_agent_benchmark_runs_models_concurrently(self):
        from calibrate.llm import tests

        # A 2-party barrier proves real concurrency: each model's request blocks
        # until *both* models have arrived. If the runner were sequential the
        # first model would wait alone and time out, failing the test.
        barrier = asyncio.Barrier(2)

        async def gated_external(*args, **kwargs):
            await asyncio.wait_for(barrier.wait(), timeout=2)
            return {
                "output": {"response": "Hi", "tool_calls": []},
                "metrics": {"passed": True, "judge_results": {}},
            }

        fake_agent = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_tests.run_test_external",
                   AsyncMock(side_effect=gated_external)):
            result = await tests.run(
                test_cases=[{
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "x"},
                }],
                output_dir=tmp,
                agent=fake_agent,
                models=["a", "b"],
                max_parallel=2,
            )
        self.assertEqual(set(result.keys()), {"a", "b"})
        self.assertEqual(result["a"]["metrics"]["passed"], 1)
        self.assertEqual(result["b"]["metrics"]["passed"], 1)

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
