"""Extra coverage for calibrate/llm/run_simulation.py — helpers, eval-only path, agent simulation."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


def _bin_ev(name):
    return {"name": name, "system_prompt": "x", "judge_model": "openai/gpt-4.1"}


def _rate_ev(name, lo=1, hi=5):
    return {
        "name": name, "system_prompt": "x", "judge_model": "openai/gpt-4.1",
        "type": "rating", "scale_min": lo, "scale_max": hi,
    }


class TestBuildEvaluationResult(unittest.TestCase):
    def test_binary(self):
        from calibrate.llm.run_simulation import _build_evaluation_result

        result = _build_evaluation_result(
            _bin_ev("x"), {"match": True, "reasoning": "ok"}
        )
        self.assertEqual(result["type"], "binary")
        self.assertEqual(result["value"], 1.0)

    def test_rating(self):
        from calibrate.llm.run_simulation import _build_evaluation_result

        result = _build_evaluation_result(
            _rate_ev("x", 1, 5), {"score": 4, "reasoning": "ok"}
        )
        self.assertEqual(result["type"], "rating")
        self.assertEqual(result["scale_min"], 1)
        self.assertEqual(result["scale_max"], 5)

    def test_with_id(self):
        from calibrate.llm.run_simulation import _build_evaluation_result

        ev = dict(_bin_ev("x"), id="ev1")
        result = _build_evaluation_result(ev, {"match": True, "reasoning": "ok"})
        self.assertEqual(result["evaluator_id"], "ev1")


class TestJudgeAndEmit(unittest.IsolatedAsyncioTestCase):
    async def test_emits_lines(self):
        from calibrate.llm.run_simulation import _judge_and_emit

        captured = []
        with patch("calibrate.llm.run_simulation.evaluate_simuation",
                   AsyncMock(return_value={"x": {"match": True, "reasoning": "ok"}})):
            results = await _judge_and_emit(
                transcript=[{"role": "user", "content": "Hi"}],
                evaluators=[_bin_ev("x")],
                fallback_judge_model="m",
                emit=captured.append,
            )
        self.assertEqual(results[0]["name"], "x")
        self.assertTrue(any("evaluator(s)" in line for line in captured))


class TestValidateSimulationEvalOnlyDataset(unittest.TestCase):
    def test_valid(self):
        from calibrate.llm.run_simulation import validate_simulation_eval_only_dataset

        ok, _ = validate_simulation_eval_only_dataset([
            {"conversation_history": [{"role": "user", "content": "hi"}]}
        ])
        self.assertTrue(ok)

    def test_not_list(self):
        from calibrate.llm.run_simulation import validate_simulation_eval_only_dataset

        ok, _ = validate_simulation_eval_only_dataset({})
        self.assertFalse(ok)

    def test_item_not_dict(self):
        from calibrate.llm.run_simulation import validate_simulation_eval_only_dataset

        ok, _ = validate_simulation_eval_only_dataset(["x"])
        self.assertFalse(ok)

    def test_missing_history(self):
        from calibrate.llm.run_simulation import validate_simulation_eval_only_dataset

        ok, _ = validate_simulation_eval_only_dataset([{}])
        self.assertFalse(ok)

    def test_history_not_list(self):
        from calibrate.llm.run_simulation import validate_simulation_eval_only_dataset

        ok, _ = validate_simulation_eval_only_dataset([{"conversation_history": "no"}])
        self.assertFalse(ok)

    def test_invalid_name(self):
        from calibrate.llm.run_simulation import validate_simulation_eval_only_dataset

        ok, _ = validate_simulation_eval_only_dataset(
            [{"conversation_history": [], "name": 5}]
        )
        self.assertFalse(ok)


class TestSaveTranscript(unittest.TestCase):
    def test_no_output_dir_returns(self):
        from calibrate.llm.run_simulation import _save_transcript

        _save_transcript(None, [{"role": "user", "content": "hi"}])

    def test_writes_file(self):
        from calibrate.llm.run_simulation import _save_transcript

        with tempfile.TemporaryDirectory() as tmp:
            _save_transcript(tmp, [{"role": "user", "content": "hi"}])
            self.assertTrue((Path(tmp) / "transcript.json").exists())


class TestRunEvalOnlySimulationTask(unittest.IsolatedAsyncioTestCase):
    async def test_basic(self):
        from calibrate.llm.run_simulation import run_eval_only_simulation_task

        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_simulation.evaluate_simuation",
                   AsyncMock(return_value={"x": {"match": True, "reasoning": "ok"}})):
            semaphore = asyncio.Semaphore(1)
            result = await run_eval_only_simulation_task(
                semaphore=semaphore,
                item={"conversation_history": [{"role": "user", "content": "hi"}], "name": "n1"},
                item_index=0,
                evaluators=[_bin_ev("x")],
                output_dir=tmp,
            )
        sim_metrics, eval_results = result
        self.assertEqual(sim_metrics["row_id"], "row_1")
        self.assertEqual(sim_metrics["name"], "n1")
        self.assertEqual(sim_metrics["x"], 1.0)


class TestRunEvalOnlySimulations(unittest.IsolatedAsyncioTestCase):
    async def test_run_eval_only_full(self):
        from calibrate.llm.run_simulation import run_eval_only_simulations

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with patch("calibrate.llm.run_simulation.evaluate_simuation",
                       AsyncMock(return_value={"x": {"match": True, "reasoning": "ok"}})):
                failed = await run_eval_only_simulations(
                    config={"evaluators": [_bin_ev("x")]},
                    dataset=[
                        {"conversation_history": [{"role": "user", "content": "hi"}], "name": "a"},
                        {"conversation_history": [{"role": "user", "content": "hi2"}]},
                    ],
                    output_dir=str(out),
                    parallel=2,
                )
            self.assertEqual(failed, 0)
            self.assertTrue((out / "dataset_map.json").exists())


class TestAggregateAndWrite(unittest.TestCase):
    def test_aggregates(self):
        from calibrate.llm.run_simulation import _aggregate_and_write_simulation_results

        with tempfile.TemporaryDirectory() as tmp:
            results = [
                (
                    {"name": "sim1", "x": 1.0},
                    [{"name": "x", "value": 1.0, "type": "binary",
                      "evaluator_id": "ev1", "scale_min": 1, "scale_max": 5}],
                ),
                (
                    {"name": "sim2", "x": 0.0},
                    [{"name": "x", "value": 0.0, "type": "binary",
                      "evaluator_id": "ev1"}],
                ),
                RuntimeError("oops"),
            ]
            failed = _aggregate_and_write_simulation_results(results, tmp)
        self.assertEqual(len(failed), 1)


class TestRunSimulationWithAgent(unittest.IsolatedAsyncioTestCase):
    async def test_basic_flow(self):
        from calibrate.llm import run_simulation as RS

        # Mock the user LLM
        fake_user_resp = MagicMock()
        fake_user_resp.choices = [
            MagicMock(message=MagicMock(content="user-msg"))
        ]

        fake_user_client = MagicMock()
        fake_user_client.chat.completions.create = AsyncMock(return_value=fake_user_resp)

        fake_agent = MagicMock()
        fake_agent.call = AsyncMock(side_effect=[
            {"response": "agent-1", "tool_calls": []},
            {"response": "agent-2", "tool_calls": []},
        ])

        with patch("openai.AsyncOpenAI", return_value=fake_user_client), \
             patch.object(RS, "evaluate_simuation",
                          AsyncMock(return_value={"x": {"match": True, "reasoning": "ok"}})):
            result = await RS.run_simulation_with_agent(
                agent=fake_agent,
                user_system_prompt="up",
                evaluators=[_bin_ev("x")],
                max_turns=1,
            )
        self.assertGreater(len(result["transcript"]), 0)
        self.assertEqual(len(result["evaluation_results"]), 1)

    async def test_agent_no_response_ends_early(self):
        from calibrate.llm import run_simulation as RS

        fake_user_client = MagicMock()
        fake_agent = MagicMock()
        # 3 failed attempts (tool calls only) returns None
        fake_agent.call = AsyncMock(return_value={"response": None, "tool_calls": [{"tool": "x"}]})

        with patch("openai.AsyncOpenAI", return_value=fake_user_client), \
             patch.object(RS, "evaluate_simuation",
                          AsyncMock(return_value={})):
            result = await RS.run_simulation_with_agent(
                agent=fake_agent,
                user_system_prompt="up",
                evaluators=[_bin_ev("x")],
                max_turns=1,
                agent_speaks_first=True,
            )
        self.assertEqual(result["transcript"], [])

    async def test_openrouter_user_provider(self):
        from calibrate.llm import run_simulation as RS

        fake_user_resp = MagicMock()
        fake_user_resp.choices = [MagicMock(message=MagicMock(content="hi"))]

        fake_user_client = MagicMock()
        fake_user_client.chat.completions.create = AsyncMock(return_value=fake_user_resp)

        fake_agent = MagicMock()
        fake_agent.call = AsyncMock(return_value={"response": "agent", "tool_calls": []})

        with patch("openai.AsyncOpenAI", return_value=fake_user_client), \
             patch.object(RS, "evaluate_simuation",
                          AsyncMock(return_value={"x": {"match": True, "reasoning": "ok"}})):
            await RS.run_simulation_with_agent(
                agent=fake_agent,
                user_system_prompt="up",
                evaluators=[_bin_ev("x")],
                max_turns=1,
                agent_speaks_first=False,
                user_provider="openrouter",
            )


class TestMainEvalOnly(unittest.IsolatedAsyncioTestCase):
    async def test_main_eval_only_run(self):
        from calibrate.llm import run_simulation as RS

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({"evaluators": [_bin_ev("x")]}))
            ds = Path(tmp) / "ds.json"
            ds.write_text(json.dumps([
                {"conversation_history": [{"role": "user", "content": "hi"}]}
            ]))
            argv = ["sim.py", "-c", str(cfg), "-o", str(Path(tmp) / "out"),
                    "--eval-only", "--dataset", str(ds)]
            with patch.object(sys, "argv", argv), \
                 patch.object(RS, "evaluate_simuation",
                              AsyncMock(return_value={"x": {"match": True, "reasoning": "ok"}})):
                await RS.main()

    async def test_main_invalid_evaluators_exits(self):
        from calibrate.llm import run_simulation as RS

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({"evaluators": []}))
            argv = ["sim.py", "-c", str(cfg), "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await RS.main()

    async def test_main_eval_only_missing_dataset(self):
        from calibrate.llm import run_simulation as RS

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({"evaluators": [_bin_ev("x")]}))
            argv = ["sim.py", "-c", str(cfg), "-o", tmp, "--eval-only"]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await RS.main()

    async def test_main_eval_only_invalid_dataset_json(self):
        from calibrate.llm import run_simulation as RS

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({"evaluators": [_bin_ev("x")]}))
            ds = Path(tmp) / "ds.json"
            ds.write_text("{bad")
            argv = ["sim.py", "-c", str(cfg), "-o", tmp,
                    "--eval-only", "--dataset", str(ds)]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await RS.main()

    async def test_main_eval_only_invalid_dataset_shape(self):
        from calibrate.llm import run_simulation as RS

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({"evaluators": [_bin_ev("x")]}))
            ds = Path(tmp) / "ds.json"
            ds.write_text(json.dumps({}))
            argv = ["sim.py", "-c", str(cfg), "-o", tmp,
                    "--eval-only", "--dataset", str(ds)]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await RS.main()


class TestResolveSimulationParallel(unittest.TestCase):
    def test_cli_value_takes_precedence(self):
        from calibrate.llm.run_simulation import _resolve_simulation_parallel
        with patch.dict("os.environ", {"CALIBRATE_SIMULATION_PARALLEL": "7"}):
            self.assertEqual(_resolve_simulation_parallel(3), 3)

    def test_env_var_used_when_no_cli(self):
        from calibrate.llm.run_simulation import _resolve_simulation_parallel
        with patch.dict("os.environ", {"CALIBRATE_SIMULATION_PARALLEL": "7"}):
            self.assertEqual(_resolve_simulation_parallel(None), 7)

    def test_default_when_neither_set(self):
        from calibrate.llm.run_simulation import (
            _resolve_simulation_parallel,
            DEFAULT_SIMULATION_PARALLEL,
        )
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("CALIBRATE_SIMULATION_PARALLEL", None)
            self.assertEqual(
                _resolve_simulation_parallel(None), DEFAULT_SIMULATION_PARALLEL
            )

    def test_invalid_env_falls_back_to_default(self):
        from calibrate.llm.run_simulation import (
            _resolve_simulation_parallel,
            DEFAULT_SIMULATION_PARALLEL,
        )
        with patch.dict("os.environ", {"CALIBRATE_SIMULATION_PARALLEL": "abc"}):
            self.assertEqual(
                _resolve_simulation_parallel(None), DEFAULT_SIMULATION_PARALLEL
            )

    def test_non_positive_values_ignored(self):
        from calibrate.llm.run_simulation import (
            _resolve_simulation_parallel,
            DEFAULT_SIMULATION_PARALLEL,
        )
        with patch.dict("os.environ", {"CALIBRATE_SIMULATION_PARALLEL": "0"}):
            self.assertEqual(
                _resolve_simulation_parallel(0), DEFAULT_SIMULATION_PARALLEL
            )


if __name__ == "__main__":
    unittest.main()
