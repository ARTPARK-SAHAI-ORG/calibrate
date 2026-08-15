"""Unit tests for calibrate_agent/general/eval.py.

Covers dataset validation, evaluator resolution from config, and the
end-to-end run_general_eval path (with the judge mocked) producing
metrics.json + results.csv.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, AsyncMock

import pandas as pd

from calibrate_agent.general.eval import (
    validate_general_eval_dataset,
    _resolve_evaluators,
    run_general_eval,
    main as eval_main,
)


BINARY_EV = {
    "name": "faithful",
    "system_prompt": "judge faithfulness",
    "judge_model": "openai/gpt-4.1",
}


def _write_json(obj) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
    json.dump(obj, f)
    f.close()
    return f.name


class TestValidateDataset(unittest.TestCase):
    def test_missing_file(self):
        ok, err, rows = validate_general_eval_dataset("/no/such/file.json")
        self.assertFalse(ok)
        self.assertIn("does not exist", err)

    def test_not_a_list(self):
        path = _write_json({"id": "1"})
        try:
            ok, err, _ = validate_general_eval_dataset(path)
        finally:
            os.unlink(path)
        self.assertFalse(ok)
        self.assertIn("list", err)

    def test_empty_list(self):
        path = _write_json([])
        try:
            ok, err, _ = validate_general_eval_dataset(path)
        finally:
            os.unlink(path)
        self.assertFalse(ok)
        self.assertIn("empty", err)

    def test_missing_fields(self):
        path = _write_json([{"id": "1", "input": "x"}])  # no output
        try:
            ok, err, _ = validate_general_eval_dataset(path)
        finally:
            os.unlink(path)
        self.assertFalse(ok)
        self.assertIn("output", err)

    def test_duplicate_ids(self):
        path = _write_json(
            [
                {"id": "1", "input": "a", "output": "b"},
                {"id": "1", "input": "c", "output": "d"},
            ]
        )
        try:
            ok, err, _ = validate_general_eval_dataset(path)
        finally:
            os.unlink(path)
        self.assertFalse(ok)
        self.assertIn("Duplicate", err)

    def test_valid(self):
        rows_in = [
            {"id": "1", "input": "a", "output": "b"},
            {"id": "2", "input": "c", "output": "d"},
        ]
        path = _write_json(rows_in)
        try:
            ok, err, rows = validate_general_eval_dataset(path)
        finally:
            os.unlink(path)
        self.assertTrue(ok)
        self.assertEqual(err, "")
        self.assertEqual(rows, rows_in)

    def test_valid_with_arguments_dict(self):
        # arguments is keyed by evaluator name → that evaluator's var dict.
        rows_in = [
            {
                "id": "1",
                "input": "a",
                "output": "b",
                "arguments": {"faithful": {"reference": "v"}},
            },
        ]
        path = _write_json(rows_in)
        try:
            ok, err, rows = validate_general_eval_dataset(path)
        finally:
            os.unlink(path)
        self.assertTrue(ok)
        self.assertEqual(err, "")
        self.assertEqual(rows, rows_in)

    def test_valid_without_arguments(self):
        # arguments is optional — rows missing it are still valid.
        rows_in = [{"id": "1", "input": "a", "output": "b"}]
        path = _write_json(rows_in)
        try:
            ok, err, _ = validate_general_eval_dataset(path)
        finally:
            os.unlink(path)
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_arguments_not_a_dict_rejected(self):
        path = _write_json(
            [{"id": "1", "input": "a", "output": "b", "arguments": "nope"}]
        )
        try:
            ok, err, _ = validate_general_eval_dataset(path)
        finally:
            os.unlink(path)
        self.assertFalse(ok)
        self.assertEqual(err, "Row 0 field 'arguments' must be an object")

    def test_arguments_evaluator_value_not_a_dict_rejected(self):
        path = _write_json(
            [
                {
                    "id": "1",
                    "input": "a",
                    "output": "b",
                    "arguments": {"faithful": "nope"},
                }
            ]
        )
        try:
            ok, err, _ = validate_general_eval_dataset(path)
        finally:
            os.unlink(path)
        self.assertFalse(ok)
        self.assertEqual(
            err,
            "Row 0 field 'arguments['faithful']' must be an object "
            "mapping variable names to values",
        )


class TestResolveEvaluators(unittest.TestCase):
    def test_missing_evaluators_raises(self):
        with self.assertRaises(ValueError):
            _resolve_evaluators({})

    def test_empty_evaluators_raises(self):
        with self.assertRaises(ValueError):
            _resolve_evaluators({"evaluators": []})

    def test_evaluator_without_system_prompt_raises(self):
        with self.assertRaises(ValueError):
            _resolve_evaluators({"evaluators": [{"name": "x"}]})

    def test_valid_config(self):
        out = _resolve_evaluators({"evaluators": [BINARY_EV]})
        self.assertEqual(out, [BINARY_EV])


class TestRunGeneralEval(unittest.IsolatedAsyncioTestCase):
    async def test_error_status_on_bad_dataset(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = await run_general_eval(
                dataset_path="/no/such/file.json",
                output_dir=out_dir,
                evaluators=[BINARY_EV],
            )
        self.assertEqual(result["status"], "error")

    async def test_removes_stale_log_file(self):
        rows = [{"id": "1", "input": "a", "output": "b"}]
        dataset_path = _write_json(rows)
        fake_score = {
            "scores": {"faithful": {"type": "binary", "mean": 1.0}},
            "score": 1.0,
            "per_row": [{"faithful": {"reasoning": "ok", "match": True}}],
        }
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                # Pre-existing logs file should be removed at the start of the run.
                stale = os.path.join(out_dir, "logs")
                with open(stale, "w") as f:
                    f.write("old log\n")
                with patch(
                    "calibrate_agent.general.eval.get_general_judge_score",
                    AsyncMock(return_value=fake_score),
                ):
                    result = await run_general_eval(
                        dataset_path=dataset_path,
                        output_dir=out_dir,
                        evaluators=[BINARY_EV],
                    )
                self.assertEqual(result["status"], "completed")
                # The stale content is gone (file recreated fresh by the logger).
                self.assertNotIn("old log", open(stale).read())
        finally:
            os.unlink(dataset_path)

    async def test_end_to_end_writes_outputs(self):
        rows = [
            {"id": "row_a", "input": "doc A", "output": "sum A"},
            {"id": "row_b", "input": "doc B", "output": "sum B"},
        ]
        dataset_path = _write_json(rows)

        fake_score = {
            "scores": {"faithful": {"type": "binary", "mean": 0.5}},
            "score": 0.5,
            "per_row": [
                {"faithful": {"reasoning": "ok", "match": True}},
                {"faithful": {"reasoning": "no", "match": False}},
            ],
        }

        try:
            with tempfile.TemporaryDirectory() as out_dir:
                with patch(
                    "calibrate_agent.general.eval.get_general_judge_score",
                    AsyncMock(return_value=fake_score),
                ):
                    result = await run_general_eval(
                        dataset_path=dataset_path,
                        output_dir=out_dir,
                        evaluators=[BINARY_EV],
                    )

                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["metrics"]["faithful"]["mean"], 0.5)

                # metrics.json
                with open(os.path.join(out_dir, "metrics.json")) as f:
                    metrics = json.load(f)
                self.assertEqual(metrics["faithful"]["mean"], 0.5)

                # results.csv has one row per input row + per-evaluator columns
                df = pd.read_csv(os.path.join(out_dir, "results.csv"))
                self.assertEqual(len(df), 2)
                self.assertIn("faithful", df.columns)
                self.assertIn("faithful_reasoning", df.columns)
                self.assertEqual(list(df["id"]), ["row_a", "row_b"])
                self.assertEqual(bool(df.iloc[0]["faithful"]), True)
                self.assertEqual(bool(df.iloc[1]["faithful"]), False)

                # config.json captures the evaluators used
                with open(os.path.join(out_dir, "config.json")) as f:
                    cfg = json.load(f)
                self.assertEqual(cfg["evaluators"][0]["name"], "faithful")
        finally:
            os.unlink(dataset_path)

    async def test_arguments_list_passed_to_judge(self):
        rows = [
            {"id": "row_a", "input": "doc A", "output": "sum A",
             "arguments": {"faithful": {"name": "Ann"}}},
            {"id": "row_b", "input": "doc B", "output": "sum B"},
        ]
        dataset_path = _write_json(rows)

        fake_score = {
            "scores": {"faithful": {"type": "binary", "mean": 1.0}},
            "score": 1.0,
            "per_row": [
                {"faithful": {"reasoning": "ok", "match": True}},
                {"faithful": {"reasoning": "ok", "match": True}},
            ],
        }
        judge_mock = AsyncMock(return_value=fake_score)
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                with patch(
                    "calibrate_agent.general.eval.get_general_judge_score", judge_mock
                ):
                    result = await run_general_eval(
                        dataset_path=dataset_path,
                        output_dir=out_dir,
                        evaluators=[BINARY_EV],
                    )
                self.assertEqual(result["status"], "completed")
        finally:
            os.unlink(dataset_path)

        judge_mock.assert_awaited_once()
        self.assertEqual(
            judge_mock.call_args.kwargs["arguments_list"],
            [{"faithful": {"name": "Ann"}}, None],
        )


class TestGeneralPartialResults(unittest.IsolatedAsyncioTestCase):
    """results.csv holds the rows graded so far while the judge is still running."""

    async def test_partial_results_written_during_run(self):
        from calibrate_agent.general.eval import _score_and_write_results

        with tempfile.TemporaryDirectory() as out_dir:
            results_path = os.path.join(out_dir, "results.csv")
            seen = {}

            async def fake_judge(_input, output, **kwargs):
                if output == "sum B":
                    # Row A finishes first; capture what is on disk by then.
                    await asyncio.sleep(0.05)
                    seen["partial"] = pd.read_csv(results_path)
                return {"faithful": {"reasoning": output, "match": output == "sum A"}}

            with patch(
                "calibrate_agent.general.metrics.general_judge",
                AsyncMock(side_effect=fake_judge),
            ):
                metrics = await _score_and_write_results(
                    ids=["row_a", "row_b"],
                    inputs=["doc A", "doc B"],
                    outputs=["sum A", "sum B"],
                    evaluators=[BINARY_EV],
                    output_dir=out_dir,
                )

            partial = seen["partial"]
            self.assertEqual(list(partial["id"]), ["row_a"])
            self.assertEqual(bool(partial.iloc[0]["faithful"]), True)
            self.assertEqual(partial.iloc[0]["faithful_reasoning"], "sum A")

            self.assertAlmostEqual(metrics["faithful"]["mean"], 0.5)
            df = pd.read_csv(results_path)
            self.assertEqual(list(df["id"]), ["row_a", "row_b"])
            self.assertEqual(
                list(df.columns),
                ["id", "input", "output", "faithful", "faithful_reasoning"],
            )
            self.assertEqual([bool(v) for v in df["faithful"]], [True, False])

    async def test_partial_file_drops_scores_from_a_removed_evaluator(self):
        from calibrate_agent.general.eval import _score_and_write_results

        with tempfile.TemporaryDirectory() as out_dir:
            results_path = os.path.join(out_dir, "results.csv")
            seen = {}

            async def fake_judge(_input, output, **kwargs):
                if output == "sum B":
                    await asyncio.sleep(0.05)
                    seen["partial"] = pd.read_csv(results_path)
                return {"faithful": {"reasoning": output, "match": True}}

            with patch(
                "calibrate_agent.general.metrics.general_judge",
                AsyncMock(side_effect=fake_judge),
            ):
                await _score_and_write_results(
                    ids=["row_a", "row_b"],
                    inputs=["doc A", "doc B"],
                    outputs=["sum A", "sum B"],
                    evaluators=[BINARY_EV],
                    output_dir=out_dir,
                    existing_rows={
                        "row_a": {
                            "id": "row_a",
                            "input": "doc A",
                            "output": "sum A",
                            "concise": "True",
                            "concise_reasoning": "from the old evaluator",
                        }
                    },
                )

            self.assertNotIn("concise", seen["partial"].columns)


def _write_results_csv(out_dir, rows, evaluators=None) -> str:
    """Stand in for a prior run: its results.csv plus the config.json it wrote.

    Both are needed to resume — the config records how the stored scores were
    produced, and without it they cannot be trusted.
    """
    from calibrate_agent.judges import write_evaluator_config

    write_evaluator_config(out_dir, evaluators or [BINARY_EV])
    path = os.path.join(out_dir, "results.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


async def _fake_judge(_input, output, **kwargs):
    """Score every row from its own output text, so completion order cannot mix rows."""
    return {"faithful": {"reasoning": f"why {output}", "match": output.endswith("A")}}


class TestGeneralResume(unittest.IsolatedAsyncioTestCase):
    """A rerun keeps the rows already scored and judges only what is missing."""

    async def test_resumed_run_only_judges_missing_rows(self):
        rows = [
            {"id": "row_a", "input": "doc A", "output": "sum A"},
            {"id": "row_b", "input": "doc B", "output": "sum B"},
        ]
        dataset_path = _write_json(rows)
        judge = AsyncMock(side_effect=_fake_judge)
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                _write_results_csv(
                    out_dir,
                    [
                        {
                            "id": "row_a",
                            "input": "doc A",
                            "output": "sum A",
                            "faithful": True,
                            "faithful_reasoning": "why sum A",
                        }
                    ],
                )
                with patch(
                    "calibrate_agent.general.metrics.general_judge", judge
                ):
                    result = await run_general_eval(
                        dataset_path=dataset_path,
                        output_dir=out_dir,
                        evaluators=[BINARY_EV],
                    )

                self.assertEqual(result["status"], "completed")
                self.assertEqual(judge.await_count, 1)
                self.assertEqual(judge.await_args.args[1], "sum B")

                df = pd.read_csv(os.path.join(out_dir, "results.csv"))
                self.assertEqual(list(df["id"]), ["row_a", "row_b"])
                self.assertEqual([bool(v) for v in df["faithful"]], [True, False])
                self.assertEqual(
                    list(df["faithful_reasoning"]), ["why sum A", "why sum B"]
                )
        finally:
            os.unlink(dataset_path)

    async def test_resumed_metrics_match_a_single_run(self):
        rows = [
            {"id": "row_a", "input": "doc A", "output": "sum A"},
            {"id": "row_b", "input": "doc B", "output": "sum B"},
        ]
        dataset_path = _write_json(rows)
        try:
            with tempfile.TemporaryDirectory() as fresh_dir, patch(
                "calibrate_agent.general.metrics.general_judge",
                AsyncMock(side_effect=_fake_judge),
            ):
                await run_general_eval(
                    dataset_path=dataset_path,
                    output_dir=fresh_dir,
                    evaluators=[BINARY_EV],
                )
                with open(os.path.join(fresh_dir, "metrics.json")) as f:
                    single_run = json.load(f)

            with tempfile.TemporaryDirectory() as out_dir, patch(
                "calibrate_agent.general.metrics.general_judge",
                AsyncMock(side_effect=_fake_judge),
            ):
                _write_results_csv(
                    out_dir,
                    [
                        {
                            "id": "row_a",
                            "input": "doc A",
                            "output": "sum A",
                            "faithful": True,
                            "faithful_reasoning": "why sum A",
                        }
                    ],
                )
                await run_general_eval(
                    dataset_path=dataset_path,
                    output_dir=out_dir,
                    evaluators=[BINARY_EV],
                )
                with open(os.path.join(out_dir, "metrics.json")) as f:
                    resumed = json.load(f)
        finally:
            os.unlink(dataset_path)

        self.assertEqual(resumed, single_run)
        self.assertAlmostEqual(resumed["faithful"]["mean"], 0.5)

    async def test_overwrite_rejudges_every_row(self):
        rows = [
            {"id": "row_a", "input": "doc A", "output": "sum A"},
            {"id": "row_b", "input": "doc B", "output": "sum B"},
        ]
        dataset_path = _write_json(rows)
        judge = AsyncMock(side_effect=_fake_judge)
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                _write_results_csv(
                    out_dir,
                    [
                        {
                            "id": "row_a",
                            "input": "doc A",
                            "output": "sum A",
                            "faithful": True,
                            "faithful_reasoning": "stale",
                        }
                    ],
                )
                with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                    json.dump({"stale": True}, f)

                with patch("calibrate_agent.general.metrics.general_judge", judge):
                    await run_general_eval(
                        dataset_path=dataset_path,
                        output_dir=out_dir,
                        evaluators=[BINARY_EV],
                        overwrite=True,
                    )

                self.assertEqual(judge.await_count, 2)
                df = pd.read_csv(os.path.join(out_dir, "results.csv"))
                self.assertEqual(list(df["id"]), ["row_a", "row_b"])
                self.assertEqual(
                    list(df["faithful_reasoning"]), ["why sum A", "why sum B"]
                )
                with open(os.path.join(out_dir, "metrics.json")) as f:
                    metrics = json.load(f)
                self.assertNotIn("stale", metrics)
        finally:
            os.unlink(dataset_path)

    async def test_changed_evaluator_set_is_not_mixed_with_the_old_one(self):
        rows = [{"id": "row_a", "input": "doc A", "output": "sum A"}]
        dataset_path = _write_json(rows)
        judge = AsyncMock(side_effect=_fake_judge)
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                _write_results_csv(
                    out_dir,
                    [
                        {
                            "id": "row_a",
                            "input": "doc A",
                            "output": "sum A",
                            "concise": True,
                            "concise_reasoning": "from the old evaluator",
                        }
                    ],
                )
                with patch("calibrate_agent.general.metrics.general_judge", judge):
                    await run_general_eval(
                        dataset_path=dataset_path,
                        output_dir=out_dir,
                        evaluators=[BINARY_EV],
                    )

                self.assertEqual(judge.await_count, 1)
                df = pd.read_csv(os.path.join(out_dir, "results.csv"))
                self.assertNotIn("concise", df.columns)
                self.assertEqual(list(df["faithful_reasoning"]), ["why sum A"])
        finally:
            os.unlink(dataset_path)

    async def test_numeric_looking_id_resumes(self):
        rows = [
            {"id": 1, "input": "doc A", "output": "sum A"},
            {"id": 2, "input": "doc B", "output": "sum B"},
        ]
        dataset_path = _write_json(rows)
        judge = AsyncMock(side_effect=_fake_judge)
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                _write_results_csv(
                    out_dir,
                    [
                        {
                            "id": 1,
                            "input": "doc A",
                            "output": "sum A",
                            "faithful": True,
                            "faithful_reasoning": "why sum A",
                        }
                    ],
                )
                with patch("calibrate_agent.general.metrics.general_judge", judge):
                    await run_general_eval(
                        dataset_path=dataset_path,
                        output_dir=out_dir,
                        evaluators=[BINARY_EV],
                    )

                self.assertEqual(judge.await_count, 1)
                self.assertEqual(judge.await_args.args[1], "sum B")
                df = pd.read_csv(os.path.join(out_dir, "results.csv"))
                self.assertEqual(list(df["id"]), [1, 2])
        finally:
            os.unlink(dataset_path)

    async def test_row_with_blank_score_is_judged_again(self):
        rows = [{"id": "row_a", "input": "doc A", "output": "sum A"}]
        dataset_path = _write_json(rows)
        judge = AsyncMock(side_effect=_fake_judge)
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                _write_results_csv(
                    out_dir,
                    [
                        {
                            "id": "row_a",
                            "input": "doc A",
                            "output": "sum A",
                            "faithful": "",
                            "faithful_reasoning": "",
                        }
                    ],
                )
                with patch("calibrate_agent.general.metrics.general_judge", judge):
                    await run_general_eval(
                        dataset_path=dataset_path,
                        output_dir=out_dir,
                        evaluators=[BINARY_EV],
                    )

                self.assertEqual(judge.await_count, 1)
                df = pd.read_csv(os.path.join(out_dir, "results.csv"))
                self.assertEqual(list(df["faithful_reasoning"]), ["why sum A"])
        finally:
            os.unlink(dataset_path)

    async def test_row_missing_from_the_dataset_is_dropped(self):
        rows = [{"id": "row_a", "input": "doc A", "output": "sum A"}]
        dataset_path = _write_json(rows)
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                _write_results_csv(
                    out_dir,
                    [
                        {
                            "id": "row_a",
                            "input": "doc A",
                            "output": "sum A",
                            "faithful": True,
                            "faithful_reasoning": "why sum A",
                        },
                        {
                            "id": "row_gone",
                            "input": "doc X",
                            "output": "sum X",
                            "faithful": True,
                            "faithful_reasoning": "why sum X",
                        },
                    ],
                )
                with patch(
                    "calibrate_agent.general.metrics.general_judge",
                    AsyncMock(side_effect=_fake_judge),
                ):
                    result = await run_general_eval(
                        dataset_path=dataset_path,
                        output_dir=out_dir,
                        evaluators=[BINARY_EV],
                    )

                df = pd.read_csv(os.path.join(out_dir, "results.csv"))
                self.assertEqual(list(df["id"]), ["row_a"])
                self.assertAlmostEqual(result["metrics"]["faithful"]["mean"], 1.0)
        finally:
            os.unlink(dataset_path)

    async def test_row_arguments_still_match_their_row_on_a_resumed_run(self):
        templated = {
            "name": "faithful",
            "system_prompt": "check against {{reference}}",
            "judge_model": "openai/gpt-4.1",
        }
        rows = [
            {"id": "row_a", "input": "doc A", "output": "sum A",
             "arguments": {"faithful": {"reference": "REF-A"}}},
            {"id": "row_b", "input": "doc B", "output": "sum B",
             "arguments": {"faithful": {"reference": "REF-B"}}},
            {"id": "row_c", "input": "doc C", "output": "sum C",
             "arguments": {"faithful": {"reference": "REF-C"}}},
        ]
        dataset_path = _write_json(rows)
        prompts_by_output = {}

        async def capture(_input, output, **kwargs):
            prompts_by_output[output] = kwargs["evaluators"][0]["system_prompt"]
            return await _fake_judge(_input, output, **kwargs)

        try:
            with tempfile.TemporaryDirectory() as out_dir:
                _write_results_csv(
                    out_dir,
                    [
                        {
                            "id": "row_a",
                            "input": "doc A",
                            "output": "sum A",
                            "faithful": True,
                            "faithful_reasoning": "why sum A",
                        }
                    ],
                )
                with patch(
                    "calibrate_agent.general.metrics.general_judge",
                    AsyncMock(side_effect=capture),
                ):
                    await run_general_eval(
                        dataset_path=dataset_path,
                        output_dir=out_dir,
                        evaluators=[templated],
                    )
        finally:
            os.unlink(dataset_path)

        self.assertEqual(
            prompts_by_output,
            {
                "sum B": "check against REF-B",
                "sum C": "check against REF-C",
            },
        )


class TestMain(unittest.IsolatedAsyncioTestCase):
    """Cover the CLI entry point branches of calibrate_agent.general.eval.main()."""

    def _argv(self, dataset, config, out):
        return ["calibrate_agent", "--dataset", dataset, "-c", config, "-o", out]

    async def test_config_not_found_exits(self):
        with tempfile.TemporaryDirectory() as out_dir:
            ds = _write_json([{"id": "1", "input": "a", "output": "b"}])
            try:
                with patch.object(
                    sys, "argv", self._argv(ds, "/no/such/config.json", out_dir)
                ):
                    with self.assertRaises(SystemExit) as cm:
                        await eval_main()
            finally:
                os.unlink(ds)
        self.assertEqual(cm.exception.code, 1)

    async def test_config_bad_json_exits(self):
        with tempfile.TemporaryDirectory() as out_dir:
            ds = _write_json([{"id": "1", "input": "a", "output": "b"}])
            bad = tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            )
            bad.write("{not valid json")
            bad.close()
            try:
                with patch.object(sys, "argv", self._argv(ds, bad.name, out_dir)):
                    with self.assertRaises(SystemExit) as cm:
                        await eval_main()
            finally:
                os.unlink(ds)
                os.unlink(bad.name)
        self.assertEqual(cm.exception.code, 1)

    async def test_config_without_evaluators_exits(self):
        with tempfile.TemporaryDirectory() as out_dir:
            ds = _write_json([{"id": "1", "input": "a", "output": "b"}])
            cfg = _write_json({"evaluators": []})
            try:
                with patch.object(sys, "argv", self._argv(ds, cfg, out_dir)):
                    with self.assertRaises(SystemExit) as cm:
                        await eval_main()
            finally:
                os.unlink(ds)
                os.unlink(cfg)
        self.assertEqual(cm.exception.code, 1)

    async def test_error_status_exits(self):
        with tempfile.TemporaryDirectory() as out_dir:
            ds = _write_json([{"id": "1", "input": "a", "output": "b"}])
            cfg = _write_json({"evaluators": [BINARY_EV]})
            try:
                with patch.object(sys, "argv", self._argv(ds, cfg, out_dir)), patch(
                    "calibrate_agent.general.eval.run_general_eval",
                    AsyncMock(return_value={"status": "error", "error": "boom"}),
                ):
                    with self.assertRaises(SystemExit) as cm:
                        await eval_main()
            finally:
                os.unlink(ds)
                os.unlink(cfg)
        self.assertEqual(cm.exception.code, 1)

    async def test_success_prints_summary(self):
        with tempfile.TemporaryDirectory() as out_dir:
            ds = _write_json([{"id": "1", "input": "a", "output": "b"}])
            cfg = _write_json({"evaluators": [BINARY_EV]})
            completed = {
                "status": "completed",
                "metrics": {"faithful": {"type": "binary", "mean": 1.0}},
                "output_dir": out_dir,
            }
            run_mock = AsyncMock(return_value=completed)
            try:
                with patch.object(sys, "argv", self._argv(ds, cfg, out_dir)), patch(
                    "calibrate_agent.general.eval.run_general_eval", run_mock
                ):
                    # Should not raise SystemExit on success
                    await eval_main()
            finally:
                os.unlink(ds)
                os.unlink(cfg)
            # run_general_eval received the resolved evaluators from config
            self.assertEqual(
                run_mock.call_args.kwargs["evaluators"][0]["name"], "faithful"
            )

    async def test_overwrite_flag_rejudges_every_row(self):
        with tempfile.TemporaryDirectory() as out_dir:
            ds = _write_json([{"id": "row_a", "input": "doc A", "output": "sum A"}])
            cfg = _write_json({"evaluators": [BINARY_EV]})
            _write_results_csv(
                out_dir,
                [
                    {
                        "id": "row_a",
                        "input": "doc A",
                        "output": "sum A",
                        "faithful": True,
                        "faithful_reasoning": "stale",
                    }
                ],
            )
            judge = AsyncMock(side_effect=_fake_judge)
            argv = self._argv(ds, cfg, out_dir) + ["--overwrite"]
            try:
                with patch.object(sys, "argv", argv), patch(
                    "calibrate_agent.general.metrics.general_judge", judge
                ):
                    await eval_main()
            finally:
                os.unlink(ds)
                os.unlink(cfg)
            self.assertEqual(judge.await_count, 1)
            df = pd.read_csv(os.path.join(out_dir, "results.csv"))
            self.assertEqual(list(df["faithful_reasoning"]), ["why sum A"])

    async def test_overwrite_defaults_to_off(self):
        with tempfile.TemporaryDirectory() as out_dir:
            ds = _write_json([{"id": "1", "input": "a", "output": "b"}])
            cfg = _write_json({"evaluators": [BINARY_EV]})
            completed = {"status": "completed", "metrics": {}, "output_dir": out_dir}
            run_mock = AsyncMock(return_value=completed)
            try:
                with patch.object(sys, "argv", self._argv(ds, cfg, out_dir)), patch(
                    "calibrate_agent.general.eval.run_general_eval", run_mock
                ):
                    await eval_main()
            finally:
                os.unlink(ds)
                os.unlink(cfg)
            self.assertFalse(run_mock.call_args.kwargs["overwrite"])

    async def test_success_with_no_scores_prints_placeholder(self):
        with tempfile.TemporaryDirectory() as out_dir:
            ds = _write_json([{"id": "1", "input": "a", "output": "b"}])
            cfg = _write_json({"evaluators": [BINARY_EV]})
            # metrics with no type-bearing dicts → the "(no scores)" branch
            completed = {"status": "completed", "metrics": {}, "output_dir": out_dir}
            try:
                with patch.object(sys, "argv", self._argv(ds, cfg, out_dir)), patch(
                    "calibrate_agent.general.eval.run_general_eval",
                    AsyncMock(return_value=completed),
                ):
                    await eval_main()  # should not raise
            finally:
                os.unlink(ds)
                os.unlink(cfg)


class TestCliDispatch(unittest.TestCase):
    """Cover the `general` branch wired into calibrate_agent.cli.main().

    Plain (sync) TestCase because ``cli.main()`` calls ``asyncio.run()`` itself,
    which cannot nest inside a running event loop.
    """

    def test_dispatch_invokes_eval_main(self):
        from calibrate_agent import cli

        eval_main_mock = AsyncMock(return_value=None)
        argv = ["calibrate_agent", "general", "--dataset", "d.json", "-c", "c.json"]
        with patch.object(sys, "argv", argv), patch(
            "calibrate_agent.general.eval.main", eval_main_mock
        ):
            cli.main()
        eval_main_mock.assert_awaited_once()

    def test_dispatch_missing_dataset_exits(self):
        from calibrate_agent import cli

        argv = ["calibrate_agent", "general", "-c", "c.json"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 1)

    def test_dispatch_missing_config_exits(self):
        from calibrate_agent import cli

        argv = ["calibrate_agent", "general", "--dataset", "d.json"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 1)


class TestGeneralResumeRejudges(unittest.IsolatedAsyncioTestCase):
    """A stored score is only reused when it still describes this run."""

    DATASET = [
        {"id": "row_a", "input": "doc A", "output": "sum A"},
        {"id": "row_b", "input": "doc B", "output": "sum B"},
    ]

    def _write_prior_run(self, out_dir, stored_evaluators, stored_row):
        from calibrate_agent.judges import write_evaluator_config

        write_evaluator_config(out_dir, stored_evaluators)
        pd.DataFrame([stored_row]).to_csv(
            os.path.join(out_dir, "results.csv"), index=False
        )

    async def _run(self, out_dir, evaluators, judge):
        dataset_path = _write_json(self.DATASET)
        try:
            with patch("calibrate_agent.general.metrics.general_judge", judge):
                return await run_general_eval(
                    dataset_path=dataset_path,
                    output_dir=out_dir,
                    evaluators=evaluators,
                )
        finally:
            os.unlink(dataset_path)

    async def test_evaluator_that_became_a_rating_rejudges_every_row(self):
        rating_ev = {
            "name": "faithful",
            "system_prompt": "judge faithfulness",
            "judge_model": "openai/gpt-4.1",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
        }

        async def fake(_input, output, **kwargs):
            return {"faithful": {"reasoning": f"why {output}", "score": 4}}

        judge = AsyncMock(side_effect=fake)
        with tempfile.TemporaryDirectory() as out_dir:
            self._write_prior_run(
                out_dir,
                [BINARY_EV],
                {
                    "id": "row_a",
                    "input": "doc A",
                    "output": "sum A",
                    "faithful": True,
                    "faithful_reasoning": "pass/fail from the old run",
                },
            )
            await self._run(out_dir, [rating_ev], judge)

            self.assertEqual(judge.await_count, 2)
            df = pd.read_csv(os.path.join(out_dir, "results.csv"))
            self.assertEqual(list(df["faithful"]), [4, 4])
            with open(os.path.join(out_dir, "metrics.json")) as f:
                metrics = json.load(f)
            self.assertEqual(metrics["faithful"]["mean"], 4.0)

    async def test_widened_rating_range_rejudges_every_row(self):
        def rating(scale_max):
            return {
                "name": "faithful",
                "system_prompt": "judge faithfulness",
                "judge_model": "openai/gpt-4.1",
                "type": "rating",
                "scale_min": 1,
                "scale_max": scale_max,
            }

        async def fake(_input, output, **kwargs):
            return {"faithful": {"reasoning": f"why {output}", "score": 9}}

        judge = AsyncMock(side_effect=fake)
        with tempfile.TemporaryDirectory() as out_dir:
            self._write_prior_run(
                out_dir,
                [rating(5)],
                {
                    "id": "row_a",
                    "input": "doc A",
                    "output": "sum A",
                    "faithful": 5,
                    "faithful_reasoning": "top of the old range",
                },
            )
            await self._run(out_dir, [rating(10)], judge)

            self.assertEqual(judge.await_count, 2)
            df = pd.read_csv(os.path.join(out_dir, "results.csv"))
            self.assertEqual(list(df["faithful"]), [9, 9])

    async def test_reworded_prompt_still_reuses_stored_scores(self):
        reworded = dict(BINARY_EV, system_prompt="judge faithfulness, carefully")

        async def fake(_input, output, **kwargs):
            return {"faithful": {"reasoning": f"why {output}", "match": False}}

        judge = AsyncMock(side_effect=fake)
        with tempfile.TemporaryDirectory() as out_dir:
            self._write_prior_run(
                out_dir,
                [BINARY_EV],
                {
                    "id": "row_a",
                    "input": "doc A",
                    "output": "sum A",
                    "faithful": True,
                    "faithful_reasoning": "kept",
                },
            )
            await self._run(out_dir, [reworded], judge)

            self.assertEqual(judge.await_count, 1)
            df = pd.read_csv(os.path.join(out_dir, "results.csv"))
            self.assertEqual(list(df["faithful_reasoning"]), ["kept", "why sum B"])

    async def test_changed_output_text_is_judged_again(self):
        async def fake(_input, output, **kwargs):
            return {"faithful": {"reasoning": f"why {output}", "match": False}}

        judge = AsyncMock(side_effect=fake)
        with tempfile.TemporaryDirectory() as out_dir:
            self._write_prior_run(
                out_dir,
                [BINARY_EV],
                {
                    "id": "row_a",
                    "input": "doc A",
                    "output": "an older summary",
                    "faithful": True,
                    "faithful_reasoning": "scored the older summary",
                },
            )
            await self._run(out_dir, [BINARY_EV], judge)

            self.assertEqual(judge.await_count, 2)
            df = pd.read_csv(os.path.join(out_dir, "results.csv"))
            self.assertEqual(
                list(df["faithful_reasoning"]), ["why sum A", "why sum B"]
            )

    async def test_changed_input_text_is_judged_again(self):
        async def fake(_input, output, **kwargs):
            return {"faithful": {"reasoning": f"why {output}", "match": False}}

        judge = AsyncMock(side_effect=fake)
        with tempfile.TemporaryDirectory() as out_dir:
            self._write_prior_run(
                out_dir,
                [BINARY_EV],
                {
                    "id": "row_a",
                    "input": "an older document",
                    "output": "sum A",
                    "faithful": True,
                    "faithful_reasoning": "scored the older document",
                },
            )
            await self._run(out_dir, [BINARY_EV], judge)

            self.assertEqual(judge.await_count, 2)


class TestGeneralDuplicateIdTypes(unittest.TestCase):
    def test_ids_that_differ_only_in_type_are_duplicates(self):
        path = _write_json(
            [
                {"id": 1, "input": "a", "output": "b"},
                {"id": "1", "input": "c", "output": "d"},
            ]
        )
        try:
            ok, err, _ = validate_general_eval_dataset(path)
        finally:
            os.unlink(path)
        self.assertFalse(ok)
        self.assertIn("Duplicate row id", err)


if __name__ == "__main__":
    unittest.main()
