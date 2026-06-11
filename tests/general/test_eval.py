"""Unit tests for calibrate/general/eval.py.

Covers dataset validation, evaluator resolution from config, and the
end-to-end run_general_eval path (with the judge mocked) producing
metrics.json + results.csv.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, AsyncMock

import pandas as pd

from calibrate.general.eval import (
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
                    "calibrate.general.eval.get_general_judge_score",
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
                    "calibrate.general.eval.get_general_judge_score",
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
                    "calibrate.general.eval.get_general_judge_score", judge_mock
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


class TestMain(unittest.IsolatedAsyncioTestCase):
    """Cover the CLI entry point branches of calibrate.general.eval.main()."""

    def _argv(self, dataset, config, out):
        return ["calibrate", "--dataset", dataset, "-c", config, "-o", out]

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
                    "calibrate.general.eval.run_general_eval",
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
                    "calibrate.general.eval.run_general_eval", run_mock
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

    async def test_success_with_no_scores_prints_placeholder(self):
        with tempfile.TemporaryDirectory() as out_dir:
            ds = _write_json([{"id": "1", "input": "a", "output": "b"}])
            cfg = _write_json({"evaluators": [BINARY_EV]})
            # metrics with no type-bearing dicts → the "(no scores)" branch
            completed = {"status": "completed", "metrics": {}, "output_dir": out_dir}
            try:
                with patch.object(sys, "argv", self._argv(ds, cfg, out_dir)), patch(
                    "calibrate.general.eval.run_general_eval",
                    AsyncMock(return_value=completed),
                ):
                    await eval_main()  # should not raise
            finally:
                os.unlink(ds)
                os.unlink(cfg)


class TestCliDispatch(unittest.TestCase):
    """Cover the `general` branch wired into calibrate.cli.main().

    Plain (sync) TestCase because ``cli.main()`` calls ``asyncio.run()`` itself,
    which cannot nest inside a running event loop.
    """

    def test_dispatch_invokes_eval_main(self):
        from calibrate import cli

        eval_main_mock = AsyncMock(return_value=None)
        argv = ["calibrate", "general", "--dataset", "d.json", "-c", "c.json"]
        with patch.object(sys, "argv", argv), patch(
            "calibrate.general.eval.main", eval_main_mock
        ):
            cli.main()
        eval_main_mock.assert_awaited_once()

    def test_dispatch_missing_dataset_exits(self):
        from calibrate import cli

        argv = ["calibrate", "general", "-c", "c.json"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 1)

    def test_dispatch_missing_config_exits(self):
        from calibrate import cli

        argv = ["calibrate", "general", "--dataset", "d.json"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
