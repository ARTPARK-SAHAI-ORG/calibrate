"""Unit tests for calibrate/general/eval.py.

Covers dataset validation, evaluator resolution from config, and the
end-to-end run_general_eval path (with the judge mocked) producing
metrics.json + results.csv.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, AsyncMock

import pandas as pd

from calibrate.general.eval import (
    validate_general_eval_dataset,
    _resolve_evaluators,
    run_general_eval,
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


if __name__ == "__main__":
    unittest.main()
