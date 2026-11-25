"""General task evaluation runner + CLI.

File-based pathway for the general (non-conversational) task judge. Reads a
JSON dataset of ``{id, input, output}`` rows plus a list of evaluators, runs
the judge over every row, and writes ``results.csv`` + ``metrics.json``.

Invoked via ``calibrate-agent general --dataset data.json --config config.json``.
"""

import argparse
import asyncio
import json
import os
import sys
from os.path import exists, join
from typing import List, Optional

import pandas as pd

from calibrate_agent.general.metrics import get_general_judge_score
from calibrate_agent.judges import (
    PartialResultsWriter,
    evaluator_row_columns,
    require_unique_evaluator_names,
    write_evaluator_config,
)
from calibrate_agent.utils import (
    provider_log as _log,
    provider_log_file as _current_log_file,
)


# Required fields for every dataset row.
GENERAL_DATASET_FIELDS = ("id", "input", "output")


def validate_general_eval_dataset(
    dataset_path: str,
    evaluator_names: Optional[set] = None,
) -> tuple[bool, str, list[dict]]:
    """Validate a general eval dataset JSON file.

    Expected format: a JSON list of objects, each with ``id``, ``input`` and
    ``output`` fields. Each row may optionally carry an ``arguments`` object
    keyed by evaluator ``name`` → that evaluator's variable dict, used to render
    that evaluator's ``{{var}}`` placeholders (mirroring ``llm`` per-criteria
    ``arguments``). If present, ``arguments`` and each of its values must be
    objects.

    A row may also carry ``evaluators`` (the names of the evaluators to run on
    it; omitted means all) and ``tool_calls`` / ``expected_tool_calls`` lists,
    which make the row judged on the tool calls it made. ``evaluator_names``,
    when given, is the set of known names an ``evaluators`` list is checked
    against.

    Returns:
        tuple[bool, str, list[dict]]: (is_valid, error_message, parsed_rows)
    """
    if not exists(dataset_path):
        return False, f"Dataset file does not exist: {dataset_path}", []

    try:
        with open(dataset_path) as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Failed to parse dataset JSON: {e}", []

    if not isinstance(data, list):
        return False, "Dataset must be a JSON list of objects", []

    if not data:
        return False, "Dataset is empty — provide at least one row", []

    required = set(GENERAL_DATASET_FIELDS)
    seen_ids: set = set()
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            return False, f"Row {i} is not an object", []
        missing = required - row.keys()
        if missing:
            return (
                False,
                f"Row {i} missing required fields: {sorted(missing)}. "
                f"Each row needs 'id', 'input', 'output'.",
                [],
            )
        if "arguments" in row:
            row_args = row["arguments"]
            if not isinstance(row_args, dict):
                return False, f"Row {i} field 'arguments' must be an object", []
            for ev_name, ev_args in row_args.items():
                if not isinstance(ev_args, dict):
                    return (
                        False,
                        f"Row {i} field 'arguments[{ev_name!r}]' must be an "
                        f"object mapping variable names to values",
                        [],
                    )
        if "evaluators" in row:
            names = row["evaluators"]
            if (
                not isinstance(names, list)
                or not names
                or not all(isinstance(n, str) for n in names)
            ):
                return (
                    False,
                    f"Row {i} field 'evaluators' must be a non-empty list of "
                    f"evaluator names",
                    [],
                )
            if evaluator_names is not None:
                unknown = [n for n in names if n not in evaluator_names]
                if unknown:
                    return False, f"Row {i} names unknown evaluators: {unknown}", []
        for key in ("tool_calls", "expected_tool_calls"):
            if key in row and not isinstance(row[key], list):
                return False, f"Row {i} field '{key}' must be a list", []
            if key in row and not all(isinstance(tc, dict) for tc in row[key]):
                return False, f"Row {i} field '{key}' entries must be objects", []
        row_id = row["id"]
        if row_id in seen_ids:
            return False, f"Duplicate row id: {row_id!r}", []
        seen_ids.add(row_id)

    return True, "", data


def _resolve_evaluators(config: Optional[dict]) -> List[dict]:
    """Pull the evaluator list out of a config dict, validating it.

    The general task judge has no implicit default, so the config must define a
    non-empty ``evaluators`` list. Raises ``ValueError`` otherwise.
    """
    evaluators = (config or {}).get("evaluators")
    if not isinstance(evaluators, list) or len(evaluators) == 0:
        raise ValueError(
            "Config must define a non-empty `evaluators` list. Each evaluator "
            "needs a `name` and `system_prompt` (general task evaluation has no "
            "implicit default)."
        )
    for ev in evaluators:
        if not isinstance(ev, dict) or "name" not in ev or "system_prompt" not in ev:
            raise ValueError(
                "Each evaluator must be a dict with 'name' and 'system_prompt' "
                "(got: " + repr(ev) + ")"
            )
    require_unique_evaluator_names(evaluators)
    return evaluators


async def _score_and_write_results(
    ids: list,
    inputs: List[Optional[str]],
    outputs: List[str],
    evaluators: List[dict],
    output_dir: str,
    arguments_list: Optional[List[Optional[dict]]] = None,
    evaluator_names_list: Optional[List[Optional[List[str]]]] = None,
    tool_calls_list: Optional[List[Optional[dict]]] = None,
) -> dict:
    """Run the general judge over (input, output) pairs and write outputs.

    Writes ``results.csv`` and ``metrics.json`` plus the resolved evaluator
    ``config.json`` under ``output_dir``. Returns the metrics_data dict.

    Each row is appended to ``results.csv`` as its judge result arrives, so a
    run killed part-way still leaves the graded rows on disk. Nothing else owns
    that file here — the general task has no inference step and no resume, so
    the judge is free to write it from the first row onwards.

    Args:
        arguments_list: Optional per-row ``arguments`` dicts (one entry per
            row, ``None`` for rows with no injection) forwarded to the judge to
            render evaluator ``{{var}}`` placeholders.
    """
    write_evaluator_config(output_dir, evaluators)

    evaluators_by_name = {ev["name"]: ev for ev in evaluators}
    partial_writer = PartialResultsWriter(
        join(output_dir, "results.csv"),
        evaluators_by_name,
        [
            {"id": _id, "input": input_text, "output": output_text}
            for _id, input_text, output_text in zip(ids, inputs, outputs)
        ],
    )
    try:
        llm_results = await get_general_judge_score(
            inputs,
            outputs,
            evaluators=evaluators,
            arguments_list=arguments_list,
            on_row=partial_writer.write,
            evaluator_names_list=evaluator_names_list,
            tool_calls_list=tool_calls_list,
        )
    finally:
        partial_writer.close()
    for name, score_dict in llm_results["scores"].items():
        _log(f"  {name}: {score_dict['mean']:.4f}")

    metrics_data: dict = {}
    for name, score_dict in llm_results["scores"].items():
        metrics_data[name] = score_dict

    data = []
    for _id, input_text, output_text, llm_row in zip(
        ids, inputs, outputs, llm_results["per_row"]
    ):
        row = {
            "id": _id,
            "input": input_text,
            "output": output_text,
        }
        row.update(evaluator_row_columns(evaluators_by_name, llm_row))
        data.append(row)

    with open(join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)

    pd.DataFrame(data).to_csv(join(output_dir, "results.csv"), index=False)

    return metrics_data


async def run_general_eval(
    dataset_path: str,
    output_dir: str,
    evaluators: List[dict],
) -> dict:
    """Run general task evaluators on a dataset of ``{id, input, output}`` rows.

    Writes ``metrics.json`` and ``results.csv`` under ``output_dir``.

    Args:
        dataset_path: Path to a JSON file with a list of {"id", "input",
            "output"} rows.
        output_dir: Directory to write results and metrics.
        evaluators: List of evaluator dicts (each with ``name`` and
            ``system_prompt``).

    Returns:
        dict with ``status`` and, on success, ``metrics`` and ``output_dir``.
    """
    os.makedirs(output_dir, exist_ok=True)

    log_save_path = join(output_dir, "logs")
    if exists(log_save_path):
        os.remove(log_save_path)

    token = _current_log_file.set(log_save_path)
    try:
        _log("--------------------------------")
        _log("\033[33mRunning general task evaluation on dataset\033[0m")
        _log(f"Dataset: {dataset_path}")

        is_valid, error_msg, rows = validate_general_eval_dataset(
            dataset_path, {ev["name"] for ev in evaluators}
        )
        if not is_valid:
            _log(f"\033[31mError: {error_msg}\033[0m")
            return {"status": "error", "error": error_msg}

        ids = [r["id"] for r in rows]
        inputs = [str(r["input"]) if r["input"] is not None else "" for r in rows]
        outputs = [str(r["output"]) if r["output"] is not None else "" for r in rows]
        arguments_list = [r.get("arguments") for r in rows]

        metrics_data = await _score_and_write_results(
            ids=ids,
            inputs=inputs,
            outputs=outputs,
            evaluators=evaluators,
            output_dir=output_dir,
            arguments_list=arguments_list,
            evaluator_names_list=[r.get("evaluators") for r in rows],
            tool_calls_list=[
                {
                    "tool_calls": r.get("tool_calls") or [],
                    "expected_tool_calls": r.get("expected_tool_calls"),
                }
                if "tool_calls" in r or "expected_tool_calls" in r
                else None
                for r in rows
            ],
        )

        return {
            "status": "completed",
            "metrics": metrics_data,
            "output_dir": output_dir,
        }
    finally:
        _current_log_file.reset(token)


async def main():
    """CLI entry point for general task evaluation."""
    parser = argparse.ArgumentParser(
        description="Run general (non-conversational) task evaluation on a dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSON (list of {id, input, output})",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file defining the `evaluators` list",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./out",
        help="Path to the output directory to save the results",
    )

    args = parser.parse_args()

    if not exists(args.config):
        print(f"\033[31mError: config file not found: {args.config}\033[0m")
        sys.exit(1)

    try:
        with open(args.config) as f:
            config = json.load(f)
    except Exception as e:
        print(f"\033[31mError: failed to parse config JSON: {e}\033[0m")
        sys.exit(1)

    try:
        evaluators = _resolve_evaluators(config)
    except ValueError as e:
        print(f"\033[31mError: {e}\033[0m")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n\033[91mGeneral Task Evaluation\033[0m\n")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print("")

    result = await run_general_eval(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        evaluators=evaluators,
    )

    print(f"\n\033[92m{'='*60}\033[0m")
    print(f"\033[92mSummary\033[0m")
    print(f"\033[92m{'='*60}\033[0m\n")

    if result.get("status") == "error":
        print(f"  \033[31mError - {result.get('error')}\033[0m")
        sys.exit(1)

    metrics = result.get("metrics", {})
    # Evaluator entries are dicts carrying a ``type`` field; that's the marker
    # used to pick them out from any other top-level metrics.
    judge_scores = {
        k: v["mean"]
        for k, v in metrics.items()
        if isinstance(v, dict) and "type" in v
    }
    judge_str = ", ".join(f"{k}={v:.4f}" for k, v in judge_scores.items())
    print(f"  {judge_str}" if judge_str else "  (no scores)")
    print(f"\n  Results written to {result.get('output_dir')}")


if __name__ == "__main__":
    asyncio.run(main())
