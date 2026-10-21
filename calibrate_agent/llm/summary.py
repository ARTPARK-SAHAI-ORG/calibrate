"""Pass rates, per-evaluator totals and their 95% likely ranges for LLM test runs.

Standard library only, so a caller that reads stored results can import it
without loading pipecat, pandas or numpy.
"""

import math
import statistics
from collections import defaultdict
from typing import List, Optional

Z_95 = 1.96

DEFAULT_LLM_TEST_EVALUATOR_NAME = "correctness"

# Test case types graded by the evaluators named in ``evaluation.criteria``;
# ``tool_call`` cases are graded against an expected tool-call list instead.
EVALUATOR_TEST_TYPES = ("response", "conversation", "general")


def wilson_interval(passed: int, total: int) -> Optional[dict]:
    """95% Wilson score interval for ``passed`` out of ``total``, on the 0-100 scale."""
    if total <= 0:
        return None
    p = passed / total
    z2 = Z_95 * Z_95
    denom = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    half = Z_95 * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denom
    return {
        "low": max(0.0, (center - half) * 100),
        "high": min(100.0, (center + half) * 100),
    }


def mean_interval(
    scores: List[float], scale_min: float, scale_max: float
) -> Optional[dict]:
    """95% interval around the mean of ``scores``, clamped to the rating scale.

    ``None`` with fewer than two scores or when every score is the same.
    """
    if len(set(scores)) < 2:
        return None
    mean = statistics.fmean(scores)
    half = Z_95 * statistics.stdev(scores) / math.sqrt(len(scores))
    return {
        "low": max(scale_min, mean - half),
        "high": min(scale_max, mean + half),
    }


NO_ANSWER = "No test case got an answer"


def _percent(count: int, answered: int) -> Optional[float]:
    return count / answered * 100 if answered > 0 else None


def _pass_rate(passed: int, answered: int) -> dict:
    return {
        "pass_rate": _percent(passed, answered),
        "pass_rate_ci": wilson_interval(passed, answered),
    }


def pass_rate_summary(results: List[dict]) -> dict:
    """Pass rate and its likely range over the test cases that got an answer.

    A row marked ``error: True`` got no answer and is left out. An empty reply
    is a real answer that failed.
    """
    answered = [r for r in results if not r.get("error")]
    passed = sum(1 for r in answered if (r.get("metrics") or {}).get("passed"))
    return _pass_rate(passed, len(answered))


def answered_count(metrics: dict) -> int:
    """Test cases in a ``metrics.json`` block that got an answer."""
    return int(metrics.get("total", 0)) - int(metrics.get("errored", 0))


def pass_rate_from_metrics(metrics: dict) -> dict:
    """:func:`pass_rate_summary` from the counts in a ``metrics.json`` block."""
    return _pass_rate(int(metrics.get("passed", 0)), answered_count(metrics))


def format_share(count: int, answered: int) -> str:
    """``count/answered (rate%)``, or :data:`NO_ANSWER` when nothing was answered."""
    rate = _percent(count, answered)
    if rate is None:
        return NO_ANSWER
    return f"{count}/{answered} ({rate:.1f}%)"


def format_pass_rate(metrics: dict) -> str:
    """:func:`format_share` of the passed cases in a ``metrics.json`` block."""
    return format_share(int(metrics.get("passed", 0)), answered_count(metrics))


def _normalize_criteria_refs(criteria) -> list[dict]:
    """Coerce a test case's ``evaluation.criteria`` into a list of evaluator refs.

    Accepted shapes:
    - ``str`` → ``[{"name": DEFAULT_LLM_TEST_EVALUATOR_NAME, "arguments": {"criteria": <str>}}]``
    - ``list[{name, arguments}]`` → returned as-is

    Any other shape raises ``ValueError``.
    """
    if isinstance(criteria, str):
        return [
            {
                "name": DEFAULT_LLM_TEST_EVALUATOR_NAME,
                "arguments": {"criteria": criteria},
            }
        ]
    if isinstance(criteria, list):
        for entry in criteria:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError(
                    "evaluation.criteria entries must be dicts with a 'name' key "
                    "(and optional 'arguments'); got: " + repr(entry)
                )
        return criteria
    raise ValueError(
        f"evaluation.criteria must be a string or list of evaluator refs; got {type(criteria).__name__}"
    )


def aggregate_criteria(results: List[dict], name_to_evaluator: dict) -> dict:
    """Aggregate per-evaluator metrics across test case results.

    Each ``response`` / ``conversation`` / ``general`` test case contributes to
    the totals for the evaluators referenced in its ``evaluation.criteria``.
    ``name_to_evaluator`` maps evaluator name to its config dict (type/scale).

    Per-evaluator output shape depends on the evaluator's type:
    - binary: ``{"type": "binary", "passed": int, "total": int, "pass_rate": float,
                  "ci": {"low", "high"}}`` (``pass_rate`` and ``ci`` on 0-100)
    - rating: ``{"type": "rating", "mean": float, "min": int, "max": int,
                  "count": int, "scale_min": int, "scale_max": int,
                  "ci": {"low", "high"} | None}``
    """
    binary_totals: defaultdict = defaultdict(lambda: {"passed": 0, "total": 0})
    rating_scores: defaultdict = defaultdict(list)
    rating_scale: dict = {}

    for result in results:
        metrics = result.get("metrics", {})
        evaluation = result.get("test_case", {}).get("evaluation", {})

        if evaluation.get("type") not in EVALUATOR_TEST_TYPES:
            continue

        judge_results = metrics.get("judge_results")
        if not judge_results:
            continue

        for ref in _normalize_criteria_refs(evaluation.get("criteria")):
            name = ref["name"]
            ev = name_to_evaluator.get(name)
            if ev is None:
                continue
            ev_data = judge_results.get(name, {})
            if ev.get("type") == "rating":
                if "score" in ev_data:
                    rating_scores[name].append(int(ev_data["score"]))
                    rating_scale[name] = (int(ev["scale_min"]), int(ev["scale_max"]))
            else:
                binary_totals[name]["total"] += 1
                if ev_data.get("match"):
                    binary_totals[name]["passed"] += 1

    aggregated: dict = {}
    for name, c in binary_totals.items():
        aggregated[name] = {
            "type": "binary",
            "passed": c["passed"],
            "total": c["total"],
            "pass_rate": (c["passed"] / c["total"]) * 100,
            "ci": wilson_interval(c["passed"], c["total"]),
        }
    for name, scores in rating_scores.items():
        lo, hi = rating_scale[name]
        aggregated[name] = {
            "type": "rating",
            "mean": float(sum(scores) / len(scores)),
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
            "scale_min": lo,
            "scale_max": hi,
            "ci": mean_interval(scores, lo, hi),
        }
    for name, entry in aggregated.items():
        ev = name_to_evaluator.get(name)
        if ev and "id" in ev:
            entry["evaluator_id"] = ev["id"]
    return aggregated
