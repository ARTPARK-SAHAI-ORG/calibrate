import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from calibrate.llm._metrics_utils import _numeric_or_none


def generate_leaderboard(output_dir: str, save_dir: str) -> None:
    """
    Generate leaderboard from model results in output_dir.

    Expected structure:
        output_dir/
            model1/
                metrics.json  (contains {"total": N, "passed": M, "criteria": {...}})
            model2/
                metrics.json
            ...

    The leaderboard shows:
    - Overall `pass_rate` (all test cases)
    - Per-criterion pass rate columns when the suite has response-type tests
      (e.g., `accuracy`, `tone`)

    Args:
        output_dir: Directory containing model subdirectories with metrics.json files
        save_dir: Directory where leaderboard artifacts will be saved
    """
    base_path = Path(output_dir).expanduser().resolve()
    save_path = Path(save_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        raise FileNotFoundError(f"Output directory does not exist: {base_path}")

    # Find model directories (skip 'leaderboard' folder if present)
    model_dirs = sorted(
        p for p in base_path.iterdir()
        if p.is_dir() and p.name != "leaderboard"
    )

    if not model_dirs:
        print(f"No model folders found under {base_path}")
        return

    model_data: Dict[str, dict] = {}
    for model_dir in model_dirs:
        data = _read_metrics(model_dir / "metrics.json")
        if data is None:
            continue
        model_data[model_dir.name] = data

    if not model_data:
        print("No results found to compile.")
        return

    # Collect union of criterion names across all models (sorted for stable column order)
    criterion_names: List[str] = sorted(
        {
            name
            for data in model_data.values()
            for name in (data.get("criteria") or {}).keys()
        }
    )

    leaderboard_df = _build_leaderboard(model_data, criterion_names)
    csv_path = save_path / "llm_leaderboard.csv"
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"Saved leaderboard CSV to {csv_path}")


def _read_metrics(metrics_path: Path) -> Optional[dict]:
    if not metrics_path.exists():
        print(f"[WARN] metrics.json missing for {metrics_path.parent}")
        return None

    try:
        with metrics_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError:
        print(f"[WARN] Could not parse {metrics_path}")
        return None

    return data


def _to_percent(passed: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return (passed / total) * 100


def _build_leaderboard(
    model_data: Dict[str, dict],
    criterion_names: List[str],
) -> pd.DataFrame:
    """Build leaderboard DataFrame.

    Columns: model, passed, total, pass_rate, latency_p50, latency_p95,
    latency_p99, cost, total_tokens, [criterion_1, ...]

    ``latency_p50``/``latency_p95``/``latency_p99`` are percentiles of the
    per-test-case response-generation latency (in milliseconds, judge time
    excluded); ``None`` for runs without it (e.g. eval-only).

    ``cost`` is the mean per-test-case LLM cost in USD (``None`` when the run
    reported no cost, e.g. the OpenAI provider or external agents).

    ``total_tokens`` is the mean per-test-case total token usage (``None`` when
    the run reported no token counts, e.g. external agents that don't report
    usage or eval-only).

    Per-criterion column values:
    - binary criterion → pass_rate (%)
    - rating criterion → mean score (raw, on the criterion's scale)
    """
    rows = []
    for model_name in sorted(model_data):
        data = model_data[model_name]
        passed = int(data.get("passed", 0))
        total = int(data.get("total", 0))
        latency = data.get("latency_ms") if isinstance(data.get("latency_ms"), dict) else {}
        cost = data.get("cost") if isinstance(data.get("cost"), dict) else {}
        total_tokens = (
            data.get("total_tokens")
            if isinstance(data.get("total_tokens"), dict)
            else {}
        )
        row: Dict[str, object] = {
            "model": model_name,
            "passed": passed,
            "total": total,
            "pass_rate": _to_percent(passed, total),
            "latency_p50": latency.get("p50"),
            "latency_p95": latency.get("p95"),
            "latency_p99": latency.get("p99"),
            "cost": _numeric_or_none(cost.get("mean")),
            "total_tokens": _numeric_or_none(total_tokens.get("mean")),
        }

        criteria = data.get("criteria") or {}
        for name in criterion_names:
            crit = criteria.get(name)
            if not crit:
                row[name] = None
            elif crit.get("type") == "rating":
                row[name] = crit.get("mean")
            else:
                row[name] = crit.get("pass_rate")

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory with scenario subdirectories",
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        required=True,
        help="Directory where leaderboard artifacts will be stored",
    )
    args = parser.parse_args()
    generate_leaderboard(args.output_dir, args.save_dir)


if __name__ == "__main__":
    main()
