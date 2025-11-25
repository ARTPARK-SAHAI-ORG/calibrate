import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None


def generate_leaderboard(output_dir: str, save_dir: str) -> None:
    base_path = Path(output_dir).expanduser().resolve()
    save_path = Path(save_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        raise FileNotFoundError(f"Output directory does not exist: {base_path}")

    scenario_dirs = sorted(p for p in base_path.iterdir() if p.is_dir())
    if not scenario_dirs:
        print(f"No scenario folders found under {base_path}")
        return

    model_results: Dict[str, Dict[str, float]] = {}
    overall_totals: Dict[str, Dict[str, int]] = {}
    scenario_names: List[str] = []

    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        scenario_names.append(scenario_name)
        model_dirs = sorted(p for p in scenario_dir.iterdir() if p.is_dir())

        if not model_dirs:
            print(f"[WARN] No model folders found for scenario {scenario_name}")
            continue

        for model_dir in model_dirs:
            result = _read_results(model_dir / "metrics.json")
            if result is None:
                continue

            passed, total = result
            pass_percent = _to_percent(passed, total)
            if pass_percent is None:
                continue

            model_name = model_dir.name
            model_results.setdefault(model_name, {})[scenario_name] = pass_percent
            totals = overall_totals.setdefault(model_name, {"passed": 0, "total": 0})
            totals["passed"] += passed
            totals["total"] += total

    if not model_results:
        print("No results found to compile.")
        return

    leaderboard_df = _build_leaderboard(model_results, scenario_names, overall_totals)
    csv_path = save_path / "llm_leaderboard.csv"
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"Saved leaderboard CSV to {csv_path}")

    chart_path = save_path / "llm_leaderboard.png"
    _create_comparison_chart(leaderboard_df, chart_path)


def _read_results(results_path: Path) -> Optional[Tuple[int, int]]:
    if not results_path.exists():
        print(f"[WARN] metrics.json missing for {results_path.parent}")
        return None

    try:
        with results_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError:
        print(f"[WARN] Could not parse {results_path}")
        return None

    total = int(data.get("total", 0))
    passed = int(data.get("passed", 0))
    return passed, total


def _build_leaderboard(
    model_results: Dict[str, Dict[str, float]],
    scenario_names: List[str],
    overall_totals: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    # Deduplicate scenarios while maintaining order
    seen: set[str] = set()
    ordered_scenarios: List[str] = []
    for scenario in scenario_names:
        if scenario not in seen:
            ordered_scenarios.append(scenario)
            seen.add(scenario)

    rows = []
    for model_name in sorted(model_results):
        row: Dict[str, Optional[float]] = {"model": model_name}
        for scenario in ordered_scenarios:
            row[scenario] = model_results[model_name].get(scenario)

        totals = overall_totals.get(model_name, {"passed": 0, "total": 0})
        row["overall"] = _to_percent(totals["passed"], totals["total"])
        rows.append(row)

    return pd.DataFrame(rows)


def _to_percent(passed: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return (passed / total) * 100


def _create_comparison_chart(df: pd.DataFrame, chart_path: Path) -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required to generate charts. Please install it."
        )

    if df.empty:
        print("Leaderboard dataframe is empty, skipping chart creation.")
        return

    metric_columns = [col for col in df.columns if col != "model"]
    if not metric_columns:
        print("No scenarios available for charting.")
        return

    plot_df = df.set_index("model")[metric_columns].T

    fig, ax = plt.subplots(figsize=(max(8, len(metric_columns) * 1.2), 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Pass %")
    ax.set_xlabel("Scenario")
    ax.set_ylim(0, 105)
    ax.set_title("Model Pass Percentage by Scenario")
    ax.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison chart to {chart_path}")


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
