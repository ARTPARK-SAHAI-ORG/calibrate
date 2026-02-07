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
    """
    Generate leaderboard from model results in output_dir.
    
    Expected structure:
        output_dir/
            model1/
                metrics.json  (contains {"total": N, "passed": M})
            model2/
                metrics.json
            ...
    
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

    model_results: Dict[str, Dict[str, float]] = {}
    overall_totals: Dict[str, Dict[str, int]] = {}

    for model_dir in model_dirs:
        result = _read_results(model_dir / "metrics.json")
        if result is None:
            continue

        passed, total = result
        pass_percent = _to_percent(passed, total)
        if pass_percent is None:
            continue

        model_name = model_dir.name
        model_results[model_name] = {"pass_rate": pass_percent}
        overall_totals[model_name] = {"passed": passed, "total": total}

    if not model_results:
        print("No results found to compile.")
        return

    leaderboard_df = _build_leaderboard_flat(model_results, overall_totals)
    csv_path = save_path / "llm_leaderboard.csv"
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"Saved leaderboard CSV to {csv_path}")

    chart_path = save_path / "llm_leaderboard.png"
    _create_comparison_chart_flat(leaderboard_df, chart_path)


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


def _build_leaderboard_flat(
    model_results: Dict[str, Dict[str, float]],
    overall_totals: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    """Build leaderboard dataframe for flat model structure."""
    rows = []
    for model_name in sorted(model_results):
        totals = overall_totals.get(model_name, {"passed": 0, "total": 0})
        row = {
            "model": model_name,
            "passed": totals["passed"],
            "total": totals["total"],
            "pass_rate": _to_percent(totals["passed"], totals["total"]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _to_percent(passed: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return (passed / total) * 100


def _create_comparison_chart_flat(df: pd.DataFrame, chart_path: Path) -> None:
    """Create comparison chart for flat model structure."""
    if plt is None:
        raise ImportError(
            "matplotlib is required to generate charts. Please install it."
        )

    if df.empty:
        print("Leaderboard dataframe is empty, skipping chart creation.")
        return

    if "pass_rate" not in df.columns:
        print("No pass_rate column available for charting.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.5), 5))
    
    models = df["model"].tolist()
    pass_rates = df["pass_rate"].tolist()
    
    bars = ax.bar(models, pass_rates, color="steelblue")
    
    # Add value labels on bars
    for bar, rate in zip(bars, pass_rates):
        if rate is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    
    ax.set_ylabel("Pass Rate (%)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 105)
    ax.set_title("LLM Test Pass Rate by Model")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    # Rotate x-axis labels if many models
    if len(models) > 3:
        plt.xticks(rotation=45, ha="right")
    
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
