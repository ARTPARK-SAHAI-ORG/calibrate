import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

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

    model_dirs = sorted(p for p in base_path.iterdir() if p.is_dir())
    if not model_dirs:
        print(f"No model folders found under {base_path}")
        return

    model_results: Dict[str, Dict[str, float]] = {}
    metric_names: List[str] = []

    for model_dir in model_dirs:
        model_name = model_dir.name
        result = _read_metrics(model_dir / "metrics.json")
        if result is None:
            continue

        model_results[model_name] = result

        # Collect all metric names
        for metric_name in result.keys():
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    if not model_results:
        print("No results found to compile.")
        return

    leaderboard_df = _build_leaderboard(model_results, metric_names)
    csv_path = save_path / "simulation_leaderboard.csv"
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"Saved leaderboard CSV to {csv_path}")

    chart_path = save_path / "simulation_leaderboard.png"
    _create_comparison_chart(leaderboard_df, chart_path)


def _read_metrics(metrics_path: Path) -> Optional[Dict[str, float]]:
    if not metrics_path.exists():
        print(f"[WARN] metrics.json missing for {metrics_path.parent}")
        return None

    try:
        with metrics_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError:
        print(f"[WARN] Could not parse {metrics_path}")
        return None

    # Extract mean values for each metric
    result: Dict[str, float] = {}
    for metric_name, metric_data in data.items():
        if isinstance(metric_data, dict) and "mean" in metric_data:
            result[metric_name] = metric_data["mean"] * 100  # Convert to percentage
        elif isinstance(metric_data, (int, float)):
            result[metric_name] = float(metric_data) * 100

    return result


def _build_leaderboard(
    model_results: Dict[str, Dict[str, float]],
    metric_names: List[str],
) -> pd.DataFrame:
    rows = []
    for model_name in sorted(model_results):
        row: Dict[str, Optional[float]] = {"model": model_name}
        for metric in metric_names:
            row[metric] = model_results[model_name].get(metric)

        # Calculate overall average across all metrics
        values = [v for v in row.values() if isinstance(v, (int, float))]
        row["overall"] = sum(values) / len(values) if values else None
        rows.append(row)

    return pd.DataFrame(rows)


def _create_comparison_chart(df: pd.DataFrame, chart_path: Path) -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required to generate charts. Please install it."
        )

    if df.empty:
        print("Leaderboard dataframe is empty, skipping chart creation.")
        return

    metric_columns = [col for col in df.columns if col not in ["model", "overall"]]
    if not metric_columns:
        print("No metrics available for charting.")
        return

    plot_df = df.set_index("model")[metric_columns].T

    fig, ax = plt.subplots(figsize=(max(8, len(metric_columns) * 1.5), 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score %")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 105)
    ax.set_title("Model Score by Metric")
    ax.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
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
        help="Path to the output directory with model subdirectories",
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
