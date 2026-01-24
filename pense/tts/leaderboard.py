import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None

TARGET_METRICS = [
    "llm_judge_score",
    "ttfb",
]
INVALID_SHEET_CHARS = set("[]:*?/\\")


def generate_leaderboard(output_dir: str, save_dir: str) -> None:
    base_path = Path(output_dir).expanduser().resolve()
    save_path = Path(save_dir).expanduser().resolve()

    save_path.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        raise FileNotFoundError(f"Output directory does not exist: {base_path}")

    run_dirs = sorted(p for p in base_path.iterdir() if p.is_dir())
    if not run_dirs:
        print(f"No run folders found under {base_path}")
        return

    summary_rows: List[Dict[str, object]] = []
    run_results: Dict[str, pd.DataFrame] = {}

    for run_dir in run_dirs:
        metrics = _read_metrics(run_dir / "metrics.json")
        results_df = _read_results(run_dir / "results.csv")

        row = {"run": run_dir.name, "count": len(results_df)}
        for metric in TARGET_METRICS:
            row[metric] = metrics.get(metric)

        summary_rows.append(row)
        run_results[run_dir.name] = results_df

    summary_df = pd.DataFrame(summary_rows)

    _create_metric_charts(summary_df, save_path)
    workbook_path = save_path / "tts_leaderboard.xlsx"
    _write_workbook(summary_df, run_results, workbook_path)

    print(f"Saved leaderboard workbook to {workbook_path}")


def _read_metrics(metrics_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not metrics_path.exists():
        print(f"[WARN] metrics.json missing for {metrics_path.parent.name}")
        return metrics

    with metrics_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    # Handle new dict format (preferred)
    if isinstance(data, dict) and "metric_name" not in data:
        for key, value in data.items():
            if isinstance(value, dict) and "mean" in value:
                # Nested dict with mean/std/values (e.g., ttfb, processing_time)
                metrics[key] = value["mean"]
            elif isinstance(value, (int, float)):
                # Simple float value (e.g., llm_judge_score)
                metrics[key] = float(value)
        return metrics

    # Handle legacy list format for backwards compatibility
    if isinstance(data, dict):
        data = [data]

    for entry in data:
        if not isinstance(entry, dict):
            continue

        metric_name = entry.get("metric_name")
        if metric_name:
            metrics[metric_name] = entry["mean"]
            continue

        for key, value in entry.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

    return metrics


def _read_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        print(f"[WARN] results.csv missing for {results_path.parent.name}")
        return pd.DataFrame()
    return pd.read_csv(results_path)


def _create_metric_charts(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required to generate charts. Please install it."
        )

    available_metrics = [m for m in TARGET_METRICS if m in summary_df.columns]
    if not available_metrics:
        print("No metrics available to plot.")
        return

    import numpy as np  # local import to avoid dependency if charts not needed

    runs = summary_df["run"].tolist()
    total_runs = len(runs)

    if total_runs == 0:
        print("No runs available to plot.")
        return

    # Create a separate chart for each metric
    for metric in available_metrics:
        metric_values = summary_df[metric].tolist()

        # Skip if all values are NaN
        if all(pd.isna(v) for v in metric_values):
            print(f"Skipping {metric} chart - no values available.")
            continue

        # Replace NaN with 0 for plotting
        metric_values = [0 if pd.isna(v) else v for v in metric_values]

        fig, ax = plt.subplots(figsize=(max(6, total_runs * 0.8), 5))

        x = np.arange(total_runs)
        bar_width = 0.6

        bars = ax.bar(x, metric_values, width=bar_width, color="steelblue")

        # Add value labels on top of bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            # Format as integer if it's a whole number, otherwise show decimals
            if value == int(value):
                label = f"{int(value)}"
            else:
                label = f"{value:.4f}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        metric_title = metric.replace("_", " ").title()
        ax.set_title(f"{metric_title} by Provider")
        ax.set_ylabel(metric_title)
        ax.set_xlabel("Provider")
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=45, ha="right")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()

        chart_path = output_dir / f"{metric}.png"
        fig.savefig(chart_path, dpi=300)
        plt.close(fig)

        print(f"Saved {metric} chart at {chart_path}")


def _write_workbook(
    summary_df: pd.DataFrame, run_results: Dict[str, pd.DataFrame], workbook_path: Path
) -> None:
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names: Set[str] = set()

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)

        for run_name, df in run_results.items():
            if "llm_judge_score" not in df.columns:
                continue

            df = df[~df["llm_judge_score"]]
            sheet_name = _unique_sheet_name(run_name, sheet_names)
            if df.empty:
                pd.DataFrame({"info": ["No results.csv found"]}).to_excel(
                    writer, sheet_name=sheet_name, index=False
                )
            else:
                df.to_excel(writer, sheet_name=sheet_name, index=False)


def _unique_sheet_name(run_name: str, existing: Set[str]) -> str:
    sanitized = "".join("_" if ch in INVALID_SHEET_CHARS else ch for ch in run_name)
    sanitized = sanitized.strip() or "run"
    sanitized = sanitized[:31]

    candidate = sanitized
    suffix = 1
    while candidate in existing:
        trimmed = sanitized[: 31 - (len(str(suffix)) + 1)]
        candidate = f"{trimmed}_{suffix}"
        suffix += 1

    existing.add(candidate)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory which has directories for each provider",
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        required=True,
        help="Path to the directory where the leaderboard results will be saved",
    )
    args = parser.parse_args()
    generate_leaderboard(args.output_dir, args.save_dir)


if __name__ == "__main__":
    main()
