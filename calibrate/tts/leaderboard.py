"""
TTS Leaderboard Generation

Generates comparison leaderboard from TTS evaluation results.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

LEADERBOARD_METRICS = ["llm_judge_score", "ttfb"]
INVALID_SHEET_CHARS = set("[]:*?/\\")


def generate_leaderboard(output_dir: str, save_dir: str | None = None) -> str:
    """Generate leaderboard comparing all provider results in output_dir.

    Args:
        output_dir: Directory containing provider result subdirectories
        save_dir: Directory to save leaderboard files (defaults to output_dir/leaderboard)

    Returns:
        Path to the leaderboard directory
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    base_path = Path(output_dir).expanduser().resolve()

    if save_dir is None:
        save_path = base_path / "leaderboard"
    else:
        save_path = Path(save_dir).expanduser().resolve()

    save_path.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        raise FileNotFoundError(f"Output directory does not exist: {base_path}")

    run_dirs = sorted(
        p for p in base_path.iterdir() if p.is_dir() and p.name != "leaderboard"
    )
    if not run_dirs:
        print(f"No provider folders found under {base_path}")
        return str(save_path)

    summary_rows = []
    run_results = {}

    for run_dir in run_dirs:
        metrics = _read_leaderboard_metrics(run_dir / "metrics.json")
        results_df = _read_leaderboard_results(run_dir / "results.csv")

        row = {"run": run_dir.name, "count": len(results_df)}
        for metric in LEADERBOARD_METRICS:
            row[metric] = metrics.get(metric)

        summary_rows.append(row)
        run_results[run_dir.name] = results_df

    summary_df = pd.DataFrame(summary_rows)

    if plt is not None:
        _create_leaderboard_charts(summary_df, save_path, plt)
    else:
        print("matplotlib not available, skipping chart generation")

    workbook_path = save_path / "tts_leaderboard.xlsx"
    _write_leaderboard_workbook(summary_df, run_results, workbook_path)
    print(f"Saved leaderboard workbook to {workbook_path}")

    return str(save_path)


def _read_leaderboard_metrics(metrics_path: Path) -> dict:
    """Read metrics from metrics.json file."""
    if not metrics_path.exists():
        print(f"[WARN] metrics.json missing for {metrics_path.parent.name}")
        return {}

    with metrics_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    metrics = {}
    if isinstance(data, dict) and "metric_name" not in data:
        for key, value in data.items():
            if isinstance(value, dict) and "mean" in value:
                metrics[key] = value["mean"]
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
        return metrics

    # Legacy format
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


def _read_leaderboard_results(results_path: Path) -> pd.DataFrame:
    """Read results from results.csv file."""
    if not results_path.exists():
        print(f"[WARN] results.csv missing for {results_path.parent.name}")
        return pd.DataFrame()
    return pd.read_csv(results_path)


def _create_leaderboard_charts(summary_df: pd.DataFrame, output_dir: Path, plt) -> None:
    """Create bar charts for each metric."""
    import numpy as np

    available_metrics = [m for m in LEADERBOARD_METRICS if m in summary_df.columns]
    if not available_metrics:
        print("No metrics available to plot.")
        return

    runs = summary_df["run"].tolist()
    total_runs = len(runs)
    if total_runs == 0:
        print("No runs available to plot.")
        return

    for metric in available_metrics:
        metric_values = summary_df[metric].tolist()
        if all(pd.isna(v) for v in metric_values):
            print(f"Skipping {metric} chart - no values available.")
            continue

        metric_values = [0 if pd.isna(v) else v for v in metric_values]
        fig, ax = plt.subplots(figsize=(max(6, total_runs * 0.8), 5))
        x = np.arange(total_runs)
        bars = ax.bar(x, metric_values, width=0.6, color="steelblue")

        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            label = f"{int(value)}" if value == int(value) else f"{value:.4f}"
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


def _write_leaderboard_workbook(
    summary_df: pd.DataFrame, run_results: dict, workbook_path: Path
) -> None:
    """Write leaderboard Excel workbook."""
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = set()

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)

        for run_name, df in run_results.items():
            if "llm_judge_score" in df.columns:
                df = df[~df["llm_judge_score"]]
            sheet_name = _unique_sheet_name(run_name, sheet_names)
            if df.empty:
                pd.DataFrame({"info": ["No results.csv found"]}).to_excel(
                    writer, sheet_name=sheet_name, index=False
                )
            else:
                df.to_excel(writer, sheet_name=sheet_name, index=False)


def _unique_sheet_name(run_name: str, existing: set) -> str:
    """Generate unique Excel sheet name."""
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


def main():
    """CLI entry point for leaderboard generation."""
    parser = argparse.ArgumentParser(description="Generate TTS evaluation leaderboard")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing provider result subdirectories",
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save leaderboard files (defaults to output_dir/leaderboard)",
    )

    args = parser.parse_args()

    generate_leaderboard(output_dir=args.output_dir, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
