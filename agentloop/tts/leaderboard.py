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
    "processing_time",
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

    plot_df = summary_df[["run"] + available_metrics].copy()
    plot_df = plot_df.melt(
        id_vars="run",
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    )
    plot_df = plot_df.dropna(subset=["value"])
    if plot_df.empty:
        print("No metric values available to plot.")
        return

    runs = plot_df["run"].unique()
    metrics = plot_df["metric"].unique()

    import numpy as np  # local import to avoid dependency if charts not needed

    x = np.arange(len(metrics))
    total_runs = len(runs)
    bar_width = min(0.8 / total_runs, 0.2)
    offsets = (np.arange(total_runs) - (total_runs - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 0.9), 5.5))

    pivot_df = plot_df.pivot(index="metric", columns="run", values="value").reindex(
        metrics
    )

    for idx, run in enumerate(runs):
        heights = pivot_df[run].to_numpy()
        ax.bar(
            x + offsets[idx],
            heights,
            width=bar_width,
            label=run,
        )

    ax.set_title("Metrics by Run")
    ax.set_ylabel("Value")
    ax.set_xlabel("Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [metric.replace("_", " ").title() for metric in metrics],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Run", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()

    chart_path = output_dir / "all_metrics_by_run.png"
    fig.savefig(chart_path, dpi=300)
    plt.close(fig)

    print(f"Saved combined chart at {chart_path}")


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
