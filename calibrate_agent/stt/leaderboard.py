"""
STT Leaderboard Generation

Generates comparison leaderboard from STT evaluation results.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from calibrate_agent.utils import read_leaderboard_metrics

INVALID_SHEET_CHARS = set("[]:*?/\\")
PAST_RUNS_SHEET = "past_runs"


def generate_leaderboard(output_dir: str, save_dir: str | None = None) -> str:
    """Generate leaderboard comparing all provider results in output_dir.

    Args:
        output_dir: Directory containing provider result subdirectories
        save_dir: Directory to save leaderboard files (defaults to output_dir/leaderboard)

    Returns:
        Path to the leaderboard directory
    """
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
        metrics = read_leaderboard_metrics(run_dir / "metrics.json")
        results_df = _read_leaderboard_results(run_dir / "results.csv")

        row = {"run": run_dir.name, "count": len(results_df)}
        row.update(metrics)

        summary_rows.append(row)
        run_results[run_dir.name] = results_df

    summary_df = pd.DataFrame(summary_rows)

    workbook_path = save_path / "stt_leaderboard.xlsx"
    past_runs_df = _load_and_extend_past_runs(workbook_path, summary_df)
    _write_leaderboard_workbook(
        summary_df, run_results, workbook_path, past_runs_df=past_runs_df
    )
    print(f"Saved leaderboard workbook to {workbook_path}")

    return str(save_path)


def _read_leaderboard_results(results_path: Path) -> pd.DataFrame:
    """Read results from results.csv file."""
    if not results_path.exists():
        print(f"[WARN] results.csv missing for {results_path.parent.name}")
        return pd.DataFrame()
    return pd.read_csv(results_path)


def _load_and_extend_past_runs(
    workbook_path: Path, new_summary: pd.DataFrame
) -> pd.DataFrame:
    """Archive the previous summary into ``past_runs`` and return the full history.

    If ``workbook_path`` already exists, its current ``summary`` sheet is stamped
    with ``archived_at`` and appended under any existing ``past_runs`` rows. The
    newly computed ``new_summary`` is not archived yet — only the previous
    workbook's snapshot is. Returns an empty frame when there is nothing to
    archive.
    """
    if not workbook_path.exists():
        return pd.DataFrame()

    try:
        existing_past = pd.read_excel(workbook_path, sheet_name=PAST_RUNS_SHEET)
    except ValueError:
        existing_past = pd.DataFrame()
    except Exception:
        existing_past = pd.DataFrame()

    try:
        previous_summary = pd.read_excel(workbook_path, sheet_name="summary")
    except Exception:
        return existing_past

    if previous_summary.empty:
        return existing_past

    stamped = previous_summary.copy()
    stamped.insert(
        0,
        "archived_at",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    frames = [df for df in (existing_past, stamped) if not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _write_leaderboard_workbook(
    summary_df: pd.DataFrame,
    run_results: dict,
    workbook_path: Path,
    past_runs_df: pd.DataFrame | None = None,
) -> None:
    """Write leaderboard Excel workbook."""
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = set()

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary_df.to_excel(
            writer, sheet_name="summary", index=False, float_format="%.5f"
        )
        sheet_names.add("summary")

        if past_runs_df is not None and not past_runs_df.empty:
            past_runs_df.to_excel(
                writer,
                sheet_name=PAST_RUNS_SHEET,
                index=False,
                float_format="%.5f",
            )
            sheet_names.add(PAST_RUNS_SHEET)

        for run_name, df in run_results.items():
            sheet_name = _unique_sheet_name(run_name, sheet_names)
            if df.empty:
                pd.DataFrame({"info": ["No results.csv found"]}).to_excel(
                    writer, sheet_name=sheet_name, index=False
                )
            else:
                df.to_excel(
                    writer, sheet_name=sheet_name, index=False, float_format="%.5f"
                )


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
    parser = argparse.ArgumentParser(description="Generate STT evaluation leaderboard")
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
