"""Shared output helpers for LLM benchmark results."""

import sys

from calibrate_agent.llm.summary import format_pass_rate


def print_benchmark_summary(
    models: list,
    model_results: dict,
    leaderboard_dir: str,
    model_label=None,
) -> bool:
    """Print the standard benchmark summary and return True if any errors occurred.

    Args:
        models: Ordered list of model names.
        model_results: Dict of model → result. Each value must have shape:
            {"metrics": {"passed": N, "total": M}}
        leaderboard_dir: Path where leaderboard was saved.
        model_label: Optional callable to format display label from model name.
    """
    print(f"\n\033[92m{'='*60}\033[0m")
    print(f"\033[92mOverall Summary\033[0m")
    print(f"\033[92m{'='*60}\033[0m\n")

    has_errors = False
    for model in models:
        label = model_label(model) if model_label else model
        mr = model_results.get(model, {})
        if not isinstance(mr, dict) or mr.get("status") == "error":
            print(f"  {label}: \033[31mError - {mr.get('error') if isinstance(mr, dict) else mr}\033[0m")
            has_errors = True
        else:
            print(f"  {label}: {format_pass_rate(mr.get('metrics', {}))}")

    print(f"\n\033[92mLeaderboard saved to {leaderboard_dir}\033[0m")
    return has_errors
