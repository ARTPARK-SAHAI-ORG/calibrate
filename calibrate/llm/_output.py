"""Shared output helpers for LLM benchmark results."""

import os
import sys
from os.path import exists, join
from typing import Awaitable, Callable, Optional


async def run_benchmark_cli(
    *,
    output_dir: str,
    models: list,
    runner: "Callable[[], Awaitable[dict]]",
    config_path: Optional[str] = None,
    provider: Optional[str] = None,
    model_label=None,
) -> None:
    """Run a multi-model LLM benchmark with consolidated logging + summary.

    Shared scaffolding for both the direct-provider benchmark
    (``llm/benchmark.py``) and the agent-connection benchmark (the ``llm`` CLI
    path). Mirrors stdout/stderr into a per-run ``logs`` file — so concurrent
    per-model output doesn't interleave on the terminal — prints the banner,
    awaits ``runner`` (which returns a ``{model: result}`` dict), writes the
    leaderboard, and prints the consolidated summary. Exits non-zero if any
    model errored.

    Args:
        output_dir: Base output directory (per-model subfolders live here).
        models: Ordered list of model names.
        runner: Zero-arg coroutine factory returning ``{model: result}``.
        config_path: Optional config path shown in the banner.
        provider: Optional provider shown in the banner (omitted for agents).
        model_label: Optional callable to format display labels.
    """
    from calibrate.utils import StreamTee
    from calibrate.llm.tests_leaderboard import generate_leaderboard

    os.makedirs(output_dir, exist_ok=True)

    # When the interactive UI runs each model in its own subprocess, several
    # processes target the same ``logs`` path; ``CALIBRATE_LLM_LOG_APPEND=1``
    # makes them append instead of racing to truncate each other's output.
    log_path = join(output_dir, "logs")
    append_mode = os.environ.get("CALIBRATE_LLM_LOG_APPEND") == "1"
    if not append_mode and exists(log_path):
        os.remove(log_path)
    log_file = open(log_path, "a" if append_mode else "w")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = StreamTee(original_stdout, log_file)
    sys.stderr = StreamTee(original_stderr, log_file)

    label_fn = model_label or (lambda m: m)
    try:
        print("\n\033[91mLLM Tests Benchmark\033[0m\n")
        if config_path:
            print(f"Config: {config_path}")
        print(f"Model(s): {', '.join(label_fn(m) for m in models)}")
        if provider:
            print(f"Provider: {provider}")
        print(f"Output: {output_dir}")
        print("")

        model_results = await runner()

        leaderboard_dir = join(output_dir, "leaderboard")
        try:
            generate_leaderboard(output_dir=output_dir, save_dir=leaderboard_dir)
        except Exception as e:
            print(f"\033[31mLeaderboard generation failed: {e}\033[0m")

        has_errors = print_benchmark_summary(
            models=models,
            model_results=model_results,
            leaderboard_dir=leaderboard_dir,
            model_label=model_label,
        )
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    if has_errors:
        sys.exit(1)


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
            metrics = mr.get("metrics", {})
            passed = metrics.get("passed", 0)
            total = metrics.get("total", 0)
            pct = (passed / total * 100) if total > 0 else 0
            print(f"  {label}: {passed}/{total} ({pct:.1f}%)")

    print(f"\n\033[92mLeaderboard saved to {leaderboard_dir}\033[0m")
    return has_errors
