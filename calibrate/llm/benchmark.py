"""
LLM Tests Benchmark — Multi-model parallel evaluation with leaderboard generation.

This module handles running LLM tests across multiple models in parallel
and automatically generates a leaderboard after all models complete.

CLI Usage:
    calibrate llm -c config.json -m model1 model2 -p openrouter -o ./out

Python SDK:
    from calibrate.llm import tests
    import asyncio
    asyncio.run(tests.run(
        system_prompt="...",
        tools=[...],
        test_cases=[...],
        models=["gpt-4.1", "claude-3.5-sonnet"],
        provider="openrouter"
    ))
"""

import argparse
import asyncio
import json
import os
import sys
from os.path import exists, join

from calibrate.llm.run_tests import display_label, run_model_tests
from calibrate.llm.tests_leaderboard import generate_leaderboard
from calibrate.llm._output import print_benchmark_summary
from calibrate.utils import StreamTee, apply_debug_limit

# Maximum number of models to run in parallel
MAX_PARALLEL_MODELS = 2


async def run(
    config: dict,
    models: list[str],
    provider: str,
    output_dir: str = "./out",
    max_parallel: int = MAX_PARALLEL_MODELS,
    test_parallel: int | None = None,
    overwrite: bool = False,
) -> dict:
    """
    Run LLM tests for multiple models in parallel and generate a leaderboard.

    This is the main entry point for multi-model LLM benchmarks.

    Args:
        config: Test configuration dict containing system_prompt, tools, test_cases
        models: List of model names to evaluate
        provider: LLM provider (openai or openrouter)
        output_dir: Path to output directory for results (default: ./out)
            Results saved to output_dir/model_name/ for each model
        max_parallel: Maximum number of models to run in parallel (default: 2)
        test_parallel: Max test cases to evaluate concurrently per model.
        overwrite: When False (default), resume each model from its prior
            ``results.json`` instead of re-evaluating completed test cases.

    Returns:
        dict: Results summary with status and output paths

    Example:
        >>> import asyncio
        >>> import json
        >>> config = json.load(open("tests.json"))
        >>> from calibrate.llm.benchmark import run
        >>> result = asyncio.run(run(
        ...     config=config,
        ...     models=["gpt-4.1", "claude-3.5-sonnet"],
        ...     provider="openrouter",
        ...     output_dir="./out"
        ... ))
    """
    results = {}
    semaphore = asyncio.Semaphore(max_parallel)

    async def run_model(model: str) -> tuple[str, dict]:
        """Run tests for a single model with semaphore control."""
        async with semaphore:
            result = await run_model_tests(
                model=model,
                provider=provider,
                config=config,
                output_dir=output_dir,
                test_parallel=test_parallel,
                overwrite=overwrite,
            )
            return (model, result)

    # Run all models with limited parallelism
    tasks = [run_model(model) for model in models]
    model_results = await asyncio.gather(*tasks)

    for model, result in model_results:
        results[model] = result

    # Generate leaderboard from output_dir (which contains model folders)
    leaderboard_dir = join(output_dir, "leaderboard")
    try:
        generate_leaderboard(output_dir=output_dir, save_dir=leaderboard_dir)
    except Exception as e:
        results["leaderboard"] = f"error: {e}"

    return {
        "status": "completed",
        "output_dir": output_dir,
        "leaderboard_dir": leaderboard_dir,
        "models": results,
    }


async def main():
    """CLI entry point for multi-model LLM benchmark."""
    parser = argparse.ArgumentParser(
        description="LLM Tests Benchmark - run multiple models in parallel"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file for the tests",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./out",
        help="Path to the output directory to save the results",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        nargs="+",
        required=True,
        help="Model(s) to use for evaluation (space-separated for multiple)",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=["openai", "openrouter"],
        default="openrouter",
        help="LLM provider to use (openai or openrouter)",
    )
    parser.add_argument(
        "-n",
        "--parallel",
        type=int,
        default=None,
        help="Number of test cases to evaluate in parallel per model",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force a clean run instead of resuming completed test cases from a prior results.json",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Debug mode: evaluate only the first N test cases (see --debug_count)",
    )
    parser.add_argument(
        "-dc",
        "--debug_count",
        type=int,
        default=5,
        help="Number of test cases to evaluate in debug mode (default: 5)",
    )

    args = parser.parse_args()

    models = args.model

    config = json.load(open(args.config))

    if args.debug and config.get("test_cases"):
        config["test_cases"] = apply_debug_limit(
            config["test_cases"], args.debug, args.debug_count
        )

    # ``exist_ok=True`` makes this safe when several ``calibrate llm``
    # subprocesses (e.g. one per model spawned by the interactive UI) race to
    # create the output dir — the previous ``if not exists: makedirs(...)``
    # pattern was non-atomic and the loser raised ``FileExistsError``.
    os.makedirs(args.output_dir, exist_ok=True)

    # Mirror everything written to stdout/stderr into a single output-dir-level
    # `logs` file so the full terminal session (banner, per-model output,
    # leaderboard prints, summary) is captured in one place — same pattern as
    # the STT/TTS benchmark CLIs.
    #
    # When the interactive UI runs each model in its own ``calibrate llm``
    # subprocess, multiple processes target the same ``logs`` path concurrently;
    # the UI sets ``CALIBRATE_LLM_LOG_APPEND=1`` so subprocesses append instead
    # of racing to truncate each other's output. The UI itself clears the file
    # once before kicking off the run.
    log_path = join(args.output_dir, "logs")
    append_mode = os.environ.get("CALIBRATE_LLM_LOG_APPEND") == "1"
    if not append_mode and exists(log_path):
        os.remove(log_path)
    log_file = open(log_path, "a" if append_mode else "w")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = StreamTee(original_stdout, log_file)
    sys.stderr = StreamTee(original_stderr, log_file)

    try:
        print("\n\033[91mLLM Tests Benchmark\033[0m\n")
        print(f"Config: {args.config}")
        print(f"Model(s): {', '.join(display_label(args.provider, m) for m in models)}")
        print(f"Provider: {args.provider}")
        print(f"Output: {args.output_dir}")
        print("")

        result = await run(
            config=config,
            models=models,
            provider=args.provider,
            output_dir=args.output_dir,
            test_parallel=args.parallel,
            overwrite=args.overwrite,
        )

        has_errors = print_benchmark_summary(
            models=models,
            model_results=result["models"],
            leaderboard_dir=result["leaderboard_dir"],
            model_label=lambda m: display_label(args.provider, m),
        )

        if has_errors:
            sys.exit(1)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    asyncio.run(main())
