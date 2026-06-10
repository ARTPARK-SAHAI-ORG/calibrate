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
from os.path import join

from calibrate.llm.run_tests import display_label, run_model_tests
from calibrate.llm.tests_leaderboard import generate_leaderboard
from calibrate.llm._output import run_benchmark_cli

# Maximum number of models to run in parallel
MAX_PARALLEL_MODELS = 2


async def _run_models(
    config: dict,
    models: list[str],
    provider: str,
    output_dir: str,
    max_parallel: int = MAX_PARALLEL_MODELS,
    test_parallel: int | None = None,
) -> dict:
    """Run tests for each model with bounded parallelism.

    Returns a ``{model: result}`` dict (no leaderboard side effect), so it can
    be reused as the ``runner`` for :func:`run_benchmark_cli`.
    """
    results: dict = {}
    semaphore = asyncio.Semaphore(max_parallel)

    async def run_model(model: str) -> tuple[str, dict]:
        async with semaphore:
            result = await run_model_tests(
                model=model,
                provider=provider,
                config=config,
                output_dir=output_dir,
                test_parallel=test_parallel,
            )
            return (model, result)

    tasks = [run_model(model) for model in models]
    for model, result in await asyncio.gather(*tasks):
        results[model] = result
    return results


async def run(
    config: dict,
    models: list[str],
    provider: str,
    output_dir: str = "./out",
    max_parallel: int = MAX_PARALLEL_MODELS,
    test_parallel: int | None = None,
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
    results = await _run_models(
        config=config,
        models=models,
        provider=provider,
        output_dir=output_dir,
        max_parallel=max_parallel,
        test_parallel=test_parallel,
    )

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

    args = parser.parse_args()

    models = args.model

    config = json.load(open(args.config))

    await run_benchmark_cli(
        output_dir=args.output_dir,
        models=models,
        runner=lambda: _run_models(
            config=config,
            models=models,
            provider=args.provider,
            output_dir=args.output_dir,
            test_parallel=args.parallel,
        ),
        config_path=args.config,
        provider=args.provider,
        model_label=lambda m: display_label(args.provider, m),
    )


if __name__ == "__main__":
    asyncio.run(main())
