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

from calibrate.llm.run_tests import run_model_tests
from calibrate.llm.tests_leaderboard import generate_leaderboard

# Maximum number of models to run in parallel
MAX_PARALLEL_MODELS = 2


async def run(
    config: dict,
    models: list[str],
    provider: str,
    output_dir: str = "./out",
    max_parallel: int = MAX_PARALLEL_MODELS,
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

    args = parser.parse_args()

    models = args.model

    print("\n\033[91mLLM Tests Benchmark\033[0m\n")
    print(f"Config: {args.config}")
    print(f"Model(s): {', '.join([f'{args.provider}/{m}' for m in models])}")
    print(f"Provider: {args.provider}")
    print(f"Output: {args.output_dir}")
    print("")

    config = json.load(open(args.config))

    if not exists(args.output_dir):
        os.makedirs(args.output_dir)

    result = await run(
        config=config,
        models=models,
        provider=args.provider,
        output_dir=args.output_dir,
    )

    # Print summary
    print(f"\n\033[92m{'='*60}\033[0m")
    print(f"\033[92mOverall Summary\033[0m")
    print(f"\033[92m{'='*60}\033[0m\n")

    has_errors = False
    for model in models:
        model_result = result["models"].get(model, {})
        if isinstance(model_result, dict):
            if model_result.get("status") == "error":
                print(
                    f"  {args.provider}/{model}: \033[31mError - {model_result.get('error')}\033[0m"
                )
                has_errors = True
            else:
                passed = model_result.get("passed", 0)
                total = model_result.get("total", 0)
                pct = (passed / total * 100) if total > 0 else 0
                print(f"  {args.provider}/{model}: {passed}/{total} ({pct:.1f}%)")

    print(f"\n\033[92mLeaderboard saved to {result['leaderboard_dir']}\033[0m")

    if has_errors:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
