"""
TTS Benchmark — Multi-provider parallel evaluation with leaderboard generation.

This module handles running TTS evaluation across multiple providers in parallel
and automatically generates a leaderboard after all providers complete.

CLI Usage:
    calibrate tts -p provider1 provider2 -i input.csv -l english -o ./out

Python SDK:
    from calibrate.tts import run
    import asyncio
    asyncio.run(run(providers=["google", "openai"], language="english", input="./data.csv"))
"""

import argparse
import asyncio
import os
import sys
from os.path import exists, join
from typing import Literal

from calibrate.tts.eval import (
    TTS_LANGUAGES,
    TTS_PROVIDERS,
    run_single_provider_eval,
    validate_tts_input_file,
)
from calibrate.tts.leaderboard import generate_leaderboard
from calibrate.utils import StreamTee

# Maximum number of providers to run in parallel
MAX_PARALLEL_PROVIDERS = 2


async def run(
    input: str,
    providers: list[
        Literal[
            "cartesia", "openai", "groq", "google", "elevenlabs", "sarvam", "smallest"
        ]
    ],
    language: Literal[
        "english",
        "hindi",
        "kannada",
        "bengali",
        "malayalam",
        "marathi",
        "odia",
        "punjabi",
        "tamil",
        "telugu",
        "gujarati",
        "sindhi",
    ] = "english",
    output_dir: str = "./out",
    debug: bool = False,
    debug_count: int = 5,
    overwrite: bool = False,
    max_parallel: int = MAX_PARALLEL_PROVIDERS,
    judge_evaluators: list[dict] = None,
) -> dict:
    """
    Run TTS evaluation for multiple providers in parallel and generate a leaderboard.

    This is the main entry point for multi-provider TTS benchmarks.

    Args:
        input: Path to input CSV file containing texts to synthesize
        providers: List of TTS providers to evaluate
        language: Language for synthesis
        output_dir: Path to output directory for results (default: ./out)
        debug: Run evaluation on first N texts only
        debug_count: Number of texts to run in debug mode (default: 5)
        overwrite: Overwrite existing results instead of resuming from checkpoint (default: False)
        max_parallel: Maximum number of providers to run in parallel (default: 2)
        judge_evaluators: Optional list of evaluator dicts (each with ``name``,
            ``system_prompt``, ``judge_model``, ``type``, ...). When omitted
            the implicit default TTS evaluator runs.

    Returns:
        dict: Results summary with status and output paths

    Example:
        >>> import asyncio
        >>> from calibrate.tts import run
        >>> result = asyncio.run(run(
        ...     providers=["google", "openai", "elevenlabs"],
        ...     language="english",
        ...     input="./data/sample.csv",
        ...     output_dir="./out"
        ... ))
    """
    results = {}
    semaphore = asyncio.Semaphore(max_parallel)

    async def run_provider(provider: str) -> tuple[str, dict]:
        """Run evaluation for a single provider with semaphore control."""
        async with semaphore:
            result = await run_single_provider_eval(
                provider=provider,
                language=language,
                input_file=input,
                output_dir=output_dir,
                debug=debug,
                debug_count=debug_count,
                overwrite=overwrite,
                judge_evaluators=judge_evaluators,
            )
            return (provider, result)

    # Run all providers with limited parallelism
    tasks = [run_provider(provider) for provider in providers]
    provider_results = await asyncio.gather(*tasks)

    for provider, result in provider_results:
        results[provider] = result

    # Generate leaderboard
    leaderboard_dir = f"{output_dir}/leaderboard"
    try:
        generate_leaderboard(output_dir=output_dir, save_dir=leaderboard_dir)
    except Exception as e:
        results["leaderboard"] = f"error: {e}"

    return {
        "status": "completed",
        "output_dir": output_dir,
        "leaderboard_dir": leaderboard_dir,
        "providers": results,
    }


async def main():
    """CLI entry point for multi-provider TTS benchmark."""
    parser = argparse.ArgumentParser(
        description="TTS Benchmark - run multiple providers in parallel"
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        nargs="+",
        required=True,
        help="TTS provider(s) to use for evaluation (space-separated for multiple)",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="english",
        choices=TTS_LANGUAGES,
        help="Language of the audio files",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file containing the texts to synthesize",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./out",
        help="Path to the output directory to save the results",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Run the evaluation on the first N texts only",
    )
    parser.add_argument(
        "-dc",
        "--debug_count",
        help="Number of texts to run the evaluation on",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of resuming from last checkpoint",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to optional JSON config file with an `evaluators` list",
    )

    args = parser.parse_args()

    providers = args.provider

    # Validate all providers
    for provider in providers:
        if provider not in TTS_PROVIDERS:
            print(f"\033[31mError: Invalid provider '{provider}'.\033[0m")
            print(f"Available providers: {', '.join(TTS_PROVIDERS)}")
            sys.exit(1)

    # Validate input CSV file
    is_valid, error_msg = validate_tts_input_file(args.input)
    if not is_valid:
        print(f"\033[31mInput validation error: {error_msg}\033[0m")
        sys.exit(1)

    # ``exist_ok=True`` makes this safe when several ``calibrate tts``
    # subprocesses race to create the output dir on first use; the previous
    # ``if not exists: makedirs(...)`` pattern was non-atomic and the loser
    # raised ``FileExistsError``.
    os.makedirs(args.output_dir, exist_ok=True)

    # Mirror everything written to stdout/stderr into a single output-dir-level
    # `logs` file so the full terminal session (header, per-provider output,
    # tqdm progress, leaderboard prints, summary) is captured in one place.
    log_path = join(args.output_dir, "logs")
    if exists(log_path):
        os.remove(log_path)
    log_file = open(log_path, "w")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = StreamTee(original_stdout, log_file)
    sys.stderr = StreamTee(original_stderr, log_file)

    try:
        print("\n\033[91mTTS Benchmark\033[0m\n")
        print(f"Provider(s): {', '.join(providers)}")
        print(f"Language: {args.language}")
        print(f"Input: {args.input}")
        print(f"Output: {args.output_dir}")
        print("")

        # Load evaluators from optional config file
        judge_evaluators = None
        if args.config:
            import json as _json
            with open(args.config) as _f:
                _cfg = _json.load(_f)
            judge_evaluators = _cfg.get("evaluators")

        result = await run(
            input=args.input,
            providers=providers,
            language=args.language,
            output_dir=args.output_dir,
            debug=args.debug,
            debug_count=args.debug_count,
            overwrite=args.overwrite,
            judge_evaluators=judge_evaluators,
        )

        # Print summary
        print(f"\n\033[92m{'='*60}\033[0m")
        print(f"\033[92mSummary\033[0m")
        print(f"\033[92m{'='*60}\033[0m\n")

        has_errors = False
        for provider in providers:
            provider_result = result["providers"].get(provider, {})
            if isinstance(provider_result, dict):
                if provider_result.get("status") == "error":
                    print(
                        f"  {provider}: \033[31mError - {provider_result.get('error')}\033[0m"
                    )
                    has_errors = True
                else:
                    metrics = provider_result.get("metrics", {})
                    # Evaluator entries are dicts carrying a ``type`` field;
                    # ttfb has no ``type`` so it's correctly excluded.
                    judge_scores = {
                        k: v["mean"]
                        for k, v in metrics.items()
                        if isinstance(v, dict) and "type" in v
                    }
                    ttfb_data = metrics.get("ttfb", {})
                    ttfb_p50 = ttfb_data.get("p50", "N/A") if isinstance(ttfb_data, dict) else "N/A"
                    judge_str = ", ".join(f"{k}={v:.2f}" for k, v in judge_scores.items())
                    ttfb_str = f"TTFB(p50)={ttfb_p50:.3f}s" if isinstance(ttfb_p50, float) else f"TTFB(p50)={ttfb_p50}"
                    print(f"  {provider}: {judge_str}, {ttfb_str}")

        print(f"\n\033[92mLeaderboard saved to {result['leaderboard_dir']}\033[0m")

        if has_errors:
            sys.exit(1)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    asyncio.run(main())
