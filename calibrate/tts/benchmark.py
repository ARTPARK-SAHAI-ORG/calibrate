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
            try:
                result = await run_single_provider_eval(
                    provider=provider,
                    language=language,
                    input_file=input,
                    output_dir=output_dir,
                    debug=debug,
                    debug_count=debug_count,
                    overwrite=overwrite,
                )
                return (provider, result)
            except Exception as e:
                return (provider, {"status": "error", "error": str(e)})

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

    if not exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("\n\033[91mTTS Benchmark\033[0m\n")
    print(f"Provider(s): {', '.join(providers)}")
    print(f"Language: {args.language}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print("")

    result = await run(
        input=args.input,
        providers=providers,
        language=args.language,
        output_dir=args.output_dir,
        debug=args.debug,
        debug_count=args.debug_count,
        overwrite=args.overwrite,
    )

    # Print summary
    print(f"\n\033[92m{'='*60}\033[0m")
    print(f"\033[92mSummary\033[0m")
    print(f"\033[92m{'='*60}\033[0m\n")

    for provider in providers:
        provider_result = result["providers"].get(provider, {})
        if isinstance(provider_result, dict):
            if provider_result.get("status") == "error":
                print(
                    f"  {provider}: \033[31mError - {provider_result.get('error')}\033[0m"
                )
            else:
                metrics = provider_result.get("metrics", {})
                llm_score = metrics.get("llm_judge_score", "N/A")
                ttfb_data = metrics.get("ttfb", {})
                ttfb_mean = ttfb_data.get("mean", "N/A") if ttfb_data else "N/A"
                if isinstance(llm_score, float) and isinstance(ttfb_mean, float):
                    print(
                        f"  {provider}: LLM Score={llm_score:.2f}, TTFB={ttfb_mean:.3f}s"
                    )
                elif isinstance(llm_score, float):
                    print(f"  {provider}: LLM Score={llm_score:.2f}, TTFB={ttfb_mean}")
                else:
                    print(f"  {provider}: LLM Score={llm_score}, TTFB={ttfb_mean}")

    print(f"\n\033[92mLeaderboard saved to {result['leaderboard_dir']}\033[0m")


if __name__ == "__main__":
    asyncio.run(main())
