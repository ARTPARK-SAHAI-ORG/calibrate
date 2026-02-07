"""
STT Benchmark - Run multiple providers in parallel and generate leaderboard.

This is the main entry point for the Python SDK and CLI when evaluating
multiple STT providers.

Usage:
    # Python SDK
    from calibrate.stt import run
    result = asyncio.run(run(
        providers=["deepgram", "google", "sarvam"],
        input_dir="./data",
        output_dir="./out"
    ))

    # CLI
    calibrate stt -p deepgram google sarvam -i ./data -o ./out
"""

import asyncio
import argparse
import sys
import os
from os.path import exists, join
from typing import Literal

from calibrate.stt.eval import (
    run_single_provider_eval,
    validate_stt_input_dir,
    STT_PROVIDERS,
    STT_LANGUAGES,
)
from calibrate.stt.leaderboard import generate_leaderboard


# Maximum number of providers to run in parallel
MAX_PARALLEL_PROVIDERS = 2


async def run(
    providers: list[
        Literal[
            "deepgram",
            "openai",
            "cartesia",
            "smallest",
            "groq",
            "google",
            "sarvam",
            "elevenlabs",
        ]
    ],
    input_dir: str,
    output_dir: str = "./out",
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
    input_file_name: str = "stt.csv",
    debug: bool = False,
    debug_count: int = 5,
    ignore_retry: bool = False,
    overwrite: bool = False,
    max_parallel: int = MAX_PARALLEL_PROVIDERS,
) -> dict:
    """
    Run STT evaluation for one or more providers and generate a leaderboard.

    Evaluates providers in parallel (max 2 by default), then generates a comparison leaderboard.

    Args:
        providers: List of STT providers to evaluate
        input_dir: Path to input directory containing audio files and stt.csv
        output_dir: Path to output directory for results (default: ./out)
        language: Language of the audio files
        input_file_name: Name of the input CSV file (default: stt.csv)
        debug: Run evaluation on first N audio files only
        debug_count: Number of audio files to run in debug mode (default: 5)
        ignore_retry: Skip retry if not all audios are processed
        overwrite: Overwrite existing results instead of resuming from checkpoint (default: False)
        max_parallel: Maximum number of providers to run in parallel (default: 2)

    Returns:
        dict: Results summary with status and output paths

    Example:
        >>> import asyncio
        >>> from calibrate.stt import run
        >>> result = asyncio.run(run(
        ...     providers=["deepgram", "google", "sarvam"],
        ...     language="english",
        ...     input_dir="./data",
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
                    input_dir=input_dir,
                    input_file_name=input_file_name,
                    output_dir=output_dir,
                    debug=debug,
                    debug_count=debug_count,
                    ignore_retry=ignore_retry,
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
    """CLI entry point for multi-provider STT benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark multiple STT providers in parallel"
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        nargs="+",
        required=True,
        help="STT provider(s) to evaluate (space-separated for multiple)",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="english",
        choices=STT_LANGUAGES,
        help="Language of the audio files",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing the audio files and stt.csv",
    )
    parser.add_argument(
        "-f",
        "--input-file-name",
        type=str,
        default="stt.csv",
        help="Name of the input file containing the dataset to evaluate",
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
        help="Run the evaluation on the first N audio files",
    )
    parser.add_argument(
        "-dc",
        "--debug_count",
        type=int,
        default=5,
        help="Number of audio files to run the evaluation on in debug mode",
    )
    parser.add_argument(
        "--ignore_retry",
        action="store_true",
        help="Ignore retrying if all the audios are not processed and move on to LLM judge",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of resuming from last checkpoint",
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        help="Directory to save leaderboard results (defaults to output_dir/leaderboard)",
    )

    args = parser.parse_args()

    providers = args.provider

    # Validate all providers
    for provider in providers:
        if provider not in STT_PROVIDERS:
            print(f"\033[31mError: Invalid provider '{provider}'.\033[0m")
            print(f"Available providers: {', '.join(STT_PROVIDERS)}")
            sys.exit(1)

    # Validate input directory structure
    is_valid, error_msg = validate_stt_input_dir(args.input_dir, args.input_file_name)
    if not is_valid:
        print(f"\033[31mInput validation error: {error_msg}\033[0m")
        sys.exit(1)

    if not exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("\n\033[91mSTT Benchmark\033[0m\n")
    print(f"Provider(s): {', '.join(providers)}")
    print(f"Language: {args.language}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Parallel: max {MAX_PARALLEL_PROVIDERS} at a time")
    print("")

    # Run benchmark
    result = await run(
        providers=providers,
        language=args.language,
        input_dir=args.input_dir,
        input_file_name=args.input_file_name,
        output_dir=args.output_dir,
        debug=args.debug,
        debug_count=args.debug_count,
        ignore_retry=args.ignore_retry,
        overwrite=args.overwrite,
    )

    # Print summary
    print(f"\n\033[92m{'='*60}\033[0m")
    print(f"\033[92mSummary\033[0m")
    print(f"\033[92m{'='*60}\033[0m\n")

    provider_results = result.get("providers", {})
    for provider in providers:
        prov_result = provider_results.get(provider, {})
        if isinstance(prov_result, str) and prov_result.startswith("error"):
            print(f"  {provider}: \033[31m{prov_result}\033[0m")
        elif prov_result.get("status") == "error":
            print(f"  {provider}: \033[31mError - {prov_result.get('error')}\033[0m")
        else:
            metrics = prov_result.get("metrics", {})
            wer = metrics.get("wer", 0)
            llm_score = metrics.get("llm_judge_score", 0)
            print(f"  {provider}: WER={wer:.4f}, LLM Score={llm_score:.4f}")

    print(f"\n\033[92mLeaderboard saved to: {result.get('leaderboard_dir')}\033[0m")


if __name__ == "__main__":
    asyncio.run(main())
