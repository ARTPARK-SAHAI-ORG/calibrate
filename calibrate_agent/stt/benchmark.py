"""
STT Benchmark - Run multiple providers in parallel and generate leaderboard.

This is the main entry point for the Python SDK and CLI when evaluating
multiple STT providers.

Usage:
    # Python SDK
    from calibrate_agent.stt import run
    result = asyncio.run(run(
        providers=["deepgram", "google", "sarvam"],
        input_dir="./data",
        output_dir="./out"
    ))

    # CLI
    calibrate-agent stt -p deepgram google sarvam -i ./data -o ./out
"""

import asyncio
import argparse
import sys
import os
from os.path import exists, join
from typing import Literal

from calibrate_agent.stt.eval import (
    run_single_provider_eval,
    run_eval_only,
    validate_stt_input_dir,
    format_metrics_summary,
    STT_PROVIDERS,
    STT_LANGUAGES,
)
from calibrate_agent.stt.leaderboard import generate_leaderboard
from calibrate_agent._env import env_int
from calibrate_agent.utils import StreamTee


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
        "maithili",
    ] = "english",
    input_file_name: str = "stt.csv",
    debug: bool = False,
    debug_count: int = 5,
    ignore_retry: bool = False,
    overwrite: bool = False,
    max_parallel: int = MAX_PARALLEL_PROVIDERS,
    judge_evaluators: list[dict] = None,
    run_llm_judges: bool = True,
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
        judge_evaluators: Optional list of evaluator dicts (each with ``name``,
            ``system_prompt``, ``judge_model``, ``type``, ...). When omitted no
            LLM judge runs and only WER/CER are reported.
        run_llm_judges: Run the Sarvam intent/entity judge (default True;
            loads a normalizer model + makes per-row judge calls). Set to False
            to skip it.

    Returns:
        dict: Results summary with status and output paths

    Example:
        >>> import asyncio
        >>> from calibrate_agent.stt import run
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
                judge_evaluators=judge_evaluators,
                run_llm_judges=run_llm_judges,
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
    """CLI entry point for multi-provider STT benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark multiple STT providers in parallel"
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        nargs="+",
        help="STT provider(s) to evaluate (space-separated for multiple). Not required with --eval-only.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip STT inference and run evaluators directly on a dataset of (gt, pred) pairs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON (list of {id, gt, pred}). Required with --eval-only.",
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
        help="Path to the input directory containing the audio files and stt.csv. Not required with --eval-only.",
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
        help="Ignore retrying if all the audios are not processed and move on to evaluators",
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
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to optional JSON config file with an `evaluators` list",
    )
    parser.add_argument(
        "--skip-llm-judges",
        action="store_true",
        help=(
            "Skip the extra LLM-based judges (Sarvam intent & entity "
            "preservation, Sarvam LLM-WER/CER, and pipecat-style semantic WER). "
            "They all run by default; passing this reports WER/CER only."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=env_int("CALIBRATE_STT_MAX_PARALLEL", MAX_PARALLEL_PROVIDERS),
        help=(
            "Number of providers to evaluate in parallel (default: 2, or "
            "$CALIBRATE_STT_MAX_PARALLEL)."
        ),
    )

    args = parser.parse_args()

    providers = args.provider

    if args.eval_only:
        if not args.dataset:
            print("\033[31mError: --dataset is required with --eval-only\033[0m")
            sys.exit(1)
    else:
        if not providers:
            print("\033[31mError: --provider is required (omit only with --eval-only)\033[0m")
            sys.exit(1)
        if not args.input_dir:
            print("\033[31mError: --input-dir is required (omit only with --eval-only)\033[0m")
            sys.exit(1)

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

    # ``exist_ok=True`` makes this safe when several ``calibrate-agent stt``
    # subprocesses race to create the output dir on first use; the previous
    # ``if not exists: makedirs(...)`` pattern was non-atomic and the loser
    # raised ``FileExistsError``.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load evaluators from optional config file (shared by both flows)
    judge_evaluators = None
    if args.config:
        import json as _json

        with open(args.config) as _f:
            _cfg = _json.load(_f)
        judge_evaluators = _cfg.get("evaluators")

    # Eval-only mode owns ``output_dir/logs`` itself via the provider_log
    # contextvar, so we don't set up a benchmark-level StreamTee here —
    # otherwise both writers would target the same path (and the eval-only
    # ``os.remove`` of that path would unlink the active handle on POSIX or
    # raise PermissionError on Windows).
    if args.eval_only:
        print("\n\033[91mSTT Eval-Only\033[0m\n")
        print(f"Dataset: {args.dataset}")
        print(f"Language: {args.language}")
        print(f"Output: {args.output_dir}")
        print("")

        result = await run_eval_only(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            judge_evaluators=judge_evaluators,
            language=args.language,
            run_llm_judges=not args.skip_llm_judges,
        )

        print(f"\n\033[92m{'='*60}\033[0m")
        print(f"\033[92mSummary\033[0m")
        print(f"\033[92m{'='*60}\033[0m\n")

        if result.get("status") == "error":
            print(f"  \033[31mError - {result.get('error')}\033[0m")
            sys.exit(1)

        metrics = result.get("metrics", {})
        print(format_metrics_summary(metrics))
        return

    # Benchmark (multi-provider) mode: mirror stdout/stderr into a single
    # output-dir-level ``logs`` file so the full terminal session (header,
    # per-provider output, tqdm progress, leaderboard prints, summary) is
    # captured in one place.
    log_path = join(args.output_dir, "logs")
    if exists(log_path):
        os.remove(log_path)
    log_file = open(log_path, "w")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = StreamTee(original_stdout, log_file)
    sys.stderr = StreamTee(original_stderr, log_file)

    try:
        print("\n\033[91mSTT Benchmark\033[0m\n")
        print(f"Provider(s): {', '.join(providers)}")
        print(f"Language: {args.language}")
        print(f"Input: {args.input_dir}")
        print(f"Output: {args.output_dir}")
        print("")

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
            judge_evaluators=judge_evaluators,
            run_llm_judges=not args.skip_llm_judges,
            max_parallel=args.max_parallel,
        )

        # Print summary
        print(f"\n\033[92m{'='*60}\033[0m")
        print(f"\033[92mSummary\033[0m")
        print(f"\033[92m{'='*60}\033[0m\n")

        has_errors = False
        provider_results = result.get("providers", {})
        for provider in providers:
            prov_result = provider_results.get(provider, {})
            if isinstance(prov_result, str) and prov_result.startswith("error"):
                print(f"  {provider}: \033[31m{prov_result}\033[0m")
            elif prov_result.get("status") == "error":
                print(
                    f"  {provider}: \033[31mError - {prov_result.get('error')}\033[0m"
                )
                has_errors = True
            else:
                metrics = prov_result.get("metrics", {})
                print(format_metrics_summary(metrics, prefix=f"{provider}: "))

        if has_errors:
            sys.exit(1)

        print(
            f"\n\033[92mLeaderboard saved to: {result.get('leaderboard_dir')}\033[0m"
        )
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    asyncio.run(main())
