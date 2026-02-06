# calibrate.stt module
"""
Speech-to-Text evaluation and benchmarking module.

Library Usage:
    from calibrate.stt import eval, leaderboard

    # Run STT evaluation
    import asyncio
    result = asyncio.run(eval(
        provider="deepgram",
        language="english",
        input_dir="./data",
        output_dir="./out"
    ))

    # Generate leaderboard
    leaderboard(output_dir="./out", save_dir="./leaderboard")
"""

from typing import Literal


async def eval(
    provider: Literal[
        "deepgram",
        "openai",
        "cartesia",
        "smallest",
        "groq",
        "google",
        "sarvam",
        "elevenlabs",
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
    port: int = 8765,
) -> dict:
    """
    Run STT evaluation for a given provider.

    Args:
        provider: STT provider to evaluate (deepgram, openai, cartesia, smallest, groq, google, sarvam, elevenlabs)
        input_dir: Path to input directory containing audio files and stt.csv
        output_dir: Path to output directory for results (default: ./out)
        language: Language of the audio files (english, hindi, kannada, bengali, malayalam, marathi, odia, punjabi, tamil, telugu, gujarati, sindhi)
        input_file_name: Name of the input CSV file (default: stt.csv)
        debug: Run evaluation on first N audio files only
        debug_count: Number of audio files to run in debug mode (default: 5)
        ignore_retry: Skip retry if not all audios are processed
        overwrite: Overwrite existing results instead of resuming from checkpoint (default: False)
        port: WebSocket port for STT bot (default: 8765)

    Returns:
        dict: Results containing transcripts and metrics

    Example:
        >>> import asyncio
        >>> from calibrate.stt import eval
        >>> result = asyncio.run(eval(
        ...     provider="deepgram",
        ...     language="english",
        ...     input_dir="./data",
        ...     output_dir="./out"
        ... ))
    """
    # Import here to avoid circular imports and heavy imports at module load
    from calibrate.stt.eval import main as _stt_eval_main
    import sys
    import argparse

    # Build argument list
    argv = [
        "calibrate",
        "--provider",
        provider,
        "--language",
        language,
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--input-file-name",
        input_file_name,
        "--debug_count",
        str(debug_count),
        "--port",
        str(port),
    ]

    if debug:
        argv.append("--debug")
    if ignore_retry:
        argv.append("--ignore_retry")
    if overwrite:
        argv.append("--overwrite")

    # Save original sys.argv and restore after
    original_argv = sys.argv
    try:
        sys.argv = argv
        await _stt_eval_main()
        return {"status": "completed", "output_dir": output_dir}
    finally:
        sys.argv = original_argv


def leaderboard(output_dir: str, save_dir: str) -> None:
    """
    Generate STT leaderboard from evaluation results.

    Args:
        output_dir: Path to directory containing provider evaluation results
        save_dir: Path to directory where leaderboard will be saved

    Example:
        >>> from calibrate.stt import leaderboard
        >>> leaderboard(output_dir="./out", save_dir="./leaderboard")
    """
    from calibrate.stt.leaderboard import generate_leaderboard

    generate_leaderboard(output_dir=output_dir, save_dir=save_dir)


__all__ = ["eval", "leaderboard"]
