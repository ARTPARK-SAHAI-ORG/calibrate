# calibrate.tts module
"""
Text-to-Speech evaluation and benchmarking module.

Library Usage:
    from calibrate.tts import eval, leaderboard

    # Run TTS evaluation
    import asyncio
    result = asyncio.run(eval(
        provider="google",
        language="english",
        input="./data/sample.csv",
        output_dir="./out"
    ))

    # Generate leaderboard
    leaderboard(output_dir="./out", save_dir="./leaderboard")
"""

from typing import Literal


async def eval(
    input: str,
    provider: Literal[
        "cartesia", "openai", "groq", "google", "elevenlabs", "sarvam", "smallest"
    ] = "google",
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
) -> dict:
    """
    Run TTS evaluation for a given provider.

    Args:
        input: Path to input CSV file containing texts to synthesize
        provider: TTS provider to evaluate (cartesia, openai, groq, google, elevenlabs, sarvam, smallest)
        language: Language for synthesis (english, hindi, kannada, bengali, malayalam, marathi, odia, punjabi, tamil, telugu, gujarati, sindhi)
        output_dir: Path to output directory for results (default: ./out)
        debug: Run evaluation on first N texts only
        debug_count: Number of texts to run in debug mode (default: 5)
        overwrite: Overwrite existing results instead of resuming from checkpoint (default: False)

    Returns:
        dict: Results containing audio paths and metrics

    Example:
        >>> import asyncio
        >>> from calibrate.tts import eval
        >>> result = asyncio.run(eval(
        ...     provider="google",
        ...     language="english",
        ...     input="./data/sample.csv",
        ...     output_dir="./out"
        ... ))
    """
    from calibrate.tts.eval import main as _tts_eval_main
    import sys

    # Build argument list
    argv = [
        "calibrate",
        "--provider",
        provider,
        "--language",
        language,
        "--input",
        input,
        "--output-dir",
        output_dir,
        "--debug_count",
        str(debug_count),
    ]

    if debug:
        argv.append("--debug")
    if overwrite:
        argv.append("--overwrite")

    # Save original sys.argv and restore after
    original_argv = sys.argv
    try:
        sys.argv = argv
        await _tts_eval_main()
        return {"status": "completed", "output_dir": output_dir}
    finally:
        sys.argv = original_argv


def leaderboard(output_dir: str, save_dir: str) -> None:
    """
    Generate TTS leaderboard from evaluation results.

    Args:
        output_dir: Path to directory containing provider evaluation results
        save_dir: Path to directory where leaderboard will be saved

    Example:
        >>> from calibrate.tts import leaderboard
        >>> leaderboard(output_dir="./out", save_dir="./leaderboard")
    """
    from calibrate.tts.leaderboard import generate_leaderboard

    generate_leaderboard(output_dir=output_dir, save_dir=save_dir)


__all__ = ["eval", "leaderboard"]
