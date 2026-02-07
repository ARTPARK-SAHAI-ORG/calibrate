# calibrate.stt module
"""
Speech-to-Text evaluation and benchmarking module.

Library Usage:
    from calibrate.stt import run

    # Run STT evaluation across providers and generate leaderboard
    import asyncio
    result = asyncio.run(run(
        providers=["deepgram", "google"],
        language="english",
        input_dir="./data",
        output_dir="./out"
    ))

For single-provider evaluation without leaderboard:
    from calibrate.stt import run_single

    result = asyncio.run(run_single(
        provider="deepgram",
        language="english",
        input_dir="./data",
        output_dir="./out"
    ))
"""

# Main entry point - benchmark multiple providers with leaderboard
from calibrate.stt.benchmark import run

# Single provider evaluation (no leaderboard)
from calibrate.stt.eval import run_single_provider_eval as run_single

# Leaderboard generation
from calibrate.stt.leaderboard import generate_leaderboard

# Input validation
from calibrate.stt.eval import validate_stt_input_dir

__all__ = ["run", "run_single", "generate_leaderboard", "validate_stt_input_dir"]
