# calibrate_agent.stt module
"""
Speech-to-Text evaluation and benchmarking module.

Library Usage:
    from calibrate_agent.stt import run

    # Run STT evaluation across providers and generate leaderboard
    import asyncio
    result = asyncio.run(run(
        providers=["deepgram", "google"],
        language="english",
        input_dir="./data",
        output_dir="./out"
    ))

For single-provider evaluation without leaderboard:
    from calibrate_agent.stt import run_single

    result = asyncio.run(run_single(
        provider="deepgram",
        language="english",
        input_dir="./data",
        output_dir="./out"
    ))
"""

# Main entry point - benchmark multiple providers with leaderboard
from calibrate_agent.stt.benchmark import run

# Single provider evaluation (no leaderboard)
from calibrate_agent.stt.eval import run_single_provider_eval as run_single

# Run evaluators only on a pre-existing (gt, pred) dataset
from calibrate_agent.stt.eval import run_eval_only

# Leaderboard generation
from calibrate_agent.stt.leaderboard import generate_leaderboard

# Input validation
from calibrate_agent.stt.eval import validate_stt_input_dir

__all__ = [
    "run",
    "run_single",
    "run_eval_only",
    "generate_leaderboard",
    "validate_stt_input_dir",
]
