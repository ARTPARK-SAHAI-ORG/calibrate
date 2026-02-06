"""
calibrate - A package for voice agent evaluation and benchmarking.

Library Usage:
    # STT Evaluation
    from calibrate.stt import eval, leaderboard
    import asyncio
    asyncio.run(eval(provider="deepgram", input_dir="./data", output_dir="./out"))
    leaderboard(output_dir="./out", save_dir="./leaderboard")

    # TTS Evaluation
    from calibrate.tts import eval, leaderboard
    asyncio.run(eval(provider="google", input="./data/sample.csv", output_dir="./out"))
    leaderboard(output_dir="./out", save_dir="./leaderboard")

    # LLM Tests
    from calibrate.llm import tests
    asyncio.run(tests.run(config="./config.json", output_dir="./out"))
    tests.leaderboard(output_dir="./out", save_dir="./leaderboard")

    # LLM Simulations
    from calibrate.llm import simulations
    asyncio.run(simulations.run(config="./config.json", output_dir="./out"))
    simulations.leaderboard(output_dir="./out", save_dir="./leaderboard")

    # Agent Simulation
    from calibrate.agent import simulation
    asyncio.run(simulation.run(config="./config.json", output_dir="./out"))
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("calibrate-agent")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

# Lazy imports for submodules
from calibrate import stt
from calibrate import tts
from calibrate import llm
from calibrate import agent

__all__ = ["stt", "tts", "llm", "agent", "__version__"]
