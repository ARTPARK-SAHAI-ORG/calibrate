"""
calibrate - A package for voice agent evaluation and benchmarking.

Library Usage:
    # STT Evaluation (runs eval across providers + generates leaderboard)
    from calibrate.stt import run
    import asyncio
    asyncio.run(run(providers=["deepgram", "google"], input_dir="./data", output_dir="./out"))

    # TTS Evaluation (runs eval across providers + generates leaderboard)
    from calibrate.tts import run
    asyncio.run(run(providers=["google", "openai"], input="./data/sample.csv", output_dir="./out"))

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

# Suppress noisy startup logs from pipecat (loguru) and transformers
# This MUST happen before any submodule imports that trigger pipecat/transformers
import sys
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from loguru import logger

logger.remove()  # Remove default handler to suppress pipecat startup message
logger.add(sys.stderr, level="WARNING")

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
