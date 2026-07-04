"""
calibrate_agent - A package for voice agent evaluation and benchmarking.

Library Usage:
    # STT Evaluation (runs eval across providers + generates leaderboard)
    from calibrate_agent.stt import run
    import asyncio
    asyncio.run(run(providers=["deepgram", "google"], input_dir="./data", output_dir="./out"))

    # TTS Evaluation (runs eval across providers + generates leaderboard)
    from calibrate_agent.tts import run
    asyncio.run(run(providers=["google", "openai"], input="./data/sample.csv", output_dir="./out"))

    # LLM Tests
    from calibrate_agent.llm import tests
    asyncio.run(tests.run(config="./config.json", output_dir="./out"))
    tests.leaderboard(output_dir="./out", save_dir="./leaderboard")

    # LLM Simulations
    from calibrate_agent.llm import simulations
    asyncio.run(simulations.run(config="./config.json", output_dir="./out"))
    simulations.leaderboard(output_dir="./out", save_dir="./leaderboard")

    # Agent Simulation
    from calibrate_agent.agent import simulation
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

# Lazy imports for submodules.
#
# Each submodule pulls in heavy, optional provider SDKs (deepgram, cartesia,
# google-cloud-speech, pipecat, ...). Importing them eagerly means a single
# broken/incompatible provider dependency makes `import calibrate_agent` fail
# entirely. Use PEP 562 module-level __getattr__ so a submodule is only
# imported when it is actually accessed (e.g. `calibrate_agent.stt`), while keeping
# the same `calibrate_agent.stt` / `from calibrate_agent import stt` access pattern.
import importlib

_SUBMODULES = ("stt", "tts", "llm", "agent")


def __getattr__(name):
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_SUBMODULES))


__all__ = ["stt", "tts", "llm", "agent", "__version__"]
