"""
pense - A package for voice agent evaluation and benchmarking.

Library Usage:
    # STT Evaluation
    from pense.stt import eval, leaderboard
    import asyncio
    asyncio.run(eval(provider="deepgram", input_dir="./data", output_dir="./out"))
    leaderboard(output_dir="./out", save_dir="./leaderboard")
    
    # TTS Evaluation
    from pense.tts import eval, leaderboard
    asyncio.run(eval(provider="google", input="./data/sample.csv", output_dir="./out"))
    leaderboard(output_dir="./out", save_dir="./leaderboard")
    
    # LLM Tests
    from pense.llm import tests
    asyncio.run(tests.run(config="./config.json", output_dir="./out"))
    tests.leaderboard(output_dir="./out", save_dir="./leaderboard")
    
    # LLM Simulations
    from pense.llm import simulations
    asyncio.run(simulations.run(config="./config.json", output_dir="./out"))
    simulations.leaderboard(output_dir="./out", save_dir="./leaderboard")
    
    # Agent Simulation
    from pense.agent import simulation
    asyncio.run(simulation.run(config="./config.json", output_dir="./out"))
"""

__version__ = "0.1.0"

# Lazy imports for submodules
from pense import stt
from pense import tts
from pense import llm
from pense import agent

__all__ = ["stt", "tts", "llm", "agent", "__version__"]
