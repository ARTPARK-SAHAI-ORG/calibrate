# pense.agent module
"""
Voice agent testing and simulation module.

Library Usage:
    from pense.agent import simulation
    
    # Run agent simulation
    import asyncio
    result = asyncio.run(simulation.run(
        config="./config.json",
        output_dir="./out"
    ))
    
    # Run a single simulation with custom parameters
    result = asyncio.run(simulation.run_single(
        system_prompt="You are a helpful assistant...",
        language="english",
        gender="female",
        evaluation_criteria=[...],
        output_dir="./out"
    ))
"""

from typing import Literal, List
from dataclasses import dataclass


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text service."""

    provider: str = "google"


@dataclass
class TTSConfig:
    """Configuration for Text-to-Speech service."""

    provider: str = "google"


@dataclass
class LLMConfig:
    """Configuration for LLM service."""

    provider: str = "openrouter"
    model: str = "openai/gpt-4.1"


class _Simulation:
    """Voice Agent Simulation API."""

    @staticmethod
    async def run(
        config: str,
        output_dir: str = "./out",
        port: int = 8765,
    ) -> dict:
        """
        Run voice agent simulations from a configuration file.

        Args:
            config: Path to JSON configuration file containing simulation config
            output_dir: Path to output directory for results (default: ./out)
            port: Base WebSocket port for simulations (default: 8765)

        Returns:
            dict: Results containing simulation outcomes and metrics

        Example:
            >>> import asyncio
            >>> from pense.agent import simulation
            >>> result = asyncio.run(simulation.run(
            ...     config="./config.json",
            ...     output_dir="./out"
            ... ))
        """
        from pense.agent.run_simulation import main as _agent_simulation_main
        import sys

        # Build argument list
        argv = [
            "pense",
            "--config",
            config,
            "--output-dir",
            output_dir,
            "--port",
            str(port),
        ]

        # Save original sys.argv and restore after
        original_argv = sys.argv
        try:
            sys.argv = argv
            await _agent_simulation_main()
            return {"status": "completed", "output_dir": output_dir}
        finally:
            sys.argv = original_argv

    @staticmethod
    async def run_single(
        system_prompt: str,
        language: Literal["english", "hindi"],
        gender: Literal["male", "female"],
        evaluation_criteria: List[dict],
        output_dir: str,
        interrupt_probability: float = 0.0,
        port: int = 8765,
        agent_speaks_first: bool = True,
        max_turns: int = 50,
    ) -> dict:
        """
        Run a single voice agent simulation.

        Args:
            system_prompt: System prompt for the simulated user
            language: Language for the simulation (english or hindi)
            gender: Gender for TTS voice selection
            evaluation_criteria: List of evaluation criteria dicts
            output_dir: Output directory for results
            interrupt_probability: Probability of user interrupting the agent (0.0-1.0)
            port: WebSocket port for the simulation
            agent_speaks_first: Whether the agent initiates the conversation
            max_turns: Maximum number of assistant turns

        Returns:
            dict: Simulation result with transcript, metrics, and evaluation

        Example:
            >>> import asyncio
            >>> from pense.agent import simulation
            >>> result = asyncio.run(simulation.run_single(
            ...     system_prompt="You are simulating a user...",
            ...     language="english",
            ...     gender="female",
            ...     evaluation_criteria=[{"name": "completeness", "description": "..."}],
            ...     output_dir="./out"
            ... ))
        """
        from pense.agent.run_simulation import run_simulation as _run_simulation

        return await _run_simulation(
            system_prompt=system_prompt,
            language=language,
            gender=gender,
            evaluation_criteria=evaluation_criteria,
            output_dir=output_dir,
            interrupt_probability=interrupt_probability,
            port=port,
            agent_speaks_first=agent_speaks_first,
            max_turns=max_turns,
        )

    @staticmethod
    async def run_with_config(
        config: dict,
        persona_index: int,
        user_persona: dict,
        scenario_index: int,
        scenario: dict,
        output_dir: str,
        base_port: int = 8765,
    ) -> tuple:
        """
        Run a single simulation task with config and persona/scenario.

        Args:
            config: Full configuration dict
            persona_index: Index of the persona in config
            user_persona: Persona dict for the simulated user
            scenario_index: Index of the scenario in config
            scenario: Scenario dict
            output_dir: Output directory for results
            base_port: Base WebSocket port

        Returns:
            tuple: (simulation_metrics, evaluation_results, stt_judge_result)
        """
        from pense.agent.run_simulation import (
            run_single_simulation_task as _run_single_simulation_task,
        )

        # Default interrupt sensitivity map
        interrupt_sensitivity_map = {
            "none": 0,
            "low": 0.25,
            "medium": 0.5,
            "high": 0.8,
        }

        return await _run_single_simulation_task(
            config=config,
            persona_index=persona_index,
            user_persona=user_persona,
            scenario_index=scenario_index,
            scenario=scenario,
            output_dir=output_dir,
            interrupt_sensitivity_map=interrupt_sensitivity_map,
            base_port=base_port,
        )


# Create singleton instance
simulation = _Simulation()

# Re-export config classes
__all__ = ["simulation", "STTConfig", "TTSConfig", "LLMConfig"]
