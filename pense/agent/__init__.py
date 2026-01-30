# pense.agent module
"""
Voice agent testing and simulation module.

Library Usage:
    from pense.agent import simulation
    
    # Run agent simulation
    import asyncio
    result = asyncio.run(simulation.run(
        system_prompt="You are a helpful assistant...",
        tools=[...],
        personas=[...],
        scenarios=[...],
        evaluation_criteria=[...],
        output_dir="./out"
    ))
"""

from typing import Literal, List, Optional, Dict, Any
from dataclasses import dataclass, field
import os
import json
import asyncio
from collections import defaultdict
import numpy as np
import pandas as pd


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text service."""

    provider: str = "google"

    def to_dict(self) -> dict:
        return {"provider": self.provider}


@dataclass
class TTSConfig:
    """Configuration for Text-to-Speech service."""

    provider: str = "google"
    voice_id: Optional[str] = None

    def to_dict(self) -> dict:
        result = {"provider": self.provider}
        if self.voice_id:
            result["voice_id"] = self.voice_id
        return result


@dataclass
class LLMConfig:
    """Configuration for LLM service."""

    provider: str = "openrouter"
    model: str = "openai/gpt-4.1"

    def to_dict(self) -> dict:
        return {"provider": self.provider, "model": self.model}


class _Simulation:
    """Voice Agent Simulation API."""

    @staticmethod
    async def run(
        system_prompt: str,
        tools: List[dict],
        personas: List[dict],
        scenarios: List[dict],
        evaluation_criteria: List[dict],
        output_dir: str = "./out",
        stt: Optional[STTConfig] = None,
        tts: Optional[TTSConfig] = None,
        llm: Optional[LLMConfig] = None,
        agent_speaks_first: bool = True,
        max_turns: int = 50,
        port: int = 8765,
    ) -> dict:
        """
        Run voice agent simulations with the given configuration.

        Args:
            system_prompt: System prompt for the voice agent
            tools: List of tool definitions available to the agent
            personas: List of persona dicts with 'characteristics', 'gender', 'language', optional 'interruption_sensitivity'
            scenarios: List of scenario dicts with 'description'
            evaluation_criteria: List of criteria dicts with 'name' and 'description'
            output_dir: Path to output directory for results (default: ./out)
            stt: STT configuration (default: Google)
            tts: TTS configuration (default: Google)
            llm: LLM configuration (default: OpenRouter with gpt-4.1)
            agent_speaks_first: Whether the agent initiates the conversation (default: True)
            max_turns: Maximum number of assistant turns (default: 50)
            port: Base WebSocket port for simulations (default: 8765)

        Returns:
            dict: Results containing simulation outcomes and metrics

        Example:
            >>> import asyncio
            >>> from pense.agent import simulation, STTConfig, TTSConfig, LLMConfig
            >>> result = asyncio.run(simulation.run(
            ...     system_prompt="You are a helpful nurse...",
            ...     tools=[...],
            ...     personas=[{
            ...         "characteristics": "A shy mother named Geeta...",
            ...         "gender": "female",
            ...         "language": "english",
            ...         "interruption_sensitivity": "medium"
            ...     }],
            ...     scenarios=[{"description": "User completes the form"}],
            ...     evaluation_criteria=[{"name": "completeness", "description": "All questions answered"}],
            ...     output_dir="./out",
            ...     stt=STTConfig(provider="google"),
            ...     tts=TTSConfig(provider="google"),
            ...     llm=LLMConfig(provider="openrouter", model="openai/gpt-4.1"),
            ... ))
        """
        from pense.agent.run_simulation import run_single_simulation_task
        import gc

        os.makedirs(output_dir, exist_ok=True)

        # Build config dict
        config = {
            "system_prompt": system_prompt,
            "tools": tools,
            "personas": personas,
            "scenarios": scenarios,
            "evaluation_criteria": evaluation_criteria,
            "settings": {"agent_speaks_first": agent_speaks_first, "max_turns": max_turns},
        }

        if stt:
            config["stt"] = stt.to_dict() if hasattr(stt, "to_dict") else stt
        if tts:
            config["tts"] = tts.to_dict() if hasattr(tts, "to_dict") else tts
        if llm:
            config["llm"] = llm.to_dict() if hasattr(llm, "to_dict") else llm

        # Mapping from interruption_sensitivity labels to probabilities
        interrupt_sensitivity_map = {
            "none": 0,
            "low": 0.25,
            "medium": 0.5,
            "high": 0.8,
        }

        # Run simulations sequentially
        results = []
        total_simulations = len(personas) * len(scenarios)
        simulation_count = 0

        for persona_index, user_persona in enumerate(personas):
            for scenario_index, scenario in enumerate(scenarios):
                simulation_count += 1
                try:
                    result = await run_single_simulation_task(
                        config=config,
                        persona_index=persona_index,
                        user_persona=user_persona,
                        scenario_index=scenario_index,
                        scenario=scenario,
                        output_dir=output_dir,
                        interrupt_sensitivity_map=interrupt_sensitivity_map,
                        base_port=port,
                    )
                    results.append(result)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    results.append(e)
                finally:
                    # Cleanup between simulations
                    if simulation_count < total_simulations:
                        gc.collect()
                        await asyncio.sleep(3.0)

        # Aggregate metrics
        all_simulation_metrics = []
        metrics_by_criterion = defaultdict(list)
        stt_llm_judge_scores = []

        for result in results:
            if isinstance(result, Exception):
                continue
            if result is None:
                continue

            sim_metrics_row, evaluation_results, stt_judge = result
            if sim_metrics_row is None:
                continue

            all_simulation_metrics.append(sim_metrics_row)

            for eval_result in evaluation_results:
                criterion_name = eval_result["name"]
                match_value = float(eval_result["match"])
                metrics_by_criterion[criterion_name].append(match_value)

            if stt_judge and "score" in stt_judge:
                stt_llm_judge_scores.append(stt_judge["score"])

        # Compute summary
        metrics_summary = {}
        for criterion_name, values in metrics_by_criterion.items():
            metrics_summary[criterion_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

        if stt_llm_judge_scores:
            metrics_summary["stt_llm_judge"] = {
                "mean": float(np.mean(stt_llm_judge_scores)),
                "std": float(np.std(stt_llm_judge_scores)),
                "values": stt_llm_judge_scores,
            }

        # Save results
        if all_simulation_metrics:
            df = pd.DataFrame(all_simulation_metrics)
            df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_summary, f, indent=4)

        return {
            "status": "completed",
            "output_dir": output_dir,
            "metrics": metrics_summary,
        }

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


# Create singleton instance
simulation = _Simulation()

# Re-export config classes
__all__ = ["simulation", "STTConfig", "TTSConfig", "LLMConfig"]
