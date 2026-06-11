# calibrate.agent module
"""
Voice agent testing and simulation module.

Library Usage:
    from calibrate.agent import simulation

    # Run agent simulation
    import asyncio
    result = asyncio.run(simulation.run(
        system_prompt="You are a helpful assistant...",
        tools=[...],
        personas=[...],
        scenarios=[...],
        evaluators=[...],
        output_dir="./out"
    ))
"""

from typing import Literal, List, Optional, Dict, Any
from dataclasses import dataclass, field
import os
import json
import asyncio
from collections import defaultdict
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
        evaluators: List[dict],
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
            evaluators: List of evaluator dicts with 'name', 'system_prompt', and optional 'judge_model'/'type'/'scale_min'/'scale_max'
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
            >>> from calibrate.agent import simulation, STTConfig, TTSConfig, LLMConfig
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
            ...     evaluators=[{"name": "completeness", "system_prompt": "Evaluate whether all questions were answered..."}],
            ...     output_dir="./out",
            ...     stt=STTConfig(provider="google"),
            ...     tts=TTSConfig(provider="google"),
            ...     llm=LLMConfig(provider="openrouter", model="openai/gpt-4.1"),
            ... ))
        """
        from calibrate.judges import require_simulation_evaluators, write_evaluator_config
        from calibrate.agent.run_simulation import run_single_simulation_task
        import gc

        require_simulation_evaluators(evaluators)

        os.makedirs(output_dir, exist_ok=True)
        write_evaluator_config(output_dir, evaluators)

        # Build config dict
        config = {
            "system_prompt": system_prompt,
            "tools": tools,
            "personas": personas,
            "scenarios": scenarios,
            "evaluators": evaluators,
            "settings": {
                "agent_speaks_first": agent_speaks_first,
                "max_turns": max_turns,
            },
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

        failed_simulations = [r for r in results if isinstance(r, Exception)]
        if failed_simulations:
            error_msgs = "; ".join(str(e) for e in failed_simulations)
            raise RuntimeError(
                f"{len(failed_simulations)} simulation(s) failed: {error_msgs}"
            )

        for result in results:
            if result is None:
                continue

            sim_metrics_row, evaluation_results, stt_judge = result
            if sim_metrics_row is None:
                continue

            all_simulation_metrics.append(sim_metrics_row)

            for eval_result in evaluation_results:
                criterion_name = eval_result["name"]
                # value works for both binary (0/1) and rating (int score)
                metrics_by_criterion[criterion_name].append(
                    float(eval_result["value"])
                )

            if stt_judge and "score" in stt_judge:
                stt_llm_judge_scores.append(stt_judge["score"])

        # Track criterion types and scale bounds for metrics.json enrichment
        criterion_types: dict = {}
        criterion_ids: dict = {}
        criterion_scales: dict = {}
        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            _, evaluation_results, _ = result
            for eval_result in evaluation_results or []:
                criterion_types.setdefault(
                    eval_result["name"], eval_result.get("type", "binary")
                )
                if "evaluator_id" in eval_result:
                    criterion_ids.setdefault(
                        eval_result["name"], eval_result["evaluator_id"]
                    )
                if "scale_min" in eval_result and "scale_max" in eval_result:
                    criterion_scales.setdefault(
                        eval_result["name"],
                        (
                            int(eval_result["scale_min"]),
                            int(eval_result["scale_max"]),
                        ),
                    )

        # Compute summary
        from calibrate.utils import summarize_metric_distribution

        metrics_summary = {}
        for criterion_name, values in metrics_by_criterion.items():
            metrics_summary[criterion_name] = summarize_metric_distribution(
                values,
                metric_type=criterion_types.get(criterion_name, "binary"),
                scale=criterion_scales.get(criterion_name),
                evaluator_id=criterion_ids.get(criterion_name),
            )

        if stt_llm_judge_scores:
            metrics_summary["stt_llm_judge"] = summarize_metric_distribution(
                stt_llm_judge_scores
            )

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
        evaluators: List[dict],
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
            evaluators: List of evaluator dicts with 'name', 'system_prompt', and optional 'judge_model'/'type'/'scale_min'/'scale_max'
            output_dir: Output directory for results
            interrupt_probability: Probability of user interrupting the agent (0.0-1.0)
            port: WebSocket port for the simulation
            agent_speaks_first: Whether the agent initiates the conversation
            max_turns: Maximum number of assistant turns

        Returns:
            dict: Simulation result with transcript, metrics, and evaluation

        Example:
            >>> import asyncio
            >>> from calibrate.agent import simulation
            >>> result = asyncio.run(simulation.run_single(
            ...     system_prompt="You are simulating a user...",
            ...     language="english",
            ...     gender="female",
            ...     evaluators=[{"name": "completeness", "system_prompt": "..."}],
            ...     output_dir="./out"
            ... ))
        """
        from calibrate.judges import require_simulation_evaluators, write_evaluator_config
        from calibrate.agent.run_simulation import run_simulation as _run_simulation

        require_simulation_evaluators(evaluators)
        write_evaluator_config(output_dir, evaluators)

        return await _run_simulation(
            system_prompt=system_prompt,
            language=language,
            gender=gender,
            evaluators=evaluators,
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
