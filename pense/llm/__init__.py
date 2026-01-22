# pense.llm module
"""
LLM evaluation module for tests and simulations.

Library Usage:
    from pense.llm import tests, simulations
    
    # Run LLM tests
    import asyncio
    result = asyncio.run(tests.run(
        system_prompt="You are a helpful assistant...",
        tools=[...],
        test_cases=[...],
        output_dir="./out",
        model="gpt-4.1",
        provider="openrouter"
    ))
    
    # Generate tests leaderboard
    tests.leaderboard(output_dir="./out", save_dir="./leaderboard")
    
    # Run LLM simulations
    result = asyncio.run(simulations.run(
        system_prompt="You are a helpful assistant...",
        tools=[...],
        personas=[...],
        scenarios=[...],
        evaluation_criteria=[...],
        output_dir="./out",
        model="gpt-4.1",
        provider="openrouter"
    ))
    
    # Generate simulations leaderboard
    simulations.leaderboard(output_dir="./out", save_dir="./leaderboard")
"""

from typing import Literal, Optional, List, Dict, Any
import os
import json
import asyncio
from collections import defaultdict
import numpy as np
import pandas as pd


class _Tests:
    """LLM Tests API."""

    @staticmethod
    async def run(
        system_prompt: str,
        tools: List[dict],
        test_cases: List[dict],
        output_dir: str = "./out",
        model: str = "gpt-4.1",
        provider: Literal["openai", "openrouter"] = "openrouter",
        run_name: Optional[str] = None,
    ) -> dict:
        """
        Run LLM tests with the given configuration.

        Args:
            system_prompt: System prompt for the LLM
            tools: List of tool definitions available to the LLM
            test_cases: List of test case dicts, each containing 'history', 'evaluation', and optional 'settings'
            output_dir: Path to output directory for results (default: ./out)
            model: Model name to use for evaluation
            provider: LLM provider (openai or openrouter)
            run_name: Optional name for this run (used in output folder name)

        Returns:
            dict: Results containing test outcomes and metrics

        Example:
            >>> import asyncio
            >>> from pense.llm import tests
            >>> result = asyncio.run(tests.run(
            ...     system_prompt="You are a helpful assistant...",
            ...     tools=[{
            ...         "type": "client",
            ...         "name": "get_weather",
            ...         "description": "Get weather",
            ...         "parameters": [{"id": "location", "type": "string", "description": "City", "required": True}]
            ...     }],
            ...     test_cases=[{
            ...         "history": [{"role": "user", "content": "What's the weather in NYC?"}],
            ...         "evaluation": {"type": "tool_call", "tool_calls": [{"tool": "get_weather"}]}
            ...     }],
            ...     output_dir="./out",
            ...     model="gpt-4.1",
            ...     provider="openrouter"
            ... ))
        """
        from pense.llm.run_tests import run_test as _run_test
        from pense.utils import configure_print_logger, log_and_print

        # Create output directory
        save_folder_name = (
            f"{provider}/{model}" if provider == "openai" else f"{model}"
        )
        save_folder_name = save_folder_name.replace("/", "__")
        
        if run_name:
            final_output_dir = os.path.join(output_dir, run_name, save_folder_name)
        else:
            final_output_dir = os.path.join(output_dir, save_folder_name)

        os.makedirs(final_output_dir, exist_ok=True)

        log_save_path = os.path.join(final_output_dir, "logs")
        if os.path.exists(log_save_path):
            os.remove(log_save_path)

        print_log_save_path = os.path.join(final_output_dir, "results.log")
        if os.path.exists(print_log_save_path):
            os.remove(print_log_save_path)

        configure_print_logger(print_log_save_path)

        results = []
        results_file_path = os.path.join(final_output_dir, "results.json")

        for test_case_index, test_case in enumerate(test_cases):
            agent_language = test_case.get("settings", {}).get("language", "english")
            result = await _run_test(
                chat_history=test_case["history"],
                evaluation=test_case["evaluation"],
                system_prompt=system_prompt + f"\n\nYou must always speak in {agent_language}.",
                model=model,
                provider=provider,
                tools=tools,
            )

            if result["metrics"]["passed"]:
                log_and_print(f"✅ Test case {test_case_index + 1} passed")
            else:
                log_and_print(f"❌ Test case {test_case_index + 1} failed")
                if "reasoning" in result["metrics"]:
                    log_and_print(result["metrics"]["reasoning"])

            result["test_case"] = test_case
            results.append(result)

            # Save intermediate results
            with open(results_file_path, "w") as f:
                json.dump(results, f, indent=4)

            log_and_print("-" * 40)

        total_passed = sum(1 for r in results if r["metrics"]["passed"])
        total_tests = len(results)
        failed_count = total_tests - total_passed

        if total_passed == total_tests:
            log_and_print("🎉 All tests passed!")
        elif failed_count == total_tests:
            log_and_print("❌ All tests failed!")
        else:
            log_and_print(f"✅ Total Passed: {total_passed}/{total_tests} ({(total_passed/total_tests)*100:.1f}%)")
            log_and_print(f"❌ Total Failed: {failed_count}/{total_tests} ({(failed_count/total_tests)*100:.1f}%)")

        # Save final results
        with open(os.path.join(final_output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        metrics = {"total": total_tests, "passed": total_passed}
        with open(os.path.join(final_output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        return {
            "status": "completed",
            "output_dir": final_output_dir,
            "results": results,
            "metrics": metrics,
        }

    @staticmethod
    def leaderboard(output_dir: str, save_dir: str) -> None:
        """
        Generate LLM tests leaderboard from evaluation results.

        Args:
            output_dir: Path to directory containing test results
            save_dir: Path to directory where leaderboard will be saved

        Example:
            >>> from pense.llm import tests
            >>> tests.leaderboard(output_dir="./out", save_dir="./leaderboard")
        """
        from pense.llm.tests_leaderboard import generate_leaderboard

        generate_leaderboard(output_dir=output_dir, save_dir=save_dir)

    @staticmethod
    async def run_test(
        chat_history: List[dict],
        evaluation: dict,
        system_prompt: str,
        model: str,
        provider: str,
        tools: List[dict] = None,
    ) -> dict:
        """
        Run a single LLM test case.

        Args:
            chat_history: List of chat messages (role/content dicts)
            evaluation: Evaluation criteria dict
            system_prompt: System prompt for the LLM
            model: Model name
            provider: LLM provider
            tools: Optional list of tool definitions

        Returns:
            dict: Test result with output and metrics
        """
        from pense.llm.run_tests import run_test as _run_test

        return await _run_test(
            chat_history=chat_history,
            evaluation=evaluation,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            tools=tools or [],
        )

    @staticmethod
    async def run_inference(
        chat_history: List[dict],
        system_prompt: str,
        model: str,
        provider: str,
        tools: List[dict] = None,
    ) -> dict:
        """
        Run LLM inference without evaluation.

        Args:
            chat_history: List of chat messages (role/content dicts)
            system_prompt: System prompt for the LLM
            model: Model name
            provider: LLM provider
            tools: Optional list of tool definitions

        Returns:
            dict: Response and tool calls from the LLM
        """
        from pense.llm.run_tests import run_inference as _run_inference

        return await _run_inference(
            chat_history=chat_history,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            tools=tools or [],
        )


class _Simulations:
    """LLM Simulations API."""

    @staticmethod
    async def run(
        system_prompt: str,
        tools: List[dict],
        personas: List[dict],
        scenarios: List[dict],
        evaluation_criteria: List[dict],
        output_dir: str = "./out",
        model: str = "gpt-4.1",
        provider: Literal["openai", "openrouter"] = "openrouter",
        parallel: int = 1,
        agent_speaks_first: bool = True,
        max_turns: int = 50,
    ) -> dict:
        """
        Run LLM simulations with the given configuration.

        Args:
            system_prompt: System prompt for the bot/agent
            tools: List of tool definitions available to the agent
            personas: List of persona dicts with 'characteristics', 'gender', 'language'
            scenarios: List of scenario dicts with 'description'
            evaluation_criteria: List of criteria dicts with 'name' and 'description'
            output_dir: Path to output directory for results (default: ./out)
            model: Model name to use for both agent and user LLMs
            provider: LLM provider (openai or openrouter)
            parallel: Number of simulations to run in parallel (default: 1)
            agent_speaks_first: Whether the agent initiates the conversation (default: True)
            max_turns: Maximum number of assistant turns (default: 50)

        Returns:
            dict: Results containing simulation outcomes and metrics

        Example:
            >>> import asyncio
            >>> from pense.llm import simulations
            >>> result = asyncio.run(simulations.run(
            ...     system_prompt="You are a helpful nurse...",
            ...     tools=[...],
            ...     personas=[{"characteristics": "...", "gender": "female", "language": "english"}],
            ...     scenarios=[{"description": "User completes the form"}],
            ...     evaluation_criteria=[{"name": "completeness", "description": "All questions answered"}],
            ...     output_dir="./out",
            ...     model="gpt-4.1",
            ...     provider="openrouter"
            ... ))
        """
        from pense.llm.run_simulation import run_single_simulation_task

        os.makedirs(output_dir, exist_ok=True)

        # Build config dict for run_single_simulation_task
        config = {
            "system_prompt": system_prompt,
            "tools": tools,
            "personas": personas,
            "scenarios": scenarios,
            "evaluation_criteria": evaluation_criteria,
            "settings": {"agent_speaks_first": agent_speaks_first},
            "max_turns": max_turns,
        }

        # Create a mock args object
        class Args:
            pass
        args = Args()
        args.model = model
        args.provider = provider

        # Create semaphore for parallel execution
        semaphore = asyncio.Semaphore(parallel)

        # Create all simulation tasks
        tasks = []
        for persona_index, user_persona in enumerate(personas):
            for scenario_index, scenario in enumerate(scenarios):
                task = run_single_simulation_task(
                    semaphore=semaphore,
                    config=config,
                    persona_index=persona_index,
                    user_persona=user_persona,
                    scenario_index=scenario_index,
                    scenario=scenario,
                    output_dir=output_dir,
                    args=args,
                )
                tasks.append(task)

        # Run all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect metrics
        metrics_by_criterion = defaultdict(list)
        all_simulation_metrics = []

        for result in results:
            if isinstance(result, Exception):
                continue
            if result is None:
                continue

            simulation_metrics, evaluation_results = result
            if simulation_metrics:
                all_simulation_metrics.append(simulation_metrics)
                for eval_result in evaluation_results:
                    metrics_by_criterion[eval_result["name"]].append(float(eval_result["value"]))

        # Compute summary
        metrics_summary = {}
        for criterion_name, values in metrics_by_criterion.items():
            metrics_summary[criterion_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
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
    def leaderboard(output_dir: str, save_dir: str) -> None:
        """
        Generate LLM simulations leaderboard from evaluation results.

        Args:
            output_dir: Path to directory containing simulation results
            save_dir: Path to directory where leaderboard will be saved

        Example:
            >>> from pense.llm import simulations
            >>> simulations.leaderboard(output_dir="./out", save_dir="./leaderboard")
        """
        from pense.llm.simulation_leaderboard import generate_leaderboard

        generate_leaderboard(output_dir=output_dir, save_dir=save_dir)

    @staticmethod
    async def run_simulation(
        bot_system_prompt: str,
        tools: List[dict],
        user_system_prompt: str,
        evaluation_criteria: List[dict],
        bot_model: str = "gpt-4.1",
        user_model: str = "gpt-4.1",
        bot_provider: str = "openai",
        user_provider: str = "openai",
        agent_speaks_first: bool = True,
        max_turns: int = 50,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Run a single LLM simulation.

        Args:
            bot_system_prompt: System prompt for the bot/agent
            tools: List of tool definitions available to the bot
            user_system_prompt: System prompt for the simulated user
            evaluation_criteria: List of evaluation criteria dicts
            bot_model: Model name for the bot
            user_model: Model name for the simulated user
            bot_provider: LLM provider for the bot
            user_provider: LLM provider for the simulated user
            agent_speaks_first: Whether the agent initiates the conversation
            max_turns: Maximum number of assistant turns
            output_dir: Optional output directory for intermediate transcripts

        Returns:
            dict: Simulation result with transcript and evaluation
        """
        from pense.llm.run_simulation import run_simulation as _run_simulation

        return await _run_simulation(
            bot_system_prompt=bot_system_prompt,
            tools=tools,
            user_system_prompt=user_system_prompt,
            evaluation_criteria=evaluation_criteria,
            bot_model=bot_model,
            user_model=user_model,
            bot_provider=bot_provider,
            user_provider=user_provider,
            agent_speaks_first=agent_speaks_first,
            max_turns=max_turns,
            output_dir=output_dir,
        )


# Create singleton instances
tests = _Tests()
simulations = _Simulations()

__all__ = ["tests", "simulations"]
