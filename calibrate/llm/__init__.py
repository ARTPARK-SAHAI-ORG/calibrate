# calibrate.llm module
"""
LLM evaluation module for tests and simulations.

Library Usage:
    from calibrate.llm import tests, simulations

    # Run LLM benchmark across multiple models (parallel + auto-leaderboard)
    import asyncio
    result = asyncio.run(tests.run(
        system_prompt="You are a helpful assistant...",
        tools=[...],
        test_cases=[...],
        output_dir="./out",
        models=["gpt-4.1", "claude-3.5-sonnet", "gemini-2.0-flash"],
        provider="openrouter"
    ))

    # Run single model evaluation (no leaderboard)
    result = asyncio.run(tests.run_single(
        system_prompt="You are a helpful assistant...",
        tools=[...],
        test_cases=[...],
        output_dir="./out",
        model="gpt-4.1",
        provider="openrouter"
    ))

    # Generate tests leaderboard separately
    tests.leaderboard(output_dir="./out", save_dir="./leaderboard")

    # Run LLM simulations benchmark across multiple models
    result = asyncio.run(simulations.run(
        system_prompt="You are a helpful assistant...",
        tools=[...],
        personas=[...],
        scenarios=[...],
        evaluators=[...],
        output_dir="./out",
        models=["gpt-4.1", "claude-3.5-sonnet"],
        provider="openrouter"
    ))

    # Run single model simulation (no leaderboard)
    result = asyncio.run(simulations.run_single(
        system_prompt="You are a helpful assistant...",
        tools=[...],
        personas=[...],
        scenarios=[...],
        evaluators=[...],
        output_dir="./out",
        model="gpt-4.1",
        provider="openrouter"
    ))

    # Generate simulations leaderboard separately
    simulations.leaderboard(output_dir="./out", save_dir="./leaderboard")
"""

from typing import Literal, Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from calibrate.connections import TextAgentConnection
import os
import json
import asyncio
from collections import defaultdict
import pandas as pd


class _Tests:
    """LLM Tests API."""

    @staticmethod
    async def _run_single_model(
        system_prompt: str,
        tools: List[dict],
        test_cases: List[dict],
        output_dir: str,
        model: str,
        provider: str,
        run_name: Optional[str] = None,
        agent: Optional["TextAgentConnection"] = None,
        evaluators: Optional[List[dict]] = None,
        test_parallel: Optional[int] = None,
    ) -> dict:
        """Run tests for a single model (or external agent)."""
        from calibrate.llm.run_tests import (
            run_test as _run_test,
            run_test_external as _run_test_external,
            _get_name_to_evaluator_dict,
            _evaluators_for_config_output,
            _resolve_evaluators_for_test_case,
            _run_items_parallel,
        )
        from calibrate.judges import write_evaluator_config
        from calibrate.utils import configure_print_logger, log_and_print

        # Create output directory
        if agent is not None and model:
            # Benchmarking with agent: use model name as subfolder so leaderboard works.
            save_folder_name = model.replace("/", "__")
            if run_name:
                final_output_dir = os.path.join(output_dir, run_name, save_folder_name)
            else:
                final_output_dir = os.path.join(output_dir, save_folder_name)
        elif agent is not None:
            # Single agent connection run: save directly to output_dir (no subfolder).
            final_output_dir = os.path.join(output_dir, run_name) if run_name else output_dir
        else:
            save_folder_name = f"{provider}/{model}" if provider == "openai" else f"{model}"
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

        results_file_path = os.path.join(final_output_dir, "results.json")

        # Pass model name to agent for benchmark routing; None for single runs.
        agent_model_hint: Optional[str] = model if (agent is not None and model) else None

        evaluator_config = {"evaluators": evaluators or []}
        name_to_evaluator = _get_name_to_evaluator_dict(evaluator_config)
        write_evaluator_config(
            output_dir, _evaluators_for_config_output(evaluator_config)
        )

        async def process(test_case_index: int, test_case: dict) -> dict:
            evaluation = test_case["evaluation"]
            resolved_evaluators = (
                _resolve_evaluators_for_test_case(
                    evaluation,
                    _get_name_to_evaluator_dict(
                        evaluator_config,
                        include_default=(evaluation.get("type") == "response"),
                    ),
                )
                if evaluation.get("type") in ("response", "conversation")
                else None
            )
            if agent is not None:
                result = await _run_test_external(
                    chat_history=test_case["history"],
                    evaluation=evaluation,
                    agent=agent,
                    model=agent_model_hint,
                    evaluators=resolved_evaluators,
                )
            else:
                result = await _run_test(
                    chat_history=test_case["history"],
                    evaluation=evaluation,
                    system_prompt=system_prompt,
                    model=model,
                    provider=provider,
                    tools=tools,
                    unique_id=run_name or "",
                    evaluators=resolved_evaluators,
                )

            if result["metrics"]["passed"]:
                log_and_print(f"✅ Test case {test_case_index + 1} passed")
            else:
                log_and_print(f"❌ Test case {test_case_index + 1} failed")
            if "reasoning" in result["metrics"]:
                log_and_print(result["metrics"]["reasoning"])

            if "id" in test_case:
                result["test_case_id"] = test_case["id"]
            result["test_case"] = test_case
            log_and_print("-" * 40)
            return result

        results = await _run_items_parallel(
            test_cases, process, results_file_path, test_parallel
        )

        total_passed = sum(1 for r in results if r["metrics"]["passed"])
        total_tests = len(results)
        failed_count = total_tests - total_passed

        if total_passed == total_tests:
            log_and_print("🎉 All tests passed!")
        elif failed_count == total_tests:
            log_and_print("❌ All tests failed!")
        else:
            log_and_print(
                f"✅ Total Passed: {total_passed}/{total_tests} ({(total_passed/total_tests)*100:.1f}%)"
            )
            log_and_print(
                f"❌ Total Failed: {failed_count}/{total_tests} ({(failed_count/total_tests)*100:.1f}%)"
            )

        # Save final results
        with open(os.path.join(final_output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        from calibrate.llm.run_tests import (
            _aggregate_criteria,
            _aggregate_tool_calls,
            _aggregate_cost,
            _aggregate_latency,
            _aggregate_total_tokens,
        )
        metrics = {
            "total": total_tests,
            "passed": total_passed,
            "criteria": _aggregate_criteria(results, name_to_evaluator),
            "tool_calls": _aggregate_tool_calls(results),
        }
        cost = _aggregate_cost(results)
        if cost is not None:
            metrics["cost"] = cost
        latency = _aggregate_latency(results)
        if latency is not None:
            metrics["latency_ms"] = latency
        total_tokens = _aggregate_total_tokens(results)
        if total_tokens is not None:
            metrics["total_tokens"] = total_tokens
        with open(os.path.join(final_output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        return {
            "status": "completed",
            "model": model,
            "output_dir": final_output_dir,
            "results": results,
            "metrics": metrics,
        }

    @staticmethod
    async def run(
        test_cases: List[dict],
        system_prompt: str = "",
        tools: List[dict] = None,
        output_dir: str = "./out",
        models: Optional[List[str]] = None,
        model: str = "gpt-4.1",
        provider: Literal["openai", "openrouter"] = "openrouter",
        run_name: Optional[str] = None,
        max_parallel: int = 2,
        agent: Optional["TextAgentConnection"] = None,
        evaluators: Optional[List[dict]] = None,
        test_parallel: Optional[int] = None,
    ) -> dict:
        """
        Run LLM tests with the given configuration.

        Pass ``agent`` to evaluate an external text agent instead of an internal
        LLM. When ``agent`` is provided, ``system_prompt``, ``tools``, ``model``,
        and ``provider`` are ignored.

        Supports running tests across multiple models in parallel (ignored when
        ``agent`` is provided).

        Args:
            test_cases: List of test case dicts, each containing 'history' and 'evaluation'
            system_prompt: System prompt for the LLM (ignored when agent is provided)
            tools: List of tool definitions available to the LLM (ignored when agent is provided)
            output_dir: Path to output directory for results (default: ./out)
            models: List of model names to evaluate (if provided, runs in parallel)
            model: Single model name to use (used if models is not provided)
            provider: LLM provider (openai or openrouter)
            run_name: Optional name for this run (used in output folder name)
            max_parallel: Maximum number of models to run in parallel (default: 2)
            test_parallel: Max test cases to evaluate concurrently per model.
            agent: Optional external agent connection. When provided, routes all
                test cases to the external agent instead of an internal LLM.
            evaluators: Optional list of evaluator dicts (each with ``name``,
                ``system_prompt``, ``judge_model``, ``type``, ...). Each test
                case's ``evaluation.criteria`` references these by name.
                If omitted, the implicit default LLM-test evaluator is used.

        Returns:
            dict: Results containing test outcomes and metrics for all models

        Example:
            >>> import asyncio
            >>> from calibrate.llm import tests
            >>> # Internal model
            >>> result = asyncio.run(tests.run(
            ...     system_prompt="You are a helpful assistant...",
            ...     tools=[...],
            ...     test_cases=[...],
            ...     model="gpt-4.1"
            ... ))
            >>> # External agent
            >>> from calibrate.connections import TextAgentConnection
            >>> result = asyncio.run(tests.run(
            ...     agent=TextAgentConnection(url="https://your-agent.com/chat"),
            ...     test_cases=[...],
            ... ))
        """
        tools = tools or []

        # External agent benchmark: run once per model, passing model hint in each request
        if agent is not None and models and len(models) > 0:
            semaphore = asyncio.Semaphore(max_parallel)

            async def run_agent_model(m: str) -> dict:
                async with semaphore:
                    return await _Tests._run_single_model(
                        system_prompt="",
                        tools=[],
                        test_cases=test_cases,
                        output_dir=output_dir,
                        model=m,
                        provider=provider,
                        run_name=run_name,
                        agent=agent,
                        evaluators=evaluators,
                        test_parallel=test_parallel,
                    )

            results = await asyncio.gather(*[run_agent_model(m) for m in models])
            return {m: r for m, r in zip(models, results)}

        # External agent: single run (no model selection) — pass empty model so
        # no model — save directly to output_dir (no subfolder)
        if agent is not None:
            return await _Tests._run_single_model(
                system_prompt=system_prompt,
                tools=tools,
                test_cases=test_cases,
                output_dir=output_dir,
                model="",
                provider=provider,
                run_name=run_name,
                agent=agent,
                evaluators=evaluators,
                test_parallel=test_parallel,
            )

        # If models list is provided, run in parallel
        if models and len(models) > 0:
            semaphore = asyncio.Semaphore(max_parallel)

            async def run_with_semaphore(m: str) -> dict:
                async with semaphore:
                    return await _Tests._run_single_model(
                        system_prompt=system_prompt,
                        tools=tools,
                        test_cases=test_cases,
                        output_dir=output_dir,
                        model=m,
                        provider=provider,
                        run_name=run_name,
                        evaluators=evaluators,
                        test_parallel=test_parallel,
                    )

            tasks = [run_with_semaphore(m) for m in models]
            model_results = await asyncio.gather(*tasks, return_exceptions=True)

            results_by_model = {}
            for i, result in enumerate(model_results):
                model_name = models[i]
                if isinstance(result, Exception):
                    results_by_model[model_name] = {
                        "status": "error",
                        "error": str(result),
                    }
                else:
                    results_by_model[model_name] = result

            return {
                "status": "completed",
                "output_dir": output_dir,
                "models": results_by_model,
            }

        # Single model - use original behavior
        return await _Tests._run_single_model(
            system_prompt=system_prompt,
            tools=tools,
            test_cases=test_cases,
            output_dir=output_dir,
            model=model,
            provider=provider,
            run_name=run_name,
            evaluators=evaluators,
            test_parallel=test_parallel,
        )

    @staticmethod
    async def run_single(
        test_cases: List[dict],
        system_prompt: str = "",
        tools: List[dict] = None,
        output_dir: str = "./out",
        model: str = "gpt-4.1",
        provider: Literal["openai", "openrouter"] = "openrouter",
        run_name: Optional[str] = None,
        agent: Optional["TextAgentConnection"] = None,
        evaluators: Optional[List[dict]] = None,
    ) -> dict:
        """
        Run LLM tests for a single model or external agent (no leaderboard).

        Args:
            test_cases: List of test case dicts
            system_prompt: System prompt for the LLM (ignored when agent is provided)
            tools: List of tool definitions available to the LLM (ignored when agent is provided)
            output_dir: Path to output directory for results (default: ./out)
            model: Model name to use
            provider: LLM provider (openai or openrouter)
            run_name: Optional name for this run (used in output folder name)
            agent: Optional external agent connection.

        Returns:
            dict: Results containing test outcomes and metrics
        """
        return await _Tests._run_single_model(
            system_prompt=system_prompt,
            tools=tools or [],
            test_cases=test_cases,
            output_dir=output_dir,
            model=model,
            provider=provider,
            run_name=run_name,
            agent=agent,
            evaluators=evaluators,
        )

    @staticmethod
    def leaderboard(output_dir: str, save_dir: str) -> None:
        """
        Generate LLM tests leaderboard from evaluation results.

        Args:
            output_dir: Path to directory containing test results
            save_dir: Path to directory where leaderboard will be saved

        Example:
            >>> from calibrate.llm import tests
            >>> tests.leaderboard(output_dir="./out", save_dir="./leaderboard")
        """
        from calibrate.llm.tests_leaderboard import generate_leaderboard

        generate_leaderboard(output_dir=output_dir, save_dir=save_dir)

    @staticmethod
    async def run_test(
        chat_history: List[dict],
        evaluation: dict,
        system_prompt: str,
        model: str,
        provider: str,
        tools: List[dict] = None,
        evaluators: Optional[List[dict]] = None,
    ) -> dict:
        """
        Run a single LLM test case.

        Args:
            chat_history: List of chat messages (role/content dicts)
            evaluation: Evaluation dict with ``type`` and (for ``response``)
                ``criteria`` referencing evaluators by name.
            system_prompt: System prompt for the LLM
            model: Model name
            provider: LLM provider
            tools: Optional list of tool definitions
            evaluators: Optional list of evaluator dicts. If omitted, the
                implicit default LLM-test evaluator is used.

        Returns:
            dict: Test result with output and metrics
        """
        from calibrate.llm.run_tests import (
            run_test as _run_test,
            _get_name_to_evaluator_dict,
            _resolve_evaluators_for_test_case,
        )

        evaluator_config = {"evaluators": evaluators or []}
        resolved_evaluators = (
            _resolve_evaluators_for_test_case(
                evaluation,
                _get_name_to_evaluator_dict(
                    evaluator_config,
                    include_default=(evaluation.get("type") == "response"),
                ),
            )
            if evaluation.get("type") in ("response", "conversation")
            else None
        )

        return await _run_test(
            chat_history=chat_history,
            evaluation=evaluation,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            tools=tools or [],
            unique_id="",
            evaluators=resolved_evaluators,
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
        from calibrate.llm.run_tests import run_inference as _run_inference

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
    async def _run_single_model(
        system_prompt: str,
        tools: List[dict],
        personas: List[dict],
        scenarios: List[dict],
        evaluators: List[dict],
        output_dir: str,
        model: str,
        provider: str,
        parallel: int,
        agent_speaks_first: bool,
        max_turns: int,
        agent: Optional["TextAgentConnection"] = None,
        _flat_output: bool = False,
    ) -> dict:
        """Run simulations for a single model or external agent.

        Args:
            _flat_output: When True, save results directly to output_dir instead of
                a model-specific subfolder. Used by the single-model path in run()
                for backward compatibility.
        """
        from calibrate.judges import require_simulation_evaluators, write_evaluator_config
        from calibrate.llm.run_simulation import run_single_simulation_task

        require_simulation_evaluators(evaluators or [])

        # Create output directory — flat for single-model runs, model-scoped for benchmarks
        if _flat_output:
            final_output_dir = output_dir
        else:
            save_folder_name = f"{provider}/{model}" if provider == "openai" else f"{model}"
            save_folder_name = save_folder_name.replace("/", "__")
            final_output_dir = os.path.join(output_dir, save_folder_name)

        os.makedirs(final_output_dir, exist_ok=True)
        write_evaluator_config(output_dir, evaluators or [])

        # Build config dict for run_single_simulation_task
        config = {
            "system_prompt": system_prompt,
            "tools": tools,
            "personas": personas,
            "scenarios": scenarios,
            "evaluators": evaluators or [],
            "settings": {
                "agent_speaks_first": agent_speaks_first,
                "max_turns": max_turns,
            },
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
                    output_dir=final_output_dir,
                    args=args,
                    agent=agent,
                )
                tasks.append(task)

        # Run all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect metrics
        metrics_by_criterion = defaultdict(list)
        criterion_types: dict = {}  # name -> "binary" | "rating"
        criterion_ids: dict = {}
        criterion_scales: dict = {}  # name -> (scale_min, scale_max) for ratings
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
                    metrics_by_criterion[eval_result["name"]].append(
                        float(eval_result["value"])
                    )
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

        # Save results
        if all_simulation_metrics:
            df = pd.DataFrame(all_simulation_metrics)
            df.to_csv(os.path.join(final_output_dir, "results.csv"), index=False)

        with open(os.path.join(final_output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_summary, f, indent=4)

        return {
            "status": "completed",
            "model": model,
            "output_dir": final_output_dir,
            "metrics": metrics_summary,
        }

    @staticmethod
    async def run(
        personas: List[dict],
        scenarios: List[dict],
        evaluators: List[dict],
        system_prompt: str = "",
        tools: List[dict] = None,
        output_dir: str = "./out",
        models: Optional[List[str]] = None,
        model: str = "gpt-4.1",
        provider: Literal["openai", "openrouter"] = "openrouter",
        parallel: int = 1,
        agent_speaks_first: bool = True,
        max_turns: int = 50,
        max_parallel_models: int = 2,
        agent: Optional["TextAgentConnection"] = None,
    ) -> dict:
        """
        Run LLM simulations with the given configuration.

        Pass ``agent`` to simulate against an external text agent instead of an
        internal LLM. When ``agent`` is provided, ``system_prompt``, ``tools``,
        ``model``, ``provider``, and ``models`` are ignored.

        Args:
            personas: List of persona dicts with 'characteristics', 'gender', 'language'
            scenarios: List of scenario dicts with 'description'
            evaluators: List of evaluator dicts with 'name', 'system_prompt',
                'judge_model', and optional 'type'/'scale_min'/'scale_max'.
                At least one evaluator is required (simulations have no implicit default).
            system_prompt: System prompt for the bot/agent (ignored when agent is provided)
            tools: List of tool definitions available to the agent (ignored when agent is provided)
            output_dir: Path to output directory for results (default: ./out)
            models: List of model names to evaluate (if provided, runs in parallel)
            model: Single model name to use (used if models is not provided)
            provider: LLM provider (openai or openrouter)
            parallel: Number of simulations to run in parallel per model (default: 1)
            agent_speaks_first: Whether the agent initiates the conversation (default: True)
            max_turns: Maximum number of assistant turns (default: 50)
            max_parallel_models: Maximum number of models to run in parallel (default: 2)
            agent: Optional external agent connection.

        Returns:
            dict: Results containing simulation outcomes and metrics

        Example:
            >>> import asyncio
            >>> from calibrate.llm import simulations
            >>> # Internal model
            >>> result = asyncio.run(simulations.run(
            ...     system_prompt="You are a helpful nurse...",
            ...     tools=[...],
            ...     personas=[...],
            ...     scenarios=[...],
            ...     evaluators=[...],
            ...     model="gpt-4.1"
            ... ))
            >>> # External agent
            >>> from calibrate.connections import TextAgentConnection
            >>> result = asyncio.run(simulations.run(
            ...     agent=TextAgentConnection(url="https://your-agent.com/chat"),
            ...     personas=[...],
            ...     scenarios=[...],
            ...     evaluators=[...],
            ... ))
        """
        tools = tools or []
        os.makedirs(output_dir, exist_ok=True)

        # External agent: single run, no multi-model loop
        if agent is not None:
            return await _Simulations._run_single_model(
                system_prompt=system_prompt,
                tools=tools,
                personas=personas,
                scenarios=scenarios,
                evaluators=evaluators,
                output_dir=output_dir,
                model=model,
                provider=provider,
                parallel=parallel,
                agent_speaks_first=agent_speaks_first,
                max_turns=max_turns,
                agent=agent,
            )

        # If models list is provided, run in parallel
        if models and len(models) > 0:
            semaphore = asyncio.Semaphore(max_parallel_models)

            async def run_with_semaphore(m: str) -> dict:
                async with semaphore:
                    return await _Simulations._run_single_model(
                        system_prompt=system_prompt,
                        tools=tools,
                        personas=personas,
                        scenarios=scenarios,
                        evaluators=evaluators,
                        output_dir=output_dir,
                        model=m,
                        provider=provider,
                        parallel=parallel,
                        agent_speaks_first=agent_speaks_first,
                        max_turns=max_turns,
                    )

            tasks = [run_with_semaphore(m) for m in models]
            model_results = await asyncio.gather(*tasks, return_exceptions=True)

            results_by_model = {}
            for i, result in enumerate(model_results):
                model_name = models[i]
                if isinstance(result, Exception):
                    results_by_model[model_name] = {
                        "status": "error",
                        "error": str(result),
                    }
                else:
                    results_by_model[model_name] = result

            return {
                "status": "completed",
                "output_dir": output_dir,
                "models": results_by_model,
            }

        # Single model — save directly to output_dir (no model subfolder) for backward compatibility
        return await _Simulations._run_single_model(
            system_prompt=system_prompt,
            tools=tools,
            personas=personas,
            scenarios=scenarios,
            evaluators=evaluators,
            output_dir=output_dir,
            model=model,
            provider=provider,
            parallel=parallel,
            agent_speaks_first=agent_speaks_first,
            max_turns=max_turns,
            _flat_output=True,
        )

    @staticmethod
    async def run_single(
        personas: List[dict],
        scenarios: List[dict],
        evaluators: List[dict],
        system_prompt: str = "",
        tools: List[dict] = None,
        output_dir: str = "./out",
        model: str = "gpt-4.1",
        provider: Literal["openai", "openrouter"] = "openrouter",
        parallel: int = 1,
        agent_speaks_first: bool = True,
        max_turns: int = 50,
        agent: Optional["TextAgentConnection"] = None,
    ) -> dict:
        """
        Run LLM simulations for a single model or external agent (no leaderboard).

        Args:
            personas: List of persona dicts with 'characteristics', 'gender', 'language'
            scenarios: List of scenario dicts with 'description'
            evaluators: List of evaluator dicts (top-level); at least one is required.
            system_prompt: System prompt for the bot/agent (ignored when agent is provided)
            tools: List of tool definitions available to the agent (ignored when agent is provided)
            output_dir: Path to output directory for results (default: ./out)
            model: Model name to use
            provider: LLM provider (openai or openrouter)
            parallel: Number of simulations to run in parallel (default: 1)
            agent_speaks_first: Whether the agent initiates the conversation (default: True)
            max_turns: Maximum number of assistant turns (default: 50)
            agent: Optional external agent connection.

        Returns:
            dict: Results containing simulation outcomes and metrics
        """
        return await _Simulations._run_single_model(
            system_prompt=system_prompt,
            tools=tools or [],
            personas=personas,
            scenarios=scenarios,
            evaluators=evaluators,
            output_dir=output_dir,
            model=model,
            provider=provider,
            parallel=parallel,
            agent_speaks_first=agent_speaks_first,
            max_turns=max_turns,
            agent=agent,
        )

    @staticmethod
    def leaderboard(output_dir: str, save_dir: str) -> None:
        """
        Generate LLM simulations leaderboard from evaluation results.

        Args:
            output_dir: Path to directory containing simulation results
            save_dir: Path to directory where leaderboard will be saved

        Example:
            >>> from calibrate.llm import simulations
            >>> simulations.leaderboard(output_dir="./out", save_dir="./leaderboard")
        """
        from calibrate.llm.simulation_leaderboard import generate_leaderboard

        generate_leaderboard(output_dir=output_dir, save_dir=save_dir)

    @staticmethod
    async def run_simulation(
        bot_system_prompt: str,
        tools: List[dict],
        user_system_prompt: str,
        evaluators: List[dict],
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
            evaluators: List of evaluator dicts (top-level); at least one is required.
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
        from calibrate.llm.run_simulation import run_simulation as _run_simulation

        return await _run_simulation(
            bot_system_prompt=bot_system_prompt,
            tools=tools,
            user_system_prompt=user_system_prompt,
            evaluators=evaluators,
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
