# pense.llm module
"""
LLM evaluation module for tests and simulations.

Library Usage:
    from pense.llm import tests, simulations
    
    # Run LLM tests
    import asyncio
    result = asyncio.run(tests.run(
        config="./config.json",
        output_dir="./out",
        model="gpt-4.1",
        provider="openrouter"
    ))
    
    # Generate tests leaderboard
    tests.leaderboard(output_dir="./out", save_dir="./leaderboard")
    
    # Run LLM simulations
    result = asyncio.run(simulations.run(
        config="./config.json",
        output_dir="./out",
        model="gpt-4.1",
        provider="openrouter"
    ))
    
    # Generate simulations leaderboard
    simulations.leaderboard(output_dir="./out", save_dir="./leaderboard")
"""

from typing import Literal, Optional, List


class _Tests:
    """LLM Tests API."""

    @staticmethod
    async def run(
        config: str,
        output_dir: str = "./out",
        model: str = "gpt-4.1",
        provider: Literal["openai", "openrouter"] = "openrouter",
    ) -> dict:
        """
        Run LLM tests from a configuration file.

        Args:
            config: Path to JSON configuration file containing test cases
            output_dir: Path to output directory for results (default: ./out)
            model: Model name to use for evaluation
            provider: LLM provider (openai or openrouter)

        Returns:
            dict: Results containing test outcomes and metrics

        Example:
            >>> import asyncio
            >>> from pense.llm import tests
            >>> result = asyncio.run(tests.run(
            ...     config="./config.json",
            ...     output_dir="./out",
            ...     model="gpt-4.1",
            ...     provider="openrouter"
            ... ))
        """
        from pense.llm.run_tests import main as _llm_tests_main
        import sys

        # Build argument list
        argv = [
            "pense",
            "--config",
            config,
            "--output-dir",
            output_dir,
            "--model",
            model,
            "--provider",
            provider,
        ]

        # Save original sys.argv and restore after
        original_argv = sys.argv
        try:
            sys.argv = argv
            await _llm_tests_main()
            return {"status": "completed", "output_dir": output_dir}
        finally:
            sys.argv = original_argv

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

    # Allow calling run_test and run_inference directly for more control
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
        config: str,
        output_dir: str = "./out",
        model: str = "gpt-4.1",
        provider: Literal["openai", "openrouter"] = "openrouter",
        parallel: int = 1,
    ) -> dict:
        """
        Run LLM simulations from a configuration file.

        Args:
            config: Path to JSON configuration file containing simulation config
            output_dir: Path to output directory for results (default: ./out)
            model: Model name to use for both agent and user LLMs
            provider: LLM provider (openai or openrouter)
            parallel: Number of simulations to run in parallel (default: 1)

        Returns:
            dict: Results containing simulation outcomes and metrics

        Example:
            >>> import asyncio
            >>> from pense.llm import simulations
            >>> result = asyncio.run(simulations.run(
            ...     config="./config.json",
            ...     output_dir="./out",
            ...     model="gpt-4.1",
            ...     provider="openrouter"
            ... ))
        """
        from pense.llm.run_simulation import main as _llm_simulation_main
        import sys

        # Build argument list
        argv = [
            "pense",
            "--config",
            config,
            "--output-dir",
            output_dir,
            "--model",
            model,
            "--provider",
            provider,
            "--parallel",
            str(parallel),
        ]

        # Save original sys.argv and restore after
        original_argv = sys.argv
        try:
            sys.argv = argv
            await _llm_simulation_main()
            return {"status": "completed", "output_dir": output_dir}
        finally:
            sys.argv = original_argv

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

    # Allow calling run_simulation directly for more control
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
