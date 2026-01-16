import asyncio
import argparse
import json
import sys
from typing import List, Optional, Literal
from loguru import logger
import os
from os.path import join, exists, splitext, basename
import shutil
from collections import defaultdict
import traceback
from pense.utils import configure_print_logger, log_and_print
from pipecat.frames.frames import (
    TranscriptionFrame,
    LLMRunFrame,
    EndFrame,
    EndTaskFrame,
    LLMFullResponseEndFrame,
    CancelFrame,
    LLMMessagesAppendFrame,
    TextFrame,
    FunctionCallResultProperties,
)
from openai import AsyncOpenAI
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pense.llm.metrics import evaluate_simuation
import pandas as pd
import numpy as np

DEFAULT_MAX_TURNS = 50


class ConversationState:
    """Tracks conversation turns and coordinates termination across pipelines."""

    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self.turn_count = 0
        self.finished = False
        self._lock = asyncio.Lock()
        self._finish_notified = False
        self._is_max_turns_reached = False

    async def record_turn(self) -> bool:
        """Register a completed message turn.

        Returns:
            True if the conversation should continue, False if it reached the limit.
        """
        async with self._lock:
            if self.finished:
                return False

            self.turn_count += 1

            if self.turn_count >= self.max_turns:
                self._is_max_turns_reached = True
                self.finished = True
                return False

            return True

    async def mark_finished(self) -> bool:
        """Mark the conversation as finished and ensure it's done only once.

        Returns:
            True if this call marked the conversation finished, False otherwise.
        """
        async with self._lock:
            if self._finish_notified:
                return False

            self._finish_notified = True
            self.finished = True
            return True


class Processor(FrameProcessor):
    """Processor that captures LLM text output."""

    def __init__(
        self,
        speaks_first: bool,
        *,
        conversation_state: "ConversationState",
        name: str = "Processor",
        role: Literal["agent", "user"] = "agent",
    ):
        super().__init__(enable_direct_mode=True, name=name)
        self._current_response = ""
        self._collecting_response = False
        self._ready = False
        self._speaks_first = speaks_first
        self._conversation_state = conversation_state
        self._partner_task: Optional["PipelineTask"] = None
        self._self_end_sent = False
        self._role = role

    def set_task(self, task: "PipelineTask"):
        """Set the task reference after task creation."""
        self._task = task

    def set_partner(self, task: "PipelineTask"):
        """Set the partner task to exchange messages with."""
        self._partner_task = task

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        logger.info(f"text output processor frame: {frame}")

        if not self._ready:
            self._ready = True
            if self._task and self._speaks_first:
                await self._task.queue_frames(
                    [
                        LLMRunFrame(),
                    ]
                )

        # Capture text frames from LLM
        if isinstance(frame, TextFrame):
            logger.info(f"Received text frame: {frame}")
            text = frame.text
            if text:
                self._collecting_response = True
                self._current_response += text
                logger.info(f"Received text chunk: {text}")
                frame.includes_inter_frame_spaces = True

        # When we get an EndFrame after collecting text, save the complete response
        if isinstance(frame, LLMFullResponseEndFrame):
            response = self._current_response.strip()
            self._current_response = ""
            self._collecting_response = False

            if response:
                await self._handle_completed_response(response)
            elif self._conversation_state and self._conversation_state.finished:
                await self._end_conversation()

        await self.push_frame(frame, direction)

    async def _handle_completed_response(self, response: str):
        should_continue = True

        # Log the LLM message with role color
        color = (
            "\033[94m" if self._role == "agent" else "\033[93m"
        )  # Blue for bot, Yellow for user
        log_and_print(f"{color}[{self._role.capitalize()}]: {response}\033[0m")

        if self._conversation_state and self._role == "agent":
            should_continue = await self._conversation_state.record_turn()

        await self._forward_to_partner(response, run_partner=should_continue)

        if self._conversation_state and not should_continue:
            await self._end_conversation()

    async def _forward_to_partner(self, response: str, *, run_partner: bool):
        if not self._partner_task or not response:
            return

        frame = LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": response}],
            run_llm=run_partner,
        )

        await self._partner_task.queue_frames([frame])

    async def _end_conversation(self):
        notify_partner = False

        if self._conversation_state:
            notify_partner = await self._conversation_state.mark_finished()

        if notify_partner and self._partner_task:
            await self._partner_task.queue_frames([EndFrame()])

        if self._task and not self._self_end_sent:
            self._self_end_sent = True
            await self._task.queue_frames([EndFrame()])


async def run_simulation(
    bot_system_prompt: str,
    tools: List[dict],
    user_system_prompt: str,
    evaluation_criteria: list[dict],
    bot_model: str = "gpt-4.1",
    user_model: str = "gpt-4.1",
    bot_provider: str = "openai",
    user_provider: str = "openai",
    agent_speaks_first: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> List[str]:
    """Runs a text-only bot that processes text inputs through an LLM and returns text outputs."""
    # Create LLM service

    if bot_provider == "openrouter":
        bot_llm = OpenRouterLLMService(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=bot_model,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        bot_llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=bot_model,
        )

    if user_provider == "openrouter":
        user_llm = OpenRouterLLMService(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=user_model,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        user_llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=user_model,
        )

    conversation_state = ConversationState(max_turns=max_turns)

    # Create processors (text_output needs to be created first for reference)
    bot_processor = Processor(
        role="agent",
        speaks_first=agent_speaks_first,
        conversation_state=conversation_state,
        name="BotProcessor",
    )
    user_processor = Processor(
        role="user",
        speaks_first=not agent_speaks_first,
        conversation_state=conversation_state,
        name="UserProcessor",
    )

    # Create context with system prompt
    messages = [{"role": "system", "content": bot_system_prompt}]

    async def _exec_call_call():
        try:
            await bot_processor._end_conversation()
        except Exception as exc:
            logger.warning(
                f"Unable to cancel task after end_call (no tool_call_id): {exc}"
            )

    async def end_call(params: FunctionCallParams):
        reason = params.arguments.get("reason") if params.arguments else None
        if reason:
            log_and_print(f"tool call: end_call invoked by LLM: {reason}")
        else:
            log_and_print("tool call: end_call invoked by LLM")

        await params.result_callback(
            None, properties=FunctionCallResultProperties(run_llm=False)
        )
        await _exec_call_call()
        return

    async def generic_function_call(params: FunctionCallParams):
        log_and_print(
            f"tool call: {params.function_name} invoked with arguments: {params.arguments}"
        )

        await params.result_callback(
            {"status": "received"},
        )
        return

    end_call_tool = FunctionSchema(
        name="end_call",
        description="End the current call when the conversation is complete.",
        properties={
            "reason": {
                "type": "string",
                "description": "Optional explanation for why the call should end.",
            }
        },
        required=[],
    )
    standard_tools = [end_call_tool]
    bot_llm.register_function("end_call", end_call)

    for tool in tools:
        properties = {}
        for parameter in tool["parameters"]:
            prop = {
                "type": parameter["type"],
                "description": parameter["description"],
            }

            if "items" in parameter:
                prop["items"] = parameter["items"]

            if "enum" in parameter:
                prop["enum"] = parameter["enum"]

            properties[parameter["id"]] = prop

        tool_function = FunctionSchema(
            name=tool["name"],
            description=tool["description"],
            properties=properties,
            required=[],
        )
        standard_tools.append(tool_function)
        bot_llm.register_function(tool["name"], generic_function_call)

    tools = ToolsSchema(standard_tools=standard_tools)

    bot_context = LLMContext(messages, tools=tools)
    bot_context_aggregator = LLMContextAggregatorPair(bot_context)

    messages = [{"role": "system", "content": user_system_prompt}]
    user_context = LLMContext(messages)
    user_context_aggregator = LLMContextAggregatorPair(user_context)

    # Build pipeline with all processors
    bot_pipeline = Pipeline(
        [
            # text_input,
            bot_context_aggregator.user(),
            bot_llm,
            bot_processor,
            bot_context_aggregator.assistant(),
        ]
    )

    user_pipeline = Pipeline(
        [
            # text_input,
            user_context_aggregator.user(),
            user_llm,
            user_processor,
            user_context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        bot_pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[LLMLogObserver()],
    )

    user_task = PipelineTask(
        user_pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[LLMLogObserver()],
    )

    # Set task reference for text_input processor
    bot_processor.set_task(task)
    user_processor.set_task(user_task)

    bot_processor.set_partner(user_task)
    user_processor.set_partner(task)

    runner = PipelineRunner(handle_sigint=False)
    user_runner = PipelineRunner(handle_sigint=False)

    try:
        await asyncio.gather(
            runner.run(task),
            user_runner.run(user_task),
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise e

    transcript = [
        message
        for message in bot_context._messages
        if message["role"] not in ["system"]
    ]

    if conversation_state._is_max_turns_reached:
        transcript.append(
            {
                "role": "end_reason",
                "content": "max_turns",
            }
        )

    log_and_print(
        f"Evaluating the conversation based on the criteria:\n\n{evaluation_criteria}"
    )
    llm_judge_result = await evaluate_simuation(transcript, evaluation_criteria)

    evaluation_results = [
        {
            "name": criterion["name"],
            "value": int(llm_judge_result[criterion["name"]]["match"]),
            "reasoning": llm_judge_result[criterion["name"]]["reasoning"],
        }
        for criterion in evaluation_criteria
    ]

    return {
        "transcript": transcript,
        "evaluation_results": evaluation_results,
    }


async def run_single_simulation_task(
    semaphore: asyncio.Semaphore,
    config: dict,
    persona_index: int,
    user_persona: dict,
    scenario_index: int,
    scenario: dict,
    output_dir: str,
    args,
):
    """Run a single simulation task with semaphore for concurrency control."""
    async with semaphore:
        characteristics = user_persona.get("characteristics", "")
        gender = user_persona.get("gender", "")
        language = user_persona.get("language", "english")

        scenario_description = scenario.get("description", "")

        gender_prompt = f"\n\nYour gender is {gender}." if gender else ""
        user_system_prompt = f"You are a user speaking to an agent. This is your persona:\n\n{characteristics}{gender_prompt}\n\nThe following scenario will be played out: {scenario_description}. Make sure to respond to the agent to match the given scenario as per the given persona for you. You always speak in {language}."

        simulation_name = (
            f"simulation_persona_{persona_index + 1}_scenario_{scenario_index + 1}"
        )

        simulation_output_dir = f"{output_dir}/{simulation_name}"

        if exists(simulation_output_dir):
            shutil.rmtree(simulation_output_dir)

        os.makedirs(simulation_output_dir)

        logs_file_path = f"{output_dir}/{simulation_name}/logs"

        # Remove default logger to prevent logs from being printed to terminal
        logger.remove()

        # Create a unique logger for this simulation
        log_file_id = logger.add(
            logs_file_path,
            level="DEBUG",
            colorize=False,
        )

        agent_speaks_first = config.get("settings", {}).get("agent_speaks_first", True)

        print_log_save_path = f"{output_dir}/{simulation_name}/results.log"
        configure_print_logger(print_log_save_path)

        command = " ".join(sys.argv)
        log_and_print(f"\033[33mRunning command\033[0m: {command}")

        log_and_print("--------------------------------")
        log_and_print(f"""Running simulation \033[93m{simulation_name}\033[0m""")
        log_and_print(f"\033[93mPersona:\033[0m\n{characteristics}")
        log_and_print(f"\033[93mGender:\033[0m {gender}" if gender else "")
        log_and_print(f"\033[93mLanguage:\033[0m {language}")
        log_and_print(f"\033[93mScenario:\033[0m\n{scenario_description}")
        log_and_print(f"\033[93mAgent Speaks First:\033[0m {agent_speaks_first}")
        log_and_print("--------------------------------")

        try:
            output = await run_simulation(
                bot_system_prompt=config["system_prompt"]
                + f"\n\nYou must always speak in {language}.",
                tools=config["tools"],
                user_system_prompt=user_system_prompt,
                evaluation_criteria=config["evaluation_criteria"],
                bot_model=args.model,
                user_model=args.model,
                bot_provider=args.provider,
                user_provider=args.provider,
                agent_speaks_first=agent_speaks_first,
                max_turns=config.get("max_turns", DEFAULT_MAX_TURNS),
            )

            simulation_metrics = {
                "name": simulation_name,
            }

            for metric_dict in output["evaluation_results"]:
                simulation_metrics[metric_dict["name"]] = float(metric_dict["value"])

            with open(join(simulation_output_dir, "transcript.json"), "w") as f:
                json.dump(output["transcript"], f, indent=4)

            df = pd.DataFrame(output["evaluation_results"])
            df.to_csv(
                join(simulation_output_dir, "evaluation_results.csv"), index=False
            )

            # Save persona dict and scenario dict
            with open(join(simulation_output_dir, "config.json"), "w") as f:
                json.dump(
                    {
                        "persona": user_persona,
                        "scenario": scenario,
                    },
                    f,
                    indent=4,
                )

            return simulation_metrics, output["evaluation_results"]
        except Exception as e:
            traceback.print_exc()
        finally:
            logger.remove(log_file_id)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the config JSON file containing the evaluation config",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./out",
        help="Path to the output directory to save the results",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model to use for the evaluation",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default="openai",
        help="LLM provider to use (openai or openrouter)",
    )
    parser.add_argument(
        "-n",
        "--parallel",
        type=int,
        default=1,
        help="Number of simulations to run in parallel",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Create semaphore to limit parallel executions
    semaphore = asyncio.Semaphore(args.parallel)

    # Create all simulation tasks
    tasks = []
    for persona_index, user_persona in enumerate(config["personas"]):
        for scenario_index, scenario in enumerate(config["scenarios"]):
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

    # Run all tasks with controlled parallelism
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect metrics from results
    metrics = defaultdict(list)
    all_simulation_metrics = []

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Simulation failed with error: {result}")
            continue

        simulation_metrics, evaluation_results = result
        all_simulation_metrics.append(simulation_metrics)

        for metric_dict in evaluation_results:
            metrics[metric_dict["name"]].append(float(metric_dict["value"]))

    metrics_summary = {}

    for metric_name, metric_values in metrics.items():
        metrics_summary[metric_name] = {
            "mean": np.mean(metric_values),
            "std": np.std(metric_values),
            "values": metric_values,
        }

    df = pd.DataFrame(all_simulation_metrics)
    df.to_csv(join(output_dir, "results.csv"), index=False)

    with open(join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
