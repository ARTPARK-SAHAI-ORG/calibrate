import asyncio
import argparse
import json
import sys
from typing import List, Optional, Literal
from loguru import logger
from dotenv import load_dotenv
import os
from os.path import join, exists
import shutil
from collections import defaultdict

from agentloop.utils import configure_print_logger, log_and_print
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
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from agentloop.llm.metrics import evaluate_simuation
import pandas as pd
import numpy as np

load_dotenv(override=True)


class ConversationState:
    """Tracks conversation turns and coordinates termination across pipelines."""

    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self.turn_count = 0
        self.finished = False
        self._lock = asyncio.Lock()
        self._finish_notified = False

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

        if self._conversation_state:
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
    bot_speaks_first: bool = True,
    max_turns: int = 10,
) -> List[str]:
    """Runs a text-only bot that processes text inputs through an LLM and returns text outputs."""
    # Create LLM service

    bot_llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=bot_model,
    )

    user_llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=user_model,
    )

    conversation_state = ConversationState(max_turns=max_turns)

    # Create processors (text_output needs to be created first for reference)
    bot_processor = Processor(
        role="agent",
        speaks_first=bot_speaks_first,
        conversation_state=conversation_state,
        name="BotProcessor",
    )
    user_processor = Processor(
        role="user",
        speaks_first=not bot_speaks_first,
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
        raise

    transcript = [
        message
        for message in bot_context._messages
        if message["role"] not in ["system", "tool"]
    ]

    llm_judge_result = await evaluate_simuation(transcript, evaluation_criteria)

    evaluation_results = [
        {
            "name": criterion["name"],
            "match": llm_judge_result[criterion["name"]]["match"],
            "reasoning": llm_judge_result[criterion["name"]]["reasoning"],
        }
        for criterion in evaluation_criteria
    ]

    return {
        "transcript": transcript,
        "evaluation_results": evaluation_results,
    }


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

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    metrics = defaultdict(list)
    all_simulation_metrics = []

    for persona_index, user_persona in enumerate(config["personas"]):
        for scenario_index, scenario in enumerate(config["scenarios"]):

            user_system_prompt = f"You are a user speaking to an agent. This is your persona:\n\n{user_persona}\n\nThe following scenario will be played out: {scenario}. Make sure to respond to the agent to match the given scenario as per the given persona for you."

            simulation_name = (
                f"simulation_persona_{persona_index + 1}_scenario_{scenario_index + 1}"
            )

            simulation_output_dir = f"{args.output_dir}/{simulation_name}"

            if exists(simulation_output_dir):
                shutil.rmtree(simulation_output_dir)

            os.makedirs(simulation_output_dir)

            logs_file_path = f"{args.output_dir}/{simulation_name}/logs"

            logger.remove()
            log_file_id = logger.add(
                logs_file_path,
                level="DEBUG",
                colorize=False,
            )

            print_log_save_path = f"{args.output_dir}/{simulation_name}/results.log"
            configure_print_logger(print_log_save_path)

            command = " ".join(sys.argv)
            log_and_print(f"\033[33mRunning command\033[0m: {command}")

            log_and_print("--------------------------------")
            log_and_print(f"""Running simulation \033[93m{simulation_name}\033[0m""")
            log_and_print(f"\033[93mPersona:\033[0m\n{user_persona}")
            log_and_print(f"\033[93mScenario:\033[0m\n{scenario}")
            log_and_print("--------------------------------")

            output = await run_simulation(
                bot_system_prompt=config["agent_system_prompt"]
                + f"\n\nYou must always speak in {config['language']}.",
                tools=config["tools"],
                user_system_prompt=user_system_prompt,
                evaluation_criteria=config["evaluation_criteria"],
                bot_model=args.model,
                user_model=args.model,
                max_turns=config.get("max_turns", 50),
            )

            simulation_metrics = {
                "name": simulation_name,
            }

            for metric_dict in output["evaluation_results"]:
                metrics[metric_dict["name"]].append(float(metric_dict["match"]))
                simulation_metrics[metric_dict["name"]] = float(metric_dict["match"])

            all_simulation_metrics.append(simulation_metrics)

            with open(join(simulation_output_dir, "transcript.json"), "w") as f:
                json.dump(output["transcript"], f, indent=4)

            df = pd.DataFrame(output["evaluation_results"])
            df.to_csv(
                join(simulation_output_dir, "evaluation_results.csv"), index=False
            )

            logger.remove(log_file_id)

        #     break

        # break

    metrics_summary = {}

    for metric_name, metric_values in metrics.items():
        metrics_summary[metric_name] = {
            "mean": np.mean(metric_values),
            "std": np.std(metric_values),
            "values": metric_values,
        }

    df = pd.DataFrame(all_simulation_metrics)
    df.to_csv(join(args.output_dir, "results.csv"), index=False)

    with open(join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
