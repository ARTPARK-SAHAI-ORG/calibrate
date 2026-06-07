import asyncio
import argparse
import json
import sys
import uuid
from typing import List, Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from calibrate.connections import TextAgentConnection
from loguru import logger
import os
from os.path import join, exists, splitext, basename
import shutil
from collections import defaultdict
import traceback
from calibrate.utils import (
    configure_print_logger,
    log_and_print,
    build_tools_schema,
    make_webhook_call,
    cleanup_print_logger,
    current_simulation_name,
    provider_log_file,
)
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
from pipecat.adapters.schemas.function_schema import FunctionSchema
from calibrate.judges import require_simulation_evaluators
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
from calibrate.llm.metrics import evaluate_simuation, DEFAULT_SIMULATION_JUDGE_MODEL
from calibrate.judges import (
    attach_evaluator_id,
    evaluator_result_value,
    format_evaluation_result_lines,
    is_rating,
    write_evaluator_config,
)


async def _judge_and_emit(
    transcript: list,
    evaluators: list,
    fallback_judge_model: str,
    emit,
) -> list[dict]:
    """Run the simulation judge, build evaluation rows, and stream them via ``emit``.

    ``emit`` is called with the prelude line and each formatted evaluator-row
    line. Caller controls the sink (``log_and_print`` for live runs that have a
    per-simulation logger context, ``print`` with a name prefix for eval-only).
    """
    emit(f"Evaluating the conversation against {len(evaluators)} evaluator(s).")
    llm_judge_result = await evaluate_simuation(
        transcript, evaluators, fallback_model=fallback_judge_model
    )
    evaluation_results = [
        _build_evaluation_result(ev, llm_judge_result[ev["name"]])
        for ev in evaluators
    ]
    for row in evaluation_results:
        for line in format_evaluation_result_lines(row):
            emit(line)
    return evaluation_results


def _build_evaluation_result(evaluator: dict, judge_row: dict) -> dict:
    """Build a per-row evaluation result, carrying rating scale bounds when relevant.

    The scale bounds propagate through to ``metrics.json`` so the simulation
    leaderboard can normalize rating means correctly when computing the
    ``overall`` column.
    """
    result = {
        "name": evaluator["name"],
        "type": "rating" if is_rating(evaluator) else "binary",
        "value": evaluator_result_value(evaluator, judge_row),
        "reasoning": judge_row["reasoning"],
    }
    result = attach_evaluator_id(evaluator, result)
    if is_rating(evaluator):
        result["scale_min"] = int(evaluator["scale_min"])
        result["scale_max"] = int(evaluator["scale_max"])
    return result
import pandas as pd
import numpy as np

from pipecat.utils.tracing.setup import setup_tracing
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from calibrate.utils import patch_langfuse_trace

IS_TRACING_ENABLED = bool(os.getenv("ENABLE_TRACING"))

# Initialize tracing if enabled
if IS_TRACING_ENABLED:
    patch_langfuse_trace(trace_name="text_simulation")

    exporter = OTLPSpanExporter()

    setup_tracing(
        service_name="text_simulation",
        exporter=exporter,
    )
    logger.info("OpenTelemetry tracing initialized")


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
        context: Optional["LLMContext"] = None,
        output_dir: Optional[str] = None,
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
        self._context = context
        self._output_dir = output_dir

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

    def _save_intermediate_transcript(self):
        """Save intermediate transcript to file."""
        if not self._context or not self._output_dir:
            return

        transcript = [
            message
            for message in self._context._messages
            if message["role"] not in ["system"]
        ]

        transcript_path = join(self._output_dir, "transcript.json")
        with open(transcript_path, "w") as f:
            json.dump(transcript, f, indent=4)

    async def _handle_completed_response(self, response: str):
        should_continue = True

        # Log the LLM message with role color
        color = (
            "\033[94m" if self._role == "agent" else "\033[93m"
        )  # Blue for bot, Yellow for user
        log_and_print(f"{color}[{self._role.capitalize()}]: {response}\033[0m")

        if self._conversation_state and self._role == "agent":
            should_continue = await self._conversation_state.record_turn()

        self._save_intermediate_transcript()

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
    evaluators: list[dict],
    bot_model: str = "gpt-4.1",
    user_model: str = "gpt-4.1",
    bot_provider: str = "openai",
    user_provider: str = "openai",
    agent_speaks_first: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
    output_dir: Optional[str] = None,
    fallback_judge_model: str = DEFAULT_SIMULATION_JUDGE_MODEL,
) -> List[str]:
    """Runs a text-only bot that processes text inputs through an LLM and returns text outputs."""
    require_simulation_evaluators(evaluators)

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

    # Create context with system prompt (before processors so we can pass it)
    messages = [{"role": "system", "content": bot_system_prompt}]
    # Note: bot_context will be created after tools are set up, but we need a placeholder
    # We'll set the context on the processor later

    # Create processors (text_output needs to be created first for reference)
    bot_processor = Processor(
        role="agent",
        speaks_first=agent_speaks_first,
        conversation_state=conversation_state,
        name="BotProcessor",
        output_dir=output_dir,
    )
    user_processor = Processor(
        role="user",
        speaks_first=not agent_speaks_first,
        conversation_state=conversation_state,
        name="UserProcessor",
    )

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

    # Build tool schemas using common utility
    tool_schemas, webhook_configs = build_tools_schema(tools)
    standard_tools = [end_call_tool] + tool_schemas

    def create_webhook_function_call(webhook_config: dict):
        async def webhook_function_call(params: FunctionCallParams):
            log_and_print(
                f"tool call: {params.function_name} (webhook) invoked with arguments: {params.arguments}"
            )

            result = await make_webhook_call(webhook_config, params.arguments or {})

            await params.result_callback(result)
            return

        return webhook_function_call

    # Register function handlers
    bot_llm.register_function("end_call", end_call)
    for tool_schema in tool_schemas:
        if tool_schema.name in webhook_configs:
            bot_llm.register_function(
                tool_schema.name,
                create_webhook_function_call(webhook_configs[tool_schema.name]),
            )
        else:
            bot_llm.register_function(tool_schema.name, generic_function_call)

    tools = ToolsSchema(standard_tools=standard_tools)

    bot_context = LLMContext(messages, tools=tools)
    bot_context_aggregator = LLMContextAggregatorPair(bot_context)

    # Set context on bot_processor for intermediate transcript saving
    bot_processor._context = bot_context

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
        enable_tracing=IS_TRACING_ENABLED,
        conversation_id=str(uuid.uuid4()) if IS_TRACING_ENABLED else None,
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

    # Capture ERROR-level logs to surface pipecat internal errors
    # and cancel pipeline tasks immediately on critical errors
    captured_errors: list[str] = []
    _error_triggered = False

    def error_capture_sink(message):
        nonlocal _error_triggered
        record = message.record
        if record["level"].name in ("ERROR", "CRITICAL"):
            captured_errors.append(record["message"])

            if not _error_triggered:
                _error_triggered = True
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(lambda: asyncio.ensure_future(task.cancel()))
                loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(user_task.cancel())
                )

    error_sink_id = logger.add(error_capture_sink, level="ERROR")

    try:
        await asyncio.gather(
            runner.run(task),
            user_runner.run(user_task),
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise e
    finally:
        logger.remove(error_sink_id)

    # Check if errors were captured during the run
    if captured_errors:
        raise RuntimeError(
            f"Simulation failed with pipeline errors: {'; '.join(captured_errors)}"
        )

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

    evaluation_results = await _judge_and_emit(
        transcript, evaluators, fallback_judge_model, emit=log_and_print
    )

    return {
        "transcript": transcript,
        "evaluation_results": evaluation_results,
    }


def _save_transcript(output_dir: str | None, transcript: list[dict]) -> None:
    if not output_dir:
        return

    transcript_path = join(output_dir, "transcript.json")
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, indent=4)


async def run_simulation_with_agent(
    agent: "TextAgentConnection",
    user_system_prompt: str,
    evaluators: list[dict],
    agent_speaks_first: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
    user_model: str = "gpt-4.1",
    user_provider: str = "openai",
    output_dir: Optional[str] = None,
    fallback_judge_model: str = DEFAULT_SIMULATION_JUDGE_MODEL,
) -> dict:
    """Run a text simulation where the agent is an external HTTP endpoint.

    The user side is an internal LLM acting as a persona/scenario-driven user.
    The agent side is called via HTTP POST on each turn using
    :meth:`~calibrate.connections.TextAgentConnection.call`.

    Args:
        agent: External agent connection.
        user_system_prompt: System prompt for the simulated user.
        evaluators: List of evaluator dicts with ``name`` and ``system_prompt``.
        agent_speaks_first: Whether the agent sends the first message. Default: True.
        max_turns: Maximum agent turns before ending. Default: 50.
        user_model: Model for the user simulator. Default: ``gpt-4.1``.
        user_provider: Provider for the user simulator. Default: ``openai``.
        output_dir: Optional directory to save intermediate transcript.

    Returns:
        dict with ``transcript`` and ``evaluation_results`` keys.
    """
    require_simulation_evaluators(evaluators)

    from openai import AsyncOpenAI as _AsyncOpenAI

    # User LLM client
    if user_provider == "openrouter":
        user_client = _AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        user_client = _AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Messages the user simulator tracks (role-flipped: agent→user, user→assistant)
    user_messages = [{"role": "system", "content": user_system_prompt}]
    # Messages sent to the external agent (standard role: user→user, agent→assistant)
    agent_messages: list[dict] = []
    # Full transcript for evaluation
    transcript: list[dict] = []
    max_turns_reached = False

    _MAX_TOOL_CALL_RETRIES = 3

    async def _get_agent_response(messages: list) -> Optional[str]:
        """Call agent, retrying up to 3 times if it responds with tool calls but no text.

        Each tool-call-only response is recorded in messages before retrying so the
        agent sees its own prior tool calls. Returns None if no text response after
        3 attempts, logging a warning.
        """
        for attempt in range(_MAX_TOOL_CALL_RETRIES):
            output = await agent.call(messages)
            if output.get("response"):
                return output["response"]
            tool_calls = output.get("tool_calls") or []
            if tool_calls:
                messages.append(
                    {"role": "assistant", "content": json.dumps(tool_calls)}
                )
        log_and_print(
            f"\033[91m[Warning]: Agent made tool calls but returned no text response "
            f"after {_MAX_TOOL_CALL_RETRIES} attempts. Ending simulation.\033[0m"
        )
        return None

    if agent_speaks_first:
        agent_messages.append({"role": "user", "content": "Hi"})
        agent_text = await _get_agent_response(agent_messages)
        if agent_text is None:
            return {"transcript": transcript, "evaluation_results": []}
        agent_messages.append({"role": "assistant", "content": agent_text})
        transcript.append({"role": "assistant", "content": agent_text})
        log_and_print(f"\033[94m[Agent]: {agent_text}\033[0m")
        # Prepare user simulator: agent's first message becomes user's input
        user_messages.append({"role": "user", "content": agent_text})

    for turn in range(max_turns):
        # --- User simulator turn ---
        user_resp = await user_client.chat.completions.create(
            model=user_model,
            messages=user_messages,
        )
        user_message = user_resp.choices[0].message.content.strip()
        user_messages.append({"role": "assistant", "content": user_message})
        transcript.append({"role": "user", "content": user_message})
        log_and_print(f"\033[93m[User]: {user_message}\033[0m")
        _save_transcript(output_dir, transcript)

        # --- External agent turn ---
        agent_messages.append({"role": "user", "content": user_message})
        agent_text = await _get_agent_response(agent_messages)
        if agent_text is None:
            break
        agent_messages.append({"role": "assistant", "content": agent_text})
        transcript.append({"role": "assistant", "content": agent_text})
        log_and_print(f"\033[94m[Agent]: {agent_text}\033[0m")
        _save_transcript(output_dir, transcript)

        # Prepare user simulator for next turn
        user_messages.append({"role": "user", "content": agent_text})

        if turn + 1 >= max_turns:
            max_turns_reached = True

    if max_turns_reached:
        transcript.append({"role": "end_reason", "content": "max_turns"})

    _save_transcript(output_dir, transcript)

    evaluation_results = await _judge_and_emit(
        transcript, evaluators, fallback_judge_model, emit=log_and_print
    )

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
    agent: Optional["TextAgentConnection"] = None,
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

        # Generate a unique ID for this simulation run to avoid conflicts
        # when multiple simulations with the same name run in parallel
        simulation_run_id = str(uuid.uuid4())

        # Create a unique loguru sink for this simulation with a strict filter
        # that only accepts logs from this simulation's context
        def simulation_filter(record):
            sim_id = record["extra"].get("simulation")
            return sim_id == simulation_run_id

        log_file_id = logger.add(
            logs_file_path,
            level="DEBUG",
            colorize=False,
            filter=simulation_filter,
        )

        # Route judge LLM input/output into this simulation's logs file.
        judge_log_token = provider_log_file.set(logs_file_path)

        agent_speaks_first = config.get("settings", {}).get("agent_speaks_first", True)

        # Configure print logger with unique ID for parallel execution
        print_log_save_path = f"{output_dir}/{simulation_name}/results.log"
        configure_print_logger(print_log_save_path, simulation_name=simulation_run_id)
        current_simulation_name.set(simulation_run_id)

        # Use contextualize to bind simulation ID to ALL log calls within this context
        # This includes logs from libraries like pipecat
        with logger.contextualize(simulation=simulation_run_id):
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

            evaluators = config.get("evaluators") or []

            try:
                if agent is not None:
                    output = await run_simulation_with_agent(
                        agent=agent,
                        user_system_prompt=user_system_prompt,
                        evaluators=evaluators,
                        agent_speaks_first=agent_speaks_first,
                        max_turns=config.get("settings", {}).get(
                            "max_turns", DEFAULT_MAX_TURNS
                        ),
                        user_model="gpt-4.1",
                        user_provider="openai",
                        output_dir=simulation_output_dir,
                    )
                else:
                    output = await run_simulation(
                        bot_system_prompt=config["system_prompt"]
                        + f"\n\nYou must always speak in {language}.",
                        tools=config["tools"],
                        user_system_prompt=user_system_prompt,
                        evaluators=evaluators,
                        bot_model=args.model,
                        user_model="gpt-4.1",
                        bot_provider=args.provider,
                        user_provider="openai",
                        agent_speaks_first=agent_speaks_first,
                        max_turns=config.get("settings", {}).get(
                            "max_turns", DEFAULT_MAX_TURNS
                        ),
                        output_dir=simulation_output_dir,
                    )

                simulation_metrics = {
                    "name": simulation_name,
                }

                for metric_dict in output["evaluation_results"]:
                    simulation_metrics[metric_dict["name"]] = float(
                        metric_dict["value"]
                    )

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
                raise e
            finally:
                try:
                    logger.remove(log_file_id)
                except ValueError:
                    pass  # Handler was already removed by another task
                provider_log_file.reset(judge_log_token)
                # Clean up the print logger for this simulation using the unique run ID
                cleanup_print_logger(simulation_run_id)


async def main():
    # Remove default loguru handler (stderr) to prevent all logs from showing on terminal
    # This is done once at startup, not per-simulation
    logger.remove()

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
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip simulation and run evaluators on a dataset of pre-existing transcripts",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON for --eval-only (list of {conversation_history, name?})",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    try:
        require_simulation_evaluators(config.get("evaluators"))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir

    if args.eval_only:
        if not args.dataset:
            print("Error: --dataset is required with --eval-only", file=sys.stderr)
            sys.exit(1)
        try:
            with open(args.dataset) as _f:
                dataset = json.load(_f)
        except Exception as e:
            print(
                f"Error: failed to read dataset {args.dataset}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        is_valid, err = validate_simulation_eval_only_dataset(dataset)
        if not is_valid:
            print(f"Dataset validation error: {err}", file=sys.stderr)
            sys.exit(1)

        print("\n\033[91mText Simulation Eval-Only\033[0m\n")
        print(f"Config: {args.config}")
        print(f"Dataset: {args.dataset}")
        print(f"Output: {output_dir}\n")

        failed_count = await run_eval_only_simulations(
            config=config,
            dataset=dataset,
            output_dir=output_dir,
            parallel=args.parallel,
        )
        if failed_count:
            print(f"\n\033[31m{failed_count} simulation(s) failed\033[0m")
            sys.exit(1)
        return

    os.makedirs(output_dir, exist_ok=True)
    write_evaluator_config(output_dir, config["evaluators"])

    # Create semaphore to limit parallel executions
    semaphore = asyncio.Semaphore(args.parallel)

    # Detect agent connection path
    agent = None
    if config.get("agent_url"):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(
            url=config["agent_url"],
            headers=config.get("agent_headers"),
        )

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
                agent=agent,
            )
            tasks.append(task)

    # Run all tasks with controlled parallelism
    results = await asyncio.gather(*tasks, return_exceptions=True)

    failed = _aggregate_and_write_simulation_results(results, output_dir)

    if failed:
        print(f"\n\033[31m{len(failed)} simulation(s) failed:\033[0m")
        for err in failed:
            print(f"  \033[31m- {err}\033[0m")
        sys.exit(1)


def _aggregate_and_write_simulation_results(
    results: list, output_dir: str
) -> list:
    """Aggregate per-simulation results into ``results.csv`` and ``metrics.json``.

    Each non-Exception result is a ``(simulation_metrics, evaluation_results)``
    tuple. Returns the list of Exception entries (failed simulations) so the
    caller can surface them.
    """
    metrics = defaultdict(list)
    all_simulation_metrics = []
    failed_simulations = []
    criterion_types: dict = {}
    criterion_ids: dict = {}
    criterion_scales: dict = {}

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Simulation failed with error: {result}")
            failed_simulations.append(result)
            continue

        simulation_metrics, evaluation_results = result
        all_simulation_metrics.append(simulation_metrics)

        for metric_dict in evaluation_results:
            metrics[metric_dict["name"]].append(float(metric_dict["value"]))
            criterion_types.setdefault(
                metric_dict["name"], metric_dict.get("type", "binary")
            )
            if "evaluator_id" in metric_dict:
                criterion_ids.setdefault(
                    metric_dict["name"], metric_dict["evaluator_id"]
                )
            if "scale_min" in metric_dict and "scale_max" in metric_dict:
                criterion_scales.setdefault(
                    metric_dict["name"],
                    (int(metric_dict["scale_min"]), int(metric_dict["scale_max"])),
                )

    metrics_summary = {}
    for metric_name, metric_values in metrics.items():
        entry = {
            "type": criterion_types.get(metric_name, "binary"),
            "mean": np.mean(metric_values),
            "std": np.std(metric_values),
            "values": metric_values,
        }
        if metric_name in criterion_scales:
            entry["scale_min"], entry["scale_max"] = criterion_scales[metric_name]
        if metric_name in criterion_ids:
            entry["evaluator_id"] = criterion_ids[metric_name]
        metrics_summary[metric_name] = entry

    df = pd.DataFrame(all_simulation_metrics)
    df.to_csv(join(output_dir, "results.csv"), index=False)

    with open(join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    return failed_simulations


def validate_simulation_eval_only_dataset(dataset: object) -> tuple[bool, str]:
    """Validate the shape of a text-simulation eval-only dataset.

    Each item must be ``{"conversation_history": list, "name"?: str}``.
    Returns ``(is_valid, error_message)``.
    """
    if not isinstance(dataset, list):
        return False, "Dataset must be a JSON list of {conversation_history, name?} items"

    for i, item in enumerate(dataset):
        if not isinstance(item, dict):
            return False, f"Item {i}: must be an object"
        if "conversation_history" not in item:
            return False, f"Item {i}: missing required field 'conversation_history'"
        if not isinstance(item["conversation_history"], list):
            return False, f"Item {i}: 'conversation_history' must be a list"
        if "name" in item and not isinstance(item["name"], str):
            return False, f"Item {i}: 'name' must be a string when provided"

    return True, ""


async def run_eval_only_simulation_task(
    semaphore: asyncio.Semaphore,
    item: dict,
    item_index: int,
    evaluators: list[dict],
    output_dir: str,
    fallback_judge_model: str = DEFAULT_SIMULATION_JUDGE_MODEL,
):
    """Evaluate a single pre-existing transcript and write per-simulation files.

    ``item`` schema: ``{"conversation_history": [...], "name": str?}``. Only
    ``conversation_history`` is required — it's the sole input to the
    evaluators. ``name`` is optional metadata preserved into the aggregated
    outputs and ``dataset_map.json`` for traceability.

    The per-simulation output subdirectory is derived from a stable internal
    ``row_id`` (``row_<1-based-index>``), not from the user-provided ``name``,
    so duplicate or empty names cannot collide on disk.
    """
    async with semaphore:
        transcript = item["conversation_history"]
        row_id = f"row_{item_index + 1}"
        display_name = item.get("name") or row_id

        simulation_output_dir = join(output_dir, row_id)
        if exists(simulation_output_dir):
            shutil.rmtree(simulation_output_dir)
        os.makedirs(simulation_output_dir)

        evaluation_results = await _judge_and_emit(
            transcript,
            evaluators,
            fallback_judge_model,
            emit=lambda line: print(f"[{display_name}] {line}"),
        )

        with open(join(simulation_output_dir, "transcript.json"), "w") as f:
            json.dump(transcript, f, indent=4)

        pd.DataFrame(evaluation_results).to_csv(
            join(simulation_output_dir, "evaluation_results.csv"), index=False
        )

        simulation_metrics = {"row_id": row_id, "name": display_name}
        for metric_dict in evaluation_results:
            simulation_metrics[metric_dict["name"]] = float(metric_dict["value"])

        return simulation_metrics, evaluation_results


async def run_eval_only_simulations(
    config: dict,
    dataset: list[dict],
    output_dir: str,
    parallel: int = 1,
) -> int:
    """Run evaluators on a pre-existing dataset of simulation transcripts.

    Returns the number of failed simulations.
    """
    evaluators = config.get("evaluators") or []
    require_simulation_evaluators(evaluators)

    os.makedirs(output_dir, exist_ok=True)
    write_evaluator_config(output_dir, evaluators)

    # Map internal row_id ↔ original dataset row, so the caller can correlate
    # per-simulation subdirectories back to whatever name/index they passed in.
    dataset_map = {
        f"row_{i + 1}": {
            "index": i,
            "name": item.get("name"),
        }
        for i, item in enumerate(dataset)
    }
    with open(join(output_dir, "dataset_map.json"), "w") as f:
        json.dump(dataset_map, f, indent=4)

    semaphore = asyncio.Semaphore(parallel)
    tasks = [
        run_eval_only_simulation_task(
            semaphore=semaphore,
            item=item,
            item_index=i,
            evaluators=evaluators,
            output_dir=output_dir,
        )
        for i, item in enumerate(dataset)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    failed = _aggregate_and_write_simulation_results(results, output_dir)
    return len(failed)


if __name__ == "__main__":
    asyncio.run(main())
