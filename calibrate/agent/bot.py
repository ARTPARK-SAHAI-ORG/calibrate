import os
from typing import Any, Dict, Literal

from loguru import logger
import asyncio

from pydantic import BaseModel
from calibrate.utils import (
    create_stt_service,
    create_tts_service,
    build_tools_schema,
    make_webhook_call,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    Frame,
    BotSpeakingFrame,
    UserSpeakingFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    TTSTextFrame,
    LLMRunFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    FunctionCallResultFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    FunctionCallResultProperties,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments

from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport

# from pipecat.transports.daily.transport import DailyParams
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
    RTVIClientMessageFrame,
)
from pipecat.utils.time import time_now_iso8601

from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.services.llm_service import FunctionCallParams
from pipecat.observers.loggers.user_bot_latency_log_observer import (
    UserBotLatencyLogObserver,
)

bot_logger = logger.bind(source="BOT")


class MetricsLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, TTFBMetricsData):
                    bot_logger.info(f"!!! MetricsFrame: {frame}, ttfb: {d.value}")
                elif isinstance(d, ProcessingMetricsData):
                    bot_logger.info(f"!!! MetricsFrame: {frame}, processing: {d.value}")
                elif isinstance(d, LLMUsageMetricsData):
                    tokens = d.value
                    bot_logger.info(
                        f"!!! MetricsFrame: {frame}, tokens: {tokens.prompt_tokens}, characters: {tokens.completion_tokens}"
                    )
                elif isinstance(d, TTSUsageMetricsData):
                    bot_logger.info(f"!!! MetricsFrame: {frame}, characters: {d.value}")
        await self.push_frame(frame, direction)


class STTConfig(BaseModel):
    provider: Literal[
        "deepgram", "google", "openai", "cartesia", "groq", "elevenlabs", "sarvam"
    ] = "deepgram"


class TTSConfig(BaseModel):
    provider: Literal[
        "cartesia", "google", "openai", "groq", "elevenlabs", "sarvam"
    ] = "google"
    instructions: str = None


class LLMConfig(BaseModel):
    provider: Literal["openai", "openrouter"] = "openrouter"
    model: str = "openai/gpt-4.1"


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    system_prompt: str,
    tools: list[dict],
    stt_config: STTConfig,
    tts_config: TTSConfig,
    llm_config: LLMConfig,
    language: Literal["english", "hindi", "kannada"],
    mode: Literal["run", "eval"] = "run",
    agent_speaks_first: bool = True,
):
    if language not in ["english", "hindi", "kannada"]:
        bot_logger.error(f"Invalid language: {language}")
        return

    bot_logger.info(f"Starting bot")

    # Create STT service using common utility
    stt = create_stt_service(stt_config.provider, language)

    # Create TTS service using common utility
    tts = create_tts_service(
        tts_config.provider,
        language,
        instructions=tts_config.instructions,
    )

    if llm_config.provider == "openai":
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"), model=llm_config.model
        )
    elif llm_config.provider == "openrouter":
        llm = OpenRouterLLMService(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=llm_config.model,
            base_url="https://openrouter.ai/api/v1",
        )

    ml = MetricsLogger()

    transcript = TranscriptProcessor()

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    async def _exec_call_call():
        try:
            await task.cancel()
        except Exception as exc:
            bot_logger.warning(
                f"Unable to cancel task after end_call (no tool_call_id): {exc}"
            )

    async def end_call(params: FunctionCallParams):
        reason = params.arguments.get("reason") if params.arguments else None
        if reason:
            bot_logger.info(f"end_call tool invoked by LLM: {reason}")
        else:
            bot_logger.info("end_call tool invoked by LLM")

        if mode == "run":
            await params.result_callback(
                None, properties=FunctionCallResultProperties(run_llm=False)
            )
            await _exec_call_call()
            return

        tool_call_id = params.tool_call_id
        pending_tool_calls[tool_call_id] = params

        # Create an event to wait for the result from the eval side
        result_event = asyncio.Event()
        pending_tool_call_events[tool_call_id] = result_event

        try:
            await rtvi.handle_function_call(params)

            # Wait for the result from the eval side (with timeout)
            try:
                await asyncio.wait_for(result_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                bot_logger.warning(
                    f"Timeout waiting for function call result for end_call:{tool_call_id}"
                )
                pending_tool_calls.pop(tool_call_id, None)
                pending_tool_call_events.pop(tool_call_id, None)
                return

            # Get the result and call the callback
            result = pending_tool_call_results.pop(tool_call_id, None)
            pending_tool_calls.pop(tool_call_id, None)
            pending_tool_call_events.pop(tool_call_id, None)

            await params.result_callback(
                result, properties=FunctionCallResultProperties(run_llm=False)
            )
            bot_logger.debug(
                f"Delivered function call result for end_call:{tool_call_id}"
            )

            await _exec_call_call()

        except Exception as exc:
            pending_tool_calls.pop(tool_call_id, None)
            pending_tool_call_events.pop(tool_call_id, None)
            pending_tool_call_results.pop(tool_call_id, None)
            bot_logger.warning(f"Unable to forward end_call to client: {exc}")

    async def generic_function_call(params: FunctionCallParams):
        bot_logger.info(
            f"{params.function_name} invoked with arguments: {params.arguments}"
        )

        if mode == "run":
            await params.result_callback(
                {"status": "received"},
            )
            return

        tool_call_id = params.tool_call_id
        pending_tool_calls[tool_call_id] = params

        # Create an event to wait for the result from the eval side
        result_event = asyncio.Event()
        pending_tool_call_events[tool_call_id] = result_event

        try:
            await rtvi.handle_function_call(params)

            # Wait for the result from the eval side (with timeout)
            try:
                await asyncio.wait_for(result_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                bot_logger.warning(
                    f"Timeout waiting for function call result for {params.function_name}:{tool_call_id}"
                )
                pending_tool_calls.pop(tool_call_id, None)
                pending_tool_call_events.pop(tool_call_id, None)
                return

            # Get the result and call the callback
            result = pending_tool_call_results.pop(tool_call_id, None)
            pending_tool_calls.pop(tool_call_id, None)
            pending_tool_call_events.pop(tool_call_id, None)

            properties = (
                FunctionCallResultProperties(run_llm=False)
                if params.function_name == "end_call"
                else None
            )

            await params.result_callback(result, properties=properties)
            bot_logger.debug(
                f"Delivered function call result for {params.function_name}:{tool_call_id}"
            )

            if params.function_name == "end_call":
                await _exec_call_call()

        except Exception as exc:
            pending_tool_calls.pop(tool_call_id, None)
            pending_tool_call_events.pop(tool_call_id, None)
            pending_tool_call_results.pop(tool_call_id, None)
            bot_logger.warning(
                f"Unable to forward {params.function_name} to client: {exc}"
            )

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
            bot_logger.info(
                f"{params.function_name} (webhook) invoked with arguments: {params.arguments}"
            )

            # In "run" mode, make the actual webhook call
            if mode == "run":
                result = await make_webhook_call(webhook_config, params.arguments or {})
                await params.result_callback(result)
                return

            # In eval mode, forward to client like generic function call
            tool_call_id = params.tool_call_id
            pending_tool_calls[tool_call_id] = params

            result_event = asyncio.Event()
            pending_tool_call_events[tool_call_id] = result_event

            try:
                await rtvi.handle_function_call(params)

                try:
                    await asyncio.wait_for(result_event.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    bot_logger.warning(
                        f"Timeout waiting for function call result for {params.function_name}:{tool_call_id}"
                    )
                    pending_tool_calls.pop(tool_call_id, None)
                    pending_tool_call_events.pop(tool_call_id, None)
                    return

                result = pending_tool_call_results.pop(tool_call_id, None)
                pending_tool_calls.pop(tool_call_id, None)
                pending_tool_call_events.pop(tool_call_id, None)

                await params.result_callback(result)
                bot_logger.debug(
                    f"Delivered function call result for {params.function_name}:{tool_call_id}"
                )

            except Exception as exc:
                pending_tool_calls.pop(tool_call_id, None)
                pending_tool_call_events.pop(tool_call_id, None)
                pending_tool_call_results.pop(tool_call_id, None)
                bot_logger.warning(
                    f"Unable to forward {params.function_name} to client: {exc}"
                )

        return webhook_function_call

    # Register function handlers
    llm.register_function("end_call", end_call)
    for tool_schema in tool_schemas:
        if tool_schema.name in webhook_configs:
            llm.register_function(
                tool_schema.name,
                create_webhook_function_call(webhook_configs[tool_schema.name]),
            )
        else:
            llm.register_function(tool_schema.name, generic_function_call)

    tools = ToolsSchema(standard_tools=standard_tools)

    context = LLMContext(messages, tools)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pending_tool_calls: Dict[str, FunctionCallParams] = {}
    pending_tool_call_events: Dict[str, asyncio.Event] = {}
    pending_tool_call_results: Dict[str, Any] = {}

    class InputLogger(FrameProcessor):
        def __init__(
            self,
        ):
            super().__init__()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"input bot frame: {frame}")

            if isinstance(frame, RTVIClientMessageFrame) and frame.type == "interrupt":
                logger.info(f"Simulating user interruption of the bot")
                await self.push_interruption_task_frame_and_wait()
                await self.push_frame(UserStartedSpeakingFrame())
                await self.push_frame(
                    TranscriptionFrame(
                        "",
                        "",
                        time_now_iso8601(),
                    )
                )
                # Need to wait before sending the UserStoppedSpeakingFrame,
                # otherwise TranscriptionFrame will be processed
                # later than the UserStoppedSpeakingFrame
                await asyncio.sleep(0.1)
                await self.push_frame(UserStoppedSpeakingFrame())

            await self.push_frame(frame, direction)

    class OutputLogger(FrameProcessor):
        def __init__(self):
            super().__init__()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"output bot frame: {frame}")

            await self.push_frame(frame, direction)

    class FunctionCallResultHandler(FrameProcessor):
        def __init__(self):
            super().__init__(enable_direct_mode=True, name="FunctionCallResultHandler")

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, FunctionCallResultFrame):
                tool_call_id = frame.tool_call_id

                # Check if we're waiting for this result
                event = pending_tool_call_events.get(tool_call_id)

                if event and not event.is_set():
                    # Store the result and signal the waiting coroutine
                    pending_tool_call_results[tool_call_id] = frame.result
                    event.set()
                    bot_logger.debug(
                        f"Received function call result for {frame.function_name}:{tool_call_id}, signaling waiting handler"
                    )
                elif tool_call_id not in pending_tool_calls:
                    # Only warn if we're not expecting this tool_call_id at all
                    bot_logger.debug(
                        f"Received function call result for already-processed or unknown tool_call_id {tool_call_id} ({frame.function_name})"
                    )

            await self.push_frame(frame, direction)

    pipeline_processors = [
        transport.input(),
        rtvi,
        InputLogger(),
    ]

    if mode == "eval":
        pipeline_processors.append(FunctionCallResultHandler())

    pipeline_processors.extend(
        [
            stt,
            transcript.user(),
            context_aggregator.user(),
            llm,
            tts,
            ml,
            OutputLogger(),
            transport.output(),
            transcript.assistant(),
            context_aggregator.assistant(),
        ]
    )

    pipeline = Pipeline(pipeline_processors)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
        ),
        observers=[
            RTVIObserver(rtvi),
            LLMLogObserver(),
            UserBotLatencyLogObserver(),
        ],  # RTVI protocol events
        idle_timeout_secs=120,  # 2 minutes idle timeout
        idle_timeout_frames=(
            UserSpeakingFrame,
            BotSpeakingFrame,  # Bot speaking
            TextFrame,  # LLM generating text
            TTSTextFrame,  # TTS processing
            LLMFullResponseStartFrame,  # LLM started responding
            LLMFullResponseEndFrame,  # LLM finished responding
            OutputAudioRawFrame,  # Audio being sent out
        ),
        cancel_on_idle_timeout=True,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        bot_logger.info(f"Client connected")
        # Kick off the conversation.
        if agent_speaks_first:
            messages.append(
                {
                    "role": "user",
                    "content": "Hi",
                }
            )

            await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        bot_logger.info(f"Client disconnected")
        await task.cancel()

    @transcript.event_handler("on_transcript_update")
    async def handle_transcript_update(processor, frame):
        # Each message contains role (user/assistant), content, and timestamp
        for message in frame.messages:
            bot_logger.info(
                f"Bot transcript:[{message.timestamp}] {message.role}: {message.content}"
            )

    # Handle client connection
    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Signal bot is ready to receive messages
        await rtvi.set_bot_ready()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)
