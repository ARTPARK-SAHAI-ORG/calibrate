import asyncio
import json
import os
import sys
from os.path import join, exists
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Literal
from uuid import uuid4
import shutil
from deepgram import LiveOptions
from loguru import logger
from PIL.ImageFile import ImageFile
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from agentloop.utils import (
    current_context,
    add_default_source,
    configure_print_logger,
    log_and_print,
    save_audio_chunk,
)
from agentloop.llm.metrics import evaluate_simuation
from agentloop.stt.metrics import get_llm_judge_score as stt_llm_judge_score
import pandas as pd

USER_MESSAGE_COLOR = "\033[94m"
PARTIAL_AGENT_MESSAGE_COLOR = "\033[95m"
PARTIAL_AGENT_MESSAGE_COLOR_IGNORED = "\033[36m"
AGENT_MESSAGE_COLOR = "\033[92m"
TOOL_CALL_COLOR = "\033[33m"  # Magenta, not used for any of the above or below
GENERAL_LOG_COLOR = "\033[93m"
RESET_COLOR = "\033[0m"
INTERRUPTION_COLOR = "\033[91m"

# Create a contextual logger with EVAL prefix
eval_logger = logger.bind(source="EVAL")

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.transcriptions.language import Language

from pipecat.frames.frames import (
    EndFrame,
    AggregatedTextFrame,
    StopFrame,
    CancelFrame,
    EndTaskFrame,
    InterimTranscriptionFrame,
    LLMRunFrame,
    TTSTextFrame,
    TranscriptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    LLMMessagesAppendFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    Frame,
    InterruptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.services.google.tts import GoogleTTSService

# from pipecat.processors.transcript_processor import TranscriptProcessor
# from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from pipecat.transports.websocket.client import (
    WebsocketClientParams,
    WebsocketClientTransport,
)

from pipecat.serializers.protobuf import ProtobufFrameSerializer
from agentloop.agent.bot import run_bot, STTConfig, TTSConfig, LLMConfig
from dotenv import load_dotenv
from pipecat.utils.time import time_now_iso8601

load_dotenv(override=True)


PIPELINE_IDLE_TIMEOUT_SECS = 60
EVAL_TIMEOUT_SECS = 3000


async def start_bot(
    system_prompt: str,
    tools: list[dict] = [],
    language: Literal["english", "hindi"] = "english",
    stt_config: STTConfig = STTConfig(),
    tts_config: TTSConfig = TTSConfig(),
    llm_config: LLMConfig = LLMConfig(),
):
    current_context.set("BOT")

    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
            session_timeout=60 * 3,  # 3 minutes
        )
    )

    runner_args = RunnerArguments()
    # runner_args.pipeline_idle_timeout_secs = PIPELINE_IDLE_TIMEOUT_SECS

    await run_bot(
        transport,
        runner_args,
        system_prompt=system_prompt,
        tools=tools,
        stt_config=stt_config,
        tts_config=tts_config,
        llm_config=llm_config,
        language=language,
        mode="eval",
    )


class RTVIMessageFrameAdapter(FrameProcessor):
    def __init__(
        self,
        context: LLMContext,
        audio_buffer: AudioBufferProcessor,
        interrupt_probability: float,
        tool_calls: list[dict],
        stt_outputs: list[str],
        ttft: defaultdict,
        processing_time: defaultdict,
        output_dir: str,
    ):
        super().__init__(enable_direct_mode=True, name="RTVIMessageFrameAdapter")
        self._context = context
        self._audio_buffer = audio_buffer
        self._interrupt_probability = interrupt_probability
        self._tool_calls = tool_calls
        self._output_dir = Path(output_dir)
        self._turn_index = 0
        self._stt_outputs = stt_outputs
        self._ttft = ttft
        self._processing_time = processing_time
        self._text_buffer = ""  # buffer of the text that the bot has generated so far
        self._spoken_text_buffer = (
            ""  # buffer of the text that the bot has spoken so far
        )
        self._is_bot_interrupt_decided = False  # whether the decision to interrupt the bot by the user has been made yet
        self._is_bot_interrupt_triggered = False  # whether the spoken text buffer is complete and matches the text buffer; only when this becomes true is when the intteruption actually triggered
        self._turns_concluded = set()
        self._serialized_transcript = []  # Store transcripts for return

    async def _reset_buffers(self):
        self._turns_concluded.add(self._turn_index)  # mark the turn as concluded
        self._text_buffer = ""
        self._spoken_text_buffer = ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputAudioRawFrame) and self._is_bot_interrupt_triggered:
            # don't forward bot audio frames after the interruption has been triggered
            return

        if isinstance(frame, InputTransportMessageFrame):
            message = getattr(frame, "message", {}) or {}
            if message.get("label") == "rtvi-ai":
                msg_type = message.get("type")
                data = message.get("data") or {}
                generated_frames: list = []
                timestamp = time_now_iso8601()
                user_id = ""

                if msg_type == "bot-started-speaking":
                    self._audio_buffer._reset_all_audio_buffers()
                    self._turn_index += 1
                    eval_logger.info(f"[rtvi] turn index: {self._turn_index}")
                    generated_frames.append(UserStartedSpeakingFrame())
                elif (
                    msg_type == "bot-stopped-speaking"
                    and self._turn_index not in self._turns_concluded
                ):
                    generated_frames.extend(
                        [
                            TranscriptionFrame(
                                text=self._text_buffer,
                                user_id=user_id,
                                timestamp=timestamp,
                                result={},
                            ),
                            UserStoppedSpeakingFrame(),
                        ]
                    )
                    await self._reset_buffers()
                elif msg_type == "user-stopped-speaking":
                    # once the simulated user starts speaking, mark the bot as not
                    # interrupted anymore and spoken text buffer as not complete anymore
                    self._is_bot_interrupt_decided = False
                    self._is_bot_interrupt_triggered = False
                elif msg_type == "bot-output":
                    text = data.get("text") or ""
                    spoken = data.get("spoken") or False

                    if text:
                        log_and_print(
                            f"{INTERRUPTION_COLOR}Agent message for debugging: {data}{RESET_COLOR}"
                        )
                        if (
                            (
                                not self._is_bot_interrupt_decided or spoken
                            )  # only continue if either the decision to interrupt the bot by the user has not been made yet or the message is being spoken by the bot and does not match the interrupted text yet
                            and not self._is_bot_interrupt_triggered  # bot has not been interrupted yet
                        ):
                            if spoken and self._is_bot_interrupt_decided:
                                log_and_print(
                                    f"{GENERAL_LOG_COLOR}Agent speaking the generated message before interruption: {text}{RESET_COLOR}"
                                )

                                # the text is being spoken by the bot and the decision to interrupt
                                # the bot by the user has been made
                                self._spoken_text_buffer += " " + text

                                # once the spoken text buffer matches the text buffer, mark the spoken
                                # text buffer as complete and interrupt the bot by the simulated user
                                if self._spoken_text_buffer == self._text_buffer:
                                    self._is_bot_interrupt_triggered = True

                                    await self.push_frame(
                                        OutputTransportMessageUrgentFrame(
                                            message={
                                                "label": "rtvi-ai",
                                                "type": "client-message",
                                                "id": str(uuid4()),
                                                "data": {
                                                    "t": "interrupt",
                                                },
                                            }
                                        ),
                                        FrameDirection.DOWNSTREAM,
                                    )

                                    generated_frames.extend(
                                        [
                                            TranscriptionFrame(
                                                text=self._text_buffer,
                                                user_id=user_id,
                                                timestamp=timestamp,
                                                result={},
                                            ),
                                            UserStoppedSpeakingFrame(),
                                        ]
                                    )

                                    await self._reset_buffers()
                            elif not spoken and not self._is_bot_interrupt_decided:
                                # the text has been spoken by the bot yet and the decision to
                                # interrupt the bot by the user has not been made yet
                                log_and_print(
                                    f"{PARTIAL_AGENT_MESSAGE_COLOR}Received agent message{RESET_COLOR}: {text}{RESET_COLOR}"
                                )
                                self._text_buffer += " " + text

                                result_payload = data if data else None

                                if (
                                    np.random.rand() < self._interrupt_probability
                                    and not self._is_bot_interrupt_decided
                                ):
                                    # decide to interrupt the bot by the simulated user based on whatever they have said so far
                                    # the actual interruption will happen when the bot finishes speaking the text recorded so far
                                    log_and_print(
                                        f"--------------------------------\n{INTERRUPTION_COLOR}[User interrupts the bot]{RESET_COLOR}\n--------------------------------"
                                    )

                                    # mark the text collected so far as the text after which the user will interrupt the bot
                                    # we still need to wait for the bot to finish speaking the text after which the user will interrupt the bot
                                    self._is_bot_interrupt_decided = True

                                # add the interim transcription frame for the text buffer as usual
                                generated_frames.append(
                                    InterimTranscriptionFrame(
                                        text=self._text_buffer,
                                        user_id=user_id,
                                        timestamp=timestamp,
                                        result=result_payload,
                                    )
                                )
                            elif spoken:
                                log_and_print(
                                    f"{GENERAL_LOG_COLOR}Agent just speaking the generated message: {text}{RESET_COLOR}"
                                )

                        else:
                            if not spoken:
                                log_and_print(
                                    f"{PARTIAL_AGENT_MESSAGE_COLOR_IGNORED}Received agent message (ignored){RESET_COLOR}: {text}{RESET_COLOR}"
                                )
                            else:
                                log_and_print(
                                    f"{GENERAL_LOG_COLOR}Agent speaking the generated message 2: {text}{RESET_COLOR}"
                                )

                for generated_frame in generated_frames:
                    await self.push_frame(generated_frame, direction)

        if isinstance(frame, EndFrame) or isinstance(frame, CancelFrame):
            try:
                self._output_dir.mkdir(parents=True, exist_ok=True)

                serialized_transcript: list[dict] = []

                tool_call_current_index = 0

                for index, message in enumerate(self._context.get_messages()):
                    if not isinstance(message, dict):
                        continue
                    role = message.get("role")

                    if (
                        tool_call_current_index < len(self._tool_calls)
                        and self._tool_calls[tool_call_current_index].get("position")
                        == index
                    ):
                        serialized_transcript.append(
                            {
                                "role": "tool_call",
                                "content": self._tool_calls[
                                    tool_call_current_index
                                ].get("data"),
                            }
                        )
                        tool_call_current_index += 1

                    # flip the role as the user for the transcript is the agent and vice versa
                    if role == "user":
                        role = "assistant"
                    elif role == "assistant":
                        role = "user"

                    serialized_transcript.append(
                        {
                            "role": role,
                            "content": message.get("content", ""),
                        }
                    )

                while tool_call_current_index < len(self._tool_calls):
                    serialized_transcript.append(
                        {
                            "role": "tool_call",
                            "content": self._tool_calls[tool_call_current_index].get(
                                "data"
                            ),
                        }
                    )
                    tool_call_current_index += 1

                serialized_transcript = [
                    message
                    for message in serialized_transcript
                    if message.get("role") in {"user", "assistant", "tool_call"}
                ]

                # Store transcripts for return
                self._serialized_transcript = serialized_transcript

                with open(
                    os.path.join(self._output_dir, "transcripts.json"), "w"
                ) as transcripts_file:
                    json.dump(serialized_transcript, transcripts_file, indent=4)

                with open(
                    os.path.join(self._output_dir, "tool_calls.json"), "w"
                ) as tool_calls_file:
                    json.dump(self._tool_calls, tool_calls_file, indent=4)

                with open(
                    os.path.join(self._output_dir, "stt_outputs.json"), "w"
                ) as stt_outputs_file:
                    json.dump(self._stt_outputs, stt_outputs_file, indent=4)

            except Exception as exc:
                eval_logger.error(
                    "Failed to persist RTVI transcripts",
                    extra={"error": str(exc)},
                )

        await self.push_frame(frame, direction)


class MetricsLogger(FrameProcessor):
    def __init__(
        self, ttft: defaultdict, processing_time: defaultdict, context: LLMContext
    ):
        super().__init__(enable_direct_mode=True, name="MetricsLogger")
        self._ttft = ttft
        self._processing_time = processing_time
        self._context = context

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if (
            isinstance(frame, InputTransportMessageFrame)
            and self._context.get_messages()
        ):
            message = getattr(frame, "message", {})
            if isinstance(message, dict) and message.get("label") == "rtvi-ai":
                if message.get("type") == "metrics" and message.get("data"):
                    if message.get("data").get("ttfb"):
                        for d in message.get("data").get("ttfb"):
                            if not d.get("value"):
                                continue
                            self._ttft[d.get("processor")].append(d.get("value"))
                    if message.get("data").get("processing"):
                        for d in message.get("data").get("processing"):
                            if not d.get("value"):
                                continue
                            self._processing_time[d.get("processor")].append(
                                d.get("value")
                            )

        await self.push_frame(frame, direction)


class STTLogger(FrameProcessor):
    def __init__(self, stt_outputs: list[str], rtvi_adapter):
        super().__init__(enable_direct_mode=True, name="STTLogger")
        self._stt_outputs = stt_outputs
        self._rtvi_adapter = rtvi_adapter
        self._stt_outputs.append("")
        self.last_turn_index = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputTransportMessageFrame):
            message = getattr(frame, "message", {}) or {}
            if message.get("label") == "rtvi-ai":
                msg_type = message.get("type")
                data = message.get("data") or {}

                if msg_type == "user-transcription":
                    if (text := data.get("text")) and data.get("final"):
                        log_and_print(
                            f"{USER_MESSAGE_COLOR}[User (as transcribed by the agent)]{RESET_COLOR}: {text}"
                        )
                        if self._rtvi_adapter._turn_index > self.last_turn_index:
                            self._stt_outputs.append(text)
                            self.last_turn_index = self._rtvi_adapter._turn_index
                        else:
                            self._stt_outputs[-1] += text

        await self.push_frame(frame, direction)


class IOLogger(FrameProcessor):
    def __init__(
        self,
    ):
        super().__init__()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSTextFrame) and hasattr(frame, "text"):
            log_and_print(
                f"{USER_MESSAGE_COLOR}[User]\033[0m: {frame.text}{RESET_COLOR}"
            )

        await self.push_frame(frame, direction)


class RTVIFunctionCallResponder(FrameProcessor):
    def __init__(self, tool_calls: list[dict], context: LLMContext):
        super().__init__(enable_direct_mode=True, name="RTVIFunctionCallResponder")
        self._send_frame = None
        self._end_call_callback = None
        self._tool_calls = tool_calls
        self._context = context

    def set_frame_sender(self, sender):
        self._send_frame = sender

    def set_end_call_callback(self, callback):
        self._end_call_callback = callback

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputTransportMessageFrame):
            message = getattr(frame, "message", {})
            if isinstance(message, dict) and message.get("label") == "rtvi-ai":
                if message.get("type") == "llm-function-call":
                    tool_call_name = message.get("data", {}).get("function_name")
                    arguments = message.get("data", {}).get("args") or {}

                    log_and_print(
                        f"{TOOL_CALL_COLOR}tool call: {tool_call_name} invoked with arguments: {arguments}{RESET_COLOR}"
                    )

                    self._tool_calls.append(
                        {
                            "position": len(self._context.get_messages()),
                            "data": message.get("data"),
                        }
                    )

                    data = message.get("data") or {}
                    function_name = data.get("function_name")
                    tool_call_id = data.get("tool_call_id")
                    arguments = data.get("args") or {}

                    if function_name and tool_call_id:
                        result, post_callback = await self._execute_function(
                            function_name, arguments
                        )
                        await self._send_result_message(
                            function_name, tool_call_id, arguments, result
                        )
                        if post_callback:
                            await post_callback()

        await self.push_frame(frame, direction)

    async def _execute_function(self, function_name, arguments):
        if function_name == "end_call":
            reason = arguments.get("reason")

            async def _post_callback():
                if self._end_call_callback:
                    await self._end_call_callback(reason)

            result = {"acknowledged": True}
            if reason:
                result["reason"] = reason
            return result, _post_callback

        if function_name == "plan_next_question":
            return {"status": "received"}, None

        return {"status": "unhandled"}, None

    async def _send_result_message(
        self, function_name, tool_call_id, arguments, result
    ):
        if not self._send_frame:
            eval_logger.warning(
                "Skipping function call result send; sender not configured",
                extra={"function_name": function_name},
            )
            return

        payload = {
            "label": "rtvi-ai",
            "type": "llm-function-call-result",
            "id": str(uuid4()),
            "data": {
                "function_name": function_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments,
                "result": result,
            },
        }

        frame = OutputTransportMessageUrgentFrame(message=payload)
        await self._send_frame(frame)


async def run_simulation(
    system_prompt: str,
    language: Literal[
        "english",
        "hindi",
    ],
    evaluation_criteria: list[dict],
    output_dir: str,
    # user_speaks_first: bool = False,
    interrupt_probability: float = 0.5,  # medium
) -> dict:
    user_speaks_first = True  # hardcoded for now

    # Set context for EVAL logs
    current_context.set("EVAL")

    eval_logger.info(f"Starting evaluation pipeline")

    stt_outputs = []
    ttft = defaultdict[Any, list](list)
    processing_time = defaultdict(list)
    audio_save_dir = os.path.join(output_dir, "audios")

    if os.path.exists(audio_save_dir):
        shutil.rmtree(audio_save_dir)

    os.makedirs(audio_save_dir, exist_ok=True)

    transport = WebsocketClientTransport(
        uri="ws://localhost:8765",
        params=WebsocketClientParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
            # vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            # turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    )
    session = transport._session
    connect_lock = asyncio.Lock()
    original_connect = session.connect

    async def locked_connect(*args, **kwargs):
        async with connect_lock:
            if session._websocket:
                return

            max_attempts = 10
            base_delay = 0.5
            last_error = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await original_connect(*args, **kwargs)
                except (OSError, ConnectionError) as exc:
                    last_error = exc
                    delay = min(base_delay * (2 ** (attempt - 1)), 5.0)
                    eval_logger.warning(
                        "WebSocket connect attempt failed; retrying",
                        extra={
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "error": str(exc),
                        },
                    )
                    await asyncio.sleep(delay)

            raise (
                last_error
                if last_error
                else RuntimeError("Unknown error while connecting to WebSocket")
            )

    session.connect = locked_connect

    # Workaround for race condition: manually initialize the audio queue before connection
    transport.input()._audio_in_queue = asyncio.Queue()

    tts_language = (
        Language.EN
        if language == "english"
        else Language.HI if language == "hindi" else Language.KN
    )

    voice_id = (
        "hi-IN-Chirp3-HD-Zephyr" if language == "hindi" else "en-US-Chirp3-HD-Zephyr"
    )
    tts = GoogleTTSService(
        voice_id=voice_id,
        params=GoogleTTSService.InputParams(language=tts_language),
        credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    transcript = TranscriptProcessor()

    if not user_speaks_first:
        simulation_system_prompt = system_prompt
    else:
        simulation_system_prompt = f"{system_prompt}.\n\nBegin the conversation by saying 'Hello' to the agent."

    messages = [
        {
            "role": "system",
            "content": simulation_system_prompt,
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    audio_buffer = AudioBufferProcessor(enable_turn_audio=True)

    tool_calls = []
    function_call_handler = RTVIFunctionCallResponder(tool_calls, context)

    rtvi_message_adapter = RTVIMessageFrameAdapter(
        context,
        audio_buffer,
        interrupt_probability,
        tool_calls,
        stt_outputs,
        ttft,
        processing_time,
        output_dir,
    )

    metrics_logger = MetricsLogger(ttft, processing_time, context)

    stt_logger = STTLogger(stt_outputs, rtvi_message_adapter)

    output_logger = IOLogger()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            function_call_handler,
            rtvi_message_adapter,
            metrics_logger,
            stt_logger,
            transcript.user(),
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            output_logger,
            transport.output(),  # Transport bot output
            # transcript.assistant(),
            context_aggregator.assistant(),  # Assistant spoken responses
            audio_buffer,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            allow_interruptions=True,
        ),
        observers=[LLMLogObserver()],
        cancel_on_idle_timeout=False,
    )

    function_call_handler.set_frame_sender(task.queue_frame)

    async def _handle_end_call_request(reason):
        if reason:
            eval_logger.info("Server requested end_call", extra={"reason": reason})
        else:
            eval_logger.info("Server requested end_call")

        await task.cancel()

    function_call_handler.set_end_call_callback(_handle_end_call_request)

    @audio_buffer.event_handler("on_user_turn_audio_data")
    async def on_user_turn_audio_data(buffer, audio, sample_rate, num_channels):
        eval_logger.info(f"Audio data received - bot")
        eval_logger.info(f"[bot] turn index: {rtvi_message_adapter._turn_index}")
        audio_save_path = os.path.join(
            audio_save_dir, f"{rtvi_message_adapter._turn_index}_bot.wav"
        )
        await save_audio_chunk(audio_save_path, audio, sample_rate, num_channels)

    @audio_buffer.event_handler("on_bot_turn_audio_data")
    async def on_bot_turn_audio_data(buffer, audio, sample_rate, num_channels):
        eval_logger.info(f"Audio data received - user")
        eval_logger.info(f"[user] turn index: {rtvi_message_adapter._turn_index}")
        audio_save_path = os.path.join(
            audio_save_dir, f"{rtvi_message_adapter._turn_index}_user.wav"
        )
        await save_audio_chunk(audio_save_path, audio, sample_rate, num_channels)

    @transport.event_handler("on_connected")
    async def on_connected(transport, client):
        eval_logger.info(f"WebSocket connected")
        await audio_buffer.start_recording()

        # Default behavior is for the bot to speak first
        # If the eval bot speaks first, we append the prompt to the messages
        if user_speaks_first:
            # Always kick off the eval agent
            await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_disconnected")
    async def on_disconnected(transport, client):
        eval_logger.info(f"WebSocket disconnected")
        await task.cancel()

    @transcript.event_handler("on_transcript_update")
    async def handle_transcript_update(processor, frame):
        # Each message contains role (user/assistant), content, and timestamp
        for message in frame.messages:
            eval_logger.info(
                f"Eval transcript: [{message.timestamp}] {message.role}: {message.content}"
            )

            # since the user for the simulation pipeline is the agent we are testing
            if message.role != "user":
                continue

            log_and_print(
                f"{AGENT_MESSAGE_COLOR}[Agent]{RESET_COLOR}: {message.content}{RESET_COLOR}"
            )

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)

    transcript = rtvi_message_adapter._serialized_transcript

    log_and_print(
        f"Evaluating the conversation based on the criteria: {evaluation_criteria}"
    )
    # Get evaluation results from LLM judge
    llm_judge_result = await evaluate_simuation(transcript, evaluation_criteria)

    evaluation_results = [
        {
            "name": criterion["name"],
            "match": llm_judge_result[criterion["name"]]["match"],
            "reasoning": llm_judge_result[criterion["name"]]["reasoning"],
        }
        for criterion in evaluation_criteria
    ]

    # Get user messages from transcript (these are what the agent heard/transcribed)
    user_messages_in_transcript = [
        msg["content"]
        for msg in transcript
        if msg.get("role") == "user" and msg.get("content")
    ]

    # Filter out empty STT outputs
    filtered_stt_outputs = [s for s in stt_outputs if s.strip()]

    # # Compare STT outputs with user messages using STT LLM judge
    stt_llm_judge_result = None
    if filtered_stt_outputs and user_messages_in_transcript:
        # Align lengths - take minimum length
        log_and_print(f"Evaluating the STT outputs with user messages")
        min_len = min(len(filtered_stt_outputs), len(user_messages_in_transcript))
        if min_len > 0:
            stt_llm_judge_result = await stt_llm_judge_score(
                references=user_messages_in_transcript[:min_len],
                predictions=filtered_stt_outputs[:min_len],
            )

    # Build comprehensive metrics
    ttft_dict = dict(ttft)
    processing_time_dict = dict(processing_time)

    metrics = {
        "ttft": ttft_dict,
        "processing_time": processing_time_dict,
        "evaluation_results": evaluation_results,
        "stt_llm_judge": stt_llm_judge_result,
    }

    # Save comprehensive metrics.json
    with open(os.path.join(output_dir, "metrics.json"), "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    # Return all data
    return {
        "transcript": transcript,
        "stt_outputs": filtered_stt_outputs,
        "tool_calls": tool_calls,
        "evaluation_results": evaluation_results,
        "metrics": metrics,
    }


async def main():
    import argparse

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

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    interrupt_labels = [
        "no",
        "low",
        "medium",
        "high",
    ]
    interrupt_probabilities = [
        0,
        0.25,
        0.5,
        0.8,
    ]

    # Aggregated metrics across all simulations
    all_simulation_metrics = []
    metrics_by_criterion = defaultdict(list)
    stt_llm_judge_scores = []

    for persona_index, user_persona in enumerate(config["personas"]):
        for scenario_index, scenario in enumerate(config["scenarios"]):

            user_system_prompt = f"You are a user speaking to an agent. This is your persona:\n\n{user_persona}\n\nThe following scenario will be played out: {scenario}. Make sure to respond to the agent to match the given scenario as per the given persona for you. You must generate values like numbers, proper nouns, etc. in such a way that they can compatible with text-to-speech generation systems (e.g. write a phone number as individual digits instead of a full string or an email address like firstname.lastname@example.com as 'firstname dot lastname at example dot com') without ever mentioning in the generation that you are doing so for the TTS system."

            for prob, label in zip(interrupt_probabilities, interrupt_labels):
                simulation_name = f"simulation_persona_{persona_index + 1}_scenario_{scenario_index + 1}_{label}_interrupt"

                simulation_output_dir = f"{args.output_dir}/{simulation_name}"

                if exists(simulation_output_dir):
                    shutil.rmtree(simulation_output_dir)

                os.makedirs(simulation_output_dir)

                simulation_config = {
                    "persona": user_persona,
                    "scenario": scenario,
                    "interrupt_probability": prob,
                    "interrupt_type": label,
                }

                with open(f"{simulation_output_dir}/config.json", "w") as f:
                    json.dump(simulation_config, f, indent=4)

                logs_file_path = f"{args.output_dir}/{simulation_name}/logs"

                logger.remove()
                eval_logger.remove()

                log_file_id = eval_logger.add(
                    logs_file_path,
                    level="DEBUG",
                    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | [{extra[source]}] | {message}",
                    filter=add_default_source,
                    colorize=False,
                )

                print_log_save_path = f"{args.output_dir}/{simulation_name}/results.log"
                configure_print_logger(print_log_save_path)

                command = " ".join(sys.argv)
                log_and_print(f"\033[33mRunning command\033[0m: {command}")

                log_and_print("--------------------------------")
                log_and_print(
                    f"""Running simulation {GENERAL_LOG_COLOR}{simulation_name}{RESET_COLOR}"""
                )
                log_and_print(
                    f"{GENERAL_LOG_COLOR}Persona:{RESET_COLOR}\n{user_persona}"
                )
                log_and_print(f"{GENERAL_LOG_COLOR}Scenario:{RESET_COLOR}\n{scenario}")
                log_and_print("--------------------------------")

                simulation_result = None
                try:
                    bot_task = asyncio.create_task(
                        start_bot(
                            config["agent_system_prompt"]
                            + f"\n\nYou must always speak in {config['language']}.",
                            config["tools"],
                            config["language"],
                        )
                    )
                    sim_task = asyncio.create_task(
                        run_simulation(
                            user_system_prompt,
                            config["language"],
                            config["evaluation_criteria"],
                            simulation_output_dir,
                            # user_speaks_first=True,
                            interrupt_probability=prob,
                        )
                    )
                    tasks = [bot_task, sim_task]
                    done, pending = await asyncio.wait(tasks, timeout=EVAL_TIMEOUT_SECS)
                    if pending:
                        eval_logger.error(
                            f"ERROR: Eval timeout expired, cancelling pending tasks..."
                        )
                        # Both pipeline idle timeouts should have worked and both tasks
                        # should have exited already, but if we got here something went
                        # wrong so we perform an abrupt asyncio task cancellation, which
                        # will not cleanup things nicely.
                        for task in pending:
                            task.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)

                    # Get result from simulation task
                    if sim_task in done and not sim_task.cancelled():
                        try:
                            simulation_result = sim_task.result()
                        except Exception as e:
                            eval_logger.error(
                                f"ERROR: Failed to get simulation result: {e}"
                            )
                except Exception as e:
                    eval_logger.error(f"ERROR: Unable to run: {e}")

                # Aggregate metrics from this simulation
                if simulation_result:
                    sim_metrics_row = {"name": simulation_name}

                    # Evaluation criteria metrics
                    for eval_result in simulation_result.get("evaluation_results", []):
                        criterion_name = eval_result["name"]
                        match_value = float(eval_result["match"])
                        metrics_by_criterion[criterion_name].append(match_value)
                        sim_metrics_row[criterion_name] = match_value

                    # STT LLM judge score
                    stt_judge = simulation_result.get("metrics", {}).get(
                        "stt_llm_judge"
                    )
                    if stt_judge and "score" in stt_judge:
                        stt_llm_judge_scores.append(stt_judge["score"])
                        sim_metrics_row["stt_llm_judge_score"] = stt_judge["score"]

                    all_simulation_metrics.append(sim_metrics_row)

                eval_logger.remove(log_file_id)

    # Compute and save aggregated metrics
    metrics_summary = {}

    # Aggregate evaluation criteria metrics
    for criterion_name, values in metrics_by_criterion.items():
        metrics_summary[criterion_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }

    # Aggregate STT LLM judge scores
    if stt_llm_judge_scores:
        metrics_summary["stt_llm_judge"] = {
            "mean": float(np.mean(stt_llm_judge_scores)),
            "std": float(np.std(stt_llm_judge_scores)),
            "values": stt_llm_judge_scores,
        }

    # Save overall results.csv
    if all_simulation_metrics:
        df = pd.DataFrame(all_simulation_metrics)
        df.to_csv(join(args.output_dir, "results.csv"), index=False)

    # Save overall metrics.json
    with open(join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
