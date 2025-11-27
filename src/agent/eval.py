import asyncio
import io
import json
import os
from os.path import join, exists
import shutil
import time
import struct
import wave
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Literal
from uuid import uuid4
import shutil
import aiofiles
from deepgram import LiveOptions
from loguru import logger
from PIL.ImageFile import ImageFile
from dataclasses import dataclass
from collections import defaultdict

# Context variable to track current execution context (BOT or EVAL)
current_context: ContextVar[str] = ContextVar("current_context", default="UNKNOWN")

USER_MESSAGE_COLOR = "\033[94m"
AGENT_MESSAGE_COLOR = "\033[92m"
GENERAL_LOG_COLOR = "\033[93m"
RESET_COLOR = "\033[0m"


# Configure logger format to support source prefix
def add_default_source(record):
    """Add default source if not present in extra"""
    if "source" not in record["extra"]:
        context = current_context.get()
        record["extra"]["source"] = f"{context}-SYS"
    return True


# Create a contextual logger with EVAL prefix
eval_logger = logger.bind(source="EVAL")

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.transcriptions.language import Language

from pipecat.frames.frames import (
    EndFrame,
    StopFrame,
    CancelFrame,
    EndTaskFrame,
    InterimTranscriptionFrame,
    LLMRunFrame,
    TranscriptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    InputAudioRawFrame,
    LLMMessagesAppendFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
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
from integrations.smallest.tts import SmallestTTSService
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
from bot import run_bot, STTConfig, TTSConfig, LLMConfig
from dotenv import load_dotenv
from pipecat.utils.time import time_now_iso8601
import logging

load_dotenv(override=True)


print_logger: Optional[logging.Logger] = None


def configure_print_logger(log_path: str):
    """Configure a dedicated logger for console print mirroring."""
    global print_logger
    print_logger = logging.getLogger("agent_eval_print_logger")
    print_logger.setLevel(logging.INFO)
    print_logger.propagate = False

    for handler in list(print_logger.handlers):
        print_logger.removeHandler(handler)

    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    print_logger.addHandler(handler)


def log_and_print(message: object = ""):
    text = str(message)
    print(text)
    eval_logger.info(text)
    if print_logger:
        print_logger.info(text)


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
    runner_args.pipeline_idle_timeout_secs = PIPELINE_IDLE_TIMEOUT_SECS

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


async def save_audio_chunk(
    path: str, audio_chunk: bytes, sample_rate: int, num_channels: int
):
    if len(audio_chunk) == 0:
        eval_logger.warning(f"There's no audio to save for {path}")
        return

    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists():
        log_and_print(
            f"{GENERAL_LOG_COLOR}Creating new audio file at {filepath}{RESET_COLOR}"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_chunk)
            async with aiofiles.open(filepath, "wb") as file:
                await file.write(buffer.getvalue())
    else:
        log_and_print(
            f"{GENERAL_LOG_COLOR}Appending audio chunk to {filepath}{RESET_COLOR}"
        )
        async with aiofiles.open(filepath, "rb+") as file:
            current_size = await file.seek(0, os.SEEK_END)
            if current_size < 44:
                eval_logger.error(
                    f"Existing audio file {filepath} is too small to be a valid WAV; rewriting"
                )
                await file.seek(0)
                await file.truncate(0)
                with io.BytesIO() as buffer:
                    with wave.open(buffer, "wb") as wf:
                        wf.setsampwidth(2)
                        wf.setnchannels(num_channels)
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio_chunk)
                    await file.write(buffer.getvalue())
                return

            await file.write(audio_chunk)
            new_size = current_size + len(audio_chunk)
            data_chunk_size = max(0, new_size - 44)

            await file.seek(40)
            await file.write(struct.pack("<I", data_chunk_size))

            await file.seek(4)
            await file.write(struct.pack("<I", new_size - 8))

            await file.flush()


async def run_simulation_pipeline(
    system_prompt: str,
    language: Literal[
        "english",
        "hindi",
    ],
    output_dir: str,
    user_speaks_first: bool = False,
):
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
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
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
        simulation_system_prompt = (
            f"{system_prompt}.\n\nBegin the conversation by saying 'Hi' to the agent."
        )

    messages = [
        {
            "role": "system",
            "content": simulation_system_prompt,
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    audio_buffer = AudioBufferProcessor(enable_turn_audio=True)

    class RTVIFunctionCallResponder(FrameProcessor):
        def __init__(self, tool_calls: list[dict]):
            super().__init__(enable_direct_mode=True, name="RTVIFunctionCallResponder")
            self._send_frame = None
            self._end_call_callback = None
            self._tool_calls = tool_calls

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
                        self._tool_calls.append(
                            message.get("data"),
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

    class RTVIMessageFrameAdapter(FrameProcessor):
        def __init__(
            self,
            context: LLMContext,
            tool_calls: list[dict],
            stt_outputs: list[str],
            ttft: defaultdict,
            processing_time: defaultdict,
            output_dir: str,
        ):
            super().__init__(enable_direct_mode=True, name="RTVIMessageFrameAdapter")
            self._context = context
            self._tool_calls = tool_calls
            self._output_dir = Path(output_dir)
            self._turn_index = 0
            self._stt_outputs = stt_outputs
            self._ttft = ttft
            self._processing_time = processing_time
            self._text_buffer = ""

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            if isinstance(frame, InputTransportMessageFrame):
                message = getattr(frame, "message", {}) or {}
                if message.get("label") == "rtvi-ai":
                    msg_type = message.get("type")
                    data = message.get("data") or {}
                    generated_frames: list = []
                    timestamp = time_now_iso8601()
                    user_id = ""

                    if msg_type == "bot-started-speaking":
                        self._turn_index += 1
                        eval_logger.info(f"[rtvi] turn index: {self._turn_index}")
                        generated_frames.append(UserStartedSpeakingFrame())
                    elif msg_type == "bot-stopped-speaking":
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
                        self._text_buffer = ""
                    elif msg_type == "bot-transcription":
                        text = data.get("text") or ""
                        if text:
                            self._text_buffer += text

                            result_payload = data if data else None

                            generated_frames.append(
                                InterimTranscriptionFrame(
                                    text=self._text_buffer,
                                    user_id=user_id,
                                    timestamp=timestamp,
                                    result=result_payload,
                                )
                            )

                    for generated_frame in generated_frames:
                        await self.push_frame(generated_frame, direction)

            if isinstance(frame, EndFrame) or isinstance(frame, CancelFrame):
                try:
                    self._output_dir.mkdir(parents=True, exist_ok=True)
                    serialized_transcripts: list[dict] = []
                    for message in self._context.get_messages():
                        if not isinstance(message, dict):
                            continue
                        role = message.get("role")

                        if role in {"user", "assistant"}:
                            serialized_transcripts.append(
                                {
                                    "role": role,
                                    "content": message.get("content", ""),
                                }
                            )
                    with open(
                        os.path.join(self._output_dir, "transcripts.json"), "w"
                    ) as transcripts_file:
                        json.dump(serialized_transcripts, transcripts_file, indent=4)

                    with open(
                        os.path.join(self._output_dir, "tool_calls.json"), "w"
                    ) as tool_calls_file:
                        json.dump(self._tool_calls, tool_calls_file, indent=4)

                    with open(
                        os.path.join(self._output_dir, "stt_outputs.json"), "w"
                    ) as stt_outputs_file:
                        json.dump(self._stt_outputs, stt_outputs_file, indent=4)

                    ttft = dict(self._ttft)
                    processing_time = dict(self._processing_time)

                    with open(
                        os.path.join(self._output_dir, "metrics.json"), "w"
                    ) as metrics_file:
                        json.dump(
                            {"ttft": ttft, "processing_time": processing_time},
                            metrics_file,
                            indent=4,
                        )

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
                                f"{USER_MESSAGE_COLOR}[User (transcribed)]{RESET_COLOR}: {text}"
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

            if isinstance(frame, TextFrame) and hasattr(frame, "text"):
                log_and_print(
                    f"{USER_MESSAGE_COLOR}[User]\033[0m: {frame.text}{RESET_COLOR}"
                )

            await self.push_frame(frame, direction)

    tool_calls = []
    function_call_handler = RTVIFunctionCallResponder(tool_calls)

    rtvi_message_adapter = RTVIMessageFrameAdapter(
        context, tool_calls, stt_outputs, ttft, processing_time, output_dir
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
        idle_timeout_secs=PIPELINE_IDLE_TIMEOUT_SECS,
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
            role = (
                "Agent" if message.role == "user" else "User"
            )  # since the user for the simulation pipeline is the agent we are testing
            color = (
                AGENT_MESSAGE_COLOR if message.role == "user" else USER_MESSAGE_COLOR
            )
            log_and_print(
                f"{color}[{role}]{RESET_COLOR}: {message.content}{RESET_COLOR}"
            )

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


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

            log_and_print("--------------------------------")
            log_and_print(
                f"""Running simulation {GENERAL_LOG_COLOR}{simulation_name}{RESET_COLOR}"""
            )
            log_and_print(f"{GENERAL_LOG_COLOR}Persona:{RESET_COLOR}\n{user_persona}")
            log_and_print(f"{GENERAL_LOG_COLOR}Scenario:{RESET_COLOR}\n{scenario}")
            log_and_print("--------------------------------")

            try:
                tasks = [
                    asyncio.create_task(
                        start_bot(
                            config["agent_system_prompt"]
                            + f"\n\nYou must always speak in {config['language']}.",
                            config["tools"],
                            config["language"],
                        )
                    ),
                    asyncio.create_task(
                        run_simulation_pipeline(
                            user_system_prompt,
                            config["language"],
                            simulation_output_dir,
                            user_speaks_first=True,
                        )
                    ),
                ]
                _, pending = await asyncio.wait(tasks, timeout=EVAL_TIMEOUT_SECS)
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
            except Exception as e:
                eval_logger.error(f"ERROR: Unable to run: {e}")

            eval_logger.remove(log_file_id)


if __name__ == "__main__":
    asyncio.run(main())
