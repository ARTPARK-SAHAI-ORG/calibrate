import asyncio
import gc
import json
import os
import sys
import socket
from os.path import join, exists
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Literal
from uuid import uuid4
from deepgram import LiveOptions
from loguru import logger
from PIL.ImageFile import ImageFile
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from calibrate.utils import (
    current_context,
    add_default_source,
    configure_print_logger,
    log_and_print,
    save_audio_chunk,
    combine_turn_audio_chunks,
    combine_audio_files,
)
from calibrate.llm.metrics import evaluate_simuation
from calibrate.stt.metrics import get_llm_judge_score as stt_llm_judge_score
import pandas as pd

USER_MESSAGE_COLOR = "\033[94m"
PARTIAL_AGENT_MESSAGE_COLOR = "\033[95m"
PARTIAL_AGENT_MESSAGE_COLOR_IGNORED = "\033[36m"
AGENT_MESSAGE_COLOR = "\033[92m"
TOOL_CALL_COLOR = "\033[33m"  # Magenta, not used for any of the above or below
GENERAL_LOG_COLOR = "\033[93m"
RESET_COLOR = "\033[0m"
INTERRUPTION_COLOR = "\033[91m"
DEFAULT_MAX_TURNS = 10
DEFAULT_PORT = 8765

# Create a contextual logger with EVAL prefix
eval_logger = logger.bind(source="EVAL")

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.transcriptions.language import Language

from pipecat.frames.frames import (
    EndFrame,
    BotSpeakingFrame,
    UserSpeakingFrame,
    EndTaskFrame,
    LLMContextFrame,
    StopFrame,
    CancelFrame,
    EndFrame,
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
    TTSStoppedFrame,
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
from calibrate.agent.bot import run_bot, STTConfig, TTSConfig, LLMConfig
from pipecat.utils.time import time_now_iso8601


PIPELINE_IDLE_TIMEOUT_SECS = 120  # 2 minutes
EVAL_TIMEOUT_SECS = 3000


def is_port_available(port: int) -> bool:
    """Check if a port is available by verifying no process is listening on it."""
    # First check if we can connect to the port (if yes, something is listening)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            result = s.connect_ex(("localhost", port))
            if result == 0:
                # Connection succeeded, port is in use
                return False
    except OSError:
        pass

    # Then verify we can bind to it (without SO_REUSEADDR for stricter check)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except OSError:
        return False


def find_available_port(
    start_port: int, max_attempts: int = 100, excluded_ports: Optional[set] = None
) -> Optional[int]:
    """Find an available port starting from start_port, checking up to max_attempts ports.

    Args:
        start_port: Port to start checking from
        max_attempts: Maximum number of ports to check
        excluded_ports: Set of ports to exclude from consideration
    """
    excluded_ports = excluded_ports or set()
    for i in range(max_attempts):
        port = start_port + i
        if port in excluded_ports:
            continue
        if is_port_available(port):
            return port
    return None


async def start_bot(
    system_prompt: str,
    tools: list[dict] = [],
    language: Literal["english", "hindi"] = "english",
    port: int = DEFAULT_PORT,
    stt_config: STTConfig = STTConfig(),
    tts_config: TTSConfig = TTSConfig(),
    llm_config: LLMConfig = LLMConfig(),
    agent_speaks_first: bool = True,
):
    current_context.set("BOT")

    transport = WebsocketServerTransport(
        port=port,
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
            session_timeout=60 * 3,  # 3 minutes
        ),
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
        agent_speaks_first=agent_speaks_first,
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
        audio_save_dir: str,
    ):
        super().__init__(enable_direct_mode=True, name="RTVIMessageFrameAdapter")
        self._context = context
        self._audio_buffer = audio_buffer
        self._interrupt_probability = interrupt_probability
        self._tool_calls = tool_calls
        self._output_dir = Path(output_dir)
        self._audio_save_dir = audio_save_dir
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
        self._ended_due_to_max_turns = False
        self._bot_audio_chunk_indices = {}  # Track chunk indices for bot audio per turn

    async def _reset_buffers(self):
        concluded_turn = self._turn_index
        self._turns_concluded.add(concluded_turn)  # mark the turn as concluded
        self._text_buffer = ""
        self._spoken_text_buffer = ""

        # Save intermediate state after each turn
        await self._save_intermediate_state(concluded_turn)

    def _build_serialized_transcript(
        self, end_reason: Optional[str] = None
    ) -> list[dict]:
        """Build serialized transcript from context messages and tool calls.

        Args:
            end_reason: Optional reason for ending the conversation (e.g., "max_turns")

        Returns:
            List of transcript entries with roles flipped (user becomes assistant and vice versa)
        """
        serialized_transcript: list[dict] = []

        # Group tool calls by position
        tool_calls_by_position = defaultdict(list)
        for tool_call in self._tool_calls:
            position = tool_call.get("position")
            data = tool_call.get("data", {})
            tool_calls_by_position[position].append(
                {
                    "id": data.get("tool_call_id"),
                    "function": {
                        "name": data.get("function_name"),
                        "arguments": json.dumps(data.get("args", {})),
                    },
                    "type": "function",
                }
            )

        for index, message in enumerate(self._context.get_messages()):
            if not isinstance(message, dict):
                continue
            role = message.get("role")

            # Add tool calls that occurred at this position
            if index in tool_calls_by_position:
                serialized_transcript.append(
                    {
                        "role": "assistant",
                        "tool_calls": tool_calls_by_position[index],
                    }
                )

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

        # Add any remaining tool calls that occurred after all messages
        max_message_index = len(self._context.get_messages())
        for position in sorted(tool_calls_by_position.keys()):
            if position >= max_message_index:
                serialized_transcript.append(
                    {
                        "role": "assistant",
                        "tool_calls": tool_calls_by_position[position],
                    }
                )

        serialized_transcript = [
            message
            for message in serialized_transcript
            if message.get("role") in {"user", "assistant"}
        ]

        if end_reason:
            serialized_transcript.append(
                {
                    "role": "end_reason",
                    "content": end_reason,
                }
            )

        return serialized_transcript

    def _save_transcript(self, transcript: list[dict]):
        """Save transcript to file.

        Args:
            transcript: The serialized transcript to save
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._serialized_transcript = transcript

        with open(
            os.path.join(self._output_dir, "transcript.json"), "w"
        ) as transcripts_file:
            json.dump(transcript, transcripts_file, indent=4)

    async def _save_intermediate_state(self, concluded_turn: int):
        """Save intermediate transcript after the concluded turn."""
        try:
            transcript = self._build_serialized_transcript()
            self._save_transcript(transcript)
            eval_logger.info(
                f"Saved intermediate transcript after turn {concluded_turn}"
            )

        except Exception as exc:
            traceback.print_exc()
            eval_logger.error(
                f"Failed to save intermediate state for turn {concluded_turn}",
                extra={"error": str(exc)},
            )

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputAudioRawFrame) and self._is_bot_interrupt_triggered:
            # don't forward bot audio frames after the interruption has been triggered
            return
        elif isinstance(frame, InputAudioRawFrame) and self._turn_index:
            log_and_print("Received audio frame from agent")
            # Save bot audio chunk
            turn_index = self._turn_index
            chunk_index = self._bot_audio_chunk_indices.get(turn_index, 0)
            self._bot_audio_chunk_indices[turn_index] = chunk_index + 1
            audio_save_path = os.path.join(
                self._audio_save_dir, f"{turn_index}_bot_{chunk_index}.wav"
            )
            await save_audio_chunk(
                audio_save_path, frame.audio, frame.sample_rate, frame.num_channels
            )

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
                    # once the simulated user stops speaking, mark the bot as not
                    # interrupted anymore and spoken text buffer as not complete anymore
                    self._is_bot_interrupt_decided = False
                    self._is_bot_interrupt_triggered = False

                elif msg_type == "bot-output":
                    text = data.get("text") or ""
                    spoken = data.get("spoken") or False

                    if text:
                        # log_and_print(
                        #     f"{INTERRUPTION_COLOR}Agent message for debugging: {data}{RESET_COLOR}"
                        # )
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
                                    f"{GENERAL_LOG_COLOR}Agent speaking the generated message: {text}{RESET_COLOR}"
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
                # Build and save final transcript
                end_reason = "max_turns" if self._ended_due_to_max_turns else None
                transcript = self._build_serialized_transcript(end_reason=end_reason)
                self._save_transcript(transcript)

                with open(
                    os.path.join(self._output_dir, "tool_calls.json"), "w"
                ) as tool_calls_file:
                    json.dump(self._tool_calls, tool_calls_file, indent=4)

                with open(
                    os.path.join(self._output_dir, "stt_outputs.json"), "w"
                ) as stt_outputs_file:
                    json.dump(self._stt_outputs, stt_outputs_file, indent=4)

                # Final cleanup: combine any remaining audio chunks that weren't processed
                if os.path.exists(self._audio_save_dir):
                    combine_turn_audio_chunks(self._audio_save_dir)
                    eval_logger.info(
                        "Final cleanup: combined any remaining audio chunks"
                    )

            except Exception as exc:
                traceback.print_exc()
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


class SilencePadder(FrameProcessor):
    """Adds silence padding after TTS audio to help STT services flush transcriptions.

    Some STT services (like Sarvam) need trailing silence to properly detect
    end of speech and return final transcriptions.
    """

    def __init__(
        self,
        silence_duration_ms: int = 1000,
        chunk_ms: int = 40,
        audio_save_dir: str = None,
        rtvi_message_adapter: "RTVIMessageFrameAdapter" = None,
    ):
        super().__init__(enable_direct_mode=True, name="SilencePadder")
        self._silence_duration_ms = silence_duration_ms
        self._chunk_ms = chunk_ms
        self._last_sample_rate = 16000
        self._last_num_channels = 1
        self._audio_save_dir = audio_save_dir
        self._rtvi_message_adapter = rtvi_message_adapter
        self._user_audio_chunk_indices = (
            {}
        )  # Track chunk indices for user audio per turn

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # Track audio parameters from outgoing audio frames and save user audio chunks
        if isinstance(frame, OutputAudioRawFrame):
            self._last_sample_rate = frame.sample_rate
            self._last_num_channels = frame.num_channels
            log_and_print("Sending audio frame from simulated user")
            # Save user audio chunk
            if self._audio_save_dir and self._rtvi_message_adapter:
                turn_index = self._rtvi_message_adapter._turn_index
                chunk_index = self._user_audio_chunk_indices.get(turn_index, 0)
                self._user_audio_chunk_indices[turn_index] = chunk_index + 1
                audio_save_path = os.path.join(
                    self._audio_save_dir, f"{turn_index}_user_{chunk_index}.wav"
                )
                await save_audio_chunk(
                    audio_save_path, frame.audio, frame.sample_rate, frame.num_channels
                )

        # When TTS stops, add silence padding before pushing the frame
        if isinstance(frame, TTSStoppedFrame):
            await self._push_silence()

        await self.push_frame(frame, direction)

    async def _push_silence(self):
        """Generate and push silence frames."""
        frames_per_chunk = max(
            1, int(self._last_sample_rate * (self._chunk_ms / 1000.0))
        )
        silence_chunks = max(1, int(self._silence_duration_ms / self._chunk_ms))
        # 16-bit audio: 2 bytes per sample
        silence_audio = b"\x00" * (frames_per_chunk * self._last_num_channels * 2)

        for _ in range(silence_chunks):
            eval_logger.warning(
                "Sending simulated silence frames",
            )

            frame = OutputAudioRawFrame(
                audio=silence_audio,
                sample_rate=self._last_sample_rate,
                num_channels=self._last_num_channels,
            )
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)
            await asyncio.sleep(self._chunk_ms / 1000.0)


class MaxTurnsEndProcessor(FrameProcessor):
    """Processor that ends the task after number of assistant messages exceeds max_turns."""

    def __init__(self, max_turns: int, rtvi_adapter, context: LLMContext):
        super().__init__(enable_direct_mode=True, name="MaxTurnsEndProcessor")
        self._max_turns = max_turns
        self._rtvi_adapter = rtvi_adapter
        self._context = context

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            num_assistant_messages = len(
                [
                    message
                    for message in self._context.get_messages()
                    if message.get("role") == "assistant"
                ]
            )
            if num_assistant_messages == self._max_turns:
                log_and_print(
                    f"{INTERRUPTION_COLOR}Max turns ({self._max_turns}) reached, ending conversation{RESET_COLOR}"
                )
                self._rtvi_adapter._ended_due_to_max_turns = True
                await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

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
    gender: Literal["male", "female"],
    evaluation_criteria: list[dict],
    output_dir: str,
    interrupt_probability: float,
    port: int = DEFAULT_PORT,
    agent_speaks_first: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> dict:
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
        uri=f"ws://localhost:{port}",
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
                    traceback.print_exc()
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
                except Exception as exc:
                    traceback.print_exc()
                    raise exc

            raise (
                last_error
                if last_error
                else RuntimeError("Unknown error while connecting to WebSocket")
            )

    session.connect = locked_connect

    # Workaround for race condition: manually initialize the audio queue before connection
    transport.input()._audio_in_queue = asyncio.Queue()

    tts_language = (
        Language.KN
        if language == "kannada"
        else Language.HI if language == "hindi" else Language.EN
    )

    voice_name = "Zephyr" if gender == "female" else "Charon"
    language_code = (
        "hi-IN"
        if language == "hindi"
        else "kn-IN" if language == "kannada" else "en-US"
    )
    voice_id = f"{language_code}-Chirp3-HD-{voice_name}"
    eval_logger.info(f"Using voice ID: {voice_id}")
    tts = GoogleTTSService(
        voice_id=voice_id,
        params=GoogleTTSService.InputParams(language=tts_language),
        credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-5.2")

    transcript = TranscriptProcessor()

    simulation_system_prompt = system_prompt
    if not agent_speaks_first:
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
        audio_save_dir,
    )

    metrics_logger = MetricsLogger(ttft, processing_time, context)

    stt_logger = STTLogger(stt_outputs, rtvi_message_adapter)

    output_logger = IOLogger()

    # Add silence padding after TTS to help STT services (like Google, Sarvam) flush transcriptions
    silence_padder = SilencePadder(
        silence_duration_ms=1000,
        chunk_ms=40,
        audio_save_dir=audio_save_dir,
        rtvi_message_adapter=rtvi_message_adapter,
    )

    max_turns_end_processor = MaxTurnsEndProcessor(
        max_turns,
        rtvi_message_adapter,
        context,
    )

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
            max_turns_end_processor,  # Check and end after assistant processing if max_turns reached
            tts,  # TTS
            silence_padder,  # Add silence padding after TTS for STT flush
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

    function_call_handler.set_frame_sender(task.queue_frame)

    async def _handle_end_call_request(reason):
        if reason:
            eval_logger.info("Server requested end_call", extra={"reason": reason})
        else:
            eval_logger.info("Server requested end_call")

        await task.cancel()

    function_call_handler.set_end_call_callback(_handle_end_call_request)

    # @audio_buffer.event_handler("on_user_turn_audio_data")
    # async def on_user_turn_audio_data(buffer, audio, sample_rate, num_channels):
    #     eval_logger.info(f"Audio data received - bot")
    #     eval_logger.info(f"[bot] turn index: {rtvi_message_adapter._turn_index}")
    #     audio_save_path = os.path.join(
    #         audio_save_dir, f"{rtvi_message_adapter._turn_index}_bot.wav"
    #     )
    #     await save_audio_chunk(audio_save_path, audio, sample_rate, num_channels)

    # @audio_buffer.event_handler("on_bot_turn_audio_data")
    # async def on_bot_turn_audio_data(buffer, audio, sample_rate, num_channels):
    #     eval_logger.info(f"Audio data received - user")
    #     eval_logger.info(f"[user] turn index: {rtvi_message_adapter._turn_index}")
    #     audio_save_path = os.path.join(
    #         audio_save_dir, f"{rtvi_message_adapter._turn_index}_user.wav"
    #     )
    #     await save_audio_chunk(audio_save_path, audio, sample_rate, num_channels)

    @transport.event_handler("on_connected")
    async def on_connected(transport, client):
        eval_logger.info(f"WebSocket connected")
        await audio_buffer.start_recording()

        if not agent_speaks_first:
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
        f"Evaluating the conversation based on the criteria:\n\n{evaluation_criteria}"
    )
    # Get evaluation results from LLM judge
    llm_judge_result = await evaluate_simuation(
        transcript, evaluation_criteria, agent_system_prompt=system_prompt
    )

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

        stt_eval_references = user_messages_in_transcript[:min_len]
        stt_eval_predictions = filtered_stt_outputs[:min_len]

        if min_len > 0:
            stt_llm_judge_result = await stt_llm_judge_score(
                references=stt_eval_references,
                predictions=stt_eval_predictions,
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

    # Build evaluation_results.csv with all metrics
    evaluation_results_rows = []

    # Add evaluation criteria rows
    for eval_result in evaluation_results:
        evaluation_results_rows.append(
            {
                "name": eval_result["name"],
                "value": int(eval_result["match"]),
                "reasoning": eval_result["reasoning"],
            }
        )

    # Add latency metrics rows
    for processor, values in ttft_dict.items():
        if not values:
            continue

        processor = processor.lower()
        component = (
            "stt" if "stt" in processor else "tts" if "tts" in processor else "llm"
        )
        evaluation_results_rows.append(
            {
                "name": f"{component}/ttft",
                "value": float(np.mean(values)),
                "reasoning": "",
            }
        )

    for processor, values in processing_time_dict.items():
        if not values:
            continue

        processor = processor.lower()
        component = (
            "stt" if "stt" in processor else "tts" if "tts" in processor else "llm"
        )
        evaluation_results_rows.append(
            {
                "name": f"{component}/processing_time",
                "value": float(np.mean(values)),
                "reasoning": "",
            }
        )

    # Add STT LLM judge score row
    if stt_llm_judge_result:
        evaluation_results_rows.append(
            {
                "name": "stt_llm_judge_score",
                "value": stt_llm_judge_result["score"],
                "reasoning": "",
            }
        )

        df = pd.DataFrame(
            {
                "reference": stt_eval_references,
                "prediction": stt_eval_predictions,
                "score": [int(row["match"]) for row in stt_llm_judge_result["per_row"]],
                "reasoning": [
                    row["reasoning"] for row in stt_llm_judge_result["per_row"]
                ],
            }
        )
        df.to_csv(os.path.join(output_dir, "stt_results.csv"), index=False)

    # Save evaluation_results.csv
    if evaluation_results_rows:
        df = pd.DataFrame(evaluation_results_rows)
        df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)

    # Return all data
    return {
        "transcript": transcript,
        "stt_outputs": filtered_stt_outputs,
        "tool_calls": tool_calls,
        "evaluation_results": evaluation_results,
        "metrics": metrics,
    }


async def run_single_simulation_task(
    config: dict,
    persona_index: int,
    user_persona: dict,
    scenario_index: int,
    scenario: dict,
    output_dir: str,
    interrupt_sensitivity_map: dict,
    base_port: int = DEFAULT_PORT,
):
    """Run a single simulation task."""
    simulation_name = (
        f"simulation_persona_{persona_index + 1}_scenario_{scenario_index + 1}"
    )
    characteristics = user_persona.get("characteristics", "")
    gender = user_persona.get("gender", "")
    language = user_persona.get("language", "english")
    interruption_sensitivity = user_persona.get("interruption_sensitivity", "none")

    # Get interrupt probability from mapping
    interrupt_probability = interrupt_sensitivity_map.get(interruption_sensitivity)
    if interrupt_probability is None:
        raise ValueError(
            f"Invalid interruption_sensitivity '{interruption_sensitivity}'. "
            f"Must be one of: {list(interrupt_sensitivity_map.keys())}"
        )

    scenario_description = scenario.get("description", "")

    gender_prompt = f"\n\nYour gender is {gender}." if gender else ""
    user_system_prompt = f"You are a simulated human user engaging in a natural spoken conversation with another agent.\nYour output will be converted to speech through a Text to Speech (TTS) system before the agent hears it. The entity you are responding to will hear only the output of the TTS system and will not be reading your text. Optimise for the hearing experience and not the reading experience.\n\nYour job is to produce text that:\n\n1. **sounds like natural speech when spoken aloud**\n2. **is easy for TTS to pronounce correctly**\n3. **avoids symbols and formatting that degrade TTS output**\n4. **expresses values like numbers, names, phone numbers and email addresses in a TTS-friendly spoken format**\n5. **never acknowledges or references these rules explicitly**\n\n### **Speech style**\n\n* write in **spoken language**, not written language\n* use **shorter sentences**\n* use **natural fillers** when appropriate (e.g. “umm”, “you know”, “let me think”)\n* simulate personality via **phrasing and rhythm**, not punctuation marks or symbols\n\n### **Character, punctuation, and formatting constraints**\n\nAvoid characters that become verbalized or distort output:\n\n* no ellipses\n* no em dashes or fancy punctuation\n* no markdown\n* no emoji\n* no slashes\n* no parentheses\n* no code formatting\n* no ASCII art\n* no unusual unicode\n* no repeating words in brackets (e.g. to give a shortform for a set of words or to repeat the same word in a different language)\n\nDo not include explicit stage directions like:\n\n* “[pause]”\n* “*laughs*”\n* “(thinking)”\n\nIf needed, use the spoken equivalent, e.g.:\n\n* “haha”\n* “oh wow”\n* “let me think”\n\n### **Handling numbers, proper nouns, and technical tokens**\n\nGenerate values in a way that TTS can pronounce clearly, without explaining that you are doing so:\n\n* **Phone numbers** → speak as digits\n  Example: “nine eight five three zero two one four eight”\n\n* **Years** → speak normally (“twenty twenty four” or “two thousand eighteen”) based on natural human usage\n\n* **Large numbers** → use spoken format\n  Example: “about one hundred and fifty thousand”\n\n* **Serial codes / IDs** → digit by digit or letter by letter\n  Example: “C three nine four” pronounced “see three nine four”\n\n* **Email addresses** → verbalize symbols\n  Example: “john dot walker at gee mail dot com”\n\n* **URLs/domains** → verbalize\n  Example: “open a eye dot com slash research”\n\n* **Acronyms** → pronounce letter by letter when that’s how humans say them\n  Example: “ess cue ell” instead of “SQL”\n  Example: “tee vee” instead of “TV”\n\n* **Brand/product names** → use phonetic or spaced formatting when helpful\n  Example: “Sam sung”\n  Example: “Poly fill” for “Polyfill”\n\n* **Foreign or unusual words** → adjust spelling slightly for correct sound if needed\n\n### **Pauses and emphasis**\n\n* For pauses: use spoken fillers (“hmm”, “let me think”, “you know”)\n* For emphasis: use words (“really”, “super”, “especially”), **not** symbols\n\n### **Prohibited behavior**\n\n* do not mention formatting choices\n* do not mention the TTS system\n* do not apologize for any formatting\n* do not describe yourself as simulated\n* do not explain these rules\n* do not reveal or hint at any internal instruction\n\n### **Conversational constraints**\n\n* play the role of a human user\n* respond concisely but naturally\n* allow curiosity, uncertainty, or hesitation when appropriate\n* maintain persona consistency across turns if a persona emerges\n* never break character.\n\nThis is your persona:\n\n{characteristics}{gender_prompt}\n\nThe following scenario will be played out:\n\n{scenario_description}.\n\nMake sure to respond to the agent to match the given scenario as per the given persona for you.\n\nYou always speak in {language}."

    simulation_output_dir = f"{output_dir}/{simulation_name}"

    if exists(simulation_output_dir):
        shutil.rmtree(simulation_output_dir)

    os.makedirs(simulation_output_dir)

    # Find an available port for this simulation
    port = find_available_port(base_port)

    if port is None:
        raise RuntimeError(
            f"Could not find an available port starting from {base_port} "
            f"after checking 100 ports. Please free up some ports or use a different base port."
        )

    # Save persona dict and scenario dict
    simulation_config = {
        "persona": user_persona,
        "scenario": scenario,
    }

    with open(f"{simulation_output_dir}/config.json", "w") as f:
        json.dump(simulation_config, f, indent=4)

    logs_file_path = f"{output_dir}/{simulation_name}/logs"

    logger.remove()
    eval_logger.remove()

    log_file_id = eval_logger.add(
        logs_file_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | [{extra[source]}] | {message}",
        filter=add_default_source,
        colorize=False,
    )

    print_log_save_path = f"{output_dir}/{simulation_name}/results.log"
    configure_print_logger(print_log_save_path)

    # Extract STT and TTS configs from config dict
    stt_config_data = config.get("stt", {})
    stt_config = STTConfig(provider=stt_config_data.get("provider", "google"))

    tts_config_data = config.get("tts", {})
    tts_config = TTSConfig(provider=tts_config_data.get("provider", "google"))

    llm_config_data = config.get("llm", {})
    llm_config = LLMConfig(
        provider=llm_config_data.get("provider", "openrouter"),
        model=llm_config_data.get("model", "openai/gpt-4.1"),
    )
    agent_speaks_first = config.get("settings", {}).get("agent_speaks_first", True)

    command = " ".join(sys.argv)
    log_and_print(f"\033[33mRunning command\033[0m: {command}")

    log_and_print("--------------------------------")
    log_and_print(f"""Running simulation \033[93m{simulation_name}\033[0m""")
    log_and_print(f"\033[93mPersona:\033[0m\n{characteristics}")
    log_and_print(f"\033[93mGender:\033[0m {gender}" if gender else "")
    log_and_print(f"\033[93mLanguage:\033[0m {language}")
    log_and_print(f"\033[93mScenario:\033[0m\n{scenario_description}")
    log_and_print(f"\033[93mPort:\033[0m {port}")
    log_and_print(f"\033[93mSTT Config:\033[0m {stt_config}")
    log_and_print(f"\033[93mTTS Config:\033[0m {tts_config}")
    log_and_print(f"\033[93mLLM Config:\033[0m {llm_config}")
    log_and_print(f"\033[93mAgent Speaks First:\033[0m {agent_speaks_first}")
    log_and_print("--------------------------------")

    simulation_result = None
    bot_task = None
    sim_task = None
    try:
        bot_task = asyncio.create_task(
            start_bot(
                config["system_prompt"] + f"\n\nYou must always speak in {language}.",
                config["tools"],
                language,
                port=port,
                stt_config=stt_config,
                tts_config=tts_config,
                llm_config=llm_config,
                agent_speaks_first=agent_speaks_first,
            )
        )
        # Give the bot a moment to start listening before connecting
        await asyncio.sleep(1.0)
        sim_task = asyncio.create_task(
            run_simulation(
                user_system_prompt,
                language,
                gender,
                config["evaluation_criteria"],
                simulation_output_dir,
                interrupt_probability=interrupt_probability,
                port=port,
                agent_speaks_first=agent_speaks_first,
                max_turns=config.get("settings", {}).get(
                    "max_turns", DEFAULT_MAX_TURNS
                ),
            )
        )
        simulation_tasks = [bot_task, sim_task]
        done, pending = await asyncio.wait(simulation_tasks, timeout=EVAL_TIMEOUT_SECS)
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
                traceback.print_exc()
                eval_logger.error(f"ERROR: Failed to get simulation result: {e}")
    except Exception as e:
        traceback.print_exc()
        eval_logger.error(f"ERROR: Unable to run: {e}")
    finally:
        # Ensure all tasks are fully cancelled and cleaned up
        for task in [bot_task, sim_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Give async cleanup tasks time to complete (WebSocket close, STT stream close, etc.)
        await asyncio.sleep(0.5)

        eval_logger.remove(log_file_id)

    # Combine audio chunks for each turn into single turn files, then combine all into conversation.wav
    audio_dir = os.path.join(simulation_output_dir, "audios")
    conversation_audio_path = os.path.join(simulation_output_dir, "conversation.wav")
    if os.path.exists(audio_dir):
        # First, combine chunks for each turn (e.g., 0_bot_0.wav, 0_bot_1.wav -> 0_bot.wav)
        combine_turn_audio_chunks(audio_dir)
        log_and_print(f"Combined turn audio chunks in {audio_dir}")
        # Then combine all turn files into conversation.wav
        combine_audio_files(audio_dir, conversation_audio_path)
        log_and_print(f"Combined audio saved to {conversation_audio_path}")

    # Return metrics for aggregation
    if simulation_result:
        sim_metrics_row = {"name": simulation_name}

        # Evaluation criteria metrics
        for eval_result in simulation_result.get("evaluation_results", []):
            criterion_name = eval_result["name"]
            match_value = float(eval_result["match"])
            sim_metrics_row[criterion_name] = match_value

        # STT LLM judge score
        stt_judge = simulation_result.get("metrics", {}).get("stt_llm_judge")
        if stt_judge and "score" in stt_judge:
            sim_metrics_row["stt_llm_judge_score"] = stt_judge["score"]

        return (
            sim_metrics_row,
            simulation_result.get("evaluation_results", []),
            stt_judge,
        )

    return None, [], None


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
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Websocket port to run the simulations on",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Mapping from interruption_sensitivity labels to probabilities
    interrupt_sensitivity_map = {
        "none": 0,
        "low": 0.25,
        "medium": 0.5,
        "high": 0.8,
    }

    # Run simulations sequentially
    results = []
    total_simulations = len(config["personas"]) * len(config["scenarios"])
    simulation_count = 0

    for persona_index, user_persona in enumerate(config["personas"]):
        for scenario_index, scenario in enumerate(config["scenarios"]):
            simulation_count += 1
            try:
                result = await run_single_simulation_task(
                    config=config,
                    persona_index=persona_index,
                    user_persona=user_persona,
                    scenario_index=scenario_index,
                    scenario=scenario,
                    output_dir=args.output_dir,
                    interrupt_sensitivity_map=interrupt_sensitivity_map,
                    base_port=args.port,
                )
                results.append(result)
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Simulation failed with error: {e}")
                results.append(e)
            finally:
                # Cleanup between simulations to prevent state leakage
                # This is critical when running multiple simulations sequentially
                if simulation_count < total_simulations:
                    # Force garbage collection to clean up lingering objects
                    # (WebSocket connections, STT streams, turn analyzer state, etc.)
                    gc.collect()

                    # Wait for ports to be fully released (TIME_WAIT state)
                    # and for async cleanup tasks to complete
                    cleanup_delay = 3.0  # seconds
                    log_and_print(
                        f"Waiting {cleanup_delay}s for cleanup before next simulation..."
                    )
                    await asyncio.sleep(cleanup_delay)

    # Aggregated metrics across all simulations
    all_simulation_metrics = []
    metrics_by_criterion = defaultdict(list)
    stt_llm_judge_scores = []

    # Collect metrics from results
    for result in results:
        if isinstance(result, Exception):
            continue

        sim_metrics_row, evaluation_results, stt_judge = result
        if sim_metrics_row is None:
            continue

        all_simulation_metrics.append(sim_metrics_row)

        # Evaluation criteria metrics
        for eval_result in evaluation_results:
            criterion_name = eval_result["name"]
            match_value = float(eval_result["match"])
            metrics_by_criterion[criterion_name].append(match_value)

        # STT LLM judge score
        if stt_judge and "score" in stt_judge:
            stt_llm_judge_scores.append(stt_judge["score"])

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
