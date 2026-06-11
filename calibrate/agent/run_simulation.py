import aiohttp
import asyncio
import gc
import json
import os
import sys
import socket
from os.path import join, exists
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Literal
import traceback
from uuid import uuid4
from deepgram import LiveOptions
from loguru import logger
from PIL.ImageFile import ImageFile
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from calibrate.utils import (
    current_context,
    current_simulation_name,
    add_default_source,
    configure_print_logger,
    cleanup_print_logger,
    log_and_print,
    save_audio_chunk,
    combine_turn_audio_chunks,
    combine_audio_files,
    build_tools_schema,
    make_webhook_call,
    provider_log_file,
    summarize_metric_distribution,
)
from calibrate.llm.metrics import evaluate_simuation, DEFAULT_SIMULATION_JUDGE_MODEL
from calibrate.stt.metrics import (
    get_llm_judge_score as stt_llm_judge_score,
    DEFAULT_STT_JUDGE_MODEL,
)
from calibrate.judges import (
    DEFAULT_STT_EVALUATOR,
    attach_evaluator_id,
    evaluator_result_value,
    format_evaluation_result_lines,
    is_rating,
    require_simulation_evaluators,
    write_evaluator_config,
)
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
DEFAULT_AGENT_SPEAKS_FIRST = True

# Pipecat logs Google STT gRPC 409 (idle stream) at ERROR while reconnecting; that is
# recoverable and must not trigger our error sink (which cancels the eval pipeline).
_BENIGN_GOOGLE_STT_IDLE_TIMEOUT = (
    "409 Stream timed out after receiving no more client requests"
)


def _is_benign_google_stt_idle_error(log_message: str) -> bool:
    return (
        "GoogleSTTService" in log_message
        and _BENIGN_GOOGLE_STT_IDLE_TIMEOUT in log_message
    )


def count_agent_message_turns(messages: list) -> int:
    """Count tested-agent speaking turns in the simulation LLM context.

    The remote agent's speech is stored as ``user`` messages; the serialized
    transcript flips roles so those become ``assistant``. Consecutive ``user``
    messages (e.g. streaming fragments) count as one turn.
    """
    count = 0
    in_user_run = False
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user":
            if not in_user_run:
                count += 1
                in_user_run = True
        elif role is not None:
            in_user_run = False
    return count


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
    InterruptionTaskFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
    ErrorFrame,
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
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService

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
TRANSCRIPT_FILE_NAME = "transcript.json"


def find_available_port() -> int:
    """Find an available port by letting the OS assign one.

    This is the most robust approach for scalability when multiple processes
    may be trying to find ports simultaneously. The OS handles the race condition
    by assigning unique ephemeral ports.

    Returns:
        An available port number.

    Raises:
        RuntimeError: If unable to find an available port.
    """
    try:
        # Let the OS assign an available port by binding to port 0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            return port
    except OSError as e:
        raise RuntimeError(f"Could not find an available port: {e}")


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
        agent_speaks_first: bool = True,
        max_turns: int = DEFAULT_MAX_TURNS,
    ):
        super().__init__(enable_direct_mode=True, name="RTVIMessageFrameAdapter")
        self._context = context
        self._audio_buffer = audio_buffer
        self._agent_speaks_first = agent_speaks_first
        self._max_turns = max_turns
        self._interrupt_probability = interrupt_probability
        self._tool_calls = tool_calls
        self._output_dir = Path(output_dir)
        self._audio_save_dir = audio_save_dir
        self._turn_index = (
            0  # last active transcript line index (for logs / checkpoints)
        )
        # Single 1-based line index for WAV names; see _assign_next_transcript_audio_line.
        self._active_transcript_audio_index = 0
        # Role that owns ``_active_transcript_audio_index``. Audio chunks are only
        # saved when the inbound speaker matches this role — prevents a tool-only
        # bot turn from saving ``{prev_user_idx}_bot.wav`` chunks against a stale
        # index that still points at the previous sim-user line.
        self._active_transcript_audio_role: Optional[str] = None
        self._last_transcript_line_assigned = (
            0  # strict sequence when context lags commits
        )
        self._stt_turn_index = (
            0  # increments when a bot line is reserved (on first real speech)
        )
        self._stt_outputs = stt_outputs
        self._ttft = ttft
        self._processing_time = processing_time
        self._text_buffer = ""  # buffer of the text that the bot has generated so far (received stream; used for interrupt completion matching)
        self._heard_text_buffer = ""  # buffer of agent text actually spoken (fed to simulated-user STT as what was heard)
        self._spoken_text_buffer = (
            ""  # buffer of the text that the bot has spoken so far
        )
        self._is_bot_interrupt_decided = False  # whether the decision to interrupt the bot by the user has been made yet
        self._is_bot_interrupt_triggered = False  # whether the spoken text buffer is complete and matches the text buffer; only when this becomes true is when the intteruption actually triggered
        self._pending_user_turn = False  # set True on bot-started-speaking; False on bot-stopped-speaking after frames are pushed
        self._turns_concluded = set()
        self._serialized_transcript = []  # Store transcripts for return
        self._ended_due_to_max_turns = False
        self._bot_audio_chunk_indices = {}  # Track chunk indices for bot audio per turn
        self._awaiting_first_bot_audio_chunk = (
            False  # True after bot-started until first qualifying ``spoken`` RTVI text
        )
        # Set by SimulatedUserTurnIndexHook on LLMFullResponseStartFrame; consumed
        # lazily by SilencePadder when the first sim-user TTS audio frame arrives.
        # Avoids allocating an index for sim-user turns that get interrupted before
        # any audio actually flows.
        self._sim_user_turn_pending = False
        # Inbound bot audio frames that arrive after ``bot-started-speaking`` but
        # before the first ``spoken=True`` RTVI confirmation. TTS audio commonly
        # streams ~1s ahead of its ``spoken`` event, so the first sentence's
        # audio would otherwise be dropped. Flushed when the bot line is
        # reserved; cleared on ``bot-stopped-speaking`` for tool-only turns.
        self._pending_bot_audio_frames: list = []

    async def _ensure_bot_transcript_line_for_current_turn(
        self, spoken_fragment: str = ""
    ) -> None:
        """Reserve the next transcript line only for real agent speech.

        Tool-only turns can still emit ``bot-started-speaking`` and inbound audio
        without user-facing TTS; we only reserve a line after ``spoken`` RTVI
        text, and only when there is actual wording (stream buffer or fragment).
        """
        if not self._awaiting_first_bot_audio_chunk:
            return
        lexical = (self._text_buffer or "").strip() or (spoken_fragment or "").strip()
        if len(lexical) < 2:
            return
        if not any(ch.isalpha() for ch in lexical):
            return
        self._awaiting_first_bot_audio_chunk = False
        if self._active_transcript_audio_role == "bot":
            # Bot already owns the active line — a single LLM response can be
            # TTS'd as multiple sentences, each producing its own
            # bot-started/stopped cycle. Keep appending chunks to the same file
            # rather than splitting one transcript entry across files.
            line = self._active_transcript_audio_index
            eval_logger.info(
                f"[rtvi] continuing bot transcript audio index (same turn): {line}"
            )
        else:
            self._stt_turn_index += 1
            line = self._assign_next_transcript_audio_line(role="bot")
            self._turn_index = line
            eval_logger.info(f"[rtvi] transcript audio index (bot, on speech): {line}")
        await self._flush_pending_bot_audio()

    async def _flush_pending_bot_audio(self) -> None:
        """Save buffered bot audio frames against the just-reserved line."""
        if not self._pending_bot_audio_frames or not self._audio_save_dir:
            self._pending_bot_audio_frames = []
            return
        turn_index = self._active_transcript_audio_index
        for frame in self._pending_bot_audio_frames:
            chunk_index = self._bot_audio_chunk_indices.get(turn_index, 0)
            self._bot_audio_chunk_indices[turn_index] = chunk_index + 1
            audio_save_path = os.path.join(
                self._audio_save_dir, f"{turn_index}_bot_{chunk_index}.wav"
            )
            await save_audio_chunk(
                audio_save_path, frame.audio, frame.sample_rate, frame.num_channels
            )
        self._pending_bot_audio_frames = []

    def _assign_next_transcript_audio_line(self, role: str) -> int:
        """Next 1-based transcript line for ``{N}_bot`` / ``{N}_user`` chunk files.

        Strictly monotonic: each segment that gets a WAV (sim-user LLM start, or
        first real agent speech for a turn) advances the index so files are
        ``1_bot``, ``2_user``, ``3_bot``, … in conversation order.
        """
        candidate = self._last_transcript_line_assigned + 1
        self._last_transcript_line_assigned = candidate
        self._active_transcript_audio_index = candidate
        self._active_transcript_audio_role = role
        return candidate

    async def _reset_buffers(self):
        concluded_turn = self._turn_index
        self._turns_concluded.add(concluded_turn)  # mark the turn as concluded
        self._text_buffer = ""
        self._heard_text_buffer = ""
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

        # Merge consecutive content messages from the same role into one. Tool-call
        # entries (no ``content`` key) act as separators and are left untouched.
        merged_transcript: list[dict] = []
        for message in serialized_transcript:
            if (
                merged_transcript
                and "content" in message
                and "content" in merged_transcript[-1]
                and message.get("role") == merged_transcript[-1].get("role")
            ):
                prev = merged_transcript[-1].get("content") or ""
                curr = message.get("content") or ""
                if prev and curr:
                    merged_transcript[-1]["content"] = f"{prev} {curr}"
                else:
                    merged_transcript[-1]["content"] = prev or curr
            else:
                merged_transcript.append(dict(message))
        serialized_transcript = merged_transcript

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
            os.path.join(self._output_dir, TRANSCRIPT_FILE_NAME), "w"
        ) as transcripts_file:
            json.dump(transcript, transcripts_file, indent=4)

    async def _save_intermediate_state(
        self,
        concluded_turn: Optional[int] = None,
        *,
        reason: Optional[str] = None,
    ):
        """Save intermediate transcript after a checkpoint (turn concluded or user stopped)."""
        end_reason = "max_turns" if self._ended_due_to_max_turns else None
        transcript = self._build_serialized_transcript(end_reason=end_reason)
        self._save_transcript(transcript)
        if reason is not None:
            eval_logger.info(reason)
        elif concluded_turn is not None:
            eval_logger.info(
                f"Saved intermediate transcript after turn {concluded_turn}"
            )

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputAudioRawFrame) and self._is_bot_interrupt_triggered:
            # don't forward bot audio frames after the interruption has been triggered
            return
        elif isinstance(frame, InputAudioRawFrame):
            # Never reserve a line from raw inbound audio: tool rounds can still
            # carry silence or noise. Lines are reserved from ``spoken`` RTVI only.
            if self._awaiting_first_bot_audio_chunk and self._audio_save_dir:
                # TTS audio for the first sentence routinely arrives ~1s before
                # its ``spoken=True`` confirmation. Buffer here; flush once the
                # bot line is reserved (or drop on tool-only stop).
                self._pending_bot_audio_frames.append(frame)
            elif (
                self._active_transcript_audio_index > 0
                and self._active_transcript_audio_role == "bot"
            ):
                turn_index = self._active_transcript_audio_index
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
                    agent_turns_so_far = count_agent_message_turns(
                        self._context.get_messages()
                    )
                    if agent_turns_so_far >= self._max_turns:
                        log_and_print(
                            f"{INTERRUPTION_COLOR}Max turns ({self._max_turns}) reached, ending conversation{RESET_COLOR}"
                        )
                        self._ended_due_to_max_turns = True
                        await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
                    else:
                        self._awaiting_first_bot_audio_chunk = True
                        self._pending_user_turn = True
                        # Cancel any pending sim-user line allocation: bot has
                        # the floor and any in-flight sim-user TTS will be
                        # killed by the upcoming InterruptionTaskFrame.
                        self._sim_user_turn_pending = False
                        # The bot has the floor: stop the sim user's in-flight TTS
                        # and flush its assistant aggregator so anything spoken so
                        # far is committed as its own turn.
                        await self.push_frame(
                            InterruptionTaskFrame(), FrameDirection.UPSTREAM
                        )
                        generated_frames.append(UserStartedSpeakingFrame())
                elif msg_type == "bot-stopped-speaking" and self._pending_user_turn:
                    if self._awaiting_first_bot_audio_chunk:
                        # Tool-only turn: no ``spoken=True`` ever fired, so any
                        # buffered audio belongs to nothing the user heard.
                        self._pending_bot_audio_frames = []
                    self._awaiting_first_bot_audio_chunk = False
                    self._pending_user_turn = False
                    generated_frames.extend(
                        [
                            TranscriptionFrame(
                                text=self._heard_text_buffer,
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
                    await self._save_intermediate_state(
                        reason=(
                            "Saved intermediate transcript after simulated user stopped speaking "
                            f"(turn index {self._turn_index})"
                        )
                    )

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
                                await self._ensure_bot_transcript_line_for_current_turn(
                                    text
                                )
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
                                    self._pending_user_turn = False
                                    self._awaiting_first_bot_audio_chunk = False

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
                                # Received stream only (not yet spoken): track for interrupt
                                # completion matching but do not feed STT — the simulated user
                                # should only react to spoken audio.
                                log_and_print(
                                    f"{PARTIAL_AGENT_MESSAGE_COLOR}Received agent message{RESET_COLOR}: {text}{RESET_COLOR}"
                                )
                                self._text_buffer += " " + text
                            elif spoken and not self._is_bot_interrupt_decided:
                                await self._ensure_bot_transcript_line_for_current_turn(
                                    text
                                )
                                log_and_print(
                                    f"{GENERAL_LOG_COLOR}Agent speaking the generated message: {text}{RESET_COLOR}"
                                )
                                self._heard_text_buffer += " " + text
                                result_payload = data if data else None

                                if np.random.rand() < self._interrupt_probability:
                                    log_and_print(
                                        f"--------------------------------\n{INTERRUPTION_COLOR}[User interrupts the bot]{RESET_COLOR}\n--------------------------------"
                                    )
                                    self._is_bot_interrupt_decided = True
                                    # Align interrupt target with heard text only (received may run ahead of TTS).
                                    self._text_buffer = self._heard_text_buffer

                                generated_frames.append(
                                    InterimTranscriptionFrame(
                                        text=self._heard_text_buffer,
                                        user_id=user_id,
                                        timestamp=timestamp,
                                        result=result_payload,
                                    )
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
                                await self._ensure_bot_transcript_line_for_current_turn(
                                    text
                                )

                for generated_frame in generated_frames:
                    await self.push_frame(generated_frame, direction)

        if isinstance(frame, EndFrame) or isinstance(frame, CancelFrame):
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
                # STTLogger pre-seeds an empty entry to handle the user-speaks-first
                # case (first transcription lands at index 0). When the agent speaks
                # first, that slot is never filled and stays "" — strip empties so
                # the dumped file reflects only real transcriptions.
                json.dump(
                    [s for s in self._stt_outputs if s.strip()],
                    stt_outputs_file,
                    indent=4,
                )

            # Final cleanup: combine any remaining audio chunks that weren't processed
            if os.path.exists(self._audio_save_dir):
                combine_turn_audio_chunks(self._audio_save_dir)
                eval_logger.info("Final cleanup: combined any remaining audio chunks")

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
                        if self._rtvi_adapter._stt_turn_index > self.last_turn_index:
                            self._stt_outputs.append(text)
                            self.last_turn_index = self._rtvi_adapter._stt_turn_index
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


class SimulatedUserTurnIndexHook(FrameProcessor):
    """Marks that the sim user has started an LLM turn.

    Allocation of the transcript line itself is deferred until actual TTS audio
    is about to be written by ``SilencePadder``. This avoids "orphan" sim-user
    indices when the bot interrupts before any audio flows — those allocations
    used to flip the active role to ``"user"`` and break bot continuation
    role-stickiness, splitting a single bot message across multiple ``_bot.wav``
    files.
    """

    def __init__(self, rtvi_adapter: "RTVIMessageFrameAdapter"):
        super().__init__(enable_direct_mode=True, name="SimulatedUserTurnIndexHook")
        self._rtvi_adapter = rtvi_adapter

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            # Just signal intent — actual line allocation happens lazily in
            # SilencePadder when the first audio frame arrives.
            self._rtvi_adapter._sim_user_turn_pending = True

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
            # Lazy allocation: only reserve a sim-user line when we actually have
            # audio to write. If the sim user got interrupted before any audio
            # flowed, no line is reserved — preserving bot role-stickiness.
            if (
                self._audio_save_dir
                and self._rtvi_message_adapter
                and self._rtvi_message_adapter._sim_user_turn_pending
            ):
                self._rtvi_message_adapter._sim_user_turn_pending = False
                if self._rtvi_message_adapter._active_transcript_audio_role != "user":
                    line = self._rtvi_message_adapter._assign_next_transcript_audio_line(
                        role="user"
                    )
                    self._rtvi_message_adapter._turn_index = line
                    eval_logger.info(f"[sim user] transcript audio index: {line}")
            if (
                self._audio_save_dir
                and self._rtvi_message_adapter
                and self._rtvi_message_adapter._active_transcript_audio_index > 0
                and self._rtvi_message_adapter._active_transcript_audio_role == "user"
            ):
                turn_index = self._rtvi_message_adapter._active_transcript_audio_index
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


class RTVIFunctionCallResponder(FrameProcessor):
    def __init__(
        self,
        tool_calls: list[dict],
        context: LLMContext,
        webhook_configs: dict[str, dict] = None,
    ):
        super().__init__(enable_direct_mode=True, name="RTVIFunctionCallResponder")
        self._send_frame = None
        self._end_call_callback = None
        self._tool_calls = tool_calls
        self._context = context
        self._webhook_configs = webhook_configs or {}

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

        # Check if this is a webhook tool
        if function_name in self._webhook_configs:
            webhook_config = self._webhook_configs[function_name]
            result = await make_webhook_call(webhook_config, arguments or {})
            return result, None

        # For all other (non-webhook) tools, return status received
        return {"status": "received"}, None

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
    evaluators: list[dict],
    output_dir: str,
    interrupt_probability: float,
    port: int = DEFAULT_PORT,
    agent_speaks_first: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
    tools: list[dict] = None,
    fallback_judge_model: str = DEFAULT_SIMULATION_JUDGE_MODEL,
    fallback_stt_judge_model: str = DEFAULT_STT_JUDGE_MODEL,
) -> dict:
    require_simulation_evaluators(evaluators)

    # Set context for EVAL logs
    current_context.set("EVAL")

    # Capture ERROR-level logs to surface pipecat internal errors
    captured_errors: list[str] = []
    # Mutable container so the error sink can access the pipeline task once it's created
    _pipeline_task_ref: list = [None]
    _error_triggered = False

    def error_capture_sink(message):
        nonlocal _error_triggered
        record = message.record
        if record["level"].name in ("ERROR", "CRITICAL"):
            text = record["message"]
            if _is_benign_google_stt_idle_error(text):
                eval_logger.debug(
                    "Skipping pipeline cancel for benign Google STT idle stream timeout",
                )
                return
            captured_errors.append(text)

            # Cancel the pipeline task immediately on critical errors
            # so the simulation doesn't wait for idle timeout
            if not _error_triggered and _pipeline_task_ref[0] is not None:
                _error_triggered = True
                pipeline_task = _pipeline_task_ref[0]
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(pipeline_task.cancel())
                )

    error_sink_id = logger.add(error_capture_sink, level="ERROR")

    try:
        return await _run_simulation_inner(
            system_prompt=system_prompt,
            language=language,
            gender=gender,
            evaluators=evaluators,
            output_dir=output_dir,
            interrupt_probability=interrupt_probability,
            port=port,
            agent_speaks_first=agent_speaks_first,
            max_turns=max_turns,
            tools=tools,
            captured_errors=captured_errors,
            pipeline_task_ref=_pipeline_task_ref,
            fallback_judge_model=fallback_judge_model,
            fallback_stt_judge_model=fallback_stt_judge_model,
        )
    finally:
        logger.remove(error_sink_id)


async def _run_simulation_inner(
    system_prompt: str,
    language: Literal["english", "hindi"],
    gender: Literal["male", "female"],
    evaluators: list[dict],
    output_dir: str,
    interrupt_probability: float,
    port: int,
    agent_speaks_first: bool,
    max_turns: int,
    tools: list[dict],
    captured_errors: list[str],
    pipeline_task_ref: list,
    fallback_judge_model: str = DEFAULT_SIMULATION_JUDGE_MODEL,
    fallback_stt_judge_model: str = DEFAULT_STT_JUDGE_MODEL,
) -> dict:
    # Build webhook configs from tools for function call handling
    webhook_configs = {}
    if tools:
        _, webhook_configs = build_tools_schema(tools)

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
        Language.KN
        if language == "kannada"
        else Language.HI if language == "hindi" else Language.EN
    )

    # ElevenLabs voice IDs for the simulated user. Word-level TTS so that
    # mid-utterance interrupts commit only the words actually spoken.

    if language == "hindi":
        if gender == "female":
            voice_id = "dVTC43Yewy5fAIcmsISI"
        else:
            voice_id = "hdkYGMdbdWZpANLZvmnk"
    else:
        # Default to English voices
        voice_id = (
            "OHY6EjdeHKeQymoihwfz" if gender == "female" else "fPIfC3elMLbN9tNwMXkw"
        )
    eval_logger.info(f"Using ElevenLabs voice ID: {voice_id}")
    elevenlabs_http_session = aiohttp.ClientSession()
    tts = ElevenLabsHttpTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=voice_id,
        aiohttp_session=elevenlabs_http_session,
        params=ElevenLabsHttpTTSService.InputParams(language=tts_language),
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
    function_call_handler = RTVIFunctionCallResponder(
        tool_calls, context, webhook_configs
    )

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
        agent_speaks_first=agent_speaks_first,
        max_turns=max_turns,
    )

    simulated_user_turn_index_hook = SimulatedUserTurnIndexHook(rtvi_message_adapter)

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
            simulated_user_turn_index_hook,
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

    # Expose task reference so the error capture sink can cancel it on critical errors
    pipeline_task_ref[0] = task

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

    try:
        await runner.run(task)
    finally:
        await elevenlabs_http_session.close()

    transcript = rtvi_message_adapter._serialized_transcript

    # Check if the simulation completed with a meaningful transcript
    # Only fail if there's no meaningful conversation - benign errors like STT timeouts
    # after conversation ends should not cause failure
    meaningful_messages = [
        msg
        for msg in transcript
        if msg.get("role") in ("user", "assistant") and msg.get("content")
    ]
    if not meaningful_messages:
        error_details = ""
        if captured_errors:
            error_details = f"{'; '.join(captured_errors)}"
        raise RuntimeError(
            f"Simulation failed: no meaningful conversation occurred.{error_details}"
        )

    log_and_print(
        f"Evaluating the conversation against {len(evaluators)} evaluator(s)."
    )
    # Get evaluation results from LLM judge
    llm_judge_result = await evaluate_simuation(
        transcript, evaluators, fallback_model=fallback_judge_model
    )

    def _build_eval_row(evaluator: dict, judge_row: dict) -> dict:
        row = {
            "name": evaluator["name"],
            "type": "rating" if is_rating(evaluator) else "binary",
            "value": evaluator_result_value(evaluator, judge_row),
            "reasoning": judge_row["reasoning"],
        }
        row = attach_evaluator_id(evaluator, row)
        if is_rating(evaluator):
            row["scale_min"] = int(evaluator["scale_min"])
            row["scale_max"] = int(evaluator["scale_max"])
        return row

    evaluation_results = [
        _build_eval_row(ev, llm_judge_result[ev["name"]]) for ev in evaluators
    ]

    for row in evaluation_results:
        for line in format_evaluation_result_lines(row):
            log_and_print(line)

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
                fallback_model=fallback_stt_judge_model,
            )

            # Surface per-evaluator STT judge results in the CLI/log so the
            # voice simulation output mirrors the conversation evaluators above
            # (e.g. ``[stt:semantic_match] mean pass rate: 83.3% (5/6)`` or
            # ``[stt:fluency] mean: 4.20/5``). The aggregated mean across rows
            # is the same value persisted into evaluation_results.csv.
            for stt_ev_name, stt_score in (
                stt_llm_judge_result.get("scores") or {}
            ).items():
                if stt_score.get("type") == "rating":
                    mean = stt_score.get("mean", 0.0)
                    scale_max = stt_score.get("scale_max")
                    log_and_print(
                        f"[stt:{stt_ev_name}] mean: {mean:.2f}"
                        + (f"/{scale_max}" if scale_max is not None else "")
                    )
                else:
                    mean = stt_score.get("mean", 0.0)
                    matched = sum(
                        1
                        for row in stt_llm_judge_result.get("per_row") or []
                        if row.get(stt_ev_name, {}).get("match")
                    )
                    log_and_print(
                        f"[stt:{stt_ev_name}] mean pass rate: "
                        f"{mean * 100:.1f}% ({matched}/{min_len})"
                    )
            log_and_print(
                f"[stt] overall score: {stt_llm_judge_result.get('score', 0.0):.2f}"
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
                "evaluator_id": eval_result.get("evaluator_id"),
                "name": eval_result["name"],
                "type": eval_result.get("type", "binary"),
                "value": eval_result["value"],
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
                "score": [
                    int(row[DEFAULT_STT_EVALUATOR["name"]]["match"])
                    for row in stt_llm_judge_result["per_row"]
                ],
                "reasoning": [
                    row[DEFAULT_STT_EVALUATOR["name"]]["reasoning"]
                    for row in stt_llm_judge_result["per_row"]
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
    semaphore: asyncio.Semaphore,
    config: dict,
    persona_index: int,
    user_persona: dict,
    scenario_index: int,
    scenario: dict,
    output_dir: str,
    interrupt_sensitivity_map: dict,
):
    """Run a single simulation task with semaphore for concurrency control."""
    async with semaphore:
        return await _run_single_simulation_inner(
            config=config,
            persona_index=persona_index,
            user_persona=user_persona,
            scenario_index=scenario_index,
            scenario=scenario,
            output_dir=output_dir,
            interrupt_sensitivity_map=interrupt_sensitivity_map,
        )


async def _run_single_simulation_inner(
    config: dict,
    persona_index: int,
    user_persona: dict,
    scenario_index: int,
    scenario: dict,
    output_dir: str,
    interrupt_sensitivity_map: dict,
):
    """Inner implementation of a single simulation task."""
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
    # Uses OS-assigned ephemeral port for robustness when multiple simulations run concurrently
    port = find_available_port()

    # Save persona dict and scenario dict
    simulation_config = {
        "persona": user_persona,
        "scenario": scenario,
    }

    with open(f"{simulation_output_dir}/config.json", "w") as f:
        json.dump(simulation_config, f, indent=4)

    logs_file_path = f"{output_dir}/{simulation_name}/logs"

    # Generate a unique ID for this simulation run to avoid conflicts
    # when multiple simulations with the same name run in parallel
    simulation_run_id = str(uuid4())

    # Create a unique loguru sink for this simulation with a strict filter
    # that only accepts logs from this simulation's context
    def simulation_filter(record):
        if "source" not in record["extra"]:
            context = current_context.get()
            record["extra"]["source"] = f"{context}-SYS"
        sim_id = record["extra"].get("simulation")
        return sim_id == simulation_run_id

    log_file_id = logger.add(
        logs_file_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | [{extra[source]}] | {message}",
        filter=simulation_filter,
        colorize=False,
    )

    # Route judge LLM input/output into this simulation's logs file.
    judge_log_token = provider_log_file.set(logs_file_path)

    # Configure print logger with unique ID for parallel execution
    print_log_save_path = f"{output_dir}/{simulation_name}/results.log"
    configure_print_logger(print_log_save_path, simulation_name=simulation_run_id)
    current_simulation_name.set(simulation_run_id)

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
    agent_speaks_first = config.get("settings", {}).get(
        "agent_speaks_first", DEFAULT_AGENT_SPEAKS_FIRST
    )

    max_turns = config.get("settings", {}).get("max_turns", DEFAULT_MAX_TURNS)

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
        log_and_print(f"\033[93mPort:\033[0m {port}")
        log_and_print(f"\033[93mSTT Config:\033[0m {stt_config}")
        log_and_print(f"\033[93mTTS Config:\033[0m {tts_config}")
        log_and_print(f"\033[93mLLM Config:\033[0m {llm_config}")
        log_and_print(f"\033[93mAgent Speaks First:\033[0m {agent_speaks_first}")
        log_and_print(f"\033[93mMax Turns:\033[0m {max_turns}")
        log_and_print("--------------------------------")

        simulation_result = None
        bot_task = None
        sim_task = None
        try:
            bot_task = asyncio.create_task(
                start_bot(
                    config["system_prompt"]
                    + f"\n\nYou must always speak in {language}.",
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

            # Check if bot_task failed during startup - if so, get its result to surface the error
            if bot_task.done():
                # This will raise if the bot task failed with an exception
                bot_task.result()
                # If we get here, bot completed without exception but also without starting server
                # this is still wrong because the bot should be running, not completed
                raise RuntimeError(
                    "Bot task completed unexpectedly before simulation could connect"
                )

            evaluators = config.get("evaluators") or []

            sim_task = asyncio.create_task(
                run_simulation(
                    user_system_prompt,
                    language,
                    gender,
                    evaluators,
                    simulation_output_dir,
                    interrupt_probability=interrupt_probability,
                    port=port,
                    agent_speaks_first=agent_speaks_first,
                    max_turns=max_turns,
                    tools=config.get("tools", []),
                )
            )
            simulation_tasks = [bot_task, sim_task]
            done, pending = await asyncio.wait(
                simulation_tasks, timeout=EVAL_TIMEOUT_SECS
            )
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
            if sim_task in done:
                if sim_task.cancelled():
                    # Simulation was cancelled (likely due to websocket disconnect from error)
                    # Check if bot_task has an exception that caused this
                    if bot_task in done and not bot_task.cancelled():
                        # This will raise if bot_task failed with an exception
                        bot_task.result()
                    raise RuntimeError("Simulation task was cancelled unexpectedly")
                else:
                    simulation_result = sim_task.result()
        except Exception as e:
            raise e
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

            try:
                logger.remove(log_file_id)
            except ValueError:
                pass  # Handler was already removed
            provider_log_file.reset(judge_log_token)
            # Clean up the print logger for this simulation using the unique run ID
            cleanup_print_logger(simulation_run_id)

    # Combine audio chunks for each turn into single turn files, then combine all into conversation.wav
    audio_dir = os.path.join(simulation_output_dir, "audios")
    conversation_audio_path = os.path.join(simulation_output_dir, "conversation.wav")
    if os.path.exists(audio_dir):
        # First, combine chunks for each turn (e.g., 0_bot_0.wav, 0_bot_1.wav -> 0_bot.wav)
        combine_turn_audio_chunks(audio_dir)
        log_and_print(f"Combined turn audio chunks in {audio_dir}")
        # Then combine all turn files into conversation.wav, using transcript for correct ordering
        transcript_path = os.path.join(simulation_output_dir, TRANSCRIPT_FILE_NAME)
        combine_audio_files(audio_dir, conversation_audio_path, transcript_path)
        log_and_print(f"Combined audio saved to {conversation_audio_path}")

    # Return metrics for aggregation
    if simulation_result:
        sim_metrics_row = {"name": simulation_name}

        # Evaluation criteria metrics (value works for both binary 0/1 and rating score)
        for eval_result in simulation_result.get("evaluation_results", []):
            criterion_name = eval_result["name"]
            sim_metrics_row[criterion_name] = float(eval_result["value"])

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
    # Remove default loguru handler (stderr) to prevent all logs from showing on terminal
    # This is done once at startup, not per-simulation
    logger.remove()

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
        "-n",
        "--parallel",
        type=int,
        default=1,
        help="Number of simulations to run in parallel",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    try:
        require_simulation_evaluators(config.get("evaluators"))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    write_evaluator_config(args.output_dir, config["evaluators"])

    # Mapping from interruption_sensitivity labels to probabilities
    interrupt_sensitivity_map = {
        "none": 0,
        "low": 0.25,
        "medium": 0.5,
        "high": 0.8,
    }

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
                output_dir=args.output_dir,
                interrupt_sensitivity_map=interrupt_sensitivity_map,
            )
            tasks.append(task)

    # Run all tasks with controlled parallelism
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregated metrics across all simulations
    all_simulation_metrics = []
    metrics_by_criterion = defaultdict(list)
    stt_llm_judge_scores = []

    # Collect metrics from results
    failed_simulations = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Simulation failed with error: {result}")
            failed_simulations.append(result)
            continue

        sim_metrics_row, evaluation_results, stt_judge = result
        if sim_metrics_row is None:
            continue

        all_simulation_metrics.append(sim_metrics_row)

        # Evaluation criteria metrics (value works for both binary 0/1 and rating score)
        for eval_result in evaluation_results:
            criterion_name = eval_result["name"]
            metrics_by_criterion[criterion_name].append(float(eval_result["value"]))

        # STT LLM judge score
        if stt_judge and "score" in stt_judge:
            stt_llm_judge_scores.append(stt_judge["score"])

    # Compute and save aggregated metrics
    metrics_summary = {}

    # Track criterion types and scale bounds
    criterion_types: dict = {}
    criterion_ids: dict = {}
    criterion_scales: dict = {}
    for result in results:
        if isinstance(result, Exception) or result is None:
            continue
        _, evaluation_results, _ = result
        for eval_result in evaluation_results or []:
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

    # Aggregate evaluation criteria metrics
    for criterion_name, values in metrics_by_criterion.items():
        metrics_summary[criterion_name] = summarize_metric_distribution(
            values,
            metric_type=criterion_types.get(criterion_name, "binary"),
            scale=criterion_scales.get(criterion_name),
            evaluator_id=criterion_ids.get(criterion_name),
        )

    # Aggregate STT LLM judge scores
    if stt_llm_judge_scores:
        metrics_summary["stt_llm_judge"] = summarize_metric_distribution(
            stt_llm_judge_scores
        )

    # Save overall results.csv
    if all_simulation_metrics:
        df = pd.DataFrame(all_simulation_metrics)
        df.to_csv(join(args.output_dir, "results.csv"), index=False)

    # Save overall metrics.json
    with open(join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    if failed_simulations:
        print(f"\n\033[31m{len(failed_simulations)} simulation(s) failed:\033[0m")
        for err in failed_simulations:
            print(f"  \033[31m- {err}\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
