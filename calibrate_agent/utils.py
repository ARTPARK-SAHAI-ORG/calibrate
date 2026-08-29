import asyncio
import io
import json
import logging
import os
import struct
import threading
import wave
from collections import defaultdict
from contextvars import ContextVar
from pathlib import Path
from typing import Literal, Optional, Any

import aiofiles
import aiohttp
import backoff
import numpy as np
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    InputTransportMessageFrame,
    TTSAudioRawFrame,
    ErrorFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transcriptions.language import Language

# Context variable to track current execution context (BOT or EVAL)
current_context: ContextVar[str] = ContextVar("current_context", default="UNKNOWN")


_usd_to_inr_cache: float | None = None

# Live USD-base FX sources (no API key). Both return {"rates": {"INR": <rate>}}.
# Tried in order each attempt; a second source guards against one being down or
# rate-limiting a given IP.
_FX_ENDPOINTS = (
    "https://api.frankfurter.dev/v1/latest?base=USD&symbols=INR",
    "https://open.er-api.com/v6/latest/USD",
)


@backoff.on_exception(backoff.expo, Exception, max_tries=4)
def _fetch_usd_to_inr() -> float:
    import urllib.request

    last_error: Exception | None = None
    for url in _FX_ENDPOINTS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "calibrate-agent"})
            with urllib.request.urlopen(request, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
            rate = float(data["rates"]["INR"])
            if rate > 0:
                return rate
            last_error = ValueError(f"Non-positive USD→INR rate from FX API: {rate}")
        except Exception as error:  # try the next source before retrying
            last_error = error
    raise last_error or RuntimeError("No FX source returned a USD→INR rate")


def get_usd_to_inr_rate() -> float:
    """Return the USD→INR rate (INR per 1 USD), fetched live and cached.

    Fetched once per process from Frankfurter (ECB reference rates, no API key)
    with exponential-backoff retries, and cached. Raises if the live rate cannot
    be fetched — there is no hardcoded fallback, so a reported INR→USD cost
    always reflects a real exchange rate. Used to convert INR-billed provider
    prices (e.g. Sarvam) to USD for cost reporting.
    """
    global _usd_to_inr_cache
    if _usd_to_inr_cache is None:
        _usd_to_inr_cache = _fetch_usd_to_inr()
    return _usd_to_inr_cache


def get_audio_duration_seconds(audio_path: str | Path) -> float | None:
    """Return audio duration in seconds, or None when it cannot be read.

    Uses pydub (``ffmpeg``) so any format the transcription path can decode —
    WAV, MP3, etc. — yields a duration, keeping cost coverage in sync with the
    files actually transcribed. Logs a warning when a file can't be read.
    """
    try:
        from pydub import AudioSegment

        return AudioSegment.from_file(audio_path).duration_seconds
    except Exception as e:
        logger.warning(f"Could not read audio duration for {audio_path}: {e}")
        return None


def patch_langfuse_trace(trace_name: str):
    from pipecat.utils.tracing import service_decorators

    original = service_decorators.add_llm_span_attributes
    first_call = [True]

    def patched(span, *args, **kwargs):
        original(span, *args, **kwargs)

        if first_call[0]:
            span.set_attribute("langfuse.trace.name", trace_name)
            first_call[0] = False

        # Set the input of the first LLM call as the trace input
        if first_call[0] and kwargs.get("messages"):
            span.set_attribute("langfuse.trace.input", kwargs["messages"])
            first_call[0] = False

        # Set the output of each LLM call as the trace output (last one wins)
        orig_set = span.set_attribute

        def new_set(key, value):
            orig_set(key, value)
            if key == "output":
                orig_set("langfuse.trace.output", value)

        span.set_attribute = new_set

    service_decorators.add_llm_span_attributes = patched


def add_default_source(record):
    """Add default source if not present in extra"""
    if "source" not in record["extra"]:
        context = current_context.get()
        record["extra"]["source"] = f"{context}-SYS"
    return True


# Global print logger instance (for backwards compatibility with single simulation)
_print_logger: Optional[logging.Logger] = None

# Thread-local storage for per-simulation print loggers
_simulation_print_loggers: dict[str, logging.Logger] = {}

# Context variable for current simulation name (for log_and_print to know which logger to use)
current_simulation_name: ContextVar[str] = ContextVar(
    "current_simulation_name", default=""
)


def configure_print_logger(
    log_path: str,
    logger_name: str = "print_logger",
    simulation_name: str = "",
):
    """Configure a dedicated logger for console print mirroring.

    Args:
        log_path: Path to the log file
        logger_name: Name for the logger instance (default: "print_logger")
        simulation_name: Unique name for this simulation (for parallel execution)
    """
    global _print_logger

    # Use unique logger name for parallel simulations
    unique_logger_name = (
        f"{logger_name}_{simulation_name}" if simulation_name else logger_name
    )
    sim_logger = logging.getLogger(unique_logger_name)
    sim_logger.setLevel(logging.INFO)
    sim_logger.propagate = False

    for handler in list(sim_logger.handlers):
        sim_logger.removeHandler(handler)

    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    sim_logger.addHandler(handler)

    if simulation_name:
        _simulation_print_loggers[simulation_name] = sim_logger
        current_simulation_name.set(simulation_name)
    else:
        _print_logger = sim_logger

    return sim_logger


def cleanup_print_logger(simulation_name: str = ""):
    """Clean up the print logger for a simulation.

    Args:
        simulation_name: Name of the simulation to clean up
    """
    if simulation_name and simulation_name in _simulation_print_loggers:
        sim_logger = _simulation_print_loggers.pop(simulation_name)
        for handler in list(sim_logger.handlers):
            handler.close()
            sim_logger.removeHandler(handler)


def log_and_print(
    message: object = "", use_loguru: bool = True, simulation_name: str = ""
):
    """Print to stdout and mirror the message to both loguru and file logger.

    Args:
        message: Message to print and log
        use_loguru: Whether to also log via loguru (default: True)
        simulation_name: Override simulation name (uses context var if not provided)
    """
    text = str(message)
    print(text, flush=True)

    if use_loguru:
        # logger.info will inherit the simulation context from logger.contextualize()
        logger.info(text)

    # Determine which print logger to use
    sim_name = simulation_name or current_simulation_name.get()
    if sim_name and sim_name in _simulation_print_loggers:
        _simulation_print_loggers[sim_name].info(text)
    elif _print_logger:
        _print_logger.info(text)


# =============================================================================
# Per-provider Logging (shared by STT and TTS evaluations)
# =============================================================================
#
# A single ``logs`` file is written per provider. Everything printed to the
# terminal during that provider's run is mirrored into this file. There is no
# separate ``results.log`` and no loguru sink — ``provider_log`` is the only
# hook used inside the STT/TTS evaluation flows.

provider_log_file: ContextVar[Optional[str]] = ContextVar(
    "provider_log_file", default=None
)


def provider_log(message: object = "", *, to_terminal: bool = True) -> None:
    """Append ``message`` to the per-provider log file and (by default) print it.

    The active log file is resolved from ``provider_log_file`` (a
    ``ContextVar``), so each provider running in parallel writes to its own
    file without cross-contamination.
    """
    text = str(message)
    if to_terminal:
        print(text, flush=True)
    log_path = provider_log_file.get()
    if log_path:
        with open(log_path, "a") as f:
            f.write(text + "\n")


def apply_debug_limit(
    items: list,
    debug: bool,
    debug_count: int,
    *,
    label: str = "test cases",
) -> list:
    """Truncate ``items`` to the first ``debug_count`` when ``debug`` is on.

    Mirrors the STT/TTS ``--debug`` behaviour for the LLM commands: a quick
    smoke-run mode that limits how many items are evaluated. Returns the
    (possibly truncated) list and prints a banner when truncation happens.
    """
    if not debug or not items:
        return items
    n = min(debug_count, len(items))
    print(
        f"\033[93mrunning in debug mode: using first {n} {label} for evaluation\033[0m"
    )
    return items[:n]


# Serializes concurrent judge-log writes within the process so two judges
# running at once never split each other's entry.
_judge_log_lock = threading.Lock()


def log_judge_io(
    *,
    evaluator: str,
    model: str,
    system_prompt: str,
    user_input: str,
    output: object,
) -> None:
    """Append one judge LLM call's input/output to the active run log file.

    The whole entry is written as a **single atomic append** so concurrent
    writers (other judges running in parallel, or the run's loguru sink sharing
    the same file) can never interleave a judge's input and output. Never
    prints to the terminal. No-op when no log file is bound to the current
    context, so SDK callers outside a run are unaffected. Used by the judge
    calls in :mod:`calibrate_agent.judges` so every module (LLM, STT, TTS,
    simulation) captures judge prompts and responses locally, independent of
    Langfuse.
    """
    log_path = provider_log_file.get()
    if log_path is None:
        return
    block = (
        "──── judge call ────\n"
        f"evaluator: {evaluator}\n"
        f"model: {model}\n"
        f"system_prompt:\n{system_prompt}\n"
        f"input:\n{user_input}\n"
        f"output: {output}\n"
        "────────────────────\n"
    )
    data = block.encode("utf-8", errors="replace")
    # Single O_APPEND write: the kernel appends each write atomically, and the
    # lock guards against torn writes if a block ever exceeds the atomic-write
    # size or another thread logs concurrently.
    with _judge_log_lock:
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)


class StreamTee:
    """Mirror writes to the original stream and a log file.

    Used by STT/TTS benchmarks to capture everything printed to
    ``stdout``/``stderr`` during a run into an output-dir-level ``logs`` file,
    without altering what the user sees on the terminal.
    """

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        self._log_file.write(data)
        return len(data) if isinstance(data, str) else 0

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    def isatty(self):
        # Preserve TTY semantics so tqdm keeps rendering as a progress bar
        return getattr(self._original, "isatty", lambda: False)()

    def __getattr__(self, item):
        return getattr(self._original, item)


async def save_audio_chunk(
    path: str, audio_chunk: bytes, sample_rate: int, num_channels: int
):
    """Save or append audio data to a WAV file.

    Args:
        path: Path to the audio file
        audio_chunk: Raw audio bytes to save
        sample_rate: Audio sample rate
        num_channels: Number of audio channels
    """
    if len(audio_chunk) == 0:
        logger.warning(f"There's no audio to save for {path}")
        return

    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists():
        # log_and_print(f"\033[92mCreating new audio file at {filepath}\033[0m")
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_chunk)
            async with aiofiles.open(filepath, "wb") as file:
                await file.write(buffer.getvalue())
    else:
        # log_and_print(f"\033[92mAppending audio chunk to {filepath}\033[0m")
        async with aiofiles.open(filepath, "rb+") as file:
            current_size = await file.seek(0, os.SEEK_END)
            if current_size < 44:
                logger.error(
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


def combine_turn_audio_chunks_for_turn(audio_dir: str, turn_index: int) -> bool:
    """Combine audio chunks for a specific turn into single turn audio files.

    Groups files like {turn_index}_{role}_{chunk_index}.wav and combines them
    into {turn_index}_{role}.wav, then deletes the original chunks.

    Args:
        audio_dir: Directory containing the audio chunk files
        turn_index: The specific turn index to combine chunks for

    Returns:
        True if successful, False otherwise
    """
    import glob
    import re

    # Pattern to match chunk files for the specific turn: {turn_index}_{role}_{chunk_index}.wav
    chunk_pattern = re.compile(rf"^{turn_index}_(bot|user)_(\d+)\.wav$")

    audio_files = glob.glob(os.path.join(audio_dir, f"{turn_index}_*_*.wav"))

    if not audio_files:
        logger.info(f"No audio chunks found for turn {turn_index} in {audio_dir}")
        return True

    # Group files by role
    role_groups = defaultdict(list)
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        match = chunk_pattern.match(filename)
        if match:
            role = match.group(1)
            chunk_index = int(match.group(2))
            role_groups[role].append((chunk_index, audio_file))

    if not role_groups:
        logger.info(f"No chunk files found to combine for turn {turn_index}")
        return True

    # Combine each role group
    for role, chunks in role_groups.items():
        # Sort by chunk index
        chunks.sort(key=lambda x: x[0])
        chunk_files = [f for _, f in chunks]

        output_path = os.path.join(audio_dir, f"{turn_index}_{role}.wav")

        # Read and combine audio data
        combined_audio = b""
        sample_rate = None
        num_channels = None
        sample_width = None

        for chunk_file in chunk_files:
            try:
                with wave.open(chunk_file, "rb") as wf:
                    if sample_rate is None:
                        sample_rate = wf.getframerate()
                        num_channels = wf.getnchannels()
                        sample_width = wf.getsampwidth()
                    else:
                        # Verify audio parameters match
                        if (
                            wf.getframerate() != sample_rate
                            or wf.getnchannels() != num_channels
                            or wf.getsampwidth() != sample_width
                        ):
                            logger.warning(
                                f"Audio parameters mismatch in {chunk_file}, skipping"
                            )
                            continue

                    combined_audio += wf.readframes(wf.getnframes())
            except Exception as e:
                logger.error(f"Error reading {chunk_file}: {e}")
                continue

        if not combined_audio or sample_rate is None:
            logger.warning(f"No valid audio data for turn {turn_index} {role}")
            continue

        # Write combined audio
        try:
            with wave.open(output_path, "wb") as wf:
                wf.setsampwidth(sample_width)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(combined_audio)
            logger.info(f"Combined turn audio saved to {output_path}")

            # Delete the original chunk files
            for chunk_file in chunk_files:
                try:
                    os.remove(chunk_file)
                except Exception as e:
                    logger.warning(f"Failed to delete chunk file {chunk_file}: {e}")

        except Exception as e:
            logger.error(f"Error writing combined turn audio: {e}")
            continue

    return True


def combine_turn_audio_chunks(audio_dir: str) -> bool:
    """Combine audio chunks for each turn into single turn audio files.

    Groups files like {turn_index}_{role}_{chunk_index}.wav and combines them
    into {turn_index}_{role}.wav, then deletes the original chunks.

    Args:
        audio_dir: Directory containing the audio chunk files

    Returns:
        True if successful, False otherwise
    """
    import glob
    import re
    from collections import defaultdict

    # Pattern to match chunk files: {turn_index}_{role}_{chunk_index}.wav
    chunk_pattern = re.compile(r"^(\d+)_(bot|user)_(\d+)\.wav$")

    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))

    if not audio_files:
        logger.warning(f"No audio files found in {audio_dir}")
        return False

    # Group files by turn_index and role
    turn_groups = defaultdict(list)
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        match = chunk_pattern.match(filename)
        if match:
            turn_index = int(match.group(1))
            role = match.group(2)
            chunk_index = int(match.group(3))
            turn_groups[(turn_index, role)].append((chunk_index, audio_file))

    if not turn_groups:
        logger.info("No chunk files found to combine")
        return True

    # Combine each group
    for (turn_index, role), chunks in turn_groups.items():
        # Sort by chunk index
        chunks.sort(key=lambda x: x[0])
        chunk_files = [f for _, f in chunks]

        output_path = os.path.join(audio_dir, f"{turn_index}_{role}.wav")

        # Read and combine audio data
        combined_audio = b""
        sample_rate = None
        num_channels = None
        sample_width = None

        for chunk_file in chunk_files:
            try:
                with wave.open(chunk_file, "rb") as wf:
                    if sample_rate is None:
                        sample_rate = wf.getframerate()
                        num_channels = wf.getnchannels()
                        sample_width = wf.getsampwidth()
                    else:
                        # Verify audio parameters match
                        if (
                            wf.getframerate() != sample_rate
                            or wf.getnchannels() != num_channels
                            or wf.getsampwidth() != sample_width
                        ):
                            logger.warning(
                                f"Audio parameters mismatch in {chunk_file}, skipping"
                            )
                            continue

                    combined_audio += wf.readframes(wf.getnframes())
            except Exception as e:
                logger.error(f"Error reading {chunk_file}: {e}")
                continue

        if not combined_audio or sample_rate is None:
            logger.warning(f"No valid audio data for turn {turn_index} {role}")
            continue

        # Write combined audio
        try:
            with wave.open(output_path, "wb") as wf:
                wf.setsampwidth(sample_width)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(combined_audio)
            logger.info(f"Combined turn audio saved to {output_path}")

            # Delete the original chunk files
            for chunk_file in chunk_files:
                try:
                    os.remove(chunk_file)
                except Exception as e:
                    logger.warning(f"Failed to delete chunk file {chunk_file}: {e}")

        except Exception as e:
            logger.error(f"Error writing combined turn audio: {e}")
            continue

    return True


TRANSCRIPT_FILE_NAME = "transcript.json"


def save_transcript(output_dir: str | Path | None, transcript: list[dict]) -> None:
    """Write conversation messages to transcript.json under output_dir."""
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, TRANSCRIPT_FILE_NAME), "w") as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)


def combine_audio_files(
    audio_dir: str, output_path: str, transcript_path: str = None
) -> bool:
    """Combine all WAV files in a directory into a single conversation WAV file.

    Uses the transcript to determine the correct order of audio files.
    For each content message in the transcript (skipping tool_calls-only messages),
    the corresponding audio file is added in order using a single 1-based line
    index ``N`` that matches ``transcript.json`` order:
      - ``assistant`` content → ``{N}_bot.wav``
      - ``user`` content → ``{N}_user.wav``

    Falls back to sorting by filename if no transcript is provided.

    Args:
        audio_dir: Directory containing the audio files
        output_path: Path to save the combined audio file
        transcript_path: Path to the transcript.json file for ordering

    Returns:
        True if successful, False otherwise
    """
    import glob

    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))

    if not audio_files:
        logger.warning(f"No audio files found in {audio_dir}")
        return False

    sorted_files = []

    if not transcript_path or not os.path.exists(transcript_path):
        raise FileNotFoundError(f"No transcript file found at {transcript_path}")

    with open(transcript_path, "r") as f:
        transcript = json.load(f)

    msg_index = 1

    for msg in transcript:
        # Skip messages that only have tool_calls and no content
        if "content" not in msg or msg.get("content") is None:
            continue

        role = msg.get("role")
        if role == "assistant":
            file_path = os.path.join(audio_dir, f"{msg_index}_bot.wav")
            if os.path.exists(file_path):
                sorted_files.append(file_path)
            else:
                logger.warning(f"Expected audio file not found: {file_path}")
            msg_index += 1
        elif role == "user":
            file_path = os.path.join(audio_dir, f"{msg_index}_user.wav")
            if os.path.exists(file_path):
                sorted_files.append(file_path)
            else:
                logger.warning(f"Expected audio file not found: {file_path}")
            msg_index += 1

    if not sorted_files:
        raise ValueError("No audio files matched from transcript")

    # Read all audio data
    combined_audio = b""
    sample_rate = None
    num_channels = None
    sample_width = None

    for audio_file in sorted_files:
        try:
            with wave.open(audio_file, "rb") as wf:
                if sample_rate is None:
                    sample_rate = wf.getframerate()
                    num_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                else:
                    # Verify audio parameters match
                    if (
                        wf.getframerate() != sample_rate
                        or wf.getnchannels() != num_channels
                        or wf.getsampwidth() != sample_width
                    ):
                        logger.warning(
                            f"Audio parameters mismatch in {audio_file}, skipping"
                        )
                        continue

                combined_audio += wf.readframes(wf.getnframes())
        except Exception as e:
            logger.error(f"Error reading {audio_file}: {e}")
            continue

    if not combined_audio or sample_rate is None:
        logger.error("No valid audio data to combine")
        return False

    # Write combined audio
    try:
        with wave.open(output_path, "wb") as wf:
            wf.setsampwidth(sample_width)
            wf.setnchannels(num_channels)
            wf.setframerate(sample_rate)
            wf.writeframes(combined_audio)
        logger.info(f"Combined audio saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing combined audio: {e}")
        return False


class MetricsLogger(FrameProcessor):
    """Frame processor that logs RTVI metrics (TTFB and processing time)."""

    def __init__(
        self,
        ttfb: defaultdict,
        processing_time: defaultdict,
    ):
        super().__init__(enable_direct_mode=True, name="MetricsLogger")
        self._ttfb = ttfb
        self._processing_time = processing_time

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputTransportMessageFrame):
            message = getattr(frame, "message", {})
            if isinstance(message, dict) and message.get("label") == "rtvi-ai":
                if message.get("type") == "metrics" and message.get("data"):
                    if message.get("data").get("ttfb"):
                        for d in message.get("data").get("ttfb"):
                            if not d.get("value"):
                                continue
                            self._ttfb[d.get("processor")].append(d.get("value"))
                    if message.get("data").get("processing"):
                        for d in message.get("data").get("processing"):
                            if not d.get("value"):
                                continue
                            self._processing_time[d.get("processor")].append(
                                d.get("value")
                            )

        await self.push_frame(frame, direction)


# =============================================================================
# Language Code Utilities
# =============================================================================

# Sarvam supported language codes (Indian languages).
# TTS and STT diverge: STT's saaras:v3 additionally supports Maithili.
SARVAM_LANGUAGE_CODES = {
    "english": "en-IN",
    "hindi": "hi-IN",
    "kannada": "kn-IN",
    "bengali": "bn-IN",
    "malayalam": "ml-IN",
    "marathi": "mr-IN",
    "odia": "od-IN",
    "punjabi": "pa-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "gujarati": "gu-IN",
}

SARVAM_STT_LANGUAGE_CODES = {
    **SARVAM_LANGUAGE_CODES,
    "maithili": "mai-IN",
}

SARVAM_TTS_LANGUAGE_CODES = SARVAM_LANGUAGE_CODES

# Default language codes (ISO 639-1)
DEFAULT_LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hi",
    "kannada": "kn",
    "bengali": "bn",
    "malayalam": "ml",
    "marathi": "mr",
    "odia": "od",
    "punjabi": "pa",
    "tamil": "ta",
    "telugu": "te",
    "gujarati": "gu",
}

# =============================================================================
# STT Provider Language Codes
# =============================================================================

# Deepgram STT supported language codes (STT only)
DEEPGRAM_STT_LANGUAGE_CODES = {
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "finnish": "fi",
    "flemish": "nl-BE",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hindi": "hi",
    "hungarian": "hu",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "kannada": "kn",
    "korean": "ko",
    "latvian": "lv",
    "lithuanian": "lt",
    "macedonian": "mk",
    "malay": "ms",
    "marathi": "mr",
    "norwegian": "no",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "spanish": "es",
    "swedish": "sv",
    "tagalog": "tl",
    "tamil": "ta",
    "telugu": "te",
    "turkish": "tr",
    "ukrainian": "uk",
    "vietnamese": "vi",
}

# OpenAI STT (Whisper) supported language codes (ISO 639-1)
OPENAI_STT_LANGUAGE_CODES = {
    "afrikaans": "af",
    "arabic": "ar",
    "armenian": "hy",
    "azerbaijani": "az",
    "belarusian": "be",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "chinese": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "german": "de",
    "greek": "el",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "kannada": "kn",
    "kazakh": "kk",
    "korean": "ko",
    "latvian": "lv",
    "lithuanian": "lt",
    "macedonian": "mk",
    "malay": "ms",
    "marathi": "mr",
    "maori": "mi",
    "nepali": "ne",
    "norwegian": "no",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "spanish": "es",
    "swahili": "sw",
    "swedish": "sv",
    "tagalog": "tl",
    "tamil": "ta",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "vietnamese": "vi",
    "welsh": "cy",
}

# Groq STT (Whisper) — full multilingual language set
GROQ_STT_LANGUAGE_CODES = {
    "afrikaans": "af",
    "albanian": "sq",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "assamese": "as",
    "azerbaijani": "az",
    "bashkir": "ba",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "breton": "br",
    "bulgarian": "bg",
    "burmese": "my",
    "catalan": "ca",
    "cantonese": "yue",
    "chinese": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "faroese": "fo",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "haitian": "ht",
    "haitian creole": "ht",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "korean": "ko",
    "lao": "lo",
    "latin": "la",
    "latvian": "lv",
    "lingala": "ln",
    "lithuanian": "lt",
    "luxembourgish": "lb",
    "macedonian": "mk",
    "malagasy": "mg",
    "malay": "ms",
    "malayalam": "ml",
    "maltese": "mt",
    "maori": "mi",
    "marathi": "mr",
    "mongolian": "mn",
    "myanmar": "my",
    "nepali": "ne",
    "norwegian": "no",
    "nynorsk": "nn",
    "occitan": "oc",
    "pashto": "ps",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "serbian": "sr",
    "shona": "sn",
    "sindhi": "sd",
    "sinhala": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tagalog": "tl",
    "tajik": "tg",
    "tamil": "ta",
    "tatar": "tt",
    "telugu": "te",
    "thai": "th",
    "tibetan": "bo",
    "turkish": "tr",
    "turkmen": "tk",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "yiddish": "yi",
    "yoruba": "yo",
    # Aliases
    "castilian": "es",
    "flemish": "nl",
    "letzeburgesch": "lb",
    "mandarin": "zh",
    "moldavian": "ro",
    "moldovan": "ro",
    "panjabi": "pa",
    "pushto": "ps",
    "sinhalese": "si",
    "valencian": "ca",
}

# Cartesia STT supported language codes (Whisper-based, very extensive)
CARTESIA_STT_LANGUAGE_CODES = {
    "afrikaans": "af",
    "albanian": "sq",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "assamese": "as",
    "azerbaijani": "az",
    "bashkir": "ba",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "breton": "br",
    "bulgarian": "bg",
    "burmese": "my",
    "catalan": "ca",
    "chinese": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "faroese": "fo",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "haitian": "ht",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "korean": "ko",
    "lao": "lo",
    "latin": "la",
    "latvian": "lv",
    "lingala": "ln",
    "lithuanian": "lt",
    "luxembourgish": "lb",
    "macedonian": "mk",
    "malagasy": "mg",
    "malay": "ms",
    "malayalam": "ml",
    "maltese": "mt",
    "maori": "mi",
    "marathi": "mr",
    "mongolian": "mn",
    "nepali": "ne",
    "norwegian": "no",
    "nynorsk": "nn",
    "occitan": "oc",
    "pashto": "ps",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "serbian": "sr",
    "shona": "sn",
    "sindhi": "sd",
    "sinhala": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tagalog": "tl",
    "tajik": "tg",
    "tamil": "ta",
    "tatar": "tt",
    "telugu": "te",
    "thai": "th",
    "tibetan": "bo",
    "turkish": "tr",
    "turkmen": "tk",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "yiddish": "yi",
    "yoruba": "yo",
    "cantonese": "yue",
}

# Smallest STT supported language codes
SMALLEST_STT_LANGUAGE_CODES = {
    "bengali": "bn",
    "bulgarian": "bg",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "gujarati": "gu",
    "hindi": "hi",
    "hungarian": "hu",
    "italian": "it",
    "kannada": "kn",
    "latvian": "lv",
    "lithuanian": "lt",
    "malayalam": "ml",
    "maltese": "mt",
    "marathi": "mr",
    "odia": "or",
    "polish": "pl",
    "portuguese": "pt",
    # "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "slovak": "sk",
    "spanish": "es",
    "swedish": "sv",
    "tamil": "ta",
    "telugu": "te",
    "ukrainian": "uk",
}

# Google STT supported language codes (BCP-47)
GOOGLE_STT_LANGUAGE_CODES = {
    "afrikaans": "af-ZA",
    "amharic": "am-ET",
    "arabic": "ar-XA",
    "armenian": "hy-AM",
    "assamese": "as-IN",
    "azerbaijani": "az-AZ",
    "bengali": "bn-IN",
    "bulgarian": "bg-BG",
    "burmese": "my-MM",
    "catalan": "ca-ES",
    "chinese": "cmn-Hans-CN",
    "croatian": "hr-HR",
    "czech": "cs-CZ",
    "danish": "da-DK",
    "dutch": "nl-NL",
    "english": "en-US",
    "estonian": "et-EE",
    "filipino": "fil-PH",
    "finnish": "fi-FI",
    "french": "fr-FR",
    "galician": "gl-ES",
    "georgian": "ka-GE",
    "german": "de-DE",
    "greek": "el-GR",
    "gujarati": "gu-IN",
    "hebrew": "iw-IL",
    "hindi": "hi-IN",
    "hungarian": "hu-HU",
    "icelandic": "is-IS",
    "indonesian": "id-ID",
    "italian": "it-IT",
    "japanese": "ja-JP",
    "javanese": "jv-ID",
    "kannada": "kn-IN",
    "kazakh": "kk-KZ",
    "khmer": "km-KH",
    "korean": "ko-KR",
    "lao": "lo-LA",
    "latvian": "lv-LV",
    "lithuanian": "lt-LT",
    "macedonian": "mk-MK",
    "malay": "ms-MY",
    "malayalam": "ml-IN",
    "marathi": "mr-IN",
    "mongolian": "mn-MN",
    "nepali": "ne-NP",
    "norwegian": "no-NO",
    "odia": "or-IN",
    "persian": "fa-IR",
    "polish": "pl-PL",
    "portuguese": "pt-BR",
    "punjabi": "pa-Guru-IN",
    "romanian": "ro-RO",
    "russian": "ru-RU",
    "sepedi": "nso-ZA",
    "serbian": "sr-RS",
    "sindhi": "sd-IN",
    "slovak": "sk-SK",
    "slovenian": "sl-SI",
    "spanish": "es-ES",
    "swahili": "sw-KE",
    "swedish": "sv-SE",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "thai": "th-TH",
    "turkish": "tr-TR",
    "ukrainian": "uk-UA",
    "urdu": "ur-PK",
    "uzbek": "uz-UZ",
    "vietnamese": "vi-VN",
    "xhosa": "xh-ZA",
    "zulu": "zu-ZA",
}

# ElevenLabs STT supported language codes (ISO 639-3, extensive)
ELEVENLABS_STT_LANGUAGE_CODES = {
    "afrikaans": "afr",
    "amharic": "amh",
    "arabic": "ara",
    "armenian": "hye",
    "assamese": "asm",
    "asturian": "ast",
    "azerbaijani": "aze",
    "belarusian": "bel",
    "bengali": "ben",
    "bosnian": "bos",
    "bulgarian": "bul",
    "burmese": "mya",
    "cantonese": "yue",
    "catalan": "cat",
    "croatian": "hrv",
    "czech": "ces",
    "danish": "dan",
    "dutch": "nld",
    "english": "eng",
    "estonian": "est",
    "filipino": "fil",
    "finnish": "fin",
    "french": "fra",
    "galician": "glg",
    "ganda": "lug",
    "georgian": "kat",
    "german": "deu",
    "greek": "ell",
    "gujarati": "guj",
    "hausa": "hau",
    "hebrew": "heb",
    "hindi": "hin",
    "hungarian": "hun",
    "icelandic": "isl",
    "igbo": "ibo",
    "indonesian": "ind",
    "irish": "gle",
    "italian": "ita",
    "japanese": "jpn",
    "javanese": "jav",
    "kannada": "kan",
    "kazakh": "kaz",
    "khmer": "khm",
    "korean": "kor",
    "kurdish": "kur",
    "kyrgyz": "kir",
    "lao": "lao",
    "latvian": "lav",
    "lithuanian": "lit",
    "luxembourgish": "ltz",
    "macedonian": "mkd",
    "malay": "msa",
    "malayalam": "mal",
    "maltese": "mlt",
    "mandarin": "zho",
    "chinese": "zho",
    "maori": "mri",
    "marathi": "mar",
    "mongolian": "mon",
    "nepali": "nep",
    "northern_sotho": "nso",
    "norwegian": "nor",
    "occitan": "oci",
    "odia": "ori",
    "pashto": "pus",
    "persian": "fas",
    "polish": "pol",
    "portuguese": "por",
    "punjabi": "pan",
    "romanian": "ron",
    "russian": "rus",
    "serbian": "srp",
    "shona": "sna",
    "sindhi": "snd",
    "slovak": "slk",
    "slovenian": "slv",
    "somali": "som",
    "spanish": "spa",
    "swahili": "swa",
    "swedish": "swe",
    "tajik": "tgk",
    "tamil": "tam",
    "telugu": "tel",
    "thai": "tha",
    "turkish": "tur",
    "ukrainian": "ukr",
    "urdu": "urd",
    "uzbek": "uzb",
    "vietnamese": "vie",
    "welsh": "cym",
    "wolof": "wol",
    "xhosa": "xho",
    "yoruba": "yor",
    "zulu": "zul",
}

# =============================================================================
# TTS Provider Language Codes
# =============================================================================

# Cartesia TTS supported language codes (more limited than STT)
CARTESIA_TTS_LANGUAGE_CODES = {
    "arabic": "ar",
    "bengali": "bn",
    "bulgarian": "bg",
    "chinese": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "finnish": "fi",
    "french": "fr",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "kannada": "kn",
    "korean": "ko",
    "malayalam": "ml",
    "marathi": "mr",
    "norwegian": "no",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "slovak": "sk",
    "spanish": "es",
    "swedish": "sv",
    "tagalog": "tl",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "vietnamese": "vi",
}

# Google TTS supported language codes (BCP-47, more limited than STT)
GOOGLE_TTS_LANGUAGE_CODES = {
    "arabic": "ar-XA",
    "bengali": "bn-IN",
    "bulgarian": "bg-BG",
    "cantonese": "yue-HK",
    "chinese": "cmn-CN",
    "croatian": "hr-HR",
    "czech": "cs-CZ",
    "danish": "da-DK",
    "dutch": "nl-NL",
    "english": "en-US",
    "estonian": "et-EE",
    "finnish": "fi-FI",
    "french": "fr-FR",
    "german": "de-DE",
    "greek": "el-GR",
    "gujarati": "gu-IN",
    "hebrew": "he-IL",
    "hindi": "hi-IN",
    "hungarian": "hu-HU",
    "indonesian": "id-ID",
    "italian": "it-IT",
    "japanese": "ja-JP",
    "kannada": "kn-IN",
    "korean": "ko-KR",
    "latvian": "lv-LV",
    "lithuanian": "lt-LT",
    "malayalam": "ml-IN",
    "marathi": "mr-IN",
    "norwegian": "nb-NO",
    "odia": "or-IN",
    "polish": "pl-PL",
    "portuguese": "pt-BR",
    "punjabi": "pa-IN",
    "romanian": "ro-RO",
    "russian": "ru-RU",
    "serbian": "sr-RS",
    "sindhi": "sd-IN",
    "slovak": "sk-SK",
    "slovenian": "sl-SI",
    "spanish": "es-ES",
    "swahili": "sw-KE",
    "swedish": "sv-SE",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "thai": "th-TH",
    "turkish": "tr-TR",
    "ukrainian": "uk-UA",
    "urdu": "ur-IN",
    "vietnamese": "vi-VN",
}

# ElevenLabs TTS supported language codes (more limited than STT)
ELEVENLABS_TTS_LANGUAGE_CODES = {
    "arabic": "ara",
    "bulgarian": "bul",
    "chinese": "zho",
    "croatian": "hrv",
    "czech": "ces",
    "danish": "dan",
    "dutch": "nld",
    "english": "eng",
    "filipino": "fil",
    "finnish": "fin",
    "french": "fra",
    "german": "deu",
    "greek": "ell",
    "hindi": "hin",
    "indonesian": "ind",
    "italian": "ita",
    "japanese": "jpn",
    "korean": "kor",
    "malay": "msa",
    "polish": "pol",
    "portuguese": "por",
    "romanian": "ron",
    "russian": "rus",
    "sindhi": "sd",
    "slovak": "slk",
    "spanish": "spa",
    "swedish": "swe",
    "tamil": "tam",
    "turkish": "tur",
    "ukrainian": "ukr",
}

# OpenAI TTS supported language codes (similar to STT)
OPENAI_TTS_LANGUAGE_CODES = OPENAI_STT_LANGUAGE_CODES

# Groq TTS supported language codes (only English for Orpheus)
GROQ_TTS_LANGUAGE_CODES = {
    "english": "en",
}

# Smallest TTS supported language codes (same as STT)
SMALLEST_TTS_LANGUAGE_CODES = SMALLEST_STT_LANGUAGE_CODES

# =============================================================================
# Legacy aliases for backwards compatibility
# =============================================================================
DEEPGRAM_LANGUAGE_CODES = DEEPGRAM_STT_LANGUAGE_CODES
OPENAI_LANGUAGE_CODES = OPENAI_STT_LANGUAGE_CODES
GROQ_LANGUAGE_CODES = GROQ_STT_LANGUAGE_CODES
CARTESIA_LANGUAGE_CODES = CARTESIA_STT_LANGUAGE_CODES
GOOGLE_LANGUAGE_CODES = GOOGLE_STT_LANGUAGE_CODES
ELEVENLABS_LANGUAGE_CODES = ELEVENLABS_STT_LANGUAGE_CODES
SMALLEST_LANGUAGE_CODES = SMALLEST_STT_LANGUAGE_CODES

# Gemini is a Google model and uses the same BCP-47 language codes, so it reuses
# Google's maps. Benchmark-only (calibrate_agent/{stt,tts}/eval.py via
# google-genai) — there is no cascaded pipecat Gemini STT/TTS service, so it is
# not wired into create_stt_service / create_tts_service.
GEMINI_STT_LANGUAGE_CODES = GOOGLE_STT_LANGUAGE_CODES
GEMINI_TTS_LANGUAGE_CODES = GOOGLE_TTS_LANGUAGE_CODES


def get_stt_language_code(language: str, provider: str) -> str:
    """Get the appropriate language code string for an STT provider.

    Args:
        language: The language name (e.g., english, hindi, kannada)
        provider: The STT provider name (sarvam, google, deepgram, openai, etc.)

    Returns:
        The appropriate language code string for the STT provider

    Examples:
        >>> get_stt_language_code("hindi", "sarvam")
        'hi-IN'
        >>> get_stt_language_code("hindi", "deepgram")
        'hi'
        >>> get_stt_language_code("english", "google")
        'en-US'
    """
    language = language.lower()

    if provider == "sarvam":
        return SARVAM_STT_LANGUAGE_CODES.get(language, "en-IN")
    elif provider == "google":
        return GOOGLE_STT_LANGUAGE_CODES.get(language, "en-US")
    elif provider == "gemini":
        return GEMINI_STT_LANGUAGE_CODES.get(language, "en-US")
    elif provider == "smallest":
        return SMALLEST_STT_LANGUAGE_CODES.get(language, "multi")
    elif provider == "cartesia":
        return CARTESIA_STT_LANGUAGE_CODES.get(language, "en")
    elif provider == "elevenlabs":
        return ELEVENLABS_STT_LANGUAGE_CODES.get(language, "eng")
    elif provider == "openai":
        return OPENAI_STT_LANGUAGE_CODES.get(language, "en")
    elif provider == "groq":
        return GROQ_STT_LANGUAGE_CODES.get(language, "en")
    elif provider == "deepgram":
        return DEEPGRAM_STT_LANGUAGE_CODES.get(language, "en")

    # Default: use ISO 639-1 codes
    return DEFAULT_LANGUAGE_CODES.get(language, "en")


def get_tts_language_code(language: str, provider: str) -> str:
    """Get the appropriate language code string for a TTS provider.

    Args:
        language: The language name (e.g., english, hindi, kannada)
        provider: The TTS provider name (sarvam, google, cartesia, openai, etc.)

    Returns:
        The appropriate language code string for the TTS provider

    Examples:
        >>> get_tts_language_code("hindi", "sarvam")
        'hi-IN'
        >>> get_tts_language_code("hindi", "cartesia")
        'hi'
        >>> get_tts_language_code("english", "groq")
        'en'
    """
    language = language.lower()

    if provider == "sarvam":
        return SARVAM_TTS_LANGUAGE_CODES.get(language, "en-IN")
    elif provider == "google":
        return GOOGLE_TTS_LANGUAGE_CODES.get(language, "en-US")
    elif provider == "gemini":
        return GEMINI_TTS_LANGUAGE_CODES.get(language, "en-US")
    elif provider == "smallest":
        return SMALLEST_TTS_LANGUAGE_CODES.get(language, "en")
    elif provider == "cartesia":
        return CARTESIA_TTS_LANGUAGE_CODES.get(language, "en")
    elif provider == "elevenlabs":
        return ELEVENLABS_TTS_LANGUAGE_CODES.get(language, "eng")
    elif provider == "openai":
        return OPENAI_TTS_LANGUAGE_CODES.get(language, "en")
    elif provider == "groq":
        return GROQ_TTS_LANGUAGE_CODES.get(language, "en")

    # Default: use ISO 639-1 codes
    return DEFAULT_LANGUAGE_CODES.get(language, "en")


def get_language_code(language: str, provider: str) -> str:
    """Get the appropriate language code string for a provider.

    DEPRECATED: Use get_stt_language_code() or get_tts_language_code() instead.
    This function defaults to STT language codes for backwards compatibility.

    Args:
        language: The language name (english, hindi, kannada, etc.)
        provider: The provider name (sarvam, google, deepgram, openai, etc.)

    Returns:
        The appropriate language code string for the provider
    """
    return get_stt_language_code(language, provider)


def validate_stt_language(language: str, provider: str) -> None:
    """Validate that a language is supported by the given STT provider.

    Args:
        language: The language name (e.g., english, hindi, kannada)
        provider: The STT provider name

    Raises:
        ValueError: If the language is not supported by the provider
    """
    language = language.lower()

    # Map providers to their STT language code dictionaries
    provider_languages = {
        "sarvam": SARVAM_STT_LANGUAGE_CODES,
        "google": GOOGLE_STT_LANGUAGE_CODES,
        "gemini": GEMINI_STT_LANGUAGE_CODES,
        "smallest": SMALLEST_STT_LANGUAGE_CODES,
        "cartesia": CARTESIA_STT_LANGUAGE_CODES,
        "elevenlabs": ELEVENLABS_STT_LANGUAGE_CODES,
        "openai": OPENAI_STT_LANGUAGE_CODES,
        "groq": GROQ_STT_LANGUAGE_CODES,
        "deepgram": DEEPGRAM_STT_LANGUAGE_CODES,
    }

    if provider not in provider_languages:
        raise ValueError(f"Unknown STT provider: {provider}")

    supported_languages = provider_languages[provider]

    if language not in supported_languages:
        supported_list = sorted(supported_languages.keys())
        raise ValueError(
            f"Language '{language}' is not supported by {provider} STT.\n"
            f"Supported languages for {provider} STT: {', '.join(supported_list)}"
        )


def validate_tts_language(language: str, provider: str) -> None:
    """Validate that a language is supported by the given TTS provider.

    Args:
        language: The language name (e.g., english, hindi, kannada)
        provider: The TTS provider name

    Raises:
        ValueError: If the language is not supported by the provider
    """
    language = language.lower()

    # Map providers to their TTS language code dictionaries
    provider_languages = {
        "sarvam": SARVAM_TTS_LANGUAGE_CODES,
        "google": GOOGLE_TTS_LANGUAGE_CODES,
        "gemini": GEMINI_TTS_LANGUAGE_CODES,
        "cartesia": CARTESIA_TTS_LANGUAGE_CODES,
        "elevenlabs": ELEVENLABS_TTS_LANGUAGE_CODES,
        "openai": OPENAI_TTS_LANGUAGE_CODES,
        "groq": GROQ_TTS_LANGUAGE_CODES,
        "smallest": SMALLEST_TTS_LANGUAGE_CODES,
    }

    if provider not in provider_languages:
        raise ValueError(f"Unknown TTS provider: {provider}")

    supported_languages = provider_languages[provider]

    if language not in supported_languages:
        supported_list = sorted(supported_languages.keys())
        raise ValueError(
            f"Language '{language}' is not supported by {provider} TTS.\n"
            f"Supported languages for {provider} TTS: {', '.join(supported_list)}"
        )


# =============================================================================
# STT/TTS Provider Factory Functions
# =============================================================================


# Base pipecat ``Language`` enum per language name, used by the STT factory
# (create_stt_service) and the pipeline benchmark engine. Mirrors the coverage of
# the direct-path ``get_stt_language_code`` maps so both engines support the same
# 13 languages. Sarvam takes India-regional (``*_IN``) variants; every other
# provider takes the base enum and pipecat's per-service layer translates it.
_STT_LANGUAGE_ENUMS = {
    "english": Language.EN,
    "hindi": Language.HI,
    "kannada": Language.KN,
    "bengali": Language.BN,
    "malayalam": Language.ML,
    "marathi": Language.MR,
    "odia": Language.OR,
    "punjabi": Language.PA,
    "tamil": Language.TA,
    "telugu": Language.TE,
    "gujarati": Language.GU,
    "sindhi": Language.SD,
    "maithili": Language.MAI,
}

_SARVAM_STT_LANGUAGE_ENUMS = {
    "english": Language.EN_IN,
    "hindi": Language.HI_IN,
    "kannada": Language.KN_IN,
    "bengali": Language.BN_IN,
    "malayalam": Language.ML_IN,
    "marathi": Language.MR_IN,
    "odia": Language.OR_IN,
    "punjabi": Language.PA_IN,
    "tamil": Language.TA_IN,
    "telugu": Language.TE_IN,
    "gujarati": Language.GU_IN,
    "sindhi": Language.SD_IN,
    "maithili": Language.MAI_IN,
}


def get_stt_language(
    language: str,
    provider: str,
) -> Language:
    """Get the appropriate Language enum for STT based on language and provider.

    Args:
        language: The language name (e.g. english, hindi, tamil — any of the 13
            supported STT languages).
        provider: The STT provider name

    Returns:
        The appropriate Language enum value (falls back to English for an
        unknown language, matching the direct-path code maps' defaults).
    """
    language = language.lower()
    table = _SARVAM_STT_LANGUAGE_ENUMS if provider == "sarvam" else _STT_LANGUAGE_ENUMS
    return table.get(language, table["english"])


def get_tts_language(
    language: Literal["english", "hindi", "kannada"],
    provider: str,
) -> Language:
    """Get the appropriate Language enum for TTS based on language and provider.

    Args:
        language: The language name (english, hindi, kannada)
        provider: The TTS provider name

    Returns:
        The appropriate Language enum value
    """
    # Sarvam uses regional language codes
    if provider == "sarvam":
        if language == "kannada":
            return Language.KN_IN
        elif language == "hindi":
            return Language.HI_IN
        else:
            return Language.EN_IN

    # Default language codes
    if language == "kannada":
        return Language.KN
    elif language == "hindi":
        return Language.HI
    else:
        return Language.EN


# Per-provider DEFAULT TTS voice, used when a language has no entry in
# TTS_PROVIDER_VOICES_BY_LANGUAGE below. Shared by the live agent
# (create_tts_service) and the benchmark (calibrate_agent/tts/eval.py) through
# get_tts_voice() so a leaderboard reflects the exact voice the agent deploys.
# Guarded by tests/test_utils_factories.py. Google's default embeds the language
# code, built from GOOGLE_TTS_VOICE_FAMILY. Deepgram TTS isn't benchmarked, so
# its voice stays inline in create_tts_service.
TTS_PROVIDER_VOICES = {
    "cartesia": "faf0731e-dfb9-4cfc-8119-259a79b27e12",  # riya
    "openai": "coral",
    "groq": "troy",
    "elevenlabs": "m5qndnI7u4OAdXhH0Mr5",  # Krishna (pre-made voice)
    "sarvam": "aditya",
    "smallest": "aditi",
    # Gemini prebuilt voice; language-agnostic (one voice speaks all languages),
    # so no TTS_PROVIDER_VOICES_BY_LANGUAGE entry is needed.
    "gemini": "Kore",
}

# Google TTS voice family; the full name is "{lang_code}-{family}" (e.g.
# en-US-Chirp3-HD-Achernar), built per language in get_tts_voice. Google needs no
# TTS_PROVIDER_VOICES_BY_LANGUAGE entry — the language code in the name already
# makes it per-language, so this family IS the single source for every language.
GOOGLE_TTS_VOICE_FAMILY = "Chirp3-HD-Achernar"

# Per-language voice OVERRIDES. A (provider, language) entry here wins over the
# provider's TTS_PROVIDER_VOICES default; languages without an entry fall back to
# it. This restores the per-language voices the live agent shipped before voices
# were single-sourced. Read by both sides through get_tts_voice(), so the live
# agent and benchmark still can't drift.
TTS_PROVIDER_VOICES_BY_LANGUAGE = {
    "cartesia": {
        "english": "66c6b81c-ddb7-4892-bdd5-19b5a7be38e7",
        "hindi": "28ca2041-5dda-42df-8123-f58ea9c3da00",
        "kannada": "7c6219d2-e8d2-462c-89d8-7ecba7c75d65",
    },
    "elevenlabs": {
        "english": "90ipbRoKi4CpHXvKVtl0",
        "hindi": "jUjRbhZWoMK4aDciW36V",
        "kannada": "90ipbRoKi4CpHXvKVtl0",  # falls back to the english voice
    },
    "smallest": {
        "english": "aarushi",
        "hindi": "aarushi",
        "kannada": "vijay",
    },
}


def get_tts_voice(provider: str, language: str) -> str:
    """Return the default TTS voice for a provider + language.

    A per-language override in TTS_PROVIDER_VOICES_BY_LANGUAGE wins; otherwise the
    provider's language-agnostic default (Google builds
    "{lang_code}-{GOOGLE_TTS_VOICE_FAMILY}"). Shared by create_tts_service and
    calibrate_agent/tts/eval.py so the live agent and benchmark pick the same voice.
    """
    override = TTS_PROVIDER_VOICES_BY_LANGUAGE.get(provider, {}).get(language)
    if override is not None:
        return override
    if provider == "google":
        return f"{get_tts_language_code(language, 'google')}-{GOOGLE_TTS_VOICE_FAMILY}"
    return TTS_PROVIDER_VOICES[provider]


def get_gemini_api_key(required: bool = True) -> Optional[str]:
    """Return the Gemini API key from GOOGLE_API_KEY.

    GOOGLE_API_KEY is google-genai's canonical variable. Single source for the
    benchmark Gemini STT/TTS paths (calibrate_agent/{stt,tts}/eval.py) and the
    `status` check.

    Raises ValueError when ``required`` and the variable is not set.
    """
    key = os.getenv("GOOGLE_API_KEY")
    if required and not key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    return key


# Single source of truth for per-provider default model names. The live agent
# (create_stt_service / create_tts_service below) AND the benchmarks
# (calibrate_agent/stt/eval.py, calibrate_agent/tts/eval.py) both read these, so a change in
# one place applies to both and a leaderboard always reflects the deployed
# config. See CLAUDE.md ("Keeping live-agent and eval defaults in sync"); the
# parity is guarded by tests/test_utils_factories.py. Sarvam STT uses saaras:v3
# with mode="transcribe" — under pipecat 1.0.0 that routes to the plain STT
# streaming endpoint (use_translate_endpoint=False), matching the benchmark's
# transcribe_sarvam. (Under pipecat 0.0.98 this wasn't possible — its wrapper
# forced saaras onto the translate endpoint with no mode param — which is why it
# was previously excluded.) Providers absent here have no shared model string
# (e.g. Google/Smallest/Deepgram TTS carry the model in the voice name or set no
# model at all).
STT_PROVIDER_MODELS = {
    "deepgram": "nova-3",
    "openai": "gpt-4o-transcribe",
    "groq": "whisper-large-v3-turbo",
    "google": "chirp_3",
    "gemini": "gemini-3.5-flash",
    "cartesia": "ink-whisper",
    "elevenlabs": "scribe_v2_realtime",
    "smallest": "pulse",
    "sarvam": "saaras:v3",
}


def google_stt_model_and_location(
    language: str, model: str | None = None
) -> tuple[str, str]:
    """Return the Google STT (model, location) for a language.

    Sindhi is only served by ``chirp_2`` in ``asia-southeast1``; every other
    language uses the default model in ``us``. Single source of truth shared by
    the live-agent factory (``create_stt_service``) and the direct-path
    benchmark (``stt/eval.py`` ``transcribe_google``) so the two can't drift.
    """
    if str(language).lower() == "sindhi":
        return model or "chirp_2", "asia-southeast1"
    return model or STT_PROVIDER_MODELS["google"], "us"


TTS_PROVIDER_MODELS = {
    "gemini": "gemini-3.1-flash-tts-preview",
    "cartesia": "sonic-3.5",
    "openai": "gpt-4o-mini-tts",
    "groq": "canopylabs/orpheus-v1-english",
    "elevenlabs": "eleven_multilingual_v2",
    "sarvam": "bulbul:v3",
}


def create_stt_service(
    provider: str,
    language: Literal["english", "hindi", "kannada"],
    model: Optional[str] = None,
):
    """Create an STT service instance for the given provider and language.

    Args:
        provider: STT provider name (deepgram, openai, cartesia, google, sarvam, elevenlabs, smallest, groq)
        language: Language for transcription (english, hindi, kannada)
        model: Optional model name (uses default for provider if not specified)

    Returns:
        Configured STT service instance

    Raises:
        ValueError: If provider is not supported
    """
    # Import services here to avoid circular imports
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.openai.stt import OpenAISTTService
    from pipecat.services.google.stt import GoogleSTTService
    from pipecat.services.cartesia.stt import CartesiaSTTService
    from pipecat.services.groq.stt import GroqSTTService
    from pipecat.services.sarvam.stt import SarvamSTTService
    from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
    from pipecat.services.smallest.stt import SmallestSTTService

    stt_language = get_stt_language(language, provider)

    if provider == "deepgram":
        return DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            encoding="linear16",
            settings=DeepgramSTTService.Settings(
                language=stt_language,
                model=STT_PROVIDER_MODELS[provider],
            ),
        )
    elif provider == "sarvam":
        return SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
            mode="transcribe",
            settings=SarvamSTTService.Settings(
                language=stt_language,
                model=STT_PROVIDER_MODELS[provider],
            ),
        )
    elif provider == "elevenlabs":
        return ElevenLabsRealtimeSTTService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            settings=ElevenLabsRealtimeSTTService.Settings(
                language=stt_language,
                model=STT_PROVIDER_MODELS[provider],
            ),
        )
    elif provider == "openai":
        return OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAISTTService.Settings(
                language=stt_language, model=model or STT_PROVIDER_MODELS[provider]
            ),
        )
    elif provider == "cartesia":
        return CartesiaSTTService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            settings=CartesiaSTTService.Settings(
                language=stt_language,
                model=STT_PROVIDER_MODELS[provider],
            ),
        )
    elif provider == "smallest":
        return SmallestSTTService(
            api_key=os.getenv("SMALLEST_API_KEY"),
            settings=SmallestSTTService.Settings(
                language=stt_language,
                model=STT_PROVIDER_MODELS[provider],
            ),
        )
    elif provider == "groq":
        return GroqSTTService(
            api_key=os.getenv("GROQ_API_KEY"),
            settings=GroqSTTService.Settings(
                language=stt_language, model=model or STT_PROVIDER_MODELS[provider]
            ),
        )
    elif provider == "google":
        google_model, google_location = google_stt_model_and_location(language, model)
        return GoogleSTTService(
            sample_rate=16000,
            location=google_location,
            settings=GoogleSTTService.Settings(
                languages=[stt_language],
                model=google_model,
            ),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
    else:
        raise ValueError(f"Unsupported STT provider: {provider}")


def create_tts_service(
    provider: str,
    language: Literal["english", "hindi", "kannada"],
    voice_id: Optional[str] = None,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
):
    """Create a TTS service instance for the given provider and language.

    Args:
        provider: TTS provider name (cartesia, openai, groq, google, elevenlabs, sarvam, smallest)
        language: Language for synthesis (english, hindi, kannada)
        voice_id: Optional custom voice ID (uses default for provider/language if not specified)
        model: Optional model name (uses default for provider if not specified)
        instructions: Optional instructions for OpenAI TTS

    Returns:
        Configured TTS service instance

    Raises:
        ValueError: If provider is not supported
    """
    # Import services here to avoid circular imports
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.openai.tts import OpenAITTSService
    from pipecat.services.groq.tts import GroqTTSService
    from pipecat.services.google.tts import GoogleTTSService
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
    from pipecat.services.sarvam.tts import SarvamTTSService
    from pipecat.services.deepgram.tts import DeepgramTTSService
    from pipecat.services.smallest.tts import SmallestTTSService

    tts_language = get_tts_language(language, provider)

    if provider == "cartesia":
        return CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            settings=CartesiaTTSService.Settings(
                language=tts_language,
                model=model or TTS_PROVIDER_MODELS[provider],
                voice=voice_id or get_tts_voice(provider, language),
            ),
        )
    elif provider == "openai":
        return OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAITTSService.Settings(
                model=model or TTS_PROVIDER_MODELS[provider],
                voice=voice_id or get_tts_voice(provider, language),
                instructions=instructions,
            ),
        )
    elif provider == "groq":
        return GroqTTSService(
            api_key=os.getenv("GROQ_API_KEY"),
            settings=GroqTTSService.Settings(
                model=model or TTS_PROVIDER_MODELS[provider],
                voice=voice_id or get_tts_voice(provider, language),
            ),
        )
    elif provider == "google":
        return GoogleTTSService(
            settings=GoogleTTSService.Settings(
                language=tts_language,
                voice=voice_id or get_tts_voice(provider, language),
            ),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
    elif provider == "elevenlabs":
        return ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            settings=ElevenLabsTTSService.Settings(
                language=tts_language,
                model=TTS_PROVIDER_MODELS[provider],
                voice=voice_id or get_tts_voice(provider, language),
            ),
        )
    elif provider == "sarvam":
        return SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            settings=SarvamTTSService.Settings(
                language=tts_language,
                model=model or TTS_PROVIDER_MODELS[provider],
                voice=voice_id or get_tts_voice(provider, language),
            ),
        )
    elif provider == "deepgram":
        return DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            settings=DeepgramTTSService.Settings(voice=voice_id or "aura-2-andromeda-en"),
        )
    elif provider == "smallest":
        return SmallestTTSService(
            api_key=os.getenv("SMALLEST_API_KEY"),
            settings=SmallestTTSService.Settings(
                voice=voice_id or get_tts_voice(provider, language),
                language=tts_language,
            ),
        )
    else:
        raise ValueError(f"Unsupported TTS provider: {provider}")


def wrap_tts_with_empty_response_retry(
    tts,
    *,
    max_attempts: int = 3,
    backoff_base_secs: float = 0.5,
):
    """Retry a TTS turn that yields no audio, in place, on any pipecat TTS service.

    Some HTTP TTS providers (notably ElevenLabs) intermittently return an HTTP
    200 with an *empty* audio stream: ``run_tts`` iterates zero times, yields no
    ``TTSAudioRawFrame`` and no error, so the turn goes completely silent. In the
    voice-agent simulator that stalls the whole conversation until the pipeline
    idle timeout, leaving a truncated transcript.

    This wraps the instance's ``run_tts`` so that any synthesis attempt producing
    zero ``TTSAudioRawFrame`` is retried (exponential backoff) up to
    ``max_attempts`` times. Audio is streamed as it arrives on the successful
    attempt — only the empty case (nothing yielded) triggers a retry. If every
    attempt is empty, an ``ErrorFrame`` is surfaced instead of silence so the
    caller fails loudly rather than idle-timing-out with a corrupted transcript.

    Provider-general and safe for streaming (websocket) services: those signal
    "audio arrives via a separate receive loop" by yielding ``None`` from
    ``run_tts``; when that happens the retry is skipped and frames pass through
    untouched.

    Returns the same ``tts`` instance (mutated) for convenient chaining.
    """
    original_run_tts = tts.run_tts
    service_name = type(tts).__name__

    async def run_tts_with_retry(text: str, context_id: str):
        for attempt in range(1, max_attempts + 1):
            audio_seen = False
            streams_out_of_band = False
            deferred = []  # non-audio frames held until we commit to this attempt
            async for frame in original_run_tts(text, context_id):
                if frame is None:
                    # Websocket services deliver audio via a separate receive
                    # loop; the None sentinel means "handled elsewhere". Never
                    # retry these — pass through as-is.
                    streams_out_of_band = True
                    yield frame
                elif isinstance(frame, TTSAudioRawFrame):
                    audio_seen = True
                    yield frame
                elif isinstance(frame, ErrorFrame):
                    deferred.append(frame)
                elif audio_seen:
                    yield frame
                else:
                    deferred.append(frame)

            if audio_seen or streams_out_of_band:
                for frame in deferred:
                    if not isinstance(frame, ErrorFrame):
                        yield frame
                return

            if attempt < max_attempts:
                delay = backoff_base_secs * (2 ** (attempt - 1))
                logger.warning(
                    f"{service_name}: TTS produced no audio for [{text}] "
                    f"(attempt {attempt}/{max_attempts}); retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue

            logger.error(
                f"{service_name}: TTS produced no audio for [{text}] after "
                f"{max_attempts} attempts; surfacing error"
            )
            emitted_error = False
            for frame in deferred:
                yield frame
                if isinstance(frame, ErrorFrame):
                    emitted_error = True
            if not emitted_error:
                yield ErrorFrame(
                    error=(
                        f"{service_name} produced no audio for [{text}] after "
                        f"{max_attempts} attempts"
                    )
                )

    tts.run_tts = run_tts_with_retry
    return tts


def _build_param_property(param: dict) -> dict:
    """Build a property dict for a single parameter."""
    prop = {
        "type": param["type"],
        "description": param["description"],
    }
    if "items" in param:
        prop["items"] = param["items"]
    if "enum" in param:
        prop["enum"] = param["enum"]
    return prop


def build_tools_schema(
    tools: list[dict],
) -> tuple[list[FunctionSchema], dict[str, dict]]:
    """
    Build FunctionSchema objects from tool definitions.

    Supports two tool types:
    - structured_output (default): Parameters at top level in 'parameters' array
    - webhook: Parameters in nested 'webhook.queryParameters' and 'webhook.body.parameters'

    Args:
        tools: List of tool definition dicts

    Returns:
        tuple of (list of FunctionSchema objects, dict of webhook configs keyed by tool name)

    Raises:
        ValueError: If webhook tool is missing required fields (url, method, headers)
    """
    function_schemas = []
    webhook_configs = {}

    for tool in tools:
        properties = {}
        required = []

        if tool.get("type") == "webhook":
            # For webhook tools, structure params as nested body and query dicts
            webhook = tool.get("webhook", {})

            # Validate required webhook fields
            for field in ["url", "method"]:
                if field not in webhook:
                    raise ValueError(
                        f"Webhook tool '{tool['name']}' is missing required '{field}' field"
                    )

            webhook_configs[tool["name"]] = {
                "url": webhook["url"],
                "method": webhook["method"],
                "headers": webhook.get("headers", []),
                "timeout": webhook.get("timeout", 20),
            }

            # Build query parameters schema
            query_properties = {}
            query_required = []
            if "queryParameters" in webhook:
                for param in webhook["queryParameters"]:
                    query_properties[param["id"]] = _build_param_property(param)
                    if param.get("required"):
                        query_required.append(param["id"])

            # Build body parameters schema
            body_properties = {}
            body_required = []
            if "body" in webhook and "parameters" in webhook["body"]:
                for param in webhook["body"]["parameters"]:
                    body_properties[param["id"]] = _build_param_property(param)
                    if param.get("required"):
                        body_required.append(param["id"])

            # Create nested structure with query and body as separate objects
            if query_properties:
                properties["query"] = {
                    "type": "object",
                    "description": "Query parameters for the webhook request",
                    "properties": query_properties,
                    "required": query_required,
                }
                if query_required:
                    required.append("query")
            if body_properties:
                properties["body"] = {
                    "type": "object",
                    "description": webhook.get("body", {}).get(
                        "description", "Request body parameters"
                    ),
                    "properties": body_properties,
                    "required": body_required,
                }
                if body_required:
                    required.append("body")
        else:
            # For structured_output or default type, use tool["parameters"]
            parameters = tool.get("parameters", [])

            for parameter in parameters:
                if parameter.get("required"):
                    required.append(parameter["id"])
                properties[parameter["id"]] = _build_param_property(parameter)

        function_schema = FunctionSchema(
            name=tool["name"],
            description=tool["description"],
            properties=properties,
            required=required,
        )
        function_schemas.append(function_schema)

    return function_schemas, webhook_configs


async def make_webhook_call(
    webhook_config: dict,
    arguments: dict,
) -> dict[str, Any]:
    """
    Make an HTTP webhook call with the provided configuration and arguments.

    Args:
        webhook_config: Dict containing url, method, headers, and timeout
        arguments: Dict containing 'query' and/or 'body' parameters from LLM

    Returns:
        Dict with 'status', 'status_code', and 'response' (or 'error')
    """
    url = webhook_config["url"]
    method = webhook_config["method"].upper()
    headers_list = webhook_config["headers"]
    timeout = webhook_config.get("timeout", 20)

    # Convert headers list to dict
    headers = {}
    for header in headers_list:
        headers[header["name"]] = header["value"]

    # Extract query params and body from arguments
    query_params = arguments.get("query", {})
    body = arguments.get("body", {})

    logger.info(
        f"Making webhook call:\n"
        f"  method: {method}\n"
        f"  url: {url}\n"
        f"  headers: {headers}\n"
        f"  query: {query_params}\n"
        f"  body: {body}\n"
        f"  timeout: {timeout}s"
    )

    try:
        async with aiohttp.ClientSession() as session:
            request_kwargs = {
                "url": url,
                "headers": headers,
                "params": query_params if query_params else None,
                "timeout": aiohttp.ClientTimeout(total=timeout),
            }

            # Add body for methods that support it
            if method in ["POST", "PUT", "PATCH"] and body:
                request_kwargs["json"] = body

            async with session.request(method, **request_kwargs) as response:
                status_code = response.status
                try:
                    response_data = await response.json()
                except Exception:
                    response_data = await response.text()

                logger.info(
                    f"Webhook response: status={status_code}, response={response_data}"
                )

                return {
                    "type": "webhook_response",
                    "status": "success" if 200 <= status_code < 300 else "error",
                    "status_code": status_code,
                    "response": response_data,
                }

    except asyncio.TimeoutError:
        logger.error(f"Webhook call timed out after {timeout}s")
        return {
            "type": "webhook_response",
            "status": "error",
            "error": f"Request timed out after {timeout}s",
        }
    except aiohttp.ClientError as e:
        logger.error(f"Webhook call failed: {e}")
        return {
            "type": "webhook_response",
            "status": "error",
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Unexpected error during webhook call: {e}")
        return {
            "type": "webhook_response",
            "status": "error",
            "error": str(e),
        }


def summarize_metric_distribution(
    values: list,
    *,
    metric_type: Optional[str] = None,
    scale: Optional[tuple] = None,
    evaluator_id: Optional[str] = None,
) -> dict:
    """Build a ``metrics.json`` summary entry for a list of per-item ``values``.

    Returns ``{["type"], "mean", "std", "values"[, "scale_min", "scale_max"]
    [, "evaluator_id"]}`` with JSON-friendly float aggregates. Shared by the
    LLM and agent simulation aggregators so the entry shape stays identical
    across all of them. Optional fields are omitted when their argument is
    ``None``: ``metric_type`` (the ``stt_llm_judge`` rollup has no type),
    ``scale`` (only rating criteria), and ``evaluator_id``.
    """
    entry: dict = {}
    if metric_type is not None:
        entry["type"] = metric_type
    entry["mean"] = float(np.mean(values))
    entry["std"] = float(np.std(values))
    entry["values"] = values
    if scale is not None:
        entry["scale_min"], entry["scale_max"] = scale
    if evaluator_id is not None:
        entry["evaluator_id"] = evaluator_id
    return entry


def read_leaderboard_metrics(metrics_path: Path) -> dict:
    """Read a provider/run ``metrics.json`` into a flat ``{column: scalar}`` dict.

    Shared by the STT and TTS leaderboards. Current format: evaluator entries
    are dicts carrying a ``mean`` — extracted into the ``<key>`` column;
    latency dicts (e.g. ``ttfb``) carry ``p50``/``p95``/``p99`` — fanned out
    into ``<key>_p50``/``<key>_p95``/``<key>_p99`` columns; plain numbers (e.g.
    ``wer``) are kept as-is. The legacy ``metric_name``/list format is still
    supported for older runs.
    """
    if not metrics_path.exists():
        print(f"[WARN] metrics.json missing for {metrics_path.parent.name}")
        return {}

    with metrics_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    metrics: dict = {}
    if isinstance(data, dict) and "metric_name" not in data:
        for key, value in data.items():
            if key == "cost" and isinstance(value, dict):
                # Only the USD total is comparable across providers, so it is
                # the one cost column on the leaderboard. Per-unit rates and
                # native-currency figures live in each provider's metrics.json.
                cost_usd = value.get("cost_usd")
                if isinstance(cost_usd, (int, float)):
                    metrics["cost_usd"] = float(cost_usd)
            elif isinstance(value, dict) and "mean" in value:
                metrics[key] = value["mean"]
            elif isinstance(value, dict) and "p50" in value:
                for pct in ("p50", "p95", "p99"):
                    if pct in value:
                        metrics[f"{key}_{pct}"] = value[pct]
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
        return metrics

    # Legacy format
    if isinstance(data, dict):
        data = [data]
    for entry in data:
        if not isinstance(entry, dict):
            continue
        metric_name = entry.get("metric_name")
        if metric_name:
            metrics[metric_name] = entry["mean"]
            continue
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
    return metrics
