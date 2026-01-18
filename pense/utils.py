import asyncio
import io
import logging
import os
import struct
import wave
from collections import defaultdict
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

import aiofiles
from loguru import logger
from pipecat.frames.frames import InputTransportMessageFrame
from pipecat.processors.frame_processor import FrameProcessor

# Context variable to track current execution context (BOT or EVAL)
current_context: ContextVar[str] = ContextVar("current_context", default="UNKNOWN")


def add_default_source(record):
    """Add default source if not present in extra"""
    if "source" not in record["extra"]:
        context = current_context.get()
        record["extra"]["source"] = f"{context}-SYS"
    return True


# Global print logger instance
_print_logger: Optional[logging.Logger] = None


def configure_print_logger(log_path: str, logger_name: str = "print_logger"):
    """Configure a dedicated logger for console print mirroring.

    Args:
        log_path: Path to the log file
        logger_name: Name for the logger instance (default: "print_logger")
    """
    global _print_logger
    _print_logger = logging.getLogger(logger_name)
    _print_logger.setLevel(logging.INFO)
    _print_logger.propagate = False

    for handler in list(_print_logger.handlers):
        _print_logger.removeHandler(handler)

    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _print_logger.addHandler(handler)


def log_and_print(message: object = "", use_loguru: bool = True):
    """Print to stdout and mirror the message to both loguru and file logger.

    Args:
        message: Message to print and log
        use_loguru: Whether to also log via loguru (default: True)
    """
    text = str(message)
    print(text)
    if use_loguru:
        logger.info(text)
    if _print_logger:
        _print_logger.info(text)


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
        log_and_print(f"\033[92mCreating new audio file at {filepath}\033[0m")
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_chunk)
            async with aiofiles.open(filepath, "wb") as file:
                await file.write(buffer.getvalue())
    else:
        log_and_print(f"\033[92mAppending audio chunk to {filepath}\033[0m")
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


def combine_audio_files(audio_dir: str, output_path: str) -> bool:
    """Combine all WAV files in a directory into a single conversation WAV file.

    Files are sorted by modification time to preserve conversation order.

    Args:
        audio_dir: Directory containing the audio files
        output_path: Path to save the combined audio file

    Returns:
        True if successful, False otherwise
    """
    import glob

    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))

    if not audio_files:
        logger.warning(f"No audio files found in {audio_dir}")
        return False

    # Sort files by modification time
    sorted_files = sorted(audio_files, key=lambda f: os.path.getmtime(f))

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
