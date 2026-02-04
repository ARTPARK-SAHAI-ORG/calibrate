import asyncio
import io
import logging
import os
import struct
import wave
from collections import defaultdict
from contextvars import ContextVar
from pathlib import Path
from typing import Literal, Optional, Any

import aiofiles
import aiohttp
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import InputTransportMessageFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transcriptions.language import Language

# Context variable to track current execution context (BOT or EVAL)
current_context: ContextVar[str] = ContextVar("current_context", default="UNKNOWN")


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
    print(text)

    if use_loguru:
        # logger.info will inherit the simulation context from logger.contextualize()
        logger.info(text)

    # Determine which print logger to use
    sim_name = simulation_name or current_simulation_name.get()
    if sim_name and sim_name in _simulation_print_loggers:
        _simulation_print_loggers[sim_name].info(text)
    elif _print_logger:
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


# =============================================================================
# Language Code Utilities
# =============================================================================

# Sarvam supported language codes (Indian languages) - Same for STT and TTS
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

# Groq STT uses the same Whisper model as OpenAI
GROQ_STT_LANGUAGE_CODES = OPENAI_STT_LANGUAGE_CODES

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
        return SARVAM_LANGUAGE_CODES.get(language, "en-IN")
    elif provider == "google":
        return GOOGLE_STT_LANGUAGE_CODES.get(language, "en-US")
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
        return SARVAM_LANGUAGE_CODES.get(language, "en-IN")
    elif provider == "google":
        return GOOGLE_TTS_LANGUAGE_CODES.get(language, "en-US")
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
        "sarvam": SARVAM_LANGUAGE_CODES,
        "google": GOOGLE_STT_LANGUAGE_CODES,
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
        "sarvam": SARVAM_LANGUAGE_CODES,
        "google": GOOGLE_TTS_LANGUAGE_CODES,
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


def get_stt_language(
    language: Literal["english", "hindi", "kannada"],
    provider: str,
) -> Language:
    """Get the appropriate Language enum for STT based on language and provider.

    Args:
        language: The language name (english, hindi, kannada)
        provider: The STT provider name

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


# Voice ID mappings for TTS providers by language
TTS_VOICE_IDS = {
    "cartesia": {
        "english": "66c6b81c-ddb7-4892-bdd5-19b5a7be38e7",
        "hindi": "28ca2041-5dda-42df-8123-f58ea9c3da00",
        "kannada": "7c6219d2-e8d2-462c-89d8-7ecba7c75d65",
    },
    "google": {
        "english": "en-US-Chirp3-HD-Achernar",
        "hindi": "hi-IN-Chirp3-HD-Achernar",
        "kannada": "kn-IN-Chirp3-HD-Achernar",
    },
    "elevenlabs": {
        "english": "90ipbRoKi4CpHXvKVtl0",
        "hindi": "jUjRbhZWoMK4aDciW36V",
        "kannada": "90ipbRoKi4CpHXvKVtl0",  # fallback to english
    },
    "smallest": {
        "english": "aarushi",
        "hindi": "aarushi",
        "kannada": "vijay",
    },
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
    from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
    from pipecat.services.openai.stt import OpenAISTTService
    from pipecat.services.google.stt import GoogleSTTService
    from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaLiveOptions
    from pipecat.services.groq.stt import GroqSTTService
    from pipecat.services.sarvam.stt import SarvamSTTService
    from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
    from calibrate.integrations.smallest.stt import SmallestSTTService

    stt_language = get_stt_language(language, provider)

    if provider == "deepgram":
        return DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(language=stt_language.value, encoding="linear16"),
        )
    elif provider == "sarvam":
        return SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
            params=SarvamSTTService.InputParams(language=stt_language.value),
        )
    elif provider == "elevenlabs":
        return ElevenLabsRealtimeSTTService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            params=ElevenLabsRealtimeSTTService.InputParams(
                language_code=stt_language.value,
            ),
        )
    elif provider == "openai":
        return OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model or "gpt-4o-transcribe",
            language=stt_language,
        )
    elif provider == "cartesia":
        return CartesiaSTTService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            live_options=CartesiaLiveOptions(language=stt_language.value),
        )
    elif provider == "smallest":
        return SmallestSTTService(
            api_key=os.getenv("SMALLEST_API_KEY"),
            url="wss://waves-api.smallest.ai/api/v1/asr",
            params=SmallestSTTService.SmallestInputParams(
                audioLanguage=stt_language.value,
            ),
        )
    elif provider == "groq":
        return GroqSTTService(
            api_key=os.getenv("GROQ_API_KEY"),
            model=model or "whisper-large-v3",
            language=stt_language,
        )
    elif provider == "google":
        return GoogleSTTService(
            sample_rate=16000,
            location="us",
            params=GoogleSTTService.InputParams(
                languages=stt_language,
                model=model or "chirp_3",
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
    from calibrate.integrations.smallest.tts import SmallestTTSService

    tts_language = get_tts_language(language, provider)

    # Get default voice ID if not provided
    if voice_id is None and provider in TTS_VOICE_IDS:
        voice_id = TTS_VOICE_IDS[provider].get(language)

    if provider == "cartesia":
        return CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            model=model or "sonic-3",
            params=CartesiaTTSService.InputParams(language=tts_language),
            voice_id=voice_id or "95d51f79-c397-46f9-b49a-23763d3eaa2d",
        )
    elif provider == "openai":
        return OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice=voice_id or "fable",
            instructions=instructions,
        )
    elif provider == "groq":
        return GroqTTSService(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=model or "canopylabs/orpheus-v1-english",
            voice_id=voice_id or "autumn",
        )
    elif provider == "google":
        return GoogleTTSService(
            voice_id=voice_id
            or TTS_VOICE_IDS["google"].get(language, "en-US-Chirp3-HD-Charon"),
            params=GoogleTTSService.InputParams(language=tts_language),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
    elif provider == "elevenlabs":
        return ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            model="eleven_multilingual_v2",
            voice_id=voice_id
            or TTS_VOICE_IDS["elevenlabs"].get(language, "90ipbRoKi4CpHXvKVtl0"),
            params=ElevenLabsTTSService.InputParams(language=tts_language),
        )
    elif provider == "sarvam":
        return SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            model=model or "bulbul:v2",
            voice_id=voice_id or "abhilash",
            params=SarvamTTSService.InputParams(language=tts_language),
        )
    elif provider == "deepgram":
        return DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            voice=voice_id or "aura-2-andromeda-en",
        )
    elif provider == "smallest":
        return SmallestTTSService(
            api_key=os.getenv("SMALLEST_API_KEY"),
            voice_id=voice_id or TTS_VOICE_IDS["smallest"].get(language, "aarushi"),
            params=SmallestTTSService.InputParams(language=tts_language),
        )
    else:
        raise ValueError(f"Unsupported TTS provider: {provider}")


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
            for field in ["url", "method", "headers"]:
                if field not in webhook:
                    raise ValueError(
                        f"Webhook tool '{tool['name']}' is missing required '{field}' field"
                    )

            webhook_configs[tool["name"]] = {
                "url": webhook["url"],
                "method": webhook["method"],
                "headers": webhook["headers"],
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
                    "status": "success" if 200 <= status_code < 300 else "error",
                    "status_code": status_code,
                    "response": response_data,
                }

    except asyncio.TimeoutError:
        logger.error(f"Webhook call timed out after {timeout}s")
        return {
            "status": "error",
            "error": f"Request timed out after {timeout}s",
        }
    except aiohttp.ClientError as e:
        logger.error(f"Webhook call failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Unexpected error during webhook call: {e}")
        return {
            "status": "error",
            "error": str(e),
        }
