import asyncio
import io
import logging
import os
import struct
import wave
from collections import defaultdict
from contextvars import ContextVar
from pathlib import Path
from typing import Literal, Optional

import aiofiles
from loguru import logger
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

# Sarvam supported language codes (Indian languages)
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

# Google Speech-to-Text language codes (BCP-47)
GOOGLE_LANGUAGE_CODES = {
    # GA (Generally Available)
    "catalan": "ca-ES",
    "chinese": "cmn-Hans-CN",
    "mandarin": "cmn-Hans-CN",
    "croatian": "hr-HR",
    "danish": "da-DK",
    "dutch": "nl-NL",
    "english": "en-US",
    "finnish": "fi-FI",
    "french": "fr-FR",
    "german": "de-DE",
    "greek": "el-GR",
    "hindi": "hi-IN",
    "italian": "it-IT",
    "japanese": "ja-JP",
    "korean": "ko-KR",
    "polish": "pl-PL",
    "portuguese": "pt-BR",
    "romanian": "ro-RO",
    "russian": "ru-RU",
    "spanish": "es-ES",
    "swedish": "sv-SE",
    "turkish": "tr-TR",
    "ukrainian": "uk-UA",
    "vietnamese": "vi-VN",
    # Preview
    "arabic": "ar-XA",
    "armenian": "hy-AM",
    "bengali": "bn-IN",
    "bulgarian": "bg-BG",
    "burmese": "my-MM",
    "cantonese": "yue-Hant-HK",
    "czech": "cs-CZ",
    "estonian": "et-EE",
    "filipino": "fil-PH",
    "gujarati": "gu-IN",
    "hebrew": "iw-IL",
    "hungarian": "hu-HU",
    "indonesian": "id-ID",
    "kannada": "kn-IN",
    "khmer": "km-KH",
    "lao": "lo-LA",
    "latvian": "lv-LV",
    "lithuanian": "lt-LT",
    "malay": "ms-MY",
    "malayalam": "ml-IN",
    "marathi": "mr-IN",
    "nepali": "ne-NP",
    "norwegian": "no-NO",
    "persian": "fa-IR",
    "punjabi": "pa-Guru-IN",
    "serbian": "sr-RS",
    "slovak": "sk-SK",
    "slovenian": "sl-SI",
    "swahili": "sw",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "thai": "th-TH",
    "uzbek": "uz-UZ",
}

# ElevenLabs supported language codes (ISO 639-3)
ELEVENLABS_LANGUAGE_CODES = {
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
    "cebuano": "ceb",
    "chichewa": "nya",
    "croatian": "hrv",
    "czech": "ces",
    "danish": "dan",
    "dutch": "nld",
    "english": "eng",
    "estonian": "est",
    "filipino": "fil",
    "finnish": "fin",
    "french": "fra",
    "fulah": "ful",
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
    "kabuverdianu": "kea",
    "kannada": "kan",
    "kazakh": "kaz",
    "khmer": "khm",
    "korean": "kor",
    "kurdish": "kur",
    "kyrgyz": "kir",
    "lao": "lao",
    "latvian": "lav",
    "lingala": "lin",
    "lithuanian": "lit",
    "luo": "luo",
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
    "tamil": "tam",
    "tajik": "tgk",
    "telugu": "tel",
    "thai": "tha",
    "turkish": "tur",
    "ukrainian": "ukr",
    "umbundu": "umb",
    "urdu": "urd",
    "uzbek": "uzb",
    "vietnamese": "vie",
    "welsh": "cym",
    "wolof": "wol",
    "xhosa": "xho",
    "zulu": "zul",
}

# Deepgram supported language codes
DEEPGRAM_LANGUAGE_CODES = {
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

# OpenAI/Groq Whisper supported language codes (ISO 639-1)
OPENAI_LANGUAGE_CODES = {
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

# Groq uses the same Whisper model as OpenAI
GROQ_LANGUAGE_CODES = OPENAI_LANGUAGE_CODES

# Cartesia supported language codes
CARTESIA_LANGUAGE_CODES = {
    "english": "en",
    "chinese": "zh",
    "german": "de",
    "spanish": "es",
    "russian": "ru",
    "korean": "ko",
    "french": "fr",
    "japanese": "ja",
    "portuguese": "pt",
    "turkish": "tr",
    "polish": "pl",
    "catalan": "ca",
    "dutch": "nl",
    "arabic": "ar",
    "swedish": "sv",
    "italian": "it",
    "indonesian": "id",
    "hindi": "hi",
    "finnish": "fi",
    "vietnamese": "vi",
    "hebrew": "he",
    "ukrainian": "uk",
    "greek": "el",
    "malay": "ms",
    "czech": "cs",
    "romanian": "ro",
    "danish": "da",
    "hungarian": "hu",
    "tamil": "ta",
    "norwegian": "no",
    "thai": "th",
    "urdu": "ur",
    "croatian": "hr",
    "bulgarian": "bg",
    "lithuanian": "lt",
    "latin": "la",
    "maori": "mi",
    "malayalam": "ml",
    "welsh": "cy",
    "slovak": "sk",
    "telugu": "te",
    "persian": "fa",
    "latvian": "lv",
    "bengali": "bn",
    "serbian": "sr",
    "azerbaijani": "az",
    "slovenian": "sl",
    "kannada": "kn",
    "estonian": "et",
    "macedonian": "mk",
    "breton": "br",
    "basque": "eu",
    "icelandic": "is",
    "armenian": "hy",
    "nepali": "ne",
    "mongolian": "mn",
    "bosnian": "bs",
    "kazakh": "kk",
    "albanian": "sq",
    "swahili": "sw",
    "galician": "gl",
    "marathi": "mr",
    "punjabi": "pa",
    "sinhala": "si",
    "khmer": "km",
    "shona": "sn",
    "yoruba": "yo",
    "somali": "so",
    "afrikaans": "af",
    "occitan": "oc",
    "georgian": "ka",
    "belarusian": "be",
    "tajik": "tg",
    "sindhi": "sd",
    "gujarati": "gu",
    "amharic": "am",
    "yiddish": "yi",
    "lao": "lo",
    "uzbek": "uz",
    "faroese": "fo",
    "haitian": "ht",
    "pashto": "ps",
    "turkmen": "tk",
    "nynorsk": "nn",
    "maltese": "mt",
    "sanskrit": "sa",
    "luxembourgish": "lb",
    "burmese": "my",
    "tibetan": "bo",
    "tagalog": "tl",
    "malagasy": "mg",
    "assamese": "as",
    "tatar": "tt",
    "hawaiian": "haw",
    "lingala": "ln",
    "hausa": "ha",
    "bashkir": "ba",
    "javanese": "jw",
    "sundanese": "su",
    "cantonese": "yue",
    "odia": "or",  # Adding odia with standard code
}

# Smallest supported language codes
SMALLEST_LANGUAGE_CODES = {
    "italian": "it",
    "spanish": "es",
    "english": "en",
    "portuguese": "pt",
    "hindi": "hi",
    "german": "de",
    "french": "fr",
    "ukrainian": "uk",
    "russian": "ru",
    "kannada": "kn",
    "malayalam": "ml",
    "polish": "pl",
    "marathi": "mr",
    "gujarati": "gu",
    "czech": "cs",
    "slovak": "sk",
    "telugu": "te",
    "odia": "or",  # Smallest uses 'or' for Odia
    "dutch": "nl",
    "bengali": "bn",
    "latvian": "lv",
    "estonian": "et",
    "romanian": "ro",
    "punjabi": "pa",
    "finnish": "fi",
    "swedish": "sv",
    "bulgarian": "bg",
    "tamil": "ta",
    "hungarian": "hu",
    "danish": "da",
    "lithuanian": "lt",
    "maltese": "mt",
}


def get_language_code(language: str, provider: str) -> str:
    """Get the appropriate language code string for a provider.

    This is a global utility function that returns string language codes
    suitable for direct API calls to various STT/TTS providers.

    Args:
        language: The language name (english, hindi, kannada, bengali, malayalam,
                  marathi, odia, punjabi, tamil, telugu, gujarati)
        provider: The provider name (sarvam, google, deepgram, openai, etc.)

    Returns:
        The appropriate language code string for the provider

    Examples:
        >>> get_language_code("hindi", "sarvam")
        'hi-IN'
        >>> get_language_code("hindi", "deepgram")
        'hi'
        >>> get_language_code("english", "google")
        'en-US'
    """
    language = language.lower()

    # Sarvam uses Indian regional codes
    if provider == "sarvam":
        return SARVAM_LANGUAGE_CODES.get(language, "en-IN")

    # Google uses BCP-47 codes
    if provider == "google":
        return GOOGLE_LANGUAGE_CODES.get(language, "en-US")

    # Smallest uses ISO 639-1 codes with 'or' for Odia
    if provider == "smallest":
        return SMALLEST_LANGUAGE_CODES.get(language, "en")

    # Cartesia uses ISO 639-1 codes
    if provider == "cartesia":
        return CARTESIA_LANGUAGE_CODES.get(language, "en")

    # ElevenLabs uses ISO 639-3 codes
    if provider == "elevenlabs":
        return ELEVENLABS_LANGUAGE_CODES.get(language, "eng")

    # OpenAI uses ISO 639-1 codes
    if provider == "openai":
        return OPENAI_LANGUAGE_CODES.get(language, "en")

    # Groq uses same codes as OpenAI (Whisper model)
    if provider == "groq":
        return GROQ_LANGUAGE_CODES.get(language, "en")

    # Deepgram uses ISO 639-1 codes
    if provider == "deepgram":
        return DEEPGRAM_LANGUAGE_CODES.get(language, "en")

    # Default: use ISO 639-1 codes
    return DEFAULT_LANGUAGE_CODES.get(language, "en")


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
    from pense.integrations.smallest.stt import SmallestSTTService

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
    from pense.integrations.smallest.tts import SmallestTTSService

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
