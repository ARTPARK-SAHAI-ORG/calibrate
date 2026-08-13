import asyncio
import argparse
import sys
import os
import json
import time
from os.path import join, exists
from pathlib import Path
from typing import Dict, List
import base64
import wave

from openai import AsyncOpenAI
from elevenlabs import VoiceSettings
from elevenlabs.client import AsyncElevenLabs
from groq import AsyncGroq
from cartesia import AsyncCartesia
from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse
from google.cloud import texttospeech
from google import genai
from google.genai import types as genai_types

import numpy as np
import pandas as pd

import backoff

from calibrate_agent.utils import (
    get_tts_language_code,
    get_audio_duration_seconds,
    validate_tts_language,
    provider_log as _log,
    provider_log_file as _current_log_file,
    TTS_PROVIDER_MODELS,
    get_tts_voice,
    get_gemini_api_key,
)
from calibrate_agent.pricing import TTS_DEFAULT_MODELS, cost_breakdown, resolve_pricing
from calibrate_agent.judge_store import JudgeStore
from calibrate_agent.tts.metrics import get_tts_llm_judge_score
from calibrate_agent.llm._metrics_utils import _latency_percentiles
from calibrate_agent.judges import (
    is_rating,
    DEFAULT_TTS_EVALUATOR,
    require_unique_evaluator_names,
    write_evaluator_config,
)
from calibrate_agent.langfuse import (
    observe,
    langfuse,
    langfuse_enabled,
    create_langfuse_audio_media,
)
from calibrate_agent.rate_limit import SARVAM_TTS_STREAMING_LIMITER


# Subdirectory (under a run's output dir) where synthesized audio is written.
# Single-sourced: the synthesis path writes here and eval-only reads from here.
TTS_AUDIO_SUBDIR = "audios"


# =============================================================================
# TTS Provider API Methods
# =============================================================================


def _default_tts_model(provider: str, language: str | None = None) -> str | None:
    """Resolve the pricing model name for a provider/language.

    Sindhi uses a different model for some providers than their default: Google
    synthesizes it with the synchronous Gemini-TTS model instead of the
    streaming Chirp3-HD voices, and ElevenLabs uses ``eleven_v3`` instead of
    ``eleven_multilingual_v2`` (see ``synthesize_google`` / ``synthesize_elevenlabs``).
    """
    is_sindhi = bool(language) and language.lower() == "sindhi"
    if is_sindhi and provider == "google":
        return "gemini-2.5-flash-tts"
    if is_sindhi and provider == "elevenlabs":
        return "eleven_v3"
    return TTS_DEFAULT_MODELS.get(provider)


def _build_tts_cost_metrics(
    provider: str,
    texts: list | None,
    audio_paths: list | None = None,
    model: str | None = None,
) -> dict | None:
    """Build TTS cost metrics from provider price config.

    Each provider is priced in its native billing unit — no unit conversion.
    Character-billed providers cost on the total input characters; audio-billed
    providers (OpenAI, Gemini) cost on the measured output audio duration read
    from ``audio_paths``. Returns ``None`` when there is nothing to price or no
    pricing is configured for the provider/model.
    """
    pricing = resolve_pricing("tts", provider, model=model)
    if not pricing:
        return None

    if pricing["billing_unit"] == "minute":
        durations = [
            d
            for d in (get_audio_duration_seconds(p) for p in (audio_paths or []))
            if d is not None
        ]
        if not durations:
            return None
        total_seconds = float(sum(durations))
        total_minutes = total_seconds / 60.0
        metrics = {
            "provider": provider,
            "pricing_model": pricing["model"],
            "billing_unit": "minute",
            "total_seconds": total_seconds,
            "audio_minutes": round(total_minutes, 4),
        }
        metrics.update(cost_breakdown(pricing, total_minutes, "cost_per_minute"))
        return metrics

    total_characters = sum(
        len(str(text))
        for text in (texts or [])
        if text is not None and not (isinstance(text, float) and pd.isna(text))
    )
    if total_characters <= 0:
        return None

    metrics = {
        "provider": provider,
        "pricing_model": pricing["model"],
        "billing_unit": "character",
        "total_characters": total_characters,
    }
    metrics.update(
        cost_breakdown(pricing, total_characters / 1_000_000.0, "cost_per_million_chars")
    )
    return metrics


def save_audio(audio_bytes: bytes, output_path: str, sample_rate: int = 24000):
    """Save audio bytes to a WAV file.

    Args:
        audio_bytes: Raw audio bytes (PCM or WAV format)
        output_path: Path to save the WAV file
        sample_rate: Audio sample rate (default: 24000)
    """
    import wave

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Check if audio_bytes is already a WAV file
    if audio_bytes[:4] == b"RIFF":
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
    else:
        # Raw PCM data - wrap in WAV
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)


def convert_mp3_to_wav(mp3_path: str, wav_path: str, cleanup: bool = True):
    """Convert MP3 file to WAV format.

    Args:
        mp3_path: Path to the input MP3 file
        wav_path: Path to save the output WAV file
        cleanup: If True, delete the MP3 file after conversion (default: True)
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    if cleanup:
        os.remove(mp3_path)


async def synthesize_openai(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using OpenAI's TTS API and stream directly to file."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = AsyncOpenAI()

    start_time = time.time()
    ttfb = None

    # Stream directly to file
    with open(audio_path, "wb") as f:
        async with client.audio.speech.with_streaming_response.create(
            model=TTS_PROVIDER_MODELS["openai"],
            voice=get_tts_voice("openai", language),
            input=text,
            response_format="wav",
        ) as response:
            async for chunk in response.iter_bytes():
                if ttfb is None:
                    ttfb = time.time() - start_time
                f.write(chunk)

    return {"ttfb": ttfb}


async def synthesize_google(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using Google Cloud Text-to-Speech API and save to file."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

    lang_code = get_tts_language_code(language, "google")

    client = texttospeech.TextToSpeechClient()

    # Sindhi requires synchronous API with Gemini-TTS model (streaming API doesn't support Sindhi)
    # See: https://cloud.google.com/text-to-speech/docs/gemini-tts
    if language.lower() == "sindhi":
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name="Charon",
            model_name="gemini-2.5-flash-tts",
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
        )

        start_time = time.time()
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice_params, audio_config=audio_config
        )
        ttfb = time.time() - start_time

        # Save the audio content
        save_audio(response.audio_content, audio_path, sample_rate=24000)

        return {}

    # For other languages, use streaming API with Chirp3-HD voices
    streaming_audio_config = texttospeech.StreamingAudioConfig(
        audio_encoding=texttospeech.AudioEncoding.PCM,
        sample_rate_hertz=24000,
    )

    voice_params = texttospeech.VoiceSelectionParams(
        name=get_tts_voice("google", language),
        language_code=lang_code,
    )

    streaming_config = texttospeech.StreamingSynthesizeConfig(
        voice=voice_params,
        streaming_audio_config=streaming_audio_config,
    )

    # Set the config for your stream. The first request must contain your config, and then each subsequent request must contain text.
    config_request = texttospeech.StreamingSynthesizeRequest(
        streaming_config=streaming_config
    )

    start_time = time.time()
    ttfb = None

    # Request generator. Consider using Gemini or another LLM with output streaming as a generator.
    def request_generator():
        yield config_request
        # for text in text_iterator:
        yield texttospeech.StreamingSynthesizeRequest(
            input=texttospeech.StreamingSynthesisInput(text=text)
        )

    streaming_responses = client.streaming_synthesize(request_generator())

    # Collect audio chunks and save to file
    audio_chunks = []
    for response in streaming_responses:
        if ttfb is None:
            ttfb = time.time() - start_time

        audio_chunks.append(response.audio_content)

    # Save combined PCM audio as WAV
    audio_bytes = b"".join(audio_chunks)
    save_audio(audio_bytes, audio_path, sample_rate=24000)

    return {"ttfb": ttfb}


async def synthesize_elevenlabs(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using ElevenLabs' TTS API and stream directly to file."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")

    start_time = time.time()
    ttfb = None

    elevenlabs = AsyncElevenLabs(api_key=api_key)

    voice_id = get_tts_voice("elevenlabs", language)
    output_format = "mp3_24000_48"

    if language.lower() == "sindhi":
        model_id = "eleven_v3"

        response = elevenlabs.text_to_dialogue.stream(
            output_format=output_format,
            inputs=[
                {"text": text, "voice_id": voice_id},
            ],
            language_code="sd",
            model_id="eleven_v3",
        )

    else:
        model_id = TTS_PROVIDER_MODELS["elevenlabs"]

        response = elevenlabs.text_to_speech.stream(
            voice_id=voice_id,
            output_format=output_format,
            text=text,
            model_id=model_id,
            # Optional voice settings that allow you to customize the output
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            ),
        )

    mp3_path = audio_path.replace(".wav", ".mp3")
    with open(mp3_path, "wb") as f:
        async for chunk in response:
            if ttfb is None:
                ttfb = time.time() - start_time

            if chunk:
                f.write(chunk)

    convert_mp3_to_wav(mp3_path, audio_path)

    return {"ttfb": ttfb}


async def synthesize_cartesia(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using Cartesia's TTS API and stream directly to file."""
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        raise ValueError("CARTESIA_API_KEY environment variable not set")

    lang_code = get_tts_language_code(language, "cartesia")

    client = AsyncCartesia(api_key=api_key)

    # Default voice ID
    with open(audio_path, "wb") as f:
        start_time = time.time()
        ttfb = None

        bytes_iter = client.tts.bytes(
            model_id=TTS_PROVIDER_MODELS["cartesia"],
            transcript=text,
            voice={
                "mode": "id",
                "id": get_tts_voice("cartesia", language),
            },
            language=lang_code,
            output_format={
                "container": "wav",
                "sample_rate": 24000,
                "encoding": "pcm_f32le",
            },
        )

        async for chunk in bytes_iter:
            if ttfb is None:
                ttfb = time.time() - start_time

            f.write(chunk)

    return {"ttfb": ttfb}


async def synthesize_groq(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using Groq's TTS API and save to file."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    client = AsyncGroq(api_key=api_key)

    model = TTS_PROVIDER_MODELS["groq"]
    voice = get_tts_voice("groq", language)
    response_format = "wav"

    response = await client.audio.speech.create(
        model=model, voice=voice, input=text, response_format=response_format
    )

    _log(f"\033[93mStoring generated audio to {audio_path}\033[0m")
    await response.write_to_file(audio_path)

    return {}


async def synthesize_sarvam(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using Sarvam's TTS API and save to file."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY environment variable not set")

    lang_code = get_tts_language_code(language, "sarvam")

    await SARVAM_TTS_STREAMING_LIMITER.acquire()

    client = AsyncSarvamAI(api_subscription_key=api_key)

    start_time = time.time()
    ttfb = None

    async with client.text_to_speech_streaming.connect(
        model=TTS_PROVIDER_MODELS["sarvam"], send_completion_event=True
    ) as ws:
        await ws.configure(
            target_language_code=lang_code,
            speaker=get_tts_voice("sarvam", language),
            output_audio_codec="mp3",
            speech_sample_rate=22050,
            enable_preprocessing=True,
        )

        await ws.convert(text)
        # print("Sent text message")

        await ws.flush()
        # print("Flushed buffer")

        mp3_path = str(Path(audio_path).with_suffix(".mp3"))
        chunk_count = 0
        with open(mp3_path, "wb") as f:
            async for message in ws:
                if isinstance(message, AudioOutput):
                    if ttfb is None:
                        ttfb = time.time() - start_time
                        # Print "Started audio generation" in yellow using ANSI escape code for yellow
                        _log(
                            f"\033[93mStoring generated audio to {audio_path}\033[0m",
                        )

                    chunk_count += 1
                    audio_chunk = base64.b64decode(message.data.audio)
                    f.write(audio_chunk)
                    f.flush()
                elif isinstance(message, EventResponse):
                    # Break when we receive the final event
                    if message.data.event_type == "final":
                        break

        convert_mp3_to_wav(mp3_path, audio_path)
        # print(f"All {chunk_count} chunks saved to output.wav")
        _log("\033[93mAudio generation complete\033[0m")
        if hasattr(ws, "_websocket") and not ws._websocket.closed:
            await ws._websocket.close()
            print("WebSocket connection closed.")

    return {
        "ttfb": ttfb,
    }


async def synthesize_smallest(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using Smallest AI's lightning-v3.1 streaming TTS.

    lightning-v2 (the standalone smallestai SDK's hard-coded model) was retired
    and now returns HTTP 410, so this talks to the current lightning-v3.1
    WebSocket directly — the same endpoint pipecat's SmallestTTSService uses.
    """
    from websockets.asyncio.client import connect as websocket_connect

    api_key = os.getenv("SMALLEST_API_KEY")
    if not api_key:
        raise ValueError("SMALLEST_API_KEY environment variable not set")

    lang_code = get_tts_language_code(language, "smallest")
    ws_url = "wss://waves-api.smallest.ai/api/v1/lightning-v3.1/get_speech/stream"
    payload = {
        "text": text,
        "voice_id": get_tts_voice("smallest", language),
        "language": lang_code,
        "sample_rate": 24000,
        "speed": 1.0,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    start_time = time.time()
    ttfb = None
    audio_chunks: List[bytes] = []

    async with websocket_connect(ws_url, additional_headers=headers) as ws:
        await ws.send(json.dumps(payload))
        async for message in ws:
            msg = json.loads(message)
            status = msg.get("status")
            if status == "chunk":
                audio_b64 = msg.get("data", {}).get("audio")
                if audio_b64:
                    if ttfb is None:
                        ttfb = time.time() - start_time
                    audio_chunks.append(base64.b64decode(audio_b64))
            elif status == "error":
                error = msg.get("error") or msg.get("message") or msg
                raise RuntimeError(f"Smallest TTS error: {error}")
            elif status == "complete":
                break

    save_audio(b"".join(audio_chunks), audio_path, sample_rate=24000)

    return {"ttfb": ttfb}


def _gemini_audio_chunk(chunk) -> bytes | None:
    """Extract inline PCM audio bytes from a streamed Gemini response chunk."""
    candidates = getattr(chunk, "candidates", None)
    if not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return None
    inline = getattr(parts[0], "inline_data", None)
    return getattr(inline, "data", None) if inline else None


async def synthesize_gemini(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech with a Gemini TTS model via the google-genai API.

    Streamed so ttfb reflects the first audio chunk (comparable to the other
    streaming providers), not the full synthesis time. Gemini TTS returns raw
    24 kHz, 16-bit mono PCM. Benchmark-only — no cascaded pipecat Gemini TTS
    service is wired into create_tts_service.
    """
    client = genai.Client(api_key=get_gemini_api_key())

    voice = get_tts_voice("gemini", language)
    lang_code = get_tts_language_code(language, "gemini")

    config = genai_types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=genai_types.SpeechConfig(
            language_code=lang_code,
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                    voice_name=voice
                )
            ),
        ),
    )

    start_time = time.time()
    ttfb = None
    audio_chunks = []

    stream = await client.aio.models.generate_content_stream(
        model=TTS_PROVIDER_MODELS["gemini"],
        contents=text,
        config=config,
    )
    async for chunk in stream:
        data = _gemini_audio_chunk(chunk)
        if data:
            if ttfb is None:
                ttfb = time.time() - start_time
            audio_chunks.append(data)

    # A blocked or text-only response yields no audio parts. Fail cleanly (the
    # router's @backoff retries transient blocks, then the row is marked failed)
    # rather than writing an empty WAV.
    if not audio_chunks:
        raise ValueError("Gemini TTS returned no audio")

    save_audio(b"".join(audio_chunks), audio_path, sample_rate=24000)

    return {"ttfb": ttfb}


# =============================================================================
# Main Synthesis Router
# =============================================================================


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(name="tts", capture_input=False, capture_output=False)
async def synthesize_speech(
    text: str,
    provider: str,
    language: str,
    audio_path: str,
) -> Dict:
    """Route speech synthesis to the appropriate provider and save to audio_path."""
    provider_methods = {
        "openai": synthesize_openai,
        "google": synthesize_google,
        "gemini": synthesize_gemini,
        "elevenlabs": synthesize_elevenlabs,
        "cartesia": synthesize_cartesia,
        "groq": synthesize_groq,
        "sarvam": synthesize_sarvam,
        "smallest": synthesize_smallest,
    }

    if provider not in provider_methods:
        raise ValueError(f"Unsupported TTS provider: {provider}")

    method = provider_methods[provider]
    metrics = await method(text, language, audio_path)

    audio_media = create_langfuse_audio_media(audio_path)

    if langfuse_enabled and langfuse:
        langfuse.update_current_trace(
            input={"text": text, "language": language, "provider": provider},
            output=audio_media,
            metadata={
                "input": f"Text: {text}\nLanguage: {language}\nProvider: {provider}\nAudio path: {audio_path}",
                "metrics": metrics,
            },
        )

    return metrics


# =============================================================================
# TTS Evaluation Main
# =============================================================================


async def run_tts_eval(
    gt_data: List[Dict],
    provider: str,
    language: str,
    output_dir: str,
    results_csv_path: Path,
    overwrite: bool = False,
) -> int:
    """Process texts and synthesize speech, saving results immediately to CSV.

    Args:
        gt_data: List of {"id": ..., "text": ...} for each text to process
        provider: TTS provider name
        language: Language code
        output_dir: Directory to save audio files
        results_csv_path: Path to save results CSV
        overwrite: If True, overwrite existing results instead of resuming

    Returns:
        Number of texts successfully synthesized in this run.
    """
    # Load existing results to skip already processed texts (unless overwrite is True)
    if overwrite:
        processed_ids = set()
        # Remove existing results file if overwriting
        if exists(results_csv_path):
            os.remove(results_csv_path)
    elif exists(results_csv_path):
        existing_df = pd.read_csv(results_csv_path)
        processed_ids = set(existing_df["id"].tolist())
    else:
        processed_ids = set()

    audio_output_dir = join(output_dir, TTS_AUDIO_SUBDIR)
    os.makedirs(audio_output_dir, exist_ok=True)

    success_count = 0
    ttfb_values = []

    for i, item in enumerate(gt_data):
        _id = item["id"]
        text = item["text"]

        # Skip if already processed
        if _id in processed_ids:
            _log(f"Skipping already processed: {_id}")
            continue

        _log(f"Processing [{i+1}/{len(gt_data)}]: {_id}")

        audio_path = join(audio_output_dir, f"{_id}.wav")
        try:
            result = await synthesize_speech(text, provider, language, audio_path)
        except Exception as e:
            _log(f"\033[91mFailed to synthesize {_id}: {e}\033[0m")
            raise

        # Handle optional ttfb (some providers may not return it)
        ttfb = result.get("ttfb")
        if ttfb is not None:
            ttfb_values.append(ttfb)

        # Prepare row data
        row_data = {
            "id": _id,
            "text": text,
            "audio_path": audio_path,
            "ttfb": ttfb,
        }

        # Append to CSV immediately for crash recovery
        row_df = pd.DataFrame([row_data])
        if exists(results_csv_path):
            row_df.to_csv(results_csv_path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(results_csv_path, index=False)

        success_count += 1
        if ttfb is not None:
            _log(f"\n\033[93m  TTFB: {ttfb:.3f}s\033[0m")

    return {
        "success_count": success_count,
        "ttfb_values": ttfb_values,
    }


def validate_tts_input_file(input_path: str) -> tuple[bool, str]:
    """Validate TTS input CSV file.

    Expected format:
        id,text
        row_1,hello world
        row_2,this is a test

    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    # Check if file exists
    if not exists(input_path):
        return False, f"Input file does not exist: {input_path}"

    if not input_path.lower().endswith(".csv"):
        return False, f"Input must be a CSV file. Got: {input_path}"

    # Read CSV and validate columns
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        return False, f"Failed to read CSV file: {e}"

    if "id" not in df.columns:
        return (
            False,
            f"CSV file missing required column 'id'. Found columns: {list(df.columns)}",
        )

    if "text" not in df.columns:
        return (
            False,
            f"CSV file missing required column 'text'. Found columns: {list(df.columns)}",
        )

    if len(df) == 0:
        return False, "CSV file is empty (no rows found)"

    # Check for empty text values
    empty_texts = df[df["text"].isna() | (df["text"].astype(str).str.strip() == "")]
    if len(empty_texts) > 0:
        empty_ids = empty_texts["id"].tolist()[:5]
        if len(empty_texts) <= 5:
            return False, f"CSV has rows with empty text: {empty_ids}"
        else:
            return (
                False,
                f"CSV has {len(empty_texts)} rows with empty text. First 5 IDs: {empty_ids}",
            )

    return True, ""


# Expected base columns in results.csv for TTS evaluation
# (judge columns are dynamic based on criteria, so only check base columns)
TTS_RESULTS_COLUMNS = [
    "id",
    "text",
    "audio_path",
    "ttfb",
]


def validate_existing_results_csv(results_csv_path: str) -> tuple[bool, str]:
    """Validate existing results.csv file structure.

    Checks if the file is either empty or has the expected columns for TTS results.

    Args:
        results_csv_path: Path to the results.csv file

    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    if not exists(results_csv_path):
        return True, ""  # File doesn't exist, that's fine

    try:
        df = pd.read_csv(results_csv_path)
    except Exception as e:
        return False, f"Failed to read existing results.csv: {e}"

    # Empty file is valid (will be overwritten)
    if len(df) == 0:
        return True, ""

    # Check if all expected columns are present
    missing_columns = [col for col in TTS_RESULTS_COLUMNS if col not in df.columns]
    if missing_columns:
        return False, (
            f"Existing results.csv has incompatible structure. "
            f"Missing columns: {missing_columns}. "
            f"Expected columns: {TTS_RESULTS_COLUMNS}. "
            f"Found columns: {list(df.columns)}. "
            f"Use --overwrite to replace the file or delete it manually."
        )

    return True, ""


TTS_PROVIDERS = [
    "cartesia",
    "openai",
    "groq",
    "google",
    "gemini",
    "elevenlabs",
    "sarvam",
    "smallest",
]

TTS_LANGUAGES = [
    "english",
    "hindi",
    "kannada",
    "bengali",
    "malayalam",
    "marathi",
    "odia",
    "punjabi",
    "tamil",
    "telugu",
    "gujarati",
    "sindhi",
]


async def _score_and_write_results(
    ids: list,
    texts: list[str],
    audio_paths: list[str],
    output_dir: str,
    evaluator_config_dir: str,
    judge_evaluators: list[dict] = None,
    ttfb_values: list = None,
    cost_metrics: dict = None,
    overwrite: bool = False,
) -> dict:
    """Run the TTS audio judge over (audio_path, text) pairs and write outputs.

    Writes ``metrics.json`` and ``results.csv`` under ``output_dir`` and the
    resolved evaluator config under ``evaluator_config_dir``. Returns the
    metrics_data dict.

    When ``ttfb_values`` (aligned with ``ids``) is provided, a ``ttfb`` column
    and TTFB percentile metrics are included; when None (eval-only, where
    nothing is synthesized) both are omitted. When ``cost_metrics`` is provided
    it is attached under the ``cost`` key.

    A ``JudgeStore`` checkpoint (``judge_cache.jsonl``) is loaded from
    ``output_dir``, so a (row, evaluator) result already graded there is
    reused instead of re-billing the audio judge for it. ``overwrite=True``
    clears that checkpoint along with ``results.csv``, so a fresh run does
    not resurrect stale grades.
    """
    _log("Running evaluators...")
    _evaluators = judge_evaluators if judge_evaluators else [DEFAULT_TTS_EVALUATOR]
    require_unique_evaluator_names(_evaluators)
    write_evaluator_config(evaluator_config_dir, _evaluators)

    store = JudgeStore.load(output_dir)
    if overwrite:
        store.clear()
    cached_before = len(store)
    if cached_before:
        _log(
            f"Resuming from checkpoint: {cached_before} cached judge result(s) "
            f"found in {store.path}, skipping their judge calls"
        )

    llm_judge_results = await get_tts_llm_judge_score(
        audio_paths,
        texts,
        evaluators=_evaluators,
        store=store,
        row_ids=ids,
    )
    for name, score_dict in llm_judge_results["scores"].items():
        _log(f"  {name}: {score_dict['mean']:.4f}")

    # Map evaluator name → evaluator dict (for per-row value extraction).
    _evaluators_by_name = {ev["name"]: ev for ev in _evaluators}

    # Each evaluator gets one entry keyed by its name. The value is the full
    # per-criterion dict (``type``, ``mean``, plus ``scale_min``/``scale_max``
    # for ratings). Downstream consumers (leaderboard, summary print, UI)
    # detect evaluators as dict values that carry a ``type`` field.
    metrics_data = {}
    for name, score_dict in llm_judge_results["scores"].items():
        metrics_data[name] = score_dict

    if cost_metrics:
        metrics_data["cost"] = cost_metrics

    # TTFB percentile metrics (filter out None/NaN values). Only for the
    # synthesis path — eval-only passes ttfb_values=None.
    if ttfb_values is not None:
        valid_ttfb = [
            t
            for t in ttfb_values
            if t is not None and not (isinstance(t, float) and np.isnan(t))
        ]
        ttfb_pct = _latency_percentiles(valid_ttfb)
        if ttfb_pct is not None:
            metrics_data["ttfb"] = {
                "p50": float(ttfb_pct["p50"]),
                "p95": float(ttfb_pct["p95"]),
                "p99": float(ttfb_pct["p99"]),
                "count": ttfb_pct["count"],
            }

    metrics_save_path = join(output_dir, "metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    _log(f"Metrics saved to: {metrics_save_path}")

    ttfb_iter = ttfb_values if ttfb_values is not None else [None] * len(ids)
    data = []
    for _id, text, audio_path, ttfb, llm_row in zip(
        ids, texts, audio_paths, ttfb_iter, llm_judge_results["per_row"]
    ):
        row = {"id": _id, "text": text, "audio_path": audio_path}
        if ttfb_values is not None:
            row["ttfb"] = ttfb
        for name, ev in _evaluators_by_name.items():
            ev_result = llm_row[name]
            if is_rating(ev):
                row[name] = ev_result["score"]
            else:
                row[name] = bool(ev_result["match"])
            row[f"{name}_reasoning"] = ev_result["reasoning"]
        data.append(row)

    results_csv_path = join(output_dir, "results.csv")
    pd.DataFrame(data).to_csv(results_csv_path, index=False)
    _log(f"Results saved to: {results_csv_path}")

    return metrics_data


async def run_single_provider_eval(
    provider: str,
    language: str,
    input_file: str,
    output_dir: str,
    debug: bool,
    debug_count: int,
    overwrite: bool,
    judge_evaluators: list[dict] = None,
) -> dict:
    """Run TTS evaluation for a single provider."""
    provider_output_dir = os.path.join(output_dir, provider)
    os.makedirs(provider_output_dir, exist_ok=True)

    log_save_path = join(provider_output_dir, "logs")
    if exists(log_save_path):
        os.remove(log_save_path)

    # Drop any stale results.log left over from the previous (loguru-based) layout
    legacy_results_log = join(provider_output_dir, "results.log")
    if exists(legacy_results_log):
        os.remove(legacy_results_log)

    token = _current_log_file.set(log_save_path)
    try:
        _log("--------------------------------")
        _log(f"\033[33mRunning TTS evaluation for provider: {provider}\033[0m")

        # Validate language is supported by the provider
        validate_tts_language(language, provider)

        df = pd.read_csv(input_file)

        ids = df["id"].tolist()
        texts = df["text"].astype(str).tolist()

        if debug:
            ids = ids[:debug_count]
            texts = texts[:debug_count]

        gt_data = [{"id": _id, "text": text} for _id, text in zip(ids, texts)]

        results_csv_path = join(provider_output_dir, "results.csv")

        # Validate existing results.csv structure (if not overwriting)
        if not overwrite:
            is_valid, error_msg = validate_existing_results_csv(results_csv_path)
            if not is_valid:
                _log(f"\033[31mError: {error_msg}\033[0m")
                return {"provider": provider, "status": "error", "error": error_msg}

        _log(f"Processing {len(gt_data)} texts with provider: {provider}")
        _log("--------------------------------")

        # Run TTS evaluation
        eval_results = await run_tts_eval(
            gt_data=gt_data,
            provider=provider,
            language=language,
            output_dir=provider_output_dir,
            results_csv_path=results_csv_path,
            overwrite=overwrite,
        )

        _log("--------------------------------")
        _log(f"Successfully synthesized: {eval_results['success_count']} texts")

        # Reload the final results from CSV
        if exists(results_csv_path):
            final_df = pd.read_csv(results_csv_path)
            all_ids = final_df["id"].tolist()
            all_texts = final_df["text"].astype(str).tolist()
            all_audio_paths = final_df["audio_path"].tolist()
            all_ttfb = final_df["ttfb"].tolist()
        else:
            _log("No results found")
            return {
                "provider": provider,
                "status": "error",
                "error": "No results found",
            }

        # Run evaluators, write metrics.json + results.csv (evaluator config
        # goes to the parent output_dir, shared across providers).
        cost_metrics = _build_tts_cost_metrics(
            provider=provider,
            texts=all_texts,
            audio_paths=all_audio_paths,
            model=_default_tts_model(provider, language),
        )
        metrics_data = await _score_and_write_results(
            ids=all_ids,
            texts=all_texts,
            audio_paths=all_audio_paths,
            output_dir=provider_output_dir,
            evaluator_config_dir=output_dir,
            judge_evaluators=judge_evaluators,
            ttfb_values=all_ttfb,
            cost_metrics=cost_metrics,
            overwrite=overwrite,
        )

        return {
            "provider": provider,
            "status": "completed",
            "metrics": metrics_data,
            "output_dir": provider_output_dir,
        }
    finally:
        _current_log_file.reset(token)


def validate_tts_eval_only_dataset(run_dir: str) -> tuple[bool, str, list[dict]]:
    """Validate a TTS run directory for eval-only and resolve its audio paths.

    ``run_dir`` is a prior TTS run's output directory (e.g. ``./out/run/openai``)
    containing a ``results.csv`` with ``id``, ``text`` and ``audio_path``
    columns — no transformation needed. Extra columns (e.g. ``ttfb``) are
    ignored. Each ``audio_path`` is resolved to an existing absolute path,
    tried as given (relative to the CWD) and, as a fallback, under
    ``{run_dir}/audios/{basename}`` (the fixed run layout) so it resolves
    regardless of the CWD.

    Returns:
        tuple[bool, str, list[dict]]: (is_valid, error_message, parsed_rows)
    """
    if not exists(run_dir):
        return False, f"Dataset directory does not exist: {run_dir}", []
    if not os.path.isdir(run_dir):
        return False, f"--dataset must be a run directory, got a file: {run_dir}", []

    csv_path = join(run_dir, "results.csv")
    if not exists(csv_path):
        return False, f"No results.csv found in directory: {run_dir}", []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"Failed to read results.csv: {e}", []

    missing = {"id", "text", "audio_path"} - set(df.columns)
    if missing:
        return (
            False,
            f"results.csv missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}.",
            [],
        )

    if len(df) == 0:
        return False, f"results.csv is empty: {csv_path}", []

    rows = []
    for i, r in df[["id", "text", "audio_path"]].iterrows():
        audio_path = str(r["audio_path"])
        candidates = [audio_path]
        if not os.path.isabs(audio_path):
            candidates.append(
                join(run_dir, TTS_AUDIO_SUBDIR, os.path.basename(audio_path))
            )
        resolved = next((os.path.abspath(c) for c in candidates if exists(c)), None)
        if resolved is None:
            return False, f"Row {i} audio file does not exist: {audio_path}", []
        rows.append({"id": r["id"], "text": r["text"], "audio_path": resolved})

    return True, "", rows


async def run_eval_only(
    dataset_path: str,
    output_dir: str,
    judge_evaluators: list[dict] = None,
    overwrite: bool = False,
) -> dict:
    """Run the TTS audio judge only, on a prior run's audio. Skips synthesis
    and reads the run directory's ``results.csv`` directly. Writes
    ``metrics.json`` and ``results.csv`` under ``output_dir``.

    Args:
        dataset_path: A prior TTS run directory (e.g. ``./out/run/openai``)
            whose ``results.csv`` is read directly. See
            :func:`validate_tts_eval_only_dataset`.
        output_dir: Directory to write results and metrics.
        judge_evaluators: Optional list of evaluator dicts. Defaults to the
            built-in TTS evaluator (``DEFAULT_TTS_EVALUATOR``) when omitted.
        overwrite: If True, clear ``output_dir``'s judge checkpoint
            (``judge_cache.jsonl``) before scoring instead of resuming from it.

    Returns:
        dict with status, metrics, and output_dir.
    """
    # Refuse to write results back into the run dir being judged — that would
    # overwrite its results.csv/metrics.json (losing the run's ttfb column).
    if os.path.abspath(output_dir) == os.path.abspath(dataset_path):
        return {
            "status": "error",
            "error": (
                f"--output-dir must differ from the run dir being judged: {dataset_path}. "
                "Choose a different -o so the run's results.csv/metrics.json aren't overwritten."
            ),
        }

    os.makedirs(output_dir, exist_ok=True)

    log_save_path = join(output_dir, "logs")
    if exists(log_save_path):
        os.remove(log_save_path)

    token = _current_log_file.set(log_save_path)
    try:
        _log("--------------------------------")
        _log("\033[33mRunning TTS eval-only on dataset\033[0m")
        _log(f"Dataset: {dataset_path}")

        is_valid, error_msg, rows = validate_tts_eval_only_dataset(dataset_path)
        if not is_valid:
            _log(f"\033[31mError: {error_msg}\033[0m")
            return {"status": "error", "error": error_msg}

        ids = [r["id"] for r in rows]
        texts = [str(r["text"]) for r in rows]
        audio_paths = [r["audio_path"] for r in rows]

        # No synthesis here, so no TTFB: ttfb_values=None omits the column and
        # the percentile metrics. Metrics + results + evaluator config all land
        # in ``output_dir``.
        metrics_data = await _score_and_write_results(
            ids=ids,
            texts=texts,
            audio_paths=audio_paths,
            output_dir=output_dir,
            evaluator_config_dir=output_dir,
            judge_evaluators=judge_evaluators,
            ttfb_values=None,
            overwrite=overwrite,
        )

        return {
            "status": "completed",
            "metrics": metrics_data,
            "output_dir": output_dir,
        }
    finally:
        _current_log_file.reset(token)


async def main():
    """CLI entry point for single-provider TTS evaluation.

    Used by the Ink UI which spawns individual provider processes.
    For multi-provider benchmark, use benchmark.py via `calibrate-agent tts -p provider1 provider2 ...`
    """
    parser = argparse.ArgumentParser(
        description="Single-provider TTS evaluation (used by Ink UI)"
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        required=True,
        help="TTS provider to use for evaluation",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="english",
        choices=TTS_LANGUAGES,
        help="Language of the audio files",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file containing the texts to synthesize",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./out",
        help="Path to the output directory to save the results",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Run the evaluation on the first N texts only",
    )
    parser.add_argument(
        "-dc",
        "--debug_count",
        help="Number of texts to run the evaluation on",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of resuming from last checkpoint",
    )
    args = parser.parse_args()

    provider = args.provider

    # Validate provider
    if provider not in TTS_PROVIDERS:
        print(f"\033[31mError: Invalid provider '{provider}'.\033[0m")
        print(f"Available providers: {', '.join(TTS_PROVIDERS)}")
        sys.exit(1)

    # Validate input CSV file
    is_valid, error_msg = validate_tts_input_file(args.input)
    if not is_valid:
        print(f"\033[31mInput validation error: {error_msg}\033[0m")
        sys.exit(1)

    # ``exist_ok=True`` makes this safe when several ``calibrate-agent tts``
    # subprocesses race to create the output dir on first use; the previous
    # ``if not exists: makedirs(...)`` pattern was non-atomic and the loser
    # raised ``FileExistsError``.
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n\033[91mTTS Evaluation\033[0m\n")
    print(f"Provider: {provider}")
    print(f"Language: {args.language}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print("")

    # Run single provider evaluation
    result = await run_single_provider_eval(
        provider=provider,
        language=args.language,
        input_file=args.input,
        output_dir=args.output_dir,
        debug=args.debug,
        debug_count=args.debug_count,
        overwrite=args.overwrite,
    )

    # Print summary
    print(f"\n\033[92m{'='*60}\033[0m")
    print(f"\033[92mSummary\033[0m")
    print(f"\033[92m{'='*60}\033[0m\n")

    if result.get("status") == "error":
        print(f"  {provider}: \033[31mError - {result.get('error')}\033[0m")
    else:
        metrics = result.get("metrics", {})
        # Evaluator entries are dicts carrying a ``type`` field; ttfb has no
        # ``type`` so it's correctly excluded from the judge-score string.
        judge_scores = {
            k: v["mean"]
            for k, v in metrics.items()
            if isinstance(v, dict) and "type" in v
        }
        ttfb_data = metrics.get("ttfb", {})
        ttfb_p50 = (
            ttfb_data.get("p50", "N/A") if isinstance(ttfb_data, dict) else "N/A"
        )
        judge_str = ", ".join(f"{k}={v:.2f}" for k, v in judge_scores.items())
        ttfb_str = (
            f"TTFB(p50)={ttfb_p50:.3f}s"
            if isinstance(ttfb_p50, float)
            else f"TTFB(p50)={ttfb_p50}"
        )
        print(f"  {provider}: {judge_str}, {ttfb_str}")


if __name__ == "__main__":
    asyncio.run(main())
