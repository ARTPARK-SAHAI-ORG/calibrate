import asyncio
import argparse
import sys
import os
import json
import base64
import time
import httpx
from os.path import join, exists
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlencode
import backoff
from sarvamai import AsyncSarvamAI
from openai import AsyncOpenAI
from groq import AsyncGroq
from cartesia import AsyncCartesia
import uuid
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types
from google.api_core.client_options import ClientOptions

import numpy as np
import pandas as pd

from calibrate.utils import (
    get_stt_language_code,
    validate_stt_language,
    provider_log as _log,
    provider_log_file as _current_log_file,
)
from calibrate.stt.metrics import (
    get_wer_score,
    get_llm_judge_score,
)
from calibrate.judges import (
    is_rating,
    DEFAULT_STT_EVALUATOR,
    require_unique_evaluator_names,
    write_evaluator_config,
)
from calibrate.langfuse import (
    create_langfuse_audio_media,
    observe,
    langfuse,
    langfuse_enabled,
)
from calibrate.rate_limit import (
    STT_PROVIDER_TIMEOUT_SECONDS,
    STT_STREAMING_IDLE_TIMEOUT_SECONDS,
    SARVAM_STT_STREAMING_LIMITER,
)


# =============================================================================
# STT Provider API Methods
# =============================================================================


def load_audio(audio_path: Path, as_file: bool = False, raw_pcm: bool = False):
    """
    Load audio file and convert to mono 16 kHz, 16-bit audio.

    Args:
        audio_path: Path to audio file.
        as_file: If True, return a file-like BytesIO object. If False, return bytes.
        raw_pcm: If True, return raw PCM bytes instead of WAV bytes.

    Returns:
        Bytes or BytesIO of audio in mono, 16 kHz, 16-bit PCM format.
    """
    import io

    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for audio conversion. Install with 'pip install pydub'."
        )

    # Load audio using pydub (auto-detects format)
    audio = AudioSegment.from_file(audio_path)
    # Convert to mono, 16 kHz, 16-bit PCM
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio = audio.normalize()
    audio = audio.strip_silence(silence_len=100, silence_thresh=-40)

    if raw_pcm:
        return audio.raw_data

    # Export to WAV bytes
    out_io = io.BytesIO()
    audio.export(out_io, format="wav")

    if as_file:
        out_io.seek(0)  # Reset position to start for reading
        out_io.name = "audio.wav"  # Set filename for APIs that need it
        return out_io

    return out_io.getvalue()


async def _aiter_with_idle_timeout(aiter, timeout: float):
    """Async-iterate ``aiter`` enforcing a per-message idle timeout.

    Raises ``asyncio.TimeoutError`` if no next message arrives within
    ``timeout`` seconds. Useful for STT WebSocket loops where the SDK's
    own timeout settings don't apply to the receive side and a stalled
    server can otherwise block forever.
    """
    it = aiter.__aiter__() if hasattr(aiter, "__aiter__") else aiter
    while True:
        try:
            msg = await asyncio.wait_for(it.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        yield msg


async def transcribe_deepgram_streaming(audio_path: Path, language: str) -> str:
    """Transcribe audio using Deepgram's live streaming WebSocket API.

    Uses the raw WebSocket protocol (``wss://api.deepgram.com/v1/listen``)
    rather than the threaded SDK because the SDK's live client is sync /
    callback-based and doesn't compose with our async pipeline. See
    https://developers.deepgram.com/docs/live-streaming-audio.
    """
    try:
        from websockets.asyncio.client import connect as websocket_connect
    except ModuleNotFoundError as e:
        raise ImportError(
            "websockets is required for Deepgram streaming STT. "
            "Install with 'pip install websockets'."
        ) from e

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "deepgram")

    endpoint = "wss://api.deepgram.com/v1/listen"
    params = {
        "model": "nova-3",
        "language": lang_code,
        "encoding": "linear16",
        "sample_rate": "16000",
        "channels": "1",
        "smart_format": "true",
    }
    ws_url = f"{endpoint}?{urlencode(params)}"
    headers = {"Authorization": f"Token {api_key}"}

    audio = load_audio(audio_path, raw_pcm=True)
    chunk_size = 4096

    async with websocket_connect(ws_url, additional_headers=headers) as ws:

        async def send_audio():
            for start in range(0, len(audio), chunk_size):
                chunk = audio[start : start + chunk_size]
                if chunk:
                    await ws.send(chunk)
                    await asyncio.sleep(0.05)

            # Tells Deepgram we've sent all audio so it flushes and closes.
            await ws.send(json.dumps({"type": "CloseStream"}))

        start_time = time.perf_counter()
        ttft = None
        sender = asyncio.create_task(send_audio())
        transcript_parts = []

        try:
            async for message in _aiter_with_idle_timeout(
                ws, STT_STREAMING_IDLE_TIMEOUT_SECONDS
            ):
                try:
                    output = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    continue

                if not isinstance(output, dict):
                    continue

                msg_type = output.get("type")

                if msg_type == "Results":
                    alternatives = output.get("channel", {}).get("alternatives") or []
                    if not alternatives:
                        continue

                    transcript = alternatives[0].get("transcript", "")
                    if transcript and ttft is None:
                        ttft = time.perf_counter() - start_time

                    if output.get("is_final") and transcript:
                        transcript_parts.append(transcript)
                elif msg_type == "Metadata":
                    # Sent after CloseStream once Deepgram has flushed all
                    # final transcripts — safe to stop reading.
                    break
        finally:
            if sender.done():
                await sender
            else:
                sender.cancel()
                try:
                    await sender
                except asyncio.CancelledError:
                    pass

    return {
        "transcript": " ".join(
            part.strip() for part in transcript_parts if part.strip()
        ),
        "ttft": ttft,
    }


async def transcribe_openai_streaming(audio_path: Path, language: str) -> str:
    """Transcribe audio using OpenAI's transcriptions API with ``stream=True``.

    Streams ``transcript.text.delta`` events as soon as each segment is ready
    and finishes on ``transcript.text.done`` which carries the full transcript.
    See https://developers.openai.com/api/docs/guides/speech-to-text#streaming.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = AsyncOpenAI()

    audio_file = load_audio(audio_path, as_file=True)

    stream = await client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file,
        response_format="text",
        stream=True,
    )

    start_time = time.perf_counter()
    ttft = None
    transcript = ""

    async for event in _aiter_with_idle_timeout(
        stream, STT_STREAMING_IDLE_TIMEOUT_SECONDS
    ):
        event_type = getattr(event, "type", None)

        if event_type == "transcript.text.delta":
            if ttft is None and getattr(event, "delta", ""):
                ttft = time.perf_counter() - start_time

        elif event_type == "transcript.text.done":
            transcript = getattr(event, "text", "") or ""
            if ttft is None:
                ttft = time.perf_counter() - start_time
            break

    return {
        "transcript": transcript,
        "ttft": ttft,
    }


async def transcribe_groq(audio_path: Path, language: str) -> str:
    """Transcribe audio using Groq's Whisper API."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "groq")

    client = AsyncGroq(api_key=api_key)

    audio_file = load_audio(audio_path, as_file=True)

    transcription = await asyncio.wait_for(
        client.audio.transcriptions.create(
            file=audio_file,  # Required audio file
            model="whisper-large-v3-turbo",  # Required model to use for transcription
            response_format="text",  # Optional
            language=lang_code,  # Optional
            temperature=0.0,  # Optional
        ),
        timeout=STT_PROVIDER_TIMEOUT_SECONDS,
    )

    return {
        "transcript": transcription.strip(),
    }


def _transcribe_google_streaming(
    audio_path: Path,
    lang_code: str,
    model: str = "chirp_3",
    region: str = "us",
) -> cloud_speech_types.StreamingRecognizeResponse:
    """Transcribes audio from an audio file stream using Google Cloud Speech-to-Text API.
    Args:
        stream_file (str): Path to the local audio file to be transcribed.
            Example: "resources/audio.wav"
        model (str): The model to use for transcription (default: chirp_3)
        region (str): The region for the API endpoint (default: us)
    Returns:
        list[cloud_speech_types.StreamingRecognizeResponse]: A list of objects.
            Each response includes the transcription results for the corresponding audio segment.
    """
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")

    # Instantiates a client
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{region}-speech.googleapis.com",
        )
    )

    # Reads a file as bytes
    audio_content = load_audio(audio_path)

    # In practice, stream should be a generator yielding chunks of audio data
    # Chunk size must be < 25KB per Google STT API limitations
    # Use 24KB for a safe margin
    max_chunk_size = 24 * 1024  # 24KB = 24 * 1024 bytes
    stream = [
        audio_content[start : start + max_chunk_size]
        for start in range(0, len(audio_content), max_chunk_size)
    ]
    audio_requests = (
        cloud_speech_types.StreamingRecognizeRequest(audio=audio) for audio in stream
    )

    recognition_config = cloud_speech_types.RecognitionConfig(
        auto_decoding_config=cloud_speech_types.AutoDetectDecodingConfig(),
        language_codes=[lang_code],
        model=model,
    )
    streaming_config = cloud_speech_types.StreamingRecognitionConfig(
        config=recognition_config
    )
    config_request = cloud_speech_types.StreamingRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{region}/recognizers/_",
        streaming_config=streaming_config,
    )

    def requests(
        config: cloud_speech_types.StreamingRecognizeRequest,
        audio: list,
    ) -> list:
        yield config
        for req in audio:
            yield req
            # Pace 24KB chunks to avoid flooding Google's streaming gRPC endpoint.
            time.sleep(0.5)

    # Transcribes the audio into text
    responses_iterator = client.streaming_recognize(
        requests=requests(config_request, audio_requests)
    )
    all_interim_transcripts = []

    for response in responses_iterator:
        for result in response.results:
            interim_transcript = result.alternatives[0].transcript.strip()
            if not interim_transcript:
                continue

            all_interim_transcripts.append(interim_transcript)

    return " ".join(all_interim_transcripts)


async def transcribe_google(audio_path: Path, language: str) -> str:
    """Transcribe audio using Google Cloud Speech-to-Text API."""
    from google.cloud import speech_v1 as speech

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

    lang_code = get_stt_language_code(language, "google")

    # Use chirp_2 model and asia-southeast1 region for Sindhi
    if language.lower() == "sindhi":
        model = "chirp_2"
        region = "asia-southeast1"
    else:
        model = "chirp_3"
        region = "us"

    transcript = await asyncio.wait_for(
        asyncio.to_thread(
            _transcribe_google_streaming,
            audio_path,
            lang_code,
            model,
            region,
        ),
        timeout=STT_PROVIDER_TIMEOUT_SECONDS,
    )

    return {
        "transcript": transcript.strip(),
    }


async def transcribe_sarvam(audio_path: Path, language: str) -> str:
    """Transcribe audio using Sarvam's STT API."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "sarvam")

    audio_data = base64.b64encode(load_audio(audio_path)).decode("utf-8")

    await SARVAM_STT_STREAMING_LIMITER.acquire()

    client = AsyncSarvamAI(api_subscription_key=api_key)

    transcript = ""
    ttft = None

    async with client.speech_to_text_streaming.connect(
        language_code=lang_code,
        model="saaras:v3",
        mode="transcribe",
        flush_signal=True,
    ) as ws:
        # Send audio
        await ws.transcribe(audio=audio_data, encoding="audio/wav", sample_rate=16000)

        # Force immediate processing
        await ws.flush()
        _log("⚡ Processing forced - getting immediate results")

        # Get results
        async for message in _aiter_with_idle_timeout(
            ws, STT_STREAMING_IDLE_TIMEOUT_SECONDS
        ):
            if getattr(message, "type", None) == "error":
                error = getattr(message.data, "error", "Unknown Sarvam STT error")
                raise RuntimeError(error)
            if getattr(message, "type", None) != "data":
                continue

            transcript = getattr(message.data, "transcript", "")
            metrics = getattr(message.data, "metrics", None)
            ttft = getattr(metrics, "processing_latency", None)
            break

    return {
        "transcript": transcript,
        "ttft": ttft,
    }


async def transcribe_cartesia(audio_path: Path, language: str) -> str:
    """Transcribe audio using Cartesia's STT API."""
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        raise ValueError("CARTESIA_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "cartesia")

    client = AsyncCartesia(api_key=api_key)

    try:
        # Create websocket connection with voice activity detection
        ws = await client.stt.websocket(
            model="ink-whisper",  # Model (required)
            language=lang_code,  # Language of your audio (required)
            encoding="pcm_s16le",  # Audio encoding format (required)
            sample_rate=16000,  # Audio sample rate (required)
            min_volume=0.15,  # Volume threshold for voice activity detection
            max_silence_duration_secs=0.3,  # Maximum silence duration before endpointing
        )

        # Simulate streaming audio data (replace with your audio source)
        async def audio_stream():
            """Simulate real-time audio streaming - replace with actual audio capture"""
            # Load audio file for simulation
            audio_data = load_audio(audio_path)

            # Stream in 100ms chunks (realistic for real-time processing)
            chunk_size = int(16000 * 0.1 * 2)  # 100ms at 16kHz, 16-bit

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                if chunk:
                    yield chunk
                    # Simulate real-time streaming delay
                    await asyncio.sleep(0.1)

        # Send audio and receive results concurrently
        async def send_audio():
            """Send audio chunks to the STT websocket"""
            async for chunk in audio_stream():
                await ws.send(chunk)
                # print(f"Sent audio chunk of {len(chunk)} bytes")
                # Small delay to simulate realtime applications
                await asyncio.sleep(0.02)

            # Signal end of audio stream
            await ws.send("finalize")
            await ws.send("done")
            # print("Audio streaming completed")

        start_time = time.perf_counter()
        ttft_holder = {"ttft": None}

        async def receive_transcripts():
            """Receive and process transcription results with word timestamps"""
            full_transcript = ""

            async for result in _aiter_with_idle_timeout(
                ws.receive(), STT_STREAMING_IDLE_TIMEOUT_SECONDS
            ):
                if result["type"] == "transcript":
                    text = result["text"]
                    is_final = result["is_final"]

                    if text and ttft_holder["ttft"] is None:
                        ttft_holder["ttft"] = time.perf_counter() - start_time

                    if is_final:
                        full_transcript += text + " "

                elif result["type"] == "done":
                    break

            return full_transcript.strip()

        _, final_transcript = await asyncio.gather(send_audio(), receive_transcripts())

        await ws.close()

        return {"transcript": final_transcript, "ttft": ttft_holder["ttft"]}

    finally:
        await client.close()


async def transcribe_smallest_streaming(audio_path: Path, language: str) -> str:
    """Transcribe audio using Smallest's Pulse STT WebSocket API."""
    try:
        from websockets.asyncio.client import connect as websocket_connect
    except ModuleNotFoundError as e:
        raise ImportError(
            "websockets is required for Smallest streaming STT. "
            "Install with 'pip install websockets'."
        ) from e

    api_key = os.getenv("SMALLEST_API_KEY")
    if not api_key:
        raise ValueError("SMALLEST_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "smallest")
    endpoint = "wss://api.smallest.ai/waves/v1/pulse/get_text"
    params = {
        "language": lang_code,
        "encoding": "linear16",
        "sample_rate": "16000",
        "word_timestamps": "false",
    }
    ws_url = f"{endpoint}?{urlencode(params)}"
    headers = {"Authorization": f"Bearer {api_key}"}
    audio = load_audio(audio_path, raw_pcm=True)
    chunk_size = 4096

    async with websocket_connect(ws_url, additional_headers=headers) as ws:

        async def send_audio():
            for start in range(0, len(audio), chunk_size):
                chunk = audio[start : start + chunk_size]
                if chunk:
                    await ws.send(chunk)
                    await asyncio.sleep(0.05)

            await ws.send(json.dumps({"type": "close_stream"}))

        start_time = time.perf_counter()
        ttft = None
        sender = asyncio.create_task(send_audio())
        transcript_parts = []

        try:
            async for message in _aiter_with_idle_timeout(
                ws, STT_STREAMING_IDLE_TIMEOUT_SECONDS
            ):
                try:
                    output = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if not isinstance(output, dict):
                    continue

                if output.get("type") == "error" or output.get("error"):
                    error = output.get("message") or output.get("error") or output
                    raise RuntimeError(f"Smallest streaming STT error: {error}")

                transcript = output.get("transcript", "")
                if transcript and ttft is None:
                    ttft = time.perf_counter() - start_time

                if output.get("is_final") and transcript:
                    transcript_parts.append(transcript)

                if output.get("is_last"):
                    break
        finally:
            if sender.done():
                await sender
            else:
                sender.cancel()
                try:
                    await sender
                except asyncio.CancelledError:
                    pass

    return {
        "transcript": " ".join(
            part.strip() for part in transcript_parts if part.strip()
        ),
        "ttft": ttft,
    }


async def transcribe_elevenlabs_streaming(audio_path: Path, language: str) -> str:
    """Transcribe audio using ElevenLabs' Scribe v2 Realtime WebSocket API.

    Uses the documented raw-WebSocket protocol (manual commit strategy) rather
    than the SDK's higher-level ``speech_to_text.realtime.connect`` because the
    installed elevenlabs SDK version doesn't ship the realtime STT client.
    See https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/server-side-streaming.
    """
    try:
        from websockets.asyncio.client import connect as websocket_connect
    except ModuleNotFoundError as e:
        raise ImportError(
            "websockets is required for ElevenLabs streaming STT. "
            "Install with 'pip install websockets'."
        ) from e

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "elevenlabs")

    endpoint = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    params = {
        "model_id": "scribe_v2_realtime",
        "audio_format": "pcm_16000",
        "commit_strategy": "manual",
        "language_code": lang_code,
    }
    ws_url = f"{endpoint}?{urlencode(params)}"
    headers = {"xi-api-key": api_key}

    audio = load_audio(audio_path, raw_pcm=True)
    chunk_size = 32000  # 1s of 16 kHz, 16-bit mono PCM

    async with websocket_connect(ws_url, additional_headers=headers) as ws:
        # The server emits ``session_started`` before accepting audio.
        await asyncio.wait_for(ws.recv(), timeout=STT_STREAMING_IDLE_TIMEOUT_SECONDS)

        async def send_audio():
            for start in range(0, len(audio), chunk_size):
                chunk = audio[start : start + chunk_size]
                if chunk:
                    await ws.send(
                        json.dumps(
                            {
                                "message_type": "input_audio_chunk",
                                "audio_base_64": base64.b64encode(chunk).decode(
                                    "utf-8"
                                ),
                                "commit": False,
                                "sample_rate": 16000,
                            }
                        )
                    )
                    await asyncio.sleep(0.05)

            # Final commit finalises the segment and triggers committed_transcript.
            await ws.send(
                json.dumps(
                    {
                        "message_type": "input_audio_chunk",
                        "audio_base_64": "",
                        "commit": True,
                        "sample_rate": 16000,
                    }
                )
            )

        start_time = time.perf_counter()
        ttft = None
        sender = asyncio.create_task(send_audio())
        transcript_parts = []

        try:
            async for message in _aiter_with_idle_timeout(
                ws, STT_STREAMING_IDLE_TIMEOUT_SECONDS
            ):
                try:
                    output = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if not isinstance(output, dict):
                    continue

                message_type = output.get("message_type")

                if message_type == "input_error":
                    raise RuntimeError(f"ElevenLabs streaming STT error: {output}")

                if message_type == "partial_transcript":
                    if output.get("text") and ttft is None:
                        ttft = time.perf_counter() - start_time

                if message_type == "committed_transcript":
                    text = output.get("text", "")
                    if text:
                        if ttft is None:
                            ttft = time.perf_counter() - start_time
                        transcript_parts.append(text)
                    break
        finally:
            if sender.done():
                await sender
            else:
                sender.cancel()
                try:
                    await sender
                except asyncio.CancelledError:
                    pass

    return {
        "transcript": " ".join(
            part.strip() for part in transcript_parts if part.strip()
        ),
        "ttft": ttft,
    }


# =============================================================================
# Main Transcription Router
# =============================================================================


@backoff.on_exception(backoff.expo, Exception, max_tries=3, factor=2)
@observe(name="stt", capture_input=False, capture_output=False)
async def transcribe_audio(
    audio_path: Path,
    reference: str,
    provider: str,
    language: str,
    unique_id: str,
) -> str:
    """Route audio transcription to the appropriate provider."""
    provider_methods = {
        "deepgram": transcribe_deepgram_streaming,
        "openai": transcribe_openai_streaming,
        "groq": transcribe_groq,
        "google": transcribe_google,
        "sarvam": transcribe_sarvam,
        "elevenlabs": transcribe_elevenlabs_streaming,
        "cartesia": transcribe_cartesia,
        "smallest": transcribe_smallest_streaming,
    }

    if provider not in provider_methods:
        raise ValueError(f"Unsupported STT provider: {provider}")

    method = provider_methods[provider]
    output = await method(audio_path, language)

    transcript = output["transcript"].strip()
    ttft = output.get("ttft")

    if langfuse_enabled and langfuse:
        # Download the audio from path and add to input in langfuse
        input_audio_media = create_langfuse_audio_media(audio_path)

        langfuse.update_current_trace(
            input={
                "audio": input_audio_media,
                "reference": reference,
                "language": language,
                "provider": provider,
            },
            output=transcript,
            metadata={
                "provider": provider,
                "language": language,
                "reference": reference,
                "ttft": ttft,
            },
            session_id=unique_id,
        )

    return {"transcript": transcript, "ttft": ttft}


# =============================================================================
# STT Evaluation Main
# =============================================================================


async def run_stt_eval(
    gt_data: List[Dict],
    audio_dir: Path,
    provider: str,
    language: str,
    results_csv_path: Path,
) -> int:
    """Process audio files and save results immediately to CSV.

    Row-level failures (after ``transcribe_audio``'s backoff retries are
    exhausted) are recorded as ``status="failed"`` with an ``error`` message
    in the CSV row and do NOT abort the run. The loop continues so a single
    bad row can't sink an otherwise-good benchmark. A summary of all failed
    rows is logged at the end.

    Args:
        gt_data: List of {"id": ..., "gt": ...} for each file to process
        audio_dir: Directory containing audio files
        provider: STT provider name
        language: Language code
        results_csv_path: Path to save results CSV

    Returns:
        Number of files successfully transcribed (non-empty) in this run.
    """
    # Load existing results to skip already processed files
    if exists(results_csv_path):
        existing_df = pd.read_csv(results_csv_path)
        results = existing_df.to_dict("records")
        processed_ids = set(existing_df["id"].tolist())
    else:
        results = []
        processed_ids = set()

    success_count = 0
    row_errors: List[tuple] = []

    unique_id = str(uuid.uuid4())

    for i, gt_info in enumerate(gt_data):
        # Skip if already processed
        if gt_info["id"] in processed_ids:
            continue

        audio_path = audio_dir / f"{gt_info['id']}.wav"

        _log(f"--------------------------------")
        _log(f"Processing audio [{i + 1}/{len(gt_data)}]: {audio_path.name}")

        transcript = ""
        ttft = None
        status = "success"
        error = None
        try:
            output = await transcribe_audio(
                audio_path, gt_info["gt"], provider, language, unique_id
            )
            transcript = output["transcript"]
            ttft = output.get("ttft")
            _log(f"\033[33mTranscript: {transcript}\033[0m")
            if ttft is not None:
                _log(f"TTFT: {ttft:.3f}s")
            if transcript:
                success_count += 1
        except Exception as e:
            status = "failed"
            error = f"{type(e).__name__}: {e}"
            row_errors.append((gt_info["id"], error))
            _log(f"\033[91mFailed to transcribe {audio_path}: {error}\033[0m")

        # Save immediately after each file
        results.append(
            {
                "id": gt_info["id"],
                "gt": gt_info["gt"],
                "pred": transcript,
                "ttft": ttft,
                "status": status,
                "error": error,
            }
        )
        pd.DataFrame(results).to_csv(results_csv_path, index=False)

    if row_errors:
        _log("--------------------------------")
        _log(
            f"\033[91m{len(row_errors)} row(s) failed for provider '{provider}':\033[0m"
        )
        for row_id, err in row_errors:
            _log(f"  - {row_id}: {err}")

    return success_count


def validate_stt_input_dir(input_dir: str, input_file_name: str) -> tuple[bool, str]:
    """Validate STT input directory structure.

    Expected structure:
        input_dir/
        ├── stt.csv (or custom input_file_name)
        └── audios/
            ├── audio_1.wav
            └── audio_2.wav

    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    input_path = Path(input_dir)

    # Check if directory exists
    if not input_path.exists():
        return False, f"Input directory does not exist: {input_dir}"

    if not input_path.is_dir():
        return False, f"Input path is not a directory: {input_dir}"

    # Check if CSV file exists
    csv_path = input_path / input_file_name
    if not csv_path.exists():
        return False, f"CSV file not found: {csv_path}"

    # Check if audios directory exists
    audios_dir = input_path / "audios"
    if not audios_dir.exists():
        return False, f"Audios directory not found: {audios_dir}"

    if not audios_dir.is_dir():
        return False, f"Audios path is not a directory: {audios_dir}"

    # Read CSV and validate columns
    try:
        df = pd.read_csv(csv_path)
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

    # Check if all audio files referenced in CSV exist
    missing_files = []
    for row_id in df["id"]:
        audio_path = audios_dir / f"{row_id}.wav"
        if not audio_path.exists():
            missing_files.append(f"{row_id}.wav")

    if missing_files:
        if len(missing_files) <= 5:
            return False, f"Missing audio files in audios/: {', '.join(missing_files)}"
        else:
            return (
                False,
                f"Missing {len(missing_files)} audio files in audios/. First 5: {', '.join(missing_files[:5])}",
            )

    return True, ""


# Expected columns in results.csv for STT evaluation
STT_RESULTS_COLUMNS = [
    "id",
    "gt",
    "pred",
]


def validate_existing_results_csv(results_csv_path: str) -> tuple[bool, str]:
    """Validate existing results.csv file structure.

    Checks if the file is either empty or has the expected columns for STT results.

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
    missing_columns = [col for col in STT_RESULTS_COLUMNS if col not in df.columns]
    if missing_columns:
        return False, (
            f"Existing results.csv has incompatible structure. "
            f"Missing columns: {missing_columns}. "
            f"Expected columns: {STT_RESULTS_COLUMNS}. "
            f"Found columns: {list(df.columns)}. "
            f"Use --overwrite to replace the file or delete it manually."
        )

    return True, ""


STT_PROVIDERS = [
    "deepgram",
    "openai",
    "cartesia",
    "smallest",
    "groq",
    "google",
    "sarvam",
    "elevenlabs",
]

STT_LANGUAGES = [
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


async def run_single_provider_eval(
    provider: str,
    language: str,
    input_dir: str,
    input_file_name: str,
    output_dir: str,
    debug: bool,
    debug_count: int,
    ignore_retry: bool,
    overwrite: bool,
    judge_evaluators: list[dict] = None,
) -> dict:
    """Run STT evaluation for a single provider."""
    provider_output_dir = join(output_dir, provider)

    # ``exist_ok=True`` keeps this safe when the same provider folder is
    # created concurrently by multiple eval coroutines/subprocesses.
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
        _log(f"\033[33mRunning STT evaluation for provider: {provider}\033[0m")

        # Validate language is supported by the provider
        validate_stt_language(language, provider)

        # Audio files are expected in audios/*.wav
        audio_dir = Path(input_dir) / "audios"
        gt_file = join(input_dir, input_file_name)
        results_csv_path = Path(provider_output_dir) / "results.csv"

        # Validate existing results.csv structure (if not overwriting)
        if not overwrite:
            is_valid, error_msg = validate_existing_results_csv(str(results_csv_path))
            if not is_valid:
                _log(f"\033[31mError: {error_msg}\033[0m")
                return {"provider": provider, "status": "error", "error": error_msg}

        # Delete existing results if overwrite is set
        if overwrite and exists(results_csv_path):
            os.remove(results_csv_path)
            _log("Overwrite enabled - deleted existing results.csv")

        gt = pd.read_csv(gt_file)

        if debug:
            _log(
                f"running in debug mode: using first {debug_count} audio files for evaluation",
                to_terminal=False,
            )
            gt = gt.head(debug_count)

        total_expected = len(gt)
        gt_data = [{"id": row["id"], "gt": row["text"]} for _, row in gt.iterrows()]

        # Process with retry loop
        previous_processed_count = -1

        while True:
            # Check current progress
            if exists(results_csv_path):
                current_df = pd.read_csv(results_csv_path)
                current_processed = len(current_df)

                if current_processed >= total_expected:
                    _log(f"All {total_expected} audio files processed")
                    break

                _log(f"Progress: {current_processed}/{total_expected} processed")
            else:
                current_processed = 0

            # Check if no progress was made
            if current_processed == previous_processed_count:
                _log(
                    f"No progress made - {total_expected - current_processed} files failed. "
                    f"Saving empty transcripts and exiting."
                )
                # Add empty transcripts for unprocessed files
                if exists(results_csv_path):
                    results = pd.read_csv(results_csv_path).to_dict("records")
                    processed_ids = {r["id"] for r in results}
                else:
                    results = []
                    processed_ids = set()

                for gt_info in gt_data:
                    if gt_info["id"] not in processed_ids:
                        results.append(
                            {
                                "id": gt_info["id"],
                                "gt": gt_info["gt"],
                                "pred": "",
                                "ttft": None,
                                "status": "failed",
                                "error": "not processed (no progress after retry)",
                            }
                        )

                pd.DataFrame(results).to_csv(results_csv_path, index=False)
                break

            previous_processed_count = current_processed

            # Run transcription
            success_count = await run_stt_eval(
                gt_data=gt_data,
                audio_dir=audio_dir,
                provider=provider,
                language=language,
                results_csv_path=results_csv_path,
            )

            if ignore_retry:
                break

        # Load final results for metrics
        results_df = pd.read_csv(results_csv_path)
        all_ids = results_df["id"].tolist()
        all_gt_transcripts = results_df["gt"].astype(str).tolist()
        all_pred_transcripts = results_df["pred"].fillna("").astype(str).tolist()

        if "ttft" in results_df.columns:
            all_ttfts = [
                None if pd.isna(v) else float(v) for v in results_df["ttft"].tolist()
            ]
        else:
            all_ttfts = [None] * len(all_ids)
        if "status" in results_df.columns:
            all_statuses = results_df["status"].fillna("success").astype(str).tolist()
        else:
            all_statuses = ["success"] * len(all_ids)
        if "error" in results_df.columns:
            all_errors = [
                None if pd.isna(v) else str(v) for v in results_df["error"].tolist()
            ]
        else:
            all_errors = [None] * len(all_ids)

        _log(f"gt_transcripts: {all_gt_transcripts}", to_terminal=False)
        _log(f"pred_transcripts: {all_pred_transcripts}", to_terminal=False)

        # Evaluator config is written at the parent ``output_dir`` (shared
        # across providers in a benchmark run), while per-provider results
        # live in ``provider_output_dir``.
        metrics_data = await _score_and_write_results(
            ids=all_ids,
            gt_transcripts=all_gt_transcripts,
            pred_transcripts=all_pred_transcripts,
            output_dir=provider_output_dir,
            evaluator_config_dir=output_dir,
            judge_evaluators=judge_evaluators,
            ttfts=all_ttfts,
            statuses=all_statuses,
            errors=all_errors,
        )

        return {
            "provider": provider,
            "status": "completed",
            "metrics": metrics_data,
            "output_dir": provider_output_dir,
        }
    finally:
        _current_log_file.reset(token)


def validate_stt_eval_only_dataset(dataset_path: str) -> tuple[bool, str, list[dict]]:
    """Validate an eval-only dataset JSON file.

    Expected format: a JSON list of objects with ``id``, ``gt`` and ``pred`` fields.

    Returns:
        tuple[bool, str, list[dict]]: (is_valid, error_message, parsed_rows)
    """
    if not exists(dataset_path):
        return False, f"Dataset file does not exist: {dataset_path}", []

    try:
        with open(dataset_path) as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Failed to parse dataset JSON: {e}", []

    if not isinstance(data, list):
        return False, "Dataset must be a JSON list of objects", []

    required = {"id", "gt", "pred"}
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            return False, f"Row {i} is not an object", []
        missing = required - row.keys()
        if missing:
            return (
                False,
                f"Row {i} missing required fields: {sorted(missing)}. Each row needs 'id', 'gt', 'pred'.",
                [],
            )

    return True, "", data


async def _score_and_write_results(
    ids: list,
    gt_transcripts: list[str],
    pred_transcripts: list[str],
    output_dir: str,
    evaluator_config_dir: str,
    judge_evaluators: list[dict] = None,
    ttfts: list = None,
    statuses: list = None,
    errors: list = None,
) -> dict:
    """Run WER + LLM-judge evaluators over (gt, pred) pairs and write outputs.

    Writes ``results.csv`` and ``metrics.json`` under ``output_dir`` and the
    resolved evaluator config under ``evaluator_config_dir``. Returns the
    metrics_data dict.
    """
    n = len(ids)
    if ttfts is None:
        ttfts = [None] * n
    if statuses is None:
        statuses = ["success"] * n
    if errors is None:
        errors = [None] * n

    wer_results = get_wer_score(gt_transcripts, pred_transcripts)
    # Exclude failed rows from per-row WER and the corpus aggregate so a
    # provider timeout/error isn't conflated with a bad transcription.
    per_row_wer = [
        wer if status == "success" else None
        for wer, status in zip(wer_results["per_row"], statuses)
    ]
    valid_wer = [w for w in per_row_wer if w is not None]
    overall_wer = float(np.mean(valid_wer)) if valid_wer else None
    _log(f"WER: {overall_wer}", to_terminal=False)

    _evaluators = judge_evaluators if judge_evaluators else [DEFAULT_STT_EVALUATOR]
    require_unique_evaluator_names(_evaluators)
    write_evaluator_config(evaluator_config_dir, _evaluators)

    # Only run judges on successful rows — failed rows have empty pred and
    # would otherwise waste API calls and pollute the aggregate scores.
    success_idx = [i for i, s in enumerate(statuses) if s == "success"]
    if success_idx:
        llm_results = await get_llm_judge_score(
            [gt_transcripts[i] for i in success_idx],
            [pred_transcripts[i] for i in success_idx],
            evaluators=_evaluators,
        )
        for name, score_dict in llm_results["scores"].items():
            _log(f"  {name}: {score_dict['mean']:.4f}")
    else:
        llm_results = {"scores": {}, "score": None, "per_row": []}

    # Expand per-row judge results back to full length (None for failed rows).
    per_row_llm = [None] * n
    for offset, i in enumerate(success_idx):
        per_row_llm[i] = llm_results["per_row"][offset]

    _evaluators_by_name = {ev["name"]: ev for ev in _evaluators}

    metrics_data = {"wer": overall_wer}
    for name, score_dict in llm_results["scores"].items():
        metrics_data[name] = score_dict

    valid_ttfts = [t for t in ttfts if t is not None]
    if valid_ttfts:
        metrics_data["ttft"] = {
            "mean": float(np.mean(valid_ttfts)),
            "std": float(np.std(valid_ttfts)),
            "values": [float(t) for t in valid_ttfts],
        }

    data = []
    for _id, gt_text, pred_text, wer, llm_row, ttft, status, error in zip(
        ids,
        gt_transcripts,
        pred_transcripts,
        per_row_wer,
        per_row_llm,
        ttfts,
        statuses,
        errors,
    ):
        row = {
            "id": _id,
            "gt": gt_text,
            "pred": pred_text,
            "ttft": ttft,
            "status": status,
            "error": error,
            "wer": wer,
        }
        for name, ev in _evaluators_by_name.items():
            if llm_row is None:
                row[name] = None
                row[f"{name}_reasoning"] = None
                continue
            ev_result = llm_row[name]
            if is_rating(ev):
                row[name] = ev_result["score"]
            else:
                row[name] = bool(ev_result["match"])
            row[f"{name}_reasoning"] = ev_result["reasoning"]
        data.append(row)

    with open(join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)

    pd.DataFrame(data).to_csv(join(output_dir, "results.csv"), index=False)

    return metrics_data


async def run_eval_only(
    dataset_path: str,
    output_dir: str,
    judge_evaluators: list[dict] = None,
) -> dict:
    """Run evaluators only on a pre-existing dataset of (gt, pred) pairs.

    Skips STT inference. Writes ``metrics.json`` and ``results.csv`` directly
    under ``output_dir``.

    Args:
        dataset_path: Path to a JSON file with a list of {"id", "gt", "pred"} rows.
        output_dir: Directory to write results and metrics.
        judge_evaluators: Optional list of evaluator dicts. Defaults to the
            built-in STT evaluator when omitted.

    Returns:
        dict with status, metrics, and output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    log_save_path = join(output_dir, "logs")
    if exists(log_save_path):
        os.remove(log_save_path)

    token = _current_log_file.set(log_save_path)
    try:
        _log("--------------------------------")
        _log("\033[33mRunning STT eval-only on dataset\033[0m")
        _log(f"Dataset: {dataset_path}")

        is_valid, error_msg, rows = validate_stt_eval_only_dataset(dataset_path)
        if not is_valid:
            _log(f"\033[31mError: {error_msg}\033[0m")
            return {"status": "error", "error": error_msg}

        ids = [r["id"] for r in rows]
        gts = [str(r["gt"]) for r in rows]
        preds = [str(r["pred"]) if r["pred"] is not None else "" for r in rows]

        metrics_data = await _score_and_write_results(
            ids=ids,
            gt_transcripts=gts,
            pred_transcripts=preds,
            output_dir=output_dir,
            evaluator_config_dir=output_dir,
            judge_evaluators=judge_evaluators,
        )

        return {
            "status": "completed",
            "metrics": metrics_data,
            "output_dir": output_dir,
        }
    finally:
        _current_log_file.reset(token)


async def main():
    """CLI entry point for single-provider STT evaluation.

    For multiple providers, use `calibrate stt -p provider1 provider2 ...` which
    routes to benchmark.py, or use the Python SDK's `run()` function.
    """
    parser = argparse.ArgumentParser(
        description="Run STT evaluation for a single provider"
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        required=True,
        help="STT provider to use for evaluation",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="english",
        choices=STT_LANGUAGES,
        help="Language of the audio files",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing the audio files and stt.csv",
    )
    parser.add_argument(
        "-f",
        "--input-file-name",
        type=str,
        default="stt.csv",
        help="Name of the input file containing the dataset to evaluate",
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
        help="Run the evaluation on the first N audio files",
    )
    parser.add_argument(
        "-dc",
        "--debug_count",
        type=int,
        default=5,
        help="Number of audio files to run the evaluation on in debug mode",
    )
    parser.add_argument(
        "--ignore_retry",
        action="store_true",
        help="Ignore retrying if all the audios are not processed and move on to evaluators",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of resuming from last checkpoint",
    )

    args = parser.parse_args()

    provider = args.provider

    # Validate provider
    if provider not in STT_PROVIDERS:
        print(f"\033[31mError: Invalid provider '{provider}'.\033[0m")
        print(f"Available providers: {', '.join(STT_PROVIDERS)}")
        sys.exit(1)

    # Validate input directory structure
    is_valid, error_msg = validate_stt_input_dir(args.input_dir, args.input_file_name)
    if not is_valid:
        print(f"\033[31mInput validation error: {error_msg}\033[0m")
        sys.exit(1)

    # ``exist_ok=True`` makes this safe when several ``calibrate stt``
    # subprocesses race to create the output dir on first use; the previous
    # ``if not exists: makedirs(...)`` pattern was non-atomic and the loser
    # raised ``FileExistsError``.
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n\033[91mSTT Evaluation\033[0m\n")
    print(f"Provider: {provider}")
    print(f"Language: {args.language}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("")

    # Run single provider evaluation
    result = await run_single_provider_eval(
        provider=provider,
        language=args.language,
        input_dir=args.input_dir,
        input_file_name=args.input_file_name,
        output_dir=args.output_dir,
        debug=args.debug,
        debug_count=args.debug_count,
        ignore_retry=args.ignore_retry,
        overwrite=args.overwrite,
    )

    # Print summary
    print(f"\n\033[92m{'='*60}\033[0m")
    print(f"\033[92mSummary\033[0m")
    print(f"\033[92m{'='*60}\033[0m\n")

    if result.get("status") == "error":
        print(f"  {provider}: \033[31mError - {result.get('error')}\033[0m")
        sys.exit(1)
    else:
        metrics = result.get("metrics", {})
        wer = metrics.get("wer")
        # Evaluator entries are dicts carrying a ``type`` field; that's the
        # marker we use to pick them out from other top-level metrics.
        judge_scores = {
            k: v.get("mean")
            for k, v in metrics.items()
            if isinstance(v, dict) and "type" in v
        }

        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"

        judge_str = ", ".join(f"{k}={_fmt(v)}" for k, v in judge_scores.items())
        print(f"  {provider}: WER={_fmt(wer)}, {judge_str}")


if __name__ == "__main__":
    asyncio.run(main())
