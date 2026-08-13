import asyncio
import argparse
import sys
import os
import json
import base64
from os.path import join, exists
from datetime import datetime, timezone
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
from google import genai
from google.genai import types as genai_types

import pandas as pd

from calibrate_agent.utils import (
    get_stt_language_code,
    get_audio_duration_seconds,
    validate_stt_language,
    STT_PROVIDER_MODELS,
    google_stt_model_and_location,
    create_stt_service,
    get_gemini_api_key,
    provider_log as _log,
    provider_log_file as _current_log_file,
)
from calibrate_agent.stt.metrics import (
    get_wer_score,
    get_cer_score,
    get_llm_judge_score,
    get_intent_entity_score,
    get_llm_wer_cer_score,
    get_semantic_wer_score,
    get_ttfs_stats,
)
from calibrate_agent.stt.pipeline_eval import transcribe_via_pipeline
from calibrate_agent.judge_store import JudgeStore
from calibrate_agent._env import resolve_stt_max_concurrency
from calibrate_agent._cli_args import (
    DEFAULT_STT_LLM_JUDGES,
    STT_LLM_JUDGES,
    add_stt_eval_args,
    resolve_stt_llm_judges,
)
from calibrate_agent.judges import (
    is_rating,
    require_unique_evaluator_names,
    write_evaluator_config,
)
from calibrate_agent.langfuse import (
    create_langfuse_audio_media,
    observe,
    langfuse,
    langfuse_enabled,
)
from calibrate_agent.pricing import cost_breakdown, resolve_pricing
from calibrate_agent.rate_limit import (
    STT_PROVIDER_TIMEOUT_SECONDS,
    STT_STREAMING_IDLE_TIMEOUT_SECONDS,
    SARVAM_STT_STREAMING_LIMITER,
)


# =============================================================================
# STT Provider API Methods
# =============================================================================


def _default_stt_model(provider: str, language: str | None = None) -> str | None:
    # Mirrors the model each transcribe_* uses (table, or the Google helper for
    # the Sindhi exception) so cost prices the model actually benchmarked. If a
    # provider gains a language-specific model, update it here too.
    if provider == "google":
        model, _region = google_stt_model_and_location(language)
        return model
    return STT_PROVIDER_MODELS.get(provider)


def _stt_result_row(
    row_id: object,
    gt_text: object,
    pred_text: str,
    audio_dir: Path,
) -> dict:
    return {
        "id": row_id,
        "gt": gt_text,
        "pred": pred_text,
        "audio_duration_seconds": get_audio_duration_seconds(audio_dir / f"{row_id}.wav"),
    }


def _build_stt_cost_metrics(
    provider: str,
    audio_duration_seconds: list[float | None] | None,
    model: str | None = None,
) -> dict | None:
    """Build STT cost metrics from audio duration and provider price config."""
    durations = []
    excluded_row_indices = []
    for index, duration in enumerate(audio_duration_seconds or []):
        if duration is not None and not pd.isna(duration):
            durations.append(float(duration))
        else:
            excluded_row_indices.append(index)
    if not durations:
        return None

    pricing = resolve_pricing("stt", provider, model=model)
    if not pricing:
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
    if excluded_row_indices:
        metrics["excluded_row_indices"] = excluded_row_indices
    return metrics


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
        "model": STT_PROVIDER_MODELS["deepgram"],
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

            # Tells Deepgram we've sent all audio so it flushes and closes.
            await ws.send(json.dumps({"type": "CloseStream"}))

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

    lang_code = get_stt_language_code(language, "openai")

    client = AsyncOpenAI()

    audio_file = load_audio(audio_path, as_file=True)

    # Supplying the input language (ISO-639-1) improves accuracy and latency
    # per the OpenAI transcription docs. For the gpt-4o transcribe models the
    # `language` param is only a soft hint, so we also steer the output language
    # via `prompt` — OpenAI's own recommended workaround for the model emitting
    # the wrong script (e.g. Urdu for English/Hindi audio).
    # https://developers.openai.com/api/docs/guides/speech-to-text
    stream = await client.audio.transcriptions.create(
        model=STT_PROVIDER_MODELS["openai"],
        file=audio_file,
        language=lang_code,
        prompt=f"Transcribe the audio in {language.capitalize()}.",
        response_format="text",
        stream=True,
    )

    transcript = ""

    async for event in _aiter_with_idle_timeout(
        stream, STT_STREAMING_IDLE_TIMEOUT_SECONDS
    ):
        event_type = getattr(event, "type", None)
        if event_type == "transcript.text.done":
            transcript = getattr(event, "text", "") or ""
            break

    return {
        "transcript": transcript,
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
            model=STT_PROVIDER_MODELS["groq"],  # Required model to use for transcription
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
        config=recognition_config,
        streaming_features=cloud_speech_types.StreamingRecognitionFeatures(
            interim_results=True,
        ),
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

    # Transcribes the audio into text
    responses_iterator = client.streaming_recognize(
        requests=requests(config_request, audio_requests)
    )
    final_transcripts = []

    for response in responses_iterator:
        for result in response.results:
            transcript = result.alternatives[0].transcript.strip()
            if not transcript:
                continue

            # Interim results are enabled, so only final results contribute to
            # the transcript to avoid concatenating duplicated, evolving partial
            # hypotheses.
            if result.is_final:
                final_transcripts.append(transcript)

    return {
        "transcript": " ".join(final_transcripts),
    }


async def transcribe_google(audio_path: Path, language: str) -> str:
    """Transcribe audio using Google Cloud Speech-to-Text API."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

    lang_code = get_stt_language_code(language, "google")

    model, region = google_stt_model_and_location(language)

    result = await asyncio.wait_for(
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
        "transcript": result["transcript"].strip(),
    }


def _gemini_client() -> genai.Client:
    """Build a google-genai client (key from GOOGLE_API_KEY)."""
    return genai.Client(api_key=get_gemini_api_key())


async def transcribe_gemini(audio_path: Path, language: str) -> Dict:
    """Transcribe audio with a Gemini multimodal model via the google-genai API.

    Gemini has no dedicated STT endpoint; transcription is a multimodal
    generate_content call over the audio. Benchmark-only — no cascaded pipecat
    Gemini STT service exists, so this is not mirrored in create_stt_service.
    """
    client = _gemini_client()

    lang_code = get_stt_language_code(language, "gemini")
    audio_bytes = load_audio(audio_path)

    prompt = (
        f"Transcribe the following {language} ({lang_code}) audio to text "
        "verbatim. Output only the exact spoken words in the original language, "
        "with no translation, no commentary, and no surrounding quotation marks. "
        "If the audio contains no speech, output an empty string."
    )

    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=STT_PROVIDER_MODELS["gemini"],
            contents=[
                prompt,
                genai_types.Part.from_bytes(
                    data=audio_bytes, mime_type="audio/wav"
                ),
            ],
        ),
        timeout=STT_PROVIDER_TIMEOUT_SECONDS,
    )

    return {
        "transcript": (response.text or "").strip(),
    }


# Max seconds to wait for a transcript frame after flushing the Sarvam
# websocket. The SDK does not forward its `timeout` to the websocket, so this
# guards against clips that never produce a `data`/`error` frame.
SARVAM_STT_RECV_TIMEOUT = 60.0


async def transcribe_sarvam(audio_path: Path, language: str) -> str:
    """Transcribe audio using Sarvam's STT API."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "sarvam")

    audio_data = base64.b64encode(load_audio(audio_path)).decode("utf-8")

    await SARVAM_STT_STREAMING_LIMITER.acquire()

    client = AsyncSarvamAI(api_subscription_key=api_key, timeout=120.0)

    transcript = ""
    ttft = None

    async with client.speech_to_text_streaming.connect(
        language_code=lang_code,
        model=STT_PROVIDER_MODELS["sarvam"],
        mode="transcribe",
        flush_signal=True,
    ) as ws:
        # Send audio
        await ws.transcribe(audio=audio_data, encoding="audio/wav", sample_rate=16000)

        # Force immediate processing
        await ws.flush()
        _log("⚡ Processing forced - getting immediate results")

        # Get results. The Sarvam SDK forwards no timeout to the underlying
        # websocket, so a clip that yields no transcript (e.g. silence) leaves
        # `recv()` blocked forever. Bound the wait explicitly.
        try:
            async with asyncio.timeout(SARVAM_STT_RECV_TIMEOUT):
                async for message in ws:
                    if getattr(message, "type", None) == "error":
                        error = getattr(
                            message.data, "error", "Unknown Sarvam STT error"
                        )
                        raise RuntimeError(error)
                    if getattr(message, "type", None) != "data":
                        continue

                    transcript = getattr(message.data, "transcript", "")
                    metrics = getattr(message.data, "metrics", None)
                    ttft = getattr(metrics, "processing_latency", None)
                    break
        except asyncio.TimeoutError:
            _log(
                f"[WARN] Sarvam returned no result for {audio_path.name} within "
                f"{SARVAM_STT_RECV_TIMEOUT}s; treating as empty transcript"
            )

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
            model=STT_PROVIDER_MODELS["cartesia"],  # Model (required)
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

        # Send audio and receive results concurrently
        async def send_audio():
            """Send audio chunks to the STT websocket"""
            async for chunk in audio_stream():
                await ws.send(chunk)
                # print(f"Sent audio chunk of {len(chunk)} bytes")

            # Signal end of audio stream
            await ws.send("finalize")
            await ws.send("close")
            # print("Audio streaming completed")

        async def receive_transcripts():
            """Receive and process transcription results with word timestamps"""
            full_transcript = ""

            async for result in _aiter_with_idle_timeout(
                ws.receive(), STT_STREAMING_IDLE_TIMEOUT_SECONDS
            ):
                if result["type"] == "transcript":
                    text = result["text"]
                    is_final = result["is_final"]
                    if is_final:
                        full_transcript += text + " "

                elif result["type"] == "done":
                    break

            return full_transcript.strip()

        _, final_transcript = await asyncio.gather(send_audio(), receive_transcripts())

        await ws.close()

        return {"transcript": final_transcript}

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
    endpoint = "wss://api.smallest.ai/waves/v1/stt/live"
    params = {
        "model": STT_PROVIDER_MODELS["smallest"],
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

            await ws.send(json.dumps({"type": "close_stream"}))

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
    }


# How long to wait for another ElevenLabs ``committed_transcript`` segment
# before treating the stream as finished. The server auto-commits long audio
# into multiple segments (~every 36s) and sends no end-of-transcript marker,
# so completion is detected by this inter-segment idle gap.
ELEVENLABS_SEGMENT_IDLE_SECONDS = 3.0


async def transcribe_elevenlabs_streaming(audio_path: Path, language: str) -> str:
    """Transcribe audio using ElevenLabs' Scribe v2 Realtime via the official SDK.

    Uses the SDK's ``speech_to_text.realtime.connect`` (manual commit strategy).
    See https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/server-side-streaming.
    """
    from elevenlabs import (
        AudioFormat,
        CommitStrategy,
        ElevenLabs,
        RealtimeAudioOptions,
        RealtimeEvents,
    )

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "elevenlabs")

    audio = load_audio(audio_path, raw_pcm=True)
    chunk_size = 32000  # 1s of 16 kHz, 16-bit mono PCM

    transcript_parts: List[str] = []
    fatal_error: Dict[str, object] = {"data": None}
    session_started = asyncio.Event()
    # Set whenever a committed_transcript arrives (to detect the stream going
    # idle) and whenever the stream closes or errors.
    segment_event = asyncio.Event()
    closed = asyncio.Event()

    def _on_session_started(_data):
        session_started.set()

    def _on_committed_transcript(data):
        # The server segments long audio and auto-commits roughly every 36s, so
        # a single manual commit can still yield MULTIPLE committed_transcript
        # messages (one per segment) with no terminal marker. Accumulate every
        # segment instead of stopping at the first, or long audio gets truncated.
        if isinstance(data, dict):
            text = data.get("text", "")
            if text:
                transcript_parts.append(text)
        segment_event.set()

    def _on_error(data):
        # ``insufficient_audio_activity`` fires for clips with no committable
        # speech — treat it as a graceful end-of-stream (empty transcript)
        # rather than a hard error that would trigger the @backoff retries.
        if (
            isinstance(data, dict)
            and data.get("message_type") == "insufficient_audio_activity"
        ):
            closed.set()
            segment_event.set()
            return
        fatal_error["data"] = data
        closed.set()
        segment_event.set()

    def _on_close(*_args):
        closed.set()
        segment_event.set()

    client = ElevenLabs(api_key=api_key)
    connection = await client.speech_to_text.realtime.connect(
        RealtimeAudioOptions(
            model_id=STT_PROVIDER_MODELS["elevenlabs"],
            audio_format=AudioFormat.PCM_16000,
            sample_rate=16000,
            commit_strategy=CommitStrategy.MANUAL,
            language_code=lang_code,
        )
    )

    connection.on(RealtimeEvents.SESSION_STARTED, _on_session_started)
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, _on_committed_transcript)
    connection.on(RealtimeEvents.ERROR, _on_error)
    connection.on(RealtimeEvents.CLOSE, _on_close)

    try:
        # The server emits ``session_started`` before accepting audio.
        try:
            await asyncio.wait_for(
                session_started.wait(), timeout=STT_STREAMING_IDLE_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            pass

        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            if chunk:
                await connection.send(
                    {"audio_base_64": base64.b64encode(chunk).decode("utf-8")}
                )

        # Let the trailing chunk land before finalising (per ElevenLabs docs);
        # without this, sub-second clips close with insufficient_audio_activity
        # and never emit a committed_transcript.
        await asyncio.sleep(0.5)
        await connection.commit()

        # Collect ALL committed segments. There is no end-of-transcript marker,
        # so we finish once the stream goes idle (no new segment within the idle
        # window), closes, or errors. The first segment is bounded by the longer
        # streaming idle timeout; once segments start arriving they come
        # back-to-back, so a short inter-segment gap signals completion.
        received_any = bool(transcript_parts)
        while not closed.is_set():
            segment_event.clear()
            timeout = (
                ELEVENLABS_SEGMENT_IDLE_SECONDS
                if received_any
                else STT_STREAMING_IDLE_TIMEOUT_SECONDS
            )
            try:
                await asyncio.wait_for(segment_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            received_any = True
    finally:
        await connection.close()

    if fatal_error["data"] is not None and not transcript_parts:
        raise RuntimeError(f"ElevenLabs streaming STT error: {fatal_error['data']}")

    return {
        "transcript": " ".join(
            part.strip() for part in transcript_parts if part.strip()
        ),
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
        "gemini": transcribe_gemini,
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
            },
            session_id=unique_id,
        )

    return {
        "transcript": transcript,
    }


# =============================================================================
# STT Evaluation Main
# =============================================================================


@backoff.on_exception(backoff.expo, Exception, max_tries=3, factor=2)
@observe(name="stt", capture_input=False, capture_output=False)
async def transcribe_audio_pipeline(
    audio_path: Path,
    reference: str,
    provider: str,
    language: str,
    unique_id: str,
) -> dict:
    """Transcribe one clip by streaming it through a real pipecat pipeline.

    The "pipeline" engine: feeds the WAV at real-time pace through the same
    ``create_stt_service`` the live agent deploys, so the benchmark reflects the
    shipped STT config and also captures TTFS latency (speech-stop -> final
    transcript). Contrast with ``transcribe_audio`` (the "direct" engine), which
    calls each provider SDK directly.

    Returns ``{"transcript": str, "ttfs": float | None}``.
    """
    # Sarvam's streaming endpoint is rate-limited per account; honour the same
    # limiter the direct engine uses so concurrent clips don't blow the cap.
    if provider == "sarvam":
        await SARVAM_STT_STREAMING_LIMITER.acquire()

    pcm = load_audio(audio_path, raw_pcm=True)
    stt_service = create_stt_service(provider, language)
    output = await transcribe_via_pipeline(pcm, stt_service)

    transcript = output["transcript"].strip()
    ttfs = output["ttfs"]

    if langfuse_enabled and langfuse:
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
                "ttfs": ttfs,
                "engine": "pipeline",
            },
            session_id=unique_id,
        )

    return {"transcript": transcript, "ttfs": ttfs}


async def _run_stt_eval_concurrent(
    gt_data: List[Dict],
    audio_dir: Path,
    provider: str,
    language: str,
    results_csv_path: Path,
    max_concurrency: int,
    transcribe_fn,
    with_ttfs: bool,
) -> int:
    """Transcribe clips with bounded concurrency, writing rows as they complete.

    Shared by both engines: ``transcribe_fn`` is the per-clip coroutine
    (``transcribe_audio`` for direct, ``transcribe_audio_pipeline`` for pipeline)
    and ``with_ttfs`` adds the ``ttfs`` column (pipeline only). Up to
    ``max_concurrency`` clips run at once (for pipeline each is a 1x real-time
    pipeline, so concurrency keeps a large run tractable); each result is written
    to ``results.csv`` under a lock (resumable, skip-processed). A failed clip is
    logged and left unwritten for the outer retry loop / no-progress fill,
    without a gather-wide cancel.
    """
    if exists(results_csv_path):
        existing_df = pd.read_csv(results_csv_path)
        results = existing_df.to_dict("records")
        processed_ids = set(existing_df["id"].tolist())
    else:
        results = []
        processed_ids = set()

    pending = [g for g in gt_data if g["id"] not in processed_ids]
    if not pending:
        return 0

    unique_id = str(uuid.uuid4())
    semaphore = asyncio.Semaphore(max_concurrency)
    write_lock = asyncio.Lock()
    success_count = 0

    async def worker(index: int, gt_info: Dict) -> None:
        nonlocal success_count
        async with semaphore:
            audio_path = audio_dir / f"{gt_info['id']}.wav"
            _log("--------------------------------")
            _log(f"Processing audio [{index}/{len(pending)}]: {audio_path.name}")
            try:
                output = await transcribe_fn(
                    audio_path, gt_info["gt"], provider, language, unique_id
                )
            except Exception as e:
                # Don't raise: leave this id unprocessed so the retry loop can
                # re-attempt it (and eventually fill an empty transcript) without
                # cancelling the other in-flight clips.
                _log(f"\033[91mFailed to transcribe {audio_path}: {e}\033[0m")
                return

            transcript = output["transcript"]
            _log(f"\033[33mTranscript: {transcript}\033[0m")
            async with write_lock:
                if transcript:
                    success_count += 1
                row = _stt_result_row(
                    gt_info["id"], gt_info["gt"], transcript, audio_dir
                )
                if with_ttfs:
                    row["ttfs"] = output.get("ttfs")
                results.append(row)
                pd.DataFrame(results).to_csv(results_csv_path, index=False)

    await asyncio.gather(
        *(worker(i + 1, gt_info) for i, gt_info in enumerate(pending))
    )
    return success_count


async def run_stt_eval(
    gt_data: List[Dict],
    audio_dir: Path,
    provider: str,
    language: str,
    results_csv_path: Path,
    engine: str = "pipeline",
    max_concurrency: int = None,
) -> int:
    """Process audio files and save results immediately to CSV.

    Args:
        gt_data: List of {"id": ..., "gt": ...} for each file to process
        audio_dir: Directory containing audio files
        provider: STT provider name
        language: Language code
        results_csv_path: Path to save results CSV
        engine: ``"direct"`` (per-provider SDK calls) or ``"pipeline"`` (stream
            through a real pipecat agent pipeline, capturing TTFS latency).
        max_concurrency: Concurrent clips per provider (both engines). ``None``
            resolves per engine (pipeline 1, direct 4) via
            ``resolve_stt_parallelism``.

    Returns:
        Number of files successfully transcribed (non-empty) in this run.
    """
    max_concurrency = resolve_stt_max_concurrency(engine, max_concurrency)
    transcribe_fn = (
        transcribe_audio_pipeline if engine == "pipeline" else transcribe_audio
    )
    return await _run_stt_eval_concurrent(
        gt_data,
        audio_dir,
        provider,
        language,
        results_csv_path,
        max_concurrency,
        transcribe_fn=transcribe_fn,
        with_ttfs=(engine == "pipeline"),
    )

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
    "gemini",
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
    "maithili",
]


# metrics.json keys written by each built-in LLM judge.
_BUILT_IN_JUDGE_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "intent": ("sarvam_intent_score", "sarvam_entity_score"),
    "llm_wer": ("sarvam_llm_wer", "sarvam_llm_cer"),
    "semantic_wer": ("semantic_wer",),
}

# results.csv columns written by each built-in LLM judge.
_BUILT_IN_JUDGE_RESULT_COLUMNS: dict[str, tuple[str, ...]] = {
    "intent": (
        "sarvam_intent_score",
        "sarvam_intent_reasoning",
        "sarvam_entity_score",
        "sarvam_entity_reasoning",
    ),
    "llm_wer": (
        "sarvam_llm_wer",
        "sarvam_llm_cer",
        "sarvam_llm_wer_reasoning",
    ),
    "semantic_wer": (
        "semantic_wer",
        "semantic_wer_metadata",
        "semantic_wer_reasoning",
    ),
}

# Always refreshed on every scoring pass; never restored from a prior file.
_ALWAYS_FRESH_METRIC_KEYS = frozenset({"wer", "cer", "cost", "ttfs"})
_ALWAYS_FRESH_RESULT_COLUMNS = frozenset(
    {"id", "gt", "pred", "wer", "cer", "audio_duration_seconds", "ttfs"}
)


def _load_prior_metrics(output_dir: str) -> dict | None:
    path = join(output_dir, "metrics.json")
    if not exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_prior_results(output_dir: str) -> pd.DataFrame | None:
    path = join(output_dir, "results.csv")
    if not exists(path):
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def _archive_prior_metrics(output_dir: str, prior: dict) -> None:
    """Write a timestamped copy of ``prior`` under ``judge_history/``."""
    history_dir = Path(output_dir) / "judge_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = history_dir / f"metrics_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prior, f, indent=4)


def _prior_evaluator_names(prior_metrics: dict) -> set[str]:
    """Names of config-evaluator entries in a prior ``metrics.json``."""
    names: set[str] = set()
    for key, value in prior_metrics.items():
        if key in _ALWAYS_FRESH_METRIC_KEYS:
            continue
        if any(key in keys for keys in _BUILT_IN_JUDGE_METRIC_KEYS.values()):
            continue
        if isinstance(value, dict) and "type" in value and "mean" in value:
            names.add(key)
    return names


def _merge_prior_judge_outputs(
    metrics_data: dict,
    rows: list[dict],
    output_dir: str,
    *,
    overwrite: bool,
    enabled_judges: frozenset[str],
    this_run_evaluator_names: set[str],
) -> tuple[dict, list[dict]]:
    """Keep prior built-in / config-evaluator outputs that this run did not refresh.

    Always archives an existing ``metrics.json`` under ``judge_history/`` before
    replacing it. When ``overwrite`` is True, returns the fresh artifacts
    unchanged after that archive. Otherwise copies metrics keys and per-row
    columns for judges (and config evaluators) that were not part of this
    scoring pass, matched by ``id``.
    """
    prior_metrics = _load_prior_metrics(output_dir)
    prior_df = _load_prior_results(output_dir)
    if prior_metrics is None and prior_df is None:
        return metrics_data, rows

    if prior_metrics is not None:
        _archive_prior_metrics(output_dir, prior_metrics)

    if overwrite:
        return metrics_data, rows

    if prior_metrics is not None:
        fresh_metric_keys = set(_ALWAYS_FRESH_METRIC_KEYS)
        for name in enabled_judges:
            fresh_metric_keys.update(_BUILT_IN_JUDGE_METRIC_KEYS.get(name, ()))
        fresh_metric_keys.update(this_run_evaluator_names)

        for key, value in prior_metrics.items():
            if key in fresh_metric_keys:
                continue
            if key not in metrics_data:
                metrics_data[key] = value

    if prior_df is not None and not prior_df.empty and "id" in prior_df.columns:
        prior_by_id = {
            str(r["id"]): r for r in prior_df.to_dict("records")
        }
        preserve_columns: set[str] = set()
        for name in STT_LLM_JUDGES:
            if name in enabled_judges:
                continue
            preserve_columns.update(_BUILT_IN_JUDGE_RESULT_COLUMNS.get(name, ()))
        if prior_metrics is not None:
            for ev_name in _prior_evaluator_names(prior_metrics):
                if ev_name in this_run_evaluator_names:
                    continue
                preserve_columns.add(ev_name)
                preserve_columns.add(f"{ev_name}_reasoning")

        preserve_columns = {
            c
            for c in preserve_columns
            if c in prior_df.columns and c not in _ALWAYS_FRESH_RESULT_COLUMNS
        }
        if preserve_columns:
            for row in rows:
                prior_row = prior_by_id.get(str(row["id"]))
                if prior_row is None:
                    continue
                for col in preserve_columns:
                    if col in row:
                        continue
                    value = prior_row.get(col)
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        continue
                    row[col] = value

    return metrics_data, rows


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
    llm_judges: frozenset[str] | None = None,
    engine: str = "pipeline",
    max_concurrency: int = None,
) -> dict:
    """Run STT evaluation for a single provider.

    ``max_concurrency=None`` is forwarded to ``run_stt_eval``, which resolves the
    per-engine default (pipeline 1, direct 4).
    """
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
                            _stt_result_row(
                                gt_info["id"],
                                gt_info["gt"],
                                "",
                                audio_dir,
                            )
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
                engine=engine,
                max_concurrency=max_concurrency,
            )

            if ignore_retry:
                break

        # Load final results for metrics
        results_df = pd.read_csv(results_csv_path)
        all_ids = results_df["id"].tolist()
        all_gt_transcripts = results_df["gt"].astype(str).tolist()
        all_pred_transcripts = results_df["pred"].fillna("").astype(str).tolist()
        audio_durations = []
        for row_index, row_id in enumerate(all_ids):
            duration = None
            if "audio_duration_seconds" in results_df.columns:
                duration = results_df.iloc[row_index]["audio_duration_seconds"]
            if duration is None or pd.isna(duration):
                duration = get_audio_duration_seconds(audio_dir / f"{row_id}.wav")
            audio_durations.append(duration)
        cost_metrics = _build_stt_cost_metrics(
            provider=provider,
            audio_duration_seconds=audio_durations,
            model=_default_stt_model(provider, language),
        )

        # TTFS latency is only produced by the pipeline engine; NaN/absent -> None.
        if "ttfs" in results_df.columns:
            all_ttfs = [
                None if pd.isna(v) else float(v) for v in results_df["ttfs"].tolist()
            ]
        else:
            all_ttfs = [None] * len(all_ids)

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
            language=language,
            llm_judges=llm_judges,
            audio_durations=audio_durations,
            cost_metrics=cost_metrics,
            ttfs_values=all_ttfs,
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
    language: str = "english",
    llm_judges: frozenset[str] | None = None,
    cost_metrics: dict | None = None,
    audio_durations: list[float | None] | None = None,
    ttfs_values: list = None,
    overwrite: bool = False,
) -> dict:
    """Run WER/CER (and optional LLM-judge evaluators) over (gt, pred) pairs.

    Writes ``results.csv`` and ``metrics.json`` under ``output_dir``. WER and
    CER are always computed. The LLM judge is opt-in: when ``judge_evaluators``
    is empty/omitted, no judge runs, no evaluator config is written, and the
    evaluator columns/metrics are omitted. Returns the metrics_data dict.

    Built-in LLM judges are selected via ``llm_judges`` (``intent``,
    ``llm_wer``, ``semantic_wer``). ``None`` runs all three; an empty
    frozenset skips them entirely — no normalizer model is loaded, no judge
    calls are made, and the corresponding columns/metrics are omitted.

    Each judge is isolated: if any judge raises, the failure is logged and that
    judge's columns/metrics are dropped, but WER/CER and any judges that
    succeeded are still written.

    Every judge call is checkpointed through a :class:`JudgeStore` loaded
    from ``output_dir``, keyed by ``ids``: a row already graded in a prior
    (interrupted or completed) run for the same input, evaluator prompt, and
    model is reused instead of re-judged. ``overwrite=True`` discards that
    checkpoint (along with ``results.csv``) before scoring so every row is
    graded fresh.
    """
    wer_results = get_wer_score(gt_transcripts, pred_transcripts, language=language)
    _log(f"WER: {wer_results['score']}", to_terminal=False)

    cer_results = get_cer_score(gt_transcripts, pred_transcripts, language=language)
    _log(f"CER: {cer_results['score']}", to_terminal=False)

    store = JudgeStore.load(output_dir)
    if overwrite:
        store.clear()
        _log("Overwrite enabled - cleared cached judge grades", to_terminal=False)
    elif len(store) > 0:
        _log(
            f"Resuming judge grading: {len(store)} cached result(s) found in "
            f"{store.path} and will be reused",
            to_terminal=False,
        )

    # Each judge below runs independently and is isolated: a failure in one
    # is logged and that judge's columns/metrics are omitted, but WER/CER and
    # any judges that succeeded are still written.
    enabled = DEFAULT_STT_LLM_JUDGES if llm_judges is None else llm_judges

    intent_entity_results = None
    llm_wer_results = None
    if "intent" in enabled:
        try:
            intent_entity_results = await get_intent_entity_score(
                gt_transcripts,
                pred_transcripts,
                language=language,
                store=store,
                row_ids=ids,
            )
            _log(
                f"Sarvam Intent Score: {intent_entity_results['intent']:.4f}  Sarvam Entity Score: {intent_entity_results['entity']:.4f}",
                to_terminal=False,
            )
        except Exception as e:
            intent_entity_results = None
            _log(f"Sarvam intent/entity judge failed, skipping: {e}")
    if "llm_wer" in enabled:
        try:
            llm_wer_results = await get_llm_wer_cer_score(
                gt_transcripts,
                pred_transcripts,
                language=language,
                store=store,
                row_ids=ids,
            )
            _log(
                f"Sarvam LLM WER: {llm_wer_results['llm_wer']:.4f}  Sarvam LLM CER: {llm_wer_results['llm_cer']:.4f}",
                to_terminal=False,
            )
        except Exception as e:
            llm_wer_results = None
            _log(f"Sarvam LLM-WER/CER judge failed, skipping: {e}")

    semantic_wer_results = None
    if "semantic_wer" in enabled:
        try:
            semantic_wer_results = await get_semantic_wer_score(
                gt_transcripts, pred_transcripts, store=store, row_ids=ids
            )
            _log(
                f"Semantic WER: {semantic_wer_results['semantic_wer']:.4f}",
                to_terminal=False,
            )
        except Exception as e:
            semantic_wer_results = None
            _log(f"Semantic WER judge failed, skipping: {e}")

    # The LLM judge is opt-in for STT: when no evaluators are passed we report
    # WER/CER only and skip the judge entirely (no evaluator config, no judge
    # calls, no evaluator columns/metrics).
    _evaluators = judge_evaluators or []
    llm_results = None
    if _evaluators:
        require_unique_evaluator_names(_evaluators)
        write_evaluator_config(evaluator_config_dir, _evaluators)
        try:
            llm_results = await get_llm_judge_score(
                gt_transcripts,
                pred_transcripts,
                evaluators=_evaluators,
                store=store,
                row_ids=ids,
            )
            for name, score_dict in llm_results["scores"].items():
                _log(f"  {name}: {score_dict['mean']:.4f}")
        except Exception as e:
            llm_results = None
            _log(f"LLM judge failed, skipping evaluator columns: {e}")

    # Only surface evaluator columns for judges that actually produced results.
    _evaluators_by_name = (
        {ev["name"]: ev for ev in _evaluators} if llm_results is not None else {}
    )

    metrics_data = {
        "wer": wer_results["score"],
        "cer": cer_results["score"],
    }
    if intent_entity_results is not None:
        metrics_data["sarvam_intent_score"] = intent_entity_results["intent"]
        metrics_data["sarvam_entity_score"] = intent_entity_results["entity"]
    if llm_wer_results is not None:
        metrics_data["sarvam_llm_wer"] = llm_wer_results["llm_wer"]
        metrics_data["sarvam_llm_cer"] = llm_wer_results["llm_cer"]
    if semantic_wer_results is not None:
        metrics_data["semantic_wer"] = semantic_wer_results["semantic_wer"]
    if llm_results is not None:
        for name, score_dict in llm_results["scores"].items():
            metrics_data[name] = score_dict
    if cost_metrics:
        metrics_data["cost"] = cost_metrics

    # TTFS latency (pipeline engine only). Emitted as a {p50,p95,p99,mean} dict
    # so read_leaderboard_metrics fans it into ttfs_p50/ttfs_p95/ttfs_p99 columns.
    ttfs_per_row = list(ttfs_values) if ttfs_values is not None else [None] * len(ids)
    if len(ttfs_per_row) < len(ids):
        ttfs_per_row.extend([None] * (len(ids) - len(ttfs_per_row)))
    elif len(ttfs_per_row) > len(ids):
        ttfs_per_row = ttfs_per_row[: len(ids)]
    has_ttfs = any(v is not None for v in ttfs_per_row)
    if has_ttfs:
        ttfs_stats = get_ttfs_stats(ttfs_per_row)
        if ttfs_stats is not None:
            metrics_data["ttfs"] = ttfs_stats

    ie_per_row = (
        intent_entity_results["per_row"]
        if intent_entity_results is not None
        else [None] * len(ids)
    )
    llm_wer_per_row = (
        llm_wer_results["per_row"]
        if llm_wer_results is not None
        else [None] * len(ids)
    )
    llm_per_row = (
        llm_results["per_row"] if llm_results is not None else [None] * len(ids)
    )
    semantic_wer_per_row = (
        semantic_wer_results["per_row"]
        if semantic_wer_results is not None
        else [None] * len(ids)
    )

    audio_durations = list(audio_durations or [])
    if len(audio_durations) < len(ids):
        audio_durations.extend([None] * (len(ids) - len(audio_durations)))
    elif len(audio_durations) > len(ids):
        audio_durations = audio_durations[: len(ids)]

    data = []
    for (
        _id,
        gt_text,
        pred_text,
        wer,
        cer,
        ttfs_row,
        ie_row,
        llm_wer_row,
        semantic_wer_row,
        llm_row,
        duration,
    ) in zip(
        ids,
        gt_transcripts,
        pred_transcripts,
        wer_results["per_row"],
        cer_results["per_row"],
        ttfs_per_row,
        ie_per_row,
        llm_wer_per_row,
        semantic_wer_per_row,
        llm_per_row,
        audio_durations,
    ):
        row = {
            "id": _id,
            "gt": gt_text,
            "pred": pred_text,
            "wer": wer,
            "cer": cer,
        }
        if duration is not None and not pd.isna(duration):
            row["audio_duration_seconds"] = duration
        if has_ttfs:
            row["ttfs"] = ttfs_row
        if ie_row is not None:
            row["sarvam_intent_score"] = int(ie_row["intent_score"])
            row["sarvam_intent_reasoning"] = ie_row["intent_explanation"]
            row["sarvam_entity_score"] = float(ie_row["entity_score"])
            row["sarvam_entity_reasoning"] = ie_row["entity_explanation"]
        if llm_wer_row is not None:
            row["sarvam_llm_wer"] = float(llm_wer_row["llm_wer"])
            row["sarvam_llm_cer"] = float(llm_wer_row["llm_cer"])
            row["sarvam_llm_wer_reasoning"] = json.dumps(
                llm_wer_row["segments"], ensure_ascii=False
            )
        if semantic_wer_row is not None:
            row["semantic_wer"] = float(semantic_wer_row["semantic_wer"])
            row["semantic_wer_metadata"] = json.dumps(
                {
                    "substitutions": semantic_wer_row["substitutions"],
                    "deletions": semantic_wer_row["deletions"],
                    "insertions": semantic_wer_row["insertions"],
                    "reference_words": semantic_wer_row["reference_words"],
                    "normalized_reference": semantic_wer_row["normalized_reference"],
                    "normalized_hypothesis": semantic_wer_row["normalized_hypothesis"],
                },
                ensure_ascii=False,
            )
            row["semantic_wer_reasoning"] = semantic_wer_row["reasoning"]
        for name, ev in _evaluators_by_name.items():
            ev_result = llm_row[name]
            if is_rating(ev):
                row[name] = ev_result["score"]
            else:
                row[name] = bool(ev_result["match"])
            row[f"{name}_reasoning"] = ev_result["reasoning"]
        data.append(row)

    metrics_data, data = _merge_prior_judge_outputs(
        metrics_data,
        data,
        output_dir,
        overwrite=overwrite,
        enabled_judges=enabled,
        this_run_evaluator_names=set(_evaluators_by_name),
    )

    with open(join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)

    pd.DataFrame(data).to_csv(join(output_dir, "results.csv"), index=False)

    return metrics_data


async def run_eval_only(
    dataset_path: str,
    output_dir: str,
    judge_evaluators: list[dict] = None,
    language: str = "english",
    llm_judges: frozenset[str] | None = None,
    overwrite: bool = False,
) -> dict:
    """Run evaluators only on a pre-existing dataset of (gt, pred) pairs.

    Skips STT inference. Writes ``metrics.json`` and ``results.csv`` directly
    under ``output_dir``.

    Args:
        dataset_path: Path to a JSON file with a list of {"id", "gt", "pred"} rows.
        output_dir: Directory to write results and metrics.
        judge_evaluators: Optional list of evaluator dicts. When omitted, no
            LLM judge runs and only WER/CER are reported.
        language: Language of the dataset, used to normalize text before the
            intent/entity judge. Defaults to ``english``.
        llm_judges: Built-in LLM judges to run (``intent``, ``llm_wer``,
            ``semantic_wer``). ``None`` runs all three; an empty frozenset
            skips them.
        overwrite: Discard any judge results checkpointed under ``output_dir``
            from a prior run before scoring, so every row is graded fresh.

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
            language=language,
            llm_judges=llm_judges,
            overwrite=overwrite,
        )

        return {
            "status": "completed",
            "metrics": metrics_data,
            "output_dir": output_dir,
        }
    finally:
        _current_log_file.reset(token)


def format_metrics_summary(metrics: dict, prefix: str = "") -> str:
    """Build the one-line ``WER / CER / Sarvam judges / evaluator`` summary
    shared by the single-provider, multi-provider, and eval-only paths.

    The Sarvam judge fields are only included when present in ``metrics``
    (i.e. when those built-in judges ran for the run).
    """
    parts = [
        f"WER={metrics.get('wer', 0):.4f}",
        f"CER={metrics.get('cer', 0):.4f}",
    ]
    for key, label in (
        ("semantic_wer", "Semantic WER"),
        ("sarvam_intent_score", "Sarvam Intent Score"),
        ("sarvam_entity_score", "Sarvam Entity Score"),
        ("sarvam_llm_wer", "Sarvam LLM WER"),
        ("sarvam_llm_cer", "Sarvam LLM CER"),
    ):
        if key in metrics:
            parts.append(f"{label}={metrics[key]:.4f}")
    # TTFS latency (pipeline engine only) — a {p50,p95,p99,mean} dict.
    ttfs = metrics.get("ttfs")
    if isinstance(ttfs, dict) and "p50" in ttfs:
        parts.append(f"TTFS p50={ttfs['p50']:.2f}s p95={ttfs['p95']:.2f}s")
    # Evaluator entries are dicts carrying a ``type`` field; that's the marker
    # we use to pick them out from other top-level metrics.
    parts.extend(
        f"{k}={v['mean']:.4f}"
        for k, v in metrics.items()
        if isinstance(v, dict) and "type" in v
    )
    cost = metrics.get("cost")
    if isinstance(cost, dict) and cost.get("cost_usd") is not None:
        parts.append(f"cost=${cost['cost_usd']:.6f}")
    return f"  {prefix}" + ", ".join(parts)


async def main():
    """CLI entry point for single-provider STT evaluation.

    For multiple providers, use `calibrate-agent stt -p provider1 provider2 ...` which
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
    add_stt_eval_args(parser, include_max_parallel=False)

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

    # ``exist_ok=True`` makes this safe when several ``calibrate-agent stt``
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
        llm_judges=resolve_stt_llm_judges(
            skip_llm_judges=args.skip_llm_judges, judges=args.judges
        ),
        engine=args.engine,
        max_concurrency=args.max_concurrency,
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
        print(format_metrics_summary(metrics, prefix=f"{provider}: "))


if __name__ == "__main__":
    asyncio.run(main())
