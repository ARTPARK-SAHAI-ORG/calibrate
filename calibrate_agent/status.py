"""
Status check for all supported providers.

Verifies each provider actually works by making a real API call:
- LLM providers: chat completion with "hi"
- TTS providers: synthesize "hi" to audio
- STT-only providers: transcribe silence
"""

import os
import asyncio
import json
import time
import struct
from urllib.parse import urlencode

import httpx

from calibrate_agent.utils import (
    TTS_PROVIDER_MODELS,
    get_tts_voice,
    get_gemini_api_key,
)


# ─────────────────────────────────────────────────────────────────────
# Provider definitions
# ─────────────────────────────────────────────────────────────────────

PROVIDERS = [
    {
        "name": "deepgram",
        "types": ["stt"],
        "env_vars": ["DEEPGRAM_API_KEY"],
    },
    {
        "name": "openai",
        "types": ["stt", "tts", "llm"],
        "env_vars": ["OPENAI_API_KEY"],
    },
    {
        "name": "groq",
        "types": ["stt", "tts"],
        "env_vars": ["GROQ_API_KEY"],
    },
    {
        "name": "google",
        "types": ["stt", "tts"],
        "env_vars": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT_ID"],
    },
    {
        "name": "gemini",
        "types": ["stt", "tts"],
        "env_vars": ["GOOGLE_API_KEY"],
    },
    {
        "name": "sarvam",
        "types": ["stt", "tts"],
        "env_vars": ["SARVAM_API_KEY"],
    },
    {
        "name": "elevenlabs",
        "types": ["stt", "tts"],
        "env_vars": ["ELEVENLABS_API_KEY"],
    },
    {
        "name": "cartesia",
        "types": ["stt", "tts"],
        "env_vars": ["CARTESIA_API_KEY"],
    },
    {
        "name": "smallest",
        "types": ["stt", "tts"],
        "env_vars": ["SMALLEST_API_KEY"],
    },
    {
        "name": "openrouter",
        "types": ["llm"],
        "env_vars": ["OPENROUTER_API_KEY"],
    },
]


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _generate_silence_wav(duration_s=0.5, sample_rate=16000):
    """Generate a minimal WAV file containing silence."""
    num_samples = int(sample_rate * duration_s)
    data = b"\x00\x00" * num_samples
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(data),
    )
    return header + data


_SILENCE_WAV = _generate_silence_wav()


# ─────────────────────────────────────────────────────────────────────
# Per-provider checks — real functional API calls
# ─────────────────────────────────────────────────────────────────────


async def _check_openai(client: httpx.AsyncClient) -> str:
    """LLM: send 'hi' via chat completion (gpt-4o-mini)."""
    api_key = os.getenv("OPENAI_API_KEY")
    resp = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("choices"):
        raise ValueError("No response from model")
    return "llm"


async def _check_openrouter(client: httpx.AsyncClient) -> str:
    """LLM: send 'hi' via chat completion (OpenRouter)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    resp = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("choices"):
        raise ValueError("No response from model")
    return "llm"


async def _check_groq(client: httpx.AsyncClient) -> str:
    """TTS: synthesize 'hi' via Groq audio/speech."""
    api_key = os.getenv("GROQ_API_KEY")
    resp = await client.post(
        "https://api.groq.com/openai/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "canopylabs/orpheus-v1-english",
            "voice": "troy",
            "input": "hi",
            "response_format": "wav",
        },
    )
    resp.raise_for_status()
    if len(resp.content) < 100:
        raise ValueError("Empty audio response")
    return "tts"


async def _check_google(client: httpx.AsyncClient) -> str:
    """TTS: synthesize 'hi' via Google Cloud TTS REST API."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")

    if not creds_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set")
    if not os.path.exists(creds_path):
        raise FileNotFoundError(f"Credentials file not found: {creds_path}")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT_ID not set")

    from google.oauth2 import service_account
    import google.auth.transport.requests

    credentials = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    credentials.refresh(google.auth.transport.requests.Request())

    resp = await client.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        json={
            "input": {"text": "hi"},
            "voice": {"languageCode": "en-US"},
            "audioConfig": {"audioEncoding": "LINEAR16"},
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("audioContent"):
        raise ValueError("Empty audio response")
    return "tts"


async def _check_sarvam(client: httpx.AsyncClient) -> str:
    """TTS: synthesize 'hi' via Sarvam streaming TTS."""
    api_key = os.getenv("SARVAM_API_KEY")
    from sarvamai import AsyncSarvamAI, AudioOutput

    sarvam_client = AsyncSarvamAI(api_subscription_key=api_key)

    async with sarvam_client.text_to_speech_streaming.connect(
        model="bulbul:v3-beta"
    ) as ws:
        await ws.configure(
            target_language_code="en-IN",
            speaker="aditya",
            output_audio_codec="wav",
        )
        await ws.convert("hi")
        await ws.flush()

        async for message in ws:
            if isinstance(message, AudioOutput):
                break

    return "tts"


async def _check_elevenlabs(client: httpx.AsyncClient) -> str:
    """TTS: synthesize 'hi' via ElevenLabs."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = "m5qndnI7u4OAdXhH0Mr5"
    resp = await client.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": "hi",
            "model_id": "eleven_multilingual_v2",
        },
    )
    resp.raise_for_status()
    if len(resp.content) < 100:
        raise ValueError("Empty audio response")
    return "tts"


async def _check_cartesia(client: httpx.AsyncClient) -> str:
    """TTS: synthesize 'hi' via Cartesia."""
    api_key = os.getenv("CARTESIA_API_KEY")
    resp = await client.post(
        "https://api.cartesia.ai/tts/bytes",
        headers={
            "X-API-Key": api_key,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json",
        },
        json={
            "model_id": "sonic-3",
            "transcript": "hi",
            "voice": {
                "mode": "id",
                "id": "faf0731e-dfb9-4cfc-8119-259a79b27e12",
            },
            "language": "en",
            "output_format": {
                "container": "wav",
                "sample_rate": 24000,
                "encoding": "pcm_f32le",
            },
        },
    )
    resp.raise_for_status()
    if len(resp.content) < 100:
        raise ValueError("Empty audio response")
    return "tts"


# Smallest retired lightning-v2 (the standalone smallestai SDK's hard-coded
# model), so it now returns HTTP 410 Gone. lightning-v3.1 is the current model
# and matches what pipecat's SmallestTTSService uses in the real TTS/agent path
# (see calibrate_agent/utils.py:create_tts_service).
_SMALLEST_TTS_WS_URL = (
    "wss://waves-api.smallest.ai/api/v1/lightning-v3.1/get_speech/stream"
)


_SMALLEST_STT_WS_URL = "wss://api.smallest.ai/waves/v1/stt/live"


async def _check_smallest_tts() -> str:
    """TTS: synthesize 'hi' via Smallest lightning-v3.1 streaming TTS."""
    from websockets.asyncio.client import connect as websocket_connect

    api_key = os.getenv("SMALLEST_API_KEY")
    payload = {
        "text": "hi",
        "voice_id": "sophia",
        "language": "en",
        "sample_rate": 24000,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    async with websocket_connect(
        _SMALLEST_TTS_WS_URL, additional_headers=headers
    ) as ws:
        await ws.send(json.dumps(payload))
        async for message in ws:
            msg = json.loads(message)
            status = msg.get("status")
            if status == "chunk" and msg.get("data", {}).get("audio"):
                return "tts"  # Got audio — TTS works
            if status == "error":
                raise RuntimeError(msg.get("error") or msg.get("message") or msg)
            if status == "complete":
                break
    raise ValueError("No audio generated")


async def _check_smallest_stt() -> str:
    """STT: stream silence through Smallest's Pulse STT WebSocket."""
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.exceptions import ConnectionClosed

    api_key = os.getenv("SMALLEST_API_KEY")
    params = {
        "model": "pulse",
        "language": "en",
        "encoding": "linear16",
        "sample_rate": "16000",
        "word_timestamps": "false",
    }
    ws_url = f"{_SMALLEST_STT_WS_URL}?{urlencode(params)}"
    headers = {"Authorization": f"Bearer {api_key}"}
    silence = b"\x00\x00" * 8000  # 0.5s @ 16 kHz, 16-bit mono PCM

    async with websocket_connect(ws_url, additional_headers=headers) as ws:
        await ws.send(silence)
        await ws.send(json.dumps({"type": "close_stream"}))
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except (asyncio.TimeoutError, ConnectionClosed):
                break  # Handshake + stream accepted, no error surfaced
            try:
                output = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(output, dict) and (
                output.get("type") == "error" or output.get("error")
            ):
                error = output.get("message") or output.get("error") or output
                raise RuntimeError(f"Smallest STT error: {error}")
            break  # Got a valid response — STT pipeline is live
    return "stt"


async def _check_smallest(client: httpx.AsyncClient) -> str:
    """Verify both of Smallest's declared capabilities: STT and TTS."""

    async def _labeled(coro, label):
        try:
            return await coro
        except Exception as e:
            raise RuntimeError(f"{label} {e}") from e

    await asyncio.gather(
        _labeled(_check_smallest_tts(), "TTS:"),
        _labeled(_check_smallest_stt(), "STT:"),
    )
    return "stt,tts"


async def _check_deepgram(client: httpx.AsyncClient) -> str:
    """STT: transcribe silence via Deepgram."""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    resp = await client.post(
        "https://api.deepgram.com/v1/listen",
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
        },
        params={"model": "nova-3", "language": "en"},
        content=_SILENCE_WAV,
    )
    resp.raise_for_status()
    return "stt"


async def _check_gemini(client: httpx.AsyncClient) -> str:
    """TTS: synthesize 'hi' via the Gemini TTS model (google-genai)."""
    from google import genai
    from google.genai import types as genai_types

    genai_client = genai.Client(api_key=get_gemini_api_key())
    resp = await genai_client.aio.models.generate_content(
        model=TTS_PROVIDER_MODELS["gemini"],
        contents="hi",
        config=genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=get_tts_voice("gemini", "english")
                    )
                ),
            ),
        ),
    )
    audio = resp.candidates[0].content.parts[0].inline_data.data
    if not audio or len(audio) < 100:
        raise ValueError("Empty audio response")
    return "tts"


_CHECK_FUNCTIONS = {
    "deepgram": _check_deepgram,
    "openai": _check_openai,
    "groq": _check_groq,
    "google": _check_google,
    "gemini": _check_gemini,
    "sarvam": _check_sarvam,
    "elevenlabs": _check_elevenlabs,
    "cartesia": _check_cartesia,
    "smallest": _check_smallest,
    "openrouter": _check_openrouter,
}


# ─────────────────────────────────────────────────────────────────────
# Check runner
# ─────────────────────────────────────────────────────────────────────


async def _emit_status_event(
    emit, provider: dict, stage: str, message: str, result=None
):
    if emit is None:
        return

    event = {
        "type": "result" if result is not None else "progress",
        "provider": provider["name"],
        "types": provider["types"],
        "stage": stage,
        "message": message,
    }
    if result is not None:
        event["result"] = _status_json_entry(result)
        event["_raw_result"] = result
    await emit(event)


async def _check_single_provider(
    provider: dict,
    client: httpx.AsyncClient,
    emit=None,
) -> dict:
    """Check a single provider's API key and make a real API call."""
    name = provider["name"]
    env_vars = provider["env_vars"]

    # Check if required env vars are set
    missing = [v for v in env_vars if not os.getenv(v)]
    if missing:
        result = {
            "name": name,
            "types": provider["types"],
            "key_set": False,
            "missing_vars": missing,
            "status": "skipped",
            "check_type": None,
            "error": None,
            "latency_ms": None,
        }
        missing_vars = ", ".join(missing)
        await _emit_status_event(
            emit,
            provider,
            "skipped",
            f"Skipped: missing {missing_vars}",
            result,
        )
        return result

    # Make a real API call to verify the provider works
    check_fn = _CHECK_FUNCTIONS[name]
    start = time.time()
    try:
        await _emit_status_event(emit, provider, "input_sent", "Input sent")
        check_type = await asyncio.wait_for(check_fn(client), timeout=30.0)
        latency_ms = round((time.time() - start) * 1000)
        await _emit_status_event(emit, provider, "output_received", "Output received")
        result = {
            "name": name,
            "types": provider["types"],
            "key_set": True,
            "missing_vars": [],
            "status": "ok",
            "check_type": check_type,
            "error": None,
            "latency_ms": latency_ms,
        }
        await _emit_status_event(emit, provider, "working", "Working", result)
        return result
    except asyncio.TimeoutError:
        latency_ms = round((time.time() - start) * 1000)
        result = {
            "name": name,
            "types": provider["types"],
            "key_set": True,
            "missing_vars": [],
            "status": "fail",
            "check_type": None,
            "error": "Timed out (30s)",
            "latency_ms": latency_ms,
        }
        await _emit_status_event(emit, provider, "not_working", "Not working", result)
        return result
    except httpx.HTTPStatusError as e:
        latency_ms = round((time.time() - start) * 1000)
        await _emit_status_event(emit, provider, "output_received", "Output received")
        result = {
            "name": name,
            "types": provider["types"],
            "key_set": True,
            "missing_vars": [],
            "status": "fail",
            "check_type": None,
            "error": f"HTTP {e.response.status_code}",
            "latency_ms": latency_ms,
        }
        await _emit_status_event(emit, provider, "not_working", "Not working", result)
        return result
    except Exception as e:
        latency_ms = round((time.time() - start) * 1000)
        error_msg = str(e)
        if len(error_msg) > 50:
            error_msg = error_msg[:50] + "..."
        result = {
            "name": name,
            "types": provider["types"],
            "key_set": True,
            "missing_vars": [],
            "status": "fail",
            "check_type": None,
            "error": error_msg,
            "latency_ms": latency_ms,
        }
        await _emit_status_event(emit, provider, "not_working", "Not working", result)
        return result


def _status_json_entry(result: dict) -> dict:
    """Convert an internal provider result to the public status shape."""
    public_status = result["status"]
    if public_status == "ok":
        public_status = "pass"

    return {
        "status": public_status,
        "types": result["types"],
        "error": result["error"],
        "latency_ms": result["latency_ms"],
        "missing_vars": result["missing_vars"],
        "key_set": result["key_set"],
    }


async def iter_provider_results():
    """Yield internal provider check results as soon as each provider finishes."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            asyncio.create_task(_check_single_provider(provider, client))
            for provider in PROVIDERS
        ]

        try:
            for task in asyncio.as_completed(tasks):
                yield await task
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def iter_status_events(include_internal: bool = False):
    """Yield progress and result events as provider checks run."""
    queue = asyncio.Queue()

    async def emit(event):
        await queue.put(event)

    async with httpx.AsyncClient(timeout=30.0) as client:

        async def run_check(provider):
            result = await _check_single_provider(provider, client, emit=emit)
            await queue.put(
                {"type": "_done", "provider": provider["name"], "result": result}
            )

        tasks = [asyncio.create_task(run_check(provider)) for provider in PROVIDERS]
        finished = 0

        try:
            while finished < len(tasks):
                event = await queue.get()
                if event["type"] == "_done":
                    finished += 1
                    continue
                if not include_internal:
                    event.pop("_raw_result", None)
                yield event
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def iter_status():
    """Yield public JSON-ready status results as soon as each provider finishes."""
    async for event in iter_status_events():
        if event["type"] == "result":
            yield {
                "provider": event["provider"],
                "result": event["result"],
            }


async def run_status_live(table: bool = False):
    """Print live status progress and return the final status JSON."""
    if table:
        print("\n  Checking provider status...")

    status_json = {}
    raw_results = []
    async for event in iter_status_events(include_internal=True):
        raw_result = event.pop("_raw_result", None)
        if table:
            _print_status_event(event)
        else:
            print(json.dumps(event), flush=True)

        if event["type"] == "result":
            status_json[event["provider"]] = event["result"]
            if raw_result is not None:
                raw_results.append(raw_result)

    if table:
        _print_results(raw_results)
    else:
        print(
            json.dumps({"type": "summary", "result": status_json}, indent=2), flush=True
        )

    return status_json


# ─────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────


def _print_results(results: list) -> None:
    """Print a formatted status table."""
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    print()
    print(f"  {BOLD}Provider Status{RESET}")
    print(f"  {'─' * 72}")
    print(f"  {'Provider':<14} {'Type':<14} {'API Key':<14} {'Status'}")
    print(f"  {'─' * 72}")

    for r in results:
        name = r["name"]
        types_str = ",".join(t.upper() for t in r["types"])

        if r["key_set"]:
            key_text = "✓ Set"
            key_col = GREEN
        else:
            key_text = "✗ Not set"
            key_col = RED

        if r["status"] == "ok":
            check = r["check_type"].upper()
            latency = f" {DIM}({r['latency_ms']}ms){RESET}"
            status_col = GREEN
            status_text = f"✓ {check} call OK"
        elif r["status"] == "skipped":
            status_col = YELLOW
            status_text = "— Skipped"
            latency = ""
        else:
            status_col = RED
            status_text = f"✗ {r['error']}"
            latency = ""

        print(
            f"  {name:<14} {types_str:<14} "
            f"{key_col}{key_text:<14}{RESET} "
            f"{status_col}{status_text}{RESET}{latency}"
        )

    print(f"  {'─' * 72}")

    # Summary
    ok_count = sum(1 for r in results if r["status"] == "ok")
    error_count = sum(1 for r in results if r["status"] == "fail")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")

    parts = []
    if ok_count:
        parts.append(f"{GREEN}{ok_count} ok{RESET}")
    if error_count:
        parts.append(f"{RED}{error_count} error{RESET}")
    if skipped_count:
        parts.append(f"{YELLOW}{skipped_count} skipped (no API key){RESET}")

    print(f"\n  {' · '.join(parts)}")
    print()


def _print_status_event(event: dict) -> None:
    """Print one live status event."""
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    provider = event["provider"]
    stage = event["stage"]

    if event["type"] == "progress":
        icon = "→" if stage == "input_sent" else "←"
        print(f"  {DIM}{icon} {provider}: {event['message']}{RESET}", flush=True)
        return

    result = event["result"]
    if result["status"] == "pass":
        latency = (
            f" ({result['latency_ms']}ms)" if result["latency_ms"] is not None else ""
        )
        print(f"  {GREEN}✓ {provider}: working{latency}{RESET}", flush=True)
    elif result["status"] == "skipped":
        missing = ", ".join(result["missing_vars"])
        print(f"  {YELLOW}— {provider}: skipped, missing {missing}{RESET}", flush=True)
    else:
        print(
            f"  {RED}✗ {provider}: not working - {result['error']}{RESET}", flush=True
        )


def _print_status_summary(status_json: dict) -> None:
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    pass_count = sum(1 for r in status_json.values() if r["status"] == "pass")
    fail_count = sum(1 for r in status_json.values() if r["status"] == "fail")
    skipped_count = sum(1 for r in status_json.values() if r["status"] == "skipped")

    parts = []
    if pass_count:
        parts.append(f"{GREEN}{pass_count} working{RESET}")
    if fail_count:
        parts.append(f"{RED}{fail_count} not working{RESET}")
    if skipped_count:
        parts.append(f"{YELLOW}{skipped_count} skipped (no API key){RESET}")

    print(f"\n  {' · '.join(parts)}\n", flush=True)


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────


async def main(quiet: bool = False):
    """Run all provider status checks, display results, and return JSON status.

    Args:
        quiet: If True, suppress the printed table (for machine-readable output).
    """
    if not quiet:
        return await run_status_live(table=True)

    status_json = {}
    async for result in iter_provider_results():
        status_json[result["name"]] = _status_json_entry(result)

    return status_json
