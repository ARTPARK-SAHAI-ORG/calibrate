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

import httpx


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


async def _check_smallest(client: httpx.AsyncClient) -> str:
    """TTS: synthesize 'hi' via Smallest streaming TTS."""
    api_key = os.getenv("SMALLEST_API_KEY")

    def _sync_check():
        from smallestai.waves import TTSConfig, WavesStreamingTTS

        config = TTSConfig(
            voice_id="aditi",
            language="en",
            api_key=api_key,
            sample_rate=24000,
        )
        tts = WavesStreamingTTS(config)
        for chunk in tts.synthesize("hi"):
            if chunk:
                return  # Got audio — TTS works
        raise ValueError("No audio generated")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _sync_check)
    return "tts"


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


_CHECK_FUNCTIONS = {
    "deepgram": _check_deepgram,
    "openai": _check_openai,
    "groq": _check_groq,
    "google": _check_google,
    "sarvam": _check_sarvam,
    "elevenlabs": _check_elevenlabs,
    "cartesia": _check_cartesia,
    "smallest": _check_smallest,
    "openrouter": _check_openrouter,
}


# ─────────────────────────────────────────────────────────────────────
# Check runner
# ─────────────────────────────────────────────────────────────────────


async def _check_single_provider(provider: dict, client: httpx.AsyncClient) -> dict:
    """Check a single provider's API key and make a real API call."""
    name = provider["name"]
    env_vars = provider["env_vars"]

    # Check if required env vars are set
    missing = [v for v in env_vars if not os.getenv(v)]
    if missing:
        return {
            "name": name,
            "types": provider["types"],
            "key_set": False,
            "missing_vars": missing,
            "status": "skipped",
            "check_type": None,
            "error": None,
            "latency_ms": None,
        }

    # Make a real API call to verify the provider works
    check_fn = _CHECK_FUNCTIONS[name]
    start = time.time()
    try:
        check_type = await asyncio.wait_for(check_fn(client), timeout=30.0)
        latency_ms = round((time.time() - start) * 1000)
        return {
            "name": name,
            "types": provider["types"],
            "key_set": True,
            "missing_vars": [],
            "status": "ok",
            "check_type": check_type,
            "error": None,
            "latency_ms": latency_ms,
        }
    except asyncio.TimeoutError:
        latency_ms = round((time.time() - start) * 1000)
        return {
            "name": name,
            "types": provider["types"],
            "key_set": True,
            "missing_vars": [],
            "status": "error",
            "check_type": None,
            "error": "Timed out (30s)",
            "latency_ms": latency_ms,
        }
    except httpx.HTTPStatusError as e:
        latency_ms = round((time.time() - start) * 1000)
        return {
            "name": name,
            "types": provider["types"],
            "key_set": True,
            "missing_vars": [],
            "status": "error",
            "check_type": None,
            "error": f"HTTP {e.response.status_code}",
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = round((time.time() - start) * 1000)
        error_msg = str(e)
        if len(error_msg) > 50:
            error_msg = error_msg[:50] + "..."
        return {
            "name": name,
            "types": provider["types"],
            "key_set": True,
            "missing_vars": [],
            "status": "error",
            "check_type": None,
            "error": error_msg,
            "latency_ms": latency_ms,
        }


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
    error_count = sum(1 for r in results if r["status"] == "error")
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


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────


async def main(quiet: bool = False):
    """Run all provider status checks, display results, and return JSON status.

    Args:
        quiet: If True, suppress the printed table (for machine-readable output).
    """
    if not quiet:
        print("\n  Checking provider status...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [_check_single_provider(p, client) for p in PROVIDERS]
        results = await asyncio.gather(*tasks)

    if not quiet:
        _print_results(list(results))

    status_json = {
        r["name"]: {
            "status": "pass" if r["status"] == "ok" else "fail",
            "types": r["types"],
            "error": r["error"],
            "latency_ms": r["latency_ms"],
        }
        for r in results
    }

    return status_json
