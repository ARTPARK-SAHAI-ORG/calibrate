"""Pricing lookup used by Calibrate metrics."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from calibrate_agent.utils import STT_PROVIDER_MODELS

# TTS pricing model labels. utils.TTS_PROVIDER_MODELS omits google/smallest
# (their synth functions don't read a shared model constant), so pricing keeps
# its own complete map of the model each provider is billed as.
TTS_DEFAULT_MODELS = {
    "cartesia": "sonic-3.5",
    "openai": "gpt-4o-mini-tts",
    "groq": "canopylabs/orpheus-v1-english",
    "google": "chirp3-hd",
    "gemini": "gemini-3.1-flash-tts-preview",
    "elevenlabs": "eleven_multilingual_v2",
    "sarvam": "bulbul:v3",
    "smallest": "lightning-v3.1",
}

# Per component: (price key in pricing_data.json, normalized billing unit).
# STT bills per minute of audio; TTS bills per input character (stored as a
# per-million-character rate for readable numbers).
_COMPONENT_PRICE = {
    "stt": ("price_per_minute_usd", "minute"),
    "tts": ("price_per_million_chars_usd", "character"),
}

_COMPONENT_DEFAULT_MODELS = {
    "stt": STT_PROVIDER_MODELS,
    "tts": TTS_DEFAULT_MODELS,
}


def resolve_pricing(
    component: str,
    provider: str,
    model: str | None = None,
) -> dict | None:
    """Return normalized pricing for a provider/model of ``component``.

    ``component`` is ``"stt"`` (per-minute pricing) or ``"tts"``
    (per-million-character pricing). Returns ``None`` when no price is
    configured for the resolved provider/model.
    """
    if component not in _COMPONENT_PRICE:
        raise ValueError(f"Unsupported pricing component: {component}")

    price_key, billing_unit = _COMPONENT_PRICE[component]

    provider_key = provider.lower()
    model_name = model or _COMPONENT_DEFAULT_MODELS[component].get(provider_key)
    if not model_name:
        return None

    component_pricing = _load_pricing_data().get(component, {})
    provider_pricing = component_pricing.get(provider_key)
    if not isinstance(provider_pricing, dict):
        return None

    model_pricing = provider_pricing.get(model_name)
    if not isinstance(model_pricing, dict):
        return None

    price = model_pricing.get(price_key)
    if not isinstance(price, (int, float)) or price < 0:
        return None

    return {
        "provider": provider,
        "model": model_name,
        "currency": "USD",
        "billing_unit": billing_unit,
        price_key: float(price),
    }


@lru_cache(maxsize=1)
def _load_pricing_data() -> dict:
    try:
        pricing_path = resources.files(__package__).joinpath("pricing_data.json")
        return json.loads(pricing_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ModuleNotFoundError):
        return {}
