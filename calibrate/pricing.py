"""Pricing lookup used by Calibrate metrics."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

STT_DEFAULT_MODELS = {
    "deepgram": "nova-3",
    "openai": "gpt-4o-transcribe",
    "groq": "whisper-large-v3-turbo",
    "google": "chirp_3",
    "sarvam": "saaras:v3",
    "elevenlabs": "scribe_v2",
    "cartesia": "ink-whisper",
    "smallest": "pulse_streaming",
}


def resolve_pricing(
    component: str,
    provider: str,
    model: str | None = None,
) -> dict | None:
    """Return normalized STT pricing for a provider/model.

    Only STT pricing is supported in this PR. Add a new component here when
    the corresponding metric pipeline is implemented.
    """
    if component != "stt":
        raise ValueError(f"Unsupported pricing component: {component}")

    provider_key = provider.lower()
    model_name = model or STT_DEFAULT_MODELS.get(provider_key)
    if not model_name:
        return None

    component_pricing = _load_pricing_data().get("stt", {})
    provider_pricing = _get_case_insensitive(component_pricing, provider_key)
    if not isinstance(provider_pricing, dict):
        return None

    model_pricing = _get_case_insensitive(provider_pricing, model_name)
    if not isinstance(model_pricing, dict):
        return None

    price = model_pricing.get("price_per_minute_usd")
    if not isinstance(price, (int, float)) or price < 0:
        return None

    return {
        "provider": provider,
        "model": model_name,
        "currency": "USD",
        "billing_unit": "minute",
        "price_per_minute_usd": float(price),
        "pricing_source": "calibrate_default",
    }


@lru_cache(maxsize=1)
def _load_pricing_data() -> dict:
    try:
        pricing_path = resources.files("calibrate").joinpath("pricing_data.json")
        return json.loads(pricing_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ModuleNotFoundError):
        return {}


def _get_case_insensitive(mapping: dict, key: str) -> object:
    if key in mapping:
        return mapping[key]

    lowered = key.lower()
    for candidate_key, value in mapping.items():
        if isinstance(candidate_key, str) and candidate_key.lower() == lowered:
            return value
    return None
