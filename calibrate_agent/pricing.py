"""Pricing lookup used by Calibrate metrics."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from calibrate_agent.utils import STT_PROVIDER_MODELS, get_usd_to_inr_rate

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

# Per component: the billing units it can use, each a
# (normalized billing unit, price-key stem). STT is always per-minute. TTS is
# usually per input character, but audio-token-billed models (OpenAI, Gemini)
# are per output minute — so a TTS model may use either. The currency is
# appended to the stem (``price_per_minute_usd`` / ``price_per_minute_inr``),
# so the stored key drives both the unit and the currency.
_COMPONENT_BILLING = {
    "stt": (("minute", "price_per_minute"),),
    "tts": (
        ("character", "price_per_million_chars"),
        ("minute", "price_per_minute"),
    ),
}

# Checked in order; first key present wins. USD is the default for providers
# that publish in USD; INR covers providers billed in rupees (e.g. Sarvam).
_SUPPORTED_CURRENCIES = ("usd", "inr")

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

    ``component`` is ``"stt"`` or ``"tts"``. The returned dict carries the
    resolved ``billing_unit`` (``"minute"`` or ``"character"``), the native
    ``currency``, and the ``native_rate`` in that currency. Returns ``None``
    when no price is configured for the resolved provider/model.
    """
    if component not in _COMPONENT_BILLING:
        raise ValueError(f"Unsupported pricing component: {component}")

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

    for billing_unit, price_stem in _COMPONENT_BILLING[component]:
        for currency in _SUPPORTED_CURRENCIES:
            price = model_pricing.get(f"{price_stem}_{currency}")
            if isinstance(price, (int, float)) and price >= 0:
                return {
                    "provider": provider,
                    "model": model_name,
                    "currency": currency.upper(),
                    "billing_unit": billing_unit,
                    "native_rate": float(price),
                }
    return None


def cost_breakdown(
    pricing: dict,
    usage_in_units: float,
    rate_field_stem: str,
) -> dict:
    """Build the cost fields for a metrics ``cost`` object.

    ``usage_in_units`` is total usage measured in the billing unit (minutes for
    STT, millions of characters for TTS). The native per-unit rate is reported
    under ``<rate_field_stem>_currency`` (denominated in ``currency``). Non-USD
    providers additionally get the native-currency total (``cost_in_currency``)
    and the ``conversion_rate`` used (units of the native currency per 1 USD).
    ``cost_usd`` is always present.
    """
    currency = pricing["currency"]
    native_rate = pricing["native_rate"]
    cost_native = usage_in_units * native_rate

    fields = {
        "currency": currency,
        f"{rate_field_stem}_currency": native_rate,
    }
    if currency == "USD":
        fields["cost_usd"] = cost_native
    else:
        # Only INR is supported today; extend here for other currencies.
        conversion_rate = get_usd_to_inr_rate()  # native units per 1 USD
        fields["cost_in_currency"] = cost_native
        fields["conversion_rate"] = conversion_rate
        fields["cost_usd"] = cost_native / conversion_rate
    return fields


@lru_cache(maxsize=1)
def _load_pricing_data() -> dict:
    try:
        pricing_path = resources.files(__package__).joinpath("pricing_data.json")
        return json.loads(pricing_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ModuleNotFoundError):
        return {}
