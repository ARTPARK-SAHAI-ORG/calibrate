"""Pricing lookup used by Calibrate metrics."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from calibrate_agent.utils import STT_PROVIDER_MODELS


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
    model_name = model or STT_PROVIDER_MODELS.get(provider_key)
    if not model_name:
        return None

    component_pricing = _load_pricing_data().get("stt", {})
    provider_pricing = component_pricing.get(provider_key)
    if not isinstance(provider_pricing, dict):
        return None

    model_pricing = provider_pricing.get(model_name)
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
        pricing_path = resources.files(__package__).joinpath("pricing_data.json")
        return json.loads(pricing_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ModuleNotFoundError):
        return {}
