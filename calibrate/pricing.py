"""Small pricing resolver used by Calibrate metrics.

The resolver is intentionally dependency-free. Defaults live in this package,
while config.json can override provider rates for custom contracts or drift.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any


def resolve_pricing(
    component: str,
    provider: str,
    overrides: dict | None = None,
    model: str | None = None,
) -> dict | None:
    """Return normalized pricing for a component/provider/model.

    Supported override shapes:
      {"price_per_minute_usd": 0.006}
      {"openai": {"price_per_minute_usd": 0.006}}
      {"stt": {"openai": {"price_per_minute_usd": 0.006}}}
      {"stt": {"openai": {"models": {"gpt-4o-transcribe": {...}}}}}

    Invalid override prices are ignored so evals can still finish; the
    resolver then falls back to defaults when a default exists.
    """
    override_entry = _find_pricing_entry(overrides, component, provider, model)
    override_pricing = _normalize_pricing_entry(
        override_entry,
        source="config_override",
        default_billing_unit=_default_billing_unit(component),
    )
    if override_pricing:
        return override_pricing

    default_entry = _find_pricing_entry(
        _load_default_pricing(component),
        component,
        provider,
        model,
    )
    return _normalize_pricing_entry(
        default_entry,
        source="calibrate_default",
        default_billing_unit=_default_billing_unit(component),
    )


@lru_cache(maxsize=None)
def _load_default_pricing(component: str) -> dict:
    try:
        pricing_path = resources.files("calibrate").joinpath(
            "pricing_data",
            f"{component}.json",
        )
        return json.loads(pricing_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ModuleNotFoundError):
        return {}


def _find_pricing_entry(
    pricing_map: dict | None,
    component: str,
    provider: str,
    model: str | None = None,
) -> dict | None:
    if not isinstance(pricing_map, dict):
        return None

    if _looks_like_price_entry(pricing_map):
        return pricing_map

    component_map = pricing_map.get(component)
    if isinstance(component_map, dict):
        entry = _find_in_provider_map(component_map, provider, model)
        if entry:
            return entry

    return _find_in_provider_map(pricing_map, provider, model)


def _find_in_provider_map(
    provider_map: dict,
    provider: str,
    model: str | None = None,
) -> dict | None:
    provider_entry = _get_case_insensitive(provider_map, provider)
    if not isinstance(provider_entry, dict):
        return None

    if model:
        model_entry = _get_case_insensitive(provider_entry, model)
        if isinstance(model_entry, dict):
            return _with_model(model_entry, model)

        models = provider_entry.get("models")
        if isinstance(models, dict):
            model_entry = _get_case_insensitive(models, model)
            if isinstance(model_entry, dict):
                return _with_model(model_entry, model)

    default_model = provider_entry.get("default_model")
    models = provider_entry.get("models")
    if isinstance(default_model, str) and isinstance(models, dict):
        model_entry = _get_case_insensitive(models, default_model)
        if isinstance(model_entry, dict):
            return _with_model(model_entry, default_model)

    if isinstance(models, dict) and len(models) == 1:
        model_name, model_entry = next(iter(models.items()))
        if isinstance(model_entry, dict):
            return _with_model(model_entry, str(model_name))

    return provider_entry


def _normalize_pricing_entry(
    entry: dict | None,
    source: str,
    default_billing_unit: str,
) -> dict | None:
    if not isinstance(entry, dict):
        return None

    price = _first_float(
        entry,
        "price_per_unit_usd",
        "price_per_audio_minute_usd",
        "price_per_minute_usd",
    )
    if price is None:
        return None

    billing_unit = str(entry.get("billing_unit") or default_billing_unit)
    return {
        "billing_unit": billing_unit,
        "price_per_unit_usd": price,
        "pricing_source": source,
        **({"model": str(entry["model"])} if entry.get("model") else {}),
    }


def _first_float(entry: dict, *keys: str) -> float | None:
    for key in keys:
        if key not in entry:
            continue
        try:
            value = float(entry[key])
        except (TypeError, ValueError):
            return None
        if value < 0:
            return None
        return value
    return None


def _get_case_insensitive(mapping: dict, key: str) -> Any:
    if key in mapping:
        return mapping[key]

    lowered = key.lower()
    for candidate_key, value in mapping.items():
        if isinstance(candidate_key, str) and candidate_key.lower() == lowered:
            return value
    return None


def _looks_like_price_entry(entry: dict) -> bool:
    return any(
        key in entry
        for key in (
            "price_per_unit_usd",
            "price_per_audio_minute_usd",
            "price_per_minute_usd",
        )
    )


def _with_model(entry: dict, model: str) -> dict:
    return {**entry, "model": model}


def _default_billing_unit(component: str) -> str:
    if component == "stt":
        return "audio_minute"
    if component == "tts":
        return "character"
    return "unit"
