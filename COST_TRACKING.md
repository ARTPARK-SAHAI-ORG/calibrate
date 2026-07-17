# Cost tracking

How Calibrate estimates per-provider cost for STT and TTS evaluation runs, the
shape of the `cost` object written to each provider's `metrics.json`, and the
caveats to surface anywhere costs are displayed.

## Where things live

- **Rates:** [`calibrate_agent/pricing_data.json`](calibrate_agent/pricing_data.json)
  — per component (`stt` / `tts`), per provider, per model. The key suffix
  encodes the billing unit and currency: `price_per_minute_usd`,
  `price_per_million_chars_usd`, `price_per_minute_inr`,
  `price_per_million_chars_inr`.
- **Resolution + math:** [`calibrate_agent/pricing.py`](calibrate_agent/pricing.py)
  — `resolve_pricing(component, provider, model)` returns the native
  `currency`, `billing_unit`, and `native_rate`; `cost_breakdown(...)` builds
  the cost fields.
- **Live FX:** `get_usd_to_inr_rate()` in
  [`calibrate_agent/utils.py`](calibrate_agent/utils.py) — fetched once per run
  from Frankfurter (ECB reference rates, no API key), cached, with a hardcoded
  fallback so runs never break on a network hiccup.
- **Cost builders:** `_build_stt_cost_metrics` in `stt/eval.py`,
  `_build_tts_cost_metrics` in `tts/eval.py`.

## The metric

Costs are normalized to a comparable **`cost_usd`** across all providers.
Each provider is priced in the **unit it actually bills in — no unit
conversion**:

- **STT** is always per **minute** of audio.
- **TTS** is per input **character** for most providers, but audio-token-billed
  models (OpenAI, Gemini, Google's Gemini-TTS used for Sindhi) are per output
  **minute**, measured from the synthesized `.wav` duration.

Providers billed in a non-USD currency (Sarvam, in INR) report their native
rate and total, plus the FX `conversion_rate` used; `cost_usd` is always
present.

## Cost object fields

Every `cost` object has: `provider`, `pricing_model`, `billing_unit`,
`currency`, `cost_usd`.

| Added when… | Fields |
| --- | --- |
| `billing_unit == "minute"` | `total_seconds`, `audio_minutes`, `cost_per_minute_currency` |
| `billing_unit == "character"` | `total_characters`, `cost_per_million_chars_currency` |
| `currency != "USD"` | `cost_in_currency` (total in the native currency), `conversion_rate` (native-currency units per 1 USD) |
| STT, some rows unreadable | `excluded_row_indices` |

The per-unit rate field is `cost_per_minute_currency` /
`cost_per_million_chars_currency` regardless of currency — the `currency` field
says which currency the rate and `cost_in_currency` are denominated in.

## The 6 output shapes

### 1. STT · minute · USD
_Deepgram, OpenAI, Groq, Google, ElevenLabs, Cartesia, Smallest_

```jsonc
{
  "provider": "deepgram", "pricing_model": "nova-3",
  "billing_unit": "minute", "total_seconds": 120.0, "audio_minutes": 2.0,
  "currency": "USD",
  "cost_per_minute_currency": 0.0048,
  "cost_usd": 0.0096
}
```

### 2. STT · minute · INR
_Sarvam (`saaras:v3`)_

```jsonc
{
  "provider": "sarvam", "pricing_model": "saaras:v3",
  "billing_unit": "minute", "total_seconds": 120.0, "audio_minutes": 2.0,
  "currency": "INR",
  "cost_per_minute_currency": 0.5,      // ₹/min
  "cost_in_currency": 1.0,              // ₹ total
  "conversion_rate": 96.35,             // ₹ per USD
  "cost_usd": 0.01038
}
```

### 3. TTS · character · USD
_ElevenLabs, Cartesia, Groq, Google `chirp3-hd`, Smallest_

```jsonc
{
  "provider": "groq", "pricing_model": "canopylabs/orpheus-v1-english",
  "billing_unit": "character", "total_characters": 500000,
  "currency": "USD",
  "cost_per_million_chars_currency": 22.0,
  "cost_usd": 11.0
}
```

### 4. TTS · character · INR
_Sarvam (`bulbul:v3`)_

```jsonc
{
  "provider": "sarvam", "pricing_model": "bulbul:v3",
  "billing_unit": "character", "total_characters": 500000,
  "currency": "INR",
  "cost_per_million_chars_currency": 3000.0,   // ₹/1M chars
  "cost_in_currency": 1500.0,                  // ₹ total
  "conversion_rate": 96.35,                    // ₹ per USD
  "cost_usd": 15.57
}
```

### 5. TTS · minute · USD
_OpenAI (`gpt-4o-mini-tts`), Gemini (`gemini-3.1-flash-tts-preview`),
Google-Sindhi (`gemini-2.5-flash-tts`)_

```jsonc
{
  "provider": "openai", "pricing_model": "gpt-4o-mini-tts",
  "billing_unit": "minute", "total_seconds": 180.0, "audio_minutes": 3.0,
  "currency": "USD",
  "cost_per_minute_currency": 0.015,
  "cost_usd": 0.045
}
```

### 6. TTS · minute · INR — schema-possible, none today
No current provider is both audio-billed and INR-priced. The schema supports it:
shape 5 plus `cost_in_currency` and `conversion_rate` (as in shape 2).

## Caveats (surface these wherever cost is displayed)

- **Bundled, point-in-time rates.** Costs are estimated from a rate table
  captured ~July 2026, not live provider pricing — they drift as providers
  change prices.
- **Entry tier assumed.** We use the standard pay-as-you-go / entry tier;
  volume or committed-use discounts and free tiers are not modeled.
- **Provider billing quirks not modeled.** Per-request minimums, billing
  increments/rounding, and taxes (GST) are ignored, so estimates run low for
  many short requests.
- **One variant per provider.** Each provider is priced at a single model/tier
  we selected — Cartesia entry (Pro) tier, ElevenLabs `eleven_multilingual_v2`,
  Deepgram nova-3 streaming, Smallest standard Lightning / standalone Pulse —
  not the provider's other variants or tiers.
- **Audio-billed TTS is approximated.** OpenAI, Gemini, and Google-Sindhi are
  billed by audio-output tokens; we estimate cost as
  `measured audio minutes × per-minute rate`, so the figure scales with the
  synthesized audio length and the same text can differ across providers.
- **INR converted via live mid-market FX.** For INR-billed providers, `cost_usd`
  uses a live mid-market USD→INR rate (excludes FX margin and GST) and falls
  back to a fixed rate if the live lookup fails.
