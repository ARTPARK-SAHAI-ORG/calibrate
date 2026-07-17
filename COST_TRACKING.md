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

### General — every provider
- **Estimates, not invoices.** Proportional estimates from published rates.
  Real bills differ due to per-request minimums, rounding, volume/committed-use
  discounts, free tiers, and taxes (e.g. GST).
- **Point-in-time rates.** Captured ~July 2026; providers change pricing and
  model versions — bundled rates can go stale.
- **Pay-as-you-go / entry tier assumed.** Where a provider only sells
  subscription tiers, the entry/standard rate is used; high-volume tiers are
  cheaper.
- **Free tiers are not deducted** (e.g. Google's 60 min/month, 1M chars/month).

### Currency / FX (INR providers — Sarvam)
- `cost_usd` uses a **live mid-market FX rate** (USD→INR, ECB via Frankfurter,
  daily). The real card charge includes an **FX margin + GST**, so actual USD
  differs.
- **FX fallback:** if the live lookup fails, a hardcoded rate (₹96.35, as of
  2026-07-16) is used and may be stale.
- **Sarvam `bulbul:v3` is beta pricing** (₹30/10K), subject to change.

### Audio-billed TTS (OpenAI, Gemini, Google-Sindhi)
- Billed by audio-output tokens; approximated as
  `measured audio minutes × per-minute rate`. Actual token count varies with
  content, and the per-minute figures ($0.015 / $0.03) are provider estimates.
- Cost depends on the synthesized **speech rate** — the same text can cost
  differently across these providers because their audio durations differ.
- **Gemini 3.1 Flash TTS is preview pricing** (pre-GA); batch mode (½ price) is
  not used.

### Per-provider
- **Cartesia (STT + TTS):** no flat rate — credit/subscription based. Priced at
  the entry **Pro** tier ($0.00005/credit); effective cost drops to ~$37/1M
  (TTS) and lower (STT) at the **Scale** tier.
- **ElevenLabs TTS ($100/1M):** `eleven_multilingual_v2` flat $0.10/1K; cheaper
  Flash/Turbo models exist. Sindhi uses `eleven_v3` (same price).
- **ElevenLabs STT ($0.0065/min):** `scribe_v2_realtime`; add-ons (diarization,
  entity detection) are extra.
- **Deepgram STT ($0.0048/min):** nova-3 streaming PAYG; multilingual is higher
  ($0.0058), committed (Growth) tier lower ($0.0042).
- **Google STT ($0.016/min):** billed in 15-second increments (rounded up);
  async Dynamic Batch is cheaper ($0.004/min).
- **Groq:** 10-second minimum per request — short clips cost more than the flat
  per-minute rate implies.
- **Smallest:** pricing page is JS-rendered/approximate; uses the standard (not
  Pro) TTS model and the standalone (not agent-bundle) STT rate.
