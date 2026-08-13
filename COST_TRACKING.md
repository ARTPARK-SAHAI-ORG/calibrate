# Cost tracking

How Calibrate tracks cost for STT and TTS evaluation runs. There are two
different kinds of number here:

- **Provider cost** is *measured* after a run, from the audio duration or
  character count actually processed. This is the bulk of the document below:
  the shape of the `cost` object written to each provider's `metrics.json`,
  and the caveats to surface anywhere it's displayed.
- **Judge cost** is *estimated* before a run's LLM-as-judge calls are made,
  from a token heuristic run over the dataset about to be graded — see
  "Judge cost" below. It is a different kind of number from provider cost and
  the two should not be added together casually.

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
  from Frankfurter (ECB reference rates, no API key), retried with exponential
  backoff, and cached. There is no hardcoded fallback, so a reported `cost_usd`
  always reflects a real exchange rate. If the rate can't be fetched,
  `cost_breakdown` reports the native-currency cost only and omits `cost_usd`
  (the provider is left out of the USD comparison; the run continues).
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
rate and total. When the live FX rate is reachable they also get the
`conversion_rate` used and `cost_usd`; if it isn't, `cost_usd` is omitted and
the provider is left out of the USD comparison.

## Cost object fields

Every `cost` object has: `provider`, `pricing_model`, `billing_unit`,
`currency`. USD providers always have `cost_usd`; non-USD providers have it only
when the FX rate was fetched.

| Added when… | Fields |
| --- | --- |
| `billing_unit == "minute"` | `total_seconds`, `audio_minutes`, `cost_per_minute_currency` |
| `billing_unit == "character"` | `total_characters`, `cost_per_million_chars_currency` |
| `currency != "USD"` | `cost_in_currency` (total in the native currency); plus `conversion_rate` and `cost_usd` when the live FX rate is available |
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

## Judge cost

An LLM-as-judge run issues one paid LLM call per dataset row per evaluator (STT
also runs three built-in extra judges beyond the configured evaluator — see
the STT/TTS CLI docs). Before any of those calls is made, Calibrate estimates
what the whole judge phase will cost and asks for confirmation. Unlike
provider cost above, this number is computed **before** the work happens, from
an approximation of the input — it is a budgeting estimate, not a bill.

**Where things live:**

- **Rates:** the `llm` section of
  [`calibrate_agent/pricing_data.json`](calibrate_agent/pricing_data.json) —
  keyed by the full model id used by the judges (e.g.
  `"anthropic/claude-sonnet-4.5"`), USD per million tokens. Each model's entry
  has two columns, `openrouter` and `direct`:

  ```jsonc
  "anthropic/claude-sonnet-4.5": {
    "openrouter": {
      "input_price_per_million_tokens_usd": 3.0,
      "output_price_per_million_tokens_usd": 15.0
    },
    "direct": {
      "input_price_per_million_tokens_usd": 3.0,
      "output_price_per_million_tokens_usd": 15.0
    }
  }
  ```

  An entry may also carry `audio_input_price_per_million_tokens_usd` /
  `audio_output_price_per_million_tokens_usd` (models priced separately for
  audio) and `reasoning_billed_as_output: true` (models that bill hidden
  reasoning tokens at the output rate). Four models are priced today, the
  judge defaults: `openai/gpt-5.4-mini` (the text evaluators, e.g.
  `semantic_match`), `openai/gpt-audio` (the TTS `pronunciation` evaluator),
  `google/gemini-2.5-flash` (the two Sarvam-derived STT judges), and
  `anthropic/claude-sonnet-4.5` (semantic WER).
- **Resolution:** `resolve_llm_pricing(model, source)` in
  [`calibrate_agent/pricing.py`](calibrate_agent/pricing.py) — looks up one
  model's rate entry for the `"openrouter"` or `"direct"` source, returning
  `None` (not an error) when the model has no rates so an unpriced judge model
  doesn't block the estimate.
- **Estimate and confirmation:**
  [`calibrate_agent/judge_cost.py`](calibrate_agent/judge_cost.py) —
  `estimate_judge_cost_all_sources` prices the run's judge workload against
  both `openrouter` and `direct`, `format_cost_estimate` renders the result as
  text, and `confirm_judge_cost` prints that text and gates the run on a `y`
  answer.
- **Resumable grading:** [`calibrate_agent/judge_store.py`](calibrate_agent/judge_store.py)
  — a separate mechanism from cost estimation. `JudgeStore` writes every
  judge result to `judge_cache.jsonl` in the run's output directory the
  moment it's computed, so an interrupted or partly failed judge run resumes
  instead of re-grading (and re-paying for) rows already graded. The cache
  key fingerprints the judge input together with the evaluator's system
  prompt and judge model, so editing a prompt, changing the model, or
  re-transcribing a row invalidates that row's cached grade rather than
  returning it stale. `--overwrite` discards the cache along with
  `results.csv`.

**The token heuristic.** There is no tokenizer dependency, so token counts are
approximated from character counts, weighted by writing system: Latin text at
about 4 characters per token, Indic scripts (Devanagari through Malayalam) and
Arabic at about 1.5 characters per token. A flat chars/4 rule holds for Latin
text but undercounts Indic text by 2-3x, and Calibrate's datasets are largely
Hindi, Kannada, and Telugu, so the split matters for the estimate to be
usable. Mixed-script text is counted character by character, landing between
the two pure rates. These are approximations of what a real tokenizer would
count, not a measurement.

**Audio is billed per token, not per minute.** The audio judge model
(`openai/gpt-audio`) charges $32.00 per million audio input tokens. Audio
duration is converted to tokens at roughly 10 audio tokens per second before
it's priced — a different unit, and a different conversion, from the
per-minute rates provider TTS/STT cost uses above.

**The thinking-token allowance.** Gemini 2.5 Flash (`reasoning_billed_as_output:
true`) bills hidden thinking tokens at its output rate; those tokens never
appear in the visible response, so an estimate built from the visible answer
alone would run low. The estimate compensates by multiplying that model's
output-token figure by 3 before pricing it — a deliberately generous
multiplier, since quoting too high is recoverable and quoting too low is not.

**OpenRouter vs. direct.** Every model is priced under both `openrouter` and
`direct` sources and both totals are shown. OpenRouter charges provider list
price on all four models currently priced, so the two totals normally match;
reporting both makes a divergence visible if one ever appears, and lets a
direct-API user read the column that applies to them.

## Caveats (surface these wherever cost is displayed)

- **Bundled, point-in-time rates.** Costs are estimated from a rate table
  captured ~July 2026, not live provider pricing — they drift as providers
  change prices. This applies to the `llm` rates behind the judge cost
  estimate exactly as it does to the `stt`/`tts` rates above.
- **Entry tier assumed.** We use the standard pay-as-you-go / entry tier;
  volume or committed-use discounts and free tiers are not modeled — for the
  judge models as much as for STT/TTS providers.
- **Provider billing quirks not modeled.** Per-request minimums, billing
  increments/rounding, and taxes (GST) are ignored, so estimates run low for
  many short requests. OpenRouter and provider billing minimums, rounding,
  and negotiated discounts are equally unmodeled in the judge cost estimate.
- **One variant per provider.** Each provider is priced at a single model/tier
  we selected — Cartesia entry (Pro) tier, ElevenLabs `eleven_multilingual_v2`,
  Deepgram nova-3 streaming, Smallest standard Lightning / standalone Pulse —
  not the provider's other variants or tiers.
- **Audio-billed TTS is approximated.** OpenAI, Gemini, and Google-Sindhi are
  billed by audio-output tokens; we estimate cost as
  `measured audio minutes × per-minute rate`, so the figure scales with the
  synthesized audio length and the same text can differ across providers.
- **INR converted via live mid-market FX.** For INR-billed providers, `cost_usd`
  uses a live mid-market USD→INR rate, which excludes the FX margin and GST a
  real payment would incur.
- **Judge cost is an estimate computed before the work happens, not a
  measurement of what was billed.** It is built from the character-based
  token heuristic above, not a real tokenizer, so the token counts — and
  therefore the total — are approximate; the provider's actual reported usage
  for the same run will differ.
- **The audio-tokens-per-second conversion is a fixed approximation.** The
  ~10-tokens/second figure used to price the audio judge is not derived from
  the specific audio being judged; actual billed audio tokens depend on the
  audio's real encoding and length.
