# Agent simulation examples

Example configs for `calibrate-agent simulations` — a simulated user (persona)
holds a conversation with your agent so you can score how the agent behaves.

Run any of them with:

```bash
uv run calibrate-agent simulations --type voice -c <config>.json -o ./out/<run-name>
# text simulations use --type text
```

Each run creates one conversation per **persona × scenario** and writes a
per-simulation folder with `transcript.json`, `conversation.wav`, evaluator
results, and metrics.

## Voice simulations

| File | What it demonstrates |
| --- | --- |
| `sample_voice.json` | Baseline voice sim — 2 personas × 2 scenarios, no noise. Sarvam STT / Cartesia TTS / Gemini LLM. |
| `sample_voice_agent_connection.json` | Driving an **external** agent over a WebSocket connection instead of the built-in bot. |

## Background-noise variants

These add a **`noise` block inside a persona** to simulate a caller in a noisy
place — the noise is mixed continuously under the caller's speech, so the
agent's STT hears `caller + background`. Noise is **off by default** (omit the
block). See `docs/noise-injection.mdx` for the full option reference.

| File | `noise` (per persona) | Simulates |
| --- | --- | --- |
| `sample_voice_noise_env.json` | `fixed` · `busy_street` · no people | Caller on a busy street (horns + engine + siren), no chatter. |
| `sample_voice_noise_crowd.json` | `fixed` · people `medium` | Caller in a cafe — background English chatter only. |
| `sample_voice_noise_mixed.json` | `fixed` · `railway_station` + `light` · `loud` | Caller at a station — train + footsteps + a few voices, loud. |
| `sample_voice_noise_hindi.json` | `fixed` · `busy_street` + `heavy` · `loud`, **Hindi** persona | Hindi caller from a crowded market — heavy **Hindi** crowd + street. Shows language-matched chatter. |
| `sample_voice_noise_random.json` | `random` · `clean_fraction 0.15` | Each run draws a random condition; ~15% stay clean as a control. |
| `sample_voice_noise_mixture.json` | `mixture` (weighted) | Runs split by weight across clean / street+light / heavy-crowd conditions. |

**Key ideas**

- **Per persona.** `noise` lives on the persona object (a caller's environment
  belongs to the caller); different personas can have different noise. A
  top-level `noise` is accepted as a default for all personas.
- **environment** (12 single sounds or 8 scenes like `busy_street`) and
  **people** (`none`/`single`/`light`/`medium`/`heavy`) are chosen
  independently and merged; **loudness** = `faint`/`moderate`/`loud`/`harsh`.
- **people** chatter is **language-matched** to the persona's `language`
  (english/hindi/kannada) and muffled to sound like distant background.
- **modes**: `off` / `fixed` (one condition) / `random` (varies per run) /
  `mixture` (weighted mix). `sweep` is not supported.
- Outputs when noise is on: `conversation.wav` is the **noisy** (agent-heard)
  audio; `clean_conversation.wav` and `clean_*.wav` are the clean reference.

## Text simulations

| File | What it demonstrates |
| --- | --- |
| `sample_text_internal_agent.json` | Text sim against the built-in agent. |
| `sample_text_agent_connection.json` | Text sim driving an external agent connection. |
| `sample_text_dataset.json` | Eval-only over a dataset of pre-existing transcripts. |
