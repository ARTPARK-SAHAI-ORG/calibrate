# Noise Injection — Implementation TODO (living doc)

Tracks the full build of the voice-sim background-noise feature. Spec:
[`noise_injection_spec.md`](noise_injection_spec.md). Kept updated as we go.
Status keys: ⬜ not started · 🔵 in progress · ✅ done · ⏸️ deferred.

## Phase 0 — Setup
- ✅ Deps: added `soundfile` + `scipy` (numpy already present). **audiomentations DROPPED** —
  it pins `soxr<1.0` which conflicts with pipecat 1.0's `soxr>=1.0`; backgroundify/filters/
  reverb/resample done with **numpy + scipy + soundfile + soxr** instead.
- ✅ Package skeleton `calibrate_agent/agent/noise/__init__.py`; package-data glob for
  `agent/assets/noise/**` registered in pyproject.

## Phase 1 — Simulation noise generator (`noise/simulation_noise_generator.py`)  ✅
- ✅ Steady loop w/ crossfade; event scatter (dog×8, car_horn×8); crowd overlay by density.
- ✅ **Backgroundify** (HP150 + LP3.5k + synthetic-RIR reverb + per-voice variation).
- ✅ env+people merge (people ~−6 dB under env), RMS-normalize −20 dBFS, seamless wrap,
     16 kHz/mono/16-bit. 8 unit tests pass.
- 🔵 **Audition set** — generate + listen (Wave 3, after wiring).

## Phase 2 — Schema + resolver  ✅ (32 tests)
- ✅ Parse/validate `noise` (off/fixed/random/mixture; sweep rejected).
- ✅ Per-run draw seeded by `run_index = persona_index*len(scenarios)+scenario_index`.
- ✅ Scene recipes; clean_fraction; `loudness → volume` via LOUDNESS_VOLUME (simple map v1;
     per-voice RMS calibration deferred — §18 #6).

## Phase 3 — Assets  ✅ (8 tests)
- ✅ `NoiseAssets` + `prepare_assets` populated `agent/assets/noise/` (16 kHz mono):
     env 10 + dog×8 + car_horn×8, speakers 25/40/25. Package-data registered.
- ⏸️ OGG compression deferred (WAV for now; shrink before shipping).

## Phase 4 — Pipecat wiring (`agent/run_simulation.py`)  🔵 (Wave 2 running)
- 🔵 Thread `noise` (ResolvedNoise) → `run_simulation` → `_run_simulation_inner`.
- 🔵 Attach `SoundfileMixer` via `audio_out_mixer=`; build track to per-sim `noise_track.wav`.

## Phase 5 — Clean/noisy save  ✅ module (`noise/save.py`, 5 tests) · 🔵 wiring in Wave 2
- ✅ `mix_noise_over_wav` (continuous) + `write_clean_and_noisy` (clean_ copies + both convs).
- ✅ `prefix` kwarg on `combine_audio_files` (4 tests).

## Phase 6 — CLI + config  ⏸️ deferred
- ⏸️ v1 is **config-JSON driven** (`noise` block). CLI `--noise`/`--loudness` overrides deferred.

## Phase 7 — Tests (`tests/agent/`)  ✅ unit (57 pass) · 🔵 integration in Wave 3
- ✅ generator / schema / resolver / assets / save / prefix unit tests.

## Phase 8 — Docs + example  ✅
- ✅ `docs/noise-injection.mdx` + `examples/noise_simulation_config.json`.
- ⬜ Add `"noise-injection"` to `docs/docs.json` nav (one-line follow-up).

## Phase 9 — Empirical validation
- ⬜ Run a noisy sim; check agent turn-taking survives continuous noise (§18 VAD note).
- ⬜ If it collapses: wire `MixerEnableFrame` turn-gating fallback.

## Phase 10 — FINAL (deferred to the very end) ⏸️
- ⏸️ **CC0 re-sourcing** of environmental clips (ESC-50 is CC BY-NC → cannot ship).
      Swap for CC0/public-domain equivalents; keep Vaani (CC BY 4.0) with attribution.
      **Do this last, after everything else works**, per user direction.
- ⏸️ NOTICE/attribution (Vaani CC BY 4.0) + license audit before release.
