# Background Noise Injection for Voice Simulations — Design Spec

**Status:** Design complete; implementation not started.
**Scope:** Voice-agent simulations only (not STT eval).
**Branch:** `claude/festive-rosalind-92d0d6` (rebased onto `main`, pipecat **1.0.0**).
**Package:** `calibrate_agent/` (post-#109 rename; was `calibrate/`). Code map re-anchored in §16.0.

---

## 1. Goal

Make the **simulated caller** in a voice simulation sound like a real person calling
from a real, noisy place — so we can measure how well a voice agent's STT / turn-taking
holds up under realistic background conditions. Noise is mixed into the simulated
user's outgoing audio, continuously, for the whole call; the agent's STT then hears
`caller speech + background`.

This is an **evaluation-realism / robustness** feature, not a production feature.

---

## 2. Motivation & references

Two external inputs shaped the design:

- **Coval** ([docs](https://docs.coval.ai/concepts/personas/overview#available-background-sounds)) —
  attaches a **looping background track** to a persona (21 built-in sounds: Office, Cafe,
  Crowd Talking, Street with Sirens, Heavy Rain, Newborn Baby Crying, …) plus custom
  upload, with a volume slider. Takeaway we adopted: **named, realistic ambience**
  beats abstract noise (white/pink), and it belongs to the *caller persona*.

- **Vistaar / "Kathbath-Hard"** (arXiv [2305.15386](https://arxiv.org/abs/2305.15386)) —
  builds a "hard" ASR benchmark by mixing **ESC-50** background clips in at a **randomly
  chosen loudness within a set range** (the paper uses a voice-to-background ratio of
  3–30 dB). Takeaway we adopted: **add the background at a controlled loudness relative to
  the speech, varied within a bounded range** — a simple, principled difficulty knob.

**Synthesis:** Coval gives the *what* (named, realistic ambience, attached to the caller);
Vistaar gives the *how* (mixing the background in at a controlled, varied loudness). We combine them, and localize to India
(the whole framework targets Indian voice agents — Kathbath/Svarah/Sarvam/Vaani).

---

## 3. Terminology

Terms used throughout this doc.

**The feature's own terms**

- **Environmental sounds** — non-speech background: rain, wind, engine, siren, dog,
  train, etc. (language-neutral).
- **Speaker sounds** — background *people talking*, in the caller's language
  (english / hindi / kannada), processed to sound distant (see *backgroundify*).
- **Density** — how many background speakers overlap: **single / light / medium / heavy**.
- **Loudness** — how loud the background is relative to the caller: **faint / moderate /
  loud / harsh** (louder = harder). Under the hood this is a voice-to-background ratio; see §12.
- **Situation / scene** — a realistic named place (e.g. `busy_street`), defined as a
  *recipe* of ingredients and built fresh for each call.
- **Background noise loop** — the single looping background track mixed under one call;
  it plays continuously for the whole call, under both speakers and during every pause.
- **Atom** — one complete condition for a single call: which environment, how many people,
  how loud.
- **Distribution** — how those conditions are handed out across a batch of calls (all the
  same, weighted, or randomly varied).

**Audio terms**

- **Backgroundify** — the processing that takes crisp close-up speech and makes it sound
  like distant, muffled room chatter.
- **Low-pass filter** — removes the high frequencies, leaving a muffled sound (like hearing
  a party through a wall).
- **High-pass filter** — removes the low frequencies, cutting the boomy close-up rumble of
  a voice recorded right on the microphone.
- **Reverb / impulse response (RIR)** — reverb is the echoey "in a room" quality of sound;
  an RIR is a fingerprint of a room's echo stamped onto a dry recording to place it in that
  room.
- **Crossfade** — fading one clip out while fading the next in, so the join has no audible
  click.
- **Seamless loop** — a clip whose end blends into its start so it can repeat forever with
  no restart click.
- **Scatter (event sounds)** — placing one-off sounds like a dog bark or car horn at random
  times and volumes, so a repeated clip doesn't sound like a metronome.
- **Mud / muddy** — what you get when too many equally-loud sounds pile up and smear into a
  formless wash (like mixing every paint colour into brown).
- **RMS-normalize** — set every track to the same average loudness so none is accidentally
  louder than another.
- **Resampling** — converting audio from one sample rate to another (e.g. 44.1 kHz → 16 kHz)
  to match the rest of the pipeline.
- **Mono** — single-channel audio, as opposed to stereo.
- **Mixer (`SoundfileMixer`)** — pipecat's component that plays a sound file on loop and
  blends it into the outgoing audio; what actually adds the noise during the live call.
- **VAD (voice activity detection)** — the agent's "is someone talking right now?" detector;
  decides when the caller starts and stops speaking.
- **Seed** — a starting number that anchors every "random" choice, so the same seed always
  reproduces the exact same result.
- **SNR / dBFS** — the underlying measures behind *loudness*: SNR is the voice-to-background
  ratio (higher = quieter background); dBFS measures level on a scale where 0 is the digital
  maximum and lower (more negative) is quieter.

---

## 4. Scope & non-goals

**In scope**
- Voice-agent simulation path only (`calibrate_agent/agent/run_simulation.py`, the simulated
  user's pipecat client transport).
- Languages: **english, hindi, kannada** (the three the sim supports — see §14).
- Environmental sounds + speaker sounds + their combination.
- Save clean + noisy audio; vary noise across a batch; reproducible via seed.

**Out of scope (deferred / separate)**
- STT-eval noise injection (offline dataset). Explicitly dropped early — this is agent-sim only.
- **Backchanneling** (caller says "mm-hmm"/"yeah") — that's a *user-simulator behavior*,
  not a background track. Separate feature.
- **Coval sounds not in ESC-50**: airport/ferry PA announcements, kids playing,
  construction, doorbell, skatepark. Would need CC0 sourcing (announcements are
  speech-like → could be Vaani-derived). Not in v1.
- Composite/pre-mixed "scenes as files" — scenes are generated live instead (§8).

---

## 5. Data assets

### 5.1 Environmental sounds (ESC-50)
Pulled one representative clip per chosen class from ESC-50 (44.1 kHz mono 16-bit, 5 s),
into `data/env_raw/`. **12 classes** (started 13, dropped `clock_alarm` — see §6.3):

`rain · wind · engine · vacuum_cleaner · train · siren · crying_baby · footsteps ·
keyboard_typing · laughing · dog · car_horn`

- `dog` and `car_horn` are **single-event** clips (~95% silence + one bark/honk), so we
  pulled **8 varied samples each** into `data/env_raw/dog/00..07.wav` and
  `data/env_raw/car_horn/00..07.wav` — needed for natural scattering (§6.2).

### 5.2 Speaker sounds (Vaani)
Pulled from **ARTPARK-IISc/Vaani** (HF, **CC BY 4.0**, gated — token required), into
`data/vaani_raw/{english,hindi,kannada}/`. Vaani is organized by `{State}_{District}`;
each row carries a `language` and `gender` field; audio is **mono 16 kHz 16-bit** (no
resampling needed). Spontaneous (image-prompted) speech → sounds like real chatter, not
read speech.

| Language | Clips | Female | Male | Source |
|---|---|---|---|---|
| english | 25 | 14 | 11 | Karnataka_Bangalore |
| hindi   | 40 | 25 | 15 | Bangalore (F) + Delhi_NewDelhi (M) |
| kannada | 25 | 20 | 5  | Karnataka_Bangalore |

- **Hindi was initially all-female** (Bangalore's first ~414 rows had no male Hindi);
  fixed by pulling 15 male Hindi from Delhi (`Delhi_NewDelhi_M_*.wav`).
- **Kannada is light on male voices (5)** — fine for single/light/medium; a "heavy"
  Kannada crowd would lean female. Deferred top-up (Mysore/Dharwad) if needed.
- `data/vaani_raw/manifest.json` records per-clip source district, gender, duration.

### 5.3 Licensing (critical)
- **ESC-50 = CC BY-NC** (non-commercial). Calibrate is OSS on PyPI → **cannot bundle
  ESC-50 clips as-is.** The pulled clips are **auditioning intermediates only.** For
  shipping, the environmental loops must be re-sourced from **CC0** equivalents (ESC-50's
  individually-CC0 Freesound origins, or CC0 Freesound loops matching the same classes).
- **Vaani = CC BY 4.0** — bundleable with attribution (NOTICE + docs).
- Raw pulls (`data/`) are **not committed** (intermediate, large, licensing).

---

## 6. Sound behavior (verified by waveform analysis, not filenames)

We measured each env clip (silence %, number of bursts, envelope modulation, spectral
flatness) rather than assuming from the name. Findings drove three handling classes:

### 6.1 Steady loopers (loop as-is)
`rain, wind, engine, vacuum_cleaner, train` — ~0% silence, low envelope modulation →
continuous. A short crossfaded loop already sounds continuous.

### 6.2 Intermittent / event sounds
- **Naturally spaced events**: `dog` (96% silence, 1 bark), `car_horn` (94% silence,
  1 honk), `footsteps` (97% silence, few steps). Looping one raw clip = a robotic
  metronome (exact same bark every 5 s). Handling: **scatter multiple different samples
  at random spacing + volume** over a quiet background noise loop → sounds like a real dog / real traffic.
  (This is why we pulled 8 dog + 8 car_horn samples.)
- **Multi-burst but frequent**: `crying_baby` (40% silence, cries in waves),
  `keyboard_typing` (77% silence, 28 taps), `laughing` (85% silence, bursts),
  `siren` (40% silence, oscillating wail). Fine with light spacing.

### 6.3 Dropped: `clock_alarm`
Measured 0% silence, dead steady → it **rings continuously** the whole clip. There's no
natural gap to make it "occasional"; looping it = a constant alarm through the entire
call (unrealistic + grating). **Dropped.** (User confirmed.)

### 6.4 How looping works (Stage-1 mechanics)
- **Seamless seam**: crossfade the clip's tail onto its head (fade tail 1→0, head 0→1,
  add) so the wrap-around has no click.
- **Length**: stitch several same-class clips with crossfades to ~30–60 s so it's not
  audibly repetitive; then crossfade the whole thing's end into its start.
- **Event scatter**: place different event samples at random offsets/gaps/volumes over a
  quiet background noise loop (not a fixed loop).
- pipecat's mixer does the actual endless repeating during the call; Stage 1 just makes
  the track loop-friendly.

*In plain terms: we build one ~30–60 second background clip. There are two kinds of joins
to hide. First, the internal joins — where we glue several short clips end-to-end to reach
that length — are smoothed with crossfades (one clip fading out as the next fades in) so
you never hear a hard cut. Second, the loop seam — where the end of the whole clip wraps
back to its beginning — is also crossfaded so it can repeat forever without an audible
click. Stage 1 only makes that one seamless clip; the mixer is what actually plays it on
repeat for the entire call.*

---

## 7. Speaker sounds (crowds)

### 7.1 Why background speakers are the hardest noise
Rain/engine sit in different frequency bands than speech, so STT can still "see" the
words underneath. **Background speakers are speech masking speech** — same frequency band, same
syllable-rate modulation — so the recognizer mistakes background words for foreground and
inserts/swaps them. **Indian-language** background speakers matter specifically: an Indian-tuned STT
exposed to background Hindi will hallucinate Hindi words — the exact production failure we
want to test; Western background speakers under-test it. (This is why Vaani, not MUSAN/Common Voice
English, is the source.)

*In plain terms: a rumbling engine and a human voice live in different "pitch lanes," so
the transcriber can still pick out the words over the engine. But other people talking sit
in the exact same lane as the caller — same pitches, same rhythm of syllables — so the
transcriber can't tell foreground from background and starts typing the wrong words. And
if that background chatter is in the same Indian language, an India-tuned transcriber is
especially likely to "hear" real words in it.*

### 7.2 Density levels
| Level | Voices | Feels like |
|---|---|---|
| single | 1 | one person nearby on another call / a family member |
| light | 2–3 | a couple people, quiet office |
| medium | 4–6 | a normal cafe |
| heavy | 8–12 | packed cafe / market / crowded station |

`single` is a distinct **hard** case (one intelligible voice → STT grabs its real words),
not just "less crowd."

### 7.3 Generation model — **B: generate live per simulation** (LOCKED)
For each call we **assemble a fresh crowd**: randomly pick N distinct speakers
(N = density) from that language's pool (25–40 clips), overlay at random offsets +
volume jitter, mixing genders where available. Not a pick-one-of-a-few-prebuilt-files;
each simulation's crowd is unique. Seeded → reproducible (run *i* → same crowd).
Consequence: we **bundle the raw ingredient clips**, and **generate the background noise loop on the fly**
at each sim start — we do NOT ship pre-rendered crowd tracks.

### 7.4 "Backgroundify" the speaker sounds (REQUIRED — not just volume)
The Vaani clips are **close-mic foreground speech** — a person talking *into* a microphone,
crisp and intelligible. Lowering the volume alone leaves it sounding like quiet
foreground speech, not background chatter. So every speaker background noise loop must pass a
**backgroundify chain** to make it read as *distant room chatter*:

- **Low-pass** (~3–4 kHz cutoff) — distance/air/obstacles kill the highs → muffled.
- **High-pass** (~150–200 Hz) — remove close-mic proximity boom/rumble.
- **Reverb / room simulation** — dry close-mic → wet distant; puts the voices *in a room*,
  not on the mic. Done with **scipy**: a synthetic impulse response is convolved onto the
  voices via `scipy.signal.fftconvolve` (a lightweight, dependency-free reverb). We do
  **not** use `audiomentations` (its `RoomSimulator`/`ApplyImpulseResponse` conflict with
  pipecat's `soxr` pin), so no `pyroomacoustics` dependency is pulled in.
- **Per-voice variation** — slightly different cutoff / level / (optional) pan per speaker
  so the crowd has depth instead of all voices at one "distance."
- **Level** — plus the `loudness`→volume placement (§12).

Effect: an **unintelligible distant murmur**, which is (a) what real background chatter
sounds like, and (b) far less likely to false-trigger the agent's speech-onset VAD than
clear foreground words (§18 VAD note). This chain lives in the Stage-1 background noise loop builder
(`simulation_noise_generator.py`) and applies to the **speaker** ingredients; environmental sounds are
already ambient and don't need it (a light shared room reverb to bind everything into one
space is optional).

*In plain terms: the crowd voices were recorded right up against a microphone, so they
sound like someone standing next to you — too clear to pass as background. Each filter
fakes the effect of distance: cutting the high frequencies makes them sound muffled (like
hearing a party through a wall), cutting the very low frequencies removes the boomy
close-up "chest" tone, and the reverb makes them sound like they're across a room instead
of on the phone. The result is an unintelligible murmur — which is both more realistic and
less likely to fool the agent into thinking the caller has started talking.*

---

## 8. Situations / scenes

Three buckets, all generated live:

1. **Environment-only** (no people): rain, vehicle, busy_street, housework, quiet_home…
2. **People-only** (density): single / light / medium / heavy.
3. **Mixed** (environment + people).

**Environment options (menu):**
- **Single sounds (12):** rain, wind, engine, vacuum, train, siren, crying_baby,
  footsteps, keyboard_typing, laughing, dog, car_horn.
- **Scenes (8)** — named *recipes* of ingredients (+ a default density):
  `busy_street`=car_horn+engine+siren · `vehicle`=engine+wind+horn ·
  `railway_station`=train+footsteps · `office`=keyboard_typing ·
  `home_with_baby`=crying_baby · `housework`=vacuum · `rainy_street`=rain+wind+horn ·
  `quiet_home`=dog.

**Scenes are live-generated from a fixed recipe**, not pre-made files. Every `busy_street`
call is a different busy street (different horns scattered, different crowd).

---

## 9. Combining environment + people

Selected **independently** and merged for the call:
- both set → mixed into **one** background noise loop, people kept **quieter under** the environment (so it
  doesn't turn to mud);
- only environment → environment-only; only people → people-only.
- One background noise loop per call — a caller is in one place; the background noise loop is **constant for the whole call**;
  variety happens *across* the batch (§11).
- pipecat's mixer plays one file, so "combine" = mix the chosen tracks into one background noise loop at
  sim start.
- Because we combine on the fly, we do **not** pre-bake mixed situations — env pool +
  people pool covers every combination.

*In plain terms: when a scene has both weather/traffic and people, we tuck the people
underneath the environmental sound rather than making them equally loud. If everything is
blaring at the same volume, the layers stop being distinct and smear into a single formless
wash ("mud") that would also drown out the actual caller. Keeping the chatter quieter under
the environment keeps the caller's voice on top and the scene sounding real.*

---

## 10. Injection architecture (4 stages)

```
BUNDLED INGREDIENTS            PER-SIMULATION (live)
env clips (CC0, compressed)    ┌─ resolve condition (env + people + loudness), seeded by run index
vaani speaker pool (3 langs) ──┤─ STAGE 1: build ONE background noise loop (loop steady / scatter events / overlay crowd, crossfade)
                               ├─ STAGE 2: attach mixer to sim-user transport → agent STT hears speech+background noise loop
                               └─ STAGE 3: save clean (clean_*) + noisy (unprefixed) + both conversation.wav
```

- **Stage 1 — build background noise loop (live):** from bundled ingredients per the condition; seamless
  loop; ~30–60 s.
- **Stage 2 — inject (the actual mixing):** attach a continuous `SoundfileMixer` to the
  simulated user's client transport output. Add `audio_out_mixer=SoundfileMixer(
  sound_files={"bg": path}, default_sound="bg", volume=<from loudness>, loop=True,
  mixing=True)` to the `WebsocketClientParams(...)` block at
  [`run_simulation.py:1481`](calibrate_agent/agent/run_simulation.py:1481). Import
  `from pipecat.audio.mixers.soundfile_mixer import SoundfileMixer`. The mixer loops the
  background noise loop and mixes it into every outgoing audio frame for the whole call. Runtime control
  (if we ever need to change volume/sound mid-call) via `MixerUpdateSettingsFrame({...})`
  / `MixerEnableFrame(enable)`.
  **HARD CONSTRAINT (verified in pipecat 1.0 source):** the mixer does **not** auto-resample
  and does **not** auto-convert to mono. The background noise loop file must be **mono, 16-bit PCM, 16000 Hz**
  (matching `audio_out_sample_rate=16000` at
  [`run_simulation.py:1644`](calibrate_agent/agent/run_simulation.py:1644)) — a sample-rate
  mismatch is **silently dropped** (no noise at all), and a stereo file **corrupts** the
  mix. So Stage 1 must emit exactly 16 kHz / mono / 16-bit. (Vaani clips are already
  16 kHz mono; ESC-50 env clips are 44.1 kHz → **must be resampled to 16 kHz** (converted
  to the lower sample rate) during background noise loop
  build.)
- **Stage 3 — save clean + noisy (CORRECTED — see §18 defect #1):** per-turn chunks are
  captured **upstream** of the mixer → they're clean. Copy them to `clean_`-prefixed
  names + build `clean_conversation.wav` from the clean turns. Because the live mixer is
  **continuous** (§18 #1: it self-generates background noise loop frames during silence and bot turns, not
  just user turns), the noisy reconstruction must mix the background noise loop **continuously
  across the whole call timeline** — NOT per-user-turn. Concretely: take the stitched
  clean conversation timeline and mix the background noise loop over its **entire** length (from t=0,
  same volume the live mixer used) → `conversation.wav`. So the *default* filenames hold
  the realistic/agent-heard audio; `clean_` files are the clean reference, present **only
  when noise is on**. (The earlier "mix into user turns only" plan was wrong — it would
  have left bot turns and inter-turn gaps silent, which the agent did NOT experience.)

*In plain terms: during the live call the background noise never stops — it plays under the
caller, under the agent, and during every pause, for the whole call. But the caller-voice
snippets we save to disk are captured before the noise is added, so they're clean. To
recreate what the agent actually heard, we take the full clean conversation (including the
agent's turns and the silent gaps) and lay the background noise over the entire timeline
from start to finish — not just over the moments the caller was speaking. That's why the
noisy file is a faithful re-creation rather than a copy.*
- **Stage 4 — config/CLI:** default off → existing runs unchanged.

**Honesty caveat:** the saved noisy audio is a faithful *reconstruction* — same background noise loop, same
level, mixed continuously over the same timeline — not a byte-identical tap of the
transmitted stream (the background noise loop's loop *phase* relative to the live call, and int16
re-quantization, differ microscopically). Byte-identical would require tapping the
transport internals.

---

## 11. Config schema

Lives **inside each persona** in the simulation config JSON (`-c`), under a
`noise` key on the persona object (a caller's environment belongs to the caller).
A top-level `noise` is also accepted as a default for all personas. Read at
`_run_single_simulation_inner` via `user_persona.get("noise", config.get("noise"))`.
Omit → clean.
CLI can override per-run.

### 11.1 Atom (one condition)
```jsonc
{
  "environment": <single | scene | [ingredients] | "none">,
  "people":      "none" | "single" | "light" | "medium" | "heavy",
  "loudness":    "faint" | "moderate" | "loud" | "harsh" | [list] | "any",
  "seed":        <int>
}
```
- `environment` three forms: a single sound, a named scene, or an explicit **merge list**
  `["rain","car_horn","engine"]`.
- `people` auto-matched to the **sim's language** (never set language here).
- `loudness` = user-facing SNR (§12). **A list is a menu sampled from** (`["faint","loud"]`
  → faint OR loud, no moderate). `"any"` = all four. Default: `moderate` in `fixed`;
  `any` in variety modes.

### 11.2 Distribution (assignment across *that persona's* runs)
Because `noise` lives on a **persona**, `mode` is resolved **per simulation** — once for
each `(persona × scenario)` pair, using that persona's noise block. So a persona's
distribution spans **that persona's own runs (one per scenario)**, not the whole config;
each persona has its own independent distribution. The seed is
`run_index = persona_index × (#scenarios) + scenario_index`.

`noise.mode` ∈ `off | fixed | random | mixture`:
- **off** — clean for all of this persona's runs.
- **fixed** — the same atom for every one of this persona's runs (controlled A/B).
- **random** — each of this persona's runs independently samples environment + people +
  loudness; `clean_fraction` makes some of them clean; seeded by `run_index`.
- **mixture** — each run weighted-picks one condition from the `{weight, spec}` list.

*In plain terms: an "atom" is one complete recipe for a single call's background — which
environment, how many people, how loud. A "distribution" is how a **persona** hands those
recipes out across its own runs (it gets one run per scenario). `off` = no noise on this
caller; `fixed` = the exact same recipe on every one of this caller's runs (clean A/B);
`random` = each of this caller's runs rolls its own recipe; `mixture` = proportions, e.g.
"20% quiet, 30% busy street, …". Consequence: `random`/`mixture` only show variety when a
persona runs across **several scenarios** — a persona with one scenario just gets one
seeded draw. Want each condition as its own labelled case → many personas on `fixed` (see
`examples/agent/simulation/sample_voice_noise_matrix.json`); want one caller hit with a
spread → one persona on `random`/`mixture` with several scenarios.*

**`sweep` was dropped for v1** (§18 #8): the batch size is fixed at `|personas| ×
|scenarios|`, so a `repeat: K` run-count knob doesn't fit. `random` gives variety within
the fixed run count; revisit `sweep` later if guaranteed even coverage is needed.

```jsonc
{ "noise": { "mode": "random", "clean_fraction": 0.15, "seed": 123 } }
{ "noise": { "mode": "mixture", "conditions": [
    { "weight": 0.2, "spec": "off" },
    { "weight": 0.3, "spec": { "environment": "busy_street", "people": "light" } },
    { "weight": 0.3, "spec": { "people": "medium" } },
    { "weight": 0.2, "spec": { "environment": "rain" } } ] } }
```

### 11.3 Resolution rules
- Per simulation: draw one atom → build one background noise loop → inject → **constant for that call**.
- Scene density precedence: **recipe default → explicit `people` override → sampled** (in
  variety modes). Finalized per-simulation.
- Seed: run *i* always gets the same draw *and* the same live-assembled crowd.

### 11.4 Built-in variations (condition catalog)
Additive flat menu (scenes already include people; do **not** multiply):
**12 environment-only + 4 people densities + 8 scenes + 1 off = ~25 named condition
types.** (Multiplicatively, environment 21 × people 5 ≈ 105, ×4 loudness ≈ 420 if each
loudness counts — but the built-in *menu* is the ~25 flat list.) The actual audio inside
any people-condition is unbounded (live crowd generation).

---

## 12. Loudness vs SNR

- **SNR** (signal-to-noise ratio, dB) = caller-voice loudness vs background. **Higher SNR
  = quieter background = easier** — inverted and jargony, so not exposed.
- User-facing name: **`loudness`**, direction-intuitive (louder = harder), named levels
  mapping to SNR internally: `faint ≈ 22 dB · moderate ≈ 16 · loud ≈ 11 · harsh ≈ 6`.
- `random` mode samples a loudness per run (Vistaar-style bounded randomness).
- Power-user escape hatch: `"loudness_db": [min, max]` (exact dB) — optional, not the
  default surface.
- Intensity default deferred by user ("configurable later"); mechanism must expose it.
- Mechanically, `loudness` → a mixer volume calibrated from the (normalized) TTS level.

*In plain terms: engineers measure difficulty as SNR — how much louder the voice is than
the background. The catch is that SNR runs backwards for a normal reader: a bigger SNR
number means an easier, quieter-background call. So we expose a plain "loudness" knob
instead, where turning it up makes the background louder and the call harder — the
intuitive direction. Under the hood we still convert those loudness levels back to SNR
numbers; we just flipped the label so "more" means "harder."*

---

## 13. Reproducibility
A single `seed` makes the whole batch deterministic: per-run condition draw + live crowd
assembly are seeded by (base seed + run index). Re-running reproduces identical audio for
debugging.

*In plain terms: even though the noise and crowds are generated randomly, the randomness is
anchored to a single starting number (the "seed"). Give it the same seed and every call
comes out byte-for-byte identical to last time — the same environment, the same crowd, the
same volumes. That means a bug you saw in run #3 will reappear exactly when you re-run it,
instead of vanishing into fresh random noise.*

---

## 14. Prerequisite done — Kannada language support

Before noise could language-match all three languages, the sim was clamped to
english/hindi (kannada half-wired, and a latent bug: kannada fell back to an English
ElevenLabs voice). Fixed (now in `main`, adapted to pipecat 1.0):
- Widened `Literal["english","hindi"]` → `+ "kannada"` in `start_bot`,
  `run_simulation`, `_run_simulation_inner`, and `_Simulation.run_single`.
- Kannada simulated user → **Google Chirp3-HD** (ElevenLabs has no real Kannada voice),
  gender-aware: **`kn-IN-Chirp3-HD-Achernar` (female) / `kn-IN-Chirp3-HD-Achird` (male)**.
- Note: Google TTS isn't word-level, so Kannada mid-utterance interrupts commit slightly
  coarser than the ElevenLabs english/hindi path.

---

## 15. Key decisions & tradeoffs (log)

| # | Decision | Why / tradeoff |
|---|---|---|
| 1 | Agent-sim only (not STT eval) | Where realism matters for turn-taking; STT-eval deferred |
| 2 | Continuous background noise loop (Coval-style), not per-utterance exact-SNR (Vistaar) | Realistic + robust + cheap; SNR is approximate/average not exact-per-turn |
| 3 | Save both: unprefixed=noisy, `clean_`=clean; both conversation files | Nothing downstream breaks; debuggable; `clean_` only when noise on |
| 4 | Vaani for speaker sounds (not MUSAN/Common Voice) | Indian, spontaneous, dialect-rich, CC BY, dogfoods ARTPARK's own data; triggers the Hindi-hallucination failure mode |
| 5 | **B: generate crowds/background noise loops live per sim** | Infinite variety across a batch; bundle ingredients not tracks; seeded = still reproducible |
| 6 | Drop `clock_alarm` | Waveform shows continuous ring → can't be "occasional" |
| 7 | Pull 8 samples each for dog/car_horn | Single-event clips; scattering needs variety or it's a metronome |
| 8 | `loudness` name, not `snr` | SNR is jargon and inverted |
| 9 | Ship compressed (OGG) | ~12 MB vs ~115 MB WAV — bundleable |
| 10 | Re-source env background noise loops CC0 for shipping | ESC-50 is CC BY-NC; can't ship |
| 11 | Combine env+people on the fly, one background noise loop/call | Any combo without pre-baking; realistic single-environment call |
| 12 | Scenes = live recipes, not files | Consistency with B; every instance differs |

---

## 16. Implementation plan

*(Grounded against the current pipecat-1.0 codebase.)*

### 16.0 Grounded code map (exact edit points)

> **Post-#109 rename:** the package moved `calibrate/` → **`calibrate_agent/`** (imports
> are now `from calibrate_agent.…`). Line numbers below are **re-anchored to the current
> tree** (they shifted ~150–200 lines vs the first grounding pass, from the rename + the
> intervening voice-sim commits). The old `calibrate/` dir is now just stale `__pycache__`.

**Injection (Stage 2)** — `calibrate_agent/agent/run_simulation.py`
- Sim-user transport at `1479-1489`: `WebsocketClientTransport` at `1479`,
  `WebsocketClientParams(` at `1481`; add `audio_out_mixer=` **inside** that block
  (~`1482-1488`, *not* on the opening line). Commented-out sim-user VAD at `1486`.
- `WebsocketClientParams` inherits `audio_out_mixer: Optional[BaseAudioMixer | Mapping]`
  from `TransportParams` — accepts the mixer directly. The base output transport runs it
  via the continuous `with_mixer` loop
  (`.venv/.../pipecat/transports/base_output.py:757-784` — see §18 #1).
- `SoundfileMixer` = `pipecat.audio.mixers.soundfile_mixer`; ctor keyword-only:
  `SoundfileMixer(*, sound_files, default_sound, volume=0.4, mixing=True, loop=True)`.
- **Requires `soundfile` — NOT installed today.**

**Config threading (Stage 4)** — `calibrate_agent/agent/run_simulation.py`
- Config `json.load` in `main()` at `2298`; persona/scenario enumerated at `2322-2323`;
  whole `config` passed to `run_single_simulation_task(...)` at `2324` (Semaphore at `2318`).
- **Read `noise` + resolve/build the background noise loop** in `_run_single_simulation_inner` (def `1999`)
  at ~`2087` (beside the `config.get("stt"/"tts"/"llm"/"settings")` reads). This site has
  `config`, `persona_index`, `scenario_index` in scope — the seed source (§18 #3).
- **Thread the temp background noise loop path + volume down** (not raw config): `run_simulation`
  (def `1354`; delegation to inner at `1410`) → `_run_simulation_inner` (def `1432`;
  consume at transport `1481`).
- CLI: `simulations` subparser at `cli.py:373`; `main()` at `cli.py:152`.

**Audio save (Stage 3)** — `calibrate_agent/agent/run_simulation.py` + `calibrate_agent/utils.py`
- Chunks via `save_audio_chunk` (`utils.py:283`) → `<out>/audios`, named
  `{turn}_{role}_{chunk}.wav`. Sim-user `_user` chunks written in `SilencePadder`
  (class `1142`) at `run_simulation.py:1201`, tapped **before** `transport.output()` →
  **saved audio is clean**; no live noisy capture → reconstruct offline.
- Stitch helpers: `combine_turn_audio_chunks` (`utils.py:447`), `combine_audio_files`
  (`utils.py:564`, transcript-ordered, hardcodes `{N}_bot.wav`/`{N}_user.wav` at `611`/`618`).
- End-of-run stitch at `run_simulation.py:2229-2233`. **Hook here (continuous model)**:
  after `combine_turn_audio_chunks` (`2229`) → (a) copy per-turn files to
  `clean_{N}_{role}.wav`; (b) stitch clean set → `clean_conversation.wav` (needs a
  **`prefix` kwarg on `combine_audio_files`**, default `""` so the existing `2233` call is
  unchanged); (c) noisy `conversation.wav` = background noise loop mixed **continuously over the whole clean
  timeline** at the live volume (§18 #1).
- All injected audio must be 16 kHz / mono / 16-bit (matches `save_audio_chunk` width=2;
  `combine_audio_files` same-format requirement).

**Deps / packaging** — `pyproject.toml`
- `numpy>=1.26.0` already present. **Add** `soundfile` and `audiomentations`
  (audiomentations transitively pulls numpy+soundfile).
- Bundle assets via `[tool.setuptools.package-data]."calibrate_agent"` at `71` (same
  mechanism as `ui/cli.bundle.mjs`): add `"agent/assets/**/*.ogg"`. Load at runtime via
  `importlib.resources.files("calibrate_agent.agent")/"assets"/...`.
- Package name `calibrate-agent` (`2`), entry `calibrate_agent.cli:main` (`64`).
- `.gitignore` bare `data` → `data/vaani_raw`, `data/env_raw` already ignored.

**Tests** — `tests/agent/` (unittest + `AsyncMock`/`MagicMock`).

### 16.1 Components
- **`calibrate_agent/agent/noise/`** (new package)
  - `assets.py` — resolve ingredient paths; the CC0-sourced env background noise loops + Vaani speaker pool.
  - `background noise loops.py` — Stage-1 background noise loop builder: steady loop (crossfade), event scatter, crowd
    overlay, **backgroundify the speaker crowd (low-pass + high-pass + reverb + per-voice
    variation — §7.4)**, env+people merge, seamless-loop wrap; RMS-normalize (set every
    background noise loop to the same average loudness) to a
    fixed reference (§18 #6); emits a **16 kHz/mono/16-bit temp WAV**. numpy + soundfile
    (+ audiomentations for the filters/reverb).
  - `schema.py` — parse/validate the `noise` config; the atom + distribution model.
  - `resolver.py` — per-run condition draw (fixed/random/mixture), seeded by
    `persona_index`/`scenario_index`; scene recipe expansion; density precedence;
    `loudness → volume` calibration (needs a per-voice TTS reference RMS — §18 #6).
  - `background noise loops.py`/wiring — **temp-background noise loop lifecycle**: write the background noise loop under the per-sim
    `simulation_output_dir` (unique — §18 #2/#4), delete in a `finally` that runs on the
    cancel path (`run_simulation.py:1403-1404`).
  - `save.py` — clean/noisy reconstruction: **continuous** offline mix over the full
    timeline (§18 #1), plus the `combine_audio_files(prefix=...)` helper change.
- **`calibrate_agent/agent/run_simulation.py`** — resolve/build background noise loop at ~`2087`; thread the
  temp path + volume → `_run_simulation_inner`; attach mixer to the sim-user transport
  (Stage 2); dual clean/noisy save (Stage 3).
- **`calibrate_agent/cli.py`** — read `noise` from the sim config; optional `--noise`/`--loudness` overrides.
- **`calibrate_agent/utils.py`** — add `prefix` kwarg (default `""`) to `combine_audio_files`.
- **`pyproject.toml`** — add `soundfile` + `audiomentations`; register
  `agent/assets/**/*.ogg` under the `"calibrate_agent"` package-data.
- **`docs/` + example config**; **`tests/agent/`**.
- **Dev-only** asset-prep scripts (Vaani/CC0 pull + OGG encode; not shipped).

### 16.2 Parallelizable workstreams
1. `noise/` package + unit tests (independent of the sim wiring).
2. `run_simulation.py` wiring (mixer attach + dual save) — depends on `noise/` API shape.
3. `cli.py` + config parsing.
4. deps/packaging + docs + example + CC0 asset re-sourcing.
(1) and (4) fully independent; (2)/(3) touch existing files, region/file split.

### 16.3 Testing
Pure unit tests (mirroring repo convention): background noise loop builder (seamless-loop RMS continuity,
event scatter count, crowd overlay N-speakers, env+people merge levels), schema parse/
validate, resolver (seeded draw determinism, scene-density precedence, loudness→volume),
clean/noisy save layout, off=unchanged. pipecat mocked (AsyncMock/MagicMock).

### 16.4 Open items before/at build
- **CC0 re-sourcing** of env background noise loops (blocking for shipping, not for a local prototype).
- **Add `soundfile`** — the mixer's `soundfile` backend is not installed today; decide
  `soundfile` standalone dep vs `pipecat-ai[soundfile]` extra (audiomentations pulls it
  transitively regardless).
- **Background noise loops must be exactly 16 kHz / mono / 16-bit** or the mixer silently drops/corrupts
  them — resample ESC-50 (44.1 kHz) during background noise loop build; Vaani is already 16 kHz mono.
- **`combine_audio_files` needs a `prefix` kwarg** (`utils.py:564`, names at `611`/`618`) for the clean stitch.
- Intensity/loudness default value (deferred by user).
- Kannada male-voice top-up (optional; heavy Kannada crowd leans female).
- Branch is on pipecat 1.0.0 — target that API.

---

## 17. Assets & paths summary

| Path | Contents | Committed? |
|---|---|---|
| `data/env_raw/*.wav`, `dog/`, `car_horn/` | ESC-50 auditioning clips (CC BY-NC) | No |
| `data/vaani_raw/{lang}/*.wav` + `manifest.json` | Vaani speaker pool (CC BY 4.0) | No |
| `calibrate_agent/agent/assets/env/*.ogg` | CC0-sourced env background noise loops (shipped, 16 kHz mono) | Yes (future) |
| `calibrate_agent/agent/assets/speakers/{lang}/*.ogg` | Vaani speaker pool, compressed (shipped, 16 kHz mono) | Yes (future) |
| `design/noise_injection_spec.md` | this doc | Yes |

---

## 18. Verification pass — defects found & corrections

Two adversarial agents verified this plan against the **installed pipecat 1.0.0 source**
and the current code. **All §16.0 line-refs held up** (only nit: the mixer kwarg goes
*inside* the `WebsocketClientParams(...)` block, not on its opening line). *(Line numbers
here re-anchored post-#109 rename — see §16.0.)* The defects below were in the plan's
*logic/completeness*, now corrected here.

**#1 — Fidelity contradiction (was the biggest bug).** Verified in pipecat source
(`base_output.py:757-784`, the `with_mixer` generator): once a mixer is attached, the
output transport runs a **self-clocked continuous loop** — on `QueueEmpty` it manufactures
`OutputAudioRawFrame(audio=mixer.mix(silence))`. So injected noise is **continuous for the
whole call** (bot turns + silence included), added on top of speech
(`clip(audio + sound*volume)`, `soundfile_mixer.py:193`). The plan's "reconstruct noisy
audio on user turns only" would have omitted bot-turn/inter-turn ambience.
**Fix:** Stage 3 now mixes the background noise loop **continuously over the full timeline** (§10, §16.0).

**#2 — Temp-background noise loop file lifecycle.** `SoundfileMixer` reads a **file path**
(`sound_files={name: path}`) at `mixer.start(sr)`, but background noise loops are generated *live in memory*.
**Fix:** the resolver must **write the generated background noise loop to a temp 16 kHz/mono/16-bit WAV**,
pass its path to the mixer, and delete it in a `finally` that runs on the **cancel path**
too (the error sink cancels the pipeline at `run_simulation.py:1403-1404`). Owner:
`noise/save.py` (or the wiring in `_run_simulation_inner`).

**#3 — Seed / run-index threading.** A stable per-run id exists — `persona_index` +
`scenario_index`, enumerated in `main()` at `2322-2323` and in scope at the
`_run_single_simulation_inner` noise-read (~`2087`). **Fix:** resolve the condition + build
the background noise loop **at that ~`2087` site** (which has config + both indices), seeding by
`persona_index * len(scenarios) + scenario_index`; thread the **temp background noise loop path + volume**
(not raw `noise` config) down `run_simulation → _run_simulation_inner → mixer`. This keeps
the seed source in scope and avoids building the background noise loop deep where indices aren't available.

**#4 — Concurrency collision.** Sims run in parallel under
`asyncio.Semaphore(args.parallel)` (`2318`) into a shared `output_dir`. **Fix:** temp background noise loop
path must be **unique per sim** — put it under the per-sim `simulation_output_dir`
(`2030`) or key it by `simulation_run_id` (uuid, `2054`).

**#5 — Bundled OGG sample rate.** The mixer needs exactly 16 kHz; the *shipped* assets are
OGG. **Fix:** guarantee **all bundled OGGs (env + speaker) are encoded 16 kHz mono**, and
the live overlay/merge re-quantizes to 16 kHz/mono/16-bit before writing the temp background noise loop.
(Vaani raw is 16 kHz; CC0-re-sourced env clips must be resampled at asset-prep time.)

**#6 — Loudness→volume needs a per-voice reference.** `volume` is a fixed scalar on the
background noise loop; hitting a target `loudness`(SNR) needs the **caller speech RMS**, which differs by
provider/voice (ElevenLabs en/hi vs Google Chirp3 kn). **Fix:** measure a per-voice
reference RMS (or RMS-normalize the TTS reference) and map `loudness → volume` against it;
document that absolute dB is approximate across providers. (Consistent with decision #2:
SNR is average/approximate, not exact-per-utterance — and the same volume is used for the
live mix and the offline reconstruction.)

**#7 — `combine_audio_files` prefix kwarg default.** Must default to `""` so the existing
`run_simulation.py:2233` call and the format-match check in `combine_audio_files`
(`utils.py:564`) are unchanged.

**#8 — `sweep` vs fixed batch size — RESOLVED: dropped for v1.** The batch size is fixed at
`|personas| × |scenarios|` (`2115-2116`), so `sweep`'s `repeat: K` run-count knob doesn't
fit. **Decision: drop `sweep`**; `random`/`mixture`/`fixed` all work within the fixed run
count. Revisit later if guaranteed even per-condition coverage is wanted.

**VAD under continuous noise — INTENDED, mitigated, test at build.** Continuous noise on
the *sim-user* transport reaches the **agent's input VAD/endpointing** during the agent's
own turns and gaps. **This is desired** — the goal is to stress the agent under realistic
always-on background (decision #2), so we keep it continuous. Two things reduce the risk
that it *mechanically* breaks turn-taking rather than fairly testing it: (1) the
**backgroundify chain (§7.4)** makes speaker noise muffled/distant/reverberant, so the
agent's SileroVAD is far less likely to latch onto it than onto clear foreground speech;
(2) the sim-user's own VAD is commented out (`run_simulation.py:1486`), so the sim side
won't self-trigger. The agent's VAD is external — **validate empirically** the first time
we run with noise on. **Fallback if turn-taking still collapses:** gate the background noise loop to the
caller's speech turns via `MixerEnableFrame(enable)` on turn boundaries — a config toggle,
less realistic but stable, not a redesign.

*(Sanity checks that passed during verification: `audio_out_mixer` IS a field on
`WebsocketClientParams`; `SoundfileMixer` silently drops rate-mismatched files and mixes
mono via a flat `np.frombuffer` add; the renamed `calibrate_agent.agent.run_simulation`
imports cleanly on pipecat 1.0.)*

> *Note:* an earlier draft had a second "Verification findings" section here; it's been
> merged into the #1–#8 list above. Two details it added are folded in: **RMS-normalize each
> background noise loop to a fixed reference** (so "medium rain" and "medium cafe" land at the same real SNR —
> part of #6), and the **`MixerEnableFrame` VAD mitigation** (above).
