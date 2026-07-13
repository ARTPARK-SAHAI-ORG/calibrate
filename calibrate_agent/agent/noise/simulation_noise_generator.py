"""Core DSP for the voice-sim background-noise feature.

Builds one continuous, seamless-looping, RMS-normalized 16 kHz mono background
track from a :class:`NoiseAtom` (environment ingredients + optional distant
crowd chatter) and writes it as a 16-bit PCM WAV.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import soxr
from scipy.signal import butter, fftconvolve, sosfilt

# These mirror calibrate_agent/agent/noise/schema.py. They are re-declared here
# (rather than imported at module top-level) so this module works even if
# schema.py is absent, and so the DSP never depends on schema import order.
SAMPLE_RATE = 16000
DENSITY_VOICES = {"none": 0, "single": 1, "light": 3, "medium": 5, "heavy": 10}
EVENT_SOUNDS = {"dog", "car_horn"}

_TARGET_RMS_DBFS = -20.0
_CROSSFADE_S = 0.05
_LOOP_WRAP_S = 0.25
_RIR_S = 0.4
_REVERB_WET = 0.30
_CROWD_UNDER_GAIN = 0.5  # people mixed ~-6 dB under env
_SILENT_FLOOR = 1e-4  # near-silent output level for empty atoms


def _load_wav_mono(path: str, sample_rate: int) -> np.ndarray | None:
    """Load a wav as float32 mono at ``sample_rate``. Returns None on failure."""
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.ascontiguousarray(data, dtype=np.float32)
    if sr != sample_rate and data.size:
        data = soxr.resample(data, sr, sample_rate).astype(np.float32)
    return data


def _tile_with_crossfades(
    clip: np.ndarray, total: int, sample_rate: int
) -> np.ndarray:
    """Loop ``clip`` to ``total`` samples with short crossfades at each join."""
    if clip.size == 0 or total <= 0:
        return np.zeros(max(total, 0), dtype=np.float32)

    fade = int(_CROSSFADE_S * sample_rate)
    fade = min(fade, clip.size // 2)
    out = np.zeros(total, dtype=np.float32)

    if fade <= 0:
        # Clip too short to crossfade; plain tile.
        reps = int(np.ceil(total / clip.size))
        tiled = np.tile(clip, reps)[:total]
        return tiled.astype(np.float32)

    fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    fade_out = fade_in[::-1]
    step = clip.size - fade  # advance per placement (overlap = fade)

    pos = 0
    first = True
    while pos < total:
        seg = clip.copy()
        if not first:
            seg[:fade] *= fade_in
        # Fade the tail so the next overlapping copy blends in cleanly.
        seg[-fade:] *= fade_out
        end = min(pos + seg.size, total)
        out[pos:end] += seg[: end - pos]
        pos += step
        first = False
    return out


def _scatter_events(
    clips: list[np.ndarray],
    total: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Scatter event clips over a silent buffer at random times/gaps/gains."""
    out = np.zeros(total, dtype=np.float32)
    clips = [c for c in clips if c.size]
    if not clips or total <= 0:
        return out

    pos = int(rng.uniform(0.0, 2.0) * sample_rate)
    while pos < total:
        clip = clips[rng.integers(0, len(clips))]
        gain = float(rng.uniform(0.7, 1.0))
        end = min(pos + clip.size, total)
        out[pos:end] += clip[: end - pos] * gain
        gap = rng.uniform(1.0, 5.0)
        pos = end + int(gap * sample_rate)
    return out


def _synthetic_rir(sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Exponentially-decaying white-noise room impulse response."""
    n = max(1, int(_RIR_S * sample_rate))
    decay = np.exp(-np.linspace(0.0, 6.0, n)).astype(np.float32)
    rir = rng.standard_normal(n).astype(np.float32) * decay
    peak = np.max(np.abs(rir))
    if peak > 0:
        rir = rir / peak
    return rir


def _backgroundify(
    crowd: np.ndarray, sample_rate: int, rng: np.random.Generator
) -> np.ndarray:
    """Make a crowd sum sound like distant background chatter."""
    if crowd.size == 0:
        return crowd
    nyq = sample_rate / 2.0
    hp = butter(2, 150.0 / nyq, btype="highpass", output="sos")
    lp = butter(4, min(3500.0, nyq - 1.0) / nyq, btype="lowpass", output="sos")
    dry = sosfilt(lp, sosfilt(hp, crowd)).astype(np.float32)

    rir = _synthetic_rir(sample_rate, rng)
    wet = fftconvolve(dry, rir)[: dry.size].astype(np.float32)
    return ((1.0 - _REVERB_WET) * dry + _REVERB_WET * wet).astype(np.float32)


def _build_crowd(
    speaker_paths: list[str],
    count: int,
    total: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sum ``count`` random speaker clips, each looped at a random offset."""
    out = np.zeros(total, dtype=np.float32)
    if count <= 0 or not speaker_paths or total <= 0:
        return out
    nyq = sample_rate / 2.0
    for _ in range(count):
        path = speaker_paths[rng.integers(0, len(speaker_paths))]
        clip = _load_wav_mono(path, sample_rate)
        if clip is None or clip.size == 0:
            continue
        voice = _tile_with_crossfades(clip, total, sample_rate)
        offset = int(rng.uniform(0.0, min(2.0, total / sample_rate)) * sample_rate)
        if offset:
            voice = np.roll(voice, offset)
        # Slight per-voice low-pass variation for depth.
        cutoff = float(rng.uniform(2500.0, 3800.0))
        lp = butter(2, min(cutoff, nyq - 1.0) / nyq, btype="lowpass", output="sos")
        voice = sosfilt(lp, voice).astype(np.float32)
        voice *= float(rng.uniform(0.7, 1.0))
        out += voice
    return _backgroundify(out, sample_rate, rng)


def _rms_normalize(track: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(track)))) if track.size else 0.0
    if rms > 0:
        target = 10.0 ** (_TARGET_RMS_DBFS / 20.0)
        track = track * (target / rms)
    return np.clip(track, -1.0, 1.0).astype(np.float32)


def _loop_wrap(track: np.ndarray, sample_rate: int) -> np.ndarray:
    """Crossfade the tail into the head so the track loops without a click."""
    fade = int(_LOOP_WRAP_S * sample_rate)
    fade = min(fade, track.size // 2)
    if fade <= 0:
        return track
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    head = track[:fade].copy()
    tail = track[-fade:].copy()
    track = track[:-fade].copy()
    track[:fade] = head * ramp + tail * (1.0 - ramp)
    return track


class SimulationNoiseGenerator:
    """Builds one background-noise track per simulation from a NoiseAtom."""

    def __init__(self, assets, sample_rate: int = SAMPLE_RATE):
        self.assets = assets
        self.sample_rate = sample_rate

    def _build_env(
        self, atom, total: int, rng: np.random.Generator
    ) -> np.ndarray | None:
        names = self.assets.resolve_environment(getattr(atom, "environment", None))
        if not names:
            return None
        out = np.zeros(total, dtype=np.float32)
        any_sound = False
        for name in names:
            paths = self.assets.env_ingredient_paths(name)
            if not paths:
                continue
            if name in EVENT_SOUNDS:
                clips = [_load_wav_mono(p, self.sample_rate) for p in paths]
                clips = [c for c in clips if c is not None and c.size]
                if clips:
                    out += _scatter_events(
                        clips, total, self.sample_rate, rng
                    )
                    any_sound = True
            else:
                clip = _load_wav_mono(paths[0], self.sample_rate)
                if clip is not None and clip.size:
                    out += _tile_with_crossfades(clip, total, self.sample_rate)
                    any_sound = True
        return out if any_sound else None

    def build(
        self,
        atom,
        *,
        language: str,
        out_path: str,
        seed: int,
        duration_s: float = 45.0,
    ) -> str:
        rng = np.random.default_rng(seed)
        target_len = int(round(duration_s * self.sample_rate))
        # Build extra tail so the loop-wrap crossfade leaves exactly target_len.
        wrap = min(int(_LOOP_WRAP_S * self.sample_rate), target_len // 2)
        total = target_len + wrap

        env = self._build_env(atom, total, rng)

        people = getattr(atom, "people", "none")
        count = DENSITY_VOICES.get(people, 0)
        crowd = None
        if count > 0:
            speakers = self.assets.speaker_pool(language)
            crowd = _build_crowd(
                speakers, count, total, self.sample_rate, rng
            )
            if not np.any(crowd):
                crowd = None

        if env is not None and crowd is not None:
            track = _rms_normalize(env + _CROWD_UNDER_GAIN * crowd)
        elif env is not None:
            track = _rms_normalize(env)
        elif crowd is not None:
            track = _rms_normalize(crowd)
        else:
            # Empty atom: near-silent noise, no RMS normalize-up.
            track = (
                rng.standard_normal(total).astype(np.float32) * _SILENT_FLOOR
            )
            track = np.clip(track, -1.0, 1.0).astype(np.float32)

        track = _loop_wrap(track, self.sample_rate)

        sf.write(
            out_path,
            track,
            self.sample_rate,
            subtype="PCM_16",
            format="WAV",
        )
        return out_path
