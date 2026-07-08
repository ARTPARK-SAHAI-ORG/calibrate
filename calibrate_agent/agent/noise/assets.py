"""Asset management for voice-sim background-noise injection.

Populates and locates the bundled noise asset directory
(``calibrate_agent/agent/assets/noise``) that the noise mixer draws from:

- ``env/<name>.wav`` — steady environmental loops (16 kHz mono 16-bit).
- ``env/dog/NN.wav`` and ``env/car_horn/NN.wav`` — single-event sample banks.
- ``speakers/{english,hindi,kannada}/*.wav`` — distant chatter speaker pool.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

try:
    from calibrate_agent.agent.noise.schema import (  # type: ignore
        EVENT_SOUNDS,
        SCENES,
        SINGLE_SOUNDS,
    )
except Exception:  # pragma: no cover - schema.py may not exist yet
    # TODO: remove this fallback once calibrate_agent/agent/noise/schema.py lands.
    EVENT_SOUNDS = ["dog", "car_horn"]
    SINGLE_SOUNDS = [
        "rain",
        "wind",
        "engine",
        "vacuum",
        "train",
        "siren",
        "crying_baby",
        "footsteps",
        "keyboard_typing",
        "laughing",
        "dog",
        "car_horn",
    ]
    SCENES = {
        "busy_street": ["car_horn", "engine", "siren"],
        "vehicle": ["engine", "wind", "car_horn"],
        "railway_station": ["train", "footsteps"],
        "office": ["keyboard_typing"],
        "home_with_baby": ["crying_baby"],
        "housework": ["vacuum"],
        "rainy_street": ["rain", "wind", "car_horn"],
        "quiet_home": ["dog"],
    }

TARGET_SR = 16000

_ENV_ASSET_NAMES = {"vacuum_cleaner": "vacuum"}
_SPEAKER_LANGUAGES = ("english", "hindi", "kannada")
_ENV_DEST_ENV_ROOT = "env"
_SPEAKERS_DEST_ROOT = "speakers"

_ASSETS_ENV = "CALIBRATE_NOISE_ASSETS"


def _default_assets_root() -> Path:
    """Locate the bundled ``assets/noise`` dir, tolerating unpacked layouts."""
    try:
        from importlib.resources import files

        base = files("calibrate_agent.agent")
        candidate = Path(str(base)) / "assets" / "noise"
        return candidate
    except Exception:  # pragma: no cover - fallback for odd install layouts
        return Path(__file__).resolve().parent.parent / "assets" / "noise"


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), sr


def _write_16k_mono(dest: Path, data: np.ndarray, sr: int) -> None:
    if sr != TARGET_SR:
        data = soxr.resample(data, sr, TARGET_SR)
    data = np.asarray(data, dtype=np.float32)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), data, TARGET_SR, subtype="PCM_16")


def _is_16k_mono_pcm16(path: Path) -> bool:
    try:
        info = sf.info(str(path))
    except Exception:
        return False
    return (
        info.samplerate == TARGET_SR
        and info.channels == 1
        and info.subtype == "PCM_16"
    )


def prepare_assets(
    src_env: str = "data/env_raw",
    src_speakers: str = "data/vaani_raw",
    dest: str | None = None,
) -> None:
    """Populate the bundled noise asset directory from raw clips.

    Idempotent: re-running overwrites existing outputs with fresh conversions.
    """
    src_env_path = Path(src_env)
    src_speakers_path = Path(src_speakers)
    dest_root = Path(dest) if dest is not None else _default_assets_root()

    env_dest = dest_root / _ENV_DEST_ENV_ROOT
    env_dest.mkdir(parents=True, exist_ok=True)

    # Steady single-clip environmental sounds.
    for wav in sorted(src_env_path.glob("*.wav")):
        stem = wav.stem
        name = _ENV_ASSET_NAMES.get(stem, stem)
        data, sr = _load_mono(wav)
        _write_16k_mono(env_dest / f"{name}.wav", data, sr)

    # Multi-sample event banks (dog, car_horn).
    for event in EVENT_SOUNDS:
        src_dir = src_env_path / event
        if not src_dir.is_dir():
            continue
        for wav in sorted(src_dir.glob("*.wav")):
            data, sr = _load_mono(wav)
            _write_16k_mono(env_dest / event / wav.name, data, sr)

    # Speaker pools (already 16k mono; verify and re-write if not).
    for language in _SPEAKER_LANGUAGES:
        src_dir = src_speakers_path / language
        if not src_dir.is_dir():
            continue
        lang_dest = dest_root / _SPEAKERS_DEST_ROOT / language
        lang_dest.mkdir(parents=True, exist_ok=True)
        for wav in sorted(src_dir.glob("*.wav")):
            out = lang_dest / wav.name
            if _is_16k_mono_pcm16(wav):
                shutil.copyfile(wav, out)
            else:
                data, sr = _load_mono(wav)
                _write_16k_mono(out, data, sr)


class NoiseAssets:
    """Locates bundled noise assets and resolves scene/sound names to paths."""

    def __init__(self, root: str | None = None):
        if root is not None:
            self.root = Path(root)
        elif os.environ.get(_ASSETS_ENV):
            self.root = Path(os.environ[_ASSETS_ENV])
        else:
            self.root = _default_assets_root()

    def resolve_environment(self, environment) -> list[str]:
        """Normalise an environment spec to a list of ingredient sound names."""
        if environment is None:
            return []
        if isinstance(environment, (list, tuple)):
            return list(environment)
        if environment == "none":
            return []
        if environment in SCENES:
            return list(SCENES[environment])
        return [environment]

    def env_ingredient_paths(self, name: str) -> list[str]:
        """Return the wav path(s) backing an environmental sound name."""
        env_root = self.root / _ENV_DEST_ENV_ROOT
        event_dir = env_root / name
        if event_dir.is_dir():
            return [str(p) for p in sorted(event_dir.glob("*.wav"))]
        steady = env_root / f"{name}.wav"
        if steady.is_file():
            return [str(steady)]
        return []

    def speaker_pool(self, language: str) -> list[str]:
        """Return all speaker wav paths for a language."""
        lang_dir = self.root / _SPEAKERS_DEST_ROOT / language
        if not lang_dir.is_dir():
            return []
        return [str(p) for p in sorted(lang_dir.glob("*.wav"))]
