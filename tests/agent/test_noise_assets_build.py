"""Tests for building and resolving a noise asset bundle from raw clips.

These use a synthetic bundle built in a temp dir, so they run in CI without the
real (gitignored, gated) audio downloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from calibrate_agent.agent.noise.assets import NoiseAssets, prepare_assets


def _tone(seconds: float, sr: int, channels: int = 1) -> np.ndarray:
    n = int(seconds * sr)
    wave = 0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / sr).astype(np.float32)
    if channels == 2:
        return np.stack([wave, wave], axis=1)
    return wave


@pytest.fixture
def built_bundle(tmp_path: Path) -> Path:
    """Build a noise bundle from synthetic raw clips and return its root."""
    env_raw = tmp_path / "env_raw"
    spk_raw = tmp_path / "vaani_raw"
    (env_raw / "dog").mkdir(parents=True)
    (spk_raw / "hindi").mkdir(parents=True)
    (spk_raw / "english").mkdir(parents=True)

    # Steady env clips: stereo + non-16k to exercise downmix and resample.
    sf.write(env_raw / "rain.wav", _tone(0.5, 44100, channels=2), 44100)
    # vacuum_cleaner is renamed to "vacuum" by the asset name map.
    sf.write(env_raw / "vacuum_cleaner.wav", _tone(0.5, 22050), 22050)
    # Event bank (scattered, multi-sample).
    sf.write(env_raw / "dog" / "00.wav", _tone(0.3, 16000), 16000)
    sf.write(env_raw / "dog" / "01.wav", _tone(0.3, 16000), 16000)

    # Speakers: one needs resampling (rewrite path), one is already 16k mono
    # PCM_16 (straight copy path).
    sf.write(spk_raw / "hindi" / "a.wav", _tone(0.4, 8000), 8000)
    sf.write(
        spk_raw / "english" / "b.wav", _tone(0.4, 16000), 16000, subtype="PCM_16"
    )

    bundle = tmp_path / "bundle"
    prepare_assets(src_env=str(env_raw), src_speakers=str(spk_raw), dest=str(bundle))
    return bundle


def test_prepare_assets_outputs_are_16k_mono_pcm16(built_bundle: Path):
    wavs = list(built_bundle.rglob("*.wav"))
    assert wavs, "bundle produced no wavs"
    for wav in wavs:
        info = sf.info(str(wav))
        assert info.samplerate == 16000, f"{wav} sr={info.samplerate}"
        assert info.channels == 1, f"{wav} channels={info.channels}"
        assert info.subtype == "PCM_16", f"{wav} subtype={info.subtype}"


def test_prepare_assets_renames_and_lays_out_files(built_bundle: Path):
    env = built_bundle / "env"
    assert (env / "rain.wav").is_file()
    assert (env / "vacuum.wav").is_file()  # renamed from vacuum_cleaner
    assert not (env / "vacuum_cleaner.wav").exists()
    assert {p.name for p in (env / "dog").glob("*.wav")} == {"00.wav", "01.wav"}
    assert (built_bundle / "speakers" / "hindi" / "a.wav").is_file()
    assert (built_bundle / "speakers" / "english" / "b.wav").is_file()


def test_env_ingredient_paths_steady_event_and_missing(built_bundle: Path):
    assets = NoiseAssets(root=str(built_bundle))
    assert len(assets.env_ingredient_paths("rain")) == 1
    assert len(assets.env_ingredient_paths("dog")) == 2
    assert assets.env_ingredient_paths("nonexistent") == []


def test_speaker_pool_present_and_missing(built_bundle: Path):
    assets = NoiseAssets(root=str(built_bundle))
    assert len(assets.speaker_pool("hindi")) == 1
    assert len(assets.speaker_pool("english")) == 1
    assert assets.speaker_pool("kannada") == []  # language dir absent


def test_noiseassets_env_var_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CALIBRATE_NOISE_ASSETS", str(tmp_path / "x"))
    assert NoiseAssets().root == tmp_path / "x"


def test_noiseassets_default_root_when_unset(monkeypatch):
    monkeypatch.delenv("CALIBRATE_NOISE_ASSETS", raising=False)
    root = NoiseAssets().root
    assert root.parts[-2:] == ("assets", "noise")


def test_resolve_environment_all_branches(built_bundle: Path):
    assets = NoiseAssets(root=str(built_bundle))
    assert assets.resolve_environment(None) == []
    assert assets.resolve_environment("none") == []
    assert assets.resolve_environment("busy_street") == ["car_horn", "engine", "siren"]
    assert assets.resolve_environment("rain") == ["rain"]
    assert assets.resolve_environment(["rain", "dog"]) == ["rain", "dog"]
