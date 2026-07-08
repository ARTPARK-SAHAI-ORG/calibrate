"""Tests for calibrate_agent.agent.noise.assets."""

from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf

from calibrate_agent.agent.noise.assets import NoiseAssets

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ENV = REPO_ROOT / "data" / "env_raw"


def _bundle_root() -> Path:
    return REPO_ROOT / "calibrate_agent" / "agent" / "assets" / "noise"


def _bundle_ready() -> bool:
    return _bundle_root().is_dir() and RAW_ENV.is_dir()


pytestmark = pytest.mark.skipif(
    not _bundle_ready(),
    reason="raw data/ clips or populated bundle absent",
)


def test_bundle_files_are_16k_mono_pcm16():
    root = _bundle_root()
    wavs = list(root.rglob("*.wav"))
    assert wavs, "no wav assets found in bundle"
    for wav in wavs:
        info = sf.info(str(wav))
        assert info.samplerate == 16000, f"{wav} sr={info.samplerate}"
        assert info.channels == 1, f"{wav} channels={info.channels}"
        assert info.subtype == "PCM_16", f"{wav} subtype={info.subtype}"


def test_resolve_environment_scene():
    assets = NoiseAssets(root=str(_bundle_root()))
    assert assets.resolve_environment("busy_street") == ["car_horn", "engine", "siren"]


def test_resolve_environment_single_sound():
    assets = NoiseAssets(root=str(_bundle_root()))
    assert assets.resolve_environment("rain") == ["rain"]


def test_resolve_environment_list_passthrough():
    assets = NoiseAssets(root=str(_bundle_root()))
    assert assets.resolve_environment(["rain", "siren"]) == ["rain", "siren"]


def test_resolve_environment_none_and_off():
    assets = NoiseAssets(root=str(_bundle_root()))
    assert assets.resolve_environment(None) == []
    assert assets.resolve_environment("none") == []


def test_env_ingredient_paths_event_bank():
    assets = NoiseAssets(root=str(_bundle_root()))
    paths = assets.env_ingredient_paths("dog")
    assert len(paths) > 1
    assert all(Path(p).is_file() for p in paths)


def test_env_ingredient_paths_steady():
    assets = NoiseAssets(root=str(_bundle_root()))
    paths = assets.env_ingredient_paths("rain")
    assert len(paths) == 1
    assert Path(paths[0]).is_file()


def test_speaker_pool_hindi():
    assets = NoiseAssets(root=str(_bundle_root()))
    pool = assets.speaker_pool("hindi")
    assert pool, "hindi speaker pool empty"
    assert all(Path(p).is_file() for p in pool)
