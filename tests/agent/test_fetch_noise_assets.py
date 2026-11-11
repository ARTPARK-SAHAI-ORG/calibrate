"""Tests for the noise-asset download module (setup check + first-use build)."""

from __future__ import annotations

from pathlib import Path

import huggingface_hub
import numpy as np
import pytest
import soundfile as sf

from calibrate_agent.agent.noise import download
from calibrate_agent.agent.noise.assets import NoiseAssets


def test_check_speaker_access_without_login(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    msg = download.check_speaker_access()
    assert msg is not None
    assert "huggingface-cli login" in msg


def test_check_speaker_access_logged_in(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_fake")
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", lambda self, name: object())
    assert download.check_speaker_access() is None


def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("download should not run")

    monkeypatch.setattr(download, "fetch_env", boom)
    monkeypatch.setattr(download, "fetch_speakers", boom)


def test_ensure_assets_noop_when_present(tmp_path: Path, monkeypatch):
    # A bundle that already has the needed env clip and hindi speaker.
    bundle = tmp_path / "noise"
    (bundle / "env").mkdir(parents=True)
    (bundle / "speakers" / "hindi").mkdir(parents=True)
    tone = 0.1 * np.sin(np.arange(1600)).astype(np.float32)
    sf.write(bundle / "env" / "rain.wav", tone, 16000, subtype="PCM_16")
    sf.write(bundle / "speakers" / "hindi" / "a.wav", tone, 16000, subtype="PCM_16")
    _no_network(monkeypatch)
    # Must return without attempting any download.
    download.ensure_assets(
        NoiseAssets(root=str(bundle)),
        env_names=["rain"],
        need_speakers=True,
        language="hindi",
    )


def test_ensure_assets_raises_without_login(tmp_path: Path, monkeypatch):
    _no_network(monkeypatch)
    monkeypatch.setattr(download, "check_speaker_access", lambda: "please log in")
    with pytest.raises(RuntimeError, match="please log in"):
        download.ensure_assets(
            NoiseAssets(root=str(tmp_path / "empty")),
            env_names=[],
            need_speakers=True,
            language="hindi",
        )


def test_ensure_assets_rejects_unsupported_language(tmp_path: Path, monkeypatch):
    _no_network(monkeypatch)
    with pytest.raises(RuntimeError, match="Supported"):
        download.ensure_assets(
            NoiseAssets(root=str(tmp_path / "empty")),
            env_names=[],
            need_speakers=True,
            language="tamil",
        )
