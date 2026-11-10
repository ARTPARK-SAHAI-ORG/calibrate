"""Tests for the noise-asset fetch script's pre-download setup check."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import huggingface_hub

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_noise_assets.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("fetch_noise_assets", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_speaker_access_without_login(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    msg = _load_script().check_speaker_access()
    assert msg is not None
    assert "huggingface-cli login" in msg
    assert "--skip-speakers" in msg


def test_check_speaker_access_logged_in(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_fake")
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", lambda self, name: object())
    assert _load_script().check_speaker_access() is None
