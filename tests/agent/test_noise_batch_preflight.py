"""Tests for the pre-run noise-asset check in the voice-sim batch runner."""

from __future__ import annotations

import pytest

from calibrate_agent.agent import run_simulation as rs


def _config(noise: dict) -> dict:
    return {
        "personas": [{"language": "hindi", "noise": noise}],
        "scenarios": [{"name": "s", "description": "d"}],
    }


def test_preflight_fails_fast_without_login(monkeypatch):
    monkeypatch.setattr(rs, "check_speaker_access", lambda: "please log in")
    downloaded = []
    monkeypatch.setattr(rs, "ensure_assets", lambda *a, **k: downloaded.append(1))
    cfg = _config({"mode": "fixed", "people": "medium", "loudness": "loud"})
    with pytest.raises(RuntimeError, match="please log in"):
        rs.prepare_noise_for_batch(cfg)
    assert downloaded == []  # nothing downloaded before the login failure


def test_preflight_env_only_skips_login_check(monkeypatch):
    def boom():
        raise AssertionError("env-only noise must not check Hugging Face login")

    monkeypatch.setattr(rs, "check_speaker_access", boom)
    needs_speakers = []
    monkeypatch.setattr(
        rs, "ensure_assets", lambda *a, **k: needs_speakers.append(k["need_speakers"])
    )
    cfg = _config(
        {"mode": "fixed", "environment": "busy_street", "people": "none", "loudness": "moderate"}
    )
    rs.prepare_noise_for_batch(cfg)
    assert needs_speakers == [False]


def test_preflight_noop_without_noise(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no noise means no asset work")

    monkeypatch.setattr(rs, "ensure_assets", boom)
    monkeypatch.setattr(rs, "check_speaker_access", boom)
    rs.prepare_noise_for_batch({"personas": [{"language": "hindi"}], "scenarios": [{"name": "s"}]})
