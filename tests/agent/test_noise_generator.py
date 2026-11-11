"""Unit tests for the simulation noise DSP generator."""

from __future__ import annotations

import filecmp
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from calibrate_agent.agent.noise.simulation_noise_generator import (
    SimulationNoiseGenerator,
)


@dataclass(frozen=True)
class _Atom:
    environment: object = None
    people: str = "none"
    loudness: str = "medium"


def _write_wav(path, sample_rate, seconds=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    data = (rng.standard_normal(n) * 0.2).astype(np.float32)
    sf.write(str(path), data, sample_rate, subtype="PCM_16", format="WAV")
    return str(path)


class _StubAssets:
    """Minimal assets stub producing small synthetic wavs."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        # A 16k steady sound, an event sound with several 44.1k clips,
        # and a pool of 16k speaker clips.
        self.steady = _write_wav(tmp_path / "traffic.wav", 16000, seed=1)
        self.dog = [
            _write_wav(tmp_path / f"dog_{i}.wav", 44100, seconds=0.5, seed=10 + i)
            for i in range(3)
        ]
        self.speakers = [
            _write_wav(tmp_path / f"spk_{i}.wav", 16000, seed=100 + i)
            for i in range(6)
        ]

    def resolve_environment(self, environment):
        if environment is None:
            return []
        if isinstance(environment, str):
            return [environment]
        return list(environment)

    def env_ingredient_paths(self, name):
        if name == "traffic":
            return [self.steady]
        if name == "dog":
            return list(self.dog)
        if name == "missing":
            return ["/nonexistent/file.wav"]
        return []

    def speaker_pool(self, language):
        return list(self.speakers)


def _info(path):
    info = sf.info(path)
    return info


def test_env_only_output_format(tmp_path):
    assets = _StubAssets(tmp_path)
    gen = SimulationNoiseGenerator(assets)
    out = tmp_path / "env.wav"
    gen.build(
        _Atom(environment="traffic"),
        language="hi",
        out_path=str(out),
        seed=7,
        duration_s=3.0,
    )
    info = _info(out)
    assert info.samplerate == 16000
    assert info.channels == 1
    assert "PCM_16" in info.subtype
    assert abs(info.duration - 3.0) < 0.1


def test_people_only_medium(tmp_path):
    assets = _StubAssets(tmp_path)
    gen = SimulationNoiseGenerator(assets)
    out = tmp_path / "people.wav"
    gen.build(
        _Atom(people="medium"),
        language="hi",
        out_path=str(out),
        seed=3,
        duration_s=3.0,
    )
    info = _info(out)
    assert info.samplerate == 16000
    assert info.channels == 1
    data, _ = sf.read(str(out))
    assert np.any(data != 0)


def test_env_plus_people(tmp_path):
    assets = _StubAssets(tmp_path)
    gen = SimulationNoiseGenerator(assets)
    out = tmp_path / "both.wav"
    gen.build(
        _Atom(environment=["traffic", "dog"], people="light"),
        language="hi",
        out_path=str(out),
        seed=11,
        duration_s=3.0,
    )
    info = _info(out)
    assert info.samplerate == 16000
    assert info.channels == 1
    data, _ = sf.read(str(out))
    assert np.max(np.abs(data)) <= 1.0
    assert np.any(data != 0)


def test_deterministic_same_seed(tmp_path):
    assets = _StubAssets(tmp_path)
    gen = SimulationNoiseGenerator(assets)
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    atom = _Atom(environment=["traffic", "dog"], people="medium")
    gen.build(atom, language="hi", out_path=str(a), seed=42, duration_s=3.0)
    gen.build(atom, language="hi", out_path=str(b), seed=42, duration_s=3.0)
    assert filecmp.cmp(str(a), str(b), shallow=False)


def test_different_seed_differs(tmp_path):
    assets = _StubAssets(tmp_path)
    gen = SimulationNoiseGenerator(assets)
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    atom = _Atom(environment=["traffic", "dog"], people="medium")
    gen.build(atom, language="hi", out_path=str(a), seed=1, duration_s=3.0)
    gen.build(atom, language="hi", out_path=str(b), seed=2, duration_s=3.0)
    assert not filecmp.cmp(str(a), str(b), shallow=False)


def test_resamples_44k_event_clip(tmp_path):
    # dog clips are 44.1k; using them must not crash and output stays 16k.
    assets = _StubAssets(tmp_path)
    gen = SimulationNoiseGenerator(assets)
    out = tmp_path / "dog.wav"
    gen.build(
        _Atom(environment="dog"),
        language="hi",
        out_path=str(out),
        seed=5,
        duration_s=3.0,
    )
    info = _info(out)
    assert info.samplerate == 16000
    data, _ = sf.read(str(out))
    assert np.any(data != 0)


def test_empty_atom_near_silent(tmp_path):
    assets = _StubAssets(tmp_path)
    gen = SimulationNoiseGenerator(assets)
    out = tmp_path / "silent.wav"
    gen.build(
        _Atom(environment=None, people="none"),
        language="hi",
        out_path=str(out),
        seed=0,
        duration_s=2.0,
    )
    info = _info(out)
    assert info.samplerate == 16000
    assert info.channels == 1
    data, _ = sf.read(str(out))
    rms = float(np.sqrt(np.mean(data**2)))
    assert rms < 0.01


def test_missing_files_skipped(tmp_path):
    assets = _StubAssets(tmp_path)
    gen = SimulationNoiseGenerator(assets)
    out = tmp_path / "missing.wav"
    # "missing" resolves to a nonexistent path -> treated as empty env.
    gen.build(
        _Atom(environment="missing"),
        language="hi",
        out_path=str(out),
        seed=0,
        duration_s=2.0,
    )
    info = _info(out)
    assert info.samplerate == 16000
    assert info.channels == 1
