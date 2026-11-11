import json
import os

import numpy as np
import soundfile as sf

from calibrate_agent.agent.noise.save import mix_noise_over_wav, write_clean_and_noisy

SR = 16000


def _write_wav(path, samples):
    sf.write(path, np.asarray(samples, dtype=np.int16), SR, subtype="PCM_16")


def _rms(path):
    data, _ = sf.read(path, dtype="int16")
    return float(np.sqrt(np.mean(np.asarray(data, dtype=np.float64) ** 2)))


def _make_sine(n, amp=8000, freq=220):
    t = np.arange(n)
    return (amp * np.sin(2 * np.pi * freq * t / SR)).astype(np.int16)


def test_mix_noise_over_wav_basic(tmp_path):
    n = SR  # 1 second
    clean = _make_sine(n)
    noise = (2000 * np.sin(2 * np.pi * 500 * np.arange(n // 3) / SR)).astype(np.int16)

    clean_p = str(tmp_path / "clean.wav")
    noise_p = str(tmp_path / "noise.wav")
    out_p = str(tmp_path / "out.wav")
    _write_wav(clean_p, clean)
    _write_wav(noise_p, noise)

    volume = 0.7
    ret = mix_noise_over_wav(clean_p, noise_p, out_p, volume)
    assert ret == out_p

    data, sr = sf.read(out_p, dtype="int16")
    assert sr == SR
    assert data.ndim == 1  # mono
    assert data.dtype == np.int16
    assert data.shape[0] == n  # same length as clean

    # Expected: clip(clean + tiled_noise * volume)
    reps = int(np.ceil(n / noise.shape[0]))
    tiled = np.tile(noise, reps)[:n]
    expected = np.clip(
        np.round(clean.astype(np.float64) + tiled.astype(np.float64) * volume),
        -32768,
        32767,
    ).astype(np.int16)
    assert np.max(np.abs(data.astype(np.int64) - expected.astype(np.int64))) <= 1


def test_mix_noise_volume_zero_is_clean(tmp_path):
    n = SR
    clean = _make_sine(n)
    noise = (3000 * np.sin(2 * np.pi * 400 * np.arange(n // 4) / SR)).astype(np.int16)

    clean_p = str(tmp_path / "clean.wav")
    noise_p = str(tmp_path / "noise.wav")
    out_p = str(tmp_path / "out.wav")
    _write_wav(clean_p, clean)
    _write_wav(noise_p, noise)

    mix_noise_over_wav(clean_p, noise_p, out_p, 0.0)
    data, _ = sf.read(out_p, dtype="int16")
    assert np.array_equal(np.asarray(data, dtype=np.int16), clean)


def test_mix_noise_louder_is_higher_rms(tmp_path):
    n = SR
    # Low-amplitude clean so headroom keeps mixing from clipping to a wall.
    clean = _make_sine(n, amp=1000)
    noise = (4000 * np.sin(2 * np.pi * 350 * np.arange(n // 5) / SR)).astype(np.int16)

    clean_p = str(tmp_path / "clean.wav")
    noise_p = str(tmp_path / "noise.wav")
    quiet_p = str(tmp_path / "quiet.wav")
    loud_p = str(tmp_path / "loud.wav")
    _write_wav(clean_p, clean)
    _write_wav(noise_p, noise)

    mix_noise_over_wav(clean_p, noise_p, quiet_p, 0.2)
    mix_noise_over_wav(clean_p, noise_p, loud_p, 1.0)
    assert _rms(loud_p) > _rms(quiet_p)


def _stub_combine(monkeypatch, audio_dir):
    """Stub combine_audio_files: concatenate {prefix}{N}_{role}.wav per transcript."""

    def _combine(a_dir, output_path, transcript_path=None, prefix=""):
        with open(transcript_path) as f:
            transcript = json.load(f)
        chunks = []
        idx = 1
        for msg in transcript:
            if msg.get("content") is None:
                continue
            role = msg["role"]
            fname = "bot" if role == "assistant" else "user"
            p = os.path.join(a_dir, f"{prefix}{idx}_{fname}.wav")
            if os.path.exists(p):
                data, _ = sf.read(p, dtype="int16")
                chunks.append(np.asarray(data, dtype=np.int16))
            idx += 1
        combined = (
            np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        )
        _write_wav(output_path, combined)
        return True

    monkeypatch.setattr("calibrate_agent.utils.combine_audio_files", _combine)


def test_write_clean_and_noisy(tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    _write_wav(str(audio_dir / "1_user.wav"), _make_sine(SR, amp=3000, freq=200))
    _write_wav(str(audio_dir / "1_bot.wav"), _make_sine(SR, amp=3000, freq=300))
    _write_wav(str(audio_dir / "2_user.wav"), _make_sine(SR, amp=3000, freq=250))

    transcript = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]
    transcript_path = str(tmp_path / "transcript.json")
    with open(transcript_path, "w") as f:
        json.dump(transcript, f)

    noise_path = str(tmp_path / "noise.wav")
    _write_wav(noise_path, (4000 * np.sin(2 * np.pi * 500 * np.arange(SR // 2) / SR)).astype(np.int16))

    conversation_path = str(tmp_path / "conversation.wav")

    _stub_combine(monkeypatch, str(audio_dir))

    write_clean_and_noisy(
        audio_dir=str(audio_dir),
        transcript_path=transcript_path,
        conversation_path=conversation_path,
        noise_track_path=noise_path,
        volume=0.8,
    )

    # Clean per-turn copies created.
    assert os.path.exists(audio_dir / "clean_1_user.wav")
    assert os.path.exists(audio_dir / "clean_1_bot.wav")
    assert os.path.exists(audio_dir / "clean_2_user.wav")

    clean_conv = tmp_path / "clean_conversation.wav"
    assert clean_conv.exists()
    assert os.path.exists(conversation_path)

    # No leftover temp files.
    leftover = [
        p
        for p in os.listdir(tmp_path)
        if p.endswith(".wav")
        and p not in ("conversation.wav", "clean_conversation.wav", "noise.wav")
    ]
    assert leftover == []

    # Noisy conversation is noisier than the clean stitch.
    assert _rms(conversation_path) > _rms(str(clean_conv))


def test_write_clean_and_noisy_no_turns_noop(tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    transcript_path = str(tmp_path / "transcript.json")
    with open(transcript_path, "w") as f:
        json.dump([], f)
    noise_path = str(tmp_path / "noise.wav")
    _write_wav(noise_path, np.zeros(SR // 2, dtype=np.int16))
    conversation_path = str(tmp_path / "conversation.wav")

    called = {"v": False}

    def _combine(*a, **k):
        called["v"] = True
        return True

    monkeypatch.setattr("calibrate_agent.utils.combine_audio_files", _combine)

    write_clean_and_noisy(
        audio_dir=str(audio_dir),
        transcript_path=transcript_path,
        conversation_path=conversation_path,
        noise_track_path=noise_path,
        volume=0.8,
    )

    assert not os.path.exists(conversation_path)
    assert not called["v"]
