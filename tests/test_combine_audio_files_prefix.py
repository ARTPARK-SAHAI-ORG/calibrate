import json
import os
import wave

from calibrate_agent.utils import combine_audio_files

SAMPLE_RATE = 16000


def _write_wav(path, frames):
    """Write a mono 16-bit 16k WAV with the given number of silent-ish frames."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        # Distinct byte content per file so combined outputs differ.
        wf.writeframes(frames)


def _read_wav_bytes(path):
    with wave.open(path, "rb") as wf:
        return wf.readframes(wf.getnframes())


def _setup_dir(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Unprefixed files (default behavior). msg_index is a single running
    # counter over transcript messages: user -> 1, assistant -> 2.
    _write_wav(str(audio_dir / "1_user.wav"), b"\x01\x00" * 100)
    _write_wav(str(audio_dir / "2_bot.wav"), b"\x02\x00" * 100)

    # Prefixed "clean_" files with different content AND length.
    _write_wav(str(audio_dir / "clean_1_user.wav"), b"\x0a\x00" * 250)
    _write_wav(str(audio_dir / "clean_2_bot.wav"), b"\x0b\x00" * 250)

    transcript = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    transcript_path = tmp_path / "transcript.json"
    with open(transcript_path, "w") as f:
        json.dump(transcript, f)

    return str(audio_dir), str(transcript_path)


def test_default_prefix_stitches_unprefixed(tmp_path):
    audio_dir, transcript_path = _setup_dir(tmp_path)
    out = str(tmp_path / "out_default.wav")

    assert combine_audio_files(audio_dir, out, transcript_path, prefix="") is True
    assert os.path.exists(out)

    expected = b"\x01\x00" * 100 + b"\x02\x00" * 100
    assert _read_wav_bytes(out) == expected


def test_prefix_stitches_prefixed_files(tmp_path):
    audio_dir, transcript_path = _setup_dir(tmp_path)
    out = str(tmp_path / "out_clean.wav")

    assert (
        combine_audio_files(audio_dir, out, transcript_path, prefix="clean_") is True
    )
    assert os.path.exists(out)

    expected = b"\x0a\x00" * 250 + b"\x0b\x00" * 250
    assert _read_wav_bytes(out) == expected


def test_prefixed_and_default_outputs_differ(tmp_path):
    audio_dir, transcript_path = _setup_dir(tmp_path)
    out1 = str(tmp_path / "out1.wav")
    out2 = str(tmp_path / "out2.wav")

    combine_audio_files(audio_dir, out1, transcript_path, prefix="")
    combine_audio_files(audio_dir, out2, transcript_path, prefix="clean_")

    assert _read_wav_bytes(out1) != _read_wav_bytes(out2)


def test_backcompat_positional_args_only(tmp_path):
    audio_dir, transcript_path = _setup_dir(tmp_path)
    out = str(tmp_path / "out_positional.wav")

    # No prefix passed at all -> default unchanged behavior.
    assert combine_audio_files(audio_dir, out, transcript_path) is True
    expected = b"\x01\x00" * 100 + b"\x02\x00" * 100
    assert _read_wav_bytes(out) == expected
