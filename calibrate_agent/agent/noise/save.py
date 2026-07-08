"""Save clean and noisy conversation audio for noisy voice simulations.

When background noise is enabled, the tested agent hears a continuous noise
track under the caller. To reflect that faithfully, the unprefixed
``conversation.wav`` is reconstructed as the NOISY (agent-heard) audio by
mixing a looping noise track over the full clean timeline, while clean
``clean_``-prefixed copies are kept as the reference.
"""

import glob
import os
import shutil
import tempfile

import numpy as np
import soundfile as sf


def mix_noise_over_wav(
    clean_wav: str,
    noise_track: str,
    out_wav: str,
    volume: float,
    sample_rate: int = 16000,
) -> str:
    """Mix a looping noise track over a clean WAV.

    Reads ``clean_wav`` and ``noise_track`` (both 16k mono 16-bit), tiles/loops
    the noise to the clean length, computes ``clip(clean_int16 + noise_int16 *
    volume)`` as int16, and writes ``out_wav`` (16k mono 16-bit). Matches
    pipecat SoundfileMixer's ``clip(audio + sound * volume)``.
    """
    clean, _ = sf.read(clean_wav, dtype="int16")
    noise, _ = sf.read(noise_track, dtype="int16")

    clean = np.asarray(clean, dtype=np.int16).reshape(-1)
    noise = np.asarray(noise, dtype=np.int16).reshape(-1)

    n = clean.shape[0]

    if noise.shape[0] == 0:
        tiled = np.zeros(n, dtype=np.int16)
    else:
        reps = int(np.ceil(n / noise.shape[0]))
        tiled = np.tile(noise, reps)[:n]

    mixed = clean.astype(np.float64) + tiled.astype(np.float64) * float(volume)
    mixed = np.clip(np.round(mixed), -32768, 32767).astype(np.int16)

    sf.write(out_wav, mixed, sample_rate, subtype="PCM_16")
    return out_wav


def write_clean_and_noisy(
    *,
    audio_dir: str,
    transcript_path: str,
    conversation_path: str,
    noise_track_path: str,
    volume: float,
    sample_rate: int = 16000,
) -> None:
    """Write clean per-turn copies and reconstruct the noisy conversation.

    1. Copy every per-turn ``{N}_user.wav`` / ``{N}_bot.wav`` in ``audio_dir``
       to ``clean_{N}_user.wav`` / ``clean_{N}_bot.wav``.
    2. Build ``clean_conversation.wav`` next to ``conversation_path`` from the
       clean-prefixed per-turn files.
    3. Build the clean full timeline, then mix the looping noise over it so the
       unprefixed ``conversation.wav`` is the noisy one.
    """
    from calibrate_agent.utils import combine_audio_files

    turn_files = [
        p
        for p in glob.glob(os.path.join(audio_dir, "*.wav"))
        if _is_turn_file(os.path.basename(p))
    ]

    if not turn_files:
        return

    for src in turn_files:
        name = os.path.basename(src)
        dst = os.path.join(audio_dir, f"clean_{name}")
        shutil.copyfile(src, dst)

    out_dir = os.path.dirname(conversation_path)
    clean_conv_path = os.path.join(out_dir, "clean_conversation.wav")
    combine_audio_files(
        audio_dir, clean_conv_path, transcript_path, prefix="clean_"
    )

    tmp_fd, tmp_clean = tempfile.mkstemp(suffix=".wav", dir=out_dir or None)
    os.close(tmp_fd)
    try:
        combine_audio_files(audio_dir, tmp_clean, transcript_path, prefix="")
        mix_noise_over_wav(
            tmp_clean,
            noise_track_path,
            conversation_path,
            volume,
            sample_rate=sample_rate,
        )
    finally:
        if os.path.exists(tmp_clean):
            os.remove(tmp_clean)


def _is_turn_file(name: str) -> bool:
    if name.startswith("clean_"):
        return False
    stem, ext = os.path.splitext(name)
    if ext.lower() != ".wav":
        return False
    parts = stem.split("_")
    if len(parts) != 2:
        return False
    idx, role = parts
    return idx.isdigit() and role in ("user", "bot")
