"""Download the raw noise clips and build the bundled asset directory.

Sources:
- Environmental sounds: ESC-50 (`ashraq/esc50` on the Hugging Face Hub, no auth).
  One clip per steady sound, a small bank for the scattered events (dog, car_horn).
- Speaker chatter: the gated ARTPARK-IISc/Vaani dataset (needs a Hugging Face
  login). Indian-language clips per simulation language, used as background voices.

``ensure_assets`` builds whatever a requested noise condition needs on first use,
so a fresh install downloads once instead of running with silent noise.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from calibrate_agent.agent.noise.assets import NoiseAssets, prepare_assets

# ESC-50 category -> how many clips to pull (1 steady loop; a bank of 8 for events).
ENV_SINGLE = [
    "rain",
    "wind",
    "engine",
    "vacuum_cleaner",
    "train",
    "siren",
    "crying_baby",
    "footsteps",
    "keyboard_typing",
    "laughing",
]
ENV_EVENTS = {"dog": 8, "car_horn": 8}

# Vaani speaker clips. The dataset is organised into one config per language;
# map each simulation language to its config and how many clips to pull.
VAANI_LANGUAGES = {
    "english": ("English", [("English", 25)]),
    "hindi": ("Hindi", [("Hindi", 40)]),
    "kannada": ("Kannada", [("Kannada", 25)]),
}

ESC50_DATASET = "ashraq/esc50"
VAANI_DATASET = "ARTPARK-IISc/Vaani"


def _decode_audio(audio: dict) -> tuple[np.ndarray, int]:
    """Decode an undecoded HF audio cell (``{'bytes':..., 'path':...}``)."""
    if audio.get("bytes"):
        arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    else:
        arr, sr = sf.read(audio["path"], dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return np.asarray(arr, dtype=np.float32), sr


def _write_wav(dest: Path, audio: np.ndarray, sr: int, overwrite: bool) -> bool:
    if dest.exists() and not overwrite:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")
    return True


def _stream_undecoded(dataset: str, config: str | None):
    """Stream a dataset with the audio column left as raw bytes (no torchcodec)."""
    from datasets import Audio, load_dataset

    ds = load_dataset(dataset, config, split="train", streaming=True)
    return ds.cast_column("audio", Audio(decode=False))


def check_speaker_access() -> str | None:
    """Verify Hugging Face login and access to the gated Vaani dataset.

    Returns a message explaining what to fix, or None when setup is good.
    """
    from huggingface_hub import HfApi, get_token

    if get_token() is None:
        return (
            f"Not logged in to Hugging Face, which the {VAANI_DATASET} speaker "
            "clips require.\n"
            "    Fix: run `huggingface-cli login` (or set HF_TOKEN),\n"
            "    or use a noise condition without background people."
        )
    try:
        HfApi().dataset_info(VAANI_DATASET)
    except Exception as exc:
        return (
            f"Logged in, but cannot access the gated dataset {VAANI_DATASET} "
            f"({type(exc).__name__}: {exc}).\n"
            f"    Fix: accept its terms at https://huggingface.co/datasets/{VAANI_DATASET}."
        )
    return None


def fetch_env(dest: Path, overwrite: bool = False) -> None:
    """Pull the ESC-50 environmental clips into ``dest``."""
    dest = Path(dest)
    wanted = {c: 1 for c in ENV_SINGLE}
    wanted.update(ENV_EVENTS)
    remaining = dict(wanted)

    print(f"[env] streaming {ESC50_DATASET} …")
    for row in _stream_undecoded(ESC50_DATASET, None):
        if not any(remaining.values()):
            break
        cat = row["category"]
        if remaining.get(cat, 0) <= 0:
            continue
        arr, sr = _decode_audio(row["audio"])
        idx = wanted[cat] - remaining[cat]
        if cat in ENV_EVENTS:
            out = dest / cat / f"{idx:02d}.wav"
        else:
            out = dest / f"{cat}.wav"
        if _write_wav(out, arr, sr, overwrite):
            print(f"[env]   {out}")
        remaining[cat] -= 1

    missing = [c for c, n in remaining.items() if n > 0]
    if missing:
        print(f"[env] WARNING: could not fill {missing} from {ESC50_DATASET}")


def fetch_speakers(languages: list[str], dest: Path, overwrite: bool = False) -> None:
    """Pull Vaani speaker clips per language into ``dest/<language>``."""
    dest = Path(dest)
    for lang in languages:
        target_lang, configs = VAANI_LANGUAGES[lang]
        out_dir = dest / lang
        written = 0
        for config, count in configs:
            print(f"[spk] streaming {VAANI_DATASET}:{config} for {target_lang} …")
            got = 0
            for row in _stream_undecoded(VAANI_DATASET, config):
                if got >= count:
                    break
                # The config already selects the language; only skip a row that
                # carries a language field and disagrees with it.
                lang_val = row.get("language")
                if lang_val is not None and str(lang_val).strip().lower() != target_lang.lower():
                    continue
                arr, sr = _decode_audio(row["audio"])
                out = out_dir / f"{config}_{written:03d}.wav"
                if _write_wav(out, arr, sr, overwrite):
                    print(f"[spk]   {out}")
                written += 1
                got += 1
            if got < count:
                print(f"[spk] WARNING: {config} yielded {got}/{count} {target_lang} clips")


def ensure_assets(
    assets: NoiseAssets,
    *,
    env_names: list[str],
    need_speakers: bool,
    language: str,
) -> None:
    """Build any noise assets a condition needs that are not already present.

    Downloads once into the bundle. Raises RuntimeError with a clear fix when
    background people are needed but Hugging Face access is missing, or when a
    download leaves a required asset absent.
    """
    missing_env = [n for n in env_names if not assets.env_ingredient_paths(n)]
    missing_speakers = need_speakers and not assets.speaker_pool(language)
    if not missing_env and not missing_speakers:
        return

    if missing_speakers and language not in VAANI_LANGUAGES:
        raise RuntimeError(
            f"No background-speaker clips for language '{language}'. "
            f"Supported: {list(VAANI_LANGUAGES)}."
        )
    if missing_speakers:
        problem = check_speaker_access()
        if problem:
            raise RuntimeError(f"Cannot build background-speaker noise.\n    {problem}")

    print("[noise] assets not found, downloading once (one-time setup) …")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_env = Path(tmp) / "env_raw"
        tmp_spk = Path(tmp) / "vaani_raw"
        if missing_env:
            fetch_env(tmp_env)
        if missing_speakers:
            fetch_speakers([language], tmp_spk)
        prepare_assets(
            src_env=str(tmp_env), src_speakers=str(tmp_spk), dest=str(assets.root)
        )

    still_env = [n for n in env_names if not assets.env_ingredient_paths(n)]
    still_speakers = need_speakers and not assets.speaker_pool(language)
    if still_env or still_speakers:
        missing = still_env + ([f"{language} speakers"] if still_speakers else [])
        raise RuntimeError(
            f"Noise assets still missing after download: {missing}. "
            "Check your network and Hugging Face access."
        )
