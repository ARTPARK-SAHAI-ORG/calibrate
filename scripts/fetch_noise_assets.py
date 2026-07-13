"""Fetch and build the voice-sim background-noise assets in one shot.

Downloads the raw clips from their public sources, places them under ``data/``,
then runs ``prepare_assets()`` to populate the bundled asset directory the noise
mixer reads from (``calibrate_agent/agent/assets/noise/``).

Two sources:
- **Environmental sounds** — ESC-50 (`ashraq/esc50` on the Hugging Face Hub, no
  auth). One clip per steady sound; a small bank for the scattered events
  (`dog`, `car_horn`).
- **Speaker chatter** — the ARTPARK-IISc/Vaani dataset (gated: set ``HF_TOKEN``
  or run ``huggingface-cli login`` first). Indian-language clips per simulation
  language, used as distant background voices.

Usage:
    uv run python scripts/fetch_noise_assets.py                 # everything
    uv run python scripts/fetch_noise_assets.py --skip-speakers # env only (no HF login)
    uv run python scripts/fetch_noise_assets.py --languages hindi,english

Idempotent: existing files are left in place unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import soundfile as sf

from calibrate_agent.agent.noise.assets import prepare_assets

# --- Environmental sounds (ESC-50 category -> how many clips to pull) ---------
# 1 clip for steady loops; a bank of 8 for the scattered single-shot events.
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

# --- Vaani speaker clips (dataset language -> [(hf config, count), ...]) -------
VAANI_LANGUAGES = {
    "english": ("English", [("Karnataka_Bangalore", 25)]),
    "hindi": ("Hindi", [("Karnataka_Bangalore", 25), ("Delhi_NewDelhi", 15)]),
    "kannada": ("Kannada", [("Karnataka_Bangalore", 25)]),
}

ESC50_DATASET = "ashraq/esc50"
VAANI_DATASET = "ARTPARK-IISc/Vaani"

ENV_RAW = Path("data/env_raw")
VAANI_RAW = Path("data/vaani_raw")


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


def fetch_env(overwrite: bool) -> None:
    """Pull the ESC-50 environmental clips into ``data/env_raw``."""
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
            dest = ENV_RAW / cat / f"{idx:02d}.wav"
        else:
            dest = ENV_RAW / f"{cat}.wav"
        if _write_wav(dest, arr, sr, overwrite):
            print(f"[env]   {dest}")
        remaining[cat] -= 1

    missing = [c for c, n in remaining.items() if n > 0]
    if missing:
        print(f"[env] WARNING: could not fill {missing} from {ESC50_DATASET}")


def fetch_speakers(languages: list[str], overwrite: bool) -> None:
    """Pull Vaani speaker clips per language into ``data/vaani_raw/<lang>``."""
    for lang in languages:
        target_lang, configs = VAANI_LANGUAGES[lang]
        out_dir = VAANI_RAW / lang
        written = 0
        for config, count in configs:
            print(f"[spk] streaming {VAANI_DATASET}:{config} for {target_lang} …")
            got = 0
            for row in _stream_undecoded(VAANI_DATASET, config):
                if got >= count:
                    break
                if str(row.get("language", "")).strip().lower() != target_lang.lower():
                    continue
                arr, sr = _decode_audio(row["audio"])
                dest = out_dir / f"{config}_{written:03d}.wav"
                if _write_wav(dest, arr, sr, overwrite):
                    print(f"[spk]   {dest}")
                written += 1
                got += 1
            if got < count:
                print(f"[spk] WARNING: {config} yielded {got}/{count} {target_lang} clips")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--languages",
        default="english,hindi,kannada",
        help="comma-separated speaker languages to pull (default: all three)",
    )
    ap.add_argument("--skip-env", action="store_true", help="skip ESC-50 environmental clips")
    ap.add_argument("--skip-speakers", action="store_true", help="skip Vaani speaker clips (no HF login needed)")
    ap.add_argument("--no-prepare", action="store_true", help="don't run prepare_assets() at the end")
    ap.add_argument("--overwrite", action="store_true", help="re-download clips that already exist")
    args = ap.parse_args()

    if not args.skip_env:
        fetch_env(args.overwrite)
    if not args.skip_speakers:
        languages = [l.strip() for l in args.languages.split(",") if l.strip()]
        unknown = [l for l in languages if l not in VAANI_LANGUAGES]
        if unknown:
            ap.error(f"unknown languages: {unknown} (choose from {list(VAANI_LANGUAGES)})")
        fetch_speakers(languages, args.overwrite)

    if not args.no_prepare:
        print("[build] prepare_assets() → calibrate_agent/agent/assets/noise/")
        prepare_assets()
    print("[done] noise assets ready.")


if __name__ == "__main__":
    main()
