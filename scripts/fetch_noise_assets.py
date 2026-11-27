"""Fetch and build the voice-sim background-noise assets in one shot.

Downloads the raw clips from their public sources, places them under ``data/``,
then bundles them into ``calibrate_agent/agent/assets/noise/`` (the directory the
noise mixer reads from). The download logic lives in
``calibrate_agent.agent.noise.download``, which the simulator also calls to build
assets on first use.

Usage:
    uv run python scripts/fetch_noise_assets.py                 # everything
    uv run python scripts/fetch_noise_assets.py --skip-speakers # env only (no HF login)
    uv run python scripts/fetch_noise_assets.py --languages hindi,english

Idempotent: existing files are left in place unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from calibrate_agent.agent.noise.assets import prepare_assets
from calibrate_agent.agent.noise.download import (
    VAANI_LANGUAGES,
    check_speaker_access,
    fetch_env,
    fetch_speakers,
)

ENV_RAW = Path("data/env_raw")
VAANI_RAW = Path("data/vaani_raw")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--languages",
        default="english,hindi,kannada",
        help="comma-separated speaker languages to pull (default: all three)",
    )
    ap.add_argument("--skip-env", action="store_true", help="skip ESC-50 environmental clips")
    ap.add_argument("--skip-speakers", action="store_true", help="skip Vaani speaker clips (no HF login needed)")
    ap.add_argument("--no-prepare", action="store_true", help="don't bundle the clips at the end")
    ap.add_argument("--overwrite", action="store_true", help="re-download clips that already exist")
    args = ap.parse_args()

    # Verify everything before downloading anything, so a setup gap fails fast
    # instead of part-way through and discarding what was already fetched.
    languages: list[str] = []
    if not args.skip_speakers:
        languages = [l.strip() for l in args.languages.split(",") if l.strip()]
        unknown = [l for l in languages if l not in VAANI_LANGUAGES]
        if unknown:
            ap.error(f"unknown languages: {unknown} (choose from {list(VAANI_LANGUAGES)})")
        problem = check_speaker_access()
        if problem:
            sys.exit(f"✗ Setup incomplete, nothing downloaded:\n    {problem}")

    if not args.skip_env:
        fetch_env(ENV_RAW, args.overwrite)
    if not args.skip_speakers:
        fetch_speakers(languages, VAANI_RAW, args.overwrite)

    if not args.no_prepare:
        print("[build] prepare_assets() → calibrate_agent/agent/assets/noise/")
        prepare_assets(src_env=str(ENV_RAW), src_speakers=str(VAANI_RAW))
    print("[done] noise assets ready.")


if __name__ == "__main__":
    main()
