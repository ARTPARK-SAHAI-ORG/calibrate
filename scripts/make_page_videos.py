"""Make a voiced video for each docs page from its screenshots.

Usage: python scripts/make_page_videos.py [--out out/videos] docs/<page>.mdx ...

Each screenshot is shown while ElevenLabs reads the page text that comes
before it (back to the previous screenshot). Code blocks, tables and the
"Next steps" section are not read. Needs ELEVENLABS_API_KEY and ffmpeg.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from elevenlabs import VoiceSettings
from elevenlabs.client import AsyncElevenLabs

from calibrate_agent.utils import get_tts_voice

DOCS = Path(__file__).resolve().parents[1] / "docs"
IMG = re.compile(r'<img\s+src="(/images/[^"]+)"[^>]*/>')
PAUSE = 0.7  # seconds of silence after each narration
MODEL = "eleven_v4_turbo"
# ElevenLabs' default steadiness. The benchmark settings (stability 0) make the
# voice drift and add words on long narration.
VOICE = VoiceSettings(stability=0.5, similarity_boost=0.75, style=0.0, use_speaker_boost=True)
FRACTIONS = {
    "1/2": "one half", "1/4": "one quarter", "2/4": "two quarters", "3/4": "three quarters",
    "2/3": "two thirds", "1/5": "one fifth", "3/5": "three fifths", "4/5": "four fifths",
}


def segments(mdx: str) -> list[tuple[str, str]]:
    """(image path, narration) pairs, in page order."""
    body = mdx.split("---", 2)[2].split("## Next steps")[0]
    # Code is not read aloud; a sentence that introduced it points at the page.
    body = re.sub(r":?\s*^[ \t]*```.*?^[ \t]*```[^\n]*$", " (shown on the page).\n", body, flags=re.M | re.S)
    parts = IMG.split(body)
    pairs = [[parts[i], speak(parts[i - 1])] for i in range(1, len(parts), 2)]
    if pairs:
        pairs[-1][1] = f"{pairs[-1][1]} {speak(parts[-1])}".strip()
    return [(img, text) for img, text in pairs]


def speak(md: str) -> str:
    """Page markdown as plain sentences for the voice."""
    lines = [l for l in md.splitlines() if not l.lstrip().startswith("|")]
    text = "\n".join(lines)
    text = re.sub(r"\{/\*.*?\*/\}", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*(#+|[-*]|\d+\.)\s*", "", text, flags=re.M)
    text = text.replace("**", "").replace("`", "")
    # Say what a reader sees, not the symbols: URLs, fractions, provider/model
    # names and code names do not read well aloud.
    text = re.sub(r"https?://\S+", "the address shown on the page", text)
    text = re.sub(r"\b\d+/\d+\b", lambda m: FRACTIONS.get(m[0], m[0].replace("/", " over ")), text)
    text = re.sub(r"(?<=\w)/(?=\w)", " ", text)
    text = text.replace(" + ", " plus ").replace("_", " ")
    # A heading or list item with no full stop still needs a pause when read.
    text = re.sub(r"([^.!?:\s])\s*\n", r"\1.\n", text)
    return re.sub(r"\s+", " ", text).strip()


async def synthesize(text: str, path: Path) -> None:
    client = AsyncElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    stream = client.text_to_speech.stream(
        voice_id=get_tts_voice("elevenlabs", "english"),
        model_id=MODEL,
        output_format="mp3_44100_128",
        text=text,
        voice_settings=VOICE,
    )
    # Write beside the target first, so an interrupted request is never reused as finished audio.
    part = path.with_suffix(".part")
    with open(part, "wb") as f:
        async for chunk in stream:
            f.write(chunk)
    part.rename(path)


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


async def make_video(page: Path, out_dir: Path, cache: Path) -> Path | None:
    pairs = segments(page.read_text())
    if not pairs:
        return None
    name = str(page.relative_to(DOCS).with_suffix("")).replace("/", "-")
    digest = hashlib.sha256(
        (MODEL + repr(VOICE)).encode() + b"".join(
            text.encode() + p.read_bytes()
            for img, text in pairs
            for p in [DOCS / img.lstrip("/"), (DOCS / img.lstrip("/")).with_suffix(".box.json")]
            if p.exists()
        )
    ).hexdigest()[:10]
    target = out_dir / f"{name}.{digest}.mp4"
    if target.exists():
        return target
    with tempfile.TemporaryDirectory() as tmp:
        clips = []
        for i, (img, text) in enumerate(pairs):
            audio = cache / f"{hashlib.sha256(((MODEL + repr(VOICE)) + text).encode()).hexdigest()[:16]}.mp3"
            if not audio.exists():
                await synthesize(text, audio)
            clip = Path(tmp) / f"{i:03}.mp4"
            vf = "format=yuv420p"
            box = (DOCS / img.lstrip("/")).with_suffix(".box.json")
            if box.exists():
                # Outline the element the step points at, saved by the screenshot script.
                b = json.loads(box.read_text())
                vf = f"drawbox=x={b['x'] - 6:.0f}:y={b['y'] - 6:.0f}:w={b['w'] + 12:.0f}:h={b['h'] + 12:.0f}:color=0x22c55e:t=5," + vf
            ffmpeg(
                "-loop", "1", "-i", str(DOCS / img.lstrip("/")), "-i", str(audio),
                "-t", f"{duration(audio) + PAUSE:.2f}",
                "-vf", vf,
                "-af", "apad", "-r", "10", "-c:v", "libx264", "-tune", "stillimage", "-crf", "28",
                "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "24000", str(clip),
            )
            clips.append(clip)
        listing = Path(tmp) / "clips.txt"
        listing.write_text("".join(f"file '{c}'\n" for c in clips))
        ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(target))
    for old in out_dir.glob(f"{name}.*.mp4"):
        if old != target:
            old.unlink()
    return target


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("out/videos"))
    parser.add_argument("pages", nargs="+", type=Path)
    args = parser.parse_args(argv)
    cache = args.out / "audio"
    cache.mkdir(parents=True, exist_ok=True)
    for page in args.pages:
        video = await make_video(page.resolve(), args.out, cache)
        print(f"{page}: {video or 'no screenshots, skipped'}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
