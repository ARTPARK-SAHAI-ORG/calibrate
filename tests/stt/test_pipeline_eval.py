"""Integration tests for the pipecat-pipeline STT engine (stt/pipeline_eval.py).

These run a REAL pipecat pipeline on the pinned pipecat 1.0.0 — no provider API
keys and no Silero model needed. A deterministic RMS-based fake ``VADAnalyzer``
drives the pipecat VAD state machine, and a minimal in-pipeline fake
``STTService`` emits a finalized ``TranscriptionFrame`` at end-of-speech — which
exercises the base STTService's real TTFB machinery, so we assert both transcript
collection and TTFS latency harvesting work end-to-end.

Run with:
    python -m unittest tests.stt.test_pipeline_eval -v
"""

import asyncio
import math
import struct
import unittest

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService

from calibrate_agent.stt.pipeline_eval import (
    SAMPLE_RATE,
    SyntheticAudioInputTransport,
    transcribe_via_pipeline,
    _STTOutputCollector,
)

# Short silence windows keep the real-time-paced tests fast.
FAST = dict(max_silence_seconds=0.6, post_transcription_silence_seconds=0.2)


def _tone_pcm(seconds: float, freq: float = 220.0) -> bytes:
    n = int(SAMPLE_RATE * seconds)
    out = bytearray()
    for i in range(n):
        val = int(0.6 * 32767 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        out += struct.pack("<h", val)
    return bytes(out)


class _RMSVADAnalyzer(VADAnalyzer):
    """Deterministic VAD: loud frames -> speech, silence -> quiet. No model.

    Implements only the two abstract hooks; the base class runs the real
    STARTING/SPEAKING/STOPPING/QUIET state machine, so this genuinely exercises
    the same VAD path Silero would, just with a trivial confidence function.
    """

    def __init__(self):
        super().__init__(
            params=VADParams(
                confidence=0.5, start_secs=0.05, stop_secs=0.1, min_volume=0.0
            )
        )

    def num_frames_required(self) -> int:
        return int(SAMPLE_RATE * 0.02)  # 20 ms frames

    def voice_confidence(self, buffer: bytes) -> float:
        if not buffer:
            return 0.0
        count = len(buffer) // 2
        samples = struct.unpack(f"<{count}h", buffer[: count * 2])
        rms = math.sqrt(sum(s * s for s in samples) / count) / 32768.0
        return 1.0 if rms > 0.05 else 0.0


class _FakeSTTService(STTService):
    """Emits a fixed transcript when VAD reports the user stopped speaking."""

    def __init__(self, text: str = "hello world", **kwargs):
        super().__init__(stt_ttfb_timeout=0.3, **kwargs)
        self._text = text

    def can_generate_metrics(self) -> bool:
        # Real provider services (Deepgram, Cartesia, ...) override this to True;
        # required for the base STTService to emit TTFB metrics.
        return True

    async def run_stt(self, audio: bytes):
        return
        yield  # pragma: no cover - makes this an async generator

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.push_frame(
                TranscriptionFrame(
                    user_id="", text=self._text, timestamp="", finalized=True
                )
            )


class TestPipelineEngine(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_produces_transcript_and_ttfs(self):
        result = await transcribe_via_pipeline(
            _tone_pcm(seconds=0.6),
            _FakeSTTService(text="hello world"),
            vad_analyzer=_RMSVADAnalyzer(),
            **FAST,
        )
        self.assertEqual(result["transcript"], "hello world")
        # VAD fired -> STTService opened+closed its TTFB window -> latency harvested.
        self.assertIsNotNone(result["ttfs"])
        self.assertGreaterEqual(result["ttfs"], 0.0)
        self.assertLess(result["ttfs"], 5.0)

    async def test_pipeline_empty_on_pure_silence(self):
        """Silence never trips VAD, so no transcript and no TTFS (not a crash)."""
        result = await transcribe_via_pipeline(
            bytes(int(SAMPLE_RATE * 0.4) * 2),  # 0.4s of 16-bit silence
            _FakeSTTService(text="should not appear"),
            vad_analyzer=_RMSVADAnalyzer(),
            **FAST,
        )
        self.assertEqual(result["transcript"], "")
        self.assertIsNone(result["ttfs"])


class TestOutputCollector(unittest.IsolatedAsyncioTestCase):
    async def test_collector_concatenates_segments(self):
        ev = asyncio.Event()
        collector = _STTOutputCollector(ev)
        stt = _FakeSTTService()

        async def push(text):
            await collector.on_push_frame(
                FramePushed(
                    source=stt,
                    destination=stt,
                    frame=TranscriptionFrame(user_id="", text=text, timestamp=""),
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )

        await push("part one")
        await push("part two")
        self.assertEqual(collector.transcript, "part one part two")
        self.assertTrue(ev.is_set())

    async def test_collector_ignores_non_stt_sources(self):
        """Frames not sourced from an STTService are ignored."""
        ev = asyncio.Event()
        collector = _STTOutputCollector(ev)

        class _NotStt:
            pass

        await collector.on_push_frame(
            FramePushed(
                source=_NotStt(),
                destination=_NotStt(),
                frame=TranscriptionFrame(user_id="", text="ignore me", timestamp=""),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        self.assertEqual(collector.transcript, "")
        self.assertFalse(ev.is_set())


class TestTransport(unittest.TestCase):
    def test_transport_chunk_size(self):
        """20 ms at 16 kHz 16-bit = 640 bytes/chunk."""
        t = SyntheticAudioInputTransport(b"", asyncio.Event())
        self.assertEqual(t._chunk_size, 640)


if __name__ == "__main__":
    unittest.main()
