"""Pipecat-pipeline STT engine for the benchmark.

This is the "real agent pipeline" transcription path: instead of calling each
provider's SDK directly (see ``transcribe_*`` in ``stt/eval.py``), a WAV is
streamed **at real-time pace** through a minimal pipecat pipeline

    [SyntheticAudioInputTransport] -> [VADProcessor(Silero)] -> [STTService]

using the exact same ``create_stt_service`` factory the live voice agent runs
(``calibrate_agent/utils.py``). This makes the benchmark test the STT config we
actually deploy, and lets us harvest pipecat's own speech-stop -> final-transcript
latency (``TTFBMetricsData``, branded "TTFS") the same way pipecat's published
per-service P99 numbers are measured (https://github.com/pipecat-ai/stt-benchmark).

Mirrors pipecat's stt-benchmark methodology on our pinned pipecat 1.0.0:
- audio is pushed as 20 ms ``InputAudioRawFrame`` chunks with a real-time sleep
  between them, then a trailing-silence tail so VAD fires ``UserStoppedSpeaking``
  and streaming services flush their final segment;
- Silero VAD (``stop_secs=0.2``) matches the live agent (``agent/bot.py``);
- on ``VADUserStoppedSpeakingFrame`` the pipecat ``STTService`` starts its TTFB
  timer at the corrected speech-end time and stops it on the finalized
  ``TranscriptionFrame`` (or after ``stt_ttfb_timeout``), emitting a
  ``MetricsFrame(TTFBMetricsData)`` we collect.
"""

import asyncio
import time
from typing import Optional

from pipecat.frames.frames import (
    InputAudioRawFrame,
    MetricsFrame,
    TranscriptionFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.services.stt_service import STTService

# 16 kHz, 16-bit mono PCM — matches the live agent's audio_in_sample_rate and
# what every ``create_stt_service`` provider is configured for.
SAMPLE_RATE = 16000
# Match the live agent's VAD (agent/bot.py). stop_secs MUST be set or the
# pipecat STTService skips TTFB measurement entirely.
VAD_STOP_SECS = 0.2
# 20 ms audio chunks, like pipecat's stt-benchmark synthetic transport.
CHUNK_MS = 20
# Leading silence streamed BEFORE the real audio. A streaming STT service (e.g.
# Deepgram) opens its websocket on the pipeline StartFrame; if we start pushing
# speech immediately, the first ~0.5s can be dropped before the socket is ready,
# truncating the opening words. Feeding silence first absorbs that connect time
# (and lets Silero VAD warm up), just like a real mic has a moment of room tone
# before the user speaks. TTFS is unaffected — it's anchored at speech-stop.
LEAD_SILENCE_SECONDS = 1.0
# How long to keep feeding trailing silence waiting for a transcript before
# giving up, and how much extra silence to send after the first transcript to
# let streaming services flush late segments.
MAX_SILENCE_SECONDS = 10.0
POST_TRANSCRIPTION_SILENCE_SECONDS = 2.0
# Overall guard so a wedged provider can't hang a benchmark run forever.
PIPELINE_RUN_TIMEOUT_SECONDS = 120.0


class SyntheticAudioInputTransport(BaseInputTransport):
    """Feeds a pre-loaded PCM buffer into a pipeline at real-time (1x) pace.

    Emulates a live microphone: pushes ``InputAudioRawFrame``s in ``CHUNK_MS``
    steps sleeping one chunk-duration between them, then a trailing-silence tail
    (so VAD detects end-of-speech and streaming STT flushes) until
    ``transcription_received`` fires or ``MAX_SILENCE_SECONDS`` elapses.
    """

    def __init__(
        self,
        pcm: bytes,
        transcription_received: asyncio.Event,
        sample_rate: int = SAMPLE_RATE,
        chunk_ms: int = CHUNK_MS,
        max_silence_seconds: float = MAX_SILENCE_SECONDS,
        post_transcription_silence_seconds: float = POST_TRANSCRIPTION_SILENCE_SECONDS,
        lead_silence_seconds: float = LEAD_SILENCE_SECONDS,
    ):
        super().__init__(
            TransportParams(
                audio_in_enabled=True,
                audio_in_passthrough=True,
                audio_in_sample_rate=sample_rate,
            )
        )
        self._pcm = pcm
        self._transcription_received = transcription_received
        self._chunk_sample_rate = sample_rate
        self._chunk_ms = chunk_ms
        # 2 bytes/sample (16-bit).
        self._chunk_size = int(sample_rate * chunk_ms / 1000) * 2
        self._max_silence_seconds = max_silence_seconds
        self._post_transcription_silence_seconds = post_transcription_silence_seconds
        self._lead_silence_seconds = lead_silence_seconds
        self._audio_complete = asyncio.Event()
        self._pump_task: Optional[asyncio.Task] = None

    async def start(self, frame):
        await super().start(frame)
        # Real transports call this from their connect handler; a synthetic
        # source has no connection, so mark ready as soon as the pipeline starts.
        await self.set_transport_ready(frame)

    async def set_transport_ready(self, frame):
        await super().set_transport_ready(frame)
        if self._pump_task is None:
            self._pump_task = self.create_task(self._pump_audio())

    async def stop(self, frame):
        await self._cancel_pump_task()
        await super().stop(frame)

    async def cancel(self, frame):
        await self._cancel_pump_task()
        await super().cancel(frame)

    async def _cancel_pump_task(self):
        if self._pump_task is not None:
            await self.cancel_task(self._pump_task)
            self._pump_task = None

    async def wait_for_audio_complete(self, timeout: float) -> None:
        """Wait until the real (non-silence) audio has been fully streamed."""
        try:
            await asyncio.wait_for(self._audio_complete.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    def _make_frame(self, chunk: bytes) -> InputAudioRawFrame:
        return InputAudioRawFrame(
            audio=chunk, sample_rate=self._chunk_sample_rate, num_channels=1
        )

    async def _pump_audio(self):
        sleep_time = self._chunk_ms / 1000.0
        silence = bytes(self._chunk_size)

        # Lead-in silence: let the STT service's websocket connect (and VAD warm
        # up) before the first real speech, so the opening words aren't dropped.
        for _ in range(int(self._lead_silence_seconds / sleep_time)):
            await self.push_audio_frame(self._make_frame(silence))
            await asyncio.sleep(sleep_time)

        # Stream the real audio at 1x wall-clock.
        for offset in range(0, len(self._pcm), self._chunk_size):
            chunk = self._pcm[offset : offset + self._chunk_size]
            if chunk:
                await self.push_audio_frame(self._make_frame(chunk))
                await asyncio.sleep(sleep_time)

        self._audio_complete.set()

        # Trailing silence: keep VAD/STT fed until we have a transcript (plus a
        # short post-transcript window for late streaming segments).
        silence_start = time.monotonic()
        post_deadline: Optional[float] = None
        while True:
            now = time.monotonic()
            if now - silence_start >= self._max_silence_seconds:
                break
            if self._transcription_received.is_set():
                if post_deadline is None:
                    post_deadline = now + self._post_transcription_silence_seconds
                elif now >= post_deadline:
                    break
            await self.push_audio_frame(self._make_frame(silence))
            await asyncio.sleep(sleep_time)


class _STTOutputCollector(BaseObserver):
    """Harvests final transcript text and TTFS latency from the STT service.

    Passive: reads frames the STT service pushes. Concatenates
    ``TranscriptionFrame`` text (services may segment long audio into several)
    and captures the ``TTFBMetricsData`` value pipecat's STTService emits for the
    speech-stop -> final-transcript interval. Sets ``transcription_received`` on
    the first non-empty transcript so the transport can wind down its silence tail.
    """

    def __init__(self, transcription_received: asyncio.Event):
        super().__init__()
        self._transcription_received = transcription_received
        self._parts: list[str] = []
        self.ttfs: Optional[float] = None

    async def on_push_frame(self, data: FramePushed):
        if not isinstance(data.source, STTService):
            return
        frame = data.frame
        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if text:
                self._parts.append(text)
                self._transcription_received.set()
        elif isinstance(frame, MetricsFrame):
            for item in getattr(frame, "data", []) or []:
                if isinstance(item, TTFBMetricsData) and item.value:
                    # Last non-zero wins (final-segment latency).
                    self.ttfs = float(item.value)

    @property
    def transcript(self) -> str:
        return " ".join(self._parts).strip()


def _default_vad_analyzer():
    """Silero VAD tuned to match the live agent (agent/bot.py)."""
    return SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS))


async def transcribe_via_pipeline(
    audio_pcm: bytes,
    stt_service: STTService,
    run_timeout: float = PIPELINE_RUN_TIMEOUT_SECONDS,
    vad_analyzer=None,
    max_silence_seconds: float = MAX_SILENCE_SECONDS,
    post_transcription_silence_seconds: float = POST_TRANSCRIPTION_SILENCE_SECONDS,
) -> dict:
    """Transcribe one clip by streaming it through a real pipecat pipeline.

    Args:
        audio_pcm: 16 kHz, 16-bit mono PCM bytes (e.g. ``load_audio(path, raw_pcm=True)``).
        stt_service: A pipecat ``STTService`` (from ``create_stt_service``). One
            fresh service per call — pipecat services are single-pipeline-lifecycle.
        run_timeout: Hard cap on the whole run.
        vad_analyzer: VAD analyzer to use. Defaults to Silero (``stop_secs=0.2``),
            matching the live agent. Injectable so tests can drive VAD
            deterministically without loading the Silero model.
        max_silence_seconds: How long to feed trailing silence waiting for a
            transcript before giving up.
        post_transcription_silence_seconds: Extra silence after the first
            transcript to let streaming services flush late segments.

    Returns:
        ``{"transcript": str, "ttfs": float | None}`` — ``ttfs`` is the
        speech-stop -> final-transcript latency in seconds (None if the provider
        emitted no TTFB metric, e.g. an empty/silent clip).
    """
    transcription_received = asyncio.Event()
    transport = SyntheticAudioInputTransport(
        audio_pcm,
        transcription_received,
        max_silence_seconds=max_silence_seconds,
        post_transcription_silence_seconds=post_transcription_silence_seconds,
    )
    vad = VADProcessor(vad_analyzer=vad_analyzer or _default_vad_analyzer())
    collector = _STTOutputCollector(transcription_received)

    pipeline = Pipeline([transport, vad, stt_service])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
        ),
        observers=[collector],
    )
    runner = PipelineRunner(handle_sigint=False)
    run_task = asyncio.create_task(runner.run(task))

    try:
        # Wait for the real audio to finish streaming, then for a transcript
        # (bounded by the transport's own silence-tail window).
        await transport.wait_for_audio_complete(timeout=run_timeout)
        try:
            await asyncio.wait_for(
                transcription_received.wait(),
                timeout=max_silence_seconds + post_transcription_silence_seconds + 1.0,
            )
        except asyncio.TimeoutError:
            pass
        # Give the post-transcription silence window time to collect the TTFB
        # metric and any late segments, then end the pipeline.
        await asyncio.sleep(post_transcription_silence_seconds + 0.5)
    finally:
        # Graceful drain (queues an EndFrame that propagates through the
        # transport, cancelling its audio pump), then hard-cancel on timeout.
        await task.stop_when_done()
        try:
            await asyncio.wait_for(run_task, timeout=run_timeout)
        except asyncio.TimeoutError:
            await task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

    return {"transcript": collector.transcript, "ttfs": collector.ttfs}
