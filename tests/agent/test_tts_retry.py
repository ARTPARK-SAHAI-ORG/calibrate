"""Tests for wrap_tts_with_empty_response_retry in calibrate_agent/utils.py.

The wrapper guards against intermittent silent-TTS turns (e.g. ElevenLabs
returning an HTTP 200 with an empty audio stream), which otherwise stall the
voice-agent simulator until the pipeline idle timeout. These tests mock a TTS
service whose run_tts yields a scripted sequence of frames per attempt and
assert the retry/recover/surface-error behavior.
"""

import unittest
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import TTSAudioRawFrame, ErrorFrame

from calibrate_agent.utils import wrap_tts_with_empty_response_retry


def _audio():
    return TTSAudioRawFrame(b"\x00\x00", 16000, 1)


class FakeTTSService:
    """A TTS service whose run_tts replays one scripted frame-list per attempt.

    ``scripts`` is a list of per-attempt frame lists. A frame value of ``None``
    is yielded verbatim (the websocket out-of-band sentinel).
    """

    def __init__(self, scripts):
        self._scripts = scripts
        self.attempts = 0

    async def run_tts(self, text, context_id):
        script = self._scripts[self.attempts]
        self.attempts += 1
        for frame in script:
            yield frame


async def _collect(tts, text="hello", context_id="ctx"):
    return [frame async for frame in tts.run_tts(text, context_id)]


class TestWrapTTSWithEmptyResponseRetry(unittest.IsolatedAsyncioTestCase):
    async def test_first_attempt_succeeds_no_retry(self):
        tts = FakeTTSService([[_audio(), _audio()]])
        wrap_tts_with_empty_response_retry(tts)

        with patch("calibrate_agent.utils.asyncio.sleep", new=AsyncMock()) as sleep:
            frames = await _collect(tts)

        self.assertEqual(tts.attempts, 1)
        self.assertEqual(len(frames), 2)
        self.assertTrue(all(isinstance(f, TTSAudioRawFrame) for f in frames))
        sleep.assert_not_awaited()

    async def test_empty_then_audio_retries_and_recovers(self):
        tts = FakeTTSService([[], [_audio()]])
        wrap_tts_with_empty_response_retry(tts, max_attempts=3, backoff_base_secs=0.5)

        with patch("calibrate_agent.utils.asyncio.sleep", new=AsyncMock()) as sleep:
            frames = await _collect(tts)

        self.assertEqual(tts.attempts, 2)
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], TTSAudioRawFrame)
        sleep.assert_awaited_once_with(0.5)

    async def test_all_empty_surfaces_synthetic_error(self):
        tts = FakeTTSService([[], [], []])
        wrap_tts_with_empty_response_retry(tts, max_attempts=3)

        with patch("calibrate_agent.utils.asyncio.sleep", new=AsyncMock()) as sleep:
            frames = await _collect(tts)

        self.assertEqual(tts.attempts, 3)
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], ErrorFrame)
        # backoff between the 3 attempts => 2 sleeps
        self.assertEqual(sleep.await_count, 2)

    async def test_all_empty_surfaces_providers_own_error(self):
        provider_error = ErrorFrame(error="ElevenLabs API error: boom")
        tts = FakeTTSService([[provider_error], [provider_error], [provider_error]])
        wrap_tts_with_empty_response_retry(tts, max_attempts=3)

        with patch("calibrate_agent.utils.asyncio.sleep", new=AsyncMock()):
            frames = await _collect(tts)

        # The provider's own ErrorFrame is surfaced; no synthetic one is added.
        self.assertEqual(len(frames), 1)
        self.assertIs(frames[0], provider_error)

    async def test_error_frame_on_empty_attempt_does_not_leak_on_recovery(self):
        # First attempt yields only an ErrorFrame (still no audio) -> retry.
        tts = FakeTTSService([[ErrorFrame(error="boom")], [_audio()]])
        wrap_tts_with_empty_response_retry(tts)

        with patch("calibrate_agent.utils.asyncio.sleep", new=AsyncMock()):
            frames = await _collect(tts)

        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], TTSAudioRawFrame)

    async def test_none_sentinel_skips_retry(self):
        # Websocket services deliver audio out-of-band and yield None; the
        # wrapper must pass None through and never retry.
        tts = FakeTTSService([[None]])
        wrap_tts_with_empty_response_retry(tts)

        with patch("calibrate_agent.utils.asyncio.sleep", new=AsyncMock()) as sleep:
            frames = await _collect(tts)

        self.assertEqual(tts.attempts, 1)
        self.assertEqual(frames, [None])
        sleep.assert_not_awaited()

    async def test_backoff_is_exponential(self):
        tts = FakeTTSService([[], [], [_audio()]])
        wrap_tts_with_empty_response_retry(tts, max_attempts=3, backoff_base_secs=0.5)

        with patch("calibrate_agent.utils.asyncio.sleep", new=AsyncMock()) as sleep:
            await _collect(tts)

        self.assertEqual([c.args[0] for c in sleep.await_args_list], [0.5, 1.0])

    async def test_returns_same_instance(self):
        tts = FakeTTSService([[_audio()]])
        self.assertIs(wrap_tts_with_empty_response_retry(tts), tts)


if __name__ == "__main__":
    unittest.main()
