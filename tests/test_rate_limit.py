import asyncio
import unittest
from unittest.mock import patch

from calibrate.rate_limit import (
    AsyncRateLimiter,
    SARVAM_STT_STREAMING_LIMITER,
    SARVAM_TTS_STREAMING_LIMITER,
)


class TestAsyncRateLimiterValidation(unittest.TestCase):
    def test_rejects_zero_max_calls(self):
        with self.assertRaises(ValueError):
            AsyncRateLimiter(max_calls=0)

    def test_rejects_negative_max_calls(self):
        with self.assertRaises(ValueError):
            AsyncRateLimiter(max_calls=-1)

    def test_rejects_zero_period(self):
        with self.assertRaises(ValueError):
            AsyncRateLimiter(max_calls=5, period=0)

    def test_rejects_negative_period(self):
        with self.assertRaises(ValueError):
            AsyncRateLimiter(max_calls=5, period=-1)


class TestAsyncRateLimiterBehavior(unittest.IsolatedAsyncioTestCase):
    async def test_under_limit_does_not_sleep(self):
        limiter = AsyncRateLimiter(max_calls=3, period=60.0)
        with patch("calibrate.rate_limit.asyncio.sleep") as mock_sleep:
            for _ in range(3):
                await limiter.acquire()
            mock_sleep.assert_not_called()
        self.assertEqual(len(limiter._calls), 3)

    async def test_over_limit_sleeps_for_remaining_window(self):
        # Three timestamps at t=0,1,2 in a 60s window. The 4th call should
        # sleep for ~58s (60 - (now=2 - oldest=0)).
        limiter = AsyncRateLimiter(max_calls=3, period=60.0)

        fake_now = [0.0]

        def mono():
            return fake_now[0]

        sleep_calls: list[float] = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)
            fake_now[0] += seconds

        with patch("calibrate.rate_limit.time.monotonic", side_effect=mono):
            with patch(
                "calibrate.rate_limit.asyncio.sleep", side_effect=fake_sleep
            ):
                await limiter.acquire()
                fake_now[0] = 1.0
                await limiter.acquire()
                fake_now[0] = 2.0
                await limiter.acquire()
                await limiter.acquire()

        self.assertEqual(len(sleep_calls), 1)
        self.assertAlmostEqual(sleep_calls[0], 58.0, places=5)

    async def test_evicts_stale_timestamps(self):
        limiter = AsyncRateLimiter(max_calls=2, period=60.0)

        fake_now = [0.0]

        def mono():
            return fake_now[0]

        with patch("calibrate.rate_limit.time.monotonic", side_effect=mono):
            with patch(
                "calibrate.rate_limit.asyncio.sleep"
            ) as mock_sleep:
                await limiter.acquire()
                fake_now[0] = 1.0
                await limiter.acquire()
                fake_now[0] = 62.0
                await limiter.acquire()
                mock_sleep.assert_not_called()

        self.assertEqual(len(limiter._calls), 1)

    async def test_concurrent_acquires_serialize(self):
        # Five awaiters racing for 2 slots should never observe more than 2
        # in-window timestamps at any point.
        limiter = AsyncRateLimiter(max_calls=2, period=60.0)

        observed_max = [0]

        async def fake_sleep(seconds):
            # Simulate window expiry on sleep
            limiter._calls.clear()

        with patch(
            "calibrate.rate_limit.asyncio.sleep", side_effect=fake_sleep
        ):

            async def worker():
                await limiter.acquire()
                observed_max[0] = max(observed_max[0], len(limiter._calls))

            await asyncio.gather(*(worker() for _ in range(5)))

        self.assertLessEqual(observed_max[0], 2)


class TestAsyncRateLimiterLoopRebind(unittest.TestCase):
    """A module-level limiter must survive multiple ``asyncio.run(...)``
    invocations in the same interpreter. ``asyncio.Lock`` binds to the loop
    that first contends on it, so without rebinding the second run would
    raise ``RuntimeError: ... is bound to a different event loop``.
    """

    def test_survives_multiple_asyncio_run_calls(self):
        limiter = AsyncRateLimiter(max_calls=5, period=60.0)

        async def two_acquires():
            await limiter.acquire()
            await limiter.acquire()

        asyncio.run(two_acquires())
        first_loop = limiter._lock_loop

        asyncio.run(two_acquires())
        second_loop = limiter._lock_loop

        self.assertIsNotNone(first_loop)
        self.assertIsNotNone(second_loop)
        self.assertIsNot(first_loop, second_loop)

    def test_survives_multiple_runs_with_contention(self):
        # ``asyncio.Lock`` calls ``_get_loop()`` only when a waiter has to
        # queue (i.e. the lock is already held). Forcing the limiter into
        # its wait branch and then racing a second acquirer makes the
        # second one queue, which binds the lock to loop A. A second
        # ``asyncio.run`` would then raise without the rebind.
        limiter = AsyncRateLimiter(max_calls=1, period=60.0)
        real_sleep = asyncio.sleep

        async def fake_sleep(_seconds):
            await real_sleep(0)  # actually yield so the second acquirer can queue
            limiter._calls.clear()

        async def race():
            limiter._calls.clear()
            with patch(
                "calibrate.rate_limit.asyncio.sleep", side_effect=fake_sleep
            ):
                await limiter.acquire()  # fills the bucket
                await asyncio.gather(limiter.acquire(), limiter.acquire())

        asyncio.run(race())
        asyncio.run(race())  # would raise without the rebind


class TestSarvamModuleLimiters(unittest.TestCase):
    def test_stt_streaming_is_20_per_minute(self):
        self.assertEqual(SARVAM_STT_STREAMING_LIMITER.max_calls, 20)
        self.assertEqual(SARVAM_STT_STREAMING_LIMITER.period, 60.0)

    def test_tts_streaming_is_60_per_minute(self):
        self.assertEqual(SARVAM_TTS_STREAMING_LIMITER.max_calls, 60)
        self.assertEqual(SARVAM_TTS_STREAMING_LIMITER.period, 60.0)


class TestSarvamCallsAcquire(unittest.IsolatedAsyncioTestCase):
    """The sarvam STT/TTS functions must call ``acquire()`` before opening
    the streaming websocket — otherwise we'd burst past the per-account cap
    on the first batch of rows in a benchmark run.
    """

    async def test_transcribe_sarvam_acquires_before_client(self):
        from pathlib import Path
        from calibrate.stt import eval as stt_eval

        with patch.dict("os.environ", {"SARVAM_API_KEY": "sk-fake"}):
            with patch.object(
                stt_eval.SARVAM_STT_STREAMING_LIMITER, "acquire"
            ) as mock_acquire:
                mock_acquire.return_value = None

                async def _mock_acquire():
                    mock_acquire.called_at = "before_client"

                mock_acquire.side_effect = _mock_acquire

                with patch.object(stt_eval, "load_audio", return_value=b""):
                    with patch.object(
                        stt_eval, "AsyncSarvamAI", side_effect=RuntimeError("stop")
                    ):
                        with self.assertRaises(RuntimeError):
                            await stt_eval.transcribe_sarvam(
                                Path("/tmp/x.wav"), "english"
                            )

                mock_acquire.assert_awaited_once()

    async def test_synthesize_sarvam_acquires_before_client(self):
        from calibrate.tts import eval as tts_eval

        with patch.dict("os.environ", {"SARVAM_API_KEY": "sk-fake"}):
            with patch.object(
                tts_eval.SARVAM_TTS_STREAMING_LIMITER, "acquire"
            ) as mock_acquire:

                async def _mock_acquire():
                    return None

                mock_acquire.side_effect = _mock_acquire

                with patch.object(
                    tts_eval, "AsyncSarvamAI", side_effect=RuntimeError("stop")
                ):
                    with self.assertRaises(RuntimeError):
                        await tts_eval.synthesize_sarvam(
                            "hello", "english", "/tmp/out.wav"
                        )

                mock_acquire.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
