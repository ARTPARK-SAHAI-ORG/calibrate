"""Async sliding-window rate limiters keyed by provider+endpoint.

Used to keep outbound provider API calls under per-account RPM caps. A
limiter tracks request timestamps in a deque; ``acquire()`` evicts entries
older than ``period`` and sleeps until the oldest in-window timestamp ages
out when the window is full.

Sarvam streaming limits (per account, sourced from the Sarvam dashboard):

- STT streaming (``speech_to_text_streaming.connect``): 20 RPM
- TTS streaming (``text_to_speech_streaming.connect``): 60 RPM
"""

from __future__ import annotations

import asyncio
import time
from collections import deque


class AsyncRateLimiter:
    """Sliding-window async rate limiter.

    ``acquire()`` returns once the call can proceed without exceeding
    ``max_calls`` over ``period`` seconds. Safe under concurrent awaiters —
    a single ``asyncio.Lock`` serializes the window check, so two coroutines
    cannot both observe a free slot and overshoot the cap.
    """

    def __init__(self, max_calls: int, period: float = 60.0):
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if period <= 0:
            raise ValueError("period must be positive")
        self.max_calls = max_calls
        self.period = period
        self._calls: deque[float] = deque()
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self) -> asyncio.Lock:
        # ``asyncio.Lock`` binds to whatever loop first awaits it. Module-level
        # limiters outlive any single ``asyncio.run(...)``, so we lazily
        # rebuild the lock whenever the running loop changes — otherwise a
        # second ``asyncio.run`` would inherit a lock bound to a closed loop
        # and raise ``RuntimeError: ... is bound to a different event loop``.
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def acquire(self) -> None:
        async with self._get_lock():
            now = time.monotonic()
            self._evict(now)

            if len(self._calls) >= self.max_calls:
                wait_time = self.period - (now - self._calls[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self._evict(time.monotonic())

            self._calls.append(time.monotonic())

    def _evict(self, now: float) -> None:
        cutoff = now - self.period
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()


SARVAM_STT_STREAMING_LIMITER = AsyncRateLimiter(max_calls=20, period=60.0)
SARVAM_TTS_STREAMING_LIMITER = AsyncRateLimiter(max_calls=60, period=60.0)
