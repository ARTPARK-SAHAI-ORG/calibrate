"""Coverage for tts/benchmark.py — benchmark-parallel resolution."""

import asyncio
import unittest
from unittest.mock import patch, AsyncMock


class TestResolveBenchmarkParallel(unittest.TestCase):
    def test_env_var_sets_parallelism(self):
        from calibrate.utils import resolve_benchmark_parallel

        with patch.dict("os.environ", {"CALIBRATE_TTS_BENCHMARK_PARALLEL": "5"}):
            self.assertEqual(resolve_benchmark_parallel("tts", None), 5)

    def test_explicit_value_overrides_env(self):
        from calibrate.utils import resolve_benchmark_parallel

        with patch.dict("os.environ", {"CALIBRATE_TTS_BENCHMARK_PARALLEL": "5"}):
            self.assertEqual(resolve_benchmark_parallel("tts", 3), 3)

    def test_invalid_env_falls_back_to_default(self):
        from calibrate.utils import resolve_benchmark_parallel, DEFAULT_BENCHMARK_PARALLEL

        with patch.dict("os.environ", {"CALIBRATE_TTS_BENCHMARK_PARALLEL": "abc"}):
            self.assertEqual(
                resolve_benchmark_parallel("tts", None), DEFAULT_BENCHMARK_PARALLEL
            )
        self.assertEqual(DEFAULT_BENCHMARK_PARALLEL, 2)

    def test_zero_env_falls_back_to_default(self):
        from calibrate.utils import resolve_benchmark_parallel, DEFAULT_BENCHMARK_PARALLEL

        with patch.dict("os.environ", {"CALIBRATE_TTS_BENCHMARK_PARALLEL": "0"}):
            self.assertEqual(
                resolve_benchmark_parallel("tts", None), DEFAULT_BENCHMARK_PARALLEL
            )


class TestRunBenchmarkParallel(unittest.IsolatedAsyncioTestCase):
    async def _run_and_measure_peak(self, env, max_parallel):
        """Run the benchmark with a fake provider eval and return peak concurrency."""
        from calibrate.tts import benchmark as B

        concurrency = {"current": 0, "peak": 0}
        gate = asyncio.Event()
        n_providers = 6
        started = {"count": 0}

        async def fake_eval(**kwargs):
            concurrency["current"] += 1
            concurrency["peak"] = max(concurrency["peak"], concurrency["current"])
            started["count"] += 1
            # Once all providers that *can* run concurrently have started, release.
            if started["count"] >= n_providers:
                gate.set()
            try:
                await asyncio.wait_for(gate.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            concurrency["current"] -= 1
            return {"status": "ok", "metrics": {}}

        providers = [f"p{i}" for i in range(n_providers)]
        with patch.dict("os.environ", env, clear=False):
            import os

            if not env:
                os.environ.pop("CALIBRATE_TTS_BENCHMARK_PARALLEL", None)
            with patch.object(
                B, "run_single_provider_eval", AsyncMock(side_effect=fake_eval)
            ), patch.object(B, "generate_leaderboard", lambda **k: None):
                await B.run(
                    input="x.csv",
                    providers=providers,
                    language="english",
                    output_dir="./out",
                    max_parallel=max_parallel,
                )
        return concurrency["peak"]

    async def test_env_var_bounds_concurrency(self):
        peak = await self._run_and_measure_peak(
            {"CALIBRATE_TTS_BENCHMARK_PARALLEL": "3"}, None
        )
        self.assertEqual(peak, 3)

    async def test_explicit_max_parallel_overrides_env(self):
        peak = await self._run_and_measure_peak(
            {"CALIBRATE_TTS_BENCHMARK_PARALLEL": "3"}, 2
        )
        self.assertEqual(peak, 2)

    async def test_default_when_env_unset(self):
        peak = await self._run_and_measure_peak({}, None)
        self.assertEqual(peak, 2)


if __name__ == "__main__":
    unittest.main()
