import os
import unittest
from unittest.mock import patch, AsyncMock

from calibrate.utils import resolve_benchmark_parallel, DEFAULT_BENCHMARK_PARALLEL


class TestResolveBenchmarkParallelStt(unittest.TestCase):
    def test_cli_value_takes_precedence(self):
        with patch.dict("os.environ", {"CALIBRATE_STT_BENCHMARK_PARALLEL": "7"}):
            self.assertEqual(resolve_benchmark_parallel("stt", 3), 3)

    def test_env_var_used_when_no_cli(self):
        with patch.dict("os.environ", {"CALIBRATE_STT_BENCHMARK_PARALLEL": "7"}):
            self.assertEqual(resolve_benchmark_parallel("stt", None), 7)

    def test_default_when_neither_set(self):
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("CALIBRATE_STT_BENCHMARK_PARALLEL", None)
            self.assertEqual(
                resolve_benchmark_parallel("stt", None), DEFAULT_BENCHMARK_PARALLEL
            )

    def test_invalid_env_falls_back_to_default(self):
        with patch.dict("os.environ", {"CALIBRATE_STT_BENCHMARK_PARALLEL": "abc"}):
            self.assertEqual(
                resolve_benchmark_parallel("stt", None), DEFAULT_BENCHMARK_PARALLEL
            )

    def test_non_positive_values_ignored(self):
        with patch.dict("os.environ", {"CALIBRATE_STT_BENCHMARK_PARALLEL": "0"}):
            # CLI 0 ignored, env 0 ignored -> default
            self.assertEqual(
                resolve_benchmark_parallel("stt", 0), DEFAULT_BENCHMARK_PARALLEL
            )


class TestRunBuildsSemaphoreFromResolvedParallel(unittest.IsolatedAsyncioTestCase):
    """``run`` must size its provider semaphore via ``resolve_benchmark_parallel``."""

    async def _run_and_capture_semaphore_value(self, *, max_parallel):
        captured = {}
        real_semaphore = __import__("asyncio").Semaphore

        def _spy_semaphore(value, *args, **kwargs):
            captured["value"] = value
            return real_semaphore(value, *args, **kwargs)

        with patch(
            "calibrate.stt.benchmark.run_single_provider_eval",
            new=AsyncMock(return_value={"status": "completed", "metrics": {}}),
        ), patch(
            "calibrate.stt.benchmark.generate_leaderboard"
        ), patch(
            "calibrate.stt.benchmark.asyncio.Semaphore", side_effect=_spy_semaphore
        ):
            from calibrate.stt.benchmark import run

            await run(
                providers=["deepgram"],
                input_dir="/tmp/does-not-matter",
                output_dir="/tmp/out",
                max_parallel=max_parallel,
            )
        return captured["value"]

    async def test_env_var_sets_parallelism(self):
        with patch.dict("os.environ", {"CALIBRATE_STT_BENCHMARK_PARALLEL": "5"}):
            value = await self._run_and_capture_semaphore_value(max_parallel=None)
        self.assertEqual(value, 5)

    async def test_explicit_arg_overrides_env(self):
        with patch.dict("os.environ", {"CALIBRATE_STT_BENCHMARK_PARALLEL": "5"}):
            value = await self._run_and_capture_semaphore_value(max_parallel=3)
        self.assertEqual(value, 3)

    async def test_invalid_env_falls_back_to_default(self):
        with patch.dict("os.environ", {"CALIBRATE_STT_BENCHMARK_PARALLEL": "0"}):
            value = await self._run_and_capture_semaphore_value(max_parallel=None)
        self.assertEqual(value, DEFAULT_BENCHMARK_PARALLEL)


if __name__ == "__main__":
    unittest.main()
