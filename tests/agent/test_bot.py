"""Tests for calibrate_agent/agent/bot.py — MetricsLogger and config models."""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock


class TestSTTConfig(unittest.TestCase):
    def test_default(self):
        from calibrate_agent.agent.bot import STTConfig

        cfg = STTConfig()
        self.assertEqual(cfg.provider, "deepgram")

    def test_custom_provider(self):
        from calibrate_agent.agent.bot import STTConfig

        cfg = STTConfig(provider="google")
        self.assertEqual(cfg.provider, "google")


class TestTTSConfig(unittest.TestCase):
    def test_default(self):
        from calibrate_agent.agent.bot import TTSConfig

        cfg = TTSConfig()
        self.assertEqual(cfg.provider, "google")
        self.assertIsNone(cfg.instructions)

    def test_with_instructions(self):
        from calibrate_agent.agent.bot import TTSConfig

        cfg = TTSConfig(provider="openai", instructions="be polite")
        self.assertEqual(cfg.instructions, "be polite")


class TestLLMConfig(unittest.TestCase):
    def test_default(self):
        from calibrate_agent.agent.bot import DEFAULT_TEST_AGENT_LLM_MODEL, LLMConfig

        cfg = LLMConfig()
        self.assertEqual(cfg.provider, "openrouter")
        self.assertEqual(cfg.model, DEFAULT_TEST_AGENT_LLM_MODEL)

    def test_custom(self):
        from calibrate_agent.agent.bot import LLMConfig

        cfg = LLMConfig(provider="openai", model="gpt-4")
        self.assertEqual(cfg.provider, "openai")


class TestMetricsLogger(unittest.IsolatedAsyncioTestCase):
    async def test_metrics_frame_with_data(self):
        from calibrate_agent.agent.bot import MetricsLogger
        from pipecat.frames.frames import MetricsFrame
        from pipecat.metrics.metrics import (
            TTFBMetricsData,
            ProcessingMetricsData,
            LLMUsageMetricsData,
            TTSUsageMetricsData,
        )
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        ml = MetricsLogger()

        ttfb = MagicMock(spec=TTFBMetricsData)
        ttfb.value = 0.5
        proc = MagicMock(spec=ProcessingMetricsData)
        proc.value = 0.3
        llm_usage = MagicMock(spec=LLMUsageMetricsData)
        llm_usage.value = MagicMock(prompt_tokens=10, completion_tokens=20)
        tts_usage = MagicMock(spec=TTSUsageMetricsData)
        tts_usage.value = 5

        frame = MagicMock(spec=MetricsFrame)
        frame.data = [ttfb, proc, llm_usage, tts_usage]

        with (
            patch.object(FrameProcessor, "process_frame", AsyncMock()),
            patch.object(MetricsLogger, "push_frame", AsyncMock()),
        ):
            await ml.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def test_non_metrics_frame(self):
        from calibrate_agent.agent.bot import MetricsLogger
        from pipecat.frames.frames import Frame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        ml = MetricsLogger()
        frame = MagicMock(spec=Frame)

        with (
            patch.object(FrameProcessor, "process_frame", AsyncMock()),
            patch.object(MetricsLogger, "push_frame", AsyncMock()),
        ):
            await ml.process_frame(frame, FrameDirection.DOWNSTREAM)


if __name__ == "__main__":
    unittest.main()
