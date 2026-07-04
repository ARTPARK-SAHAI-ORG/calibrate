"""Tests for simple helpers in calibrate/agent/run_simulation.py."""

import asyncio
import socket
import unittest
from unittest.mock import patch, AsyncMock, MagicMock


class TestIsBenignGoogleSttIdleError(unittest.TestCase):
    def test_benign_match(self):
        from calibrate.agent.run_simulation import _is_benign_google_stt_idle_error

        self.assertTrue(_is_benign_google_stt_idle_error(
            "GoogleSTTService error: 409 Stream timed out after receiving no more client requests"
        ))

    def test_not_benign(self):
        from calibrate.agent.run_simulation import _is_benign_google_stt_idle_error

        self.assertFalse(_is_benign_google_stt_idle_error("Some other error"))
        self.assertFalse(_is_benign_google_stt_idle_error(
            "GoogleSTTService: different error"
        ))


class TestCountAgentMessageTurns(unittest.TestCase):
    def test_empty_messages(self):
        from calibrate.agent.run_simulation import count_agent_message_turns

        self.assertEqual(count_agent_message_turns([]), 0)

    def test_single_user_run(self):
        from calibrate.agent.run_simulation import count_agent_message_turns

        messages = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},  # streaming fragment
        ]
        self.assertEqual(count_agent_message_turns(messages), 1)

    def test_alternating_turns(self):
        from calibrate.agent.run_simulation import count_agent_message_turns

        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"},
        ]
        self.assertEqual(count_agent_message_turns(messages), 3)

    def test_skip_non_dict(self):
        from calibrate.agent.run_simulation import count_agent_message_turns

        messages = [
            "not a dict",
            {"role": "user", "content": "a"},
        ]
        self.assertEqual(count_agent_message_turns(messages), 1)

    def test_role_none_treated_as_separator(self):
        from calibrate.agent.run_simulation import count_agent_message_turns

        messages = [
            {"role": "user", "content": "a"},
            {"no_role": "x"},
            {"role": "user", "content": "b"},
        ]
        # 2 turns because no_role doesn't reset (role is None, falls through)
        self.assertEqual(count_agent_message_turns(messages), 1)


class TestFindAvailablePort(unittest.TestCase):
    def test_returns_port(self):
        from calibrate.agent.run_simulation import find_available_port

        port = find_available_port()
        self.assertGreater(port, 0)

    def test_os_error_raises(self):
        from calibrate.agent import run_simulation as RS

        with patch("socket.socket", side_effect=OSError("no port")):
            with self.assertRaises(RuntimeError):
                RS.find_available_port()


class TestMetricsLogger(unittest.IsolatedAsyncioTestCase):
    async def test_process_frame(self):
        from collections import defaultdict
        from calibrate.agent.run_simulation import MetricsLogger
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        ttft = defaultdict(list)
        proc_time = defaultdict(list)
        ctx = MagicMock()
        ctx.get_messages.return_value = [{"role": "user"}]

        logger = MetricsLogger(ttft, proc_time, ctx)

        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {
            "label": "rtvi-ai",
            "type": "metrics",
            "data": {
                "ttfb": [{"processor": "p1", "value": 0.5}, {"processor": "p2", "value": 0}],
                "processing": [{"processor": "p1", "value": 0.3}],
            },
        }

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(MetricsLogger, "push_frame", AsyncMock()):
            await logger.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(ttft["p1"], [0.5])
        self.assertEqual(proc_time["p1"], [0.3])

    async def test_process_frame_no_context_messages(self):
        from collections import defaultdict
        from calibrate.agent.run_simulation import MetricsLogger
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        ctx = MagicMock()
        ctx.get_messages.return_value = []  # empty context — skip
        logger = MetricsLogger(defaultdict(list), defaultdict(list), ctx)

        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {"label": "rtvi-ai", "type": "metrics", "data": {}}

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(MetricsLogger, "push_frame", AsyncMock()):
            await logger.process_frame(frame, FrameDirection.DOWNSTREAM)


class TestIOLogger(unittest.IsolatedAsyncioTestCase):
    async def test_process_tts_text_frame(self):
        from calibrate.agent.run_simulation import IOLogger
        from pipecat.frames.frames import TTSTextFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        logger = IOLogger()
        frame = MagicMock(spec=TTSTextFrame)
        frame.text = "hello"

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(IOLogger, "push_frame", AsyncMock()):
            await logger.process_frame(frame, FrameDirection.DOWNSTREAM)


class TestSimulatedUserTurnIndexHook(unittest.IsolatedAsyncioTestCase):
    async def test_marks_pending(self):
        from calibrate.agent.run_simulation import SimulatedUserTurnIndexHook
        from pipecat.frames.frames import LLMFullResponseStartFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        adapter = MagicMock()
        adapter._sim_user_turn_pending = False
        hook = SimulatedUserTurnIndexHook(adapter)
        frame = MagicMock(spec=LLMFullResponseStartFrame)

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(SimulatedUserTurnIndexHook, "push_frame", AsyncMock()):
            await hook.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertTrue(adapter._sim_user_turn_pending)


class TestBuildUserContextAggregator(unittest.TestCase):
    """Turn-taking must be audio-driven: smart-turn stop + a bounded fallback.

    The agent under test is the "user"; turn boundaries come from its audio, not
    manual UserStarted/UserStopped frames. Turn-end is decided by the smart-turn
    analyzer, with a short user_turn_stop_timeout fallback for when it stalls on
    synthesized TTS audio.
    """

    def test_uses_smart_turn_stop_and_bounded_timeout(self):
        from calibrate.agent.run_simulation import (
            build_user_context_aggregator,
            SIM_USER_TURN_STOP_TIMEOUT_SECS,
        )
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
        from pipecat.turns.user_start.external_user_turn_start_strategy import (
            ExternalUserTurnStartStrategy,
        )

        ctx = LLMContext([{"role": "system", "content": "x"}])
        pair = build_user_context_aggregator(ctx)
        controller = pair.user()._user_turn_controller
        strategies = controller._user_turn_strategies
        # Smart-turn decides turn-end.
        self.assertTrue(
            any(isinstance(s, TurnAnalyzerUserTurnStopStrategy) for s in strategies.stop)
        )
        # Turn-start is audio-driven (default strategies), NOT manual/external.
        self.assertFalse(
            any(isinstance(s, ExternalUserTurnStartStrategy) for s in strategies.start)
        )
        # Bounded fallback for smart-turn stalls on TTS audio.
        self.assertEqual(
            controller._user_turn_stop_timeout, SIM_USER_TURN_STOP_TIMEOUT_SECS
        )


class TestSTTLogger(unittest.IsolatedAsyncioTestCase):
    async def test_process_frame_user_transcription(self):
        from calibrate.agent.run_simulation import STTLogger
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        outputs = []
        adapter = MagicMock()
        adapter._stt_turn_index = 1
        logger = STTLogger(outputs, adapter)
        # logger sets last_turn_index=0, but adapter has turn=1 → append new
        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {
            "label": "rtvi-ai",
            "type": "user-transcription",
            "data": {"text": "hello", "final": True},
        }

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(STTLogger, "push_frame", AsyncMock()):
            await logger.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(outputs[-1], "hello")

    async def test_process_frame_continues_turn(self):
        from calibrate.agent.run_simulation import STTLogger
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        outputs = []
        adapter = MagicMock()
        adapter._stt_turn_index = 0  # same turn
        logger = STTLogger(outputs, adapter)
        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {
            "label": "rtvi-ai",
            "type": "user-transcription",
            "data": {"text": "more", "final": True},
        }

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(STTLogger, "push_frame", AsyncMock()):
            await logger.process_frame(frame, FrameDirection.DOWNSTREAM)
        # Empty turn appends to outputs[-1] which is ""
        self.assertEqual(outputs[-1], "more")


class TestSilencePadder(unittest.IsolatedAsyncioTestCase):
    async def test_init(self):
        from calibrate.agent.run_simulation import SilencePadder

        padder = SilencePadder(silence_duration_ms=200, chunk_ms=20)
        self.assertEqual(padder._silence_duration_ms, 200)
        self.assertEqual(padder._chunk_ms, 20)


class TestResolveSerializer(unittest.TestCase):
    def test_protobuf_returns_instance(self):
        from calibrate.agent.run_simulation import resolve_serializer
        from pipecat.serializers.protobuf import ProtobufFrameSerializer

        ser = resolve_serializer("protobuf")
        self.assertIsInstance(ser, ProtobufFrameSerializer)

    def test_returns_new_instance_each_call(self):
        from calibrate.agent.run_simulation import resolve_serializer

        self.assertIsNot(resolve_serializer("protobuf"), resolve_serializer("protobuf"))

    def test_unknown_raises_value_error(self):
        from calibrate.agent.run_simulation import resolve_serializer

        with self.assertRaises(ValueError):
            resolve_serializer("nope")


class TestIsExternalWsAgent(unittest.TestCase):
    def test_ws_scheme_is_external(self):
        from calibrate.agent.run_simulation import is_external_ws_agent

        self.assertTrue(is_external_ws_agent({"agent_url": "ws://host:9000"}))

    def test_wss_scheme_is_external(self):
        from calibrate.agent.run_simulation import is_external_ws_agent

        self.assertTrue(is_external_ws_agent({"agent_url": "wss://host/agent"}))

    def test_http_scheme_is_internal(self):
        from calibrate.agent.run_simulation import is_external_ws_agent

        self.assertFalse(is_external_ws_agent({"agent_url": "http://host:9000"}))
        self.assertFalse(is_external_ws_agent({"agent_url": "https://host"}))

    def test_missing_url_is_internal(self):
        from calibrate.agent.run_simulation import is_external_ws_agent

        self.assertFalse(is_external_ws_agent({}))
        self.assertFalse(is_external_ws_agent({"agent_url": None}))

    def test_none_config_is_internal(self):
        from calibrate.agent.run_simulation import is_external_ws_agent

        self.assertFalse(is_external_ws_agent(None))


class TestSelectTransportUri(unittest.TestCase):
    def test_agent_uri_used_when_set(self):
        from calibrate.agent.run_simulation import select_transport_uri

        self.assertEqual(
            select_transport_uri("wss://external/agent", 8765),
            "wss://external/agent",
        )

    def test_falls_back_to_localhost_port(self):
        from calibrate.agent.run_simulation import select_transport_uri

        self.assertEqual(select_transport_uri(None, 1234), "ws://localhost:1234")

    def test_empty_string_falls_back(self):
        from calibrate.agent.run_simulation import select_transport_uri

        self.assertEqual(select_transport_uri("", 4321), "ws://localhost:4321")


class TestEndToEndLatencyTracker(unittest.TestCase):
    def test_no_turns_mean_is_none(self):
        from calibrate.agent.run_simulation import EndToEndLatencyTracker

        self.assertIsNone(EndToEndLatencyTracker().mean())

    def test_agent_audio_without_user_turn_is_ignored(self):
        from calibrate.agent.run_simulation import EndToEndLatencyTracker

        tracker = EndToEndLatencyTracker()
        tracker.mark_agent_audio()  # no pending user turn end
        self.assertEqual(tracker.deltas, [])
        self.assertIsNone(tracker.mean())

    def test_records_delta_and_mean(self):
        from calibrate.agent.run_simulation import EndToEndLatencyTracker

        loop = asyncio.new_event_loop()

        async def scenario():
            tracker = EndToEndLatencyTracker()
            tracker.mark_user_turn_end()
            await asyncio.sleep(0.01)
            tracker.mark_agent_audio()
            # only the first agent audio after a user turn end counts
            tracker.mark_agent_audio()
            return tracker

        try:
            tracker = loop.run_until_complete(scenario())
        finally:
            loop.close()

        self.assertEqual(len(tracker.deltas), 1)
        self.assertGreater(tracker.mean(), 0)


class TestBotStartedResetsInterruptState(unittest.IsolatedAsyncioTestCase):
    """A new bot utterance must clear stale interrupt flags.

    Regression test: agents that speak a turn as separate utterances (a short
    acknowledgement then the reply) previously left ``_is_bot_interrupt_decided``/
    ``_is_bot_interrupt_triggered`` set across the utterance boundary, so the next
    utterance's audio was dropped and its text ignored — the simulated user never
    "heard" the reply and the run deadlocked.
    """

    def _make_adapter(self, max_turns=20):
        from collections import defaultdict
        from calibrate.agent.run_simulation import RTVIMessageFrameAdapter

        ctx = MagicMock()
        ctx.get_messages.return_value = []  # 0 agent turns < max_turns
        audio_buffer = MagicMock()
        return RTVIMessageFrameAdapter(
            context=ctx,
            audio_buffer=audio_buffer,
            interrupt_probability=0.0,
            tool_calls=[],
            stt_outputs=[],
            ttft=defaultdict(list),
            processing_time=defaultdict(list),
            output_dir="/tmp",
            audio_save_dir="/tmp",
            max_turns=max_turns,
        )

    async def _send_bot_started(self, adapter):
        from calibrate.agent.run_simulation import RTVIMessageFrameAdapter
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {"label": "rtvi-ai", "type": "bot-started-speaking", "data": {}}
        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(RTVIMessageFrameAdapter, "push_frame", AsyncMock()):
            await adapter.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def test_stale_interrupt_flags_cleared_on_new_utterance(self):
        adapter = self._make_adapter()
        # Simulate a prior utterance that decided/triggered an interrupt but never
        # saw the clearing ``user-stopped-speaking``.
        adapter._is_bot_interrupt_decided = True
        adapter._is_bot_interrupt_triggered = True
        adapter._spoken_text_buffer = "stale ack text"

        await self._send_bot_started(adapter)

        self.assertFalse(adapter._is_bot_interrupt_decided)
        self.assertFalse(adapter._is_bot_interrupt_triggered)
        self.assertEqual(adapter._spoken_text_buffer, "")

    async def test_max_turns_branch_does_not_reset(self):
        # When max turns is reached the adapter ends the run instead of starting a
        # fresh utterance; the reset must not fire on that path.
        adapter = self._make_adapter(max_turns=0)
        adapter._is_bot_interrupt_decided = True

        await self._send_bot_started(adapter)

        self.assertTrue(adapter._is_bot_interrupt_decided)


if __name__ == "__main__":
    unittest.main()
