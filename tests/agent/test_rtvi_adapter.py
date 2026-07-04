"""Tests for RTVIMessageFrameAdapter helper methods that don't require pipecat runtime."""

import asyncio
import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


def _make_adapter(**overrides):
    from calibrate.agent.run_simulation import RTVIMessageFrameAdapter

    ctx = MagicMock()
    ctx.get_messages.return_value = []

    audio_buffer = MagicMock()

    defaults = dict(
        context=ctx,
        audio_buffer=audio_buffer,
        interrupt_probability=0.0,
        tool_calls=[],
        stt_outputs=[],
        ttft=defaultdict(list),
        processing_time=defaultdict(list),
        output_dir="/tmp",
        audio_save_dir="/tmp",
        agent_speaks_first=True,
        max_turns=10,
    )
    defaults.update(overrides)
    return RTVIMessageFrameAdapter(**defaults)


class TestAssignNextTranscriptAudioLine(unittest.TestCase):
    def test_monotonic_increment(self):
        adapter = _make_adapter()
        line1 = adapter._assign_next_transcript_audio_line(role="bot")
        line2 = adapter._assign_next_transcript_audio_line(role="user")
        line3 = adapter._assign_next_transcript_audio_line(role="bot")
        self.assertEqual([line1, line2, line3], [1, 2, 3])


class TestBuildSerializedTranscript(unittest.TestCase):
    def test_empty(self):
        adapter = _make_adapter()
        result = adapter._build_serialized_transcript()
        self.assertEqual(result, [])

    def test_role_flipping(self):
        ctx = MagicMock()
        ctx.get_messages.return_value = [
            {"role": "user", "content": "hi"},      # → assistant
            {"role": "assistant", "content": "hello"},  # → user
        ]
        adapter = _make_adapter(context=ctx)
        result = adapter._build_serialized_transcript()
        self.assertEqual(result[0]["role"], "assistant")
        self.assertEqual(result[0]["content"], "hi")
        self.assertEqual(result[1]["role"], "user")

    def test_merges_consecutive_same_role(self):
        ctx = MagicMock()
        ctx.get_messages.return_value = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        adapter = _make_adapter(context=ctx)
        result = adapter._build_serialized_transcript()
        # Both became assistant, merged
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "a b")

    def test_with_end_reason(self):
        adapter = _make_adapter()
        result = adapter._build_serialized_transcript(end_reason="max_turns")
        self.assertEqual(result[-1]["role"], "end_reason")
        self.assertEqual(result[-1]["content"], "max_turns")

    def test_tool_calls_inserted(self):
        ctx = MagicMock()
        ctx.get_messages.return_value = [
            {"role": "user", "content": "hi"},
        ]
        tool_calls = [{
            "position": 0,
            "data": {
                "tool_call_id": "call_1",
                "function_name": "foo",
                "args": {"x": 1},
            },
        }]
        adapter = _make_adapter(context=ctx, tool_calls=tool_calls)
        result = adapter._build_serialized_transcript()
        # First entry is tool_calls, then the message
        self.assertEqual(result[0]["role"], "assistant")
        self.assertIn("tool_calls", result[0])

    def test_tool_calls_after_messages(self):
        ctx = MagicMock()
        ctx.get_messages.return_value = [
            {"role": "user", "content": "hi"},
        ]
        tool_calls = [{
            "position": 5,
            "data": {
                "tool_call_id": "call_x",
                "function_name": "y",
                "args": {},
            },
        }]
        adapter = _make_adapter(context=ctx, tool_calls=tool_calls)
        result = adapter._build_serialized_transcript()
        # Tool call at position 5 (after all messages) appended
        self.assertEqual(result[-1]["role"], "assistant")
        self.assertIn("tool_calls", result[-1])

    def test_skip_non_dict_messages(self):
        ctx = MagicMock()
        ctx.get_messages.return_value = [
            "not a dict",
            {"role": "user", "content": "hi"},
        ]
        adapter = _make_adapter(context=ctx)
        result = adapter._build_serialized_transcript()
        self.assertEqual(len(result), 1)


class TestSaveTranscript(unittest.TestCase):
    def test_saves_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _make_adapter(output_dir=tmp)
            adapter._save_transcript([{"role": "assistant", "content": "hi"}])
            from calibrate.agent.run_simulation import TRANSCRIPT_FILE_NAME
            transcript_path = Path(tmp) / TRANSCRIPT_FILE_NAME
            self.assertTrue(transcript_path.exists())
            data = json.loads(transcript_path.read_text())
            self.assertEqual(data[0]["role"], "assistant")


class TestEnsureBotTranscriptLineForCurrentTurn(unittest.IsolatedAsyncioTestCase):
    async def test_not_awaiting_returns_early(self):
        adapter = _make_adapter()
        adapter._awaiting_first_bot_audio_chunk = False
        await adapter._ensure_bot_transcript_line_for_current_turn()  # No-op

    async def test_too_short_lexical_returns(self):
        adapter = _make_adapter()
        adapter._awaiting_first_bot_audio_chunk = True
        adapter._text_buffer = ""
        await adapter._ensure_bot_transcript_line_for_current_turn(spoken_fragment="a")
        self.assertTrue(adapter._awaiting_first_bot_audio_chunk)

    async def test_no_alpha_returns(self):
        adapter = _make_adapter()
        adapter._awaiting_first_bot_audio_chunk = True
        adapter._text_buffer = "123!"
        await adapter._ensure_bot_transcript_line_for_current_turn()
        self.assertTrue(adapter._awaiting_first_bot_audio_chunk)

    async def test_continues_bot_role(self):
        adapter = _make_adapter()
        adapter._awaiting_first_bot_audio_chunk = True
        adapter._text_buffer = "hello"
        adapter._active_transcript_audio_role = "bot"
        adapter._active_transcript_audio_index = 5
        await adapter._ensure_bot_transcript_line_for_current_turn()
        self.assertFalse(adapter._awaiting_first_bot_audio_chunk)

    async def test_new_bot_line(self):
        adapter = _make_adapter()
        adapter._awaiting_first_bot_audio_chunk = True
        adapter._text_buffer = "hello"
        adapter._active_transcript_audio_role = "user"
        await adapter._ensure_bot_transcript_line_for_current_turn()
        self.assertEqual(adapter._stt_turn_index, 1)
        self.assertEqual(adapter._active_transcript_audio_role, "bot")


class TestFlushPendingBotAudio(unittest.IsolatedAsyncioTestCase):
    async def test_no_pending(self):
        adapter = _make_adapter()
        await adapter._flush_pending_bot_audio()
        self.assertEqual(adapter._pending_bot_audio_frames, [])

    async def test_flushes(self):
        from calibrate.agent import run_simulation as RS

        with tempfile.TemporaryDirectory() as tmp:
            adapter = _make_adapter(audio_save_dir=tmp)
            adapter._active_transcript_audio_index = 1

            fake_frame = MagicMock()
            fake_frame.audio = b"\x00" * 100
            fake_frame.sample_rate = 16000
            fake_frame.num_channels = 1
            adapter._pending_bot_audio_frames = [fake_frame]

            with patch.object(RS, "save_audio_chunk", AsyncMock()):
                await adapter._flush_pending_bot_audio()
            self.assertEqual(adapter._pending_bot_audio_frames, [])


class TestResetBuffers(unittest.IsolatedAsyncioTestCase):
    async def test_clears_and_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _make_adapter(output_dir=tmp)
            adapter._text_buffer = "hi"
            adapter._heard_text_buffer = "hi"
            adapter._spoken_text_buffer = "hi"
            adapter._turn_index = 1
            await adapter._reset_buffers()
            self.assertEqual(adapter._text_buffer, "")
            self.assertEqual(adapter._spoken_text_buffer, "")


def _rtvi_frame(msg_type):
    from pipecat.frames.frames import InputTransportMessageFrame

    return InputTransportMessageFrame(
        message={"label": "rtvi-ai", "type": msg_type}
    )


def _stub_tasks(adapter):
    """Replace the pipecat task-manager hooks so no real event loop tasks run.

    ``_schedule_agent_turn_stop`` calls ``create_task`` (needs a running task
    manager the unit-test adapter doesn't have); the coalescing tests drive the
    debounce coroutine directly instead.
    """
    def _fake_create_task(coro, name=None):
        coro.close()  # avoid "coroutine was never awaited" warnings
        return MagicMock()

    adapter.create_task = _fake_create_task
    adapter.cancel_task = AsyncMock()


async def _drive_bot_started(adapter, pushed=None):
    """Push a ``bot-started-speaking`` RTVI frame through the adapter.

    Captures every ``push_frame`` call and returns the pushed (frame, direction)
    pairs so tests can assert whether an InterruptionTaskFrame was emitted.
    """
    return await _drive_rtvi(adapter, "bot-started-speaking", pushed)


async def _drive_rtvi(adapter, msg_type, pushed=None):
    from pipecat.processors.frame_processor import FrameDirection

    if pushed is None:
        pushed = []

    async def _capture(frame, direction):
        pushed.append((frame, direction))

    _stub_tasks(adapter)
    adapter.push_frame = _capture
    await adapter.process_frame(_rtvi_frame(msg_type), FrameDirection.DOWNSTREAM)
    return pushed


def _count_frames(pushed, frame_cls):
    return sum(1 for frame, _ in pushed if isinstance(frame, frame_cls))


def _count_interruptions(pushed):
    from pipecat.frames.frames import InterruptionTaskFrame

    return _count_frames(pushed, InterruptionTaskFrame)


class TestAgentFloorInterruptGating(unittest.IsolatedAsyncioTestCase):
    """Consecutive agent utterances must not interrupt (and drop) one another.

    Regression test for the bug where an external agent that speaks its opening
    as two back-to-back utterances (e.g. a greeting then the first question) had
    the greeting erased: the second utterance's InterruptionTaskFrame cancelled
    the user context aggregator's queue before the greeting was committed.
    """

    async def test_first_utterance_interrupts(self):
        adapter = _make_adapter()
        self.assertFalse(adapter._agent_has_floor)
        pushed = await _drive_bot_started(adapter)
        self.assertEqual(_count_interruptions(pushed), 1)
        self.assertTrue(adapter._agent_has_floor)

    async def test_consecutive_utterance_does_not_interrupt(self):
        adapter = _make_adapter()
        await _drive_bot_started(adapter)  # first utterance takes the floor
        pushed = await _drive_bot_started(adapter)  # second utterance, same turn
        self.assertEqual(_count_interruptions(pushed), 0)
        self.assertTrue(adapter._agent_has_floor)

    async def test_interrupts_again_after_sim_user_turn(self):
        from pipecat.frames.frames import LLMFullResponseStartFrame
        from pipecat.processors.frame_processor import FrameDirection
        from calibrate.agent.run_simulation import SimulatedUserTurnIndexHook

        adapter = _make_adapter()
        await _drive_bot_started(adapter)  # agent takes the floor

        # Simulated user starts its own turn -> agent no longer holds the floor.
        hook = SimulatedUserTurnIndexHook(adapter)
        hook.push_frame = AsyncMock()
        await hook.process_frame(
            LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM
        )
        self.assertFalse(adapter._agent_has_floor)

        # Agent taking the floor back should interrupt the sim user again.
        pushed = await _drive_bot_started(adapter)
        self.assertEqual(_count_interruptions(pushed), 1)
        self.assertTrue(adapter._agent_has_floor)


class TestAgentTurnCoalescing(unittest.IsolatedAsyncioTestCase):
    """Consecutive agent utterances coalesce into one simulated-user turn.

    Regression test for the follow-on bug: with per-utterance UserStopped frames,
    the sim user's LLM fired on the greeting before it heard the first question,
    so only sometimes did both land in the same turn. Now UserStopped is debounced
    so a multi-utterance agent turn (greeting + question) forms one sim-user turn.
    """

    async def test_single_user_started_across_consecutive_utterances(self):
        from pipecat.frames.frames import UserStartedSpeakingFrame

        adapter = _make_adapter()
        pushed = []
        await _drive_bot_started(adapter, pushed)  # first utterance opens the turn
        await _drive_bot_started(adapter, pushed)  # follow-on utterance, same turn
        self.assertTrue(adapter._agent_turn_active)
        # Only one turn opened, so exactly one UserStartedSpeakingFrame.
        self.assertEqual(_count_frames(pushed, UserStartedSpeakingFrame), 1)

    async def test_bot_stopped_defers_user_stop(self):
        from pipecat.frames.frames import (
            TranscriptionFrame,
            UserStoppedSpeakingFrame,
        )

        adapter = _make_adapter()
        adapter._heard_text_buffer = " नमस्ते!"
        await _drive_bot_started(adapter)
        pushed = await _drive_rtvi(adapter, "bot-stopped-speaking")
        # The utterance's transcription is emitted immediately...
        self.assertEqual(_count_frames(pushed, TranscriptionFrame), 1)
        # ...but the turn-ending UserStopped is deferred to the debounce.
        self.assertEqual(_count_frames(pushed, UserStoppedSpeakingFrame), 0)
        self.assertTrue(adapter._agent_turn_active)

    async def test_debounce_emits_user_stop(self):
        from unittest.mock import patch
        from pipecat.frames.frames import UserStoppedSpeakingFrame
        from pipecat.processors.frame_processor import FrameDirection

        adapter = _make_adapter()
        adapter._agent_turn_active = True
        pushed = []

        async def _capture(frame, direction):
            pushed.append((frame, direction))

        adapter.push_frame = _capture
        with patch("calibrate.agent.run_simulation.asyncio.sleep", AsyncMock()):
            await adapter._end_agent_turn_after_debounce()

        self.assertEqual(_count_frames(pushed, UserStoppedSpeakingFrame), 1)
        self.assertFalse(adapter._agent_turn_active)


if __name__ == "__main__":
    unittest.main()
