"""Tests for RTVIMessageFrameAdapter helper methods that don't require pipecat runtime."""

import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


def _make_adapter(**overrides):
    from calibrate_agent.agent.run_simulation import RTVIMessageFrameAdapter

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
        adapter = _make_adapter()
        adapter.record_agent_turn("hi")  # agent → assistant
        adapter.record_persona_text("hello")  # persona → user (flushed at build)
        result = adapter._build_serialized_transcript()
        self.assertEqual(result[0]["role"], "assistant")
        self.assertEqual(result[0]["content"], "hi")
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[1]["content"], "hello")

    def test_merges_consecutive_same_role(self):
        adapter = _make_adapter()
        adapter.record_agent_turn("a")
        adapter.record_agent_turn("b")
        result = adapter._build_serialized_transcript()
        # Both agent turns became assistant and merged
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "a b")

    def test_empty_turns_are_dropped(self):
        adapter = _make_adapter()
        adapter.record_agent_turn("")
        adapter.record_agent_turn(None)
        adapter.record_persona_text("   ")
        adapter.record_persona_text(None)
        self.assertEqual(adapter._build_serialized_transcript(), [])

    def test_with_end_reason(self):
        adapter = _make_adapter()
        result = adapter._build_serialized_transcript(end_reason="max_turns")
        self.assertEqual(result[-1]["role"], "end_reason")
        self.assertEqual(result[-1]["content"], "max_turns")

    def test_tool_call_recorded_after_the_turn_it_followed(self):
        adapter = _make_adapter()
        adapter.record_agent_turn("what is your name?")
        adapter.record_persona_text("my name is Geeta")
        adapter.record_tool_call(
            {
                "tool_call_id": "c1",
                "function_name": "plan_next_question",
                "args": {"i": 2},
            }
        )
        result = adapter._build_serialized_transcript()
        self.assertEqual(
            [m["role"] for m in result], ["assistant", "user", "assistant"]
        )
        self.assertEqual(result[0]["content"], "what is your name?")
        self.assertEqual(result[1]["content"], "my name is Geeta")
        self.assertIn("tool_calls", result[2])
        self.assertEqual(
            result[2]["tool_calls"][0]["function"]["name"], "plan_next_question"
        )
        self.assertEqual(
            result[2]["tool_calls"][0]["function"]["arguments"], '{"i": 2}'
        )

    def test_consecutive_tool_calls_grouped(self):
        adapter = _make_adapter()
        adapter.record_agent_turn("hi")
        adapter.record_tool_call(
            {"tool_call_id": "c1", "function_name": "plan_next_question", "args": {}}
        )
        adapter.record_tool_call(
            {"tool_call_id": "c2", "function_name": "end_call", "args": {}}
        )
        result = adapter._build_serialized_transcript()
        # Both calls collapse into a single assistant tool_calls entry
        tool_entries = [m for m in result if "tool_calls" in m]
        self.assertEqual(len(tool_entries), 1)
        names = [c["function"]["name"] for c in tool_entries[0]["tool_calls"]]
        self.assertEqual(names, ["plan_next_question", "end_call"])

    def test_final_persona_turn_captured_when_call_ends_abruptly(self):
        """The regression: agent ends the call the instant it hears the persona.

        The persona turn's spoken ``TTSTextFrame`` fragments must still land in the
        transcript, ahead of the tool calls that ended the call.
        """
        adapter = _make_adapter()
        adapter.record_agent_turn("what is your name?")
        for frag in ["my", "name", "is", "Geeta", "Prasad"]:
            adapter.record_persona_text(frag)
        adapter.record_tool_call(
            {"tool_call_id": "c1", "function_name": "plan_next_question", "args": {}}
        )
        adapter.record_tool_call(
            {"tool_call_id": "c2", "function_name": "end_call", "args": {}}
        )
        result = adapter._build_serialized_transcript()
        self.assertEqual(
            [m["role"] for m in result], ["assistant", "user", "assistant"]
        )
        self.assertEqual(result[1]["content"], "my name is Geeta Prasad")
        self.assertEqual(len(result[2]["tool_calls"]), 2)

    def test_persona_turn_recorded_once(self):
        """A persona turn flushed by a tool call is not duplicated afterwards."""
        adapter = _make_adapter()
        for frag in ["my", "name", "is", "Geeta"]:
            adapter.record_persona_text(frag)
        adapter.record_tool_call(
            {"tool_call_id": "c1", "function_name": "plan_next_question", "args": {}}
        )
        result = adapter._build_serialized_transcript()
        persona_lines = [m for m in result if m.get("role") == "user"]
        self.assertEqual(len(persona_lines), 1)
        self.assertEqual(persona_lines[0]["content"], "my name is Geeta")

    def test_punctuation_fragment_attaches_to_previous_word(self):
        adapter = _make_adapter()
        for frag in ["Geeta", "Prasad", "."]:
            adapter.record_persona_text(frag)
        result = adapter._build_serialized_transcript()
        self.assertEqual(result[-1]["content"], "Geeta Prasad.")

    def test_pending_persona_included_when_ending_on_persona(self):
        adapter = _make_adapter()
        adapter.record_agent_turn("anything else?")
        adapter.record_persona_text("no")
        adapter.record_persona_text("thanks")
        # No flush event follows; build must still surface the in-flight turn.
        result = adapter._build_serialized_transcript()
        self.assertEqual(result[-1]["role"], "user")
        self.assertEqual(result[-1]["content"], "no thanks")

    def test_build_does_not_mutate_pending(self):
        adapter = _make_adapter()
        adapter.record_persona_text("hello")
        adapter._build_serialized_transcript()
        # Building twice yields the same result (pending not consumed).
        again = adapter._build_serialized_transcript()
        self.assertEqual(again[-1]["content"], "hello")


class TestSaveTranscript(unittest.TestCase):
    def test_saves_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _make_adapter(output_dir=tmp)
            adapter._save_transcript([{"role": "assistant", "content": "hi"}])
            from calibrate_agent.agent.run_simulation import TRANSCRIPT_FILE_NAME

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
        from calibrate_agent.agent import run_simulation as RS

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

    return InputTransportMessageFrame(message={"label": "rtvi-ai", "type": msg_type})


async def _drive_bot_started(adapter, pushed=None):
    """Push a ``bot-started-speaking`` RTVI frame through the adapter."""
    return await _drive_rtvi(adapter, "bot-started-speaking", pushed)


async def _drive_rtvi(adapter, msg_type, pushed=None):
    """Push one RTVI message through the adapter, capturing pushed frames."""
    from pipecat.processors.frame_processor import FrameDirection

    if pushed is None:
        pushed = []

    async def _capture(frame, direction):
        pushed.append((frame, direction))

    adapter.push_frame = _capture
    await adapter.process_frame(_rtvi_frame(msg_type), FrameDirection.DOWNSTREAM)
    return pushed


def _count_frames(pushed, frame_cls):
    return sum(1 for frame, _ in pushed if isinstance(frame, frame_cls))


def _count_interruptions(pushed):
    from pipecat.frames.frames import InterruptionTaskFrame

    return _count_frames(pushed, InterruptionTaskFrame)


class TestNaturalInterrupt(unittest.IsolatedAsyncioTestCase):
    """Every agent utterance interrupts the simulated user (natural full-duplex).

    Turn boundaries are driven by the aggregator's audio strategies, so the adapter
    no longer emits UserStarted/UserStopped — it just fires the interrupt and the
    utterance transcription.
    """

    async def test_bot_started_interrupts_without_manual_userstarted(self):
        from pipecat.frames.frames import UserStartedSpeakingFrame

        adapter = _make_adapter()
        pushed = await _drive_bot_started(adapter)
        # Fires the interrupt to cut the sim user's in-flight TTS...
        self.assertEqual(_count_interruptions(pushed), 1)
        # ...but does NOT manually open the turn (audio strategies do that).
        self.assertEqual(_count_frames(pushed, UserStartedSpeakingFrame), 0)

    async def test_every_consecutive_utterance_interrupts(self):
        adapter = _make_adapter()
        pushed = []
        await _drive_bot_started(adapter, pushed)
        await _drive_bot_started(adapter, pushed)
        # Unlike the old gated behavior, a follow-on utterance still interrupts.
        self.assertEqual(_count_interruptions(pushed), 2)

    async def test_bot_stopped_emits_stripped_transcription_no_userstopped(self):
        from pipecat.frames.frames import (
            TranscriptionFrame,
            UserStoppedSpeakingFrame,
        )

        with tempfile.TemporaryDirectory() as tmp:
            adapter = _make_adapter(output_dir=tmp)
            adapter._pending_user_turn = True
            adapter._heard_text_buffer = " नमस्ते!"
            pushed = await _drive_rtvi(adapter, "bot-stopped-speaking")

        transcriptions = [f for f, _ in pushed if isinstance(f, TranscriptionFrame)]
        self.assertEqual(len(transcriptions), 1)
        # Leading space stripped so coalesced turns don't double-space.
        self.assertEqual(transcriptions[0].text, "नमस्ते!")
        # Turn-end is decided by smart-turn, not a manual UserStopped.
        self.assertEqual(_count_frames(pushed, UserStoppedSpeakingFrame), 0)


def _bot_output_frame(text, spoken):
    from pipecat.frames.frames import InputTransportMessageFrame

    return InputTransportMessageFrame(
        message={
            "label": "rtvi-ai",
            "type": "bot-output",
            "data": {"text": text, "spoken": spoken},
        }
    )


async def _drive_frame(adapter, frame, pushed=None):
    from pipecat.processors.frame_processor import FrameDirection

    if pushed is None:
        pushed = []

    async def _capture(f, direction):
        pushed.append((f, direction))

    adapter.push_frame = _capture
    await adapter.process_frame(frame, FrameDirection.DOWNSTREAM)
    return pushed


def _count_interrupt_messages(pushed):
    from pipecat.frames.frames import OutputTransportMessageUrgentFrame

    return sum(
        1
        for f, _ in pushed
        if isinstance(f, OutputTransportMessageUrgentFrame)
        and (getattr(f, "message", {}) or {}).get("data", {}).get("t") == "interrupt"
    )


class TestExternalAgentInterrupt(unittest.IsolatedAsyncioTestCase):
    """External agents get the block model: a decided interrupt cuts the agent off
    at once and blocks its output until the sim user finishes. Internal agents keep
    the word-level path (decision only; trigger later on stream/spoken match)."""

    async def test_external_interrupt_executes_immediately(self):
        from pipecat.frames.frames import (
            TranscriptionFrame,
            UserStoppedSpeakingFrame,
            InterimTranscriptionFrame,
        )

        with tempfile.TemporaryDirectory() as tmp:
            adapter = _make_adapter(
                output_dir=tmp, interrupt_probability=1.0, is_external=True
            )
            pushed = await _drive_frame(adapter, _bot_output_frame("नमस्ते!", True))

        self.assertTrue(adapter._is_bot_interrupt_triggered)
        # Tells the agent to stop, commits the partial, ends the turn.
        self.assertEqual(_count_interrupt_messages(pushed), 1)
        self.assertEqual(_count_frames(pushed, TranscriptionFrame), 1)
        self.assertEqual(_count_frames(pushed, UserStoppedSpeakingFrame), 1)
        # Nothing more is fed to the sim user (no interim after cutting in).
        self.assertEqual(_count_frames(pushed, InterimTranscriptionFrame), 0)

    async def test_internal_interrupt_only_decides(self):
        from pipecat.frames.frames import (
            UserStoppedSpeakingFrame,
            InterimTranscriptionFrame,
        )

        with tempfile.TemporaryDirectory() as tmp:
            adapter = _make_adapter(
                output_dir=tmp, interrupt_probability=1.0, is_external=False
            )
            pushed = await _drive_frame(adapter, _bot_output_frame("नमस्ते!", True))

        # Internal: decision is made but the interrupt is NOT executed yet; it waits
        # for the spoken text to catch up (word-level control).
        self.assertTrue(adapter._is_bot_interrupt_decided)
        self.assertFalse(adapter._is_bot_interrupt_triggered)
        self.assertEqual(_count_interrupt_messages(pushed), 0)
        self.assertEqual(_count_frames(pushed, UserStoppedSpeakingFrame), 0)
        # Internal still feeds what was heard up to the cut-in point.
        self.assertEqual(_count_frames(pushed, InterimTranscriptionFrame), 1)

    async def test_bot_started_blocked_while_external_interrupt_active(self):
        adapter = _make_adapter(is_external=True)
        adapter._is_bot_interrupt_triggered = True
        pushed = await _drive_bot_started(adapter)
        # Agent's attempt to speak is ignored: no interrupt of the sim user's turn.
        self.assertEqual(_count_interruptions(pushed), 0)

    async def test_agent_audio_dropped_while_external_interrupt_active(self):
        from pipecat.frames.frames import InputAudioRawFrame

        adapter = _make_adapter(is_external=True)
        adapter._is_bot_interrupt_triggered = True
        frame = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
        pushed = await _drive_frame(adapter, frame)
        # Dropped, not forwarded to the aggregator's VAD (would cancel the sim user).
        self.assertEqual(_count_frames(pushed, InputAudioRawFrame), 0)


if __name__ == "__main__":
    unittest.main()
