"""Tests for ConversationState and Processor classes in llm/run_simulation.py."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


class TestConversationState(unittest.IsolatedAsyncioTestCase):
    async def test_record_turn_normal(self):
        from calibrate_agent.llm.run_simulation import ConversationState

        state = ConversationState(max_turns=3)
        self.assertTrue(await state.record_turn())
        self.assertTrue(await state.record_turn())
        # 3rd hits max → False
        self.assertFalse(await state.record_turn())
        self.assertTrue(state.finished)

    async def test_record_turn_after_finished(self):
        from calibrate_agent.llm.run_simulation import ConversationState

        state = ConversationState(max_turns=2)
        state.finished = True
        self.assertFalse(await state.record_turn())

    async def test_mark_finished_once(self):
        from calibrate_agent.llm.run_simulation import ConversationState

        state = ConversationState(max_turns=5)
        self.assertTrue(await state.mark_finished())
        self.assertFalse(await state.mark_finished())  # second call returns False
        self.assertTrue(state.finished)


class TestProcessor(unittest.IsolatedAsyncioTestCase):
    """Smoke tests for Processor class — heavy pipecat dependencies."""

    def _make_processor(self, **kwargs):
        from calibrate_agent.llm.run_simulation import Processor, ConversationState

        state = ConversationState(max_turns=10)
        defaults = {
            "speaks_first": True,
            "conversation_state": state,
            "name": "TestProcessor",
            "role": "agent",
        }
        defaults.update(kwargs)
        return Processor(**defaults), state

    async def test_set_task_and_partner(self):
        proc, _state = self._make_processor()
        task = MagicMock()
        partner = MagicMock()
        proc.set_task(task)
        proc.set_partner(partner)
        self.assertIs(proc._task, task)
        self.assertIs(proc._partner_task, partner)

    async def test_save_intermediate_transcript(self):
        from calibrate_agent.llm.run_simulation import Processor, ConversationState

        with tempfile.TemporaryDirectory() as tmp:
            state = ConversationState(max_turns=5)
            ctx = MagicMock()
            ctx._messages = [
                {"role": "system", "content": "sp"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
            proc = Processor(
                speaks_first=True,
                conversation_state=state,
                context=ctx,
                output_dir=tmp,
            )
            proc._save_intermediate_transcript()
            transcript_path = Path(tmp) / "transcript.json"
            self.assertTrue(transcript_path.exists())
            data = json.loads(transcript_path.read_text())
            # System role filtered out
            self.assertEqual(len(data), 2)

    async def test_save_intermediate_transcript_no_context(self):
        proc, _ = self._make_processor()
        proc._save_intermediate_transcript()  # no-op, doesn't crash

    async def test_forward_to_partner_no_partner(self):
        proc, _ = self._make_processor()
        await proc._forward_to_partner("hi", run_partner=True)

    async def test_forward_to_partner_with_partner(self):
        proc, _ = self._make_processor()
        partner = MagicMock()
        partner.queue_frames = AsyncMock()
        proc.set_partner(partner)
        await proc._forward_to_partner("hi", run_partner=True)
        partner.queue_frames.assert_called_once()

    async def test_handle_completed_response_continues(self):
        proc, _state = self._make_processor()
        task = MagicMock()
        task.queue_frames = AsyncMock()
        proc.set_task(task)

        partner = MagicMock()
        partner.queue_frames = AsyncMock()
        proc.set_partner(partner)

        await proc._handle_completed_response("hello")
        # Continues, no end frame yet
        partner.queue_frames.assert_called_once()

    async def test_handle_completed_response_ends(self):
        proc, state = self._make_processor()
        state.max_turns = 1  # Will immediately end
        task = MagicMock()
        task.queue_frames = AsyncMock()
        proc.set_task(task)
        partner = MagicMock()
        partner.queue_frames = AsyncMock()
        proc.set_partner(partner)

        # First turn — records and hits max
        await proc._handle_completed_response("hello")
        self.assertTrue(state.finished)

    async def test_end_conversation(self):
        proc, _state = self._make_processor()
        task = MagicMock()
        task.queue_frames = AsyncMock()
        proc.set_task(task)
        partner = MagicMock()
        partner.queue_frames = AsyncMock()
        proc.set_partner(partner)

        await proc._end_conversation()
        # Should send EndFrame to partner and self
        task.queue_frames.assert_called_once()


class TestRunSimulationValidation(unittest.IsolatedAsyncioTestCase):
    async def test_empty_evaluators_raises(self):
        from calibrate_agent.llm.run_simulation import run_simulation

        with self.assertRaises(ValueError):
            await run_simulation(
                bot_system_prompt="bp",
                tools=[],
                user_system_prompt="up",
                evaluators=[],
            )


if __name__ == "__main__":
    unittest.main()
