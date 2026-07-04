"""Tests for Processor.process_frame in llm/run_simulation.py."""

import asyncio
import unittest
from unittest.mock import patch, AsyncMock, MagicMock


class TestProcessorProcessFrame(unittest.IsolatedAsyncioTestCase):
    async def _make_processor(self):
        from calibrate_agent.llm.run_simulation import Processor, ConversationState

        state = ConversationState(max_turns=5)
        proc = Processor(speaks_first=True, conversation_state=state)
        return proc, state

    async def test_first_frame_marks_ready_speaks_first(self):
        from calibrate_agent.llm.run_simulation import Processor
        from pipecat.frames.frames import Frame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        proc, _ = await self._make_processor()
        task = MagicMock()
        task.queue_frames = AsyncMock()
        proc.set_task(task)

        frame = MagicMock(spec=Frame)

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(Processor, "push_frame", AsyncMock()):
            await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        # Queued LLMRunFrame
        task.queue_frames.assert_called_once()
        self.assertTrue(proc._ready)

    async def test_text_frame_accumulates(self):
        from calibrate_agent.llm.run_simulation import Processor
        from pipecat.frames.frames import TextFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        proc, _ = await self._make_processor()
        proc._ready = True  # skip queueing
        frame = MagicMock(spec=TextFrame)
        frame.text = "hello"

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(Processor, "push_frame", AsyncMock()):
            await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(proc._current_response, "hello")

    async def test_end_frame_with_response(self):
        from calibrate_agent.llm.run_simulation import Processor
        from pipecat.frames.frames import LLMFullResponseEndFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        proc, _ = await self._make_processor()
        proc._ready = True
        proc._current_response = "hello world"
        partner = MagicMock()
        partner.queue_frames = AsyncMock()
        proc.set_partner(partner)

        frame = MagicMock(spec=LLMFullResponseEndFrame)

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(Processor, "push_frame", AsyncMock()):
            await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(proc._current_response, "")
        partner.queue_frames.assert_called_once()

    async def test_end_frame_no_response_state_finished(self):
        from calibrate_agent.llm.run_simulation import Processor, ConversationState
        from pipecat.frames.frames import LLMFullResponseEndFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        state = ConversationState(max_turns=2)
        state.finished = True

        proc = Processor(speaks_first=True, conversation_state=state)
        proc._ready = True
        proc._current_response = ""  # empty response
        task = MagicMock()
        task.queue_frames = AsyncMock()
        proc.set_task(task)

        frame = MagicMock(spec=LLMFullResponseEndFrame)

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(Processor, "push_frame", AsyncMock()):
            await proc.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def test_no_speaks_first_no_run(self):
        from calibrate_agent.llm.run_simulation import Processor, ConversationState
        from pipecat.frames.frames import Frame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        state = ConversationState(max_turns=5)
        proc = Processor(speaks_first=False, conversation_state=state)
        # Set task but speaks_first=False so it won't queue
        task = MagicMock()
        task.queue_frames = AsyncMock()
        proc.set_task(task)
        frame = MagicMock(spec=Frame)

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(Processor, "push_frame", AsyncMock()):
            await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertTrue(proc._ready)
        task.queue_frames.assert_not_called()


if __name__ == "__main__":
    unittest.main()
