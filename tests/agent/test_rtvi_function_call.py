"""Tests for RTVIFunctionCallResponder in agent/run_simulation.py."""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock


def _make_responder(tool_calls=None, ctx=None, webhooks=None):
    from calibrate_agent.agent.run_simulation import RTVIFunctionCallResponder

    if ctx is None:
        ctx = MagicMock()
        ctx.get_messages.return_value = []
    return RTVIFunctionCallResponder(
        tool_calls=tool_calls if tool_calls is not None else [],
        context=ctx,
        webhook_configs=webhooks or {},
    )


class TestRTVIFunctionCallResponder(unittest.IsolatedAsyncioTestCase):
    async def test_set_frame_sender(self):
        responder = _make_responder()
        sender = MagicMock()
        responder.set_frame_sender(sender)
        self.assertIs(responder._send_frame, sender)

    async def test_set_end_call_callback(self):
        responder = _make_responder()
        cb = MagicMock()
        responder.set_end_call_callback(cb)
        self.assertIs(responder._end_call_callback, cb)

    async def test_execute_end_call_with_reason(self):
        responder = _make_responder()
        end_cb = AsyncMock()
        responder.set_end_call_callback(end_cb)
        result, post_cb = await responder._execute_function(
            "end_call", {"reason": "done"}
        )
        self.assertEqual(result["acknowledged"], True)
        self.assertEqual(result["reason"], "done")
        await post_cb()
        end_cb.assert_called_once_with("done")

    async def test_execute_end_call_no_reason(self):
        responder = _make_responder()
        result, post_cb = await responder._execute_function("end_call", {})
        self.assertEqual(result, {"acknowledged": True})
        await post_cb()  # Should not crash even without callback set

    async def test_execute_webhook(self):
        from calibrate_agent.agent import run_simulation as RS

        responder = _make_responder(
            webhooks={"fn1": {"url": "http://x", "method": "GET", "headers": []}}
        )
        with patch.object(
            RS, "make_webhook_call", AsyncMock(return_value={"status": "success"})
        ):
            result, _ = await responder._execute_function("fn1", {"x": 1})
        self.assertEqual(result["status"], "success")

    async def test_execute_unknown_function(self):
        responder = _make_responder()
        result, _ = await responder._execute_function("other_fn", {})
        self.assertEqual(result, {"status": "received"})

    async def test_send_result_message_no_sender(self):
        responder = _make_responder()
        await responder._send_result_message("fn", "id1", {}, {"ok": True})
        # Should not crash without sender set

    async def test_send_result_message_with_sender(self):
        responder = _make_responder()
        sender = AsyncMock()
        responder.set_frame_sender(sender)
        await responder._send_result_message("fn", "id1", {"a": 1}, {"ok": True})
        sender.assert_called_once()

    async def test_process_frame_function_call(self):
        from calibrate_agent.agent.run_simulation import RTVIFunctionCallResponder
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        responder = _make_responder()
        sender = AsyncMock()
        responder.set_frame_sender(sender)

        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {
            "label": "rtvi-ai",
            "type": "llm-function-call",
            "data": {
                "function_name": "test_fn",
                "tool_call_id": "call_1",
                "args": {"x": 1},
            },
        }

        with (
            patch.object(FrameProcessor, "process_frame", AsyncMock()),
            patch.object(RTVIFunctionCallResponder, "push_frame", AsyncMock()),
        ):
            await responder.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(len(responder._tool_calls), 1)
        sender.assert_called_once()

    async def test_process_frame_other(self):
        from calibrate_agent.agent.run_simulation import RTVIFunctionCallResponder
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        responder = _make_responder()
        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {"label": "other"}

        with (
            patch.object(FrameProcessor, "process_frame", AsyncMock()),
            patch.object(RTVIFunctionCallResponder, "push_frame", AsyncMock()),
        ):
            await responder.process_frame(frame, FrameDirection.DOWNSTREAM)


if __name__ == "__main__":
    unittest.main()
