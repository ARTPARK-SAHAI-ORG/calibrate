"""Tests for calibrate_agent/stt_resilience.py (ResilientSarvamSTTService).

No real network: the Sarvam SDK client is patched, and the socket / reconnect
methods are stubbed so we drive the transient-drop and reconnect paths directly.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from websockets.exceptions import ConnectionClosedError

from pipecat.frames.frames import ErrorFrame


def _make_service(**kwargs):
    """Build a ResilientSarvamSTTService with the SDK client patched out."""
    with patch("pipecat.services.sarvam.stt.AsyncSarvamAI"):
        from calibrate_agent.stt_resilience import ResilientSarvamSTTService

        return ResilientSarvamSTTService(
            api_key="sk-fake",
            mode="transcribe",
            settings=ResilientSarvamSTTService.Settings(model="saaras:v3"),
            **kwargs,
        )


async def _collect(agen):
    return [f async for f in agen]


class TestIsTransientWsDisconnect(unittest.TestCase):
    def test_connection_closed_family(self):
        from calibrate_agent.stt_resilience import is_transient_ws_disconnect

        self.assertTrue(is_transient_ws_disconnect(ConnectionClosedError(None, None)))

    def test_os_and_connection_errors(self):
        from calibrate_agent.stt_resilience import is_transient_ws_disconnect

        self.assertTrue(is_transient_ws_disconnect(ConnectionError("reset")))
        self.assertTrue(is_transient_ws_disconnect(OSError("broken pipe")))

    def test_message_marker_fallback(self):
        from calibrate_agent.stt_resilience import is_transient_ws_disconnect

        self.assertTrue(
            is_transient_ws_disconnect(RuntimeError("no close frame received or sent"))
        )
        self.assertTrue(is_transient_ws_disconnect(RuntimeError("connection closed")))

    def test_non_transient_rejected(self):
        from calibrate_agent.stt_resilience import is_transient_ws_disconnect

        self.assertFalse(is_transient_ws_disconnect(ValueError("bad model")))
        self.assertFalse(is_transient_ws_disconnect(RuntimeError("auth failed")))


class TestRunSttReconnect(unittest.IsolatedAsyncioTestCase):
    async def test_transient_send_drop_reconnects_and_retries(self):
        svc = _make_service()
        # First send raises a transient drop; after a successful reconnect the
        # retry send succeeds → NO ErrorFrame should be yielded.
        good_socket = AsyncMock()
        bad_socket = AsyncMock()
        bad_socket.transcribe.side_effect = ConnectionClosedError(None, None)
        svc._socket_client = bad_socket

        async def fake_reestablish():
            svc._socket_client = good_socket
            return True

        with patch.object(svc, "_reestablish", side_effect=fake_reestablish):
            frames = await _collect(svc.run_stt(b"\x00\x00" * 160))

        self.assertFalse([f for f in frames if isinstance(f, ErrorFrame)])
        good_socket.transcribe.assert_awaited_once()

    async def test_transient_send_drop_reconnect_fails_yields_error(self):
        svc = _make_service()
        bad_socket = AsyncMock()
        bad_socket.transcribe.side_effect = ConnectionClosedError(None, None)
        svc._socket_client = bad_socket

        async def fake_reestablish():
            svc._socket_client = None
            return False

        with patch.object(svc, "_reestablish", side_effect=fake_reestablish):
            frames = await _collect(svc.run_stt(b"\x00\x00" * 160))

        errors = [f for f in frames if isinstance(f, ErrorFrame)]
        self.assertEqual(len(errors), 1)
        self.assertIn("reconnect failed", errors[0].error)

    async def test_non_transient_error_does_not_reconnect(self):
        svc = _make_service()
        sock = AsyncMock()
        sock.transcribe.side_effect = ValueError("bad audio")
        svc._socket_client = sock

        reestablish = AsyncMock()
        with patch.object(svc, "_reestablish", reestablish):
            frames = await _collect(svc.run_stt(b"\x00\x00" * 160))

        reestablish.assert_not_called()
        errors = [f for f in frames if isinstance(f, ErrorFrame)]
        self.assertEqual(len(errors), 1)
        self.assertIn("Error sending audio to Sarvam", errors[0].error)

    async def test_transient_drop_during_shutdown_does_not_reconnect(self):
        svc = _make_service()
        svc._shutting_down = True
        sock = AsyncMock()
        sock.transcribe.side_effect = ConnectionClosedError(None, None)
        svc._socket_client = sock

        reestablish = AsyncMock()
        with patch.object(svc, "_reestablish", reestablish):
            frames = await _collect(svc.run_stt(b"\x00\x00" * 160))

        reestablish.assert_not_called()
        self.assertEqual(len([f for f in frames if isinstance(f, ErrorFrame)]), 1)

    async def test_no_socket_yields_none_without_error(self):
        svc = _make_service()
        svc._socket_client = None
        frames = await _collect(svc.run_stt(b"\x00\x00" * 160))
        self.assertNotIn(ErrorFrame, [type(f) for f in frames])


class TestReestablish(unittest.IsolatedAsyncioTestCase):
    async def test_succeeds_when_connect_restores_socket(self):
        svc = _make_service(max_reconnect_attempts=3)
        svc._socket_client = None
        good = AsyncMock()

        async def fake_connect():
            svc._socket_client = good

        with patch.object(svc, "_disconnect", AsyncMock()), \
                patch.object(svc, "_connect", side_effect=fake_connect), \
                patch("calibrate_agent.stt_resilience.asyncio.sleep", AsyncMock()):
            self.assertTrue(await svc._reestablish())

    async def test_exhausts_attempts_then_fails(self):
        svc = _make_service(max_reconnect_attempts=3)
        svc._socket_client = None

        disconnect = AsyncMock()
        connect = AsyncMock()  # never restores _socket_client
        with patch.object(svc, "_disconnect", disconnect), \
                patch.object(svc, "_connect", connect), \
                patch("calibrate_agent.stt_resilience.asyncio.sleep", AsyncMock()):
            self.assertFalse(await svc._reestablish())
        self.assertEqual(connect.await_count, 3)

    async def test_stops_early_when_shutting_down(self):
        svc = _make_service(max_reconnect_attempts=5)
        svc._socket_client = None
        svc._shutting_down = True

        connect = AsyncMock()
        with patch.object(svc, "_disconnect", AsyncMock()), \
                patch.object(svc, "_connect", connect), \
                patch("calibrate_agent.stt_resilience.asyncio.sleep", AsyncMock()):
            self.assertFalse(await svc._reestablish())
        connect.assert_not_called()

    async def test_already_healthy_socket_short_circuits(self):
        svc = _make_service()
        svc._socket_client = AsyncMock()  # another path already restored it

        connect = AsyncMock()
        with patch.object(svc, "_connect", connect):
            self.assertTrue(await svc._reestablish())
        connect.assert_not_called()


class TestReceiveTaskHandler(unittest.IsolatedAsyncioTestCase):
    async def test_transient_drop_is_quiet_no_push_error(self):
        svc = _make_service()
        sock = AsyncMock()
        sock.start_listening.side_effect = ConnectionClosedError(None, None)
        svc._socket_client = sock
        svc.push_error = AsyncMock()

        await svc._receive_task_handler()

        svc.push_error.assert_not_called()

    async def test_non_transient_error_pushes_error(self):
        svc = _make_service()
        sock = AsyncMock()
        sock.start_listening.side_effect = ValueError("protocol error")
        svc._socket_client = sock
        svc.push_error = AsyncMock()

        await svc._receive_task_handler()

        svc.push_error.assert_awaited_once()

    async def test_drop_during_shutdown_pushes_error(self):
        svc = _make_service()
        svc._shutting_down = True
        sock = AsyncMock()
        sock.start_listening.side_effect = ConnectionClosedError(None, None)
        svc._socket_client = sock
        svc.push_error = AsyncMock()

        await svc._receive_task_handler()

        # During shutdown a drop is surfaced (not swallowed as a recoverable event).
        svc.push_error.assert_awaited_once()

    async def test_no_socket_returns_immediately(self):
        svc = _make_service()
        svc._socket_client = None
        svc.push_error = AsyncMock()
        await svc._receive_task_handler()
        svc.push_error.assert_not_called()


class TestStopCancelSetShuttingDown(unittest.IsolatedAsyncioTestCase):
    async def test_stop_sets_flag(self):
        from pipecat.frames.frames import EndFrame

        svc = _make_service()
        with patch(
            "pipecat.services.sarvam.stt.SarvamSTTService.stop", AsyncMock()
        ):
            await svc.stop(EndFrame())
        self.assertTrue(svc._shutting_down)

    async def test_cancel_sets_flag(self):
        from pipecat.frames.frames import CancelFrame

        svc = _make_service()
        with patch(
            "pipecat.services.sarvam.stt.SarvamSTTService.cancel", AsyncMock()
        ):
            await svc.cancel(CancelFrame())
        self.assertTrue(svc._shutting_down)


if __name__ == "__main__":
    unittest.main()
