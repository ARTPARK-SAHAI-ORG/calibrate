"""Resilience helpers for streaming STT services.

Most pipecat streaming STT services already reconnect transparently on a dropped
websocket: Cartesia, ElevenLabs (realtime), and Smallest inherit pipecat's
``WebsocketService`` (``reconnect_on_error=True`` by default), while Deepgram and
Google reconnect inside their own SDKs. ``SarvamSTTService`` is the exception — it
drives the Sarvam SDK's socket directly and has no reconnection, so a transient
websocket drop ("no close frame received or sent") permanently kills the stream:
``run_stt`` yields a (non-fatal) ``ErrorFrame`` and leaves ``_socket_client`` dead,
so no further transcriptions arrive and a voice simulation stalls out.

``ResilientSarvamSTTService`` closes that gap. It intercepts the transient-drop
exception on both the send path (``run_stt``) and the receive path
(``_receive_task_handler``) and reconnects in place via pipecat's own
``_disconnect``/``_connect``, bringing Sarvam to parity with the other streaming
providers. Genuinely terminal failures (reconnect exhausted) still surface.

See ``calibrate_agent/agent/run_simulation.py`` for the companion change: the
simulation's ERROR-log sink is taught to treat the transient-drop / mid-reconnect
ERROR logs (from pipecat's reconnect *and* this class) as recoverable, so it does
not cancel the whole pipeline before reconnection completes.
"""

import asyncio
import base64
from typing import AsyncGenerator, Optional

from loguru import logger
from websockets.exceptions import ConnectionClosed, WebSocketException

from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame, Frame
from pipecat.services.sarvam.stt import SarvamSTTService

# Substrings that identify a transient websocket disconnect we should recover from
# rather than fail on. The classic Sarvam symptom is the websockets library's
# "no close frame received or sent"; the others cover abrupt drops / idle kills.
_TRANSIENT_WS_MESSAGE_MARKERS = (
    "no close frame received or sent",
    "connection closed",
    "connectionclosed",
    "keepalive ping timeout",
    "going away",
    "1006",
    "1011",
)

# Reconnect backoff bounds (seconds). Kept short so a mid-conversation drop
# recovers quickly rather than stalling the simulation.
_RECONNECT_MIN_WAIT = 0.5
_RECONNECT_MAX_WAIT = 4.0
_DEFAULT_MAX_RECONNECT_ATTEMPTS = 4


def is_transient_ws_disconnect(exc: BaseException) -> bool:
    """True when ``exc`` looks like a recoverable streaming-STT websocket drop.

    Matches the websockets ``ConnectionClosed`` family and low-level socket errors
    directly, plus a message-substring fallback for wrappers that stringify the
    underlying cause without preserving the type.
    """
    if isinstance(exc, (ConnectionClosed, WebSocketException, ConnectionError, OSError)):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_WS_MESSAGE_MARKERS)


class ResilientSarvamSTTService(SarvamSTTService):
    """``SarvamSTTService`` that reconnects on transient websocket drops.

    Behaviour is identical to the base service except that a transient disconnect
    triggers a bounded, backed-off reconnect instead of a dead stream:

    - **Send path** (:meth:`run_stt`): on a transient failure the socket is
      re-established and the audio chunk is retried once. Only a non-transient
      error, an in-progress shutdown, or exhausted reconnect attempts yield an
      ``ErrorFrame``.
    - **Receive path** (:meth:`_receive_task_handler`): a transient failure returns
      quietly (logged at WARNING). The next audio send re-establishes the socket,
      which spawns a fresh receive task via ``_connect`` — so we avoid both a noisy
      ``ErrorFrame`` and the hazard of reconnecting from within the receive task
      that a full ``_disconnect`` would cancel.
    """

    def __init__(
        self,
        *args,
        max_reconnect_attempts: int = _DEFAULT_MAX_RECONNECT_ATTEMPTS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_lock = asyncio.Lock()
        # Set once a real shutdown (stop/cancel) is under way so the reconnect
        # paths don't fight the intentional teardown.
        self._shutting_down = False

    async def stop(self, frame: EndFrame):
        self._shutting_down = True
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._shutting_down = True
        await super().cancel(frame)

    async def _send_audio(self, audio: bytes) -> None:
        """Send one audio chunk over the current Sarvam socket.

        Mirrors the encoding/method selection in the base ``run_stt`` so the
        reconnect-and-retry path sends bytes exactly as the service normally does.
        """
        audio_base64 = base64.b64encode(audio).decode("utf-8")
        encoding = (
            self._input_audio_codec
            if self._input_audio_codec.startswith("audio/")
            else f"audio/{self._input_audio_codec}"
        )
        method_kwargs = {
            "audio": audio_base64,
            "encoding": encoding,
            "sample_rate": self.sample_rate,
        }
        if self._config.use_translate_method:
            await self._socket_client.translate(**method_kwargs)
        else:
            await self._socket_client.transcribe(**method_kwargs)

    async def _reestablish(self) -> bool:
        """Reconnect the Sarvam socket with bounded, backed-off retries.

        Serialised by a lock so the send and receive paths can't reconnect
        concurrently. Returns True once ``_connect`` has produced a live
        ``_socket_client``; False if shutting down or attempts are exhausted.
        """
        async with self._reconnect_lock:
            # Another path may have already restored the socket while we waited
            # for the lock.
            if self._socket_client is not None:
                return True
            for attempt in range(1, self._max_reconnect_attempts + 1):
                if self._shutting_down:
                    return False
                try:
                    await self._disconnect()
                    await self._connect()
                except Exception as e:  # noqa: BLE001 - retry regardless of cause
                    logger.warning(
                        f"{self} Sarvam STT reconnect attempt {attempt} error: {e}"
                    )
                if self._socket_client is not None:
                    logger.info(
                        f"{self} reconnected to Sarvam STT on attempt {attempt}"
                    )
                    return True
                wait = min(_RECONNECT_MIN_WAIT * 2 ** (attempt - 1), _RECONNECT_MAX_WAIT)
                await asyncio.sleep(wait)
            logger.error(
                f"{self} failed to reconnect after {self._max_reconnect_attempts} "
                "attempts (Sarvam STT)"
            )
            return False

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio, reconnecting and retrying once on a transient drop."""
        if not self._socket_client:
            yield None
            return

        try:
            await self._send_audio(audio)
        except Exception as e:  # noqa: BLE001 - classify below
            if self._shutting_down or not is_transient_ws_disconnect(e):
                yield ErrorFrame(error=f"Error sending audio to Sarvam: {e}", exception=e)
            else:
                logger.warning(
                    f"{self} transient Sarvam STT disconnect on send ({e}); reconnecting"
                )
                if await self._reestablish() and self._socket_client:
                    try:
                        await self._send_audio(audio)
                    except Exception as retry_error:  # noqa: BLE001
                        yield ErrorFrame(
                            error=(
                                "Error sending audio to Sarvam after reconnect: "
                                f"{retry_error}"
                            ),
                            exception=retry_error,
                        )
                else:
                    yield ErrorFrame(
                        error=f"Sarvam STT reconnect failed after transient disconnect: {e}",
                        exception=e,
                    )

        yield None

    async def _receive_task_handler(self):
        """Listen for transcriptions; recover quietly from a transient drop.

        On a transient disconnect we return without pushing an error: the next
        audio send triggers ``_reestablish`` (a full ``_connect``), which spawns a
        replacement receive task. Reconnecting here instead would mean a
        ``_disconnect`` cancelling the very task we're running in.
        """
        if not self._socket_client:
            return

        try:
            await self._socket_client.start_listening()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - classify below
            if self._shutting_down or not is_transient_ws_disconnect(e):
                await self.push_error(
                    error_msg=f"Sarvam receive task error: {e}", exception=e
                )
                return
            logger.warning(
                f"{self} transient Sarvam STT disconnect on receive ({e}); "
                "awaiting reconnect on next audio"
            )
