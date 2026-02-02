import asyncio
import json
import urllib.parse
from typing import Any, AsyncGenerator, Dict, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the Smallest STT service, you need to `pip install websockets`."
    )
    raise Exception(f"Missing module: {e}")


class SmallestSTTService(WebsocketSTTService):
    """Speech-to-text service implementation for the Smallest.ai WebSocket API."""

    class SmallestInputParams(BaseModel):
        """Configuration options used to build the Smallest.ai connection string."""

        audioLanguage: str = Field(default=Language.EN.value)
        audioEncoding: str = "linear16"
        audioSampleRate: int = 16000
        audioChannels: int = 1
        addPunctuation: bool = True

        # Allow arbitrary additional parameters to be forwarded to the service.
        class Config:
            extra = "allow"
            populate_by_name = True

        @field_validator("audioLanguage", mode="before")
        @classmethod
        def _normalise_language(cls, value: Union[Language, str]) -> str:
            if isinstance(value, Language):
                return value.value
            if isinstance(value, str):
                return value
            raise ValueError("audioLanguage must be a Language or string")

        def to_query_params(self, api_key: Optional[str] = None) -> Dict[str, str]:
            params = self.model_dump(exclude_none=True, by_alias=True)
            if api_key:
                params["api_key"] = api_key

            normalised: Dict[str, str] = {}
            for key, value in params.items():
                if isinstance(value, bool):
                    normalised[key] = "true" if value else "false"
                else:
                    normalised[key] = str(value)

            return normalised

    def __init__(
        self,
        *,
        api_key: Optional[str],
        url: str = "wss://waves-api.smallest.ai/api/v1/asr",
        params: Optional["SmallestSTTService.SmallestInputParams"] = None,
        sample_rate: Optional[int] = None,
        reconnect_on_error: bool = True,
        **kwargs,
    ):
        self._api_key = api_key
        self._url = url.rstrip("?")
        self._input_params = params or self.SmallestInputParams()
        effective_sample_rate = sample_rate or int(self._input_params.audioSampleRate)

        super().__init__(
            sample_rate=effective_sample_rate,
            reconnect_on_error=reconnect_on_error,
            **kwargs,
        )

        self.set_model_name("smallest")

        self._receive_task: Optional[asyncio.Task] = None
        self._reset_connection_task: Optional[asyncio.Task] = None
        self._default_language = self._coerce_language(self._input_params.audioLanguage)
        self._awaiting_end_of_turn = False

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    def can_generate_metrics(self) -> bool:
        return True

    async def start_metrics(self):
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            if not frame.emulated:
                self._awaiting_end_of_turn = True
                await self.start_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if not frame.emulated:
                await self._send_end_of_stream()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            if self._reset_connection_task and not self._reset_connection_task.done():
                await self._reset_connection_task

            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if not self._websocket or self._websocket.state is not State.OPEN:
                error_message = "Smallest STT websocket is not connected"
                logger.error(error_message)
                yield ErrorFrame(error_message)
                return

            await self._websocket.send(audio)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            message = f"{self}: error sending audio to Smallest STT: {e}"
            logger.error(message)
            yield ErrorFrame(message)
            return

        yield None

    async def _connect(self):
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = asyncio.create_task(
                self._receive_task_handler(self._report_error)
            )

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            params = self._input_params.to_query_params(self._api_key)
            query_string = urllib.parse.urlencode(params)
            url = self._url

            if query_string:
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}{query_string}"

            logger.debug("Connecting to Smallest STT service")
            self._websocket = await websocket_connect(url)
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self}: unable to connect to Smallest STT: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Smallest STT service")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing Smallest websocket: {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        while True:
            websocket = self._websocket
            if not websocket:
                break

            async for message in websocket:
                await self._handle_message(message)

            logger.debug(f"{self} Smallest connection was disconnected, reconnecting")
            await self._connect_websocket()

            if not self._websocket or self._websocket.state is not State.OPEN:
                break

    async def _handle_message(self, message: Any):
        if isinstance(message, (bytes, bytearray)):
            # The Smallest API sends JSON messages; ignore unexpected binary payloads.
            return

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"{self} received non-JSON message: {message}")
            return

        if not isinstance(payload, dict):
            return

        logger.debug(f"{self} received payload: {payload}")

        if self._is_error_payload(payload):
            error_message = (
                payload.get("message") or payload.get("error") or json.dumps(payload)
            )
            await self.push_error(ErrorFrame(error_message))
            await self._call_event_handler("on_connection_error", error_message)
            return

        transcript = self._extract_transcript(payload)
        if not transcript:
            return

        language = self._extract_language(payload) or self._default_language
        is_final = self._is_final_payload(payload)

        await self.stop_ttfb_metrics()

        if is_final:
            await self.push_frame(
                TranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                    result=payload,
                )
            )
            await self._handle_transcription(transcript, is_final, language)
            await self.stop_processing_metrics()
            self._schedule_reset_connection()
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                    result=payload,
                )
            )

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Hook for OpenTelemetry tracing (implemented in decorator)."""
        pass

    def _extract_transcript(self, payload: Dict[str, Any]) -> Optional[str]:
        transcript = payload.get("text") or payload.get("transcript")
        if not transcript:
            return None

        if not isinstance(transcript, str):
            transcript = str(transcript)

        transcript = transcript.strip()
        return transcript or None

    def _extract_language(self, payload: Dict[str, Any]) -> Optional[Language]:
        for key in ("language", "detectedLanguage", "audioLanguage"):
            value = payload.get(key)
            if not value:
                continue
            if isinstance(value, Language):
                return value
            if isinstance(value, str):
                try:
                    return Language(value)
                except ValueError:
                    return None
        return None

    def _is_final_payload(self, payload: Dict[str, Any]) -> bool:
        flags = (
            payload.get("isFinal"),
            payload.get("is_final"),
            payload.get("isEndOfTurn"),
            payload.get("is_end_of_turn"),
            payload.get("final"),
            payload.get("isEnd"),
            payload.get("is_end"),
        )

        for flag in flags:
            if isinstance(flag, bool) and flag:
                return True
            if isinstance(flag, str) and flag.lower() == "true":
                return True

        payload_type = payload.get("type")
        if isinstance(payload_type, str) and payload_type.lower() in {
            "final",
            "final_transcript",
        }:
            return True

        return False

    def _is_error_payload(self, payload: Dict[str, Any]) -> bool:
        if payload.get("type") == "error":
            return True
        if "error" in payload and payload["error"]:
            return True
        return False

    async def _send_end_of_stream(self):
        if not self._awaiting_end_of_turn:
            return

        self._awaiting_end_of_turn = False

        if not self._websocket or self._websocket.state is not State.OPEN:
            return

        try:
            await self._websocket.send(b"")
        except Exception as e:
            logger.warning(f"{self} error sending end-of-stream marker: {e}")

    @staticmethod
    def _coerce_language(
        language_value: Union[Language, str, None],
    ) -> Optional[Language]:
        if not language_value:
            return None
        if isinstance(language_value, Language):
            return language_value
        try:
            return Language(language_value)
        except ValueError:
            return None

    def _schedule_reset_connection(self):
        if self._reset_connection_task and not self._reset_connection_task.done():
            return

        self._reset_connection_task = asyncio.create_task(self._reset_connection())

    async def _reset_connection(self):
        try:
            await self._disconnect()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                f"{self} error resetting connection after final transcript: {exc}"
            )
        finally:
            self._reset_connection_task = None
