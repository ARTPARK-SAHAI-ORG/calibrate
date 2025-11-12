#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
# src/pipecat/services/smallest/tts.py

import base64
import json
import uuid
from enum import Enum, StrEnum
from typing import AsyncGenerator, List, Optional, Union

import websockets
from websockets.protocol import CLOSED, OPEN
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts


class SmallestTTSModel(StrEnum):
    """Supported models for the Smallest API."""

    LIGHTNING_V2 = "lightning-v2"


def language_to_smallest_language(language: Language) -> Optional[str]:
    BASE_LANGUAGES = {
        Language.AR: "ar",
        Language.BN: "bn",
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.IT: "it",
        Language.KN: "kn",
        Language.MR: "mr",
        Language.NL: "nl",
        Language.PL: "pl",
        Language.RU: "ru",
        Language.TA: "ta",
    }

    result = BASE_LANGUAGES.get(language)

    if not result:
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


def get_smallest_url(model: str) -> str:
    if model == SmallestTTSModel.LIGHTNING_V2:
        return "wss://waves-api.smallest.ai/api/v1/lightning-v2/get_speech/stream?timeout=60"
    else:
        raise ValueError(f"Invalid model: {model}")


class SmallestTTSService(InterruptibleTTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        speed: Optional[Union[str, float]] = 1.0
        consistency: Optional[float] = Field(default=0.5, ge=0, le=1)
        similarity: Optional[float] = Field(default=0, ge=0, le=1)
        enhancement: Optional[int] = Field(default=1, ge=0, le=2)
        model: Optional[str] = "lightning-v2"

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        url: str = "",
        model: str = "lightning-v2",
        sample_rate: Optional[int] = 24000,
        params: InputParams = InputParams(),
        text_aggregator: Optional[BaseTextAggregator] = None,
        **kwargs,
    ):
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            text_aggregator=text_aggregator
            or SkipTagsAggregator([("<spell>", "</spell>")]),
            **kwargs,
        )

        self._api_key = api_key
        self._url = get_smallest_url(model)
        self._settings = {
            "output_format": {
                "container": "wav",
                "sample_rate": 0,
            },
            "language": (
                self.language_to_service_language(params.language)
                if params.language
                else "en"
            ),
            "speed": params.speed,
            "consistency": params.consistency,
            "similarity": params.similarity,
            "enhancement": params.enhancement,
        }
        self.set_model_name(model)
        self.set_voice(voice_id)
        self._websocket = None
        self._receive_task = None
        self._request_id = None

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        self._model_id = model
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_smallest_language(language)

    def _build_msg(self, text: str = ""):
        voice_config = {}
        voice_config["mode"] = "id"
        voice_config["id"] = self._voice_id

        msg = {
            "text": text,
            "voice_id": self._voice_id,
            "language": self._settings["language"],
            "speed": self._settings["speed"],
            "consistency": self._settings["consistency"],
            "similarity": self._settings["similarity"],
            "enhancement": self._settings["enhancement"],
        }

        if self._request_id:
            msg["request_id"] = self._request_id

        return msg

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error)
            )

    async def _disconnect(self):
        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        logger.error(f"{self._url}")
        try:
            if self._websocket and self._websocket.state is OPEN:
                return
            self._websocket = await websockets.connect(
                f"{self._url}",
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket and self._websocket.state is not CLOSED:
                logger.debug("Disconnecting from Smallest AI")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._context_id = None
            self._websocket = None

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(
        self, frame: StartInterruptionFrame, direction: FrameDirection
    ):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._request_id = None

    async def _receive_messages(self):
        async for message in self._get_websocket():
            msg = json.loads(message)

            if msg["status"] == "complete":
                msg_request_id = msg.get("request_id")
                if (
                    self._request_id
                    and msg_request_id
                    and msg_request_id == self._request_id
                ):
                    await self.stop_all_metrics()
                    await self.push_frame(TTSStoppedFrame())
                    self._request_id = None
            elif msg["status"] == "chunk":
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(msg["data"]["audio"]),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                await self.push_frame(frame)
            elif msg["status"] == "error":
                logger.error(f"{self} error: {msg}")
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(ErrorFrame(f"{self} error: {msg['error']}"))
                self._request_id = None
            else:
                logger.error(f"{self} error, unknown message type: {msg}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        try:
            if not self._websocket or self._websocket.state is not OPEN:
                await self._connect()
            if not self._request_id:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._request_id = str(uuid.uuid4())
            try:
                msg = self._build_msg(text=text)
                await self._get_websocket().send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
                await self.start_ttfb_metrics()
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(f"{self} error: {str(e)}")
