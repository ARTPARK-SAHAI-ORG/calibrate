from contextlib import suppress
import csv
import io
import sys
import os
import numpy as np
import json
from os.path import join, exists
import pandas as pd
import shutil

from agentloop.utils import (
    configure_print_logger,
    log_and_print,
    save_audio_chunk,
    MetricsLogger,
)

from pipecat.frames.frames import (
    InputTransportMessageFrame,
    BotStoppedSpeakingFrame,
    TTSStoppedFrame,
    TTSStartedFrame,
    UserSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotSpeakingFrame,
    EndFrame,
    EndTaskFrame,
    LLMRunFrame,
    EndTaskFrame,
    CancelFrame,
    OutputTransportMessageFrame,
    OutputTransportReadyFrame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from typing import Literal, Optional, List, Dict
from pathlib import Path
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.transports.websocket.client import (
    WebsocketClientParams,
    WebsocketClientTransport,
)
import asyncio
import uuid
from collections import defaultdict
import websockets
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from agentloop.integrations.smallest.tts import SmallestTTSService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.groq.tts import GroqTTSService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.sarvam.tts import SarvamTTSService

from pipecat.transcriptions.language import Language
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from loguru import logger
from pipecat.processors.transcript_processor import TranscriptProcessor
from agentloop.tts.metrics import get_tts_llm_judge_score
from dotenv import load_dotenv

load_dotenv(override=True)

WAIT_TIME_BETWEEN_TURNS = 5


async def run_tts_bot(
    provider: str,
    language: Literal["english", "hindi", "kannada"],
    audio_out_sample_rate=24000,
    port: int = 8765,
):
    """Starts a TTS-only bot that reports RTVI transcription messages."""

    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=False,
            audio_out_enabled=True,
            add_wav_header=False,
        ),
        port=port,
    )

    class IOLogger(FrameProcessor):
        def __init__(self):
            super().__init__()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"bot frame: {frame}")
            # logger.info(f"bot frame type: {type(frame)}")

            await self.push_frame(frame, direction)

    input_logger = IOLogger()

    class TransportMessageRouter(FrameProcessor):
        def __init__(self):
            super().__init__(enable_direct_mode=True, name="TransportMessageRouter")

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            if isinstance(frame, InputTransportMessageFrame):
                message = getattr(frame, "message", {}) or {}
                if (
                    message.get("label") == "rtvi-ai"
                    and message.get("type") == "send-text"
                ):
                    data = message.get("data") or {}
                    content = data.get("content")
                    if content:
                        await self.push_frame(LLMFullResponseStartFrame(), direction)
                        await self.push_frame(TextFrame(text=content), direction)
                        await self.push_frame(LLMFullResponseEndFrame(), direction)
                    return

            await self.push_frame(frame, direction)

    class Processor(FrameProcessor):
        def __init__(self):
            super().__init__(enable_direct_mode=True, name="Processor")

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            if isinstance(frame, MetricsFrame):
                payload = {
                    "ttfb": [],
                    "processing": [],
                }

                for datum in frame.data:
                    record = {"processor": datum.processor}
                    if getattr(datum, "model", None):
                        record["model"] = datum.model

                    if isinstance(datum, TTFBMetricsData):
                        record["value"] = datum.value
                        payload["ttfb"].append(record)
                    elif isinstance(datum, ProcessingMetricsData):
                        record["value"] = datum.value
                        payload["processing"].append(record)

                payload = {key: value for key, value in payload.items() if value}

                if payload:
                    await self.push_frame(
                        OutputTransportMessageFrame(
                            message={
                                "label": "rtvi-ai",
                                "type": "metrics",
                                "data": payload,
                            }
                        ),
                        FrameDirection.DOWNSTREAM,
                    )

            if isinstance(frame, BotStoppedSpeakingFrame):
                await self.push_frame(
                    OutputTransportMessageFrame(
                        message={
                            "label": "rtvi-ai",
                            "type": "bot-stopped-speaking",
                        }
                    ),
                    FrameDirection.DOWNSTREAM,
                )

            if isinstance(frame, BotStartedSpeakingFrame):
                await self.push_frame(
                    OutputTransportMessageFrame(
                        message={
                            "label": "rtvi-ai",
                            "type": "bot-started-speaking",
                        }
                    ),
                    FrameDirection.DOWNSTREAM,
                )

            if isinstance(frame, BotSpeakingFrame):
                await self.push_frame(
                    OutputTransportMessageFrame(
                        message={
                            "label": "rtvi-ai",
                            "type": "bot-still-speaking",
                        }
                    ),
                    FrameDirection.DOWNSTREAM,
                )

            await self.push_frame(frame, direction)

    tts_language = (
        Language.EN
        if language == "english"
        else Language.HI if language == "hindi" else Language.KN
    )

    if provider == "smallest":
        tts = SmallestTTSService(
            api_key=os.getenv("SMALLEST_API_KEY"),
            params=SmallestTTSService.InputParams(language=tts_language),
            voice_id=(
                "aarushi" if tts_language in [Language.EN, Language.HI] else "vijay"
            ),
        )
    elif provider == "cartesia":
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=(
                "95d51f79-c397-46f9-b49a-23763d3eaa2d"
                if language == "english"
                else (
                    "7c6219d2-e8d2-462c-89d8-7ecba7c75d65"
                    if language == "kannada"
                    else "f8f5f1b2-f02d-4d8e-a40d-fd850a487b3d"  # english
                )
            ),  # British Reading Lady
        )
    elif provider == "openai":
        tts = OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice="fable",
        )
    elif provider == "groq":
        tts = GroqTTSService(api_key=os.getenv("GROQ_API_KEY"))
    elif provider == "google":
        voice_id = (
            "en-US-Chirp3-HD-Charon"
            if language == "english"
            else (
                "hi-IN-Chirp3-HD-Charon"
                if language == "hindi"
                else "kn-IN-Chirp3-HD-Charon"
            )
        )
        tts = GoogleTTSService(
            voice_id=voice_id,
            params=GoogleTTSService.InputParams(language=tts_language),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
    elif provider == "elevenlabs":
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=(
                "jUjRbhZWoMK4aDciW36V"
                if language == "hindi"
                else "90ipbRoKi4CpHXvKVtl0"
            ),
            params=ElevenLabsTTSService.InputParams(language=tts_language),
        )
    elif provider == "sarvam":
        tts_language = (
            Language.KN_IN
            if language == "kannada"
            else Language.HI_IN if language == "hindi" else Language.EN_IN
        )
        tts = SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            model="bulbul:v2",
            voice_id="abhilash",
            params=SarvamTTSService.InputParams(language=tts_language),
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")

    transcript = TranscriptProcessor()
    processor = Processor()
    message_router = TransportMessageRouter()

    pipeline = Pipeline(
        [
            transport.input(),
            message_router,
            input_logger,
            tts,
            processor,
            transport.output(),
            transcript.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=audio_out_sample_rate,
            enable_metrics=True,
        ),
        observers=[DebugLogObserver()],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected to TTS bot")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected from TTS bot")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    logger.info("TTS bot ready and waiting for connections")
    await runner.run(task)


async def run_tts_eval(
    texts: List[str],
    output_dir: str,
    audio_in_sample_rate: int = 24000,
    port: int = 8765,
) -> List[Dict[str, str]]:
    """Connects to the TTS bot and streams audio files sequentially."""

    transport = WebsocketClientTransport(
        uri=f"ws://localhost:{port}",
        params=WebsocketClientParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
        ),
    )
    session = transport._session
    connect_lock = asyncio.Lock()
    original_connect = session.connect

    async def locked_connect(*args, **kwargs):
        async with connect_lock:
            if session._websocket:
                return
            return await original_connect(*args, **kwargs)

    session.connect = locked_connect
    # Workaround for race condition: manually initialize the audio queue before connection
    transport.input()._audio_in_queue = asyncio.Queue()

    class BotTurnTextStreamer(FrameProcessor):
        """Processor that captures LLM text output."""

        def __init__(
            self,
            texts: list[str],
            audio_output_dir: str,
            audio_save_paths: list[str],
        ):
            super().__init__(enable_direct_mode=True, name="Processor")
            self._ready = False
            self._texts = texts
            self._turn_index = 0
            self._audio_output_dir = audio_output_dir
            self._audio_save_paths = audio_save_paths
            self._pending_advance_task = None

        def set_task(self, task: "PipelineTask"):
            """Set the task reference after task creation."""
            self._task = task

        async def _send_text_frame(self):
            log_and_print(f"--------------------------------")
            log_and_print(
                f"Streaming text [{self._turn_index + 1}/{len(self._texts)}]: {self._texts[self._turn_index]}"
            )
            await self.push_frame(
                OutputTransportMessageFrame(
                    message={
                        "label": "rtvi-ai",
                        "type": "send-text",
                        "data": {
                            "content": self._texts[self._turn_index],
                            "options": {"audio_response": True},
                        },
                    }
                ),
                FrameDirection.DOWNSTREAM,
            )
            self._turn_index += 1

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"text output processor frame: {frame}")

            if not self._ready:
                self._ready = True
                if self._texts:
                    await self._send_text_frame()

            elif isinstance(frame, InputTransportMessageFrame):
                message = getattr(frame, "message", {})
                if isinstance(message, dict) and message.get("label") == "rtvi-ai":
                    if message.get("type") == "bot-started-speaking":
                        # Cancel any pending advance task when bot starts speaking
                        if (
                            self._pending_advance_task
                            and not self._pending_advance_task.done()
                        ):
                            self._pending_advance_task.cancel()
                            self._pending_advance_task = None

                        await self.push_frame(
                            UserStartedSpeakingFrame(),
                            FrameDirection.DOWNSTREAM,
                        )
                    elif message.get("type") == "bot-still-speaking":
                        await self.push_frame(
                            UserSpeakingFrame(), FrameDirection.DOWNSTREAM
                        )

                    elif message.get("type") == "bot-stopped-speaking":
                        await self.push_frame(
                            UserStoppedSpeakingFrame(),
                            FrameDirection.DOWNSTREAM,
                        )

                        async def end_task():
                            await asyncio.sleep(WAIT_TIME_BETWEEN_TURNS)
                            await self.push_frame(
                                EndTaskFrame(), FrameDirection.UPSTREAM
                            )

                        async def advance_and_send():
                            await asyncio.sleep(WAIT_TIME_BETWEEN_TURNS)
                            await self._send_text_frame()

                        if self._turn_index == len(self._texts):
                            task_fn = end_task
                        else:
                            task_fn = advance_and_send

                        self._pending_advance_task = asyncio.create_task(task_fn())

            elif isinstance(frame, InputAudioRawFrame):
                try:
                    import sounddevice as sd
                    import numpy as np

                    logger.info(f"has attr frame audio: {hasattr(frame, 'audio')}")
                    audio_save_path = os.path.join(
                        self._audio_output_dir, f"{self._turn_index}.wav"
                    )
                    if audio_save_path not in self._audio_save_paths:
                        self._audio_save_paths.append(audio_save_path)

                    await save_audio_chunk(
                        audio_save_path,
                        frame.audio,
                        frame.sample_rate,
                        getattr(frame, "num_channels", 1),
                    )

                    # Play OutputAudioRawFrame to the laptop sound device
                    # logger.info(
                    #     f"has attr frame sample_rate: {hasattr(frame, 'sample_rate')}"
                    # )

                    # if hasattr(frame, "audio") and hasattr(frame, "sample_rate"):
                    #     # Assume 16-bit signed PCM audio as bytes
                    #     audio_bytes = frame.audio
                    #     sample_rate = frame.sample_rate
                    #     # Convert bytes to numpy int16 array
                    #     audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                    #     # If the frame has a 'num_channels', handle accordingly
                    #     num_channels = getattr(frame, "num_channels", 1)
                    #     if num_channels > 1:
                    #         audio_np = audio_np.reshape(-1, num_channels)
                    #     sd.play(audio_np, sample_rate)
                    #     sd.wait()
                except Exception as e:
                    logger.warning(f"Failed to play audio frame: {e}")

            await self.push_frame(frame, direction)

    audio_output_dir = os.path.join(output_dir, "audios")

    if exists(audio_output_dir):
        shutil.rmtree(audio_output_dir)

    os.makedirs(audio_output_dir)

    audio_paths = []

    messages = [
        {
            "role": "system",
            "content": 'You are a helpful assistant. Begin the conversation with "Hello, how can I help you today?"',
        }
    ]
    text_streamer = BotTurnTextStreamer(
        texts=texts, audio_output_dir=audio_output_dir, audio_save_paths=audio_paths
    )

    ttfb = defaultdict(list)
    processing_time = defaultdict(list)

    metrics_logger = MetricsLogger(ttfb=ttfb, processing_time=processing_time)

    class IOLogger(FrameProcessor):
        def __init__(self):
            super().__init__()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"frame input logger: {frame}")

            await self.push_frame(frame, direction)

    input_logger = IOLogger()

    pipeline = Pipeline(
        [
            transport.input(),  # Bot audio coming in
            input_logger,
            # transcription_logger,
            metrics_logger,
            text_streamer,
            transport.output(),  # Send our streamed text to bot
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=audio_in_sample_rate,
        ),
        observers=[LLMLogObserver()],
    )

    @transport.event_handler("on_connected")
    async def on_connected(transport, client):
        logger.info(f"WebSocket connected")

    @transport.event_handler("on_disconnected")
    async def on_disconnected(transport, client):
        logger.info(f"WebSocket disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)

    return {
        "audio_paths": audio_paths,
        "metrics": {
            "ttfb": ttfb,
            "processing_time": processing_time,
        },
    }


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default="smallest",
        choices=[
            # "smallest",
            "cartesia",
            "openai",
            "groq",
            "google",
            "elevenlabs",
            "sarvam",  # there is a bug with the sarvam tts not returning audio frames
        ],
        help="TTS provider to use for evaluation",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="english",
        choices=["english", "hindi", "kannada"],
        help="Language of the audio files",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file containing the texts to synthesize",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./out",
        help="Path to the output directory to save the results",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Run the evaluation on the first 5 audio files",
    )
    parser.add_argument(
        "-dc",
        "--debug_count",
        help="Number of texts to run the evaluation on",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Websocket port to connect to the STT bot",
    )

    args = parser.parse_args()

    if args.provider in ["openai"]:
        audio_sample_rate = 24000
    else:
        audio_sample_rate = 16000

    bot_task = asyncio.create_task(
        run_tts_bot(
            provider=args.provider,
            language=args.language,
            audio_out_sample_rate=audio_sample_rate,
            port=args.port,
        )
    )

    output_dir = os.path.join(args.output_dir, args.provider)

    os.makedirs(output_dir, exist_ok=True)

    log_save_path = join(output_dir, "logs")

    if exists(log_save_path):
        os.remove(log_save_path)

    logger.remove()
    logger.add(log_save_path)

    print_log_save_path = join(output_dir, "results.log")

    if exists(print_log_save_path):
        os.remove(print_log_save_path)

    configure_print_logger(print_log_save_path)

    log_and_print(f"Running on port: {args.port}")
    log_and_print("--------------------------------")

    command = " ".join(sys.argv)
    log_and_print(f"\033[33mRunning command\033[0m: {command}")

    df = pd.read_csv(args.input)
    texts = df["text"].tolist()

    if args.debug:
        texts = texts[: args.debug_count]

    try:
        # Give the bot a moment to start listening before connecting.
        await asyncio.sleep(1.0)
        results = await run_tts_eval(
            texts, output_dir, audio_in_sample_rate=audio_sample_rate, port=args.port
        )
        audio_paths = results["audio_paths"]
        metrics = results["metrics"]
    finally:
        if not bot_task.done():
            bot_task.cancel()
            with suppress(asyncio.CancelledError):
                await bot_task

    llm_judge_score = await get_tts_llm_judge_score(audio_paths, texts)
    logger.info(f"LLM Judge Score: {llm_judge_score['score']}")

    metrics_data = [
        {
            "llm_judge_score": llm_judge_score["score"],
        },
    ]

    data = []
    for (
        text,
        audio_path,
        llm_judge_score,
    ) in zip(
        texts,
        audio_paths,
        llm_judge_score["per_row"],
    ):
        data.append(
            {
                "text": text,
                "audio_path": audio_path,
                "llm_judge_score": llm_judge_score["match"],
                "llm_judge_reasoning": llm_judge_score["reasoning"],
            }
        )

    for metric_name, metric_values in metrics.items():
        for processor, values in metric_values.items():
            mean = np.mean(values)
            std = np.std(values)
            metrics_data.append(
                {
                    "metric_name": metric_name,
                    "processor": processor,
                    "mean": mean,
                    "std": std,
                    "values": values,
                }
            )

    metrics_save_path = join(output_dir, "metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    pd.DataFrame(data).to_csv(join(output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    asyncio.run(main())
