from contextlib import suppress
import csv
import io
import aiofiles
import os
import struct
import wave
import numpy as np
import json
from os.path import join, exists
import pandas as pd
import shutil
from pipecat.frames.frames import (
    InputTransportMessageFrame,
    BotStoppedSpeakingFrame,
    TTSStoppedFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    BotStartedSpeakingFrame,
    EndFrame,
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
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
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
from integrations.smallest import SmallestTTSService
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
from dotenv import load_dotenv

load_dotenv(".env", override=True)


async def run_tts_bot(
    provider: str, language: Literal["english", "hindi"], audio_out_sample_rate=24000
):
    """Starts a TTS-only bot that reports RTVI transcription messages."""

    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=False,
            audio_out_enabled=True,
            add_wav_header=False,
        ),
    )

    class IOLogger(FrameProcessor):
        def __init__(self):
            super().__init__()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"bot frame: {frame}")
            logger.info(f"bot frame type: {type(frame)}")

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
            self._still_speaking = False

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

            if (
                isinstance(frame, BotStoppedSpeakingFrame)
                or isinstance(frame, TTSStoppedFrame)
            ) and self._still_speaking:
                logger.info(f"turning off speaking with frame: {frame}")
                self._still_speaking = False
                await self.push_frame(
                    OutputTransportMessageFrame(
                        message={
                            "label": "rtvi-ai",
                            "type": "bot-stopped-speaking",
                        }
                    ),
                    FrameDirection.DOWNSTREAM,
                )

            if (
                isinstance(frame, BotStartedSpeakingFrame)
                or isinstance(frame, TTSStartedFrame)
            ) and not self._still_speaking:
                self._still_speaking = True
                logger.info(f"turning on speaking with frame: {frame}")
                await self.push_frame(
                    OutputTransportMessageFrame(
                        message={
                            "label": "rtvi-ai",
                            "type": "bot-started-speaking",
                        }
                    ),
                    FrameDirection.DOWNSTREAM,
                )

            if isinstance(frame, CancelFrame):
                logger.info("received cancel frame")
                logger.info(f"still speaking: {self._still_speaking}")

            if isinstance(frame, CancelFrame) and self._still_speaking:
                self._still_speaking = False
                await self.push_frame(
                    OutputTransportMessageFrame(
                        message={
                            "label": "rtvi-ai",
                            "type": "bot-stopped-speaking",
                        }
                    ),
                    FrameDirection.DOWNSTREAM,
                )
                return

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
            voice_id="aarushi",
        )
    elif provider == "cartesia":
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
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
            "en-US-Chirp3-HD-Charon" if language == "english" else "hi-IN-Standard-A"
        )
        tts = GoogleTTSService(
            voice_id=voice_id,
            params=GoogleTTSService.InputParams(language=tts_language),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
    elif provider == "elevenlabs":
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="Ui0HFqLn4HkcAenlJJVJ",
            params=ElevenLabsTTSService.InputParams(language=tts_language),
        )
    elif provider == "sarvam":
        tts = SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            model="bulbul:v2",
            voice_id="anushka",
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
        idle_timeout_secs=5,
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


async def save_audio_chunk(
    path: str, audio_chunk: bytes, sample_rate: int, num_channels: int
):
    if len(audio_chunk) == 0:
        logger.warning(f"There's no audio to save for {path}")
        return

    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists():
        logger.debug(f"Creating new audio file for {path} at {filepath}")
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_chunk)
            async with aiofiles.open(filepath, "wb") as file:
                await file.write(buffer.getvalue())
    else:
        logger.debug(f"Appending audio chunk for {path} to {filepath}")
        async with aiofiles.open(filepath, "rb+") as file:
            current_size = await file.seek(0, os.SEEK_END)
            if current_size < 44:
                logger.error(
                    f"Existing audio file {filepath} is too small to be a valid WAV; rewriting"
                )
                await file.seek(0)
                await file.truncate(0)
                with io.BytesIO() as buffer:
                    with wave.open(buffer, "wb") as wf:
                        wf.setsampwidth(2)
                        wf.setnchannels(num_channels)
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio_chunk)
                    await file.write(buffer.getvalue())
                return

            await file.write(audio_chunk)
            new_size = current_size + len(audio_chunk)
            data_chunk_size = max(0, new_size - 44)

            await file.seek(40)
            await file.write(struct.pack("<I", data_chunk_size))

            await file.seek(4)
            await file.write(struct.pack("<I", new_size - 8))

            await file.flush()


async def run_tts_eval(
    texts: List[str], output_dir: str, audio_in_sample_rate: int = 24000
) -> List[Dict[str, str]]:
    """Connects to the TTS bot and streams audio files sequentially."""

    transport = WebsocketClientTransport(
        uri="ws://localhost:8765",
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

    class MetricsLogger(FrameProcessor):
        def __init__(
            self,
            ttfb: defaultdict,
            processing_time: defaultdict,
        ):
            super().__init__(enable_direct_mode=True, name="MetricsLogger")
            self._ttfb = ttfb
            self._processing_time = processing_time

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            if isinstance(frame, InputTransportMessageFrame):
                message = getattr(frame, "message", {})
                if isinstance(message, dict) and message.get("label") == "rtvi-ai":
                    if message.get("type") == "metrics" and message.get("data"):
                        if message.get("data").get("ttfb"):
                            for d in message.get("data").get("ttfb"):
                                if not d.get("value"):
                                    continue
                                self._ttfb[d.get("processor")].append(d.get("value"))
                        if message.get("data").get("processing"):
                            for d in message.get("data").get("processing"):
                                if not d.get("value"):
                                    continue
                                self._processing_time[d.get("processor")].append(
                                    d.get("value")
                                )

            await self.push_frame(frame, direction)

    class BotTurnTextStreamer(FrameProcessor):
        """Processor that captures LLM text output."""

        def __init__(
            self,
            texts: list[str],
        ):
            super().__init__(enable_direct_mode=True, name="Processor")
            self._ready = False
            self._texts = texts
            self._turn_index = 0

        def set_task(self, task: "PipelineTask"):
            """Set the task reference after task creation."""
            self._task = task

        async def _send_text_frame(self):
            logger.info(f"pushing: {self._texts[self._turn_index]}")
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
                        await self.push_frame(
                            UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM
                        )
                    elif message.get("type") == "bot-stopped-speaking":
                        await self.push_frame(
                            UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
                        )

                        if self._turn_index == len(self._texts):
                            await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
                        else:
                            await asyncio.sleep(1)
                            await self._send_text_frame()

            await self.push_frame(frame, direction)

    audio_output_dir = os.path.join(output_dir, "audios")

    if exists(audio_output_dir):
        shutil.rmtree(audio_output_dir)

    os.makedirs(audio_output_dir)

    audio_paths = []

    # streamer = BotTurnAudioStreamer(
    #     audio_paths=audio_files,
    #     chunk_ms=40,
    #     transcripts=transcripts,
    # # )
    # llm = OpenAILLMService(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     model="gpt-4.1",
    # )

    messages = [
        {
            "role": "system",
            "content": 'You are a helpful assistant. Begin the conversation with "Hello, how can I help you today?"',
        }
    ]
    # context = LLMContext(
    #     messages,
    # )
    # context_aggregator = LLMContextAggregatorPair(context)
    text_streamer = BotTurnTextStreamer(texts=texts)
    # transcription_logger = TranscriptionWriter(transcripts, audio_streamer=streamer)

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

    audio_buffer = AudioBufferProcessor(enable_turn_audio=True)

    pipeline = Pipeline(
        [
            transport.input(),  # Bot audio coming in
            input_logger,
            # transcription_logger,
            metrics_logger,
            text_streamer,
            transport.output(),  # Send our streamed text to bot
            audio_buffer,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=audio_in_sample_rate,
        ),
        observers=[LLMLogObserver()],
        idle_timeout_secs=5,
    )

    # text_streamer.set_task(task)

    @audio_buffer.event_handler("on_user_turn_audio_data")
    async def on_user_turn_audio_data(buffer, audio, sample_rate, num_channels):
        logger.info(f"Audio data received")
        logger.info(f"turn index: {text_streamer._turn_index}")
        audio_save_path = os.path.join(
            audio_output_dir, f"{text_streamer._turn_index}.wav"
        )
        audio_paths.append(audio_save_path)
        await save_audio_chunk(audio_save_path, audio, sample_rate, num_channels)

    @transport.event_handler("on_connected")
    async def on_connected(transport, client):
        logger.info(f"WebSocket connected")
        await audio_buffer.start_recording()

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
            "smallest",
            "cartesia",
            "openai",
            "groq",
            "google",
            "elevenlabs",
            "sarvam",
        ],
        help="TTS provider to use for evaluation",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="english",
        choices=["english", "hindi"],
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
        )
    )

    output_dir = os.path.join(args.output_dir, args.provider)

    os.makedirs(output_dir, exist_ok=True)

    log_save_path = join(output_dir, "logs")

    if exists(log_save_path):
        os.remove(log_save_path)

    logger.add(log_save_path)

    df = pd.read_csv(args.input)
    texts = df["text"].tolist()

    try:
        # Give the bot a moment to start listening before connecting.
        await asyncio.sleep(1.0)
        results = await run_tts_eval(
            texts, output_dir, audio_in_sample_rate=audio_sample_rate
        )
        audio_paths = results["audio_paths"]
        metrics = results["metrics"]
    finally:
        if not bot_task.done():
            bot_task.cancel()
            with suppress(asyncio.CancelledError):
                await bot_task

    print(results)

    metrics_data = [
        # {
        #     "wer": wer_score["score"],
        # },
        # {
        #     "string_similarity": string_similarity["score"],
        # },
        # {
        #     "llm_judge_score": llm_judge_score["score"],
        # },
    ]

    data = []
    for (
        text,
        audio_path,
    ) in zip(
        texts,
        audio_paths,
    ):
        data.append(
            {
                "text": text,
                "audio_path": audio_path,
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
        json.dump(metrics_data, f)

    pd.DataFrame(data).to_csv(join(output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    asyncio.run(main())
