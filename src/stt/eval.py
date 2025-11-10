import asyncio
import argparse
from os.path import join, exists, basename, splitext
import os
import json
import wave
from contextlib import suppress
from pathlib import Path
from typing import Dict, List, Sequence, Literal

from dotenv import load_dotenv
from collections import defaultdict
from loguru import logger
import numpy as np
from natsort import natsorted

from pipecat.frames.frames import (
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    InputAudioRawFrame,
    EndTaskFrame,
    EndFrame,
    OutputTransportReadyFrame,
    UserStartedSpeakingFrame,
    UserSpeakingFrame,
)
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService, Language, LiveOptions

# from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
from pipecat.services.openai.stt import OpenAISTTService
from integrations.smallest.stt import SmallestSTTService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaLiveOptions
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.sarvam.stt import SarvamSTTService

from pipecat.transports.websocket.client import (
    WebsocketClientParams,
    WebsocketClientTransport,
)
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

from metrics import get_wer_score, get_llm_judge_score, get_string_similarity
import pandas as pd

load_dotenv(".env", override=True)


async def run_stt_bot(provider: str, language: Literal["english", "hindi"]):
    """Starts an STT-only bot that reports RTVI transcription messages."""

    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=False,
            add_wav_header=False,
            # audio_in_filter=KrispVivaFilter(),
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    class IOLogger(FrameProcessor):
        def __init__(self, agent: str, position: str):
            super().__init__()
            self._agent = agent
            self._position = position

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"bot frame: {frame}")
            logger.info(f"bot frame type: {type(frame)}")

            if isinstance(frame, UserSpeakingFrame):
                frame = RTVIServerMessageFrame(
                    data={
                        "type": "user-speaking",
                    }
                )
                asyncio.create_task(rtvi.push_frame(frame))

            await self.push_frame(frame, direction)

    input_logger = IOLogger(agent="eval", position="after_input")

    stt_language = (
        Language.EN
        if language == "english"
        else Language.HI if language == "hindi" else Language.KN
    )

    if provider == "deepgram":
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(language=stt_language.value),
        )
    elif provider == "sarvam":
        stt = SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
            language=stt_language,
        )
    elif provider == "google":
        stt = GoogleSTTService(
            params=GoogleSTTService.InputParams(
                languages=stt_language, model="chirp_3"
            ),
            location="us",
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
    elif provider == "openai":
        stt = OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-transcribe",
            language=stt_language,
        )
    elif provider == "cartesia":
        stt = CartesiaSTTService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            live_options=CartesiaLiveOptions(language=stt_language.value),
        )
    elif provider == "smallest":
        stt = SmallestSTTService(
            api_key=os.getenv("SMALLEST_API_KEY"),
            url="wss://waves-api.smallest.ai/api/v1/asr",
            params=SmallestSTTService.SmallestInputParams(
                audioLanguage=stt_language.value,
            ),
        )
    elif provider == "groq":
        stt = GroqSTTService(
            api_key=os.getenv("GROQ_API_KEY"),
            model="whisper-large-v3-turbo",
            language=Language.HI,
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")

    transcript = TranscriptProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            input_logger,
            stt,
            transcript.user(),
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            enable_metrics=True,
            allow_interruptions=True,
        ),
        observers=[RTVIObserver(rtvi)],
        idle_timeout_secs=200,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected to STT bot")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected from STT bot")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    logger.info("STT bot ready and waiting for connections")
    await runner.run(task)


async def run_stt_eval(audio_files: Sequence[Path]) -> List[Dict[str, str]]:
    """Connects to the STT bot and streams audio files sequentially."""

    transport = WebsocketClientTransport(
        uri="ws://localhost:8765",
        params=WebsocketClientParams(
            audio_in_enabled=False,
            audio_out_enabled=True,
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

    class TranscriptionWriter(FrameProcessor):
        def __init__(
            self,
            transcripts: list = [],
            audio_streamer=None,
        ):
            super().__init__(enable_direct_mode=True, name="TranscriptionWriter")
            self._transcripts = transcripts
            self._audio_streamer = audio_streamer

        def _is_final_user_transcript_message(
            self, frame: InputTransportMessageFrame
        ) -> str | bool:
            if hasattr(frame, "message"):
                message = frame.message
                if message.get("label") == "rtvi-ai":
                    msg_type = message.get("type")
                    if msg_type == "user-transcription":
                        data = message.get("data")
                        if data.get("final"):
                            return data["text"]
            return False

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"frame transcription logger: {frame}")
            # logger.info(f"frame transcription logger type: {type(frame)}")

            if isinstance(frame, InputTransportMessageFrame):
                if self._transcripts:
                    if user_transcript := self._is_final_user_transcript_message(frame):
                        logger.info(f"appending to user_transcript: {frame}")
                        self._transcripts[-1] += " " + user_transcript

            await self.push_frame(frame, direction)

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

    class BotTurnAudioStreamer(FrameProcessor):
        def __init__(
            self,
            audio_paths: List[Path],
            chunk_ms: int = 40,
            transcripts: list = [],
        ):
            super().__init__(enable_direct_mode=True, name="BotTurnAudioStreamer")
            self._audio_paths = audio_paths
            self._chunk_ms = chunk_ms
            self._current_audio_index = 0

            # States: 'await_bot_turn_end' -> 'streaming' -> 'await_reply_bot_turn_end' -> 'done'
            self._state = "await_bot_turn_end"
            self._output_ready = asyncio.Event()
            self._transcripts = transcripts
            self._pending_advance_task = None

        def _is_transcription_over(self, frame) -> bool:
            if isinstance(frame, InputTransportMessageFrame):
                if hasattr(frame, "message"):
                    message = frame.message
                    if message.get("label") == "rtvi-ai":
                        msg_type = message.get("type")
                        if msg_type == "user-transcription" and message.get("data").get(
                            "final"
                        ):
                            return True
            return False

        def _is_user_speaking(self, frame) -> bool:
            if isinstance(frame, InputTransportMessageFrame):
                if hasattr(frame, "message"):
                    message = frame.message
                    if message.get("label") == "rtvi-ai":
                        msg_type = message.get("type")
                        if msg_type == "user-started-speaking":
                            return True

                        if msg_type == "server-message":
                            if message.get("data").get("type") == "user-speaking":
                                return True

                        if msg_type == "user-transcription" and message.get("data").get(
                            "text"
                        ):
                            return True

            return False

        def _start_new_audio_streaming(
            self,
        ):
            self._state = "streaming"
            self._transcripts.append("")

            logger.info(f"transcripts length: {len(self._transcripts)}")
            asyncio.create_task(
                self._stream_audio(self._audio_paths[self._current_audio_index])
            )

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            # logger.info(f"frame: {frame}")
            is_bot_turn_over = self._is_transcription_over(frame)
            logger.info(f"is_bot_turn_over: {is_bot_turn_over}")

            if isinstance(frame, OutputTransportReadyFrame):
                self._output_ready.set()

            # Handle UserStartedSpeakingFrame during buffering
            if self._is_user_speaking(frame):
                if self._state == "buffering":
                    logger.info(
                        "User started speaking during buffering, cancelling pending advance"
                    )
                    if (
                        self._pending_advance_task
                        and not self._pending_advance_task.done()
                    ):
                        self._pending_advance_task.cancel()
                        self._pending_advance_task = None
                    self._state = "await_reply_bot_turn_end"

            logger.info(f"state: {self._state}")

            if self._state == "await_bot_turn_end":
                # Start streaming only after transcript signals end of bot input
                self._start_new_audio_streaming()

            elif self._state == "await_reply_bot_turn_end":
                if is_bot_turn_over:
                    if self._current_audio_index + 1 < len(self._audio_paths):

                        async def advance_and_stream():

                            await asyncio.sleep(5)
                            self._current_audio_index += 1

                            logger.info(
                                f"incrementing audio index: {self._current_audio_index}"
                            )

                            self._start_new_audio_streaming()

                        self._state = "buffering"
                        self._pending_advance_task = asyncio.create_task(
                            advance_and_stream()
                        )
                    else:

                        async def end_task_and_stream():

                            await asyncio.sleep(5)
                            logger.info(f"completed simulation message received")
                            await self.push_frame(
                                EndTaskFrame(), FrameDirection.UPSTREAM
                            )
                            await self.push_frame(EndFrame(), FrameDirection.UPSTREAM)
                            self._state = "done"

                        self._state = "buffering"
                        self._pending_advance_task = asyncio.create_task(
                            end_task_and_stream()
                        )

            # Pass every frame through unchanged
            await self.push_frame(frame, direction)

        async def _stream_audio(self, audio_path: Path):
            try:
                await self._output_ready.wait()

                logger.info(f"Starting new audio streaming: {audio_path}")

                if not audio_path.exists():
                    logger.error(f"Audio file not found for streaming: {audio_path}")
                    return

                with wave.open(str(audio_path), "rb") as source:
                    sample_rate = source.getframerate()
                    channels = source.getnchannels()
                    frames_per_chunk = max(
                        1, int(sample_rate * (self._chunk_ms / 1000.0))
                    )

                    while True:
                        data = source.readframes(frames_per_chunk)
                        if not data:
                            break

                        frame = OutputAudioRawFrame(
                            audio=data,
                            sample_rate=sample_rate,
                            num_channels=channels,
                        )
                        # Send audio downstream to the transport output
                        await self.push_frame(frame, FrameDirection.DOWNSTREAM)
                        await asyncio.sleep(1 / 1000.0)
            finally:
                # After sending our audio, wait for bot's reply to finish
                self._state = "await_reply_bot_turn_end"
                self._last_audio_ts = None
                logger.info(f"Finished streaming audio: {audio_path}")

    transcripts = []

    streamer = BotTurnAudioStreamer(
        audio_paths=audio_files,
        chunk_ms=40,
        transcripts=transcripts,
    )

    transcription_logger = TranscriptionWriter(transcripts, audio_streamer=streamer)

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
            transcription_logger,
            metrics_logger,
            streamer,  # After transcript, trigger streaming local audio as output
            transport.output(),  # Send our streamed audio to bot
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
        ),
        idle_timeout_secs=100,
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
        "transcripts": transcripts,
        "metrics": {
            "ttfb": ttfb,
            "processing_time": processing_time,
        },
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default="deepgram",
        choices=[
            "deepgram",
            "openai",
            "cartesia",
            "smallest",
            "groq",
            "google",
            "sarvam",
        ],
    )
    parser.add_argument(
        "-l", "--language", type=str, default="english", choices=["english", "hindi"]
    )
    parser.add_argument("-i", "--input-dir", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, default="./out")
    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_args()

    if not exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_dir = join(args.output_dir, f"{args.provider}_{args.language}")

    log_save_path = join(output_dir, "logs")

    if exists(log_save_path):
        os.remove(log_save_path)

    logger.add(log_save_path)

    audio_dir = join(args.input_dir, "audio")
    audio_files = natsorted(list(Path(audio_dir).glob("*_pcm16.wav")))
    # audio_files = natsorted(list(Path(audio_dir).glob("*.wav")))

    if args.debug:
        logger.debug(f"running in debug mode: using first 5 audio files for evaluation")
        audio_files = audio_files[:5]
        # audio_files = [
        #     file for file in audio_files if "3_21_english_baseline" in file.name
        # ]

    if not audio_files:
        raise ValueError(f"No *_pcm16.wav audio files found in {audio_dir}")

    logger.info(f"audio_files: {audio_files}")

    gt_file = join(args.input_dir, "gt.json")

    with open(gt_file, "r") as f:
        gt = json.load(f)

    bot_task = asyncio.create_task(
        run_stt_bot(provider=args.provider, language=args.language)
    )

    try:
        # Give the bot a moment to start listening before connecting.
        await asyncio.sleep(1.0)
        results = await run_stt_eval(audio_files)
        pred_transcripts = results["transcripts"]
        metrics = results["metrics"]
    finally:
        if not bot_task.done():
            bot_task.cancel()
            with suppress(asyncio.CancelledError):
                await bot_task

    ids_with_suffix = [splitext(basename(audio_file))[0] for audio_file in audio_files]
    # ids = [
    #     _id[: -len("_pcm16")] if _id.endswith("_pcm16") else _id
    #     for _id in ids_with_suffix
    # ]
    ids = ids_with_suffix
    gt_transcripts = [gt[id] for id in ids]
    logger.info(f"gt_transcripts: {gt_transcripts}")
    logger.info(f"pred_transcripts: {pred_transcripts}")
    logger.info(metrics)

    wer_score = get_wer_score(gt_transcripts, pred_transcripts)
    logger.info(f"WER: {wer_score['score']}")

    string_similarity = get_string_similarity(gt_transcripts, pred_transcripts)
    logger.info(f"String Similarity: {string_similarity['score']}")

    llm_judge_score = await get_llm_judge_score(gt_transcripts, pred_transcripts)
    logger.info(f"LLM Judge Score: {llm_judge_score['score']}")

    metrics_data = [
        {
            "wer": wer_score["score"],
        },
        {
            "string_similarity": string_similarity["score"],
        },
        {
            "llm_judge_score": llm_judge_score["score"],
        },
    ]

    data = []
    for (
        gt_transcript,
        pred_transcript,
        _id,
        wer,
        string_similarity,
        llm_judge_score,
    ) in zip(
        gt_transcripts,
        pred_transcripts,
        ids,
        wer_score["per_row"],
        string_similarity["per_row"],
        llm_judge_score["per_row"],
    ):
        data.append(
            {
                "id": _id,
                "gt": gt_transcript,
                "pred": pred_transcript,
                "wer": wer,
                "string_similarity": string_similarity,
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
        json.dump(metrics_data, f)

    pd.DataFrame(data).to_csv(join(output_dir, "error_analysis.csv"), index=False)


if __name__ == "__main__":
    asyncio.run(main())
