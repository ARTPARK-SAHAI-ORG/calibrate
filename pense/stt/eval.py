import asyncio
import argparse
import sys
from os.path import join, exists, basename, splitext
import os
import json
import wave
from datetime import datetime
from contextlib import suppress
from pathlib import Path
from typing import Dict, List, Sequence, Literal, Optional

from collections import defaultdict
from loguru import logger

from pense.utils import (
    current_context,
    add_default_source,
    configure_print_logger,
    log_and_print,
    MetricsLogger,
)
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
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService, Language, LiveOptions
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

# from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
from pipecat.services.openai.stt import OpenAISTTService

from pense.integrations.smallest.stt import SmallestSTTService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaLiveOptions
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver

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

from pense.stt.metrics import (
    get_wer_score,
    get_llm_judge_score,
    get_string_similarity,
)
import pandas as pd


async def run_stt_bot(
    provider: str, language: Literal["english", "hindi", "kannada"], port: int
):
    """Starts an STT-only bot that reports RTVI transcription messages."""
    current_context.set("BOT")

    transport = WebsocketServerTransport(
        port=port,
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
        def __init__(self, position: str):
            super().__init__()
            self._position = position

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"{self._position} bot frame: {frame}")

            if isinstance(frame, UserSpeakingFrame):
                frame = RTVIServerMessageFrame(
                    data={
                        "type": "user-speaking",
                    }
                )
                asyncio.create_task(rtvi.push_frame(frame))

            await self.push_frame(frame, direction)

    input_logger = IOLogger(position="input")
    output_logger = IOLogger(position="output")

    stt_language = (
        Language.KN
        if language == "kannada"
        else Language.HI if language == "hindi" else Language.EN
    )

    if provider == "deepgram":
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(language=stt_language.value, encoding="linear16"),
        )
    # elif provider == "deepgram-flux":
    #     stt = DeepgramFluxSTTService(
    #         api_key=os.getenv("DEEPGRAM_API_KEY"),
    #         live_options=LiveOptions(language=stt_language.value, encoding="linear16"),
    #     )
    elif provider == "sarvam":
        stt_language = (
            Language.KN_IN
            if language == "kannada"
            else Language.HI_IN if language == "hindi" else Language.EN_IN
        )
        stt = SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
            params=SarvamSTTService.InputParams(language=stt_language.value),
        )
    elif provider == "elevenlabs":
        stt = ElevenLabsRealtimeSTTService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            params=ElevenLabsRealtimeSTTService.InputParams(
                language_code=stt_language.value,
            ),
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
            model="whisper-large-v3",
            language=stt_language,
        )
    elif provider == "google":
        stt = GoogleSTTService(
            sample_rate=16000,
            location="us",
            params=GoogleSTTService.InputParams(
                languages=stt_language,
                model="chirp_3",
                # enable_interim_results=True,
                # enable_voice_activity_events=True,
            ),
            # location="us",
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
    # elif provider == "gemini":
    #     stt = GeminiLiveLLMService(

    #         credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    #     )
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
            output_logger,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            enable_metrics=True,
            allow_interruptions=True,
        ),
        idle_timeout_secs=60,
        observers=[RTVIObserver(rtvi), DebugLogObserver()],
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


async def run_stt_eval(
    audio_files: Sequence[Path],
    port: int,
    gt_transcript: Optional[List[Dict]] = None,
    output_path: Optional[Path] = None,
    existing_results: Optional[List[Dict]] = None,
) -> List[Dict[str, str]]:
    """Connects to the STT bot and streams audio files sequentially."""
    current_context.set("EVAL")

    transport = WebsocketClientTransport(
        uri=f"ws://localhost:{port}",
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

        def _is_user_transcript_message(
            self, frame: InputTransportMessageFrame
        ) -> str | bool:
            if hasattr(frame, "message"):
                message = frame.message
                if message.get("label") == "rtvi-ai":
                    msg_type = message.get("type")
                    if msg_type == "user-transcription":
                        data = message.get("data")
                        return data.get("text", "")

            return False

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"frame transcription logger: {frame}")
            # logger.info(f"frame transcription logger type: {type(frame)}")

            if isinstance(frame, InputTransportMessageFrame):
                if self._transcripts:
                    if user_transcript := self._is_final_user_transcript_message(frame):
                        self._transcripts[-1] += " " + user_transcript
                        log_and_print(
                            f"\033[93mappending to last user transcript: {user_transcript}\033[0m"
                        )
                        await self.push_frame(
                            TranscriptionFrame(
                                text=user_transcript,
                                user_id="",
                                timestamp=datetime.now().isoformat(),
                            ),
                            FrameDirection.UPSTREAM,
                        )
                    elif partial_transcript := self._is_user_transcript_message(frame):
                        await self.push_frame(
                            InterimTranscriptionFrame(
                                text=partial_transcript,
                                user_id="",
                                timestamp=datetime.now().isoformat(),
                            ),
                            FrameDirection.UPSTREAM,
                        )

            await self.push_frame(frame, direction)

    class BotTurnAudioStreamer(FrameProcessor):
        def __init__(
            self,
            audio_paths: List[Path],
            chunk_ms: int = 40,
            transcripts: list = [],
            gt_transcript: Optional[List[Dict]] = None,
            output_path: Optional[Path] = None,
            existing_results: Optional[List[Dict]] = None,
        ):
            super().__init__(enable_direct_mode=True, name="BotTurnAudioStreamer")
            self._audio_paths = audio_paths
            self._num_audios = len(audio_paths)
            self._chunk_ms = chunk_ms
            self._current_audio_index = 0

            # States: 'waiting_for_new_stream' -> 'streaming' -> 'stream_complete' -> 'done'
            self._state = "waiting_for_new_stream"
            self._output_ready = asyncio.Event()
            self._transcripts = transcripts
            self._pending_advance_task = None
            self._gt_transcript = gt_transcript
            self._output_path = output_path
            self._existing_results = existing_results or []

        def _save_intermediate_results(self):
            """Save current transcripts to CSV for crash recovery."""
            if not self._output_path or not self._gt_transcript:
                return

            # Merge existing results with new transcripts
            data = list(self._existing_results)

            for i in range(len(self._transcripts)):
                transcript = (
                    self._transcripts[i].strip() if self._transcripts[i] else ""
                )
                if transcript and i < len(self._gt_transcript):
                    data.append(
                        {
                            "id": self._gt_transcript[i]["id"],
                            "gt": self._gt_transcript[i]["gt"],
                            "pred": transcript,
                        }
                    )

            if data:
                pd.DataFrame(data).to_csv(self._output_path, index=False)
                log_and_print(f"Saved intermediate results: {len(data)} transcripts")

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

                        if msg_type == "user-stopped-speaking":
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
            # Save intermediate results after completing this audio
            self._save_intermediate_results()

            self._state = "streaming"
            log_and_print(f"--------------------------------")
            log_and_print(f"\033[93mCreated new user transcript\033[0m")
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

                    self._state = "stream_complete"

            logger.info(f"state: {self._state}")

            if self._state == "waiting_for_new_stream":
                # Start streaming only after transcript signals end of bot input
                self._start_new_audio_streaming()

            elif self._state == "stream_complete" and is_bot_turn_over:
                if self._current_audio_index + 1 < len(self._audio_paths):

                    logger.info(
                        "Setting state to buffering and advancing to next audio file in 5 seconds"
                    )

                    async def advance_and_stream():

                        await asyncio.sleep(5)
                        self._current_audio_index += 1

                        logger.info(
                            f"incrementing audio index: {self._current_audio_index}"
                        )

                        self._start_new_audio_streaming()

                    self._pending_advance_task = asyncio.create_task(
                        advance_and_stream()
                    )
                else:
                    logger.info(
                        "Setting state to buffering and ending the task in 5 seconds"
                    )

                    async def end_task_and_stream():
                        await asyncio.sleep(5)
                        logger.info(f"Completed streaming all audio files")
                        self._save_intermediate_results()
                        await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
                        await self.push_frame(EndFrame(), FrameDirection.UPSTREAM)
                        self._state = "done"

                    self._pending_advance_task = asyncio.create_task(
                        end_task_and_stream()
                    )

                self._state = "buffering"

            # Pass every frame through unchanged
            await self.push_frame(frame, direction)

        async def _stream_audio(self, audio_path: Path):
            try:
                await self._output_ready.wait()

                log_and_print(
                    f"Starting new audio streaming {self._current_audio_index + 1}/{self._num_audios}: {audio_path}"
                )

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

                # Pad a short burst of silence so downstream STT services (e.g. Google)
                # see a clean end-of-stream and flush their final transcripts before the
                # server times out the streaming RPC.
                silence_duration_ms = 1000
                silence_chunks = max(1, int(silence_duration_ms / self._chunk_ms))
                silence_sample_count = frames_per_chunk * channels
                silence_audio = b"\x00" * silence_sample_count * 2  # 16-bit audio

                for _ in range(silence_chunks):
                    frame = OutputAudioRawFrame(
                        audio=silence_audio,
                        sample_rate=sample_rate,
                        num_channels=channels,
                    )
                    await self.push_frame(frame, FrameDirection.DOWNSTREAM)
                    await asyncio.sleep(self._chunk_ms / 1000.0)
            finally:
                # After sending our audio, wait for bot's reply to finish
                self._state = "stream_complete"
                self._last_audio_ts = None
                log_and_print(f"Finished streaming audio: {audio_path}")

    transcripts = []

    streamer = BotTurnAudioStreamer(
        audio_paths=audio_files,
        chunk_ms=40,
        transcripts=transcripts,
        gt_transcript=gt_transcript,
        output_path=output_path,
        existing_results=existing_results,
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

            logger.info(f"eval frame: {frame}")

            await self.push_frame(frame, direction)

    input_logger = IOLogger()

    pipeline = Pipeline(
        [
            transport.input(),
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
            audio_out_sample_rate=16000,
        ),
        enable_tracing=True,
        idle_timeout_secs=100,
        cancel_on_idle_timeout=True,
        observers=[DebugLogObserver()],
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
        required=True,
        choices=[
            "deepgram",  # pcm16
            # "deepgram-flux",  # pcm16
            "openai",  # pcm16
            "cartesia",  # pcm16
            "smallest",  # pcm16
            "groq",  # wav
            "google",  # wav
            "gemini",  # pcm16
            "sarvam",  # wav
            "elevenlabs",  # wav
        ],
        help="STT provider to use for evaluation",
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
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing the audio files and stt.csv",
    )
    parser.add_argument(
        "-f",
        "--input-file-name",
        type=str,
        default="stt.csv",
        help="name of the input file containing the dataset to evaluate",
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
        type=int,
        default=5,
        help="Number of audio files to run the evaluation on",
    )
    parser.add_argument(
        "--ignore_retry",
        action="store_true",
        help="Ignore retrying if all the audios are not processed and move on to LLM judge",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Websocket port to connect to the STT bot",
    )

    args = parser.parse_args()

    if not exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_dir = join(args.output_dir, args.provider)

    log_save_path = join(output_dir, "logs")

    # if exists(log_save_path):
    #     os.remove(log_save_path)

    logger.remove()
    logger.add(
        log_save_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | [{extra[source]}] | {message}",
        filter=add_default_source,
        colorize=False,
    )

    print_log_save_path = join(output_dir, "results.log")

    if exists(print_log_save_path):
        os.remove(print_log_save_path)

    configure_print_logger(print_log_save_path)

    log_and_print(f"Running on port: {args.port}")
    log_and_print("--------------------------------")

    command = " ".join(sys.argv)
    log_and_print(f"\033[33mRunning command\033[0m: {command}")

    if args.provider in ["sarvam", "groq"]:
        audio_format = "wav"
    else:
        audio_format = "pcm16"

    audio_dir = join(args.input_dir, f"audios/{audio_format}")

    gt_file = join(args.input_dir, args.input_file_name)

    gt = pd.read_csv(gt_file)

    total_expected = len(gt)

    if args.debug:
        logger.debug(
            f"running in debug mode: using first {args.debug_count} audio files for evaluation"
        )
        gt = gt.head(args.debug_count)

    # Check for existing results and resume support
    results_csv_path = join(output_dir, "results.csv")
    existing_results = []
    previous_unprocessed_count = None

    # Loop until all results are collected
    while True:
        # Check for existing results and resume support
        if exists(results_csv_path):
            existing_df = pd.read_csv(results_csv_path)
            processed_ids = set(existing_df["id"].tolist())
            existing_results = existing_df[["id", "gt", "pred"]].to_dict("records")
            gt_for_pending = gt[~gt["id"].isin(processed_ids)].reset_index(drop=True)

            if gt_for_pending.empty:
                log_and_print("All audio files already processed, nothing to do")
                audio_files = []
                pred_transcripts = []
                break

            log_and_print(
                f"Resuming: {len(processed_ids)} processed, {len(gt_for_pending)} remaining"
            )
        else:
            gt_for_pending = gt

        current_unprocessed_count = len(gt_for_pending)

        # Check if no progress was made since last attempt
        if (
            previous_unprocessed_count is not None
            and current_unprocessed_count == previous_unprocessed_count
        ):
            log_and_print(
                f"No progress made - {current_unprocessed_count} files still unprocessed. "
                f"Saving empty transcripts for failed files and exiting."
            )
            # Add empty transcripts for failed files
            for _, row in gt_for_pending.iterrows():
                existing_results.append(
                    {
                        "id": row["id"],
                        "gt": row["text"],
                        "pred": "",
                    }
                )
            # Save results with empty transcripts
            pd.DataFrame(existing_results).to_csv(results_csv_path, index=False)
            audio_files = []
            pred_transcripts = []
            break

        previous_unprocessed_count = current_unprocessed_count

        audio_files = [
            Path(audio_dir) / f"{audio_name}.wav"
            for audio_name in gt_for_pending["id"].tolist()
        ]

        logger.info(f"Loading audio files: {audio_files}")

        if not audio_files:
            raise ValueError(f"No {audio_format} audio files found in {audio_dir}")

        logger.info(f"audio_files: {audio_files}")

        # Prepare gt_transcript for intermediate saving
        gt_transcript = [
            {"id": row["id"], "gt": row["text"]} for _, row in gt_for_pending.iterrows()
        ]

        bot_task = asyncio.create_task(
            run_stt_bot(provider=args.provider, language=args.language, port=args.port)
        )

        try:
            # Give the bot a moment to start listening before connecting.
            await asyncio.sleep(1.0)
            results = await run_stt_eval(
                audio_files,
                port=args.port,
                gt_transcript=gt_transcript,
                output_path=results_csv_path,
                existing_results=existing_results,
            )
            pred_transcripts = results["transcripts"]
            metrics = results["metrics"]
        finally:
            if not bot_task.done():
                bot_task.cancel()
                with suppress(asyncio.CancelledError):
                    await bot_task

        if args.ignore_retry:
            break

        # Check if all results are now collected
        if exists(results_csv_path):
            final_df = pd.read_csv(results_csv_path)
            if len(final_df) >= total_expected:
                log_and_print(
                    f"All {total_expected} audio files processed, exiting loop"
                )
                break
            else:
                log_and_print(
                    f"Only {len(final_df)}/{total_expected} processed, retrying..."
                )
        else:
            log_and_print("No results file found after run, retrying...")

    # Merge existing and new results for final metrics computation
    all_ids = [r["id"] for r in existing_results] + [
        splitext(basename(audio_file))[0] for audio_file in audio_files
    ]
    all_gt_transcripts = [r["gt"] for r in existing_results] + gt["text"].tolist()
    all_pred_transcripts = [r["pred"] for r in existing_results] + [
        t.strip() for t in pred_transcripts
    ]

    logger.info(f"gt_transcripts: {all_gt_transcripts}")
    logger.info(f"pred_transcripts: {all_pred_transcripts}")

    wer_score = get_wer_score(all_gt_transcripts, all_pred_transcripts)
    logger.info(f"WER: {wer_score['score']}")

    string_similarity = get_string_similarity(all_gt_transcripts, all_pred_transcripts)
    logger.info(f"String Similarity: {string_similarity['score']}")

    llm_judge_score = await get_llm_judge_score(
        all_gt_transcripts, all_pred_transcripts
    )
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
        all_gt_transcripts,
        all_pred_transcripts,
        all_ids,
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
        json.dump(metrics_data, f, indent=4)

    pd.DataFrame(data).to_csv(join(output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    asyncio.run(main())
