import asyncio
import os
import wave
from contextlib import suppress
from pathlib import Path
from typing import Dict, List, Sequence

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    EndTaskFrame,
    EndFrame,
    OutputTransportReadyFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService, Language, LiveOptions
from pipecat.transports.websocket.client import (
    WebsocketClientParams,
    WebsocketClientTransport,
)
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3


load_dotenv("./server/.env", override=True)


async def run_stt_bot(language: Language = Language.EN):
    """Starts an STT-only bot that reports RTVI transcription messages."""

    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=False,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    )

    class IOLogger(FrameProcessor):
        def __init__(self, agent: str, position: str):
            super().__init__()
            self._agent = agent
            self._position = position

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"frame: {frame}")

            await self.push_frame(frame, direction)

    input_logger = IOLogger(agent="eval", position="after_input")
    output_logger = IOLogger(agent="eval", position="before_output")

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(language=language.value),
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            input_logger,
            stt,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
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
        ):
            super().__init__(enable_direct_mode=True, name="TranscriptionWriter")
            self._transcripts = transcripts

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
            logger.info(f"frame transcription logger type: {type(frame)}")

            if isinstance(frame, InputTransportMessageFrame):
                if self._transcripts:
                    if user_transcript := self._is_final_user_transcript_message(frame):
                        self._transcripts[-1]["content"] += user_transcript

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

        def _start_new_audio_streaming(
            self,
        ):
            self._state = "streaming"
            self._transcripts.append({"role": "user", "content": ""})
            logger.info(f"transcripts length: {len(self._transcripts)}")
            asyncio.create_task(
                self._stream_audio(self._audio_paths[self._current_audio_index])
            )

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"frame: {frame}")
            is_bot_turn_over = self._is_transcription_over(frame)
            logger.info(f"is_bot_turn_over: {is_bot_turn_over}")

            if isinstance(frame, OutputTransportReadyFrame):
                self._output_ready.set()

            # Pass every frame through unchanged
            await self.push_frame(frame, direction)

            logger.info(f"state: {self._state}")

            if self._state == "await_bot_turn_end":
                # Start streaming only after transcript signals end of bot input
                self._start_new_audio_streaming()

            elif self._state == "await_reply_bot_turn_end":
                if is_bot_turn_over:
                    self._current_audio_index += 1

                    if self._current_audio_index < len(self._audio_paths):
                        self._start_new_audio_streaming()
                    else:
                        logger.info(f"completed simulation message received")
                        await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
                        await self.push_frame(EndFrame(), FrameDirection.UPSTREAM)
                        self._state = "done"

        async def _stream_audio(self, audio_path: Path):
            try:
                await self._output_ready.wait()
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

    transcripts = []

    transcription_logger = TranscriptionWriter(
        transcripts,
    )

    streamer = BotTurnAudioStreamer(
        audio_paths=audio_files,
        chunk_ms=40,
        transcripts=transcripts,
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Bot audio coming in
            # input_logger,
            transcription_logger,
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
        idle_timeout_secs=10,
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

    return transcripts


async def main():
    audio_files = [
        Path(__file__).with_name("input.wav"),
        Path(__file__).with_name("input2.wav"),
        Path(__file__).with_name("input3.wav"),
    ]

    bot_task = asyncio.create_task(run_stt_bot())

    # await run_stt_bot()

    try:
        # Give the bot a moment to start listening before connecting.
        await asyncio.sleep(1.0)
        results = await run_stt_eval(audio_files)
    finally:
        if not bot_task.done():
            bot_task.cancel()
            with suppress(asyncio.CancelledError):
                await bot_task

    if results:
        logger.info("Completed transcriptions:")
        logger.info(f"{results}")
        # for entry in results:
        #     logger.info(f"{entry['file']}: {entry['transcription']}")
    else:
        logger.warning("No transcriptions were produced")


if __name__ == "__main__":
    asyncio.run(main())
