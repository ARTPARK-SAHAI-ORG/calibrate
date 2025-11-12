from contextlib import suppress
import csv
import json
import os
import wave
from os.path import join, exists
from pipecat.frames.frames import (
    InputTransportMessageFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from typing import Literal, Optional
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
import asyncio
import websockets
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from integrations.smallest import SmallestTTSService
from pipecat.transcriptions.language import Language
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from loguru import logger
from pipecat.processors.transcript_processor import TranscriptProcessor
from dotenv import load_dotenv

load_dotenv(".env", override=True)


def chunk_text(text: str, max_chunk_size: int = 5) -> list[str]:
    """
    Chunk text with a maximum length, preferring to break on punctuation.

    Args:
        text: Input text to chunk.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        List of text chunks.
    """
    chunks: list[str] = []
    remaining = text.strip()

    punctuation_marks = ".,:;।!?"

    while remaining:
        if len(remaining) <= max_chunk_size:
            chunks.append(remaining)
            break

        chunk_end = max_chunk_size
        found_punct = False
        search_start = max(chunk_end - 50, 0)

        for i in range(chunk_end, search_start, -1):
            if i < len(remaining) and remaining[i] in punctuation_marks:
                chunk_end = i + 1
                found_punct = True
                break

        if not found_punct:
            for i in range(chunk_end, search_start, -1):
                if i < len(remaining) and remaining[i].isspace():
                    chunk_end = i
                    break

        if chunk_end == 0:
            chunk_end = max_chunk_size

        chunks.append(remaining[:chunk_end].strip())
        remaining = remaining[chunk_end:].strip()

    return [chunk for chunk in chunks if chunk]


async def run_tts_bot(provider: str, text: str, language: Literal["english", "hindi"]):
    """Starts a TTS-only bot that reports RTVI transcription messages."""

    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=False,
            audio_out_enabled=True,
            add_wav_header=False,
        ),
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    class IOLogger(FrameProcessor):
        def __init__(self):
            super().__init__()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"bot frame: {frame}")
            logger.info(f"bot frame type: {type(frame)}")

            await self.push_frame(frame, direction)

    input_logger = IOLogger()

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
    else:
        raise ValueError(f"Invalid provider: {provider}")

    transcript = TranscriptProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            input_logger,
            tts,
            transport.output(),
            transcript.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=16000,
            enable_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
        idle_timeout_secs=200,
    )

    text_enqueued = False

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal text_enqueued
        logger.info("Client connected to TTS bot")
        if text_enqueued:
            return

        text_to_enqueue = text.strip()
        if not text_to_enqueue:
            logger.warning("No text provided to TTS bot; skipping enqueue")
            text_enqueued = True
            return

        text_chunks = chunk_text(text_to_enqueue, max_chunk_size=150)

        if not text_chunks:
            logger.warning("No valid chunks produced from text; skipping enqueue")
            text_enqueued = True
            return

        frames = [LLMFullResponseStartFrame()]
        frames.extend(TextFrame(chunk) for chunk in text_chunks)
        frames.append(LLMFullResponseEndFrame())

        await task.queue_frames(frames)
        text_enqueued = True

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected from TTS bot")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    logger.info("TTS bot ready and waiting for connections")
    await runner.run(task)


def _write_wav(audio: bytes, sample_rate: int, num_channels: int, output_path: str):
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)


async def _read_audio_stream(
    websocket,
    receive_timeout: float,
) -> tuple[bytes, int, int, list[float]]:
    serializer = ProtobufFrameSerializer()
    audio_chunks: list[bytes] = []
    sample_rate: Optional[int] = None
    num_channels: Optional[int] = None
    received_audio = False
    first_frame_timeout = max(receive_timeout, 30.0)
    ttfb_values: list[float] = []

    while True:
        timeout = receive_timeout if received_audio else first_frame_timeout
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for audio frames from TTS bot")
            break
        except Exception as exc:
            logger.info(f"TTS bot websocket closed: {exc}")
            break

        frame = await serializer.deserialize(message)
        if not frame:
            continue

        if isinstance(frame, InputAudioRawFrame):
            if not received_audio:
                logger.info("Receiving audio from TTS bot")
            audio_chunks.append(frame.audio)
            sample_rate = frame.sample_rate or sample_rate
            num_channels = frame.num_channels or num_channels
            received_audio = True
        elif isinstance(frame, InputTransportMessageFrame):
            message = getattr(frame, "message", {})
            if isinstance(message, dict) and message.get("label") == "rtvi-ai":
                if message.get("type") == "metrics":
                    data = message.get("data") or {}
                    for entry in data.get("ttfb", []):
                        value = entry.get("value")
                        if value is not None:
                            try:
                                ttfb_values.append(float(value))
                            except (TypeError, ValueError):
                                continue
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug("Received BotStoppedSpeakingFrame from TTS bot")
            if received_audio:
                break
        elif isinstance(frame, EndFrame):
            logger.debug("Received EndFrame from TTS bot")
            break

    return (
        b"".join(audio_chunks),
        sample_rate or 16000,
        num_channels or 1,
        ttfb_values,
    )


async def _collect_audio_over_websocket(
    *,
    host: str,
    port: int,
    connect_timeout: float,
    receive_timeout: float,
    retry_attempts: int,
    retry_delay: float = 0.2,
) -> tuple[bytes, int, int, list[float]]:
    uri = f"ws://{host}:{port}"
    last_error: Optional[Exception] = None

    for attempt in range(retry_attempts):
        try:
            async with websockets.connect(
                uri, open_timeout=connect_timeout
            ) as websocket:
                logger.info(f"Connected to TTS bot at {uri}")
                return await _read_audio_stream(websocket, receive_timeout)
        except Exception as exc:
            last_error = exc
            logger.debug(
                f"Attempt {attempt + 1}/{retry_attempts} to connect failed: {exc}"
            )
            await asyncio.sleep(retry_delay)

    raise RuntimeError(f"Unable to connect to TTS bot at {uri}") from last_error


async def run_eval(
    provider: str,
    text: str,
    language: Literal["english", "hindi"],
    output_path: str,
    *,
    host: str = "localhost",
    port: int = 8765,
    connect_timeout: float = 5.0,
    receive_timeout: float = 5.0,
    retry_attempts: int = 25,
) -> dict[str, Optional[float] | str]:
    server_task = asyncio.create_task(run_tts_bot(provider, text, language))
    audio: bytes = b""
    sample_rate = 16000
    num_channels = 1
    ttfb_values: list[float] = []

    try:
        audio, sample_rate, num_channels, ttfb_values = (
            await _collect_audio_over_websocket(
                host=host,
                port=port,
                connect_timeout=connect_timeout,
                receive_timeout=receive_timeout,
                retry_attempts=retry_attempts,
            )
        )
    finally:
        try:
            await asyncio.wait_for(server_task, timeout=5.0)
        except asyncio.TimeoutError:
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task
        except asyncio.CancelledError:
            pass

    if not audio:
        raise RuntimeError("TTS bot did not produce any audio frames")

    _write_wav(audio, sample_rate, num_channels, output_path)
    logger.info(f"Saved synthesized audio to {output_path}")

    ttfb_value: Optional[float] = None
    for value in ttfb_values:
        if value is not None:
            ttfb_value = value

    return {"audio_path": output_path, "ttfb": ttfb_value}


async def run_batch(
    provider: str,
    language: Literal["english", "hindi"],
    input_csv: str,
    output_dir: str,
    *,
    host: str,
    port: int,
    connect_timeout: float,
    receive_timeout: float,
    retry_attempts: int,
):
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audios")
    os.makedirs(audio_dir, exist_ok=True)

    log_save_path = join(output_dir, "logs")

    if exists(log_save_path):
        os.remove(log_save_path)

    logger.add(log_save_path)

    with open(input_csv, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'text' column")
        rows = list(reader)

    results: list[dict[str, str | float | None]] = []
    ttfb_values: list[float] = []

    for index, row in enumerate(rows, start=1):
        raw_text = row.get("text", "")
        text = raw_text.strip() if raw_text else ""
        if not text:
            logger.warning(f"Skipping empty text at row {index}")
            continue

        audio_filename = f"audio_{index:04d}.wav"
        audio_path = os.path.join(audio_dir, audio_filename)

        eval_result = await run_eval(
            provider,
            text,
            language,
            audio_path,
            host=host,
            port=port,
            connect_timeout=connect_timeout,
            receive_timeout=receive_timeout,
            retry_attempts=retry_attempts,
        )

        rel_audio_path = os.path.relpath(eval_result["audio_path"], output_dir)
        ttfb = eval_result["ttfb"]
        if isinstance(ttfb, (int, float)):
            ttfb_values.append(float(ttfb))

        results.append(
            {
                "text": text,
                "audio_path": rel_audio_path,
                "ttfb": ttfb if ttfb is not None else "",
            }
        )

    results_path = os.path.join(output_dir, "results.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["text", "audio_path", "ttfb"])
        writer.writeheader()
        writer.writerows(results)

    metrics = {
        "ttfb_mean": sum(ttfb_values) / len(ttfb_values) if ttfb_values else None,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    logger.info(f"Wrote results to {results_path}")
    logger.info(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the TTS bot locally, synthesize text, and save the resulting audio."
    )
    parser.add_argument(
        "--input-csv",
        help="Path to a CSV file containing a 'text' column to synthesize in batch.",
    )
    parser.add_argument(
        "--output-dir",
        default="tts_outputs",
        help="Directory where batch outputs (results.csv, metrics.json, audios/) will be stored.",
    )
    parser.add_argument(
        "--text",
        help="Single text to synthesize (used when --input-csv is not provided).",
    )
    parser.add_argument(
        "--provider",
        default="smallest",
        help="TTS provider to use.",
    )
    parser.add_argument(
        "--language",
        choices=["english", "hindi"],
        default="english",
        help="Language to synthesize.",
    )
    parser.add_argument(
        "--output",
        default="tts_output.wav",
        help="Path where the synthesized audio should be saved for single-text mode.",
    )
    parser.add_argument(
        "--host", default="localhost", help="Websocket host for the TTS bot."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Websocket port for the TTS bot.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait when establishing the websocket connection.",
    )
    parser.add_argument(
        "--receive-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for audio frames before timing out.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=25,
        help="How many times to retry connecting to the websocket server.",
    )

    args = parser.parse_args()

    if args.input_csv:
        asyncio.run(
            run_batch(
                provider=args.provider,
                language=args.language,
                input_csv=args.input_csv,
                output_dir=args.output_dir,
                host=args.host,
                port=args.port,
                connect_timeout=args.connect_timeout,
                receive_timeout=args.receive_timeout,
                retry_attempts=args.retry_attempts,
            )
        )
    else:
        if not args.text:
            parser.error("Either --input-csv or --text must be provided.")

        result = asyncio.run(
            run_eval(
                args.provider,
                args.text,
                args.language,
                args.output,
                host=args.host,
                port=args.port,
                connect_timeout=args.connect_timeout,
                receive_timeout=args.receive_timeout,
                retry_attempts=args.retry_attempts,
            )
        )
        logger.info(
            f"Synthesized single text. Audio: {result['audio_path']}, TTFB: {result['ttfb']}"
        )
