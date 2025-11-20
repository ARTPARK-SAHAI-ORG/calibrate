# Adapted from https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/07-interruptible.py

import os
import sys
import argparse
import json
import io
from pathlib import Path
import aiofiles
import shutil
import struct
from os.path import join, exists
from dotenv import load_dotenv
from loguru import logger
from datetime import datetime
import wave
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, FunctionCallResultProperties
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.transcriptions.language import Language
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor

from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.openai.stt import OpenAISTTService
from integrations.smallest.stt import SmallestSTTService
from integrations.smallest.tts import SmallestTTSService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor


from pipecat.observers.loggers.user_bot_latency_log_observer import (
    UserBotLatencyLogObserver,
)
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver


load_dotenv(override=True)

CUSTOM_CLI_ARGS: dict[str, Any] = {}


def _store_cli_args(args: argparse.Namespace) -> None:
    """Persist custom CLI args so they are accessible inside bot()."""
    CUSTOM_CLI_ARGS.clear()
    CUSTOM_CLI_ARGS.update(
        {key: value for key, value in vars(args).items() if value is not None}
    )


def get_cli_arg(key: str, default: Optional[Any] = None) -> Optional[Any]:
    """Return a previously parsed CLI argument."""
    return CUSTOM_CLI_ARGS.get(key, default)


transport_params = {
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(),
    ),
}


@dataclass
class BotConfig:
    system_prompt: str
    language: str
    tools: list[dict]


def parse_bot_config(config_data: Dict[str, Any]) -> BotConfig:
    if "system_prompt" not in config_data:
        raise ValueError("Config missing required key 'system_prompt'")

    system_prompt = config_data["system_prompt"]
    language = config_data.get("language", "english")
    tools = config_data.get("tools", [])

    return BotConfig(
        system_prompt=system_prompt,
        language=language,
        tools=tools,
    )


async def save_audio_chunk(
    path: str, audio_chunk: bytes, sample_rate: int, num_channels: int
):
    if len(audio_chunk) == 0:
        logger.info(f"There's no audio to save for {path}")
        return

    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists():
        logger.info(f"Creating new audio file for {path} at {filepath}")
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_chunk)
            async with aiofiles.open(filepath, "wb") as file:
                await file.write(buffer.getvalue())
    else:
        logger.info(f"Appending audio chunk for {path} to {filepath}")
        async with aiofiles.open(filepath, "rb+") as file:
            current_size = await file.seek(0, os.SEEK_END)
            if current_size < 44:
                logger.info(
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


async def run_bot(
    config: BotConfig,
    transport: BaseTransport,
    runner_args: RunnerArguments,
    output_dir,
):
    logger.info(f"Starting bot")

    language = Language.HI if config.language == "hindi" else Language.EN

    # stt = DeepgramSTTService(
    #     api_key=os.getenv("DEEPGRAM_API_KEY"),
    #     live_options=LiveOptions(language=Language.HI.value),
    # )
    # stt = GoogleSTTService(
    #     params=GoogleSTTService.InputParams(
    #         languages=language,
    #         # model="chirp_3",
    #     ),
    #     credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    # )

    # stt = SmallestSTTService(
    #     api_key=os.getenv("SMALLEST_API_KEY"),
    #     url="wss://waves-api.smallest.ai/api/v1/asr",
    #     params=SmallestSTTService.SmallestInputParams(
    #         audioLanguage=language.value,
    #     ),
    # )
    # stt = SarvamSTTService(
    #     api_key=os.getenv("SARVAM_API_KEY"),
    #     params=SarvamSTTService.InputParams(
    #         language=language.value,
    #     ),
    # )

    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-transcribe",
        language=Language.HI,
    )

    tts = GoogleTTSService(
        voice_id="hi-IN-Chirp3-HD-Achernar",
        params=GoogleTTSService.InputParams(language=Language.HI_IN),
        credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    )

    # tts = SmallestTTSService(
    #     api_key=os.getenv("SMALLEST_API_KEY"),
    #     params=SmallestTTSService.InputParams(language=language),
    #     voice_id="aarushi",
    # )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1")

    messages = [
        {
            "role": "system",
            "content": config.system_prompt
            + f"\n\nYou must always speak in {config.language}.",
        },
    ]

    async def end_call(params: FunctionCallParams):
        print(f"end_call tool invoked by LLM: {params}")

        await params.result_callback(
            None, properties=FunctionCallResultProperties(run_llm=False)
        )
        try:
            await task.cancel()
        except Exception as exc:
            logger.warning(
                f"Unable to cancel task after end_call (no tool_call_id): {exc}"
            )

    async def generic_function_call(params: FunctionCallParams):
        print(f"{params.function_name} invoked with arguments: {params.arguments}")

        await params.result_callback(
            {"status": "received"},
        )

    end_call_tool = FunctionSchema(
        name="end_call",
        description="End the current call when the conversation is complete.",
        properties={
            "reason": {
                "type": "string",
                "description": "Optional explanation for why the call should end.",
            }
        },
        required=[],
    )
    standard_tools = [end_call_tool]
    llm.register_function("end_call", end_call)

    for tool in config.tools:
        properties = {}
        for parameter in tool["parameters"]:
            prop = {
                "type": parameter["type"],
                "description": parameter["description"],
            }

            if "items" in parameter:
                prop["items"] = parameter["items"]

            if "enum" in parameter:
                prop["enum"] = parameter["enum"]

            properties[parameter["id"]] = prop

        tool_function = FunctionSchema(
            name=tool["name"],
            description=tool["description"],
            properties=properties,
            required=[],
        )
        standard_tools.append(tool_function)
        llm.register_function(tool["name"], generic_function_call)

    tools = ToolsSchema(standard_tools=standard_tools)
    context = LLMContext(messages, tools)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    transcript = TranscriptProcessor()

    audio_buffer = AudioBufferProcessor(
        enable_turn_audio=True,  # Enable per-turn audio recording
    )

    turn_index = 0

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            transcript.user(),
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            audio_buffer,
            transcript.assistant(),
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi), UserBotLatencyLogObserver(), LLMLogObserver()],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        await audio_buffer.start_recording()
        # Kick off the conversation.
        messages.append(
            {"role": "system", "content": "Please introduce yourself to the user."}
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    @transcript.event_handler("on_transcript_update")
    async def handle_transcript_update(processor, frame):
        # Each message contains role (user/assistant), content, and timestamp
        for message in frame.messages:
            print(f"[{message.timestamp}] {message.role}: {message.content}")

    audio_dir = join(output_dir, "audios")

    if exists(audio_dir):
        shutil.rmtree(audio_dir)

    @audio_buffer.event_handler("on_user_turn_audio_data")
    async def on_user_turn_audio_data(buffer, audio, sample_rate, num_channels):
        nonlocal turn_index

        # Save or process the composite audio
        filename = f"{audio_dir}/{turn_index}_user.wav"

        turn_index += 1

        # Create the WAV file
        await save_audio_chunk(filename, audio, sample_rate, num_channels)

        logger.info(f"Saved recording to {filename}")

    @audio_buffer.event_handler("on_bot_turn_audio_data")
    async def on_bot_turn_audio_data(buffer, audio, sample_rate, num_channels):
        nonlocal turn_index

        # Save or process the composite audio
        filename = f"{audio_dir}/{turn_index}_bot.wav"

        turn_index += 1

        # Create the WAV file
        await save_audio_chunk(filename, audio, sample_rate, num_channels)

        logger.info(f"Saved recording to {filename}")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""

    logger.remove()

    config_path = get_cli_arg("config")
    if not config_path:
        raise RuntimeError(
            "Missing --config argument. Pass it before Pipecat runner options."
        )

    with open(config_path, "r") as cfg_file:
        config_data = json.load(cfg_file)

    output_dir = get_cli_arg("output_dir")
    logs_path = join(output_dir, "logs")

    if exists(logs_path):
        os.remove(logs_path)

    logger.add(logs_path, level="DEBUG")

    transport = await create_transport(runner_args, transport_params)

    bot_config = parse_bot_config(config_data)
    await run_bot(bot_config, transport, runner_args, output_dir)


if __name__ == "__main__":
    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the agent config JSON file.",
    )
    custom_parser.add_argument(
        "-o",
        "--output_dir",
        default="./out",
        help="Path to the output directory to save the logs and recordings.",
    )
    custom_args, runner_argv = custom_parser.parse_known_args()
    _store_cli_args(custom_args)

    # Reconstruct sys.argv so Pipecat's runner parser only sees its own arguments.
    sys.argv = [sys.argv[0], *runner_argv]

    from pipecat.runner.run import main

    main()
