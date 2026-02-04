# Adapted from https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/07-interruptible.py

import os
import sys
import argparse
import json
from os.path import join, exists
from loguru import logger
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional
from typing import Literal

from calibrate.utils import (
    save_audio_chunk,
    create_stt_service,
    create_tts_service,
    build_tools_schema,
    make_webhook_call,
)
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
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openrouter.llm import OpenRouterLLMService

from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor

from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

from pipecat.observers.loggers.user_bot_latency_log_observer import (
    UserBotLatencyLogObserver,
)
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver


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
class STTConfig:
    provider: Literal[
        "deepgram", "google", "openai", "elevenlabs", "sarvam", "cartesia", "smallest"
    ] = "elevenlabs"
    model: Optional[str] = None


@dataclass
class TTSConfig:
    provider: Literal[
        "elevenlabs",
        "cartesia",
        "google",
        "openai",
        "smallest",
        "deepgram",
        "sarvam",
    ] = "elevenlabs"
    voice_id: Optional[str] = None
    model: Optional[str] = None
    instructions: Optional[str] = None


@dataclass
class LLMConfig:
    provider: Literal["openrouter", "openai"] = "openrouter"
    model: str = "openai/gpt-4o-2024-11-20"
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class BotConfig:
    system_prompt: str
    language: str
    tools: list[dict]
    stt: STTConfig
    tts: TTSConfig
    llm: LLMConfig


def parse_bot_config(config_data: Dict[str, Any]) -> BotConfig:
    if "system_prompt" not in config_data:
        raise ValueError("Config missing required key 'system_prompt'")

    system_prompt = config_data["system_prompt"]
    language = config_data.get("language", "english")
    tools = config_data.get("tools", [])

    stt_data = config_data.get("stt", {})
    stt_config = STTConfig(
        provider=stt_data.get("provider", "elevenlabs"),
        model=stt_data.get("model"),
    )

    tts_data = config_data.get("tts", {})
    tts_config = TTSConfig(
        provider=tts_data.get("provider", "elevenlabs"),
        voice_id=tts_data.get("voice_id"),
        model=tts_data.get("model"),
        instructions=tts_data.get("instructions"),
    )

    llm_data = config_data.get("llm", {})
    llm_config = LLMConfig(
        provider=llm_data.get("provider", "openrouter"),
        model=llm_data.get("model", "openai/gpt-4o-2024-11-20"),
        base_url=llm_data.get("base_url"),
        api_key=llm_data.get("api_key"),
    )

    return BotConfig(
        system_prompt=system_prompt,
        language=language,
        tools=tools,
        stt=stt_config,
        tts=tts_config,
        llm=llm_config,
    )


async def run_bot(
    config: BotConfig,
    transport: BaseTransport,
    runner_args: RunnerArguments,
    output_dir,
):
    logger.info(f"Starting bot")

    # --- STT Setup ---
    stt = create_stt_service(
        provider=config.stt.provider,
        language=config.language,
        model=config.stt.model,
    )

    # --- TTS Setup ---
    tts = create_tts_service(
        provider=config.tts.provider,
        language=config.language,
        voice_id=config.tts.voice_id,
        model=config.tts.model,
        instructions=config.tts.instructions,
    )

    # --- LLM Setup ---
    llm_config = config.llm
    if llm_config.provider == "openrouter":
        llm = OpenRouterLLMService(
            api_key=llm_config.api_key or os.getenv("OPENROUTER_API_KEY"),
            model=llm_config.model,
            base_url=llm_config.base_url or "https://openrouter.ai/api/v1",
        )
    elif llm_config.provider == "openai":
        llm = OpenAILLMService(
            api_key=llm_config.api_key or os.getenv("OPENAI_API_KEY"),
            model=llm_config.model,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {llm_config.provider}")

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

    # Build tool schemas using common utility
    tool_schemas, webhook_configs = build_tools_schema(config.tools)
    standard_tools = [end_call_tool] + tool_schemas

    def create_webhook_function_call(webhook_config: dict):
        async def webhook_function_call(params: FunctionCallParams):
            print(
                f"{params.function_name} (webhook) invoked with arguments: {params.arguments}"
            )

            result = await make_webhook_call(webhook_config, params.arguments or {})
            await params.result_callback(result)
            return

        return webhook_function_call

    # Register function handlers
    llm.register_function("end_call", end_call)
    for tool_schema in tool_schemas:
        if tool_schema.name in webhook_configs:
            llm.register_function(
                tool_schema.name,
                create_webhook_function_call(webhook_configs[tool_schema.name]),
            )
        else:
            llm.register_function(tool_schema.name, generic_function_call)

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

    print("Conversation complete. Saving conversation transcript...")

    transcript = [
        message for message in context.get_messages() if message.get("role") != "system"
    ]

    with open(join(output_dir, "transcript.json"), "w") as transcript_file:
        json.dump(transcript, transcript_file, indent=4)

    print(f"Conversation transcript saved to {join(output_dir, 'transcript.json')}")


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
