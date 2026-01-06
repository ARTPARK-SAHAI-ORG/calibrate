# Adapted from https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/07-interruptible.py

import os
import sys
import argparse
import json
from os.path import join, exists
from dotenv import load_dotenv
from loguru import logger
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional
from typing import Literal

from pense.utils import save_audio_chunk
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
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaLiveOptions
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.llm_service import FunctionCallParams

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openrouter.llm import OpenRouterLLMService

from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor

from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pense.integrations.smallest.stt import SmallestSTTService
from pense.integrations.smallest.tts import SmallestTTSService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService

from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import (
    ElevenLabsTTSService,
)
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

    # Helper for language mapping
    def get_language_enum(lang_str: str) -> Language:
        if lang_str == "hindi":
            return Language.HI
        elif lang_str == "kannada":
            return Language.KN
        return Language.EN

    # --- STT Setup ---
    stt_config = config.stt
    stt_language = get_language_enum(config.language)

    if stt_config.provider == "deepgram":
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(language=stt_language.value),
        )
    elif stt_config.provider == "google":
        stt = GoogleSTTService(
            params=GoogleSTTService.InputParams(languages=stt_language),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
    elif stt_config.provider == "openai":
        stt = OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=stt_config.model or "gpt-4o-transcribe",
            language=stt_language,
        )
    elif stt_config.provider == "elevenlabs":
        stt = ElevenLabsRealtimeSTTService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            params=ElevenLabsRealtimeSTTService.InputParams(
                language_code=stt_language.value,
            ),
        )
    elif stt_config.provider == "sarvam":
        sarvam_lang = (
            Language.KN_IN
            if config.language == "kannada"
            else Language.HI_IN if config.language == "hindi" else Language.EN_IN
        )
        stt = SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
            params=SarvamSTTService.InputParams(language=sarvam_lang.value),
        )
    elif stt_config.provider == "cartesia":
        stt = CartesiaSTTService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            live_options=CartesiaLiveOptions(language=stt_language.value),
        )
    elif stt_config.provider == "smallest":
        stt = SmallestSTTService(
            api_key=os.getenv("SMALLEST_API_KEY"),
            url="wss://waves-api.smallest.ai/api/v1/asr",
            params=SmallestSTTService.SmallestInputParams(
                audioLanguage=stt_language.value,
            ),
        )
    else:
        raise ValueError(f"Unknown STT provider: {stt_config.provider}")

    # --- TTS Setup ---
    tts_config = config.tts
    tts_language = get_language_enum(config.language)

    if tts_config.provider == "elevenlabs":
        default_voice = (
            "jUjRbhZWoMK4aDciW36V"
            if config.language == "hindi"
            else "90ipbRoKi4CpHXvKVtl0"
        )
        voice_id = tts_config.voice_id or default_voice
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=voice_id,
            params=ElevenLabsTTSService.InputParams(language=tts_language),
        )
    elif tts_config.provider == "cartesia":
        default_voice = (
            "28ca2041-5dda-42df-8123-f58ea9c3da00"
            if config.language == "hindi"
            else "66c6b81c-ddb7-4892-bdd5-19b5a7be38e7"
        )
        voice_id = tts_config.voice_id or default_voice
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=voice_id,
            model=tts_config.model or "sonic-3",
            params=CartesiaTTSService.InputParams(language=tts_language),
        )
    elif tts_config.provider == "google":
        default_voice = (
            "hi-IN-Chirp3-HD-Achernar"
            if config.language == "hindi"
            else "en-US-Chirp3-HD-Achernar"
        )
        tts = GoogleTTSService(
            voice_id=tts_config.voice_id or default_voice,
            params=GoogleTTSService.InputParams(language=tts_language),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
    elif tts_config.provider == "openai":
        tts = OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice=tts_config.voice_id or "fable",
            params=OpenAITTSService.InputParams(instructions=tts_config.instructions),
        )
    elif tts_config.provider == "smallest":
        default_voice = "aarushi"
        tts = SmallestTTSService(
            api_key=os.getenv("SMALLEST_API_KEY"),
            voice_id=tts_config.voice_id or default_voice,
            params=SmallestTTSService.InputParams(language=tts_language),
        )
    elif tts_config.provider == "deepgram":
        tts = DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            voice=tts_config.voice_id or "aura-2-andromeda-en",
        )
    elif tts_config.provider == "sarvam":
        sarvam_lang = (
            Language.KN_IN
            if config.language == "kannada"
            else Language.HI_IN if config.language == "hindi" else Language.EN_IN
        )
        tts = SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            model=tts_config.model or "bulbul:v2",
            voice_id=tts_config.voice_id or "abhilash",
            params=SarvamTTSService.InputParams(language=sarvam_lang),
        )
    else:
        raise ValueError(f"Unknown TTS provider: {tts_config.provider}")

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
