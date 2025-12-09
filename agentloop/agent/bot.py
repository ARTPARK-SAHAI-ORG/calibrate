import os
from typing import Dict, Literal

from dotenv import load_dotenv
from loguru import logger

from pydantic import BaseModel
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMRunFrame,
    MetricsFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    InputTransportMessageFrame,
    FunctionCallResultFrame,
    UserStartedSpeakingFrame,
    FunctionCallResultProperties,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaLiveOptions

from pipecat.services.deepgram.stt import DeepgramSTTService, Language, LiveOptions
from pipecat.services.deepgram.tts import DeepgramTTSService

from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService

from agentloop.integrations.smallest.stt import SmallestSTTService
from agentloop.integrations.smallest.tts import SmallestTTSService

from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.groq.tts import GroqTTSService

from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService

from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService

from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService, Language
from pipecat.services.elevenlabs.tts import (
    ElevenLabsTTSService,
)

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

# from pipecat.transports.daily.transport import DailyParams
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.services.llm_service import FunctionCallParams
from pipecat.observers.loggers.user_bot_latency_log_observer import (
    UserBotLatencyLogObserver,
)

bot_logger = logger.bind(source="BOT")

load_dotenv(override=True)


class MetricsLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, TTFBMetricsData):
                    bot_logger.info(f"!!! MetricsFrame: {frame}, ttfb: {d.value}")
                elif isinstance(d, ProcessingMetricsData):
                    bot_logger.info(f"!!! MetricsFrame: {frame}, processing: {d.value}")
                elif isinstance(d, LLMUsageMetricsData):
                    tokens = d.value
                    bot_logger.info(
                        f"!!! MetricsFrame: {frame}, tokens: {tokens.prompt_tokens}, characters: {tokens.completion_tokens}"
                    )
                elif isinstance(d, TTSUsageMetricsData):
                    bot_logger.info(f"!!! MetricsFrame: {frame}, characters: {d.value}")
        await self.push_frame(frame, direction)


class STTConfig(BaseModel):
    provider: Literal[
        "deepgram", "google", "openai", "cartesia", "smallest", "elevenlabs", "sarvam"
    ] = "deepgram"


class TTSConfig(BaseModel):
    provider: Literal[
        "cartesia", "google", "openai", "smallest", "elevenlabs", "sarvam"
    ] = "google"
    instructions: str = None


class LLMConfig(BaseModel):
    provider: Literal["openai", "groq", "google"] = "openai"
    model: str = "gpt-4.1"


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    system_prompt: str = "You are a helpful assistant.",
    tools: list[dict] = [],
    stt_config: STTConfig = STTConfig(),
    tts_config: TTSConfig = TTSConfig(),
    llm_config: LLMConfig = LLMConfig(),
    language: Literal["english", "hindi"] = "english",
    mode: Literal["run", "eval"] = "run",
):
    if language not in ["english", "hindi"]:
        raise ValueError(f"Invalid language: {language}")

    bot_logger.info(f"Starting bot")

    stt_language = (
        Language.KN
        if language == "kannada"
        else Language.HI if language == "hindi" else Language.EN
    )

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
            model="gpt-4o-mini-transcribe",
            # prompt=prompt,
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
        stt_language = (
            Language.KN_IN
            if language == "kannada"
            else Language.HI_IN if language == "hindi" else Language.EN_IN
        )
        stt = SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
            params=SarvamSTTService.InputParams(language=stt_language.value),
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
                audioLanguage=stt_language,
            ),
        )

    tts_language = (
        Language.EN
        if language == "english"
        else Language.HI if language == "hindi" else Language.KN
    )

    language_to_voice_id = {
        "english": {
            "cartesia": "66c6b81c-ddb7-4892-bdd5-19b5a7be38e7",
            "smallest": "aarushi",
            "google": "en-US-Chirp3-HD-Achernar",
        },
        "hindi": {
            "cartesia": "28ca2041-5dda-42df-8123-f58ea9c3da00",
            "smallest": "aarushi",
            "google": "hi-IN-Chirp3-HD-Achernar",
        },
    }

    if tts_config.provider == "cartesia":
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            model="sonic-3",
            params=CartesiaTTSService.InputParams(
                language=tts_language,
            ),
            voice_id=language_to_voice_id[language][tts_config.provider],
        )
    elif tts_config.provider == "google":
        tts = GoogleTTSService(
            voice_id=language_to_voice_id[language][tts_config.provider],
            params=GoogleTTSService.InputParams(language=tts_language),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )

    elif tts_config.provider == "deepgram":
        tts = DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-2-andromeda-en"
        )

    elif tts_config.provider == "elevenlabs":
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            params=ElevenLabsTTSService.InputParams(
                language=tts_language,
            ),
            voice_id="Ui0HFqLn4HkcAenlJJVJ",
        )
    elif tts_config.provider == "sarvam":
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
    elif tts_config.provider == "openai":

        # You are an indian nurse in a public health clinic. Speak in a natural, conversational tone. Have an indian accent and you should sound indian. {extra_instructions}
        tts = OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice="fable",
            instructions=tts_config.instructions,
        )

    elif tts_config.provider == "smallest":
        tts = SmallestTTSService(
            api_key=os.getenv("SMALLEST_API_KEY"),
            voice_id=language_to_voice_id[language][tts_config.provider],
            params=SmallestTTSService.InputParams(
                language=tts_language,
            ),
        )

    # tts = ElevenLabsHttpTTSService(
    #         api_key=os.getenv("ELEVENLABS_API_KEY"),
    #         voice_id="Ui0HFqLn4HkcAenlJJVJ",
    #         aiohttp_session=session,
    #     )

    if llm_config.provider == "openai":
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"), model=llm_config.model
        )
    elif llm_config.provider == "google":
        llm = GoogleLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-2.5-flash",
            # turn on thinking if you want it
            # params=GoogleLLMService.InputParams(extra={"thinking_config": {"thinking_budget": 4096}}),)
        )
    elif llm_config.provider == "groq":
        llm = GroqLLMService(
            api_key=os.getenv("GROQ_API_KEY"),
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
        )

    ml = MetricsLogger()

    transcript = TranscriptProcessor()

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    async def _exec_call_call():
        try:
            await task.cancel()
        except Exception as exc:
            bot_logger.warning(
                f"Unable to cancel task after end_call (no tool_call_id): {exc}"
            )

    async def end_call(params: FunctionCallParams):
        reason = params.arguments.get("reason") if params.arguments else None
        if reason:
            bot_logger.info(f"end_call tool invoked by LLM: {reason}")
        else:
            bot_logger.info("end_call tool invoked by LLM")

        if mode == "run":
            await params.result_callback(
                None, properties=FunctionCallResultProperties(run_llm=False)
            )
            await _exec_call_call()
            return

        tool_call_id = params.tool_call_id
        pending_tool_calls[tool_call_id] = params

        try:
            await rtvi.handle_function_call(params)
        except Exception as exc:
            pending_tool_calls.pop(tool_call_id, None)
            bot_logger.warning(f"Unable to forward end_call to client: {exc}")

    async def generic_function_call(params: FunctionCallParams):
        bot_logger.info(
            f"{params.function_name} invoked with arguments: {params.arguments}"
        )

        if mode == "run":
            await params.result_callback(
                {"status": "received"},
            )
            return

        tool_call_id = params.tool_call_id
        pending_tool_calls[tool_call_id] = params

        try:
            await rtvi.handle_function_call(params)
        except Exception as exc:
            pending_tool_calls.pop(tool_call_id, None)
            bot_logger.warning(
                f"Unable to forward {params.function_name} to client: {exc}"
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

    for tool in tools:
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

    pending_tool_calls: Dict[str, FunctionCallParams] = {}

    class IOLogger(FrameProcessor):
        def __init__(
            self,
        ):
            super().__init__()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            logger.info(f"bot frame: {frame}")

            if isinstance(frame, InputTransportMessageFrame):
                logger.info(f"InputTransportMessageFrame yesss: {frame}")

            if isinstance(frame, OutputTransportMessageUrgentFrame):
                logger.info(f"OutputTransportMessageUrgentFrame yesss: {frame}")

            if (
                isinstance(frame, InputTransportMessageFrame)
                and hasattr(frame, "message")
                and frame.message.get("type") == "client-message"
                and frame.message.get("data", {}).get("t") == "interrupt"
            ):
                logger.info(f"Simulating user interruption of the bot")
                self.push_frame(InterruptionFrame(), FrameDirection.UPSTREAM)

            await self.push_frame(frame, direction)

    class FunctionCallResultHandler(FrameProcessor):
        def __init__(self):
            super().__init__(enable_direct_mode=True, name="FunctionCallResultHandler")

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, FunctionCallResultFrame):
                params = pending_tool_calls.pop(frame.tool_call_id, None)

                if not params:
                    bot_logger.warning(
                        f"Received function call result for unknown tool_call_id {frame.tool_call_id} ({frame.function_name})"
                    )
                else:
                    try:
                        properties = (
                            FunctionCallResultProperties(run_llm=False)
                            if params.function_name == "end_call"
                            else None
                        )

                        await params.result_callback(
                            frame.result, properties=properties
                        )
                        bot_logger.debug(
                            f"Delivered function call result for {frame.function_name}:{frame.tool_call_id}"
                        )
                    except Exception as exc:
                        bot_logger.warning(
                            f"Failed to deliver function call result for {frame.function_name}:{frame.tool_call_id}: {exc}"
                        )

                    if frame.function_name == "end_call":
                        await _exec_call_call()

            await self.push_frame(frame, direction)

    pipeline_processors = [
        transport.input(),
        rtvi,
        IOLogger(),
    ]

    if mode == "eval":
        pipeline_processors.append(FunctionCallResultHandler())

    pipeline_processors.extend(
        [
            stt,
            transcript.user(),
            context_aggregator.user(),
            llm,
            tts,
            ml,
            transport.output(),
            transcript.assistant(),
            context_aggregator.assistant(),
        ]
    )

    pipeline = Pipeline(pipeline_processors)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[
            RTVIObserver(rtvi),
            LLMLogObserver(),
            UserBotLatencyLogObserver(),
        ],  # RTVI protocol events
        cancel_on_idle_timeout=False,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        bot_logger.info(f"Client connected")
        # Kick off the conversation.
        # messages.append(
        #     {"role": "system", "content": "Please introduce yourself to the user."}
        # )
        # await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        bot_logger.info(f"Client disconnected")
        await task.cancel()

    @transcript.event_handler("on_transcript_update")
    async def handle_transcript_update(processor, frame):
        # Each message contains role (user/assistant), content, and timestamp
        for message in frame.messages:
            bot_logger.info(
                f"Bot transcript:[{message.timestamp}] {message.role}: {message.content}"
            )

    # Handle client connection
    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Signal bot is ready to receive messages
        await rtvi.set_bot_ready()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)
