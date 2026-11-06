"""Cover create_stt_service / create_tts_service provider branches.

pipecat passes per-provider config through ``<Service>.Settings(...)``, so
these tests patch the service class and inspect the recorded ``.Settings`` call.
"""

import os
import unittest
from unittest.mock import AsyncMock, patch


STT_SERVICE_TARGETS = {
    "deepgram-nova-3": "pipecat.services.deepgram.stt.DeepgramSTTService",
    "openai-gpt-4o-transcribe": "pipecat.services.openai.stt.OpenAISTTService",
    "groq-whisper-large-v3-turbo": "pipecat.services.groq.stt.GroqSTTService",
    "google-chirp-3": "pipecat.services.google.stt.GoogleSTTService",
    "cartesia-ink-whisper": "pipecat.services.cartesia.stt.CartesiaSTTService",
    "elevenlabs-scribe-v2-realtime": "pipecat.services.elevenlabs.stt.ElevenLabsRealtimeSTTService",
    "smallest-pulse": "pipecat.services.smallest.stt.SmallestSTTService",
    "sarvam-saaras-v3": "pipecat.services.sarvam.stt.SarvamSTTService",
    "deepgram-flux": "pipecat.services.deepgram.flux.stt.DeepgramFluxSTTService",
    "openai-gpt-transcribe": "pipecat.services.openai.stt.OpenAIRealtimeSTTService",
    "sarvam-saaras-v4-realtime": "pipecat.services.sarvam.stt.SarvamRealtimeSTTService",
    "sarvam-saaras-v3-realtime": "pipecat.services.sarvam.stt.SarvamRealtimeSTTService",
    "sarvam-saaras-v4": "pipecat.services.sarvam.stt.SarvamSTTService",
    "cartesia-ink-preview": "pipecat.services.cartesia.turns.stt.CartesiaTurnsSTTService",
}

# openai-gpt-live-transcribe builds a subclass of OpenAIRealtimeSTTService, which a patched
# class can't serve as a base for, so its tests construct the real service.
UNPATCHED_STT_PROVIDERS = {"openai-gpt-live-transcribe"}

TTS_SERVICE_TARGETS = {
    "cartesia": "pipecat.services.cartesia.tts.CartesiaTTSService",
    "openai": "pipecat.services.openai.tts.OpenAITTSService",
    "groq": "pipecat.services.groq.tts.GroqTTSService",
    "google": "pipecat.services.google.tts.GoogleTTSService",
    "elevenlabs": "pipecat.services.elevenlabs.tts.ElevenLabsTTSService",
    "sarvam": "pipecat.services.sarvam.tts.SarvamTTSService",
    "deepgram": "pipecat.services.deepgram.tts.DeepgramTTSService",
    "smallest": "pipecat.services.smallest.tts.SmallestTTSService",
}

# Providers that exist only in the benchmarks (calibrate_agent/{stt,tts}/eval.py),
# not in the live-agent factories — pipecat 1.0.0 has no cascaded service for them.
# Gemini STT/TTS run through google-genai in the eval path only, so they carry
# shared model/voice constants but have no create_*_service branch to keep in sync.
BENCHMARK_ONLY_PROVIDERS = {"gemini", "gemini-3.5-flash"}

ALL_KEYS = {
    "DEEPGRAM_API_KEY": "k",
    "SARVAM_API_KEY": "k",
    "OPENAI_API_KEY": "k",
    "GROQ_API_KEY": "k",
    "CARTESIA_API_KEY": "k",
    "ELEVENLABS_API_KEY": "k",
    "SMALLEST_API_KEY": "k",
    "GOOGLE_APPLICATION_CREDENTIALS": "/creds.json",
}


class TestCreateSTTService(unittest.TestCase):
    def test_each_provider(self):
        """Every provider branch constructs (pipecat service patched)."""
        from calibrate_agent.utils import create_stt_service

        for prov, target in STT_SERVICE_TARGETS.items():
            with patch.dict(os.environ, ALL_KEYS), patch(target):
                create_stt_service(prov, "english")

    def test_vendor_name_builds_the_default_provider(self):
        from calibrate_agent.utils import create_stt_service

        with patch.dict(os.environ, ALL_KEYS), \
             patch(STT_SERVICE_TARGETS["google-chirp-3"]) as google:
            create_stt_service("google", "english")
        google.assert_called_once()

    def test_stt_models_come_from_shared_constant(self):
        """Every provider's model in create_stt_service must come from
        utils.STT_PROVIDER_MODELS — the SAME dict calibrate_agent/stt/eval.py reads —
        so the live agent and the benchmark can't drift apart."""
        from calibrate_agent.utils import create_stt_service, STT_PROVIDER_MODELS

        for prov, expected in STT_PROVIDER_MODELS.items():
            if prov in BENCHMARK_ONLY_PROVIDERS | UNPATCHED_STT_PROVIDERS:
                continue
            with patch.dict(os.environ, ALL_KEYS), \
                    patch(STT_SERVICE_TARGETS[prov]) as svc:
                create_stt_service(prov, "english")
                self.assertEqual(
                    svc.Settings.call_args.kwargs["model"], expected,
                    f"{prov}: STT model default drifted from STT_PROVIDER_MODELS",
                )

        # Guard against a new provider entering the constant without matching
        # parity coverage here (benchmark-only providers excepted).
        self.assertEqual(
            set(STT_PROVIDER_MODELS) - BENCHMARK_ONLY_PROVIDERS,
            {"deepgram-nova-3", "openai-gpt-4o-transcribe",
             "groq-whisper-large-v3-turbo", "google-chirp-3", "cartesia-ink-whisper",
             "elevenlabs-scribe-v2-realtime", "smallest-pulse", "sarvam-saaras-v3",
             "deepgram-flux", "openai-gpt-live-transcribe", "openai-gpt-transcribe",
             "sarvam-saaras-v4-realtime", "sarvam-saaras-v3-realtime",
             "sarvam-saaras-v4", "cartesia-ink-preview"},
        )

    def test_unsupported_language_stops_with_alternative(self):
        from calibrate_agent.utils import create_stt_service

        for provider, alternative in (
            ("deepgram-flux", "deepgram-nova-3"),
            ("cartesia-ink-preview", "cartesia-ink-whisper"),
        ):
            with patch.dict(os.environ, ALL_KEYS), patch(STT_SERVICE_TARGETS[provider]) as svc:
                with self.assertRaisesRegex(ValueError, alternative):
                    create_stt_service(provider, "kannada")
                svc.assert_not_called()

    def test_agent_configs_accept_every_live_provider(self):
        from typing import get_args, get_type_hints
        from pydantic import ValidationError
        from calibrate_agent.utils import LIVE_STT_PROVIDERS, STT_PROVIDER_MODELS
        from calibrate_agent.agent import bot, test as agent_test

        self.assertEqual(
            set(LIVE_STT_PROVIDERS), set(STT_PROVIDER_MODELS) - BENCHMARK_ONLY_PROVIDERS
        )
        for provider in LIVE_STT_PROVIDERS:
            self.assertEqual(bot.STTConfig(provider=provider).provider, provider)
        with self.assertRaises(ValidationError):
            bot.STTConfig(provider="gemini-3.5-flash")
        self.assertEqual(
            set(get_args(get_type_hints(agent_test.STTConfig)["provider"])),
            set(LIVE_STT_PROVIDERS),
        )

    def test_sarvam_stt_transcribes(self):
        """Sarvam STT must use saaras:v3 with mode="transcribe" so pipecat routes
        it to the plain STT streaming endpoint (not translate), matching
        stt/eval.py's transcribe_sarvam."""
        from calibrate_agent.utils import create_stt_service, STT_PROVIDER_MODELS

        with patch.dict(os.environ, ALL_KEYS), \
                patch(STT_SERVICE_TARGETS["sarvam-saaras-v3"]) as svc:
            create_stt_service("sarvam-saaras-v3", "english")
            self.assertEqual(svc.call_args.kwargs["mode"], "transcribe")
            self.assertEqual(
                svc.Settings.call_args.kwargs["model"], STT_PROVIDER_MODELS["sarvam-saaras-v3"]
            )
            self.assertEqual(STT_PROVIDER_MODELS["sarvam-saaras-v3"], "saaras:v3")


class TestCreateRealtimeSTTService(unittest.IsolatedAsyncioTestCase):
    """The realtime streaming providers, built for real (offline) so the
    per-language model, hints and language codes can be read back."""

    def _build(self, provider, language):
        from calibrate_agent.utils import create_stt_service

        with patch.dict(os.environ, ALL_KEYS):
            return create_stt_service(provider, language)

    def test_providers_that_detect_turns_themselves(self):
        from calibrate_agent.utils import stt_decides_turns

        own_turns = {"deepgram-flux", "cartesia-ink-preview",
                     "sarvam-saaras-v4-realtime", "sarvam-saaras-v3-realtime"}
        for provider in own_turns | {"openai-gpt-live-transcribe", "openai-gpt-transcribe",
                                     "sarvam-saaras-v4"}:
            svc = self._build(provider, "english")
            self.assertEqual(stt_decides_turns(svc), provider in own_turns, provider)

    def test_deepgram_flux_english_uses_english_model_without_hints(self):
        svc = self._build("deepgram-flux", "english")
        self.assertEqual(svc._settings.model, "flux-general-en")
        self.assertIsNone(svc._settings.language_hints)

    def test_deepgram_flux_hindi_uses_multilingual_model_with_hint(self):
        from pipecat.transcriptions.language import Language

        svc = self._build("deepgram-flux", "hindi")
        self.assertEqual(svc._settings.model, "flux-general-multi")
        self.assertEqual(svc._settings.language_hints, [Language.HI])

    def test_openai_transcribe_sends_listed_language(self):
        svc = self._build("openai-gpt-transcribe", "hindi")
        self.assertEqual(svc._settings.model, "gpt-transcribe")
        self.assertEqual(str(svc._settings.language), "hi")

    def test_openai_transcribe_omits_unlisted_language(self):
        svc = self._build("openai-gpt-transcribe", "odia")
        self.assertIsNone(svc._settings.language)

    def test_sarvam_v4_passes_sarvam_language_code(self):
        svc = self._build("sarvam-saaras-v4-realtime", "odia")
        self.assertEqual(svc._settings.model, "saaras:v4")
        self.assertEqual(svc._settings.language_code, "od-IN")
        self.assertEqual(svc._settings.mode, "transcribe")

    def test_sarvam_v3_realtime_uses_realtime_service(self):
        from pipecat.services.sarvam.stt import SarvamRealtimeSTTService

        svc = self._build("sarvam-saaras-v3-realtime", "odia")
        self.assertIsInstance(svc, SarvamRealtimeSTTService)
        self.assertEqual(svc._settings.model, "saaras:v3-realtime")
        self.assertEqual(svc._settings.language_code, "od-IN")
        self.assertEqual(svc._settings.mode, "transcribe")

    def test_sarvam_v4_manual_uses_the_sarvam_streaming_service(self):
        """sarvam-saaras-v4 is saaras:v4 on the same service as sarvam, not on
        the realtime service that sarvam-saaras-v4-realtime uses."""
        from pipecat.services.sarvam.stt import SarvamSTTService
        from pipecat.transcriptions.language import Language

        svc = self._build("sarvam-saaras-v4", "hindi")
        self.assertIsInstance(svc, SarvamSTTService)
        self.assertEqual(svc._settings.model, "saaras:v4")
        self.assertEqual(svc._settings.language, Language.HI_IN)
        self.assertEqual(svc._mode, "transcribe")
        self.assertFalse(svc._settings.vad_signals)

    async def test_sarvam_v4_manual_flushes_on_our_voice_detection(self):
        from pipecat.frames.frames import VADUserStoppedSpeakingFrame
        from pipecat.processors.frame_processor import FrameDirection

        svc = self._build("sarvam-saaras-v4", "hindi")
        svc._socket_client = AsyncMock()
        with patch.object(type(svc).__mro__[1], "process_frame", AsyncMock()):
            await svc.process_frame(
                VADUserStoppedSpeakingFrame(stop_secs=0.2), FrameDirection.DOWNSTREAM
            )
        svc._socket_client.flush.assert_awaited_once()

    def test_cartesia_ink_preview_model(self):
        svc = self._build("cartesia-ink-preview", "hindi")
        self.assertEqual(svc._settings.model, "ink-preview")

    def test_end_of_turn_predictions_off(self):
        for provider in ("deepgram-flux", "cartesia-ink-preview"):
            svc = self._build(provider, "english")
            self.assertFalse(svc.eager_end_of_turn_enabled, provider)

    async def test_openai_live_sends_languages_list(self):
        from unittest.mock import AsyncMock
        from pipecat.services.openai.stt import OpenAIRealtimeSTTService

        svc = self._build("openai-gpt-live-transcribe", "hindi")
        self.assertIsInstance(svc, OpenAIRealtimeSTTService)
        self.assertEqual(svc._settings.model, "gpt-live-transcribe")
        with patch.object(
            OpenAIRealtimeSTTService, "_ws_send", new=AsyncMock()
        ) as sent:
            await svc._send_session_update()
        transcription = sent.call_args.args[0]["session"]["audio"]["input"][
            "transcription"
        ]
        self.assertEqual(transcription["languages"], ["hi"])
        self.assertNotIn("language", transcription)
        self.assertEqual(transcription["model"], "gpt-live-transcribe")

    def test_openai_live_service_class_built_once(self):
        first = self._build("openai-gpt-live-transcribe", "hindi")
        second = self._build("openai-gpt-live-transcribe", "english")
        self.assertIs(type(first), type(second))

    def test_openai_transcription_language(self):
        from calibrate_agent.utils import openai_transcription_language

        self.assertEqual(
            openai_transcription_language({"language": "hi"}, "openai-gpt-live-transcribe"),
            {"languages": ["hi"]},
        )
        self.assertEqual(
            openai_transcription_language({"language": "hi"}, "openai-gpt-transcribe"),
            {"language": "hi"},
        )
        self.assertEqual(
            openai_transcription_language({}, "openai-gpt-live-transcribe"), {}
        )


class TestCreateTTSService(unittest.TestCase):
    def test_each_provider(self):
        """Every provider branch constructs (pipecat service patched)."""
        from calibrate_agent.utils import create_tts_service

        for prov, target in TTS_SERVICE_TARGETS.items():
            with patch.dict(os.environ, ALL_KEYS), patch(target):
                create_tts_service(prov, "english")

    # Providers whose model create_tts_service passes explicitly (the rest carry
    # no shared TTS model — Google/Smallest set none, Deepgram isn't benchmarked).
    _TTS_MODEL_PROVIDERS = {"cartesia", "openai", "groq", "elevenlabs", "sarvam"}

    def test_defaults_match_tts_eval(self):
        """create_tts_service must pick the same model (TTS_PROVIDER_MODELS) and
        voice (get_tts_voice) that calibrate_agent/tts/eval.py uses — for every provider
        and every language — so the live agent and benchmark can't drift. Voices
        are per-language: overrides in TTS_PROVIDER_VOICES_BY_LANGUAGE, else the
        default. In pipecat 1.0.0 both are passed via
        <Service>.Settings(model=..., voice=...)."""
        from calibrate_agent.utils import (
            create_tts_service,
            TTS_PROVIDER_MODELS,
            TTS_PROVIDER_VOICES,
            TTS_PROVIDER_VOICES_BY_LANGUAGE,
            get_tts_voice,
        )

        benchmarked = set(TTS_SERVICE_TARGETS) - {"deepgram"}

        # Completeness guard: every provider carrying a shared model or voice
        # (default, per-language override, or Google's family) must be exercised
        # below. Adding one to a constant without covering it here fails the test.
        self.assertEqual(
            benchmarked,
            (
                set(TTS_PROVIDER_MODELS)
                | set(TTS_PROVIDER_VOICES)
                | set(TTS_PROVIDER_VOICES_BY_LANGUAGE)
                | {"google"}
            )
            - BENCHMARK_ONLY_PROVIDERS,
        )

        for prov in benchmarked:
            for language in ("english", "hindi", "kannada"):
                with patch.dict(os.environ, ALL_KEYS), \
                        patch(TTS_SERVICE_TARGETS[prov]) as svc:
                    create_tts_service(prov, language)
                    kwargs = svc.Settings.call_args.kwargs
                    self.assertEqual(
                        kwargs["voice"], get_tts_voice(prov, language),
                        f"{prov}/{language}: TTS voice drifted from get_tts_voice",
                    )
                    if prov in self._TTS_MODEL_PROVIDERS:
                        self.assertEqual(
                            kwargs["model"], TTS_PROVIDER_MODELS[prov],
                            f"{prov}: TTS model drifted from TTS_PROVIDER_MODELS",
                        )

        # The per-language overrides must actually be distinct — i.e. not silently
        # collapsed back to one voice per provider.
        self.assertNotEqual(
            get_tts_voice("cartesia", "hindi"), get_tts_voice("cartesia", "english")
        )
        self.assertNotEqual(
            get_tts_voice("smallest", "kannada"), get_tts_voice("smallest", "english")
        )


if __name__ == "__main__":
    unittest.main()
