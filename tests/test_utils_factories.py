"""Cover create_stt_service / create_tts_service provider branches.

pipecat 1.0.0 passes per-provider config through ``<Service>.Settings(...)``, so
these tests patch the service class and inspect the recorded ``.Settings`` call.
"""

import os
import unittest
from unittest.mock import patch


STT_SERVICE_TARGETS = {
    "deepgram": "pipecat.services.deepgram.stt.DeepgramSTTService",
    "openai": "pipecat.services.openai.stt.OpenAISTTService",
    "groq": "pipecat.services.groq.stt.GroqSTTService",
    "google": "pipecat.services.google.stt.GoogleSTTService",
    "cartesia": "pipecat.services.cartesia.stt.CartesiaSTTService",
    "elevenlabs": "pipecat.services.elevenlabs.stt.ElevenLabsRealtimeSTTService",
    "smallest": "pipecat.services.smallest.stt.SmallestSTTService",
    "sarvam": "pipecat.services.sarvam.stt.SarvamSTTService",
}

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
        from calibrate.utils import create_stt_service

        for prov, target in STT_SERVICE_TARGETS.items():
            with patch.dict(os.environ, ALL_KEYS), patch(target):
                create_stt_service(prov, "english")

    def test_stt_models_come_from_shared_constant(self):
        """Every provider's model in create_stt_service must come from
        utils.STT_PROVIDER_MODELS — the SAME dict calibrate/stt/eval.py reads —
        so the live agent and the benchmark can't drift apart."""
        from calibrate.utils import create_stt_service, STT_PROVIDER_MODELS

        for prov, expected in STT_PROVIDER_MODELS.items():
            with patch.dict(os.environ, ALL_KEYS), \
                    patch(STT_SERVICE_TARGETS[prov]) as svc:
                create_stt_service(prov, "english")
                self.assertEqual(
                    svc.Settings.call_args.kwargs["model"], expected,
                    f"{prov}: STT model default drifted from STT_PROVIDER_MODELS",
                )

        # Guard against a new provider entering the constant without matching
        # parity coverage here.
        self.assertEqual(
            set(STT_PROVIDER_MODELS),
            {"deepgram", "openai", "groq", "google", "cartesia",
             "elevenlabs", "smallest", "sarvam"},
        )

    def test_sarvam_stt_transcribes(self):
        """Sarvam STT must use saaras:v3 with mode="transcribe" so pipecat routes
        it to the plain STT streaming endpoint (not translate), matching
        stt/eval.py's transcribe_sarvam."""
        from calibrate.utils import create_stt_service, STT_PROVIDER_MODELS

        with patch.dict(os.environ, ALL_KEYS), \
                patch(STT_SERVICE_TARGETS["sarvam"]) as svc:
            create_stt_service("sarvam", "english")
            self.assertEqual(svc.call_args.kwargs["mode"], "transcribe")
            self.assertEqual(
                svc.Settings.call_args.kwargs["model"], STT_PROVIDER_MODELS["sarvam"]
            )
            self.assertEqual(STT_PROVIDER_MODELS["sarvam"], "saaras:v3")


class TestCreateTTSService(unittest.TestCase):
    def test_each_provider(self):
        """Every provider branch constructs (pipecat service patched)."""
        from calibrate.utils import create_tts_service

        for prov, target in TTS_SERVICE_TARGETS.items():
            with patch.dict(os.environ, ALL_KEYS), patch(target):
                create_tts_service(prov, "english")

    def test_defaults_match_tts_eval(self):
        """The live-agent TTS defaults must stay in sync with the benchmark
        defaults in calibrate/tts/eval.py. Both models and voices come from the
        shared utils.TTS_PROVIDER_MODELS / TTS_PROVIDER_VOICES (which tts/eval.py
        also reads), so neither can drift. In pipecat 1.0.0 both are passed via
        <Service>.Settings(model=..., voice=...)."""
        from calibrate.utils import (
            create_tts_service,
            TTS_PROVIDER_MODELS,
            TTS_PROVIDER_VOICES,
            GOOGLE_TTS_VOICE_FAMILY,
        )

        # provider -> (expected model or None, expected voice) for english.
        expected = {
            "cartesia": (TTS_PROVIDER_MODELS["cartesia"], TTS_PROVIDER_VOICES["cartesia"]),
            "openai": (TTS_PROVIDER_MODELS["openai"], TTS_PROVIDER_VOICES["openai"]),
            "groq": (TTS_PROVIDER_MODELS["groq"], TTS_PROVIDER_VOICES["groq"]),
            "google": (None, f"en-US-{GOOGLE_TTS_VOICE_FAMILY}"),
            "elevenlabs": (TTS_PROVIDER_MODELS["elevenlabs"], TTS_PROVIDER_VOICES["elevenlabs"]),
            "sarvam": (TTS_PROVIDER_MODELS["sarvam"], TTS_PROVIDER_VOICES["sarvam"]),
            "smallest": (None, TTS_PROVIDER_VOICES["smallest"]),
        }

        # Completeness guard: `expected` must cover every single-sourced TTS
        # provider (both dicts, plus Google which is single-sourced via
        # GOOGLE_TTS_VOICE_FAMILY). Adding a provider to a constant without a
        # parity row here fails the test until it's covered too.
        self.assertEqual(
            set(expected),
            set(TTS_PROVIDER_MODELS) | set(TTS_PROVIDER_VOICES) | {"google"},
        )

        for prov, (model_val, voice_val) in expected.items():
            with patch.dict(os.environ, ALL_KEYS), \
                    patch(TTS_SERVICE_TARGETS[prov]) as svc:
                create_tts_service(prov, "english")
                kwargs = svc.Settings.call_args.kwargs
                if model_val is not None:
                    self.assertEqual(
                        kwargs["model"], model_val,
                        f"{prov}: TTS model drifted from tts/eval.py",
                    )
                self.assertEqual(
                    kwargs["voice"], voice_val,
                    f"{prov}: TTS voice drifted from tts/eval.py",
                )


if __name__ == "__main__":
    unittest.main()
