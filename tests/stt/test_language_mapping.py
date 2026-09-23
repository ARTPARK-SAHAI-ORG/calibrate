"""Tests for get_stt_language 13-language coverage + Google Sindhi routing."""

import os
import unittest
from unittest.mock import patch


STT_LANGUAGES = [
    "english", "hindi", "kannada", "bengali", "malayalam", "marathi",
    "odia", "punjabi", "tamil", "telugu", "gujarati", "sindhi", "maithili",
]


class TestGetSTTLanguage(unittest.TestCase):
    def test_all_13_languages_map_to_a_language_enum(self):
        from calibrate_agent.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        for lang in STT_LANGUAGES:
            self.assertIsInstance(get_stt_language(lang, "deepgram"), Language)
            self.assertIsInstance(get_stt_language(lang, "sarvam"), Language)

    def test_non_english_does_not_fall_back_to_english(self):
        from calibrate_agent.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        # The bug this fixes: tamil/telugu/etc. used to become English.
        self.assertEqual(get_stt_language("tamil", "deepgram"), Language.TA)
        self.assertEqual(get_stt_language("telugu", "deepgram"), Language.TE)
        self.assertEqual(get_stt_language("bengali", "deepgram"), Language.BN)
        self.assertNotEqual(
            get_stt_language("tamil", "deepgram"),
            get_stt_language("english", "deepgram"),
        )

    def test_sarvam_uses_regional_variants(self):
        from calibrate_agent.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language("tamil", "sarvam"), Language.TA_IN)
        self.assertEqual(get_stt_language("hindi", "sarvam"), Language.HI_IN)

    def test_unknown_language_falls_back_to_english(self):
        from calibrate_agent.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language("klingon", "deepgram"), Language.EN)
        self.assertEqual(get_stt_language("klingon", "sarvam"), Language.EN_IN)


class TestGoogleModelAndLocation(unittest.TestCase):
    def test_sindhi_vs_default(self):
        from calibrate_agent.utils import (
            google_stt_model_and_location,
            STT_PROVIDER_MODELS,
        )

        self.assertEqual(
            google_stt_model_and_location("sindhi"), ("chirp_2", "asia-southeast1")
        )
        self.assertEqual(
            google_stt_model_and_location("hindi"),
            (STT_PROVIDER_MODELS["google"], "us"),
        )
        # Explicit model override wins for non-Sindhi.
        self.assertEqual(google_stt_model_and_location("hindi", "chirp_3"), ("chirp_3", "us"))


class TestGoogleSindhiRouting(unittest.TestCase):
    ALL_KEYS = {"GOOGLE_APPLICATION_CREDENTIALS": "/creds.json"}

    def test_sindhi_uses_chirp2_asia_southeast1(self):
        from calibrate_agent.utils import create_stt_service

        target = "pipecat.services.google.stt.GoogleSTTService"
        with patch.dict(os.environ, self.ALL_KEYS), patch(target) as svc:
            create_stt_service("google", "sindhi")
        self.assertEqual(svc.call_args.kwargs["location"], "asia-southeast1")
        self.assertEqual(svc.Settings.call_args.kwargs["model"], "chirp_2")

    def test_non_sindhi_uses_default_model_and_us(self):
        from calibrate_agent.utils import create_stt_service, STT_PROVIDER_MODELS

        target = "pipecat.services.google.stt.GoogleSTTService"
        with patch.dict(os.environ, self.ALL_KEYS), patch(target) as svc:
            create_stt_service("google", "hindi")
        self.assertEqual(svc.call_args.kwargs["location"], "us")
        self.assertEqual(
            svc.Settings.call_args.kwargs["model"], STT_PROVIDER_MODELS["google"]
        )


if __name__ == "__main__":
    unittest.main()


class TestRealtimeProviderLanguages(unittest.TestCase):
    """Language rules for the realtime streaming providers: officially listed
    languages run, auto-detecting models run everything, and the rest are
    rejected with the provider to use instead."""

    def test_rejections_name_the_alternative_provider(self):
        from calibrate_agent.utils import validate_stt_language

        for provider, language, alternative in [
            ("deepgram-flux", "kannada", "deepgram"),
            ("openai-live", "odia", "openai-transcribe"),
            ("cartesia-ink-preview", "tamil", "cartesia"),
        ]:
            with self.assertRaises(ValueError) as ctx:
                validate_stt_language(language, provider)
            self.assertIn(f"'{alternative}'", str(ctx.exception))

    def test_supported_languages_pass(self):
        from calibrate_agent.utils import validate_stt_language

        for provider, language in [
            ("deepgram-flux", "english"),
            ("deepgram-flux", "hindi"),
            ("openai-live", "kannada"),
            ("cartesia-ink-preview", "hindi"),
        ]:
            validate_stt_language(language, provider)

    def test_auto_detecting_and_full_coverage_providers_run_all_languages(self):
        from calibrate_agent.utils import validate_stt_language

        for language in STT_LANGUAGES:
            validate_stt_language(language, "openai-transcribe")
            validate_stt_language(language, "sarvam-v4")
            validate_stt_language(language, "sarvam-v3-realtime")

    def test_existing_provider_rejection_has_no_alternative(self):
        from calibrate_agent.utils import validate_stt_language

        with self.assertRaises(ValueError) as ctx:
            validate_stt_language("maithili", "deepgram")
        self.assertNotIn("instead", str(ctx.exception))

    def test_openai_transcribe_codes(self):
        from calibrate_agent.utils import get_stt_language_code

        self.assertEqual(get_stt_language_code("hindi", "openai-transcribe"), "hi")
        self.assertIsNone(get_stt_language_code("odia", "openai-transcribe"))

    def test_sarvam_v4_codes_and_enums(self):
        from calibrate_agent.utils import get_stt_language, get_stt_language_code
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language_code("sindhi", "sarvam-v4"), "sd-IN")
        self.assertEqual(get_stt_language_code("odia", "sarvam-v4"), "od-IN")
        self.assertEqual(get_stt_language("hindi", "sarvam-v4"), Language.HI_IN)
        self.assertEqual(get_stt_language_code("sindhi", "sarvam-v3-realtime"), "sd-IN")
        self.assertEqual(get_stt_language("hindi", "sarvam-v3-realtime"), Language.HI_IN)
        self.assertEqual(get_stt_language_code("maithili", "sarvam-v4-manual"), "mai-IN")
        self.assertEqual(get_stt_language("hindi", "sarvam-v4-manual"), Language.HI_IN)

    def test_stt_provider_model_per_language(self):
        from calibrate_agent.utils import stt_provider_model

        self.assertEqual(stt_provider_model("deepgram-flux", "english"), "flux-general-en")
        self.assertEqual(stt_provider_model("deepgram-flux", "hindi"), "flux-general-multi")
        self.assertEqual(stt_provider_model("deepgram-flux"), "flux-general-en")
        self.assertEqual(stt_provider_model("google", "sindhi"), "chirp_2")
        self.assertEqual(stt_provider_model("sarvam-v4", "hindi"), "saaras:v4")
        self.assertEqual(
            stt_provider_model("sarvam-v3-realtime", "hindi"), "saaras:v3-realtime"
        )
        self.assertEqual(stt_provider_model("sarvam-v4-manual", "hindi"), "saaras:v4")
