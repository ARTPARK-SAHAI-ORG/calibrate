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
            self.assertIsInstance(get_stt_language(lang, "deepgram-nova-3"), Language)
            self.assertIsInstance(get_stt_language(lang, "sarvam-saaras-v3"), Language)

    def test_non_english_does_not_fall_back_to_english(self):
        from calibrate_agent.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        # The bug this fixes: tamil/telugu/etc. used to become English.
        self.assertEqual(get_stt_language("tamil", "deepgram-nova-3"), Language.TA)
        self.assertEqual(get_stt_language("telugu", "deepgram-nova-3"), Language.TE)
        self.assertEqual(get_stt_language("bengali", "deepgram-nova-3"), Language.BN)
        self.assertNotEqual(
            get_stt_language("tamil", "deepgram-nova-3"),
            get_stt_language("english", "deepgram-nova-3"),
        )

    def test_sarvam_uses_regional_variants(self):
        from calibrate_agent.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language("tamil", "sarvam-saaras-v3"), Language.TA_IN)
        self.assertEqual(get_stt_language("hindi", "sarvam-saaras-v3"), Language.HI_IN)

    def test_unknown_language_falls_back_to_english(self):
        from calibrate_agent.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language("klingon", "deepgram-nova-3"), Language.EN)
        self.assertEqual(get_stt_language("klingon", "sarvam-saaras-v3"), Language.EN_IN)


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
            (STT_PROVIDER_MODELS["google-chirp-3"], "us"),
        )
        # Explicit model override wins for non-Sindhi.
        self.assertEqual(google_stt_model_and_location("hindi", "chirp_3"), ("chirp_3", "us"))


class TestGoogleSindhiRouting(unittest.TestCase):
    ALL_KEYS = {"GOOGLE_APPLICATION_CREDENTIALS": "/creds.json"}

    def test_sindhi_uses_chirp2_asia_southeast1(self):
        from calibrate_agent.utils import create_stt_service

        target = "pipecat.services.google.stt.GoogleSTTService"
        with patch.dict(os.environ, self.ALL_KEYS), patch(target) as svc:
            create_stt_service("google-chirp-3", "sindhi")
        self.assertEqual(svc.call_args.kwargs["location"], "asia-southeast1")
        self.assertEqual(svc.Settings.call_args.kwargs["model"], "chirp_2")

    def test_non_sindhi_uses_default_model_and_us(self):
        from calibrate_agent.utils import create_stt_service, STT_PROVIDER_MODELS

        target = "pipecat.services.google.stt.GoogleSTTService"
        with patch.dict(os.environ, self.ALL_KEYS), patch(target) as svc:
            create_stt_service("google-chirp-3", "hindi")
        self.assertEqual(svc.call_args.kwargs["location"], "us")
        self.assertEqual(
            svc.Settings.call_args.kwargs["model"], STT_PROVIDER_MODELS["google-chirp-3"]
        )


if __name__ == "__main__":
    unittest.main()


class TestSttOutputName(unittest.TestCase):
    def test_deepgram_flux_adds_the_model_for_the_language(self):
        from calibrate_agent.utils import stt_output_name

        self.assertEqual(stt_output_name("deepgram-flux", "english"), "deepgram-flux-general-en")
        self.assertEqual(stt_output_name("deepgram-flux", "English"), "deepgram-flux-general-en")
        self.assertEqual(stt_output_name("deepgram-flux", "hindi"), "deepgram-flux-general-multi")

    def test_other_providers_keep_their_name(self):
        from calibrate_agent.utils import STT_PROVIDER_MODELS, stt_output_name

        for provider in set(STT_PROVIDER_MODELS) - {"deepgram-flux"}:
            self.assertEqual(stt_output_name(provider, "hindi"), provider)


class TestRealtimeProviderLanguages(unittest.TestCase):
    """Language rules for the realtime streaming providers: officially listed
    languages run, auto-detecting models run everything, and the rest are
    rejected with the provider to use instead."""

    def test_rejections_name_the_alternative_provider(self):
        from calibrate_agent.utils import validate_stt_language

        for provider, language, alternative in [
            ("deepgram-flux", "kannada", "deepgram-nova-3"),
            ("openai-gpt-live-transcribe", "odia", "openai-gpt-transcribe"),
            ("cartesia-ink-preview", "tamil", "cartesia-ink-whisper"),
        ]:
            with self.assertRaises(ValueError) as ctx:
                validate_stt_language(language, provider)
            self.assertIn(f"'{alternative}'", str(ctx.exception))

    def test_supported_languages_pass(self):
        from calibrate_agent.utils import validate_stt_language

        for provider, language in [
            ("deepgram-flux", "english"),
            ("deepgram-flux", "hindi"),
            ("openai-gpt-live-transcribe", "kannada"),
            ("cartesia-ink-preview", "hindi"),
        ]:
            validate_stt_language(language, provider)

    def test_auto_detecting_and_full_coverage_providers_run_all_languages(self):
        from calibrate_agent.utils import validate_stt_language

        for language in STT_LANGUAGES:
            validate_stt_language(language, "openai-gpt-transcribe")
            validate_stt_language(language, "sarvam-saaras-v4-realtime")
            validate_stt_language(language, "sarvam-saaras-v3-realtime")

    def test_existing_provider_rejection_has_no_alternative(self):
        from calibrate_agent.utils import validate_stt_language

        with self.assertRaises(ValueError) as ctx:
            validate_stt_language("maithili", "deepgram-nova-3")
        self.assertNotIn("instead", str(ctx.exception))

    def test_openai_transcribe_codes(self):
        from calibrate_agent.utils import get_stt_language_code

        self.assertEqual(get_stt_language_code("hindi", "openai-gpt-transcribe"), "hi")
        self.assertIsNone(get_stt_language_code("odia", "openai-gpt-transcribe"))

    def test_sarvam_v4_codes_and_enums(self):
        from calibrate_agent.utils import get_stt_language, get_stt_language_code
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language_code("sindhi", "sarvam-saaras-v4-realtime"), "sd-IN")
        self.assertEqual(get_stt_language_code("odia", "sarvam-saaras-v4-realtime"), "od-IN")
        self.assertEqual(get_stt_language("hindi", "sarvam-saaras-v4-realtime"), Language.HI_IN)
        self.assertEqual(get_stt_language_code("sindhi", "sarvam-saaras-v3-realtime"), "sd-IN")
        self.assertEqual(get_stt_language("hindi", "sarvam-saaras-v3-realtime"), Language.HI_IN)
        self.assertEqual(get_stt_language_code("maithili", "sarvam-saaras-v4"), "mai-IN")
        self.assertEqual(get_stt_language("hindi", "sarvam-saaras-v4"), Language.HI_IN)

    def test_every_saaras_v4_provider_accepts_sindhi(self):
        from calibrate_agent.utils import get_stt_language_code, validate_stt_language

        for provider in ("sarvam-saaras-v4", "sarvam-saaras-v4-realtime"):
            validate_stt_language("sindhi", provider)
            self.assertEqual(get_stt_language_code("sindhi", provider), "sd-IN")

    def test_stt_provider_model_per_language(self):
        from calibrate_agent.utils import stt_provider_model

        self.assertEqual(stt_provider_model("deepgram-flux", "english"), "flux-general-en")
        self.assertEqual(stt_provider_model("deepgram-flux", "hindi"), "flux-general-multi")
        self.assertEqual(stt_provider_model("deepgram-flux"), "flux-general-en")
        self.assertEqual(stt_provider_model("google-chirp-3", "sindhi"), "chirp_2")
        self.assertEqual(stt_provider_model("sarvam-saaras-v4-realtime", "hindi"), "saaras:v4")
        self.assertEqual(
            stt_provider_model("sarvam-saaras-v3-realtime", "hindi"), "saaras:v3-realtime"
        )
        self.assertEqual(stt_provider_model("sarvam-saaras-v4", "hindi"), "saaras:v4")


class TestUIProviderLanguagesMatchPython(unittest.TestCase):
    """ui/source/providers.ts lists each STT provider's languages by hand; for
    every language in the UI menu they must match STT_LANGUAGE_TABLES."""

    def test_ui_stt_languages_match_python_tables(self):
        import re
        from pathlib import Path
        from calibrate_agent.utils import STT_LANGUAGE_TABLES

        src = (Path(__file__).parents[2] / "ui/source/providers.ts").read_text()
        menu = set(re.findall(
            r"'([^']+)'", re.search(r"export const LANGUAGES = \[(.*?)\]", src, re.S).group(1)
        ))
        stt_src = src[src.index("export const STT_PROVIDERS"):]
        ui_langs = {
            pid: set(re.findall(r"'([^']+)'", langs))
            for pid, langs in re.findall(
                r"id: '([^']+)'.*?languages: new Set\(\[(.*?)\]\)", stt_src, re.S
            )
        }
        self.assertTrue(ui_langs)
        for provider, langs in ui_langs.items():
            python_langs = set(STT_LANGUAGE_TABLES[provider][0])
            self.assertEqual(langs & menu, python_langs & menu, provider)


class TestLanguageHelpersAcceptVendorNames(unittest.TestCase):
    def test_vendor_name_matches_its_default_provider(self):
        from calibrate_agent.utils import (
            get_stt_language,
            get_stt_language_code,
            validate_stt_language,
        )

        self.assertEqual(get_stt_language_code("hindi", "sarvam"), "hi-IN")
        self.assertEqual(
            get_stt_language("hindi", "sarvam"),
            get_stt_language("hindi", "sarvam-saaras-v3"),
        )
        validate_stt_language("hindi", "sarvam")
        with self.assertRaises(ValueError):
            validate_stt_language("sindhi", "sarvam")
