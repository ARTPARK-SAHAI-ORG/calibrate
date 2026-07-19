"""Tests for get_stt_language 13-language coverage + Google Sindhi routing."""

import os
import unittest
from unittest.mock import patch


STT_LANGUAGES = [
    "english",
    "hindi",
    "kannada",
    "bengali",
    "malayalam",
    "marathi",
    "odia",
    "punjabi",
    "tamil",
    "telugu",
    "gujarati",
    "sindhi",
    "maithili",
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
        self.assertEqual(
            google_stt_model_and_location("hindi", "chirp_3"), ("chirp_3", "us")
        )


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
