"""Cover create_stt_service / create_tts_service provider branches."""

import os
import unittest
from unittest.mock import patch, MagicMock


class TestCreateSTTService(unittest.TestCase):
    """Each provider branch — patch the imported pipecat service classes."""

    def _patch_imports(self):
        """Patch all pipecat services imported lazily in create_stt_service."""
        patches = [
            patch("pipecat.services.deepgram.stt.DeepgramSTTService"),
            patch("pipecat.services.deepgram.stt.LiveOptions"),
            patch("pipecat.services.openai.stt.OpenAISTTService"),
            patch("pipecat.services.google.stt.GoogleSTTService"),
            patch("pipecat.services.cartesia.stt.CartesiaSTTService"),
            patch("pipecat.services.cartesia.stt.CartesiaLiveOptions"),
            patch("pipecat.services.groq.stt.GroqSTTService"),
            patch("pipecat.services.sarvam.stt.SarvamSTTService"),
            patch("pipecat.services.elevenlabs.stt.ElevenLabsRealtimeSTTService"),
            patch("pipecat.services.smallest.stt.SmallestSTTService"),
        ]
        return patches

    def test_deepgram(self):
        from calibrate.utils import create_stt_service

        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "k"}):
            for p in self._patch_imports():
                p.start()
            try:
                create_stt_service("deepgram", "english")
            finally:
                for p in self._patch_imports():
                    p.stop()

    def test_each_provider(self):
        from calibrate.utils import create_stt_service

        envs = {
            "DEEPGRAM_API_KEY": "k",
            "SARVAM_API_KEY": "k",
            "OPENAI_API_KEY": "k",
            "GROQ_API_KEY": "k",
            "CARTESIA_API_KEY": "k",
            "ELEVENLABS_API_KEY": "k",
            "SMALLEST_API_KEY": "k",
            "GOOGLE_APPLICATION_CREDENTIALS": "/creds.json",
        }
        with patch.dict(os.environ, envs):
            patches = self._patch_imports()
            started = [p.start() for p in patches]
            try:
                for prov in ["deepgram", "sarvam", "elevenlabs", "openai",
                             "cartesia", "smallest", "groq", "google"]:
                    create_stt_service(prov, "english")
            finally:
                for p in patches:
                    p.stop()


class TestCreateTTSService(unittest.TestCase):
    def _patch_imports(self):
        return [
            patch("pipecat.services.cartesia.tts.CartesiaTTSService"),
            patch("pipecat.services.openai.tts.OpenAITTSService"),
            patch("pipecat.services.groq.tts.GroqTTSService"),
            patch("pipecat.services.google.tts.GoogleTTSService"),
            patch("pipecat.services.elevenlabs.tts.ElevenLabsTTSService"),
            patch("pipecat.services.sarvam.tts.SarvamTTSService"),
            patch("pipecat.services.deepgram.tts.DeepgramTTSService"),
            patch("pipecat.services.smallest.tts.SmallestTTSService"),
        ]

    def test_each_provider(self):
        from calibrate.utils import create_tts_service

        envs = {
            "OPENAI_API_KEY": "k",
            "GROQ_API_KEY": "k",
            "CARTESIA_API_KEY": "k",
            "ELEVENLABS_API_KEY": "k",
            "SARVAM_API_KEY": "k",
            "DEEPGRAM_API_KEY": "k",
            "SMALLEST_API_KEY": "k",
            "GOOGLE_APPLICATION_CREDENTIALS": "/creds.json",
        }
        with patch.dict(os.environ, envs):
            patches = self._patch_imports()
            for p in patches:
                p.start()
            try:
                for prov in ["cartesia", "openai", "groq", "google", "elevenlabs",
                             "sarvam", "deepgram", "smallest"]:
                    create_tts_service(prov, "english")
            finally:
                for p in patches:
                    p.stop()


if __name__ == "__main__":
    unittest.main()
