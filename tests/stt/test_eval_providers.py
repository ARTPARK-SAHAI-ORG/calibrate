"""Happy path tests for STT provider transcribe methods (heavy SDK mocking)."""

import asyncio
import os
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


def _mock_load_audio(*args, **kwargs):
    """Common load_audio mock returning fake audio bytes."""
    if kwargs.get("raw_pcm"):
        return b"\x00\x00" * 100
    if kwargs.get("as_file"):
        import io
        buf = io.BytesIO(b"RIFF\x00\x00\x00\x00WAVE")
        buf.name = "audio.wav"
        return buf
    return b"RIFF\x00\x00\x00\x00WAVE"


class TestTranscribeGroq(unittest.IsolatedAsyncioTestCase):
    async def test_groq_happy(self):
        from calibrate_agent.stt import eval as E

        fake_client = MagicMock()
        fake_client.audio.transcriptions.create = AsyncMock(return_value="hello world ")

        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}), \
             patch.object(E, "load_audio", side_effect=_mock_load_audio), \
             patch.object(E, "AsyncGroq", return_value=fake_client):
            result = await E.transcribe_groq(Path("/tmp/x.wav"), "english")
        self.assertEqual(result["transcript"], "hello world")


class TestTranscribeSarvam(unittest.IsolatedAsyncioTestCase):
    async def test_sarvam_happy(self):
        from calibrate_agent.stt import eval as E

        # Build a fake message stream
        msg = MagicMock()
        msg.type = "data"
        msg.data.transcript = "hello"
        msg.data.metrics.processing_latency = 0.5

        class FakeWS:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            async def transcribe(self, **kwargs): pass
            async def flush(self): pass
            def __aiter__(self):
                async def gen():
                    yield msg
                return gen()

        fake_client = MagicMock()
        fake_client.speech_to_text_streaming.connect = MagicMock(return_value=FakeWS())

        with patch.dict(os.environ, {"SARVAM_API_KEY": "k"}), \
             patch.object(E, "load_audio", side_effect=_mock_load_audio), \
             patch.object(E, "AsyncSarvamAI", return_value=fake_client):
            result = await E.transcribe_sarvam(Path("/tmp/x.wav"), "english")
        self.assertEqual(result["transcript"], "hello")

    async def test_sarvam_error_message(self):
        from calibrate_agent.stt import eval as E

        err_msg = MagicMock()
        err_msg.type = "error"
        err_msg.data.error = "boom"

        class FakeWS:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            async def transcribe(self, **kwargs): pass
            async def flush(self): pass
            def __aiter__(self):
                async def gen():
                    yield err_msg
                return gen()

        fake_client = MagicMock()
        fake_client.speech_to_text_streaming.connect = MagicMock(return_value=FakeWS())

        with patch.dict(os.environ, {"SARVAM_API_KEY": "k"}), \
             patch.object(E, "load_audio", side_effect=_mock_load_audio), \
             patch.object(E, "AsyncSarvamAI", return_value=fake_client):
            with self.assertRaises(RuntimeError):
                await E.transcribe_sarvam(Path("/tmp/x.wav"), "english")


class TestTranscribeGoogle(unittest.IsolatedAsyncioTestCase):
    async def test_google_happy(self):
        from calibrate_agent.stt import eval as E

        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/creds.json",
                                      "GOOGLE_CLOUD_PROJECT_ID": "proj"}), \
             patch("calibrate_agent.stt.eval._transcribe_google_streaming",
                   return_value={"transcript": "hello world"}):
            result = await E.transcribe_google(Path("/tmp/x.wav"), "english")
        self.assertEqual(result["transcript"], "hello world")

    async def test_google_sindhi(self):
        from calibrate_agent.stt import eval as E

        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/creds.json",
                                      "GOOGLE_CLOUD_PROJECT_ID": "proj"}), \
             patch("calibrate_agent.stt.eval._transcribe_google_streaming",
                   return_value={"transcript": "hello"}):
            result = await E.transcribe_google(Path("/tmp/x.wav"), "sindhi")
        self.assertEqual(result["transcript"], "hello")


if __name__ == "__main__":
    unittest.main()
