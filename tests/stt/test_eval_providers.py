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


class TestTranscribeOruk(unittest.IsolatedAsyncioTestCase):
    async def test_oruk_happy(self):
        from calibrate_agent.stt import eval as E
        from calibrate_agent.utils import ORUK_STT_BASE_URL

        fake_resp = MagicMock()
        fake_resp.text = "  hello from fourier  "
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create = AsyncMock(return_value=fake_resp)

        with patch.dict(os.environ, {"ORUK_API_KEY": "ok-test"}, clear=True), \
             patch.object(E, "load_audio", side_effect=_mock_load_audio), \
             patch.object(E, "AsyncOpenAI", return_value=fake_client) as openai_cls:
            result = await E.transcribe_oruk(Path("/tmp/x.wav"), "english")

        self.assertEqual(result["transcript"], "hello from fourier")
        openai_cls.assert_called_once_with(
            api_key="ok-test", base_url=ORUK_STT_BASE_URL
        )
        call_kwargs = fake_client.audio.transcriptions.create.await_args.kwargs
        self.assertEqual(
            set(call_kwargs.keys()),
            {"file", "model", "response_format", "language"},
        )
        self.assertEqual(call_kwargs["model"], "oruk-fourier-2b")
        self.assertEqual(call_kwargs["response_format"], "json")
        self.assertEqual(call_kwargs["language"], "en")

    async def test_oruk_custom_base_url(self):
        from calibrate_agent.stt import eval as E

        fake_client = MagicMock()
        fake_client.audio.transcriptions.create = AsyncMock(return_value="plain text ")

        with patch.dict(
            os.environ,
            {
                "ORUK_API_KEY": "ok-test",
                "ORUK_BASE_URL": "https://staging.oruk.example/v1",
            },
            clear=True,
        ), patch.object(E, "load_audio", side_effect=_mock_load_audio), \
             patch.object(E, "AsyncOpenAI", return_value=fake_client) as openai_cls:
            result = await E.transcribe_oruk(Path("/tmp/x.wav"), "english")

        self.assertEqual(result["transcript"], "plain text")
        openai_cls.assert_called_once_with(
            api_key="ok-test", base_url="https://staging.oruk.example/v1"
        )

    async def test_oruk_missing_key(self):
        from calibrate_agent.stt import eval as E

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await E.transcribe_oruk(Path("/tmp/x.wav"), "english")

    async def test_oruk_rejects_non_english(self):
        from calibrate_agent.stt import eval as E

        with patch.dict(os.environ, {"ORUK_API_KEY": "ok-test"}, clear=True):
            with self.assertRaises(ValueError):
                await E.transcribe_oruk(Path("/tmp/x.wav"), "hindi")

    async def test_transcribe_audio_routes_to_oruk(self):
        from calibrate_agent.stt import eval as E

        fake_fn = AsyncMock(return_value={"transcript": "  routed  "})
        inner = E.transcribe_audio
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        with patch.object(E, "transcribe_oruk", fake_fn), \
             patch.object(E, "langfuse_enabled", False):
            result = await inner(Path("/tmp/x.wav"), "ref", "oruk", "english", "u1")

        self.assertEqual(result["transcript"], "routed")
        fake_fn.assert_awaited_once_with(Path("/tmp/x.wav"), "english")


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
