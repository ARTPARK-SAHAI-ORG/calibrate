"""Happy path tests for TTS provider synthesize methods (heavy SDK mocking)."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


class TestSynthesizeOpenAI(unittest.IsolatedAsyncioTestCase):
    async def test_openai_happy(self):
        from calibrate_agent.tts import eval as E

        class FakeResponse:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def iter_bytes(self):
                yield b"chunk1"
                yield b"chunk2"

        fake_client = MagicMock()
        fake_client.audio.speech.with_streaming_response.create = MagicMock(
            return_value=FakeResponse()
        )

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), \
             patch.object(E, "AsyncOpenAI", return_value=fake_client):
            path = Path(tmp) / "x.wav"
            result = await E.synthesize_openai("hi", "english", str(path))
        self.assertIsNotNone(result.get("ttfb"))


class TestSynthesizeGroq(unittest.IsolatedAsyncioTestCase):
    async def test_groq_happy(self):
        from calibrate_agent.tts import eval as E

        fake_response = MagicMock()
        fake_response.write_to_file = AsyncMock()

        fake_client = MagicMock()
        fake_client.audio.speech.create = AsyncMock(return_value=fake_response)

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"GROQ_API_KEY": "k"}), \
             patch.object(E, "AsyncGroq", return_value=fake_client):
            path = Path(tmp) / "x.wav"
            await E.synthesize_groq("hi", "english", str(path))


class TestSynthesizeCartesia(unittest.IsolatedAsyncioTestCase):
    async def test_cartesia_happy(self):
        from calibrate_agent.tts import eval as E

        class FakeBytesIter:
            def __aiter__(self):
                async def gen():
                    yield b"chunk1"
                    yield b"chunk2"
                return gen()

        fake_client = MagicMock()
        fake_client.tts.bytes = MagicMock(return_value=FakeBytesIter())

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"CARTESIA_API_KEY": "k"}), \
             patch.object(E, "AsyncCartesia", return_value=fake_client):
            path = Path(tmp) / "x.wav"
            result = await E.synthesize_cartesia("hi", "english", str(path))
        self.assertIsNotNone(result.get("ttfb"))


class TestSynthesizeElevenlabs(unittest.IsolatedAsyncioTestCase):
    async def test_elevenlabs_happy(self):
        from calibrate_agent.tts import eval as E

        class FakeStream:
            def __aiter__(self):
                async def gen():
                    yield b"mp3chunk1"
                    yield b"mp3chunk2"
                return gen()

        fake_client = MagicMock()
        fake_client.text_to_speech.stream = MagicMock(return_value=FakeStream())

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}), \
             patch.object(E, "AsyncElevenLabs", return_value=fake_client), \
             patch.object(E, "convert_mp3_to_wav"):
            path = Path(tmp) / "x.wav"
            await E.synthesize_elevenlabs("hi", "english", str(path))

    async def test_elevenlabs_sindhi(self):
        from calibrate_agent.tts import eval as E

        class FakeStream:
            def __aiter__(self):
                async def gen():
                    yield b"mp3"
                return gen()

        fake_client = MagicMock()
        fake_client.text_to_dialogue.stream = MagicMock(return_value=FakeStream())

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}), \
             patch.object(E, "AsyncElevenLabs", return_value=fake_client), \
             patch.object(E, "convert_mp3_to_wav"):
            path = Path(tmp) / "x.wav"
            await E.synthesize_elevenlabs("hi", "sindhi", str(path))


class TestSynthesizeSarvam(unittest.IsolatedAsyncioTestCase):
    async def test_sarvam_happy(self):
        from calibrate_agent.tts import eval as E
        from sarvamai import AudioOutput, EventResponse

        audio_msg = MagicMock()
        audio_msg.__class__ = AudioOutput
        audio_msg.data = MagicMock()
        audio_msg.data.audio = "aGVsbG8="  # base64("hello")

        event_msg = MagicMock()
        event_msg.__class__ = EventResponse
        event_msg.data = MagicMock()
        event_msg.data.event_type = "final"

        class FakeWS:
            _websocket = MagicMock(closed=True)

            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def configure(self, **kwargs): pass
            async def convert(self, text): pass
            async def flush(self): pass
            def __aiter__(self):
                async def gen():
                    yield audio_msg
                    yield event_msg
                return gen()

        fake_client = MagicMock()
        fake_client.text_to_speech_streaming.connect = MagicMock(return_value=FakeWS())

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"SARVAM_API_KEY": "k"}), \
             patch.object(E, "AsyncSarvamAI", return_value=fake_client), \
             patch.object(E, "convert_mp3_to_wav"):
            path = Path(tmp) / "x.wav"
            result = await E.synthesize_sarvam("hi", "english", str(path))
        self.assertIsNotNone(result.get("ttfb"))


class _FakeSmallestWS:
    def __init__(self, messages):
        self._messages = messages
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        self._it = iter(self._messages)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _FakeSmallestConnect:
    def __init__(self, ws):
        self._ws = ws

    async def __aenter__(self):
        return self._ws

    async def __aexit__(self, *exc):
        return False


class TestSynthesizeSmallest(unittest.IsolatedAsyncioTestCase):
    async def test_smallest_happy(self):
        import base64
        import json
        from calibrate_agent.tts import eval as E

        audio_b64 = base64.b64encode(b"\x01\x02" * 100).decode()
        messages = [
            json.dumps({"status": "chunk", "data": {"audio": audio_b64}}),
            json.dumps({"status": "chunk", "data": {"audio": audio_b64}}),
            json.dumps({"status": "complete"}),
        ]
        ws = _FakeSmallestWS(messages)

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"SMALLEST_API_KEY": "k"}), \
             patch("websockets.asyncio.client.connect",
                   MagicMock(return_value=_FakeSmallestConnect(ws))):
            path = Path(tmp) / "x.wav"
            result = await E.synthesize_smallest("hi", "english", str(path))
            self.assertIsNotNone(result.get("ttfb"))
            self.assertTrue(path.exists())
            self.assertEqual(len(ws.sent), 1)  # one JSON synthesis request

    async def test_smallest_server_error(self):
        import json
        from calibrate_agent.tts import eval as E

        ws = _FakeSmallestWS([json.dumps({"status": "error", "error": "boom"})])
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"SMALLEST_API_KEY": "k"}), \
             patch("websockets.asyncio.client.connect",
                   MagicMock(return_value=_FakeSmallestConnect(ws))):
            path = Path(tmp) / "x.wav"
            with self.assertRaises(RuntimeError):
                await E.synthesize_smallest("hi", "english", str(path))


class TestSynthesizeGoogle(unittest.IsolatedAsyncioTestCase):
    async def test_google_streaming(self):
        from calibrate_agent.tts import eval as E

        fake_response1 = MagicMock()
        fake_response1.audio_content = b"chunk1"
        fake_response2 = MagicMock()
        fake_response2.audio_content = b"chunk2"

        fake_client = MagicMock()
        fake_client.streaming_synthesize = MagicMock(
            return_value=iter([fake_response1, fake_response2])
        )

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/creds.json"}), \
             patch("calibrate_agent.tts.eval.texttospeech.TextToSpeechClient",
                   return_value=fake_client), \
             patch("calibrate_agent.tts.eval.save_audio"):
            path = Path(tmp) / "x.wav"
            result = await E.synthesize_google("hi", "english", str(path))
        self.assertIsNotNone(result.get("ttfb"))

    async def test_google_sindhi(self):
        from calibrate_agent.tts import eval as E

        fake_response = MagicMock()
        fake_response.audio_content = b"audio"

        fake_client = MagicMock()
        fake_client.synthesize_speech = MagicMock(return_value=fake_response)

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/creds.json"}), \
             patch("calibrate_agent.tts.eval.texttospeech.TextToSpeechClient",
                   return_value=fake_client), \
             patch("calibrate_agent.tts.eval.save_audio"):
            path = Path(tmp) / "x.wav"
            await E.synthesize_google("hi", "sindhi", str(path))


if __name__ == "__main__":
    unittest.main()
