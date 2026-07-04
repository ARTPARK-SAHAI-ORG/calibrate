"""Cover remaining gaps in stt/eval, status, and other modules to push to 70%."""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


# ── load_audio ImportError -------------------------------------------------

class TestLoadAudioImportError(unittest.TestCase):
    def test_missing_pydub(self):
        from calibrate.stt import eval as E

        # Force ImportError by patching the import statement
        import builtins
        orig_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pydub":
                raise ImportError("missing pydub")
            return orig_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            with self.assertRaises(ImportError):
                E.load_audio(Path("/tmp/x.wav"))


# ── transcribe_cartesia happy path ----------------------------------------

class TestTranscribeCartesia(unittest.IsolatedAsyncioTestCase):
    async def test_cartesia_happy(self):
        from calibrate.stt import eval as E

        # Build a fake websocket
        async def receive_gen():
            yield {"type": "transcript", "text": "hello ", "is_final": True}
            yield {"type": "transcript", "text": "world", "is_final": False}
            yield {"type": "done"}

        class FakeWS:
            send = AsyncMock()
            close = AsyncMock()

            def receive(self):
                return receive_gen()

        fake_ws = FakeWS()
        fake_client = MagicMock()
        fake_client.stt.websocket = AsyncMock(return_value=fake_ws)
        fake_client.close = AsyncMock()

        def _fake_load_audio(*args, **kwargs):
            return b"\x00" * 5000

        with patch.dict(os.environ, {"CARTESIA_API_KEY": "k"}), \
             patch.object(E, "load_audio", side_effect=_fake_load_audio), \
             patch.object(E, "AsyncCartesia", return_value=fake_client):
            result = await E.transcribe_cartesia(Path("/tmp/x.wav"), "english")

        self.assertIn("hello", result["transcript"])


# ── transcribe_smallest_streaming happy path -------------------------------
# (uses local import of `websockets.asyncio.client.connect` that's hard to
# patch portably — skipped here)


# ── status.py — sarvam and smallest checks (skipped earlier) --------------

class TestStatusSarvamCheck(unittest.IsolatedAsyncioTestCase):
    async def test_sarvam_check(self):
        from calibrate.status import _check_sarvam
        from sarvamai import AudioOutput

        audio_msg = MagicMock()
        audio_msg.__class__ = AudioOutput

        class FakeWS:
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
                return gen()

        fake_client = MagicMock()
        fake_client.text_to_speech_streaming.connect = MagicMock(return_value=FakeWS())

        with patch.dict(os.environ, {"SARVAM_API_KEY": "k"}), \
             patch("sarvamai.AsyncSarvamAI", return_value=fake_client):
            result = await _check_sarvam(MagicMock())
        self.assertEqual(result, "tts")


# Smallest status-check coverage lives in tests/test_status.py
# (TestCheckProviders.test_check_smallest_*), alongside the STT/TTS sub-checks.


# ── stt/eval - run_eval_only invalid dataset path ------------------------

class TestSTTRunEvalOnly(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_path(self):
        from calibrate.stt.eval import run_eval_only

        with tempfile.TemporaryDirectory() as tmp:
            result = await run_eval_only("/nonexistent/path.json", tmp)
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()
