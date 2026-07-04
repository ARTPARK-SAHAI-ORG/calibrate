"""Cover branches in calibrate_agent/langfuse.py."""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path


class TestNoOpObserveDecorator(unittest.TestCase):
    def test_observe_is_noop(self):
        from calibrate_agent import langfuse as L

        # When langfuse is not enabled, observe returns identity decorator.
        if not L.langfuse_enabled:
            @L.observe(name="x")
            def f(x):
                return x + 1
            self.assertEqual(f(1), 2)


class TestTestLangfuseConnection(unittest.TestCase):
    def test_success_path(self):
        from calibrate_agent import langfuse as L

        fake_client = MagicMock()
        fake_client.auth_check.return_value = True

        with patch("langfuse.get_client", return_value=fake_client):
            self.assertTrue(L.test_langfuse_connection())

    def test_failure_path(self):
        from calibrate_agent import langfuse as L

        with patch("langfuse.get_client", side_effect=Exception("nope")):
            self.assertFalse(L.test_langfuse_connection())


class TestCreateLangfuseAudioMedia(unittest.TestCase):
    def test_disabled_returns_none(self):
        from calibrate_agent import langfuse as L

        with patch.object(L, "langfuse_enabled", False):
            self.assertIsNone(L.create_langfuse_audio_media("/tmp/x.wav"))

    def test_lf_media_missing_returns_none(self):
        from calibrate_agent import langfuse as L

        with patch.object(L, "langfuse_enabled", True), \
             patch.object(L, "LangfuseMedia", None):
            self.assertIsNone(L.create_langfuse_audio_media("/tmp/x.wav"))

    def test_read_error_returns_none(self):
        from calibrate_agent import langfuse as L

        fake_media_cls = MagicMock()
        with patch.object(L, "langfuse_enabled", True), \
             patch.object(L, "LangfuseMedia", fake_media_cls):
            result = L.create_langfuse_audio_media("/nonexistent/audio.wav")
            self.assertIsNone(result)

    def test_success(self):
        from calibrate_agent import langfuse as L

        fake_media_cls = MagicMock(return_value="MEDIA")
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "a.wav"
            audio.write_bytes(b"\x00\x01\x02")
            with patch.object(L, "langfuse_enabled", True), \
                 patch.object(L, "LangfuseMedia", fake_media_cls):
                result = L.create_langfuse_audio_media(str(audio))
        self.assertEqual(result, "MEDIA")
        fake_media_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
