"""Test _transcribe_google_streaming."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestGoogleStreaming(unittest.TestCase):
    def test_google_streaming_happy(self):
        from calibrate_agent.stt import eval as E

        # An interim hypothesis, then the final result, then an empty result.
        # Only the final should land in the transcript (interim filtered out).
        interim = MagicMock(is_final=False)
        interim.alternatives = [MagicMock(transcript="hello")]
        final = MagicMock(is_final=True)
        final.alternatives = [MagicMock(transcript="hello world")]
        empty = MagicMock(is_final=True)
        empty.alternatives = [MagicMock(transcript="")]

        interim_response = MagicMock(results=[interim])
        final_response = MagicMock(results=[final])
        empty_response = MagicMock(results=[empty])

        fake_client = MagicMock()
        fake_client.streaming_recognize.return_value = iter(
            [interim_response, final_response, empty_response]
        )

        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT_ID": "proj"}), \
             patch.object(E, "SpeechClient", return_value=fake_client), \
             patch.object(E, "load_audio", return_value=b"\x00" * 100000):
            result = E._transcribe_google_streaming(
                Path("/tmp/x.wav"), "en-US"
            )

        # Only the final result contributes to the transcript.
        self.assertEqual(result["transcript"], "hello world")

    def test_google_streaming_interim_results_enabled(self):
        from calibrate_agent.stt import eval as E

        captured = {}

        def fake_streaming_recognize(requests):
            reqs = list(requests)
            captured["config"] = reqs[0]
            return iter([])

        fake_client = MagicMock()
        fake_client.streaming_recognize.side_effect = fake_streaming_recognize

        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT_ID": "proj"}), \
             patch.object(E, "SpeechClient", return_value=fake_client), \
             patch.object(E, "load_audio", return_value=b"\x00" * 100):
            result = E._transcribe_google_streaming(Path("/tmp/x.wav"), "en-US")

        # No results -> empty transcript.
        self.assertEqual(result, {"transcript": ""})
        # Interim results are enabled on the streaming config so that final
        # transcripts can be filtered cleanly from evolving partial hypotheses.
        self.assertTrue(
            captured["config"].streaming_config.streaming_features.interim_results
        )


if __name__ == "__main__":
    unittest.main()
