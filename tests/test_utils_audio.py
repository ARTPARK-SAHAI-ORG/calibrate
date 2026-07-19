"""Tests for utils.py audio combining functions."""

import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch, MagicMock


def _write_wav(
    path: Path,
    num_frames: int = 16000,
    sample_width: int = 2,
    num_channels: int = 1,
    frame_rate: int = 16000,
):
    with wave.open(str(path), "wb") as wf:
        wf.setsampwidth(sample_width)
        wf.setnchannels(num_channels)
        wf.setframerate(frame_rate)
        wf.writeframes(b"\x00\x00" * num_frames)


class TestCombineTurnAudioChunksForTurn(unittest.TestCase):
    def test_no_chunks(self):
        from calibrate_agent.utils import combine_turn_audio_chunks_for_turn

        with tempfile.TemporaryDirectory() as tmp:
            result = combine_turn_audio_chunks_for_turn(tmp, 1)
            self.assertTrue(result)

    def test_combines_role_chunks(self):
        from calibrate_agent.utils import combine_turn_audio_chunks_for_turn

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_wav(base / "1_bot_0.wav", num_frames=10)
            _write_wav(base / "1_bot_1.wav", num_frames=10)
            _write_wav(base / "1_user_0.wav", num_frames=10)
            result = combine_turn_audio_chunks_for_turn(tmp, 1)
            self.assertTrue(result)
            # Combined files exist, chunks deleted
            self.assertTrue((base / "1_bot.wav").exists())
            self.assertTrue((base / "1_user.wav").exists())
            self.assertFalse((base / "1_bot_0.wav").exists())

    def test_parameter_mismatch_skips_chunk(self):
        from calibrate_agent.utils import combine_turn_audio_chunks_for_turn

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_wav(base / "1_bot_0.wav", frame_rate=16000)
            _write_wav(base / "1_bot_1.wav", frame_rate=22050)  # mismatch
            result = combine_turn_audio_chunks_for_turn(tmp, 1)
            self.assertTrue(result)

    def test_read_error_handled(self):
        from calibrate_agent.utils import combine_turn_audio_chunks_for_turn

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "1_bot_0.wav").write_bytes(b"not a wav")
            result = combine_turn_audio_chunks_for_turn(tmp, 1)
            self.assertTrue(result)

    def test_no_chunk_match_pattern(self):
        from calibrate_agent.utils import combine_turn_audio_chunks_for_turn

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # File matches glob but not pattern
            _write_wav(base / "1_X_garbage.wav")
            result = combine_turn_audio_chunks_for_turn(tmp, 1)
            self.assertTrue(result)


class TestCombineTurnAudioChunks(unittest.TestCase):
    def test_no_files(self):
        from calibrate_agent.utils import combine_turn_audio_chunks

        with tempfile.TemporaryDirectory() as tmp:
            result = combine_turn_audio_chunks(tmp)
            self.assertFalse(result)

    def test_files_no_match(self):
        from calibrate_agent.utils import combine_turn_audio_chunks

        with tempfile.TemporaryDirectory() as tmp:
            _write_wav(Path(tmp) / "random.wav")
            result = combine_turn_audio_chunks(tmp)
            self.assertTrue(result)

    def test_combines_multiple_turns(self):
        from calibrate_agent.utils import combine_turn_audio_chunks

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_wav(base / "1_bot_0.wav", num_frames=10)
            _write_wav(base / "1_user_0.wav", num_frames=10)
            _write_wav(base / "2_bot_0.wav", num_frames=10)
            result = combine_turn_audio_chunks(tmp)
            self.assertTrue(result)
            self.assertTrue((base / "1_bot.wav").exists())
            self.assertTrue((base / "1_user.wav").exists())
            self.assertTrue((base / "2_bot.wav").exists())

    def test_param_mismatch(self):
        from calibrate_agent.utils import combine_turn_audio_chunks

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_wav(base / "1_bot_0.wav", frame_rate=16000)
            _write_wav(base / "1_bot_1.wav", frame_rate=22050)
            result = combine_turn_audio_chunks(tmp)
            self.assertTrue(result)

    def test_read_error(self):
        from calibrate_agent.utils import combine_turn_audio_chunks

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "1_bot_0.wav").write_bytes(b"junk")
            result = combine_turn_audio_chunks(tmp)
            self.assertTrue(result)


class TestCombineAudioFiles(unittest.TestCase):
    def test_no_audio_files(self):
        from calibrate_agent.utils import combine_audio_files

        with tempfile.TemporaryDirectory() as tmp:
            result = combine_audio_files(tmp, "/tmp/out.wav", None)
            self.assertFalse(result)

    def test_missing_transcript_raises(self):
        from calibrate_agent.utils import combine_audio_files

        with tempfile.TemporaryDirectory() as tmp:
            _write_wav(Path(tmp) / "a.wav")
            with self.assertRaises(FileNotFoundError):
                combine_audio_files(tmp, "/tmp/out.wav", "/nonexistent.json")

    def test_no_transcript_path_raises(self):
        from calibrate_agent.utils import combine_audio_files

        with tempfile.TemporaryDirectory() as tmp:
            _write_wav(Path(tmp) / "a.wav")
            with self.assertRaises(FileNotFoundError):
                combine_audio_files(tmp, "/tmp/out.wav", None)

    def test_basic_flow(self):
        from calibrate_agent.utils import combine_audio_files

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_wav(base / "1_bot.wav", num_frames=10)
            _write_wav(base / "2_user.wav", num_frames=10)
            transcript = base / "transcript.json"
            transcript.write_text(
                json.dumps(
                    [
                        {"role": "assistant", "content": "Hi"},
                        {"role": "user", "content": "Hello"},
                    ]
                )
            )
            out = base / "out.wav"
            result = combine_audio_files(tmp, str(out), str(transcript))
            self.assertTrue(result)
            self.assertTrue(out.exists())

    def test_skips_tool_calls_only(self):
        from calibrate_agent.utils import combine_audio_files

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_wav(base / "1_bot.wav", num_frames=10)
            transcript = base / "transcript.json"
            transcript.write_text(
                json.dumps(
                    [
                        {"role": "assistant", "content": "Hi"},
                        {"role": "assistant", "tool_calls": [{"x": 1}]},  # no content
                    ]
                )
            )
            out = base / "out.wav"
            result = combine_audio_files(tmp, str(out), str(transcript))
            self.assertTrue(result)

    def test_missing_audio_file_warning(self):
        from calibrate_agent.utils import combine_audio_files

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_wav(base / "1_bot.wav", num_frames=10)
            transcript = base / "transcript.json"
            transcript.write_text(
                json.dumps(
                    [
                        {"role": "assistant", "content": "Hi"},
                        {"role": "user", "content": "Hello"},  # no 2_user.wav
                    ]
                )
            )
            out = base / "out.wav"
            result = combine_audio_files(tmp, str(out), str(transcript))
            self.assertTrue(result)

    def test_no_matches_raises(self):
        from calibrate_agent.utils import combine_audio_files

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_wav(base / "random.wav")
            transcript = base / "transcript.json"
            transcript.write_text(
                json.dumps(
                    [
                        {"role": "system", "content": "..."},  # role not assistant/user
                    ]
                )
            )
            with self.assertRaises(ValueError):
                combine_audio_files(tmp, str(base / "out.wav"), str(transcript))

    def test_param_mismatch_skipped(self):
        from calibrate_agent.utils import combine_audio_files

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_wav(base / "1_bot.wav", frame_rate=16000)
            _write_wav(base / "2_user.wav", frame_rate=22050)
            transcript = base / "transcript.json"
            transcript.write_text(
                json.dumps(
                    [
                        {"role": "assistant", "content": "Hi"},
                        {"role": "user", "content": "Hello"},
                    ]
                )
            )
            result = combine_audio_files(tmp, str(base / "out.wav"), str(transcript))
            self.assertTrue(result)


class TestMetricsLogger(unittest.IsolatedAsyncioTestCase):
    async def test_process_frame(self):
        from collections import defaultdict
        from calibrate_agent.utils import MetricsLogger
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        ttfb = defaultdict(list)
        processing = defaultdict(list)
        logger = MetricsLogger(ttfb, processing)

        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {
            "label": "rtvi-ai",
            "type": "metrics",
            "data": {
                "ttfb": [
                    {"processor": "p1", "value": 0.5},
                    {"processor": "p2", "value": 0},
                ],
                "processing": [{"processor": "p1", "value": 0.3}],
            },
        }

        from unittest.mock import AsyncMock as _AsyncMock

        with (
            patch.object(
                FrameProcessor, "process_frame", _AsyncMock(return_value=None)
            ),
            patch.object(MetricsLogger, "push_frame", _AsyncMock(return_value=None)),
        ):
            await logger.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(ttfb["p1"], [0.5])
        self.assertEqual(processing["p1"], [0.3])

    async def test_process_frame_non_rtvi(self):
        from collections import defaultdict
        from calibrate_agent.utils import MetricsLogger
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        ttfb = defaultdict(list)
        processing = defaultdict(list)
        logger = MetricsLogger(ttfb, processing)

        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {"label": "other"}

        from unittest.mock import AsyncMock as _AsyncMock

        with (
            patch.object(
                FrameProcessor, "process_frame", _AsyncMock(return_value=None)
            ),
            patch.object(MetricsLogger, "push_frame", _AsyncMock(return_value=None)),
        ):
            await logger.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(len(ttfb), 0)


if __name__ == "__main__":
    unittest.main()
