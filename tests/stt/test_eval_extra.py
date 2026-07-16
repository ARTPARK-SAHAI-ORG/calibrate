"""Extra coverage for stt/eval.py — provider routers and main pathway."""

import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pandas as pd


def _fake_intent_entity(intent=1, entity=1.0):
    """Adaptive ``get_intent_entity_score`` mock — one row per input pair."""

    async def _fn(refs, preds, language="english", model=None):
        return {
            "intent": float(intent),
            "entity": float(entity),
            "per_row": [
                {
                    "intent_score": intent,
                    "intent_explanation": "ok",
                    "entity_score": entity,
                    "entity_explanation": "ok",
                }
                for _ in refs
            ],
        }

    return AsyncMock(side_effect=_fn)


def _fake_llm_wer(llm_wer=0.05, llm_cer=0.03):
    """Adaptive ``get_llm_wer_cer_score`` mock — one row per input pair."""

    async def _fn(refs, preds, language="english", model=None):
        return {
            "llm_wer": float(llm_wer),
            "llm_cer": float(llm_cer),
            "per_row": [
                {
                    "llm_wer": float(llm_wer),
                    "llm_cer": float(llm_cer),
                    "segments": [],
                }
                for _ in refs
            ],
        }

    return AsyncMock(side_effect=_fn)


# --- format_metrics_summary ----------------------------------------------

class TestFormatMetricsSummary(unittest.TestCase):
    def test_includes_sarvam_when_present(self):
        from calibrate_agent.stt.eval import format_metrics_summary

        line = format_metrics_summary(
            {
                "wer": 0.1,
                "cer": 0.2,
                "sarvam_intent_score": 0.9,
                "sarvam_entity_score": 0.8,
                "sarvam_llm_wer": 0.05,
                "sarvam_llm_cer": 0.03,
                "semantic": {"type": "binary", "mean": 0.75},
            },
            prefix="deepgram: ",
        )
        self.assertEqual(
            line,
            "  deepgram: WER=0.1000, CER=0.2000, Sarvam Intent Score=0.9000, "
            "Sarvam Entity Score=0.8000, Sarvam LLM WER=0.0500, "
            "Sarvam LLM CER=0.0300, semantic=0.7500",
        )

    def test_omits_sarvam_when_absent(self):
        from calibrate_agent.stt.eval import format_metrics_summary

        line = format_metrics_summary(
            {"wer": 0.1, "cer": 0.2, "semantic": {"type": "binary", "mean": 0.75}}
        )
        self.assertEqual(line, "  WER=0.1000, CER=0.2000, semantic=0.7500")
        self.assertNotIn("Sarvam", line)


# --- ElevenLabs realtime SDK test doubles ---------------------------------

class _FakeRealtimeConnection:
    """Minimal stand-in for the elevenlabs SDK ``RealtimeConnection``.

    Emits ``session_started`` as soon as the handler is registered, and on
    ``commit()`` emits the configured committed transcripts followed by a
    ``close`` (or an error), mirroring a server that closes once all committed
    segments for the session have been delivered.
    """

    def __init__(self, committed_texts=None, error_on_commit=None):
        from elevenlabs import RealtimeEvents

        self._events = RealtimeEvents
        self._handlers = {}
        self._committed_texts = committed_texts or []
        self._error_on_commit = error_on_commit
        self.sent = []
        self.committed = False
        self.closed = False

    def on(self, event, callback):
        self._handlers.setdefault(event, []).append(callback)
        # The real server sends session_started right after connect.
        if event == self._events.SESSION_STARTED:
            callback({"message_type": "session_started"})

    def _emit(self, event, data):
        for cb in self._handlers.get(event, []):
            cb(data)

    async def send(self, data):
        self.sent.append(data)

    async def commit(self):
        self.committed = True
        if self._error_on_commit is not None:
            self._emit(self._events.ERROR, self._error_on_commit)
            return
        for text in self._committed_texts:
            self._emit(
                self._events.COMMITTED_TRANSCRIPT,
                {"message_type": "committed_transcript", "text": text},
            )
        # The real server closes the stream once all committed segments are
        # delivered; emit CLOSE so the collector finishes promptly rather than
        # waiting out the inter-segment idle gap.
        self._emit(self._events.CLOSE, None)

    async def close(self):
        self.closed = True


def _fake_elevenlabs_client(connection):
    """Build a MagicMock ElevenLabs client whose realtime.connect returns conn."""
    client = MagicMock()

    async def _connect(options):
        return connection

    client.speech_to_text.realtime.connect = _connect
    return client


# --- load_audio -----------------------------------------------------------

class TestLoadAudio(unittest.TestCase):
    def test_load_audio_bytes(self):
        from calibrate_agent.stt import eval as E

        fake_segment = MagicMock()
        fake_segment.set_channels.return_value = fake_segment
        fake_segment.set_frame_rate.return_value = fake_segment
        fake_segment.set_sample_width.return_value = fake_segment
        fake_segment.normalize.return_value = fake_segment
        fake_segment.strip_silence.return_value = fake_segment

        def fake_export(out_io, format):
            out_io.write(b"WAVDATA")

        fake_segment.export = fake_export

        with patch("pydub.AudioSegment.from_file", return_value=fake_segment):
            result = E.load_audio(Path("/tmp/dummy.wav"))
        self.assertEqual(result, b"WAVDATA")

    def test_load_audio_raw_pcm(self):
        from calibrate_agent.stt import eval as E

        fake_segment = MagicMock()
        fake_segment.set_channels.return_value = fake_segment
        fake_segment.set_frame_rate.return_value = fake_segment
        fake_segment.set_sample_width.return_value = fake_segment
        fake_segment.normalize.return_value = fake_segment
        fake_segment.strip_silence.return_value = fake_segment
        fake_segment.raw_data = b"PCMDATA"

        with patch("pydub.AudioSegment.from_file", return_value=fake_segment):
            result = E.load_audio(Path("/tmp/x.wav"), raw_pcm=True)
        self.assertEqual(result, b"PCMDATA")

    def test_load_audio_as_file(self):
        from calibrate_agent.stt import eval as E

        fake_segment = MagicMock()
        fake_segment.set_channels.return_value = fake_segment
        fake_segment.set_frame_rate.return_value = fake_segment
        fake_segment.set_sample_width.return_value = fake_segment
        fake_segment.normalize.return_value = fake_segment
        fake_segment.strip_silence.return_value = fake_segment

        def fake_export(out_io, format):
            out_io.write(b"WAVDATA")

        fake_segment.export = fake_export

        with patch("pydub.AudioSegment.from_file", return_value=fake_segment):
            result = E.load_audio(Path("/tmp/x.wav"), as_file=True)
        self.assertTrue(hasattr(result, "read"))


# --- Provider transcribe_* missing-key paths ------------------------------

class TestProviderAPIKeyMissing(unittest.IsolatedAsyncioTestCase):
    async def test_groq_missing_key(self):
        from calibrate_agent.stt.eval import transcribe_groq

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_groq(Path("/tmp/x.wav"), "english")

    async def test_google_missing_credentials(self):
        from calibrate_agent.stt.eval import transcribe_google

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_google(Path("/tmp/x.wav"), "english")

    async def test_sarvam_missing_key(self):
        from calibrate_agent.stt.eval import transcribe_sarvam

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_sarvam(Path("/tmp/x.wav"), "english")

    async def test_cartesia_missing_key(self):
        from calibrate_agent.stt.eval import transcribe_cartesia

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_cartesia(Path("/tmp/x.wav"), "english")

    async def test_smallest_streaming_missing_key(self):
        from calibrate_agent.stt.eval import transcribe_smallest_streaming

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_smallest_streaming(Path("/tmp/x.wav"), "english")

    async def test_elevenlabs_streaming_missing_key(self):
        from calibrate_agent.stt.eval import transcribe_elevenlabs_streaming

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_elevenlabs_streaming(
                    Path("/tmp/x.wav"), "english"
                )

    async def test_openai_streaming_missing_key(self):
        from calibrate_agent.stt.eval import transcribe_openai_streaming

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_openai_streaming(
                    Path("/tmp/x.wav"), "english"
                )

    async def test_deepgram_streaming_missing_key(self):
        from calibrate_agent.stt.eval import transcribe_deepgram_streaming

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await transcribe_deepgram_streaming(
                    Path("/tmp/x.wav"), "english"
                )

    async def test_elevenlabs_streaming_happy(self):
        import elevenlabs

        from calibrate_agent.stt import eval as E

        fake_conn = _FakeRealtimeConnection(committed_texts=["hello"])
        fake_client = _fake_elevenlabs_client(fake_conn)

        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}), patch.object(
            E, "load_audio", return_value=b"\x00\x00" * 1000
        ), patch.object(elevenlabs, "ElevenLabs", return_value=fake_client):
            result = await E.transcribe_elevenlabs_streaming(
                Path("/tmp/x.wav"), "english"
            )

        self.assertEqual(result["transcript"], "hello")
        # Audio chunks must have been sent and a commit issued before completion.
        self.assertTrue(fake_conn.sent)
        self.assertTrue(fake_conn.committed)
        self.assertTrue(fake_conn.closed)

    async def test_elevenlabs_streaming_accumulates_multiple_segments(self):
        # The server auto-segments long audio into MULTIPLE committed_transcript
        # messages (it auto-commits ~every 36s). All segments must be captured —
        # stopping at the first would silently truncate long recordings.
        import elevenlabs

        from calibrate_agent.stt import eval as E

        fake_conn = _FakeRealtimeConnection(
            committed_texts=["hello", "world", "again"]
        )
        fake_client = _fake_elevenlabs_client(fake_conn)

        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}), patch.object(
            E, "load_audio", return_value=b"\x00\x00" * 1000
        ), patch.object(elevenlabs, "ElevenLabs", return_value=fake_client):
            result = await E.transcribe_elevenlabs_streaming(
                Path("/tmp/x.wav"), "english"
            )

        self.assertEqual(result["transcript"], "hello world again")

    async def test_elevenlabs_streaming_finishes_on_idle_without_close(self):
        # If the server never sends CLOSE after the final segment, the collector
        # must still finish via the inter-segment idle gap — capturing every
        # segment, with no hang and no truncation.
        import elevenlabs

        from calibrate_agent.stt import eval as E

        class _NoCloseConn(_FakeRealtimeConnection):
            async def commit(self):
                self.committed = True
                for text in self._committed_texts:
                    self._emit(
                        self._events.COMMITTED_TRANSCRIPT,
                        {"message_type": "committed_transcript", "text": text},
                    )
                # Deliberately no CLOSE — completion relies on the idle gap.

        fake_conn = _NoCloseConn(committed_texts=["hello", "world"])
        fake_client = _fake_elevenlabs_client(fake_conn)

        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}), patch.object(
            E, "load_audio", return_value=b"\x00\x00" * 1000
        ), patch.object(elevenlabs, "ElevenLabs", return_value=fake_client), patch.object(
            E, "ELEVENLABS_SEGMENT_IDLE_SECONDS", 0.05
        ):
            result = await E.transcribe_elevenlabs_streaming(
                Path("/tmp/x.wav"), "english"
            )

        self.assertEqual(result["transcript"], "hello world")

    async def test_elevenlabs_streaming_short_clip_insufficient_audio(self):
        # A sub-second clip can close with ``insufficient_audio_activity`` and
        # no committed transcript — we must return an empty transcript rather
        # than raising (which would trigger the router's @backoff retries).
        import elevenlabs

        from calibrate_agent.stt import eval as E

        fake_conn = _FakeRealtimeConnection(
            error_on_commit={
                "message_type": "insufficient_audio_activity",
                "error": "insufficient_audio_activity",
            }
        )
        fake_client = _fake_elevenlabs_client(fake_conn)

        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}), patch.object(
            E, "load_audio", return_value=b"\x00\x00" * 100
        ), patch.object(elevenlabs, "ElevenLabs", return_value=fake_client):
            result = await E.transcribe_elevenlabs_streaming(
                Path("/tmp/x.wav"), "english"
            )

        self.assertEqual(result["transcript"], "")

    async def test_elevenlabs_streaming_fatal_error_raises(self):
        import elevenlabs

        from calibrate_agent.stt import eval as E

        fake_conn = _FakeRealtimeConnection(
            error_on_commit={"message_type": "input_error", "error": "bad input"}
        )
        fake_client = _fake_elevenlabs_client(fake_conn)

        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}), patch.object(
            E, "load_audio", return_value=b"\x00\x00" * 1000
        ), patch.object(elevenlabs, "ElevenLabs", return_value=fake_client):
            with self.assertRaises(RuntimeError):
                await E.transcribe_elevenlabs_streaming(
                    Path("/tmp/x.wav"), "english"
                )

    async def test_openai_streaming_happy(self):
        from types import SimpleNamespace

        from calibrate_agent.stt import eval as E

        events = [
            SimpleNamespace(type="transcript.text.delta", delta="hello "),
            SimpleNamespace(type="transcript.text.done", text="hello world"),
        ]

        async def fake_stream():
            for ev in events:
                yield ev

        fake_client = MagicMock()
        fake_client.audio.transcriptions.create = AsyncMock(
            return_value=fake_stream()
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), patch.object(
            E, "load_audio", return_value=b"\x00" * 100
        ), patch.object(E, "AsyncOpenAI", return_value=fake_client):
            result = await E.transcribe_openai_streaming(
                Path("/tmp/x.wav"), "english"
            )

        self.assertEqual(result["transcript"], "hello world")

    async def test_deepgram_streaming_happy(self):
        import json as _json

        from calibrate_agent.stt import eval as E

        # Fake WS that emits a final Results then a Metadata to signal end.
        class FakeWS:
            def __init__(self):
                self.sent = []
                self._messages = [
                    _json.dumps(
                        {
                            "type": "Results",
                            "is_final": True,
                            "channel": {
                                "alternatives": [{"transcript": "hello world"}]
                            },
                        }
                    ),
                    _json.dumps({"type": "Metadata"}),
                ]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def send(self, payload):
                self.sent.append(payload)

            def __aiter__(self):
                msgs = self._messages

                async def gen():
                    for m in msgs:
                        yield m

                return gen()

        fake_ws = FakeWS()

        def _fake_connect(*args, **kwargs):
            return fake_ws

        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "k"}), patch.object(
            E, "load_audio", return_value=b"\x00" * 1000
        ), patch(
            "websockets.asyncio.client.connect", side_effect=_fake_connect
        ):
            result = await E.transcribe_deepgram_streaming(
                Path("/tmp/x.wav"), "english"
            )

        self.assertEqual(result["transcript"], "hello world")

    async def test_smallest_streaming_happy_sends_close_stream(self):
        import json as _json

        from calibrate_agent.stt import eval as E

        # Two final segments then is_last; the transcript must accumulate both
        # and a "close_stream" frame (per Pulse STT docs) must be sent.
        class FakeWS:
            def __init__(self):
                self.sent = []
                self._messages = [
                    _json.dumps({"transcript": "hello", "is_final": True}),
                    _json.dumps(
                        {"transcript": "world", "is_final": True, "is_last": True}
                    ),
                ]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def send(self, payload):
                self.sent.append(payload)

            def __aiter__(self):
                msgs = self._messages

                async def gen():
                    for m in msgs:
                        yield m

                return gen()

        fake_ws = FakeWS()

        with patch.dict(os.environ, {"SMALLEST_API_KEY": "k"}), patch.object(
            E, "load_audio", return_value=b"\x00" * 1000
        ), patch(
            "websockets.asyncio.client.connect", side_effect=lambda *a, **k: fake_ws
        ):
            result = await E.transcribe_smallest_streaming(
                Path("/tmp/x.wav"), "english"
            )

        self.assertEqual(result["transcript"], "hello world")
        text_frames = [s for s in fake_ws.sent if isinstance(s, str)]
        self.assertTrue(any('"close_stream"' in s for s in text_frames))
        self.assertFalse(any("finalize" in s for s in text_frames))

    async def test_cartesia_streaming_happy_sends_close(self):
        from calibrate_agent.stt import eval as E

        sent = []

        class FakeWS:
            async def send(self, payload):
                sent.append(payload)

            async def receive(self):
                yield {"type": "transcript", "text": "hello", "is_final": True}
                yield {"type": "transcript", "text": "world", "is_final": True}
                yield {"type": "done"}

            async def close(self):
                pass

        fake_ws = FakeWS()
        fake_client = MagicMock()
        fake_client.stt.websocket = AsyncMock(return_value=fake_ws)
        fake_client.close = AsyncMock()

        with patch.dict(os.environ, {"CARTESIA_API_KEY": "k"}), patch.object(
            E, "load_audio", return_value=b"\x00" * 1000
        ), patch.object(E, "AsyncCartesia", return_value=fake_client):
            result = await E.transcribe_cartesia(Path("/tmp/x.wav"), "english")

        self.assertEqual(result["transcript"], "hello world")
        # The documented client close command is "close", not "done".
        self.assertIn("close", sent)
        self.assertNotIn("done", sent)


# --- transcribe_audio router ----------------------------------------------

class TestTranscribeAudioRouter(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_provider_raises(self):
        from calibrate_agent.stt.eval import transcribe_audio

        # Use __wrapped__ to skip backoff retries
        inner = transcribe_audio
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        with self.assertRaises(ValueError):
            await inner(Path("/tmp/x.wav"), "ref", "bogus", "english", "u")

    async def test_routes_to_provider(self):
        from calibrate_agent.stt import eval as E

        fake_fn = AsyncMock(return_value={"transcript": "hello world"})
        inner = E.transcribe_audio
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        with patch.object(E, "transcribe_deepgram_streaming", fake_fn):
            result = await inner(Path("/tmp/x.wav"), "ref", "deepgram", "english", "u")
        self.assertEqual(result["transcript"], "hello world")
        fake_fn.assert_called_once()

    async def test_with_langfuse(self):
        from calibrate_agent.stt import eval as E

        fake_fn = AsyncMock(return_value={"transcript": "x"})
        inner = E.transcribe_audio
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        fake_lf = MagicMock()
        with patch.object(E, "transcribe_deepgram_streaming", fake_fn), \
             patch.object(E, "langfuse_enabled", True), \
             patch.object(E, "langfuse", fake_lf), \
             patch.object(E, "create_langfuse_audio_media", return_value=None):
            await inner(Path("/tmp/x.wav"), "ref", "deepgram", "english", "u")
        fake_lf.update_current_trace.assert_called_once()


# --- validate_existing_results_csv ----------------------------------------

class TestValidateExistingResultsCsv(unittest.TestCase):
    def test_nonexistent_returns_ok(self):
        from calibrate_agent.stt.eval import validate_existing_results_csv

        ok, _ = validate_existing_results_csv("/nonexistent/path.csv")
        self.assertTrue(ok)

    def test_empty_is_valid(self):
        from calibrate_agent.stt.eval import validate_existing_results_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "results.csv"
            pd.DataFrame(columns=["id", "gt", "pred"]).to_csv(p, index=False)
            ok, _ = validate_existing_results_csv(str(p))
            self.assertTrue(ok)

    def test_invalid_columns(self):
        from calibrate_agent.stt.eval import validate_existing_results_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "results.csv"
            pd.DataFrame({"foo": [1]}).to_csv(p, index=False)
            ok, err = validate_existing_results_csv(str(p))
            self.assertFalse(ok)
            self.assertIn("Missing columns", err)

    def test_valid_columns(self):
        from calibrate_agent.stt.eval import validate_existing_results_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "results.csv"
            pd.DataFrame({"id": [1], "gt": ["a"], "pred": ["a"]}).to_csv(p, index=False)
            ok, _ = validate_existing_results_csv(str(p))
            self.assertTrue(ok)


# --- validate_stt_eval_only_dataset --------------------------------------

class TestValidateSTTEvalOnlyDataset(unittest.TestCase):
    def test_nonexistent(self):
        from calibrate_agent.stt.eval import validate_stt_eval_only_dataset

        ok, err, _ = validate_stt_eval_only_dataset("/nonexistent.json")
        self.assertFalse(ok)

    def test_invalid_json(self):
        from calibrate_agent.stt.eval import validate_stt_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            p.write_text("{bad")
            ok, err, _ = validate_stt_eval_only_dataset(str(p))
            self.assertFalse(ok)

    def test_not_a_list(self):
        from calibrate_agent.stt.eval import validate_stt_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            p.write_text(json.dumps({"x": 1}))
            ok, _, _ = validate_stt_eval_only_dataset(str(p))
            self.assertFalse(ok)

    def test_row_not_object(self):
        from calibrate_agent.stt.eval import validate_stt_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            p.write_text(json.dumps(["x"]))
            ok, _, _ = validate_stt_eval_only_dataset(str(p))
            self.assertFalse(ok)

    def test_missing_fields(self):
        from calibrate_agent.stt.eval import validate_stt_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            p.write_text(json.dumps([{"id": "a"}]))
            ok, _, _ = validate_stt_eval_only_dataset(str(p))
            self.assertFalse(ok)

    def test_valid(self):
        from calibrate_agent.stt.eval import validate_stt_eval_only_dataset

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            p.write_text(json.dumps([{"id": "a", "gt": "x", "pred": "x"}]))
            ok, err, rows = validate_stt_eval_only_dataset(str(p))
            self.assertTrue(ok)
            self.assertEqual(len(rows), 1)


# --- _score_and_write_results --------------------------------------------

class TestScoreAndWrite(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from calibrate_agent.stt import eval as E

        async def _sem(references, predictions, model=None):
            return {
                "semantic_wer": 0.0,
                "per_row": [
                    {
                        "semantic_wer": 0.0, "substitutions": 0, "deletions": 0,
                        "insertions": 0, "reference_words": 1,
                        "normalized_reference": "", "normalized_hypothesis": "",
                        "reasoning": "",
                    }
                    for _ in references
                ],
            }

        p = patch.object(E, "get_semantic_wer_score", AsyncMock(side_effect=_sem))
        p.start()
        self.addCleanup(p.stop)

    async def test_writes_files(self):
        from calibrate_agent.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(E, "get_wer_score", return_value={"score": 0.1, "per_row": [0.1, 0.1]}), \
                 patch.object(E, "get_cer_score", return_value={"score": 0.2, "per_row": [0.2, 0.2]}), \
                 patch.object(E, "get_intent_entity_score", _fake_intent_entity()), \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                     "per_row": [
                         {"semantic_match": {"match": True, "reasoning": "ok"}},
                         {"semantic_match": {"match": True, "reasoning": "ok"}},
                     ],
                 })):
                result = await E._score_and_write_results(
                    ids=["a", "b"],
                    gt_transcripts=["x", "y"],
                    pred_transcripts=["x", "y"],
                    output_dir=tmp,
                    evaluator_config_dir=tmp,
                    judge_evaluators=[
                        {
                            "name": "semantic_match",
                            "system_prompt": "match",
                            "judge_model": "openai/gpt-4.1",
                        }
                    ],
                    run_llm_judges=False,
                )
            self.assertEqual(result["wer"], 0.1)
            self.assertEqual(result["cer"], 0.2)
            self.assertIn("semantic_match", result)
            self.assertTrue((Path(tmp) / "metrics.json").exists())

            import pandas as _pd
            df = _pd.read_csv(Path(tmp) / "results.csv")
            self.assertIn("cer", df.columns)
            self.assertEqual(list(df["cer"]), [0.2, 0.2])

    async def test_sarvam_judges_emit_llm_wer_columns(self):
        from calibrate_agent.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(E, "get_wer_score", return_value={"score": 0.1, "per_row": [0.1, 0.1]}), \
                 patch.object(E, "get_cer_score", return_value={"score": 0.2, "per_row": [0.2, 0.2]}), \
                 patch.object(E, "get_intent_entity_score", _fake_intent_entity()), \
                 patch.object(E, "get_llm_wer_cer_score", _fake_llm_wer(0.05, 0.03)), \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                     "per_row": [
                         {"semantic_match": {"match": True, "reasoning": "ok"}},
                         {"semantic_match": {"match": True, "reasoning": "ok"}},
                     ],
                 })):
                result = await E._score_and_write_results(
                    ids=["a", "b"],
                    gt_transcripts=["x", "y"],
                    pred_transcripts=["x", "y"],
                    output_dir=tmp,
                    evaluator_config_dir=tmp,
                    run_llm_judges=True,
                )
            self.assertEqual(result["sarvam_llm_wer"], 0.05)
            self.assertEqual(result["sarvam_llm_cer"], 0.03)

            import pandas as _pd
            df = _pd.read_csv(Path(tmp) / "results.csv")
            self.assertIn("sarvam_llm_wer", df.columns)
            self.assertIn("sarvam_llm_cer", df.columns)
            self.assertIn("sarvam_llm_wer_reasoning", df.columns)

    async def test_sarvam_llm_wer_absent_when_disabled(self):
        from calibrate_agent.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(E, "get_wer_score", return_value={"score": 0.1, "per_row": [0.1]}), \
                 patch.object(E, "get_cer_score", return_value={"score": 0.2, "per_row": [0.2]}), \
                 patch.object(E, "get_llm_wer_cer_score", _fake_llm_wer()) as llm_wer_mock, \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                     "per_row": [{"semantic_match": {"match": True, "reasoning": "ok"}}],
                 })):
                result = await E._score_and_write_results(
                    ids=["a"],
                    gt_transcripts=["x"],
                    pred_transcripts=["x"],
                    output_dir=tmp,
                    evaluator_config_dir=tmp,
                    run_llm_judges=False,
                )
            llm_wer_mock.assert_not_called()
            self.assertNotIn("sarvam_llm_wer", result)

            import pandas as _pd
            df = _pd.read_csv(Path(tmp) / "results.csv")
            self.assertNotIn("sarvam_llm_wer", df.columns)

    async def test_rating_evaluator(self):
        from calibrate_agent.stt import eval as E

        rating_ev = {"name": "r", "system_prompt": "x", "judge_model": "m",
                     "type": "rating", "scale_min": 1, "scale_max": 5}

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(E, "get_wer_score", return_value={"score": 0.05, "per_row": [0.05]}), \
             patch.object(E, "get_cer_score", return_value={"score": 0.03, "per_row": [0.03]}), \
             patch.object(E, "get_intent_entity_score", _fake_intent_entity()), \
             patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                 "scores": {"r": {"type": "rating", "mean": 4.0, "scale_min": 1, "scale_max": 5}},
                 "per_row": [{"r": {"score": 4, "reasoning": "ok"}}],
             })):
            await E._score_and_write_results(
                ids=["a"],
                gt_transcripts=["x"],
                pred_transcripts=["x"],
                output_dir=tmp,
                evaluator_config_dir=tmp,
                judge_evaluators=[rating_ev],
                run_llm_judges=False,
            )

    async def test_short_row_extras_do_not_truncate_results(self):
        from calibrate_agent.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(E, "get_wer_score", return_value={"score": 0.1, "per_row": [0.1, 0.2]}), \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                     "per_row": [
                         {"semantic_match": {"match": True, "reasoning": "ok"}},
                         {"semantic_match": {"match": False, "reasoning": "no"}},
                     ],
                 })):
                await E._score_and_write_results(
                    ids=["a", "b"],
                    gt_transcripts=["x", "y"],
                    pred_transcripts=["x", "z"],
                    output_dir=tmp,
                    evaluator_config_dir=tmp,
                    audio_durations=[1.0],
                )

            df = pd.read_csv(Path(tmp) / "results.csv")
            self.assertEqual(len(df), 2)
            self.assertEqual(df.iloc[0]["audio_duration_seconds"], 1.0)


class TestSTTCostMetrics(unittest.TestCase):
    def test_builds_cost_metrics_from_default_pricing(self):
        from calibrate_agent.stt import eval as E

        metrics = E._build_stt_cost_metrics(
            provider="openai",
            audio_duration_seconds=[60.0],
            model="gpt-4o-transcribe",
        )

        self.assertEqual(metrics["billing_unit"], "minute")
        self.assertEqual(metrics["pricing_source"], "calibrate_default")
        self.assertEqual(metrics["pricing_model"], "gpt-4o-transcribe")
        self.assertEqual(metrics["total_seconds"], 60.0)
        self.assertEqual(metrics["audio_minutes"], 1.0)
        self.assertEqual(metrics["cost_per_minute_usd"], 0.006)
        self.assertEqual(metrics["cost_usd"], 0.006)

    def test_builds_cost_metrics_from_multiple_durations(self):
        from calibrate_agent.stt import eval as E

        metrics = E._build_stt_cost_metrics(
            provider="openai",
            audio_duration_seconds=[30.0, 90.0, None],
            model="gpt-4o-transcribe",
        )

        self.assertEqual(metrics["billing_unit"], "minute")
        self.assertEqual(metrics["pricing_source"], "calibrate_default")
        self.assertEqual(metrics["total_seconds"], 120.0)
        self.assertEqual(metrics["audio_minutes"], 2.0)
        self.assertEqual(metrics["cost_per_minute_usd"], 0.006)
        self.assertEqual(metrics["cost_usd"], 0.012)

    def test_rounds_audio_minutes_for_cost_metrics(self):
        from calibrate_agent.stt import eval as E

        metrics = E._build_stt_cost_metrics(
            provider="openai",
            audio_duration_seconds=[100.0],
            model="gpt-4o-transcribe",
        )

        self.assertEqual(metrics["audio_minutes"], 1.6667)

    def test_uses_google_sindhi_pricing_model(self):
        from calibrate_agent.stt import eval as E

        metrics = E._build_stt_cost_metrics(
            provider="google",
            audio_duration_seconds=[60.0],
            model=E._default_stt_model("google", "sindhi"),
        )

        self.assertEqual(metrics["pricing_model"], "chirp_2")
        self.assertEqual(metrics["pricing_source"], "calibrate_default")
        self.assertEqual(metrics["cost_usd"], 0.016)

    def test_returns_none_when_no_pricing(self):
        from calibrate_agent.stt import eval as E

        metrics = E._build_stt_cost_metrics(
            provider="gemini",
            audio_duration_seconds=[60.0],
            model=E._default_stt_model("gemini", "english"),
        )

        self.assertIsNone(metrics)

    def test_all_supported_providers_with_default_pricing(self):
        from calibrate_agent.stt import eval as E
        from calibrate_agent.stt.eval import STT_PROVIDERS

        providers_without_per_minute_pricing = {"gemini"}
        for provider in STT_PROVIDERS:
            if provider in providers_without_per_minute_pricing:
                continue
            with self.subTest(provider=provider):
                metrics = E._build_stt_cost_metrics(
                    provider=provider,
                    audio_duration_seconds=[60.0],
                    model=E._default_stt_model(provider, "english"),
                )
                self.assertEqual(metrics["pricing_source"], "calibrate_default")
                self.assertIn("cost_usd", metrics)


# --- run_eval_only --------------------------------------------------------

class TestRunEvalOnly(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_dataset(self):
        from calibrate_agent.stt.eval import run_eval_only

        with tempfile.TemporaryDirectory() as tmp:
            result = await run_eval_only("/nonexistent.json", tmp)
        self.assertEqual(result["status"], "error")

    async def test_success(self):
        from calibrate_agent.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "data.json"
            ds.write_text(json.dumps([
                {"id": "a", "gt": "x", "pred": "x"},
                {"id": "b", "gt": "y", "pred": None},
            ]))
            out = Path(tmp) / "out"
            with patch.object(E, "get_wer_score", return_value={"score": 0.1, "per_row": [0.1, 0.1]}), \
                 patch.object(E, "get_cer_score", return_value={"score": 0.2, "per_row": [0.2, 0.2]}), \
                 patch.object(E, "get_intent_entity_score", _fake_intent_entity()), \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                     "per_row": [
                         {"semantic_match": {"match": True, "reasoning": "ok"}},
                         {"semantic_match": {"match": True, "reasoning": "ok"}},
                     ],
                 })):
                result = await E.run_eval_only(
                    str(ds), str(out), run_llm_judges=False
                )
        self.assertEqual(result["status"], "completed")


# --- run_stt_eval ---------------------------------------------------------

class TestRunStteval(unittest.IsolatedAsyncioTestCase):
    async def test_processes_new_and_skips_existing(self):
        from calibrate_agent.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "stuff"
            base.mkdir()
            audio_dir = base / "audios"
            audio_dir.mkdir()
            (audio_dir / "a.wav").write_bytes(b"\x00")
            (audio_dir / "b.wav").write_bytes(b"\x00")

            results_csv = base / "results.csv"
            pd.DataFrame([{"id": "a", "gt": "X", "pred": "x"}]).to_csv(
                str(results_csv), index=False
            )

            with patch.object(
                E,
                "transcribe_audio",
                AsyncMock(return_value={"transcript": "hello b"}),
            ):
                count = await E.run_stt_eval(
                    gt_data=[{"id": "a", "gt": "X"}, {"id": "b", "gt": "Y"}],
                    audio_dir=audio_dir,
                    provider="deepgram",
                    language="english",
                    results_csv_path=str(results_csv),
                )

            self.assertEqual(count, 1)
            df = pd.read_csv(str(results_csv))
            self.assertEqual(len(df), 2)
            new_row = df[df["id"] == "b"].iloc[0]
            self.assertEqual(new_row["pred"], "hello b")


# --- run_single_provider_eval --------------------------------------------

class TestRunSingleProviderEval(unittest.IsolatedAsyncioTestCase):
    async def test_overwrite_path(self):
        from calibrate_agent.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            (base / "audios" / "a.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a"], "text": ["hello"]}).to_csv(base / "stt.csv", index=False)

            output = Path(tmp) / "out"
            output.mkdir()
            (output / "deepgram").mkdir()
            # Pre-existing results.csv to trigger overwrite path
            (output / "deepgram" / "results.csv").write_text("id,gt,pred\na,hello,hi\n")

            with patch.object(E, "transcribe_audio", AsyncMock(return_value={"transcript": "hello"})), \
                 patch.object(E, "get_wer_score", return_value={"score": 0.0, "per_row": [0.0]}), \
                 patch.object(E, "get_cer_score", return_value={"score": 0.0, "per_row": [0.0]}), \
                 patch.object(E, "get_intent_entity_score", _fake_intent_entity()), \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                     "per_row": [{"semantic_match": {"match": True, "reasoning": "ok"}}],
                 })):
                result = await E.run_single_provider_eval(
                    provider="deepgram",
                    language="english",
                    input_dir=str(base),
                    input_file_name="stt.csv",
                    output_dir=str(output),
                    debug=False,
                    debug_count=5,
                    ignore_retry=False,
                    overwrite=True,
                    run_llm_judges=False,
                )
            self.assertEqual(result["status"], "completed")

    async def test_existing_invalid_csv_error(self):
        from calibrate_agent.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            (base / "audios" / "a.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a"], "text": ["hello"]}).to_csv(base / "stt.csv", index=False)

            output = Path(tmp) / "out"
            output.mkdir()
            (output / "deepgram").mkdir()
            (output / "deepgram" / "results.csv").write_text("bad,csv\n1,2\n")

            result = await E.run_single_provider_eval(
                provider="deepgram",
                language="english",
                input_dir=str(base),
                input_file_name="stt.csv",
                output_dir=str(output),
                debug=False,
                debug_count=5,
                ignore_retry=False,
                overwrite=False,
            )
            self.assertEqual(result["status"], "error")

    async def test_debug_mode_and_ignore_retry(self):
        from calibrate_agent.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            for i in ["a", "b"]:
                (base / "audios" / f"{i}.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a", "b"], "text": ["hello", "world"]}).to_csv(
                base / "stt.csv", index=False
            )
            output = Path(tmp) / "out"
            output.mkdir()

            with patch.object(E, "transcribe_audio", AsyncMock(return_value={"transcript": "hello"})), \
                 patch.object(E, "get_wer_score", return_value={"score": 0.0, "per_row": [0.0]}), \
                 patch.object(E, "get_cer_score", return_value={"score": 0.0, "per_row": [0.0]}), \
                 patch.object(E, "get_intent_entity_score", _fake_intent_entity()), \
                 patch.object(E, "get_llm_judge_score", AsyncMock(return_value={
                     "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
                     "per_row": [{"semantic_match": {"match": True, "reasoning": "ok"}}],
                 })):
                result = await E.run_single_provider_eval(
                    provider="deepgram",
                    language="english",
                    input_dir=str(base),
                    input_file_name="stt.csv",
                    output_dir=str(output),
                    debug=True,
                    debug_count=1,
                    ignore_retry=True,
                    overwrite=False,
                    run_llm_judges=False,
                )
            self.assertEqual(result["status"], "completed")


# --- main CLI -------------------------------------------------------------

class TestSTTMain(unittest.IsolatedAsyncioTestCase):
    async def test_main_invalid_provider(self):
        from calibrate_agent.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["e.py", "-p", "bogus", "-i", tmp, "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await E.main()

    async def test_main_invalid_input_dir(self):
        from calibrate_agent.stt import eval as E

        argv = ["e.py", "-p", "deepgram", "-i", "/nonexistent", "-o", "/tmp/x"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                await E.main()

    async def test_main_success(self):
        from calibrate_agent.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            (base / "audios" / "a.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(base / "stt.csv", index=False)
            output = Path(tmp) / "out"

            argv = ["e.py", "-p", "deepgram", "-i", str(base), "-o", str(output)]
            fake_result = {"provider": "deepgram", "status": "completed",
                           "metrics": {"wer": 0.1, "semantic_match": {"type": "binary", "mean": 0.9}}}
            with patch.object(sys, "argv", argv), \
                 patch.object(E, "run_single_provider_eval", AsyncMock(return_value=fake_result)):
                await E.main()

    async def test_main_error_status(self):
        from calibrate_agent.stt import eval as E

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "audios").mkdir()
            (base / "audios" / "a.wav").write_bytes(b"\x00")
            pd.DataFrame({"id": ["a"], "text": ["hi"]}).to_csv(base / "stt.csv", index=False)
            output = Path(tmp) / "out"

            argv = ["e.py", "-p", "deepgram", "-i", str(base), "-o", str(output)]
            fake_result = {"provider": "deepgram", "status": "error", "error": "fail"}
            with patch.object(sys, "argv", argv), \
                 patch.object(E, "run_single_provider_eval", AsyncMock(return_value=fake_result)):
                with self.assertRaises(SystemExit):
                    await E.main()


if __name__ == "__main__":
    unittest.main()
