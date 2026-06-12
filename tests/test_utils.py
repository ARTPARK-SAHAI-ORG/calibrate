"""Tests for calibrate/utils.py — language helpers, schema builders, webhook calls, factories."""

import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


class TestLanguageCodes(unittest.TestCase):
    def test_get_stt_language_codes_per_provider(self):
        from calibrate.utils import get_stt_language_code

        # Sample a few providers + languages
        self.assertEqual(get_stt_language_code("hindi", "sarvam"), "hi-IN")
        self.assertEqual(get_stt_language_code("english", "google"), "en-US")
        self.assertEqual(get_stt_language_code("english", "smallest"), "en")
        self.assertEqual(get_stt_language_code("english", "cartesia"), "en")
        self.assertEqual(get_stt_language_code("english", "elevenlabs"), "eng")
        self.assertEqual(get_stt_language_code("english", "openai"), "en")
        self.assertEqual(get_stt_language_code("english", "groq"), "en")
        self.assertEqual(get_stt_language_code("bengali", "groq"), "bn")
        self.assertEqual(get_stt_language_code("mandarin", "groq"), "zh")
        self.assertEqual(get_stt_language_code("myanmar", "groq"), "my")
        self.assertEqual(get_stt_language_code("burmese", "groq"), "my")
        self.assertEqual(get_stt_language_code("haitian creole", "groq"), "ht")
        self.assertEqual(get_stt_language_code("bengali", "openai"), "en")
        self.assertEqual(get_stt_language_code("english", "deepgram"), "en")
        # Unknown provider falls through to default
        self.assertEqual(get_stt_language_code("english", "unknown"), "en")

    def test_get_tts_language_codes_per_provider(self):
        from calibrate.utils import get_tts_language_code

        self.assertEqual(get_tts_language_code("hindi", "sarvam"), "hi-IN")
        self.assertEqual(get_tts_language_code("english", "google"), "en-US")
        self.assertEqual(get_tts_language_code("english", "cartesia"), "en")
        self.assertEqual(get_tts_language_code("english", "elevenlabs"), "eng")
        self.assertEqual(get_tts_language_code("english", "openai"), "en")
        self.assertEqual(get_tts_language_code("english", "groq"), "en")
        self.assertEqual(get_tts_language_code("english", "smallest"), "en")
        self.assertEqual(get_tts_language_code("english", "unknown"), "en")

    def test_legacy_get_language_code(self):
        from calibrate.utils import get_language_code

        self.assertEqual(get_language_code("hindi", "sarvam"), "hi-IN")

    def test_validate_stt_language_unknown_provider(self):
        from calibrate.utils import validate_stt_language

        with self.assertRaises(ValueError):
            validate_stt_language("english", "bogus_provider")

    def test_validate_stt_language_unsupported(self):
        from calibrate.utils import validate_stt_language

        with self.assertRaises(ValueError):
            validate_stt_language("klingon", "google")

    def test_validate_stt_language_supported(self):
        from calibrate.utils import validate_stt_language

        validate_stt_language("english", "google")

    def test_validate_tts_language_unknown_provider(self):
        from calibrate.utils import validate_tts_language

        with self.assertRaises(ValueError):
            validate_tts_language("english", "bogus_provider")

    def test_validate_tts_language_unsupported(self):
        from calibrate.utils import validate_tts_language

        with self.assertRaises(ValueError):
            validate_tts_language("klingon", "google")

    def test_validate_tts_language_supported(self):
        from calibrate.utils import validate_tts_language

        validate_tts_language("english", "google")

    def test_sarvam_maithili_stt_only(self):
        from calibrate.utils import (
            get_stt_language_code,
            get_tts_language_code,
            validate_stt_language,
            validate_tts_language,
        )

        self.assertEqual(get_stt_language_code("maithili", "sarvam"), "mai-IN")
        validate_stt_language("maithili", "sarvam")

        # Sarvam TTS does not support Maithili
        with self.assertRaises(ValueError):
            validate_tts_language("maithili", "sarvam")
        # Unsupported TTS lookup falls back to default
        self.assertEqual(get_tts_language_code("maithili", "sarvam"), "en-IN")


class TestGetSTTLanguageEnum(unittest.TestCase):
    def test_sarvam_kannada(self):
        from calibrate.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language("kannada", "sarvam"), Language.KN_IN)

    def test_sarvam_hindi(self):
        from calibrate.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language("hindi", "sarvam"), Language.HI_IN)

    def test_sarvam_english(self):
        from calibrate.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language("english", "sarvam"), Language.EN_IN)

    def test_default_kannada(self):
        from calibrate.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language("kannada", "deepgram"), Language.KN)

    def test_default_hindi(self):
        from calibrate.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language("hindi", "deepgram"), Language.HI)

    def test_default_english(self):
        from calibrate.utils import get_stt_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_stt_language("english", "deepgram"), Language.EN)


class TestGetTTSLanguageEnum(unittest.TestCase):
    def test_sarvam_kannada(self):
        from calibrate.utils import get_tts_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_tts_language("kannada", "sarvam"), Language.KN_IN)

    def test_sarvam_hindi(self):
        from calibrate.utils import get_tts_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_tts_language("hindi", "sarvam"), Language.HI_IN)

    def test_sarvam_english(self):
        from calibrate.utils import get_tts_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_tts_language("english", "sarvam"), Language.EN_IN)

    def test_default_kannada(self):
        from calibrate.utils import get_tts_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_tts_language("kannada", "cartesia"), Language.KN)

    def test_default_hindi(self):
        from calibrate.utils import get_tts_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_tts_language("hindi", "cartesia"), Language.HI)

    def test_default_english(self):
        from calibrate.utils import get_tts_language
        from pipecat.transcriptions.language import Language

        self.assertEqual(get_tts_language("english", "cartesia"), Language.EN)


class TestLoggerHelpers(unittest.TestCase):
    def test_configure_print_logger(self):
        from calibrate.utils import configure_print_logger, cleanup_print_logger

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "test.log"
            logger = configure_print_logger(str(log_path), simulation_name="test_sim_1")
            logger.info("hello world")
            self.assertTrue(log_path.exists())
            cleanup_print_logger("test_sim_1")

    def test_configure_print_logger_default(self):
        from calibrate.utils import configure_print_logger

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "test.log"
            configure_print_logger(str(log_path))

    def test_cleanup_nonexistent(self):
        from calibrate.utils import cleanup_print_logger

        cleanup_print_logger("nonexistent_simulation")

    def test_log_and_print_with_simulation_name(self):
        from calibrate.utils import (
            configure_print_logger,
            cleanup_print_logger,
            log_and_print,
        )

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "test.log"
            configure_print_logger(str(log_path), simulation_name="sim_xyz")
            log_and_print("hello", simulation_name="sim_xyz")
            cleanup_print_logger("sim_xyz")

    def test_log_and_print_no_simulation(self):
        from calibrate.utils import log_and_print

        log_and_print("simple message")


class TestProviderLog(unittest.TestCase):
    def test_provider_log_with_file(self):
        from calibrate.utils import provider_log, provider_log_file

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "log.log"
            token = provider_log_file.set(str(log_path))
            try:
                provider_log("hello world")
                self.assertTrue(log_path.exists())
                self.assertIn("hello world", log_path.read_text())
            finally:
                provider_log_file.reset(token)

    def test_provider_log_no_terminal(self):
        from calibrate.utils import provider_log, provider_log_file

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "log.log"
            token = provider_log_file.set(str(log_path))
            try:
                provider_log("quiet", to_terminal=False)
            finally:
                provider_log_file.reset(token)

    def test_provider_log_no_file(self):
        from calibrate.utils import provider_log

        provider_log("no file set")


class TestLogJudgeIO(unittest.TestCase):
    def test_writes_block_to_bound_file_without_terminal(self):
        from calibrate.utils import log_judge_io, provider_log_file

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "logs"
            token = provider_log_file.set(str(log_path))
            captured = io.StringIO()
            try:
                with patch("sys.stdout", captured):
                    log_judge_io(
                        evaluator="correctness",
                        model="openai/gpt-x",
                        system_prompt="SYS",
                        user_input="INPUT",
                        output={"match": True, "reasoning": "ok"},
                    )
            finally:
                provider_log_file.reset(token)

            text = log_path.read_text()
            self.assertIn("judge call", text)
            self.assertIn("correctness", text)
            self.assertIn("openai/gpt-x", text)
            self.assertIn("SYS", text)
            self.assertIn("INPUT", text)
            self.assertIn("reasoning", text)
            # Never echoed to the terminal
            self.assertEqual(captured.getvalue(), "")

    def test_no_op_when_unbound(self):
        from calibrate.utils import log_judge_io, provider_log_file

        self.assertIsNone(provider_log_file.get())
        # Must not raise when no run log file is bound.
        log_judge_io(
            evaluator="x", model="m", system_prompt="s",
            user_input="i", output="o",
        )

    def test_concurrent_writes_are_not_interleaved(self):
        """Each judge entry stays intact when many write to one file at once."""
        import threading
        from calibrate.utils import log_judge_io, provider_log_file

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "logs"

            def write(i):
                # Each thread binds the log file in its own context (worker
                # threads don't inherit the parent's context vars), then writes
                # a large-ish payload so a non-atomic write would split.
                tok = provider_log_file.set(str(log_path))
                try:
                    log_judge_io(
                        evaluator=f"ev{i}",
                        model="m",
                        system_prompt="S" * 5000,
                        user_input="I" * 5000,
                        output=f"OUT{i}",
                    )
                finally:
                    provider_log_file.reset(tok)

            threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            text = log_path.read_text()
            # Exactly one complete block per writer, none torn apart.
            self.assertEqual(text.count("──── judge call ────"), 20)
            self.assertEqual(text.count("────────────────────"), 20)
            # Every writer's evaluator name and output appear, intact.
            for i in range(20):
                self.assertIn(f"evaluator: ev{i}", text)
                self.assertIn(f"output: OUT{i}", text)

            # No block has another block's header spliced inside it: between a
            # header and its closing footer there must be no second header.
            blocks = text.split("──── judge call ────")[1:]
            for b in blocks:
                body = b.split("────────────────────")[0]
                self.assertNotIn("judge call", body)


class TestStreamTee(unittest.TestCase):
    def test_writes_to_both(self):
        from calibrate.utils import StreamTee

        original = io.StringIO()
        log = io.StringIO()
        tee = StreamTee(original, log)
        tee.write("hello")
        self.assertEqual(original.getvalue(), "hello")
        self.assertEqual(log.getvalue(), "hello")

    def test_flush(self):
        from calibrate.utils import StreamTee

        original = MagicMock()
        log = MagicMock()
        tee = StreamTee(original, log)
        tee.flush()
        original.flush.assert_called_once()
        log.flush.assert_called_once()

    def test_isatty(self):
        from calibrate.utils import StreamTee

        original = MagicMock()
        original.isatty.return_value = True
        log = MagicMock()
        tee = StreamTee(original, log)
        self.assertTrue(tee.isatty())

    def test_getattr_proxy(self):
        from calibrate.utils import StreamTee

        original = MagicMock()
        original.custom_attr = "X"
        log = MagicMock()
        tee = StreamTee(original, log)
        self.assertEqual(tee.custom_attr, "X")


class TestSaveAudioChunk(unittest.IsolatedAsyncioTestCase):
    async def test_empty_chunk_returns(self):
        from calibrate.utils import save_audio_chunk

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.wav"
            await save_audio_chunk(str(p), b"", 16000, 1)
            self.assertFalse(p.exists())

    async def test_creates_new_file(self):
        from calibrate.utils import save_audio_chunk

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.wav"
            await save_audio_chunk(str(p), b"\x00" * 100, 16000, 1)
            self.assertTrue(p.exists())

    async def test_appends_to_existing(self):
        from calibrate.utils import save_audio_chunk

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.wav"
            await save_audio_chunk(str(p), b"\x00" * 200, 16000, 1)
            size1 = p.stat().st_size
            await save_audio_chunk(str(p), b"\xff" * 200, 16000, 1)
            size2 = p.stat().st_size
            self.assertGreater(size2, size1)

    async def test_corrupt_file_rewrites(self):
        from calibrate.utils import save_audio_chunk

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.wav"
            p.write_bytes(b"x" * 10)  # too small for WAV
            await save_audio_chunk(str(p), b"\x00" * 100, 16000, 1)


class TestBuildParamProperty(unittest.TestCase):
    def test_simple(self):
        from calibrate.utils import _build_param_property

        prop = _build_param_property({"type": "string", "description": "d"})
        self.assertEqual(prop, {"type": "string", "description": "d"})

    def test_with_items_and_enum(self):
        from calibrate.utils import _build_param_property

        prop = _build_param_property({
            "type": "array",
            "description": "d",
            "items": {"type": "string"},
            "enum": ["a", "b"],
        })
        self.assertEqual(prop["items"], {"type": "string"})
        self.assertEqual(prop["enum"], ["a", "b"])


class TestBuildToolsSchema(unittest.TestCase):
    def test_structured_tool(self):
        from calibrate.utils import build_tools_schema

        schemas, webhooks = build_tools_schema([
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": [
                    {"id": "location", "type": "string", "description": "Where",
                     "required": True},
                    {"id": "units", "type": "string", "description": "Units"},
                ],
            },
        ])
        self.assertEqual(len(schemas), 1)
        self.assertEqual(webhooks, {})
        self.assertEqual(schemas[0].name, "get_weather")

    def test_webhook_tool_full(self):
        from calibrate.utils import build_tools_schema

        schemas, webhooks = build_tools_schema([
            {
                "name": "post_data",
                "description": "post data",
                "type": "webhook",
                "webhook": {
                    "url": "http://x/y",
                    "method": "POST",
                    "headers": [],
                    "queryParameters": [
                        {"id": "q1", "type": "string", "description": "q",
                         "required": True},
                    ],
                    "body": {
                        "description": "the body",
                        "parameters": [
                            {"id": "b1", "type": "string", "description": "b",
                             "required": True},
                        ],
                    },
                },
            },
        ])
        self.assertEqual(len(schemas), 1)
        self.assertEqual(webhooks["post_data"]["method"], "POST")

    def test_webhook_missing_url(self):
        from calibrate.utils import build_tools_schema

        with self.assertRaises(ValueError):
            build_tools_schema([{
                "name": "x", "description": "d", "type": "webhook",
                "webhook": {"method": "GET"},
            }])

    def test_webhook_missing_method(self):
        from calibrate.utils import build_tools_schema

        with self.assertRaises(ValueError):
            build_tools_schema([{
                "name": "x", "description": "d", "type": "webhook",
                "webhook": {"url": "http://x"},
            }])


class TestMakeWebhookCall(unittest.IsolatedAsyncioTestCase):
    async def test_successful_get(self):
        from calibrate import utils as U

        fake_response = MagicMock()
        fake_response.status = 200
        fake_response.json = AsyncMock(return_value={"ok": True})
        fake_response.__aenter__ = AsyncMock(return_value=fake_response)
        fake_response.__aexit__ = AsyncMock(return_value=False)

        fake_session = MagicMock()
        fake_session.request = MagicMock(return_value=fake_response)
        fake_session.__aenter__ = AsyncMock(return_value=fake_session)
        fake_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=fake_session):
            result = await U.make_webhook_call(
                {"url": "http://x", "method": "GET", "headers": [{"name": "K", "value": "V"}]},
                {"query": {"a": 1}},
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["status_code"], 200)

    async def test_post_with_body(self):
        from calibrate import utils as U

        fake_response = MagicMock()
        fake_response.status = 201
        fake_response.json = AsyncMock(side_effect=Exception("not json"))
        fake_response.text = AsyncMock(return_value="raw text")
        fake_response.__aenter__ = AsyncMock(return_value=fake_response)
        fake_response.__aexit__ = AsyncMock(return_value=False)

        fake_session = MagicMock()
        fake_session.request = MagicMock(return_value=fake_response)
        fake_session.__aenter__ = AsyncMock(return_value=fake_session)
        fake_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=fake_session):
            result = await U.make_webhook_call(
                {"url": "http://x", "method": "POST", "headers": []},
                {"body": {"k": "v"}},
            )
        self.assertEqual(result["status_code"], 201)
        self.assertEqual(result["response"], "raw text")

    async def test_timeout(self):
        from calibrate import utils as U

        fake_session = MagicMock()
        fake_session.request = MagicMock(side_effect=asyncio.TimeoutError())
        fake_session.__aenter__ = AsyncMock(return_value=fake_session)
        fake_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=fake_session):
            result = await U.make_webhook_call(
                {"url": "http://x", "method": "GET", "headers": []},
                {},
            )
        self.assertEqual(result["status"], "error")
        self.assertIn("timed out", result["error"])

    async def test_client_error(self):
        from calibrate import utils as U
        import aiohttp

        fake_session = MagicMock()
        fake_session.request = MagicMock(side_effect=aiohttp.ClientError("boom"))
        fake_session.__aenter__ = AsyncMock(return_value=fake_session)
        fake_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=fake_session):
            result = await U.make_webhook_call(
                {"url": "http://x", "method": "GET", "headers": []},
                {},
            )
        self.assertEqual(result["status"], "error")


class TestCreateSTTService(unittest.TestCase):
    def test_unknown_provider(self):
        from calibrate.utils import create_stt_service

        with self.assertRaises(ValueError):
            create_stt_service("bogus", "english")


class TestCreateTTSService(unittest.TestCase):
    def test_unknown_provider(self):
        from calibrate.utils import create_tts_service

        with self.assertRaises(ValueError):
            create_tts_service("bogus", "english")


class TestAddDefaultSource(unittest.TestCase):
    def test_adds_source_when_missing(self):
        from calibrate.utils import add_default_source

        record = {"extra": {}}
        add_default_source(record)
        self.assertIn("source", record["extra"])

    def test_keeps_existing_source(self):
        from calibrate.utils import add_default_source

        record = {"extra": {"source": "FOO"}}
        add_default_source(record)
        self.assertEqual(record["extra"]["source"], "FOO")


class TestPatchLangfuseTrace(unittest.TestCase):
    def test_patches_and_exercises(self):
        from calibrate.utils import patch_langfuse_trace
        from pipecat.utils.tracing import service_decorators

        original = service_decorators.add_llm_span_attributes
        try:
            # Replace original first with a no-op so the patched wrapper can call it
            service_decorators.add_llm_span_attributes = lambda *a, **k: None
            patch_langfuse_trace("test_trace")
            # Now exercise the patched function
            span = MagicMock()
            service_decorators.add_llm_span_attributes(span, messages=[{"role": "user"}])
            # Call set_attribute with key "output" to exercise that branch
            span.set_attribute("output", "x")
        finally:
            service_decorators.add_llm_span_attributes = original


class TestSummarizeMetricDistribution(unittest.TestCase):
    def test_minimal_entry_has_mean_std_values(self):
        from calibrate.utils import summarize_metric_distribution

        entry = summarize_metric_distribution([1.0, 2.0, 6.0])
        self.assertEqual(
            entry, {"mean": 3.0, "std": entry["std"], "values": [1.0, 2.0, 6.0]}
        )
        # No type/scale/evaluator_id when not supplied.
        self.assertNotIn("type", entry)
        self.assertNotIn("scale_min", entry)
        self.assertNotIn("evaluator_id", entry)
        # Aggregates are plain floats (JSON-serializable, not numpy types).
        for k in ("mean", "std"):
            self.assertIs(type(entry[k]), float)

    def test_optional_fields_included_when_supplied(self):
        from calibrate.utils import summarize_metric_distribution

        entry = summarize_metric_distribution(
            [4, 4, 2],
            metric_type="rating",
            scale=(1, 5),
            evaluator_id="ev_123",
        )
        self.assertEqual(entry["type"], "rating")
        self.assertEqual(entry["mean"], pytest_approx(10 / 3))
        self.assertEqual((entry["scale_min"], entry["scale_max"]), (1, 5))
        self.assertEqual(entry["evaluator_id"], "ev_123")

    def test_is_json_serializable(self):
        from calibrate.utils import summarize_metric_distribution

        entry = summarize_metric_distribution([0, 1, 1], metric_type="binary")
        json.dumps(entry)  # must not raise


class TestReadLeaderboardMetrics(unittest.TestCase):
    def _write(self, path: Path, data: dict) -> Path:
        path.write_text(json.dumps(data))
        return path

    def test_missing_file_returns_empty(self):
        from calibrate.utils import read_leaderboard_metrics

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                read_leaderboard_metrics(Path(tmp) / "nope.json"), {}
            )

    def test_current_format_extracts_mean(self):
        from calibrate.utils import read_leaderboard_metrics

        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(Path(tmp) / "m.json", {
                "wer": 0.1,
                "semantic_match": {"type": "binary", "mean": 0.85},
                "ttfb": {"mean": 0.4},
            })
            out = read_leaderboard_metrics(p)
            self.assertEqual(out["semantic_match"], 0.85)
            self.assertEqual(out["ttfb"], 0.4)
            self.assertEqual(out["wer"], 0.1)

    def test_legacy_metric_name_format(self):
        from calibrate.utils import read_leaderboard_metrics

        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(Path(tmp) / "m.json", {"metric_name": "wer", "mean": 0.2})
            self.assertEqual(read_leaderboard_metrics(p), {"wer": 0.2})


class TestApplyDebugLimit(unittest.TestCase):
    def test_truncates_to_debug_count(self):
        from calibrate.utils import apply_debug_limit

        items = list(range(10))
        result = apply_debug_limit(items, True, 3)
        self.assertEqual(result, [0, 1, 2])

    def test_noop_when_debug_off(self):
        from calibrate.utils import apply_debug_limit

        items = list(range(10))
        result = apply_debug_limit(items, False, 3)
        self.assertEqual(result, items)

    def test_count_larger_than_list_returns_all(self):
        from calibrate.utils import apply_debug_limit

        items = [1, 2]
        result = apply_debug_limit(items, True, 5)
        self.assertEqual(result, [1, 2])

    def test_empty_list(self):
        from calibrate.utils import apply_debug_limit

        self.assertEqual(apply_debug_limit([], True, 5), [])

    def test_prints_banner_only_when_truncating(self):
        from calibrate.utils import apply_debug_limit

        with patch("builtins.print") as mock_print:
            apply_debug_limit([1, 2, 3], True, 2)
        self.assertTrue(mock_print.called)

        with patch("builtins.print") as mock_print:
            apply_debug_limit([1, 2, 3], False, 2)
        self.assertFalse(mock_print.called)


def pytest_approx(value, tol=1e-9):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol

    return _Approx()


if __name__ == "__main__":
    unittest.main()
