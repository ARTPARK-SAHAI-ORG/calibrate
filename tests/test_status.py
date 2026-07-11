"""Tests for calibrate_agent/status.py — provider checks and main()."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import httpx


def _mk_resp(body=None, status=200, content=b"audio_data" * 50):
    m = MagicMock()
    m.status_code = status
    m.raise_for_status = MagicMock()
    if status >= 400:
        m.raise_for_status.side_effect = httpx.HTTPStatusError(
            "fail", request=MagicMock(), response=m,
        )
    if body is not None:
        m.json = MagicMock(return_value=body)
    m.content = content
    return m


def _mk_client(post_resp=None):
    client = AsyncMock()
    if post_resp:
        client.post = AsyncMock(return_value=post_resp)
    return client


class _FakeWS:
    """Websocket stand-in supporting both async iteration (TTS check) and
    recv() (STT check)."""

    def __init__(self, messages, recv_end=None):
        self._messages = list(messages)
        self.sent = []
        self._recv_end = recv_end or asyncio.TimeoutError()

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        self._it = iter(list(self._messages))
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def recv(self):
        if self._messages:
            return self._messages.pop(0)
        raise self._recv_end


class _FakeConnect:
    def __init__(self, ws):
        self._ws = ws

    async def __aenter__(self):
        return self._ws

    async def __aexit__(self, *exc):
        return False


def _patch_ws(messages, recv_end=None):
    """Patch the websockets connect used by the Smallest checks to yield messages."""
    ws = _FakeWS(messages, recv_end=recv_end)
    return patch(
        "websockets.asyncio.client.connect",
        MagicMock(return_value=_FakeConnect(ws)),
    ), ws


def _patch_smallest_both(tts_ws, stt_ws):
    """Route connect() to the right fake by URL for the combined check."""

    def _connect(url, *a, **k):
        return _FakeConnect(stt_ws if "stt/live" in url else tts_ws)

    return patch("websockets.asyncio.client.connect", MagicMock(side_effect=_connect))


class TestSilenceWav(unittest.TestCase):
    def test_generate_silence(self):
        from calibrate_agent.status import _generate_silence_wav

        wav = _generate_silence_wav(duration_s=0.1, sample_rate=8000)
        self.assertTrue(wav.startswith(b"RIFF"))


class TestCheckProviders(unittest.IsolatedAsyncioTestCase):
    async def test_check_openai_ok(self):
        from calibrate_agent.status import _check_openai

        client = _mk_client(_mk_resp({"choices": [{"message": {"content": "Hi"}}]}))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            result = await _check_openai(client)
        self.assertEqual(result, "llm")

    async def test_check_openai_no_choices(self):
        from calibrate_agent.status import _check_openai

        client = _mk_client(_mk_resp({}))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with self.assertRaises(ValueError):
                await _check_openai(client)

    async def test_check_openrouter_ok(self):
        from calibrate_agent.status import _check_openrouter

        client = _mk_client(_mk_resp({"choices": [{"message": {"content": "Hi"}}]}))
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}):
            result = await _check_openrouter(client)
        self.assertEqual(result, "llm")

    async def test_check_openrouter_no_choices(self):
        from calibrate_agent.status import _check_openrouter

        client = _mk_client(_mk_resp({}))
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}):
            with self.assertRaises(ValueError):
                await _check_openrouter(client)

    async def test_check_groq_ok(self):
        from calibrate_agent.status import _check_groq

        client = _mk_client(_mk_resp(content=b"x" * 200))
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            result = await _check_groq(client)
        self.assertEqual(result, "tts")

    async def test_check_groq_empty(self):
        from calibrate_agent.status import _check_groq

        client = _mk_client(_mk_resp(content=b"x"))
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with self.assertRaises(ValueError):
                await _check_groq(client)

    async def test_check_google_missing_creds(self):
        from calibrate_agent.status import _check_google

        client = _mk_client()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await _check_google(client)

    async def test_check_google_missing_file(self):
        from calibrate_agent.status import _check_google

        client = _mk_client()
        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent.json"}):
            with self.assertRaises(FileNotFoundError):
                await _check_google(client)

    async def test_check_google_missing_project(self):
        from calibrate_agent.status import _check_google

        client = _mk_client()
        with tempfile.TemporaryDirectory() as tmp:
            creds = Path(tmp) / "creds.json"
            creds.write_text("{}")
            with patch.dict(os.environ,
                            {"GOOGLE_APPLICATION_CREDENTIALS": str(creds)},
                            clear=True):
                with self.assertRaises(ValueError):
                    await _check_google(client)

    async def test_check_elevenlabs_ok(self):
        from calibrate_agent.status import _check_elevenlabs

        client = _mk_client(_mk_resp(content=b"x" * 200))
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}):
            result = await _check_elevenlabs(client)
        self.assertEqual(result, "tts")

    async def test_check_elevenlabs_empty(self):
        from calibrate_agent.status import _check_elevenlabs

        client = _mk_client(_mk_resp(content=b""))
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}):
            with self.assertRaises(ValueError):
                await _check_elevenlabs(client)

    async def test_check_cartesia_ok(self):
        from calibrate_agent.status import _check_cartesia

        client = _mk_client(_mk_resp(content=b"x" * 200))
        with patch.dict(os.environ, {"CARTESIA_API_KEY": "k"}):
            result = await _check_cartesia(client)
        self.assertEqual(result, "tts")

    async def test_check_cartesia_empty(self):
        from calibrate_agent.status import _check_cartesia

        client = _mk_client(_mk_resp(content=b""))
        with patch.dict(os.environ, {"CARTESIA_API_KEY": "k"}):
            with self.assertRaises(ValueError):
                await _check_cartesia(client)

    async def test_check_deepgram_ok(self):
        from calibrate_agent.status import _check_deepgram

        client = _mk_client(_mk_resp())
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "k"}):
            result = await _check_deepgram(client)
        self.assertEqual(result, "stt")

    async def test_check_smallest_tts_ok(self):
        import json
        from calibrate_agent.status import _check_smallest_tts

        chunk = json.dumps({"status": "chunk", "data": {"audio": "YWJj"}})
        patcher, ws = _patch_ws([chunk])
        with patch.dict(os.environ, {"SMALLEST_API_KEY": "k"}), patcher:
            result = await _check_smallest_tts()
        self.assertEqual(result, "tts")
        self.assertEqual(len(ws.sent), 1)

    async def test_check_smallest_tts_server_error(self):
        import json
        from calibrate_agent.status import _check_smallest_tts

        err = json.dumps({"status": "error", "error": "MODEL_DEPRECATED"})
        patcher, _ = _patch_ws([err])
        with patch.dict(os.environ, {"SMALLEST_API_KEY": "k"}), patcher:
            with self.assertRaises(RuntimeError):
                await _check_smallest_tts()

    async def test_check_smallest_tts_no_audio(self):
        import json
        from calibrate_agent.status import _check_smallest_tts

        done = json.dumps({"status": "complete"})
        patcher, _ = _patch_ws([done])
        with patch.dict(os.environ, {"SMALLEST_API_KEY": "k"}), patcher:
            with self.assertRaises(ValueError):
                await _check_smallest_tts()

    async def test_check_smallest_stt_ok(self):
        import json
        from calibrate_agent.status import _check_smallest_stt

        msg = json.dumps({"type": "transcription", "transcript": "", "is_last": True})
        patcher, ws = _patch_ws([msg])
        with patch.dict(os.environ, {"SMALLEST_API_KEY": "k"}), patcher:
            result = await _check_smallest_stt()
        self.assertEqual(result, "stt")
        # Sent the silence buffer and a close_stream control message.
        self.assertEqual(len(ws.sent), 2)

    async def test_check_smallest_stt_idle_closes_ok(self):
        # No messages -> recv() raises the end sentinel -> treated as success.
        from calibrate_agent.status import _check_smallest_stt

        patcher, _ = _patch_ws([])
        with patch.dict(os.environ, {"SMALLEST_API_KEY": "k"}), patcher:
            result = await _check_smallest_stt()
        self.assertEqual(result, "stt")

    async def test_check_smallest_stt_server_error(self):
        import json
        from calibrate_agent.status import _check_smallest_stt

        err = json.dumps({"type": "error", "message": "bad audio"})
        patcher, _ = _patch_ws([err])
        with patch.dict(os.environ, {"SMALLEST_API_KEY": "k"}), patcher:
            with self.assertRaises(RuntimeError):
                await _check_smallest_stt()

    async def test_check_smallest_both_ok(self):
        import json
        from calibrate_agent.status import _check_smallest

        tts_ws = _FakeWS([json.dumps({"status": "chunk", "data": {"audio": "YWJj"}})])
        stt_ws = _FakeWS([json.dumps({"type": "transcription", "is_last": True})])
        with patch.dict(os.environ, {"SMALLEST_API_KEY": "k"}), \
                _patch_smallest_both(tts_ws, stt_ws):
            result = await _check_smallest(_mk_client())
        self.assertEqual(result, "stt,tts")

    async def test_check_smallest_fails_if_stt_down(self):
        import json
        from calibrate_agent.status import _check_smallest

        tts_ws = _FakeWS([json.dumps({"status": "chunk", "data": {"audio": "YWJj"}})])
        stt_ws = _FakeWS([json.dumps({"type": "error", "message": "down"})])
        with patch.dict(os.environ, {"SMALLEST_API_KEY": "k"}), \
                _patch_smallest_both(tts_ws, stt_ws):
            with self.assertRaises(RuntimeError):
                await _check_smallest(_mk_client())


class TestCheckSingleProvider(unittest.IsolatedAsyncioTestCase):
    async def test_missing_env_vars_skips(self):
        from calibrate_agent.status import _check_single_provider

        provider = {"name": "openai", "types": ["llm"], "env_vars": ["OPENAI_API_KEY"]}
        with patch.dict(os.environ, {}, clear=True):
            result = await _check_single_provider(provider, MagicMock())
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["missing_vars"], ["OPENAI_API_KEY"])

    async def test_ok_path(self):
        from calibrate_agent import status as S

        provider = {"name": "openai", "types": ["llm"], "env_vars": ["OPENAI_API_KEY"]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), \
             patch.dict(S._CHECK_FUNCTIONS, {"openai": AsyncMock(return_value="llm")}):
            result = await S._check_single_provider(provider, MagicMock())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["check_type"], "llm")

    async def test_timeout(self):
        from calibrate_agent import status as S

        provider = {"name": "openai", "types": ["llm"], "env_vars": ["OPENAI_API_KEY"]}

        async def slow(*a, **kw):
            await asyncio.sleep(100)
            return "llm"

        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), \
             patch.dict(S._CHECK_FUNCTIONS, {"openai": slow}), \
             patch("asyncio.wait_for", AsyncMock(side_effect=asyncio.TimeoutError())):
            result = await S._check_single_provider(provider, MagicMock())
        self.assertEqual(result["status"], "fail")
        self.assertIn("Timed out", result["error"])

    async def test_http_status_error(self):
        from calibrate_agent import status as S

        provider = {"name": "openai", "types": ["llm"], "env_vars": ["OPENAI_API_KEY"]}

        async def fails(*a, **kw):
            resp = MagicMock()
            resp.status_code = 401
            raise httpx.HTTPStatusError("auth fail", request=MagicMock(), response=resp)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), \
             patch.dict(S._CHECK_FUNCTIONS, {"openai": fails}):
            result = await S._check_single_provider(provider, MagicMock())
        self.assertEqual(result["status"], "fail")
        self.assertIn("HTTP 401", result["error"])

    async def test_other_exception(self):
        from calibrate_agent import status as S

        provider = {"name": "openai", "types": ["llm"], "env_vars": ["OPENAI_API_KEY"]}

        async def fails(*a, **kw):
            raise RuntimeError("x" * 200)  # long error message

        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), \
             patch.dict(S._CHECK_FUNCTIONS, {"openai": fails}):
            result = await S._check_single_provider(provider, MagicMock())
        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["error"].endswith("..."))


class TestPrintResults(unittest.TestCase):
    def test_mixed_statuses(self):
        from calibrate_agent.status import _print_results

        results = [
            {"name": "p1", "types": ["llm"], "key_set": True, "missing_vars": [],
             "status": "ok", "check_type": "llm", "error": None, "latency_ms": 500},
            {"name": "p2", "types": ["stt"], "key_set": False, "missing_vars": ["K"],
             "status": "skipped", "check_type": None, "error": None, "latency_ms": None},
            {"name": "p3", "types": ["tts"], "key_set": True, "missing_vars": [],
             "status": "fail", "check_type": None, "error": "boom", "latency_ms": 100},
        ]
        _print_results(results)


class TestMain(unittest.IsolatedAsyncioTestCase):
    async def test_main_default(self):
        from calibrate_agent import status as S

        async def fake_check(provider, client, emit=None):
            return {"name": provider["name"], "types": provider["types"],
                    "key_set": False, "missing_vars": ["X"],
                    "status": "skipped", "check_type": None,
                    "error": None, "latency_ms": None}

        with patch.object(S, "_check_single_provider", fake_check):
            result = await S.main()
        self.assertIsInstance(result, dict)

    async def test_main_quiet(self):
        from calibrate_agent import status as S

        async def fake_check(provider, client, emit=None):
            return {"name": provider["name"], "types": provider["types"],
                    "key_set": True, "missing_vars": [],
                    "status": "ok", "check_type": "llm",
                    "error": None, "latency_ms": 200}

        with patch.object(S, "_check_single_provider", fake_check):
            result = await S.main(quiet=True)
        self.assertIsInstance(result, dict)
        for v in result.values():
            self.assertEqual(v["status"], "pass")


class TestStreamingStatus(unittest.IsolatedAsyncioTestCase):
    async def test_iter_status_events_streams_live_progress(self):
        from calibrate_agent import status as S

        providers = [
            {"name": "openai", "types": ["llm"], "env_vars": ["OPENAI_API_KEY"]},
        ]

        with patch.object(S, "PROVIDERS", providers), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), \
             patch.dict(S._CHECK_FUNCTIONS, {"openai": AsyncMock(return_value="llm")}):
            events = [event async for event in S.iter_status_events()]

        self.assertEqual([event["stage"] for event in events], [
            "input_sent",
            "output_received",
            "working",
        ])
        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["result"]["status"], "pass")

    async def test_iter_status_yields_provider_results_as_ready(self):
        from calibrate_agent import status as S

        release_slow = asyncio.Event()
        providers = [
            {"name": "slow", "types": ["llm"], "env_vars": ["SLOW_KEY"]},
            {"name": "fast", "types": ["llm"], "env_vars": ["FAST_KEY"]},
        ]

        async def fake_check(provider, client, emit=None):
            if provider["name"] == "slow":
                await release_slow.wait()
            result = {
                "name": provider["name"],
                "types": provider["types"],
                "key_set": True,
                "missing_vars": [],
                "status": "ok",
                "check_type": "llm",
                "error": None,
                "latency_ms": 10,
            }
            if emit is not None:
                await emit({
                    "type": "result",
                    "provider": provider["name"],
                    "types": provider["types"],
                    "stage": "working",
                    "message": "Working",
                    "result": S._status_json_entry(result),
                })
            return result

        with patch.object(S, "PROVIDERS", providers), \
             patch.object(S, "_check_single_provider", fake_check):
            stream = S.iter_status()
            first = await asyncio.wait_for(stream.__anext__(), timeout=1)
            release_slow.set()
            second = await asyncio.wait_for(stream.__anext__(), timeout=1)

        self.assertEqual(first["provider"], "fast")
        self.assertEqual(first["result"]["status"], "pass")
        self.assertEqual(second["provider"], "slow")


if __name__ == "__main__":
    unittest.main()
