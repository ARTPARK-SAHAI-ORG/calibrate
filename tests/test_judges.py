"""
Unit tests for calibrate_agent/judges.py — the unified judge module.

Covers the new evaluator-based API:
- is_rating / evaluator_result_value
- render_template / render_evaluator (placeholder substitution)
- text_judge fans out one LLM call per evaluator and keys results by name
- simulation_judge formats transcript and delegates to text_judge
- audio_judge attaches a base64 audio block per call
- Default evaluators for STT, TTS, and LLM-tests are well-formed

Run with:
    python -m pytest tests/test_judges.py -v
"""

import csv
import os
import unittest
from typing import Optional
from unittest.mock import patch, AsyncMock, MagicMock

from pydantic import BaseModel

from calibrate_agent.judges import (
    USAGE_FIELDS,
    call_usage,
    openrouter_client_recording_usage,
    evaluator_row_columns,
    write_judge_usage,
    text_judge,
    simulation_judge,
    general_task_judge,
    format_task_io,
    audio_judge,
    is_rating,
    evaluator_result_value,
    render_template,
    render_evaluator,
    ensure_known_evaluator_names,
    format_conversation,
    _result_model_for_evaluator,
    _sanitize_evaluator_for_tool_model,
    _normalize_judge_api_result,
    CriterionResult,
    DEFAULT_TEXT_JUDGE_MODEL,
    DEFAULT_AUDIO_JUDGE_MODEL,
    DEFAULT_SIMULATION_JUDGE_MODEL,
    DEFAULT_LLM_TEST_EVALUATOR,
    DEFAULT_GENERAL_TASK_EVALUATOR,
    DEFAULT_STT_EVALUATOR,
    DEFAULT_TTS_EVALUATOR,
)


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


class TestIsRating(unittest.TestCase):
    def test_binary_evaluator_is_not_rating(self):
        self.assertFalse(is_rating({"name": "x", "system_prompt": "y"}))
        self.assertFalse(
            is_rating({"name": "x", "type": "binary", "system_prompt": "y"})
        )

    def test_rating_evaluator(self):
        self.assertTrue(
            is_rating(
                {"name": "x", "type": "rating", "scale_min": 1, "scale_max": 5}
            )
        )


class TestEvaluatorResultValue(unittest.TestCase):
    def test_binary_true_is_one(self):
        ev = {"name": "x", "system_prompt": "y"}
        self.assertEqual(
            evaluator_result_value(ev, {"reasoning": "ok", "match": True}), 1.0
        )

    def test_binary_false_is_zero(self):
        ev = {"name": "x", "system_prompt": "y"}
        self.assertEqual(
            evaluator_result_value(ev, {"reasoning": "ok", "match": False}), 0.0
        )

    def test_rating_returns_score_as_float(self):
        ev = {
            "name": "x",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
            "system_prompt": "y",
        }
        self.assertEqual(
            evaluator_result_value(ev, {"reasoning": "ok", "score": 3}), 3.0
        )


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TestRenderTemplate(unittest.TestCase):
    def test_substitutes_placeholder(self):
        out = render_template("hello {{name}}", {"name": "world"})
        self.assertEqual(out, "hello world")

    def test_substitutes_multiple(self):
        out = render_template(
            "{{a}} and {{b}}", {"a": "foo", "b": "bar"}
        )
        self.assertEqual(out, "foo and bar")

    def test_missing_placeholder_left_intact(self):
        out = render_template("hello {{name}}", {})
        self.assertEqual(out, "hello {{name}}")

    def test_no_placeholders_unchanged(self):
        out = render_template("just text", {"name": "world"})
        self.assertEqual(out, "just text")


class TestRenderEvaluator(unittest.TestCase):
    def test_renders_system_prompt(self):
        ev = {
            "name": "default",
            "system_prompt": "Evaluate: {{criteria}}",
            "judge_model": "openai/gpt-4.1",
        }
        rendered = render_evaluator(ev, {"criteria": "be polite"})
        self.assertEqual(rendered["system_prompt"], "Evaluate: be polite")
        # Other keys preserved
        self.assertEqual(rendered["name"], "default")
        self.assertEqual(rendered["judge_model"], "openai/gpt-4.1")

    def test_does_not_mutate_input(self):
        ev = {
            "name": "default",
            "system_prompt": "Evaluate: {{criteria}}",
        }
        render_evaluator(ev, {"criteria": "x"})
        self.assertEqual(ev["system_prompt"], "Evaluate: {{criteria}}")


class TestEnsureKnownEvaluatorNames(unittest.TestCase):
    def test_all_known_is_noop(self):
        # Accepts both a set and a name->evaluator dict as `known`.
        ensure_known_evaluator_names(["a", "b"], {"a", "b", "c"})
        ensure_known_evaluator_names(["a"], {"a": {}, "b": {}})

    def test_unknown_raises_with_names_and_context(self):
        with self.assertRaises(ValueError) as ctx:
            ensure_known_evaluator_names(
                ["a", "typo"], {"a", "b"}, context="Row 3 arguments"
            )
        msg = str(ctx.exception)
        self.assertIn("typo", msg)
        self.assertIn("Row 3 arguments", msg)


class TestToolCallParamEvaluator(unittest.TestCase):
    def test_default_without_override(self):
        from calibrate_agent.judges import (
            tool_call_param_evaluator,
            DEFAULT_TOOL_CALL_PARAM_EVALUATOR,
        )

        ev = tool_call_param_evaluator()
        self.assertEqual(ev["name"], "tool_call_parameter")
        self.assertEqual(ev["type"], "binary")
        self.assertEqual(
            ev["judge_model"], DEFAULT_TOOL_CALL_PARAM_EVALUATOR["judge_model"]
        )
        self.assertIn("{{criteria}}", ev["system_prompt"])

    def test_judge_model_override(self):
        from calibrate_agent.judges import tool_call_param_evaluator

        ev = tool_call_param_evaluator("openai/gpt-4.1")
        self.assertEqual(ev["judge_model"], "openai/gpt-4.1")

    def test_does_not_mutate_default(self):
        from calibrate_agent.judges import (
            tool_call_param_evaluator,
            DEFAULT_TOOL_CALL_PARAM_EVALUATOR,
        )

        original = dict(DEFAULT_TOOL_CALL_PARAM_EVALUATOR)
        tool_call_param_evaluator("some/other-model")
        self.assertEqual(DEFAULT_TOOL_CALL_PARAM_EVALUATOR, original)


# ---------------------------------------------------------------------------
# Tool-name sanitization and API result shape
# ---------------------------------------------------------------------------


class TestSanitizeEvaluatorForToolModel(unittest.TestCase):
    def test_spaces_and_ampersand(self):
        self.assertEqual(
            _sanitize_evaluator_for_tool_model("Empathy & Tone"),
            "Empathy_Tone",
        )

    def test_goal_completion(self):
        self.assertEqual(
            _sanitize_evaluator_for_tool_model("Goal Completion"),
            "Goal_Completion",
        )

    def test_leading_digit(self):
        self.assertEqual(_sanitize_evaluator_for_tool_model("1st pass"), "E_1st_pass")


class TestNormalizeJudgeApiResult(unittest.TestCase):
    def test_flat_dict_unchanged(self):
        flat = {"reasoning": "ok", "score": 3}
        self.assertEqual(
            _normalize_judge_api_result(flat, "RatingResult_x"),
            flat,
        )

    def test_unwraps_nested_model_key(self):
        nested = {
            "RatingResult_Empathy_Tone": {"reasoning": "ok", "score": 4},
        }
        self.assertEqual(
            _normalize_judge_api_result(nested, "RatingResult_Empathy_Tone"),
            {"reasoning": "ok", "score": 4},
        )


# ---------------------------------------------------------------------------
# Result model construction
# ---------------------------------------------------------------------------


class TestResultModelForEvaluator(unittest.TestCase):
    def test_binary_uses_criterion_result(self):
        Output = _result_model_for_evaluator(
            {"name": "x", "system_prompt": "y"}
        )
        self.assertIs(Output, CriterionResult)
        instance = Output(reasoning="ok", match=True)
        self.assertTrue(instance.match)
        self.assertEqual(instance.reasoning, "ok")

    def test_rating_accepts_score_in_range(self):
        Output = _result_model_for_evaluator(
            {
                "name": "fluency",
                "type": "rating",
                "scale_min": 1,
                "scale_max": 5,
                "system_prompt": "rate fluency",
            }
        )
        self.assertTrue(issubclass(Output, BaseModel))
        instance = Output(reasoning="good", score=4)
        self.assertEqual(instance.score, 4)
        self.assertEqual(instance.reasoning, "good")

    def test_rating_coerces_string_score(self):
        Output = _result_model_for_evaluator(
            {
                "name": "pronunciation",
                "type": "rating",
                "scale_min": 1,
                "scale_max": 3,
                "system_prompt": "rate",
            }
        )
        instance = Output(reasoning="ok", score="3")
        self.assertEqual(instance.score, 3)
        self.assertIsInstance(instance.score, int)

    def test_rating_rejects_score_out_of_range(self):
        from pydantic import ValidationError

        Output = _result_model_for_evaluator(
            {
                "name": "fluency",
                "type": "rating",
                "scale_min": 1,
                "scale_max": 3,
                "system_prompt": "rate",
            }
        )
        with self.assertRaises(ValidationError):
            Output(reasoning="x", score=5)

    def test_rating_model_name_sanitizes_evaluator_title(self):
        Output = _result_model_for_evaluator(
            {
                "name": "Empathy & Tone",
                "type": "rating",
                "scale_min": 1,
                "scale_max": 5,
                "system_prompt": "rate",
            }
        )
        self.assertEqual(Output.__name__, "RatingResult_Empathy_Tone")


# ---------------------------------------------------------------------------
# text_judge
# ---------------------------------------------------------------------------


def _mock_instructor_chat_completions(return_values):
    """Build a mock OpenRouter+instructor client.

    ``return_values`` may be a single dict (returned for every call) or a list
    of dicts that will be returned in order across calls.
    """
    if isinstance(return_values, dict):
        return_values = [return_values]

    parsed_objs = []
    for v in return_values:
        parsed = MagicMock()
        parsed.model_dump.return_value = v
        parsed_objs.append(parsed)

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=parsed_objs)
    return client


def _usage_block(usage: Optional[dict]):
    """Build a reply's usage object, or None for a model that reports none."""
    if usage is None:
        return None
    usage_obj = MagicMock()
    usage_obj.model_dump.return_value = usage
    return usage_obj


def _mock_openrouter(return_values, usage=None):
    """Stand in for the OpenRouter client and instructor together.

    Returns ``(openrouter_client, apatch)`` to patch over
    ``_build_openrouter_client`` and ``instructor.apatch``. The instructor
    stand-in calls through whatever client it is handed, so the usage recording
    a live run relies on is exercised rather than bypassed.
    """
    if isinstance(return_values, dict):
        return_values = [return_values]

    parsed_objs = []
    replies = []
    for v in return_values:
        parsed = MagicMock()
        parsed.model_dump.return_value = v
        parsed_objs.append(parsed)
        reply = MagicMock()
        reply.usage = _usage_block(usage)
        replies.append(reply)

    openrouter_client = MagicMock()
    send = AsyncMock(side_effect=replies)
    openrouter_client.chat.completions.create = send
    # The recording wrapper replaces ``create``, so hold on to the mock the
    # request actually reaches for tests that assert on what was sent.
    openrouter_client.sent = send

    parsed_iter = iter(parsed_objs)
    patched = MagicMock()

    def apatch(recording_client):
        async def create(**kwargs):
            await recording_client.chat.completions.create(**kwargs)
            return next(parsed_iter)

        patched.chat.completions.create = AsyncMock(side_effect=create)
        return patched

    return openrouter_client, apatch


def _drop_usage(judge_result: dict) -> dict:
    """Strip the per-call cost/token/latency fields from a judge result.

    Lets a test assert on the graded answer alone; the usage fields have their
    own tests.
    """
    return {
        name: {k: v for k, v in payload.items() if k not in USAGE_FIELDS}
        for name, payload in judge_result.items()
    }


class TestJudgeIOLogging(unittest.IsolatedAsyncioTestCase):
    """The judge writes its prompt/response into the bound run log file."""

    async def test_logs_judge_io_to_bound_file(self):
        import tempfile, os
        from calibrate_agent.utils import provider_log_file

        client = _mock_instructor_chat_completions(
            [{"reasoning": "looks right", "match": True}]
        )
        f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log")
        f.close()
        token = provider_log_file.set(f.name)
        try:
            with patch(
                "calibrate_agent.judges.instructor.apatch", return_value=client
            ), patch(
                "calibrate_agent.judges._build_openrouter_client", return_value=MagicMock()
            ):
                await text_judge(
                    evaluators=[
                        {
                            "name": "accuracy",
                            "system_prompt": "Evaluate accuracy of: PLACEHOLDER",
                            "judge_model": "openai/gpt-4.1",
                        }
                    ],
                    user_prompt="my-context",
                )
            contents = open(f.name).read()
        finally:
            provider_log_file.reset(token)
            os.unlink(f.name)

        self.assertIn("judge call", contents)
        self.assertIn("accuracy", contents)            # evaluator name
        self.assertIn("openai/gpt-4.1", contents)      # model
        self.assertIn("Evaluate accuracy of", contents)  # system prompt
        self.assertIn("my-context", contents)          # user input
        self.assertIn("looks right", contents)         # judge output reasoning

    async def test_no_log_file_does_not_crash(self):
        from calibrate_agent.utils import provider_log_file

        # Ensure unbound (default None) — judge should run without writing anywhere.
        self.assertIsNone(provider_log_file.get())
        client = _mock_instructor_chat_completions(
            [{"reasoning": "ok", "match": True}]
        )
        with patch(
            "calibrate_agent.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate_agent.judges._build_openrouter_client", return_value=MagicMock()
        ):
            result = await text_judge(
                evaluators=[
                    {"name": "x", "system_prompt": "p", "judge_model": "m"}
                ],
                user_prompt="ctx",
            )
        self.assertEqual(_drop_usage(result), {"x": {"reasoning": "ok", "match": True}})


class TestCallUsage(unittest.TestCase):
    """What one judge call cost, used, and took is read off its usage block."""

    def test_reads_cost_tokens_and_cached_tokens(self):
        usage = _usage_block(
            {
                "cost": 0.00042,
                "prompt_tokens": 1200,
                "completion_tokens": 85,
                "prompt_tokens_details": {"cached_tokens": 1024},
            }
        )
        self.assertEqual(
            call_usage(usage, 1.2345678),
            {
                "cost_usd": 0.00042,
                "input_tokens": 1200,
                "output_tokens": 85,
                "cached_input_tokens": 1024,
                "latency_seconds": 1.2346,
            },
        )

    def test_token_counts_are_whole_numbers(self):
        usage = call_usage(
            _usage_block({"prompt_tokens": 10.0, "completion_tokens": 3.0}), 0.5
        )
        self.assertIsInstance(usage["input_tokens"], int)
        self.assertIsInstance(usage["output_tokens"], int)

    def test_model_reporting_no_usage_still_reports_latency(self):
        self.assertEqual(call_usage(None, 0.25), {"latency_seconds": 0.25})

    def test_non_numeric_usage_values_are_dropped(self):
        usage = _usage_block(
            {"cost": None, "prompt_tokens": "many", "completion_tokens": True}
        )
        self.assertEqual(call_usage(usage, 0.5), {"latency_seconds": 0.5})

    def test_partial_usage_keeps_what_it_has(self):
        self.assertEqual(
            call_usage(_usage_block({"prompt_tokens": 40}), 0.5),
            {"input_tokens": 40, "latency_seconds": 0.5},
        )

    def test_usage_that_cannot_be_dumped_is_ignored(self):
        usage = MagicMock()
        usage.model_dump.side_effect = RuntimeError("boom")
        self.assertEqual(call_usage(usage, 0.5), {"latency_seconds": 0.5})

    def test_usage_that_dumps_to_something_other_than_a_dict_is_ignored(self):
        usage = MagicMock()
        usage.model_dump.return_value = "not a usage block"
        self.assertEqual(call_usage(usage, 0.5), {"latency_seconds": 0.5})

    def test_usage_already_a_plain_dict_is_read_directly(self):
        self.assertEqual(
            call_usage({"prompt_tokens": 7}, 0.5),
            {"input_tokens": 7, "latency_seconds": 0.5},
        )


class TestUsageRecording(unittest.IsolatedAsyncioTestCase):
    """The usage block is read off the reply before instructor rebuilds it."""

    async def test_records_the_usage_of_the_reply_it_passes_through(self):
        client = MagicMock()
        reply = MagicMock()
        reply.usage = _usage_block({"cost": 0.5})
        client.chat.completions.create = AsyncMock(return_value=reply)

        with patch(
            "calibrate_agent.judges._build_openrouter_client", return_value=client
        ):
            recording_client, box = openrouter_client_recording_usage()
            self.assertEqual(box, {})
            returned = await recording_client.chat.completions.create(model="m")

        self.assertIs(returned, reply)
        self.assertIs(box["usage"], reply.usage)

    async def test_a_retry_leaves_the_usage_of_the_last_call(self):
        client = MagicMock()
        first, second = MagicMock(), MagicMock()
        first.usage = _usage_block({"cost": 0.1})
        second.usage = _usage_block({"cost": 0.9})
        client.chat.completions.create = AsyncMock(side_effect=[first, second])

        with patch(
            "calibrate_agent.judges._build_openrouter_client", return_value=client
        ):
            recording_client, box = openrouter_client_recording_usage()
            await recording_client.chat.completions.create(model="m")
            await recording_client.chat.completions.create(model="m")

        self.assertIs(box["usage"], second.usage)


class TestUsageOnJudgeResults(unittest.IsolatedAsyncioTestCase):
    """Every judge result carries what its own call cost, used, and took."""

    _USAGE = {"cost": 0.001, "prompt_tokens": 500, "completion_tokens": 20}

    async def _run_text_judge(self, openrouter_client, apatch):
        with patch(
            "calibrate_agent.judges.instructor.apatch", side_effect=apatch
        ), patch(
            "calibrate_agent.judges._build_openrouter_client",
            return_value=openrouter_client,
        ):
            return await text_judge(
                evaluators=[{"name": "accuracy", "system_prompt": "p"}],
                user_prompt="ctx",
            )

    async def test_text_judge_attaches_usage_per_evaluator(self):
        result = await self._run_text_judge(
            *_mock_openrouter(
                [{"reasoning": "good", "match": True}], usage=self._USAGE
            )
        )
        self.assertEqual(result["accuracy"]["cost_usd"], 0.001)
        self.assertEqual(result["accuracy"]["input_tokens"], 500)
        self.assertEqual(result["accuracy"]["output_tokens"], 20)
        self.assertGreaterEqual(result["accuracy"]["latency_seconds"], 0.0)
        self.assertEqual(result["accuracy"]["match"], True)

    async def test_request_asks_openrouter_for_cost(self):
        openrouter_client, apatch = _mock_openrouter(
            [{"reasoning": "good", "match": True}], usage=self._USAGE
        )
        await self._run_text_judge(openrouter_client, apatch)
        kwargs = openrouter_client.sent.await_args.kwargs
        self.assertEqual(kwargs["extra_body"], {"usage": {"include": True}})

    async def test_model_reporting_no_usage_still_grades_the_row(self):
        result = await self._run_text_judge(
            *_mock_openrouter([{"reasoning": "good", "match": True}], usage=None)
        )
        self.assertEqual(result["accuracy"]["match"], True)
        self.assertNotIn("cost_usd", result["accuracy"])
        self.assertIn("latency_seconds", result["accuracy"])

    async def test_usage_survives_a_nested_payload(self):
        result = await self._run_text_judge(
            *_mock_openrouter(
                [{"CriterionResult": {"reasoning": "good", "match": True}}],
                usage=self._USAGE,
            )
        )
        self.assertEqual(result["accuracy"]["reasoning"], "good")
        self.assertEqual(result["accuracy"]["cost_usd"], 0.001)

    async def test_audio_judge_attaches_usage_per_evaluator(self):
        import tempfile

        openrouter_client, apatch = _mock_openrouter(
            [{"reasoning": "clear", "match": True}], usage=self._USAGE
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFFfake")
            audio_path = f.name
        with patch(
            "calibrate_agent.judges.instructor.apatch", side_effect=apatch
        ), patch(
            "calibrate_agent.judges._build_openrouter_client",
            return_value=openrouter_client,
        ):
            result = await audio_judge(
                evaluators=[{"name": "intelligibility", "system_prompt": "p"}],
                audio_path=audio_path,
                reference_text="hello",
            )
        os.unlink(audio_path)
        self.assertEqual(result["intelligibility"]["cost_usd"], 0.001)
        self.assertEqual(result["intelligibility"]["input_tokens"], 500)
        kwargs = openrouter_client.sent.await_args.kwargs
        self.assertEqual(kwargs["extra_body"], {"usage": {"include": True}})


class TestEvaluatorRowColumns(unittest.TestCase):
    """results.csv keeps the score and the reasoning; usage goes elsewhere."""

    def test_usage_fields_do_not_become_columns(self):
        columns = evaluator_row_columns(
            {"accuracy": {"name": "accuracy"}},
            {
                "accuracy": {
                    "reasoning": "good",
                    "match": True,
                    "cost_usd": 0.002,
                    "latency_seconds": 0.9,
                }
            },
        )
        self.assertEqual(
            columns, {"accuracy": True, "accuracy_reasoning": "good"}
        )


class TestWriteJudgeUsage(unittest.TestCase):
    """judge_usage.csv holds one row per judge call, evaluator name as a value."""

    def _read(self, output_dir):
        with open(os.path.join(output_dir, "judge_usage.csv"), newline="") as f:
            return list(csv.DictReader(f))

    def test_one_row_per_row_and_evaluator(self):
        import tempfile

        judge_rows = [
            {
                "accuracy": {
                    "reasoning": "a",
                    "match": True,
                    "cost_usd": 0.001,
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cached_input_tokens": 0,
                    "latency_seconds": 1.5,
                },
                "tone": {"reasoning": "b", "match": False, "cost_usd": 0.002},
            }
        ]
        with tempfile.TemporaryDirectory() as d:
            write_judge_usage(d, ["row_a"], judge_rows, ["accuracy", "tone"])
            rows = self._read(d)

        self.assertEqual([r["evaluator"] for r in rows], ["accuracy", "tone"])
        self.assertEqual(rows[0]["id"], "row_a")
        self.assertEqual(rows[0]["cost_usd"], "0.001")
        self.assertEqual(rows[0]["input_tokens"], "100")
        self.assertEqual(rows[0]["latency_seconds"], "1.5")
        self.assertEqual(rows[1]["cost_usd"], "0.002")
        self.assertEqual(rows[1]["input_tokens"], "")

    def test_evaluator_names_that_look_like_columns_stay_separate(self):
        import tempfile

        judge_rows = [
            {
                "faithful": {"reasoning": "a", "match": True, "cost_usd": 0.001},
                "faithful_cost_usd": {
                    "reasoning": "b",
                    "match": False,
                    "cost_usd": 0.002,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as d:
            write_judge_usage(
                d, [1], judge_rows, ["faithful", "faithful_cost_usd"]
            )
            rows = self._read(d)

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {r["evaluator"]: r["cost_usd"] for r in rows},
            {"faithful": "0.001", "faithful_cost_usd": "0.002"},
        )

    def test_header_is_written_even_with_no_rows(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = write_judge_usage(d, [], [], ["accuracy"])
            with open(path, newline="") as f:
                header = f.readline().strip()

        self.assertEqual(
            header,
            "id,evaluator," + ",".join(USAGE_FIELDS),
        )

    def test_rows_a_judge_never_reached_are_skipped(self):
        import tempfile

        judge_rows = [None, {"accuracy": {"reasoning": "a", "match": True}}]
        with tempfile.TemporaryDirectory() as d:
            write_judge_usage(d, ["row_a", "row_b"], judge_rows, ["accuracy"])
            rows = self._read(d)

        self.assertEqual([r["id"] for r in rows], ["row_b"])

    def test_an_evaluator_missing_from_a_row_is_skipped(self):
        import tempfile

        judge_rows = [{"accuracy": {"reasoning": "a", "match": True}}]
        with tempfile.TemporaryDirectory() as d:
            write_judge_usage(d, ["row_a"], judge_rows, ["accuracy", "tone"])
            rows = self._read(d)

        self.assertEqual([r["evaluator"] for r in rows], ["accuracy"])


class TestTextJudge(unittest.IsolatedAsyncioTestCase):
    async def test_empty_evaluators_short_circuits(self):
        result = await text_judge(evaluators=[], user_prompt="ctx")
        self.assertEqual(result, {})

    async def test_returns_dict_keyed_by_evaluator_name(self):
        client = _mock_instructor_chat_completions(
            [
                {"reasoning": "good", "match": True},
                {"reasoning": "rude", "match": False},
            ]
        )

        with patch(
            "calibrate_agent.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate_agent.judges._build_openrouter_client", return_value=MagicMock()
        ):
            result = await text_judge(
                evaluators=[
                    {
                        "name": "accuracy",
                        "system_prompt": "Evaluate accuracy",
                        "judge_model": "openai/gpt-4.1",
                    },
                    {
                        "name": "tone",
                        "system_prompt": "Evaluate tone",
                        "judge_model": "openai/gpt-4.1",
                    },
                ],
                user_prompt="ctx",
            )

        self.assertEqual(
            _drop_usage(result),
            {
                "accuracy": {"reasoning": "good", "match": True},
                "tone": {"reasoning": "rude", "match": False},
            },
        )
        # One LLM call per evaluator
        self.assertEqual(client.chat.completions.create.await_count, 2)

    async def test_rating_nested_payload_keyed_by_original_evaluator_name(self):
        """Outer dict keys stay human-readable; nested tool-shaped payloads flatten."""
        client = _mock_instructor_chat_completions(
            [
                {
                    "RatingResult_Empathy_Tone": {
                        "reasoning": "warm",
                        "score": 4,
                    }
                },
            ]
        )
        with patch(
            "calibrate_agent.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate_agent.judges._build_openrouter_client", return_value=MagicMock()
        ):
            result = await text_judge(
                evaluators=[
                    {
                        "name": "Empathy & Tone",
                        "type": "rating",
                        "scale_min": 1,
                        "scale_max": 5,
                        "system_prompt": "rate empathy",
                        "judge_model": "openai/gpt-4.1",
                    },
                ],
                user_prompt="ctx",
            )
        self.assertEqual(
            _drop_usage(result),
            {"Empathy & Tone": {"reasoning": "warm", "score": 4}},
        )

    async def test_uses_evaluator_judge_model(self):
        client = _mock_instructor_chat_completions(
            {"reasoning": "ok", "match": True}
        )
        with patch(
            "calibrate_agent.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate_agent.judges._build_openrouter_client", return_value=MagicMock()
        ):
            await text_judge(
                evaluators=[
                    {
                        "name": "x",
                        "system_prompt": "sys",
                        "judge_model": "custom-model",
                    }
                ],
                user_prompt="ctx",
            )
        call_kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "custom-model")

    async def test_falls_back_when_evaluator_has_no_model(self):
        client = _mock_instructor_chat_completions(
            {"reasoning": "ok", "match": True}
        )
        with patch(
            "calibrate_agent.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate_agent.judges._build_openrouter_client", return_value=MagicMock()
        ):
            await text_judge(
                evaluators=[{"name": "x", "system_prompt": "sys"}],
                user_prompt="ctx",
                fallback_model="fallback-model",
            )
        call_kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "fallback-model")

    async def test_system_prompt_is_passed_verbatim(self):
        client = _mock_instructor_chat_completions(
            {"reasoning": "ok", "match": True}
        )
        with patch(
            "calibrate_agent.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate_agent.judges._build_openrouter_client", return_value=MagicMock()
        ):
            await text_judge(
                evaluators=[
                    {
                        "name": "x",
                        "system_prompt": "UNIQUE-SYS-PROMPT",
                        "judge_model": "openai/gpt-4.1",
                    }
                ],
                user_prompt="UNIQUE-USER-PROMPT",
            )
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        sys_msg = next(m for m in messages if m["role"] == "system")
        user_msg = next(m for m in messages if m["role"] == "user")
        self.assertEqual(sys_msg["content"], "UNIQUE-SYS-PROMPT")
        self.assertEqual(user_msg["content"], "UNIQUE-USER-PROMPT")

    async def test_uses_openrouter_client(self):
        client = _mock_instructor_chat_completions(
            {"reasoning": "ok", "match": True}
        )
        build_mock = MagicMock(return_value=MagicMock())
        with patch(
            "calibrate_agent.judges.instructor.apatch", return_value=client
        ), patch("calibrate_agent.judges._build_openrouter_client", build_mock):
            await text_judge(
                evaluators=[
                    {
                        "name": "x",
                        "system_prompt": "sys",
                        "judge_model": "openai/gpt-4.1",
                    }
                ],
                user_prompt="ctx",
            )
        build_mock.assert_called()


# ---------------------------------------------------------------------------
# simulation_judge
# ---------------------------------------------------------------------------


class TestSimulationJudge(unittest.IsolatedAsyncioTestCase):
    async def test_empty_evaluators_returns_empty_dict(self):
        result = await simulation_judge(
            conversation=[{"role": "user", "content": "Hi"}],
            evaluators=[],
        )
        self.assertEqual(result, {})

    async def test_delegates_to_text_judge_with_formatted_transcript(self):
        conversation = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        evaluators = [
            {
                "name": "greeting",
                "system_prompt": "agent greets",
                "judge_model": "openai/gpt-5.2",
            }
        ]

        mock_text_judge = AsyncMock(
            return_value={"greeting": {"reasoning": "ok", "match": True}}
        )

        with patch("calibrate_agent.judges.text_judge", mock_text_judge):
            result = await simulation_judge(
                conversation=conversation,
                evaluators=evaluators,
            )

        self.assertEqual(
            result, {"greeting": {"reasoning": "ok", "match": True}}
        )
        call_kwargs = mock_text_judge.call_args.kwargs
        self.assertEqual(call_kwargs["evaluators"], evaluators)
        # User prompt includes conversation transcript
        self.assertIn("user: Hi", call_kwargs["user_prompt"])
        self.assertIn("assistant: Hello!", call_kwargs["user_prompt"])

    async def test_tool_calls_included_in_transcript(self):
        conversation = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"SF"}',
                        }
                    }
                ],
            },
        ]
        mock_text_judge = AsyncMock(
            return_value={"x": {"reasoning": "ok", "match": True}}
        )
        with patch("calibrate_agent.judges.text_judge", mock_text_judge):
            await simulation_judge(
                conversation=conversation,
                evaluators=[
                    {
                        "name": "x",
                        "system_prompt": "y",
                        "judge_model": "openai/gpt-5.2",
                    }
                ],
            )

        prompt = mock_text_judge.call_args.kwargs["user_prompt"]
        self.assertIn("[Tool Call] get_weather", prompt)


# ---------------------------------------------------------------------------
# general_task_judge
# ---------------------------------------------------------------------------


class TestFormatTaskIO(unittest.TestCase):
    def test_input_and_output(self):
        out = format_task_io("the answer", input_text="the question")
        self.assertIn("`Input`:\n\nthe question", out)
        self.assertIn("`Output`:\n\nthe answer", out)

    def test_output_only_when_no_input(self):
        out = format_task_io("the answer")
        self.assertNotIn("`Input`", out)
        self.assertEqual(out, "`Output`:\n\nthe answer")

    def test_blank_input_is_omitted(self):
        out = format_task_io("the answer", input_text="   ")
        self.assertNotIn("`Input`", out)

    def test_not_framed_as_conversation(self):
        out = format_task_io("hello", input_text="hi")
        self.assertNotIn("Chat history", out)
        self.assertNotIn("user:", out)
        self.assertNotIn("assistant:", out)


class TestGeneralTaskJudge(unittest.IsolatedAsyncioTestCase):
    async def test_empty_evaluators_returns_empty_dict(self):
        result = await general_task_judge(evaluators=[], output="anything")
        self.assertEqual(result, {})

    async def test_delegates_to_text_judge_with_input_and_output(self):
        evaluators = [
            {
                "name": "task_quality",
                "system_prompt": "judge it",
                "judge_model": "openai/gpt-4.1",
            }
        ]
        mock_text_judge = AsyncMock(
            return_value={"task_quality": {"reasoning": "ok", "match": True}}
        )
        with patch("calibrate_agent.judges.text_judge", mock_text_judge):
            result = await general_task_judge(
                evaluators=evaluators,
                output="a faithful summary",
                input_text="a long document",
            )
        self.assertEqual(
            result, {"task_quality": {"reasoning": "ok", "match": True}}
        )
        call_kwargs = mock_text_judge.call_args.kwargs
        self.assertEqual(call_kwargs["evaluators"], evaluators)
        prompt = call_kwargs["user_prompt"]
        self.assertIn("a long document", prompt)
        self.assertIn("a faithful summary", prompt)
        # Neutral framing — not a chat transcript
        self.assertNotIn("Chat history", prompt)

    async def test_output_only_omits_input_section(self):
        mock_text_judge = AsyncMock(
            return_value={"x": {"reasoning": "ok", "match": True}}
        )
        with patch("calibrate_agent.judges.text_judge", mock_text_judge):
            await general_task_judge(
                evaluators=[
                    {"name": "x", "system_prompt": "y", "judge_model": "m"}
                ],
                output="{\"k\": 1}",
            )
        prompt = mock_text_judge.call_args.kwargs["user_prompt"]
        self.assertNotIn("`Input`", prompt)
        self.assertIn("`Output`", prompt)

    async def test_passes_fallback_model_through(self):
        mock_text_judge = AsyncMock(return_value={})
        with patch("calibrate_agent.judges.text_judge", mock_text_judge):
            await general_task_judge(
                evaluators=[{"name": "x", "system_prompt": "y"}],
                output="out",
                fallback_model="fallback-model",
            )
        self.assertEqual(
            mock_text_judge.call_args.kwargs["fallback_model"], "fallback-model"
        )

    async def test_end_to_end_with_mocked_client(self):
        client = _mock_instructor_chat_completions(
            [{"reasoning": "faithful", "match": True}]
        )
        with patch(
            "calibrate_agent.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate_agent.judges._build_openrouter_client", return_value=MagicMock()
        ):
            result = await general_task_judge(
                evaluators=[
                    render_evaluator(
                        DEFAULT_GENERAL_TASK_EVALUATOR,
                        {"criteria": "the summary is faithful to the input"},
                    )
                ],
                output="a summary",
                input_text="a document",
            )
        self.assertEqual(
            _drop_usage(result),
            {"task_quality": {"reasoning": "faithful", "match": True}},
        )


# ---------------------------------------------------------------------------
# audio_judge
# ---------------------------------------------------------------------------


class TestAudioJudge(unittest.IsolatedAsyncioTestCase):
    async def test_empty_evaluators_returns_empty_dict(self):
        result = await audio_judge(
            evaluators=[],
            audio_path="/dev/null",
            reference_text="hi",
        )
        self.assertEqual(result, {})

    async def test_builds_audio_message_per_evaluator(self):
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"FAKE_WAV_BYTES")
            audio_path = f.name

        try:
            client = _mock_instructor_chat_completions(
                [
                    {"reasoning": "clear", "match": True},
                    {"reasoning": "good", "match": True},
                ]
            )

            with patch(
                "calibrate_agent.judges.instructor.apatch", return_value=client
            ), patch(
                "calibrate_agent.judges._build_openrouter_client",
                return_value=MagicMock(),
            ):
                result = await audio_judge(
                    evaluators=[
                        {
                            "name": "intelligibility",
                            "system_prompt": "clear speech",
                            "judge_model": DEFAULT_AUDIO_JUDGE_MODEL,
                        },
                        {
                            "name": "pronunciation",
                            "system_prompt": "correct",
                            "judge_model": DEFAULT_AUDIO_JUDGE_MODEL,
                        },
                    ],
                    audio_path=audio_path,
                    reference_text="hello world",
                )

            self.assertEqual(
                _drop_usage(result),
                {
                    "intelligibility": {"reasoning": "clear", "match": True},
                    "pronunciation": {"reasoning": "good", "match": True},
                },
            )
            # One LLM call per evaluator
            self.assertEqual(client.chat.completions.create.await_count, 2)

            # First call carries the reference text and an audio block
            call_kwargs = client.chat.completions.create.call_args_list[0].kwargs
            self.assertEqual(call_kwargs["model"], DEFAULT_AUDIO_JUDGE_MODEL)
            user_msg = next(
                m for m in call_kwargs["messages"] if m["role"] == "user"
            )
            text_parts = [p for p in user_msg["content"] if p["type"] == "text"]
            self.assertTrue(any("hello world" in p["text"] for p in text_parts))
            audio_parts = [
                p for p in user_msg["content"] if p["type"] == "input_audio"
            ]
            self.assertEqual(len(audio_parts), 1)
        finally:
            os.unlink(audio_path)


# ---------------------------------------------------------------------------
# format_conversation
# ---------------------------------------------------------------------------


class TestFormatConversation(unittest.TestCase):
    def test_role_content_lines(self):
        out = format_conversation(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        )
        self.assertEqual(out, "user: Hi\nassistant: Hello!")

    def test_tool_calls_inlined(self):
        out = format_conversation(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"SF"}',
                            }
                        }
                    ],
                }
            ]
        )
        self.assertIn('[Tool Call] get_weather({"city":"SF"})', out)


# ---------------------------------------------------------------------------
# Default evaluator sanity checks
# ---------------------------------------------------------------------------


class TestDefaultEvaluators(unittest.TestCase):
    def test_llm_test_default_evaluator_shape(self):
        self.assertEqual(DEFAULT_LLM_TEST_EVALUATOR["name"], "correctness")
        self.assertIn("{{criteria}}", DEFAULT_LLM_TEST_EVALUATOR["system_prompt"])
        self.assertEqual(
            DEFAULT_LLM_TEST_EVALUATOR["judge_model"], DEFAULT_TEXT_JUDGE_MODEL
        )

    def test_general_task_default_evaluator_shape(self):
        self.assertEqual(DEFAULT_GENERAL_TASK_EVALUATOR["name"], "task_quality")
        self.assertIn(
            "{{criteria}}", DEFAULT_GENERAL_TASK_EVALUATOR["system_prompt"]
        )
        self.assertEqual(DEFAULT_GENERAL_TASK_EVALUATOR["type"], "binary")
        self.assertEqual(
            DEFAULT_GENERAL_TASK_EVALUATOR["judge_model"], DEFAULT_TEXT_JUDGE_MODEL
        )

    def test_stt_default_evaluator_shape(self):
        self.assertEqual(DEFAULT_STT_EVALUATOR["name"], "semantic_match")
        self.assertTrue(DEFAULT_STT_EVALUATOR["system_prompt"])
        self.assertEqual(
            DEFAULT_STT_EVALUATOR["judge_model"], DEFAULT_TEXT_JUDGE_MODEL
        )

    def test_tts_default_evaluator_shape(self):
        self.assertEqual(DEFAULT_TTS_EVALUATOR["name"], "pronunciation")
        self.assertTrue(DEFAULT_TTS_EVALUATOR["system_prompt"])
        self.assertEqual(
            DEFAULT_TTS_EVALUATOR["judge_model"], DEFAULT_AUDIO_JUDGE_MODEL
        )


if __name__ == "__main__":
    unittest.main()
