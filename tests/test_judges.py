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

import os
import tempfile
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from pydantic import BaseModel

from calibrate_agent.judges import (
    text_judge,
    simulation_judge,
    general_task_judge,
    format_task_io,
    audio_judge,
    is_rating,
    evaluator_result_value,
    evaluator_row_columns,
    evaluator_row_from_columns,
    read_existing_rows,
    stored_bool,
    stored_cell,
    stored_scores_are_comparable,
    write_evaluator_config,
    PartialResultsWriter,
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
        self.assertEqual(result, {"x": {"reasoning": "ok", "match": True}})


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
            result,
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
            result,
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
            result, {"task_quality": {"reasoning": "faithful", "match": True}}
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
                result,
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


class TestReadExistingRows(unittest.TestCase):
    def _write(self, directory, text):
        path = os.path.join(directory, "results.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_numeric_looking_ids_stay_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "id,gt,q\n1,a,True\n02,b,False\n")
            rows = read_existing_rows(path)
        self.assertEqual(set(rows), {"1", "02"})

    def test_missing_file_reads_as_no_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                read_existing_rows(os.path.join(tmp, "absent.csv")), {}
            )

    def test_rows_without_an_id_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "id,q\n,True\nrow_a,False\n")
            self.assertEqual(set(read_existing_rows(path)), {"row_a"})


class TestStoredCell(unittest.TestCase):
    def test_blank_and_missing_read_as_nothing(self):
        row = {"a": "", "b": "   ", "c": float("nan"), "d": None}
        for column in ("a", "b", "c", "d", "absent"):
            self.assertIsNone(stored_cell(row, column), column)

    def test_zero_is_a_real_value(self):
        self.assertEqual(stored_cell({"a": 0}, "a"), 0)
        self.assertEqual(stored_cell({"a": "0"}, "a"), "0")

    def test_no_row_reads_as_nothing(self):
        self.assertIsNone(stored_cell(None, "a"))


class TestStoredBool(unittest.TestCase):
    def test_recognised_values(self):
        for value in (True, "True", "true", "1", "1.0"):
            self.assertIs(stored_bool(value), True, value)
        for value in (False, "False", "false", "0", "0.0"):
            self.assertIs(stored_bool(value), False, value)

    def test_unrecognised_value(self):
        self.assertIsNone(stored_bool("maybe"))


class TestEvaluatorRowFromColumns(unittest.TestCase):
    BINARY = {"q": {"name": "q"}}
    RATING = {"r": {"name": "r", "type": "rating", "scale_min": 1, "scale_max": 5}}

    def test_round_trips_what_evaluator_row_columns_wrote(self):
        judge_row = {"q": {"match": True, "reasoning": "ok"}}
        columns = evaluator_row_columns(self.BINARY, judge_row)
        self.assertEqual(evaluator_row_from_columns(self.BINARY, columns), judge_row)

    def test_whole_number_ratings_stay_whole(self):
        rebuilt = evaluator_row_from_columns(
            self.RATING, {"r": "4.0", "r_reasoning": "good"}
        )
        self.assertEqual(rebuilt, {"r": {"score": 4, "reasoning": "good"}})
        self.assertIsInstance(rebuilt["r"]["score"], int)

    def test_a_different_evaluator_set_is_not_reused(self):
        columns = evaluator_row_columns(
            self.BINARY, {"q": {"match": True, "reasoning": "ok"}}
        )
        self.assertIsNone(
            evaluator_row_from_columns({"other": {"name": "other"}}, columns)
        )

    def test_a_blank_score_counts_as_unjudged(self):
        self.assertIsNone(
            evaluator_row_from_columns(self.BINARY, {"q": "", "q_reasoning": ""})
        )

    def test_an_unreadable_score_counts_as_unjudged(self):
        self.assertIsNone(
            evaluator_row_from_columns(self.RATING, {"r": "high", "r_reasoning": ""})
        )
        self.assertIsNone(
            evaluator_row_from_columns(self.BINARY, {"q": "maybe", "q_reasoning": ""})
        )

    def test_missing_reasoning_reads_as_empty(self):
        self.assertEqual(
            evaluator_row_from_columns(self.BINARY, {"q": "True"}),
            {"q": {"match": True, "reasoning": ""}},
        )


class TestStoredScoresAreComparable(unittest.TestCase):
    BINARY = {"name": "q", "system_prompt": "p"}
    RATING = {
        "name": "q",
        "system_prompt": "p",
        "type": "rating",
        "scale_min": 1,
        "scale_max": 5,
    }

    def _run_dir(self, tmp, evaluators):
        write_evaluator_config(tmp, evaluators)
        return tmp

    def test_an_unchanged_evaluator_is_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_dir(tmp, [self.BINARY])
            self.assertTrue(stored_scores_are_comparable(tmp, [self.BINARY]))

    def test_a_reworded_prompt_is_still_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_dir(tmp, [self.BINARY])
            reworded = {**self.BINARY, "system_prompt": "something else"}
            self.assertTrue(stored_scores_are_comparable(tmp, [reworded]))

    def test_a_different_judge_model_is_still_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_dir(tmp, [{**self.BINARY, "judge_model": "a"}])
            self.assertTrue(
                stored_scores_are_comparable(tmp, [{**self.BINARY, "judge_model": "b"}])
            )

    def test_pass_fail_turned_into_a_rating_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_dir(tmp, [self.BINARY])
            self.assertFalse(stored_scores_are_comparable(tmp, [self.RATING]))

    def test_a_widened_rating_range_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_dir(tmp, [self.RATING])
            widened = {**self.RATING, "scale_max": 10}
            self.assertFalse(stored_scores_are_comparable(tmp, [widened]))

    def test_an_evaluator_the_prior_run_did_not_have_is_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_dir(tmp, [self.BINARY])
            added = {"name": "extra", "system_prompt": "p"}
            self.assertTrue(stored_scores_are_comparable(tmp, [self.BINARY, added]))

    def test_no_prior_config_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(stored_scores_are_comparable(tmp, [self.BINARY]))

    def test_an_unreadable_config_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                f.write("{not json")
            self.assertFalse(stored_scores_are_comparable(tmp, [self.BINARY]))


class TestPartialResultsWriter(unittest.TestCase):
    EVALUATORS = {"q": {"name": "q"}}
    BASE = [{"id": "1", "gt": "a"}, {"id": "row_b", "gt": "b"}]

    def _judged(self, match, reasoning):
        return evaluator_row_columns(
            self.EVALUATORS, {"q": {"match": match, "reasoning": reasoning}}
        )

    def test_only_judged_rows_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.csv")
            writer = PartialResultsWriter(path, self.BASE)
            writer.update(1, self._judged(True, "ok"))
            rows = read_existing_rows(path)
            self.assertEqual(set(rows), {"row_b"})
            writer.update(0, self._judged(False, "no"))
            self.assertEqual(set(read_existing_rows(path)), {"1", "row_b"})

    def test_rows_are_written_in_dataset_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.csv")
            writer = PartialResultsWriter(path, self.BASE)
            writer.update(1, self._judged(True, "ok"))
            writer.update(0, self._judged(False, "no"))
            self.assertEqual(list(read_existing_rows(path)), ["1", "row_b"])

    def test_several_judges_fill_one_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.csv")
            writer = PartialResultsWriter(path, self.BASE)
            writer.update(0, {"semantic_wer": 0.1})
            writer.update(0, self._judged(True, "ok"))
            row = read_existing_rows(path)["1"]
            self.assertEqual(row["semantic_wer"], "0.1")
            self.assertEqual(row["q"], "True")

    def test_a_resumed_run_keeps_the_rows_it_started_with(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.csv")
            first = PartialResultsWriter(path, self.BASE)
            first.update(0, self._judged(False, "no"))

            second = PartialResultsWriter(path, self.BASE, read_existing_rows(path))
            second.update(1, self._judged(True, "ok"))

            rows = read_existing_rows(path)
            self.assertEqual(set(rows), {"1", "row_b"})
            self.assertEqual(rows["1"]["q"], "False")
            self.assertEqual(rows["1"]["q_reasoning"], "no")

    def test_a_row_not_in_the_dataset_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.csv")
            writer = PartialResultsWriter(
                path, self.BASE, {"ghost": {"id": "ghost", "q": "True"}}
            )
            writer.update(0, self._judged(True, "ok"))
            self.assertEqual(set(read_existing_rows(path)), {"1"})

    def test_the_dataset_owns_its_own_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.csv")
            stale = {"1": {"id": "1", "gt": "stale", "q": "True"}}
            writer = PartialResultsWriter(path, self.BASE, stale)
            writer.update(1, self._judged(True, "ok"))
            self.assertEqual(read_existing_rows(path)["1"]["gt"], "a")

    def test_a_blank_stored_value_is_not_carried_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.csv")
            stale = {"1": {"id": "1", "gt": "a", "q": "", "q_reasoning": ""}}
            writer = PartialResultsWriter(path, self.BASE, stale)
            writer.update(1, self._judged(True, "ok"))
            self.assertIsNone(stored_cell(read_existing_rows(path)["1"], "q"))

    def test_nothing_is_written_before_the_first_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.csv")
            PartialResultsWriter(path, self.BASE)
            self.assertFalse(os.path.exists(path))

    def test_a_write_failure_does_not_take_down_the_run(self):
        writer = PartialResultsWriter(
            os.path.join("no", "such", "dir", "results.csv"), self.BASE
        )
        writer.update(0, self._judged(True, "ok"))


if __name__ == "__main__":
    unittest.main()
