"""
Unit tests for calibrate/judges.py — the unified judge module.

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

import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from pydantic import BaseModel

from calibrate.judges import (
    text_judge,
    simulation_judge,
    audio_judge,
    is_rating,
    evaluator_result_value,
    render_template,
    render_evaluator,
    format_conversation,
    _result_model_for_evaluator,
    _sanitize_evaluator_for_tool_model,
    _normalize_judge_api_result,
    CriterionResult,
    DEFAULT_TEXT_JUDGE_MODEL,
    DEFAULT_AUDIO_JUDGE_MODEL,
    DEFAULT_SIMULATION_JUDGE_MODEL,
    DEFAULT_LLM_TEST_EVALUATOR,
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


class TestToolCallParamEvaluator(unittest.TestCase):
    def test_default_without_override(self):
        from calibrate.judges import (
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
        from calibrate.judges import tool_call_param_evaluator

        ev = tool_call_param_evaluator("openai/gpt-4.1")
        self.assertEqual(ev["judge_model"], "openai/gpt-4.1")

    def test_does_not_mutate_default(self):
        from calibrate.judges import (
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
        from calibrate.utils import provider_log_file

        client = _mock_instructor_chat_completions(
            [{"reasoning": "looks right", "match": True}]
        )
        f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log")
        f.close()
        token = provider_log_file.set(f.name)
        try:
            with patch(
                "calibrate.judges.instructor.apatch", return_value=client
            ), patch(
                "calibrate.judges._build_openrouter_client", return_value=MagicMock()
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
        from calibrate.utils import provider_log_file

        # Ensure unbound (default None) — judge should run without writing anywhere.
        self.assertIsNone(provider_log_file.get())
        client = _mock_instructor_chat_completions(
            [{"reasoning": "ok", "match": True}]
        )
        with patch(
            "calibrate.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate.judges._build_openrouter_client", return_value=MagicMock()
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
            "calibrate.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate.judges._build_openrouter_client", return_value=MagicMock()
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
            "calibrate.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate.judges._build_openrouter_client", return_value=MagicMock()
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
            "calibrate.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate.judges._build_openrouter_client", return_value=MagicMock()
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
            "calibrate.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate.judges._build_openrouter_client", return_value=MagicMock()
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
            "calibrate.judges.instructor.apatch", return_value=client
        ), patch(
            "calibrate.judges._build_openrouter_client", return_value=MagicMock()
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
            "calibrate.judges.instructor.apatch", return_value=client
        ), patch("calibrate.judges._build_openrouter_client", build_mock):
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

        with patch("calibrate.judges.text_judge", mock_text_judge):
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
        with patch("calibrate.judges.text_judge", mock_text_judge):
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
                "calibrate.judges.instructor.apatch", return_value=client
            ), patch(
                "calibrate.judges._build_openrouter_client",
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
