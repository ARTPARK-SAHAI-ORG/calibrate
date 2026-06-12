"""Extra coverage for calibrate/llm/run_tests.py — helpers, aggregation, eval-only flow."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


def _bin_ev(name):
    return {"name": name, "system_prompt": "x", "judge_model": "openai/gpt-4.1"}


def _rate_ev(name, lo=1, hi=5):
    return {
        "name": name, "system_prompt": "x", "judge_model": "openai/gpt-4.1",
        "type": "rating", "scale_min": lo, "scale_max": hi,
    }


class TestNormalizeCriteriaRefs(unittest.TestCase):
    def test_str_criteria_normalizes(self):
        from calibrate.llm.run_tests import _normalize_criteria_refs

        result = _normalize_criteria_refs("be helpful")
        self.assertEqual(len(result), 1)
        self.assertIn("arguments", result[0])
        self.assertEqual(result[0]["arguments"]["criteria"], "be helpful")

    def test_list_criteria_passes_through(self):
        from calibrate.llm.run_tests import _normalize_criteria_refs

        result = _normalize_criteria_refs([{"name": "a"}, {"name": "b"}])
        self.assertEqual(len(result), 2)

    def test_invalid_list_item_raises(self):
        from calibrate.llm.run_tests import _normalize_criteria_refs

        with self.assertRaises(ValueError):
            _normalize_criteria_refs([{"no_name": True}])

    def test_invalid_type_raises(self):
        from calibrate.llm.run_tests import _normalize_criteria_refs

        with self.assertRaises(ValueError):
            _normalize_criteria_refs(42)


class TestGetNameToEvaluatorDict(unittest.TestCase):
    def test_default_registry(self):
        from calibrate.llm.run_tests import _get_name_to_evaluator_dict
        from calibrate.judges import DEFAULT_LLM_TEST_EVALUATOR

        reg = _get_name_to_evaluator_dict({"evaluators": []})
        self.assertIn(DEFAULT_LLM_TEST_EVALUATOR["name"], reg)
        self.assertIn("default", reg)

    def test_user_evaluator_override(self):
        from calibrate.llm.run_tests import _get_name_to_evaluator_dict

        reg = _get_name_to_evaluator_dict({"evaluators": [_bin_ev("custom")]})
        self.assertIn("custom", reg)

    def test_default_alias_conflict_raises(self):
        from calibrate.llm.run_tests import _get_name_to_evaluator_dict
        from calibrate.judges import DEFAULT_LLM_TEST_EVALUATOR

        with self.assertRaises(ValueError):
            _get_name_to_evaluator_dict({
                "evaluators": [
                    _bin_ev("default"),
                    _bin_ev(DEFAULT_LLM_TEST_EVALUATOR["name"]),
                ]
            })

    def test_missing_required_field_raises(self):
        from calibrate.llm.run_tests import _get_name_to_evaluator_dict

        with self.assertRaises(ValueError):
            _get_name_to_evaluator_dict({"evaluators": [{"name": "noprompt"}]})


class TestResolveEvaluatorsForTestCase(unittest.TestCase):
    def test_unknown_evaluator_raises(self):
        from calibrate.llm.run_tests import (
            _get_name_to_evaluator_dict,
            _resolve_evaluators_for_test_case,
        )

        with self.assertRaises(ValueError):
            _resolve_evaluators_for_test_case(
                {"type": "response", "criteria": [{"name": "noexist"}]},
                _get_name_to_evaluator_dict({"evaluators": []}),
            )

    def test_resolves_with_template(self):
        from calibrate.llm.run_tests import (
            _get_name_to_evaluator_dict,
            _resolve_evaluators_for_test_case,
        )

        result = _resolve_evaluators_for_test_case(
            {
                "type": "response",
                "criteria": [{"name": "x", "arguments": {"criteria": "be polite"}}],
            },
            _get_name_to_evaluator_dict({
                "evaluators": [
                    {"name": "x", "system_prompt": "Check {{criteria}}", "judge_model": "m"}
                ]
            }),
        )
        self.assertEqual(result[0]["system_prompt"], "Check be polite")

    def test_response_uses_implicit_default(self):
        from calibrate.llm.run_tests import (
            _get_name_to_evaluator_dict,
            _resolve_evaluators_for_test_case,
        )

        resolved = _resolve_evaluators_for_test_case(
            {"type": "response", "criteria": "be nice"},
            _get_name_to_evaluator_dict({"evaluators": []}),
        )
        self.assertEqual(resolved[0]["name"], "correctness")

    def test_conversation_uses_user_evaluator(self):
        from calibrate.llm.run_tests import (
            _get_name_to_evaluator_dict,
            _resolve_evaluators_for_test_case,
        )

        resolved = _resolve_evaluators_for_test_case(
            {"type": "conversation", "criteria": [{"name": "tone"}]},
            _get_name_to_evaluator_dict(
                {"evaluators": [_bin_ev("tone")]}, include_default=False
            ),
        )
        self.assertEqual(resolved[0]["name"], "tone")

    def test_conversation_rejects_implicit_default(self):
        from calibrate.llm.run_tests import (
            _get_name_to_evaluator_dict,
            _resolve_evaluators_for_test_case,
        )

        with self.assertRaises(ValueError):
            _resolve_evaluators_for_test_case(
                {"type": "conversation", "criteria": "be nice"},
                _get_name_to_evaluator_dict(
                    {"evaluators": [_bin_ev("tone")]}, include_default=False
                ),
            )


class TestDisplayLabel(unittest.TestCase):
    def test_openrouter_strips_provider(self):
        from calibrate.llm.run_tests import display_label

        self.assertEqual(display_label("openrouter", "openai/gpt-4.1"), "openai/gpt-4.1")

    def test_other_provider(self):
        from calibrate.llm.run_tests import display_label

        self.assertEqual(display_label("openai", "gpt-4.1"), "openai/gpt-4.1")


class TestSortAndWebhookTools(unittest.TestCase):
    def test_sort_tool_calls(self):
        from calibrate.llm.run_tests import sort_tool_calls

        result = sort_tool_calls([{"tool": "b"}, {"tool": "a"}])
        self.assertEqual([t["tool"] for t in result], ["a", "b"])

    def test_get_webhook_tool_names(self):
        from calibrate.llm.run_tests import get_webhook_tool_names

        names = get_webhook_tool_names([
            {"name": "fn1", "type": "webhook"},
            {"name": "fn2", "type": "structured"},
        ])
        self.assertEqual(names, {"fn1"})


class TestPreprocessConversationHistory(unittest.TestCase):
    def test_injects_tool_response_for_structured(self):
        from calibrate.llm.run_tests import preprocess_conversation_history

        tools = [{"name": "fn1", "type": "structured"}]
        history = [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "fn1", "arguments": "{}"},
                }],
            },
        ]
        result = preprocess_conversation_history(history, tools)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[2]["role"], "tool")
        self.assertEqual(result[2]["tool_call_id"], "call_1")

    def test_keeps_existing_response(self):
        from calibrate.llm.run_tests import preprocess_conversation_history

        tools = [{"name": "fn1", "type": "structured"}]
        history = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "fn1", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "real"},
        ]
        result = preprocess_conversation_history(history, tools)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["content"], "real")

    def test_skip_webhook(self):
        from calibrate.llm.run_tests import preprocess_conversation_history

        tools = [{"name": "fn1", "type": "webhook"}]
        history = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "fn1", "arguments": "{}"},
                }],
            },
        ]
        result = preprocess_conversation_history(history, tools)
        # Webhook tool — no injection
        self.assertEqual(len(result), 1)


class TestToolCallPairs(unittest.TestCase):
    def test_pair_mismatch_wrong_tool(self):
        from calibrate.llm.run_tests import _tool_call_pair_mismatch

        reason = _tool_call_pair_mismatch(
            {"tool": "a", "arguments": {}},
            {"tool": "b", "arguments": {}},
        )
        self.assertIn("Tool call mismatch", reason)

    def test_pair_no_arguments_in_expected(self):
        from calibrate.llm.run_tests import _tool_call_pair_mismatch

        # Expected lacks 'arguments' → don't check args
        self.assertIsNone(_tool_call_pair_mismatch(
            {"tool": "a", "arguments": {"x": 1}},
            {"tool": "a"},
        ))

    def test_pair_args_mismatch(self):
        from calibrate.llm.run_tests import _tool_call_pair_mismatch

        reason = _tool_call_pair_mismatch(
            {"tool": "a", "arguments": {"x": 1}},
            {"tool": "a", "arguments": {"x": 2}},
        )
        self.assertIn("arguments mismatch", reason)
        self.assertIn("x:", reason)
        self.assertIn("value mismatch", reason)

    def test_pair_args_mismatch_type_only(self):
        from calibrate.llm.run_tests import _tool_call_pair_mismatch

        reason = _tool_call_pair_mismatch(
            {"tool": "a", "arguments": {"phone_number": 9811123401}},
            {"tool": "a", "arguments": {"phone_number": "9811123401"}},
        )
        self.assertIn("phone_number", reason)
        self.assertIn("type mismatch", reason)
        self.assertIn("same string form", reason)

    def test_pair_args_mismatch_multiple_keys(self):
        from calibrate.llm.run_tests import _tool_call_pair_mismatch

        reason = _tool_call_pair_mismatch(
            {
                "tool": "fill_form",
                "arguments": {"a": 1, "b": "ok", "c": None},
            },
            {
                "tool": "fill_form",
                "arguments": {"a": 2, "b": "no", "c": 0},
            },
        )
        self.assertIn("a:", reason)
        self.assertIn("b:", reason)
        self.assertIn("c:", reason)

    def test_pair_args_mismatch_extra_and_missing_keys(self):
        from calibrate.llm.run_tests import _tool_call_pair_mismatch

        reason = _tool_call_pair_mismatch(
            {"tool": "a", "arguments": {"x": 1, "extra": 9}},
            {"tool": "a", "arguments": {"x": 1, "y": 2}},
        )
        self.assertIn("extra", reason)
        self.assertIn("unexpected key", reason)
        self.assertIn("y", reason)
        self.assertIn("missing in actual", reason)

    def test_pair_args_mismatch_mixed_key_types_no_crash(self):
        from calibrate.llm.run_tests import _tool_call_arguments_diff_lines

        lines = _tool_call_arguments_diff_lines(
            {1: "a", "b": 2},
            {1: "x", "b": 2},
        )
        self.assertTrue(any("1" in ln for ln in lines))
        self.assertEqual(len(lines), 1)

    def test_pair_none_args_match(self):
        from calibrate.llm.run_tests import _tool_call_pair_mismatch

        self.assertIsNone(_tool_call_pair_mismatch(
            {"tool": "a", "arguments": {"x": 1}},
            {"tool": "a", "arguments": None},
        ))

    def test_evaluate_tool_calls_empty_output(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        result = asyncio.run(evaluate_tool_calls([], [{"tool": "a"}]))
        self.assertFalse(result["passed"])
        self.assertEqual(result["tool_call_results"], [{"tool": "a", "passed": False}])

    def test_evaluate_tool_calls_pass(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        result = asyncio.run(evaluate_tool_calls(
            [{"tool": "a", "arguments": {}}],
            [{"tool": "a", "arguments": {}}],
        ))
        self.assertTrue(result["passed"])
        self.assertEqual(result["tool_call_results"], [{"tool": "a", "passed": True}])

    def test_per_slot_passes_empty_expected(self):
        from calibrate.llm.run_tests import _per_slot_tool_passes

        self.assertEqual(_per_slot_tool_passes([], []), [])

    def test_per_slot_passes_no_output(self):
        from calibrate.llm.run_tests import _per_slot_tool_passes

        result = _per_slot_tool_passes([], [{"tool": "a"}, {"tool": "b"}])
        self.assertEqual(result, [("a", False), ("b", False)])

    def test_per_slot_passes_partial(self):
        from calibrate.llm.run_tests import _per_slot_tool_passes

        result = _per_slot_tool_passes(
            [{"tool": "a", "arguments": {}}],
            [{"tool": "a", "arguments": {}}, {"tool": "b", "arguments": {}}],
        )
        # First slot passes (alphabetical "a" matches), second slot fails (no output for b)
        self.assertEqual(result, [("a", True), ("b", False)])


class TestParamCriteriaSpec(unittest.TestCase):
    def test_literal_returns_none(self):
        from calibrate.llm.run_tests import _param_criteria_spec

        self.assertIsNone(_param_criteria_spec("hello", "x"))
        self.assertIsNone(_param_criteria_spec({"nested": 1}, "x"))
        self.assertIsNone(_param_criteria_spec(5, "x"))

    def test_llm_judge_spec(self):
        from calibrate.llm.run_tests import _param_criteria_spec

        spec = _param_criteria_spec(
            {"match_type": "llm_judge", "criteria": "a polite greeting",
             "judge_model": "openai/gpt-4.1"},
            "msg",
        )
        self.assertEqual(spec["match_type"], "llm_judge")
        self.assertEqual(spec["criteria"], "a polite greeting")
        self.assertEqual(spec["judge_model"], "openai/gpt-4.1")

    def test_llm_judge_requires_criteria(self):
        from calibrate.llm.run_tests import _param_criteria_spec

        with self.assertRaises(ValueError):
            _param_criteria_spec({"match_type": "llm_judge"}, "msg")
        with self.assertRaises(ValueError):
            _param_criteria_spec({"match_type": "llm_judge", "criteria": "  "}, "msg")

    def test_exact_spec(self):
        from calibrate.llm.run_tests import _param_criteria_spec

        spec = _param_criteria_spec({"match_type": "exact", "value": 42}, "x")
        self.assertEqual(spec, {"match_type": "exact", "value": 42})

    def test_exact_requires_value(self):
        from calibrate.llm.run_tests import _param_criteria_spec

        with self.assertRaises(ValueError):
            _param_criteria_spec({"match_type": "exact"}, "x")

    def test_unknown_match_type(self):
        from calibrate.llm.run_tests import _param_criteria_spec

        with self.assertRaises(ValueError):
            _param_criteria_spec({"match_type": "fuzzy", "criteria": "x"}, "x")


class TestEvaluateToolCallsCriteria(unittest.TestCase):
    @staticmethod
    def _judge_returning(match, reasoning="because"):
        async def fake_text_judge(evaluators, user_prompt, *a, **k):
            name = evaluators[0]["name"]
            return {name: {"match": match, "reasoning": reasoning}}

        return fake_text_judge

    def test_llm_judge_param_pass(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch(
            "calibrate.llm.run_tests.text_judge",
            side_effect=self._judge_returning(True),
        ) as mock_judge:
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "send_sms", "arguments": {"message": "Hi there, welcome!"}}],
                [{"tool": "send_sms", "arguments": {
                    "message": {"match_type": "llm_judge",
                                "criteria": "a friendly greeting"}}}],
            ))
        self.assertTrue(result["passed"])
        # A passing llm_judge parameter now retains its verdict + reasoning.
        self.assertEqual(
            result["tool_call_results"],
            [
                {
                    "tool": "send_sms",
                    "passed": True,
                    "param_judgments": [
                        {
                            "param": "message",
                            "match_type": "llm_judge",
                            "criteria": "a friendly greeting",
                            "match": True,
                            "reasoning": "because",
                        }
                    ],
                }
            ],
        )
        # The judged parameter's reasoning is consolidated into the overall
        # reasoning rather than discarded.
        self.assertIn("message", result["reasoning"])
        self.assertIn("criteria met", result["reasoning"])
        self.assertIn("because", result["reasoning"])
        # The judge prompt should mention the argument name and actual value.
        prompt = mock_judge.call_args.kwargs.get("user_prompt") or mock_judge.call_args.args[1]
        self.assertIn("send_sms", prompt)
        self.assertIn("message", prompt)
        self.assertIn("Hi there, welcome!", prompt)

    def test_llm_judge_param_fail(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch(
            "calibrate.llm.run_tests.text_judge",
            side_effect=self._judge_returning(False, "not a greeting"),
        ):
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "send_sms", "arguments": {"message": "Your code is 1234"}}],
                [{"tool": "send_sms", "arguments": {
                    "message": {"match_type": "llm_judge",
                                "criteria": "a friendly greeting"}}}],
            ))
        self.assertFalse(result["passed"])
        self.assertIn("message", result["reasoning"])
        self.assertIn("not a greeting", result["reasoning"])
        # A failing llm_judge parameter is captured too.
        self.assertEqual(
            result["tool_call_results"],
            [
                {
                    "tool": "send_sms",
                    "passed": False,
                    "param_judgments": [
                        {
                            "param": "message",
                            "match_type": "llm_judge",
                            "criteria": "a friendly greeting",
                            "match": False,
                            "reasoning": "not a greeting",
                        }
                    ],
                }
            ],
        )

    def test_llm_judge_not_invoked_for_exact_params(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch("calibrate.llm.run_tests.text_judge") as mock_judge:
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "a", "arguments": {"x": 1, "y": 2}}],
                [{"tool": "a", "arguments": {"x": 1, "y": 2}}],
            ))
        self.assertTrue(result["passed"])
        mock_judge.assert_not_called()

    def test_exact_spec_matches_literal_value(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch("calibrate.llm.run_tests.text_judge") as mock_judge:
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "a", "arguments": {"x": {"match_type": "exact", "value": 1}}}],
                [{"tool": "a", "arguments": {
                    "x": {"match_type": "exact",
                          "value": {"match_type": "exact", "value": 1}}}}],
            ))
        self.assertTrue(result["passed"])
        mock_judge.assert_not_called()

    def test_llm_judge_param_missing_in_output(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch("calibrate.llm.run_tests.text_judge") as mock_judge:
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "a", "arguments": {}}],
                [{"tool": "a", "arguments": {
                    "msg": {"match_type": "llm_judge", "criteria": "a greeting"}}}],
            ))
        self.assertFalse(result["passed"])
        self.assertIn("missing in actual", result["reasoning"])
        mock_judge.assert_not_called()

    def test_mixed_literal_and_judge_params(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch(
            "calibrate.llm.run_tests.text_judge",
            side_effect=self._judge_returning(True),
        ):
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "a", "arguments": {"id": 7, "note": "looks good"}}],
                [{"tool": "a", "arguments": {
                    "id": 7,
                    "note": {"match_type": "llm_judge", "criteria": "positive"}}}],
            ))
        self.assertTrue(result["passed"])

    def test_literal_mismatch_alongside_passing_judge(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch(
            "calibrate.llm.run_tests.text_judge",
            side_effect=self._judge_returning(True),
        ):
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "a", "arguments": {"id": 9, "note": "looks good"}}],
                [{"tool": "a", "arguments": {
                    "id": 7,
                    "note": {"match_type": "llm_judge", "criteria": "positive"}}}],
            ))
        self.assertFalse(result["passed"])
        self.assertIn("id", result["reasoning"])
        # Both params are captured on the failing slot: the failing exact one
        # and the passing judged one.
        self.assertEqual(
            result["tool_call_results"][0]["param_judgments"],
            [
                {
                    "param": "id",
                    "match_type": "exact",
                    "match": False,
                    "reasoning": "value mismatch — expected 7, got 9",
                },
                {
                    "param": "note",
                    "match_type": "llm_judge",
                    "criteria": "positive",
                    "match": True,
                    "reasoning": "because",
                },
            ],
        )

    def test_exact_only_pass_uses_flat_message(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch("calibrate.llm.run_tests.text_judge") as mock_judge:
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "a", "arguments": {"x": 1}}],
                [{"tool": "a", "arguments": {"x": 1}}],
            ))
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["reasoning"],
            "The agent's tools calls matches the expected tool calls",
        )
        # No judged params → no param_judgments key on the slot.
        self.assertEqual(result["tool_call_results"], [{"tool": "a", "passed": True}])
        mock_judge.assert_not_called()

    def test_pass_reasoning_consolidates_judged_params(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch(
            "calibrate.llm.run_tests.text_judge",
            side_effect=self._judge_returning(True, "reads as friendly"),
        ):
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "send_sms", "arguments": {"id": 7, "message": "Hi!"}}],
                [{"tool": "send_sms", "arguments": {
                    "id": 7,
                    "message": {"match_type": "llm_judge",
                                "criteria": "a friendly greeting"}}}],
            ))
        self.assertTrue(result["passed"])
        # Consolidated: base sentence, the exact-matched param line, and the
        # judged parameter's verdict/reasoning.
        self.assertIn(
            "The agent's tools calls matches the expected tool calls",
            result["reasoning"],
        )
        self.assertIn("id: value matches the expected value", result["reasoning"])
        self.assertIn("message: criteria met", result["reasoning"])
        self.assertIn("reads as friendly", result["reasoning"])


class TestAnyWildcardParam(unittest.TestCase):
    """A ``{"match_type": "any"}`` expected value is a wildcard — param ignored."""

    ANY = {"match_type": "any"}

    def test_is_any_spec(self):
        from calibrate.llm.run_tests import _is_any_spec

        self.assertTrue(_is_any_spec({"match_type": "any"}))
        # A plain "any" string is an ordinary literal, not the wildcard.
        self.assertFalse(_is_any_spec("any"))
        self.assertFalse(_is_any_spec("anything"))
        self.assertFalse(_is_any_spec(None))
        self.assertFalse(_is_any_spec({"match_type": "exact", "value": "any"}))
        self.assertFalse(_is_any_spec({"match_type": "llm_judge", "criteria": "x"}))

    def test_wildcard_absent_from_output_passes(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch("calibrate.llm.run_tests.text_judge") as mock_judge:
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "lookup", "arguments": {"city": "London"}}],
                [{"tool": "lookup", "arguments": {"city": "London", "token": self.ANY}}],
            ))
        self.assertTrue(result["passed"])
        self.assertEqual(result["tool_call_results"], [{"tool": "lookup", "passed": True}])
        mock_judge.assert_not_called()

    def test_wildcard_present_with_arbitrary_value_passes(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch("calibrate.llm.run_tests.text_judge") as mock_judge:
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "lookup",
                  "arguments": {"city": "London", "token": "xyz-123"}}],
                [{"tool": "lookup", "arguments": {"city": "London", "token": self.ANY}}],
            ))
        self.assertTrue(result["passed"])
        self.assertEqual(result["tool_call_results"], [{"tool": "lookup", "passed": True}])
        mock_judge.assert_not_called()

    def test_wildcard_does_not_mask_sibling_failure(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch("calibrate.llm.run_tests.text_judge") as mock_judge:
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "lookup",
                  "arguments": {"city": "Paris", "token": "whatever"}}],
                [{"tool": "lookup", "arguments": {"city": "London", "token": self.ANY}}],
            ))
        self.assertFalse(result["passed"])
        self.assertIn("city", result["reasoning"])
        # The wildcard sibling must not appear in the failure reasoning.
        self.assertNotIn("token", result["reasoning"])
        mock_judge.assert_not_called()

    def test_nested_wildcard_passes(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch("calibrate.llm.run_tests.text_judge") as mock_judge:
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "book",
                  "arguments": {"patient": {"name": "John Doe", "session": "s-999"}}}],
                [{"tool": "book",
                  "arguments": {"patient": {"name": "John Doe", "session": self.ANY}}}],
            ))
        self.assertTrue(result["passed"])
        # Also passes when the wildcard sub-key is absent entirely.
        with patch("calibrate.llm.run_tests.text_judge"):
            result_absent = asyncio.run(evaluate_tool_calls(
                [{"tool": "book", "arguments": {"patient": {"name": "John Doe"}}}],
                [{"tool": "book",
                  "arguments": {"patient": {"name": "John Doe", "session": self.ANY}}}],
            ))
        self.assertTrue(result_absent["passed"])
        mock_judge.assert_not_called()

    def test_plain_any_string_is_an_exact_literal(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        # A bare "any" string is NOT the wildcard — it must match exactly.
        with patch("calibrate.llm.run_tests.text_judge") as mock_judge:
            passing = asyncio.run(evaluate_tool_calls(
                [{"tool": "a", "arguments": {"x": "any"}}],
                [{"tool": "a", "arguments": {"x": "any"}}],
            ))
            failing = asyncio.run(evaluate_tool_calls(
                [{"tool": "a", "arguments": {"x": "other"}}],
                [{"tool": "a", "arguments": {"x": "any"}}],
            ))
        self.assertTrue(passing["passed"])
        self.assertFalse(failing["passed"])
        self.assertIn("x", failing["reasoning"])
        mock_judge.assert_not_called()

    def test_wildcard_contributes_no_param_record(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        # Pair a wildcard param with an llm_judge param so the slot carries a
        # param_judgments breakdown; the wildcard must be absent from it.
        async def fake_text_judge(evaluators, user_prompt, *a, **k):
            return {evaluators[0]["name"]: {"match": True, "reasoning": "ok"}}

        with patch("calibrate.llm.run_tests.text_judge", side_effect=fake_text_judge):
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "a",
                  "arguments": {"note": "looks good", "token": "anything-goes"}}],
                [{"tool": "a", "arguments": {
                    "note": {"match_type": "llm_judge", "criteria": "positive"},
                    "token": self.ANY}}],
            ))
        self.assertTrue(result["passed"])
        judgments = result["tool_call_results"][0]["param_judgments"]
        params = [r["param"] for r in judgments]
        self.assertNotIn("token", params)
        self.assertEqual(params, ["note"])

    def test_sync_pair_mismatch_wildcard_absent_matches(self):
        from calibrate.llm.run_tests import _tool_call_pair_mismatch

        # Parity contract: when the only difference is an absent wildcard param,
        # the sync matcher (used by _per_slot_tool_passes / the leaderboard)
        # agrees with the async path and reports a match.
        self.assertIsNone(_tool_call_pair_mismatch(
            {"tool": "lookup", "arguments": {"city": "London"}},
            {"tool": "lookup", "arguments": {"city": "London", "token": self.ANY}},
        ))

    def test_sync_pair_mismatch_wildcard_present_matches(self):
        from calibrate.llm.run_tests import _tool_call_pair_mismatch

        # Parity contract: when the ONLY difference is a wildcard param carrying
        # an arbitrary value, the sync matcher agrees with the async
        # evaluate_tool_calls and reports a match.
        self.assertIsNone(_tool_call_pair_mismatch(
            {"tool": "lookup", "arguments": {"city": "London", "token": "xyz-123"}},
            {"tool": "lookup", "arguments": {"city": "London", "token": self.ANY}},
        ))


class TestArgumentLevelTicksAndCrosses(unittest.TestCase):
    @staticmethod
    def _judge_returning(match, reasoning="because"):
        async def fake_text_judge(evaluators, user_prompt, *a, **k):
            return {evaluators[0]["name"]: {"match": match, "reasoning": reasoning}}
        return fake_text_judge

    def test_failures_listed_before_passes_with_emoji_prefixes(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch(
            "calibrate.llm.run_tests.text_judge",
            side_effect=self._judge_returning(True, "ok"),
        ):
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "a", "arguments": {"id": 9, "ok": 1, "note": "g"}}],
                [{"tool": "a", "arguments": {
                    "id": 7,
                    "ok": 1,
                    "note": {"match_type": "llm_judge", "criteria": "positive"}}}],
            ))
        self.assertFalse(result["passed"])
        reasoning = result["reasoning"]
        self.assertIn("❌ id:", reasoning)
        self.assertIn("✅ ok:", reasoning)
        self.assertIn("✅ note:", reasoning)
        # All ❌ lines must appear before any ✅ line.
        first_pass = reasoning.find("✅")
        last_fail = reasoning.rfind("❌")
        self.assertGreater(first_pass, last_fail)

    def test_exact_only_failure_includes_passing_args_with_tick(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        result = asyncio.run(evaluate_tool_calls(
            [{"tool": "a", "arguments": {"x": 1, "y": 9}}],
            [{"tool": "a", "arguments": {"x": 1, "y": 2}}],
        ))
        self.assertFalse(result["passed"])
        reasoning = result["reasoning"]
        self.assertIn("❌ y:", reasoning)
        self.assertIn("✅ x:", reasoning)
        self.assertLess(reasoning.find("❌ y:"), reasoning.find("✅ x:"))

    def test_pass_with_judge_uses_tick_prefix(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch(
            "calibrate.llm.run_tests.text_judge",
            side_effect=self._judge_returning(True, "reads as friendly"),
        ):
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "send_sms", "arguments": {"id": 7, "message": "Hi!"}}],
                [{"tool": "send_sms", "arguments": {
                    "id": 7,
                    "message": {"match_type": "llm_judge",
                                "criteria": "a friendly greeting"}}}],
            ))
        self.assertTrue(result["passed"])
        self.assertIn("✅ id:", result["reasoning"])
        self.assertIn("✅ message:", result["reasoning"])
        # No failure markers when everything passes.
        self.assertNotIn("❌", result["reasoning"])


class TestToolCallAsyncMatchers(unittest.TestCase):
    def test_pair_async_wrong_tool(self):
        from calibrate.llm.run_tests import _tool_call_pair_mismatch_async

        res = asyncio.run(_tool_call_pair_mismatch_async(
            {"tool": "a", "arguments": {}},
            {"tool": "b", "arguments": {}},
        ))
        self.assertIn("Tool call mismatch", res["mismatch"])
        self.assertEqual(res["records"], [])
        self.assertFalse(res["had_llm"])

    def test_pair_async_no_arguments_key(self):
        from calibrate.llm.run_tests import _tool_call_pair_mismatch_async

        self.assertEqual(asyncio.run(_tool_call_pair_mismatch_async(
            {"tool": "a", "arguments": {"x": 1}},
            {"tool": "a"},
        )), {"mismatch": None, "records": [], "had_llm": False})

    def test_pair_async_none_arguments(self):
        from calibrate.llm.run_tests import _tool_call_pair_mismatch_async

        self.assertEqual(asyncio.run(_tool_call_pair_mismatch_async(
            {"tool": "a", "arguments": {"x": 1}},
            {"tool": "a", "arguments": None},
        )), {"mismatch": None, "records": [], "had_llm": False})

    def test_message_async_expected_non_dict(self):
        from calibrate.llm.run_tests import _tool_call_arguments_eval_async

        res = asyncio.run(_tool_call_arguments_eval_async(
            "a", "not-a-dict", {"x": 1},
        ))
        self.assertIn("cannot diff", res["message"])
        self.assertEqual(res["records"], [])
        self.assertFalse(res["had_llm"])

    def test_message_async_actual_non_dict(self):
        from calibrate.llm.run_tests import _tool_call_arguments_eval_async

        res = asyncio.run(_tool_call_arguments_eval_async(
            "a", {"x": 1}, "not-a-dict",
        ))
        self.assertIn("expected dict", res["message"])
        self.assertEqual(res["records"], [])

    def test_evaluate_fewer_output_than_expected(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        # Output has only "a"; expected wants "a" and "b". Per zip-min the case
        # still passes, but the unmatched "b" slot is recorded as failed.
        result = asyncio.run(evaluate_tool_calls(
            [{"tool": "a", "arguments": {}}],
            [{"tool": "a", "arguments": {}}, {"tool": "b", "arguments": {}}],
        ))
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["tool_call_results"],
            [{"tool": "a", "passed": True}, {"tool": "b", "passed": False}],
        )

    def test_judge_parameter_uses_judge_model_and_value(self):
        from calibrate.llm.run_tests import _judge_tool_call_parameter

        captured = {}

        async def fake_text_judge(evaluators, user_prompt, *a, **k):
            captured["judge_model"] = evaluators[0]["judge_model"]
            captured["prompt"] = user_prompt
            return {evaluators[0]["name"]: {"match": True, "reasoning": "ok"}}

        with patch("calibrate.llm.run_tests.text_judge", side_effect=fake_text_judge):
            res = asyncio.run(_judge_tool_call_parameter(
                "send_sms",
                "message",
                {"match_type": "llm_judge", "criteria": "polite",
                 "judge_model": "openai/gpt-4.1"},
                "Hello!",
            ))
        self.assertTrue(res["match"])
        self.assertEqual(captured["judge_model"], "openai/gpt-4.1")
        self.assertIn("Hello!", captured["prompt"])

    def test_judge_parameter_non_serializable_value(self):
        from calibrate.llm.run_tests import _judge_tool_call_parameter

        async def fake_text_judge(evaluators, user_prompt, *a, **k):
            return {evaluators[0]["name"]: {"match": True, "reasoning": "ok"}}

        circular = []
        circular.append(circular)  # json.dumps raises -> repr() fallback

        with patch("calibrate.llm.run_tests.text_judge", side_effect=fake_text_judge):
            res = asyncio.run(_judge_tool_call_parameter(
                "tool", "param",
                {"match_type": "llm_judge", "criteria": "x"},
                circular,
            ))
        self.assertTrue(res["match"])

    def test_judge_parameter_missing_result_defaults_to_fail(self):
        from calibrate.llm.run_tests import _judge_tool_call_parameter

        async def fake_text_judge(evaluators, user_prompt, *a, **k):
            return {}  # judge returned nothing for this evaluator

        with patch("calibrate.llm.run_tests.text_judge", side_effect=fake_text_judge):
            res = asyncio.run(_judge_tool_call_parameter(
                "tool", "param",
                {"match_type": "llm_judge", "criteria": "x"},
                "value",
            ))
        self.assertFalse(res["match"])


class TestNestedToolCallCriteria(unittest.TestCase):
    @staticmethod
    def _judge(match, reasoning="r"):
        async def fake_text_judge(evaluators, user_prompt, *a, **k):
            return {evaluators[0]["name"]: {"match": match, "reasoning": reasoning}}
        return fake_text_judge

    def test_nested_subparam_judged_pass(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch(
            "calibrate.llm.run_tests.text_judge", side_effect=self._judge(True)
        ) as mock_judge:
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "book", "arguments": {
                    "patient": {"name": "John Doe", "note": "severe headache since morning"}}}],
                [{"tool": "book", "arguments": {
                    "patient": {
                        "name": "John Doe",                       # nested exact
                        "note": {"match_type": "llm_judge",       # nested judged
                                 "criteria": "describes a symptom"}}}}],
            ))
        self.assertTrue(result["passed"])
        # Both nested leaves are captured with their dotted paths: the exact
        # name match and the judged note.
        self.assertEqual(
            result["tool_call_results"][0]["param_judgments"],
            [
                {
                    "param": "patient.name",
                    "match_type": "exact",
                    "match": True,
                },
                {
                    "param": "patient.note",
                    "match_type": "llm_judge",
                    "criteria": "describes a symptom",
                    "match": True,
                    "reasoning": "r",
                },
            ],
        )
        self.assertIn("patient.name: value matches the expected value", result["reasoning"])
        self.assertIn("patient.note", result["reasoning"])
        # The judge prompt should carry the dotted path as the argument name.
        prompt = mock_judge.call_args.kwargs.get("user_prompt") or mock_judge.call_args.args[1]
        self.assertIn("patient.note", prompt)
        self.assertIn("severe headache", prompt)

    def test_nested_subparam_judged_fail(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch(
            "calibrate.llm.run_tests.text_judge",
            side_effect=self._judge(False, "not a symptom"),
        ):
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "book", "arguments": {
                    "patient": {"name": "John Doe", "note": "wants a discount"}}}],
                [{"tool": "book", "arguments": {
                    "patient": {
                        "name": "John Doe",
                        "note": {"match_type": "llm_judge", "criteria": "describes a symptom"}}}}],
            ))
        self.assertFalse(result["passed"])
        self.assertIn("patient.note", result["reasoning"])
        self.assertIn("not a symptom", result["reasoning"])

    def test_nested_exact_mismatch_and_judge_pass(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        # Nested exact field (name) is wrong; nested judged field passes -> fail
        # overall, and the literal mismatch is reported with its dotted path.
        with patch(
            "calibrate.llm.run_tests.text_judge", side_effect=self._judge(True)
        ):
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "book", "arguments": {
                    "patient": {"name": "Jane", "note": "fever"}}}],
                [{"tool": "book", "arguments": {
                    "patient": {
                        "name": "John Doe",
                        "note": {"match_type": "llm_judge", "criteria": "a symptom"}}}}],
            ))
        self.assertFalse(result["passed"])
        self.assertIn("patient.name", result["reasoning"])

    def test_multiple_nested_subparams_judged_concurrently(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        calls = {"n": 0}

        async def fake_text_judge(evaluators, user_prompt, *a, **k):
            calls["n"] += 1
            return {evaluators[0]["name"]: {"match": True, "reasoning": "ok"}}

        with patch("calibrate.llm.run_tests.text_judge", side_effect=fake_text_judge):
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "t", "arguments": {
                    "a": {"b": "x", "c": "y"}, "d": "z"}}],
                [{"tool": "t", "arguments": {
                    "a": {
                        "b": {"match_type": "llm_judge", "criteria": "c1"},
                        "c": {"match_type": "llm_judge", "criteria": "c2"}},
                    "d": {"match_type": "llm_judge", "criteria": "c3"}}}],
            ))
        self.assertTrue(result["passed"])
        self.assertEqual(calls["n"], 3)  # all three judged

    def test_nested_missing_subparam(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch("calibrate.llm.run_tests.text_judge") as mock_judge:
            result = asyncio.run(evaluate_tool_calls(
                [{"tool": "t", "arguments": {"a": {"b": "x"}}}],
                [{"tool": "t", "arguments": {
                    "a": {"note": {"match_type": "llm_judge", "criteria": "c"}}}}],
            ))
        self.assertFalse(result["passed"])
        self.assertIn("a.note", result["reasoning"])
        self.assertIn("missing in actual", result["reasoning"])
        mock_judge.assert_not_called()

    def test_multi_call_pass_prefixes_tool_in_reasoning(self):
        # With more than one expected tool call, each parameter in the
        # consolidated pass reasoning is prefixed with its tool so the lines
        # stay unambiguous (exact-matched line and judged-param line alike).
        from calibrate.llm.run_tests import evaluate_tool_calls

        with patch(
            "calibrate.llm.run_tests.text_judge", side_effect=self._judge(True, "ok")
        ):
            result = asyncio.run(evaluate_tool_calls(
                [
                    {"tool": "log", "arguments": {"id": 1, "note": "fever"}},
                    {"tool": "sms", "arguments": {"msg": "hello there"}},
                ],
                [
                    {"tool": "log", "arguments": {
                        "id": 1,
                        "note": {"match_type": "llm_judge", "criteria": "a symptom"}}},
                    {"tool": "sms", "arguments": {
                        "msg": {"match_type": "llm_judge", "criteria": "a greeting"}}},
                ],
            ))
        self.assertTrue(result["passed"])
        self.assertIn("log.id: value matches the expected value", result["reasoning"])
        self.assertIn("log.note: criteria met", result["reasoning"])
        self.assertIn("sms.msg: criteria met", result["reasoning"])

    def test_exact_spec_wrapping_dict_mismatch(self):
        # An `exact` spec whose value is a dict is compared verbatim; on a
        # mismatch the per-parameter record captures the nested diff.
        from calibrate.llm.run_tests import evaluate_tool_calls

        result = asyncio.run(evaluate_tool_calls(
            [{"tool": "a", "arguments": {"loc": {"city": "Lyon"}}}],
            [{"tool": "a", "arguments": {
                "loc": {"match_type": "exact", "value": {"city": "Paris"}}}}],
        ))
        self.assertFalse(result["passed"])
        self.assertIn("loc", result["reasoning"])
        self.assertIn("Paris", result["reasoning"])

    def test_exact_param_missing_in_actual(self):
        # A required exact parameter absent from the output is reported as
        # missing (no judge involved, plain exact-only mismatch message).
        from calibrate.llm.run_tests import evaluate_tool_calls

        result = asyncio.run(evaluate_tool_calls(
            [{"tool": "a", "arguments": {}}],
            [{"tool": "a", "arguments": {"x": 1}}],
        ))
        self.assertFalse(result["passed"])
        self.assertIn("x: missing in actual output", result["reasoning"])
        self.assertEqual(result["tool_call_results"], [{"tool": "a", "passed": False}])

    def test_collect_arg_diffs_no_specs_matches_sync_differ(self):
        # With no criteria specs anywhere, the recursive walk must produce the
        # same lines as the original synchronous differ.
        from calibrate.llm.run_tests import (
            _collect_arg_diffs,
            _tool_call_arguments_diff_lines,
        )

        expected = {"a": 1, "b": {"c": 2, "d": {"e": 3}}, "x": "keep"}
        actual = {"a": 9, "b": {"c": 2, "d": {"e": 4}}, "y": "extra"}
        lines, jobs = [], []
        _collect_arg_diffs(expected, actual, "", lines, jobs)
        self.assertEqual(jobs, [])
        self.assertEqual(lines, _tool_call_arguments_diff_lines(expected, actual))

    def test_value_mismatch_record_reasoning_handles_colon_in_key(self):
        # The stored reasoning is the mismatch detail itself, not the display
        # line re-parsed — so a key containing ": " does not garble it.
        from calibrate.llm.run_tests import _collect_arg_diffs

        expected = {"time: start": 9}
        actual = {"time: start": 5}
        lines, jobs, records = [], [], []
        _collect_arg_diffs(expected, actual, "", lines, jobs, records=records)
        self.assertEqual(
            records,
            [
                {
                    "param": "time: start",
                    "match_type": "exact",
                    "match": False,
                    "reasoning": "value mismatch — expected 9, got 5",
                }
            ],
        )
        self.assertEqual(lines, ["  time: start: value mismatch — expected 9, got 5"])

    def test_literal_differ_does_not_interpret_specs(self):
        # The criteria-agnostic wrapper (used by the aggregation fallback and by
        # `exact` values) must treat a spec-looking dict as a literal, never as
        # an instruction — and never raise on a malformed-looking spec.
        from calibrate.llm.run_tests import _tool_call_arguments_diff_lines

        spec_like = {"match_type": "llm_judge", "criteria": "x"}
        # Equal spec-like dicts → match (compared verbatim, not judged).
        self.assertEqual(
            _tool_call_arguments_diff_lines({"p": spec_like}, {"p": dict(spec_like)}),
            [],
        )
        # Different spec-like dicts → a plain value-mismatch line, no judging,
        # and no ValueError even though match_type would be "invalid".
        lines = _tool_call_arguments_diff_lines(
            {"p": {"match_type": "invalid"}}, {"p": "actual"}
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("p:", lines[0])


class TestAggregateToolCallsStored(unittest.TestCase):
    def test_reads_stored_tool_call_results(self):
        from calibrate.llm.run_tests import _aggregate_tool_calls

        results = [
            {
                "test_case": {"evaluation": {"type": "tool_call",
                                             "tool_calls": [{"tool": "a"}]}},
                "metrics": {"tool_call_results": [{"tool": "a", "passed": True}]},
                "output": {"tool_calls": []},
            },
            {
                "test_case": {"evaluation": {"type": "tool_call",
                                             "tool_calls": [{"tool": "a"}]}},
                "metrics": {"tool_call_results": [{"tool": "a", "passed": False}]},
                "output": {"tool_calls": []},
            },
        ]
        agg = _aggregate_tool_calls(results)
        self.assertEqual(agg["a"]["passed"], 1)
        self.assertEqual(agg["a"]["total"], 2)

    def test_falls_back_when_no_stored_results(self):
        from calibrate.llm.run_tests import _aggregate_tool_calls

        results = [
            {
                "test_case": {"evaluation": {"type": "tool_call",
                                             "tool_calls": [{"tool": "a", "arguments": {}}]}},
                "metrics": {},
                "output": {"tool_calls": [{"tool": "a", "arguments": {}}]},
            },
        ]
        agg = _aggregate_tool_calls(results)
        self.assertEqual(agg["a"]["passed"], 1)
        self.assertEqual(agg["a"]["total"], 1)


class TestNoResponseJudgeResults(unittest.TestCase):
    def test_binary_evaluators(self):
        from calibrate.llm.run_tests import _no_response_judge_results

        result = _no_response_judge_results([_bin_ev("x")], "no reply")
        self.assertEqual(result["x"]["match"], False)

    def test_rating_evaluators(self):
        from calibrate.llm.run_tests import _no_response_judge_results

        result = _no_response_judge_results([_rate_ev("x", lo=2, hi=5)], "no reply")
        self.assertEqual(result["x"]["score"], 2)

    def test_rating_with_bad_scale(self):
        from calibrate.llm.run_tests import _no_response_judge_results

        ev = {"name": "x", "system_prompt": "x", "judge_model": "m",
              "type": "rating", "scale_min": "not_int", "scale_max": 5}
        result = _no_response_judge_results([ev], "no reply")
        self.assertEqual(result["x"]["score"], 0)

    def test_skip_no_name(self):
        from calibrate.llm.run_tests import _no_response_judge_results

        result = _no_response_judge_results([{"system_prompt": "x"}], "nope")
        self.assertEqual(result, {})


class TestEvaluatorPassed(unittest.TestCase):
    def test_binary_true(self):
        from calibrate.llm.run_tests import _evaluator_passed

        self.assertTrue(_evaluator_passed(_bin_ev("x"), {"match": True}))

    def test_binary_false(self):
        from calibrate.llm.run_tests import _evaluator_passed

        self.assertFalse(_evaluator_passed(_bin_ev("x"), {"match": False}))

    def test_rating_at_max(self):
        from calibrate.llm.run_tests import _evaluator_passed

        self.assertTrue(_evaluator_passed(_rate_ev("x", 1, 5), {"score": 5}))

    def test_rating_below_max(self):
        from calibrate.llm.run_tests import _evaluator_passed

        self.assertFalse(_evaluator_passed(_rate_ev("x", 1, 5), {"score": 4}))


class TestEvaluateResponse(unittest.IsolatedAsyncioTestCase):
    async def test_evaluate_response_passing(self):
        from calibrate.llm.run_tests import _evaluate_response

        with patch("calibrate.llm.run_tests.test_response_llm_judge",
                   AsyncMock(return_value={"x": {"reasoning": "ok", "match": True}})):
            result = await _evaluate_response(
                chat_history=[],
                response="Hi",
                tool_calls=[],
                evaluators=[_bin_ev("x")],
                no_response_reasoning_with_tool_calls="x",
                no_response_reasoning_no_tool_calls="y",
            )
        self.assertTrue(result["passed"])

    async def test_evaluate_response_failing_uses_first_reasoning(self):
        from calibrate.llm.run_tests import _evaluate_response

        with patch("calibrate.llm.run_tests.test_response_llm_judge",
                   AsyncMock(return_value={
                       "x": {"reasoning": "no good", "match": False},
                       "y": {"reasoning": "ok", "match": True},
                   })):
            result = await _evaluate_response(
                chat_history=[],
                response="Hi",
                tool_calls=[],
                evaluators=[_bin_ev("x"), _bin_ev("y")],
                no_response_reasoning_with_tool_calls="x",
                no_response_reasoning_no_tool_calls="y",
            )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reasoning"], "no good")

    async def test_evaluate_response_empty_no_tool_calls(self):
        from calibrate.llm.run_tests import _evaluate_response

        result = await _evaluate_response(
            chat_history=[],
            response="",
            tool_calls=[],
            evaluators=[_bin_ev("x")],
            no_response_reasoning_with_tool_calls="WITH",
            no_response_reasoning_no_tool_calls="NONE",
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reasoning"], "NONE")

    async def test_evaluate_response_empty_with_tool_calls(self):
        from calibrate.llm.run_tests import _evaluate_response

        result = await _evaluate_response(
            chat_history=[],
            response="",
            tool_calls=[{"tool": "x"}],
            evaluators=[_bin_ev("x")],
            no_response_reasoning_with_tool_calls="WITH",
            no_response_reasoning_no_tool_calls="NONE",
        )
        self.assertEqual(result["reasoning"], "WITH")


class TestEvaluateTestCaseOutput(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_type_raises(self):
        from calibrate.llm.run_tests import evaluate_test_case_output

        with self.assertRaises(ValueError):
            await evaluate_test_case_output(
                chat_history=[],
                evaluation={"type": "bogus"},
                output={"response": "", "tool_calls": []},
            )

    async def test_tool_call_type(self):
        from calibrate.llm.run_tests import evaluate_test_case_output

        result = await evaluate_test_case_output(
            chat_history=[],
            evaluation={"type": "tool_call", "tool_calls": [{"tool": "a"}]},
            output={"response": "", "tool_calls": [{"tool": "a", "arguments": {}}]},
        )
        self.assertTrue(result["passed"])

    async def test_response_with_default_reasoning_messages(self):
        from calibrate.llm.run_tests import evaluate_test_case_output

        result = await evaluate_test_case_output(
            chat_history=[],
            evaluation={"type": "response", "criteria": "be polite"},
            output={"response": "", "tool_calls": [{"tool": "a"}]},
            evaluators=[_bin_ev("default")],
        )
        self.assertIn("Tool calls were generated", result["reasoning"])


class TestAggregateCriteria(unittest.TestCase):
    def test_binary_aggregation(self):
        from calibrate.llm.run_tests import _aggregate_criteria

        registry = {"x": dict(_bin_ev("x"), id="ev1")}
        results = [
            {
                "test_case": {"evaluation": {"type": "response", "criteria": [{"name": "x"}]}},
                "metrics": {"judge_results": {"x": {"match": True}}},
            },
            {
                "test_case": {"evaluation": {"type": "response", "criteria": [{"name": "x"}]}},
                "metrics": {"judge_results": {"x": {"match": False}}},
            },
            # Skip — wrong type
            {
                "test_case": {"evaluation": {"type": "tool_call"}},
                "metrics": {},
            },
            # Skip — no judge_results
            {
                "test_case": {"evaluation": {"type": "response", "criteria": [{"name": "x"}]}},
                "metrics": {},
            },
        ]
        agg = _aggregate_criteria(results, registry)
        self.assertEqual(agg["x"]["passed"], 1)
        self.assertEqual(agg["x"]["total"], 2)
        self.assertEqual(agg["x"]["evaluator_id"], "ev1")

    def test_rating_aggregation(self):
        from calibrate.llm.run_tests import _aggregate_criteria

        registry = {"r": _rate_ev("r", 1, 5)}
        results = [
            {
                "test_case": {"evaluation": {"type": "response", "criteria": [{"name": "r"}]}},
                "metrics": {"judge_results": {"r": {"score": 5}}},
            },
            {
                "test_case": {"evaluation": {"type": "response", "criteria": [{"name": "r"}]}},
                "metrics": {"judge_results": {"r": {"score": 3}}},
            },
        ]
        agg = _aggregate_criteria(results, registry)
        self.assertEqual(agg["r"]["type"], "rating")
        self.assertEqual(agg["r"]["mean"], 4.0)

    def test_unknown_evaluator_skipped(self):
        from calibrate.llm.run_tests import _aggregate_criteria

        registry = {}
        results = [
            {
                "test_case": {"evaluation": {"type": "response", "criteria": [{"name": "unknown"}]}},
                "metrics": {"judge_results": {"unknown": {"match": True}}},
            },
        ]
        agg = _aggregate_criteria(results, registry)
        self.assertEqual(agg, {})


class TestAggregateToolCalls(unittest.TestCase):
    def test_aggregation(self):
        from calibrate.llm.run_tests import _aggregate_tool_calls

        results = [
            {
                "test_case": {"evaluation": {"type": "tool_call", "tool_calls": [{"tool": "a", "arguments": {}}]}},
                "output": {"tool_calls": [{"tool": "a", "arguments": {}}]},
            },
            {
                "test_case": {"evaluation": {"type": "tool_call", "tool_calls": [{"tool": "a", "arguments": {}}]}},
                "output": {"tool_calls": []},
            },
            # Skip non-tool_call
            {"test_case": {"evaluation": {"type": "response"}}, "output": {}},
        ]
        agg = _aggregate_tool_calls(results)
        self.assertEqual(agg["a"]["passed"], 1)
        self.assertEqual(agg["a"]["total"], 2)


class TestValidateLLMEvalOnlyDataset(unittest.TestCase):
    def test_valid(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, err = validate_llm_eval_only_dataset([
            {
                "test_case": {
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "x"},
                },
                "output": {"response": "ok", "tool_calls": []},
            }
        ])
        self.assertTrue(is_valid)

    def test_valid_with_tool_call_output(self):
        """Optional per-tool-call ``output`` is accepted by the validator."""
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, err = validate_llm_eval_only_dataset([
            {
                "test_case": {
                    "history": [{"role": "user", "content": "weather?"}],
                    "evaluation": {"type": "tool_call", "tool_calls": [
                        {"tool": "get_weather", "arguments": {"city": "NYC"}}
                    ]},
                },
                "output": {
                    "response": None,
                    "tool_calls": [
                        {
                            "tool": "get_weather",
                            "arguments": {"city": "NYC"},
                            "output": {"temp": 72},
                        }
                    ],
                },
            }
        ])
        self.assertTrue(is_valid, err)

    def test_not_a_list(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, err = validate_llm_eval_only_dataset({})
        self.assertFalse(is_valid)

    def test_item_not_dict(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, err = validate_llm_eval_only_dataset(["not a dict"])
        self.assertFalse(is_valid)

    def test_missing_keys(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, _ = validate_llm_eval_only_dataset([{"test_case": {}}])
        self.assertFalse(is_valid)

    def test_tc_not_dict(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, _ = validate_llm_eval_only_dataset([{"test_case": "x", "output": {}}])
        self.assertFalse(is_valid)

    def test_output_not_dict(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, _ = validate_llm_eval_only_dataset([{"test_case": {}, "output": "x"}])
        self.assertFalse(is_valid)

    def test_tc_missing_history(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, _ = validate_llm_eval_only_dataset([{
            "test_case": {"evaluation": {"type": "response"}}, "output": {}
        }])
        self.assertFalse(is_valid)

    def test_history_not_list(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, _ = validate_llm_eval_only_dataset([{
            "test_case": {"history": "not list", "evaluation": {"type": "response"}},
            "output": {"response": "", "tool_calls": []},
        }])
        self.assertFalse(is_valid)

    def test_evaluation_not_dict(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, _ = validate_llm_eval_only_dataset([{
            "test_case": {"history": [], "evaluation": "x"},
            "output": {"response": "", "tool_calls": []},
        }])
        self.assertFalse(is_valid)

    def test_invalid_eval_type(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, _ = validate_llm_eval_only_dataset([{
            "test_case": {"history": [], "evaluation": {"type": "bogus"}},
            "output": {"response": "", "tool_calls": []},
        }])
        self.assertFalse(is_valid)

    def test_missing_output_keys(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, _ = validate_llm_eval_only_dataset([{
            "test_case": {"history": [], "evaluation": {"type": "response"}},
            "output": {"response": ""},
        }])
        self.assertFalse(is_valid)

    def test_tool_calls_not_list(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, _ = validate_llm_eval_only_dataset([{
            "test_case": {"history": [], "evaluation": {"type": "response"}},
            "output": {"response": "", "tool_calls": "nope"},
        }])
        self.assertFalse(is_valid)


class TestRunEvalOnlyTests(unittest.IsolatedAsyncioTestCase):
    async def test_eval_only_basic(self):
        from calibrate.llm.run_tests import run_eval_only_tests

        with tempfile.TemporaryDirectory() as tmp, \
             patch("calibrate.llm.run_tests.test_response_llm_judge",
                   AsyncMock(return_value={"correctness": {"reasoning": "ok", "match": True}})):
            result = await run_eval_only_tests(
                config={"evaluators": []},
                dataset=[
                    {
                        "test_case": {
                            "id": "tc1",
                            "history": [{"role": "user", "content": "hi"}],
                            "evaluation": {"type": "response", "criteria": "polite"},
                        },
                        "output": {"response": "Hi!", "tool_calls": []},
                    },
                    # Failing tool_call test
                    {
                        "test_case": {
                            "history": [],
                            "evaluation": {"type": "tool_call", "tool_calls": [{"tool": "x"}]},
                        },
                        "output": {"response": "", "tool_calls": []},
                    },
                ],
                output_dir=tmp,
            )
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["total"], 2)


class TestWriteTestResults(unittest.TestCase):
    def test_writes_files(self):
        from calibrate.llm.run_tests import _write_test_results_outputs

        with tempfile.TemporaryDirectory() as tmp:
            results = [
                {
                    "metrics": {"passed": True, "judge_results": {}},
                    "test_case": {"evaluation": {"type": "tool_call", "tool_calls": []}},
                    "output": {"response": "", "tool_calls": []},
                },
            ]
            passed, total = _write_test_results_outputs(results, tmp, {})
            self.assertEqual((passed, total), (1, 1))
            self.assertTrue((Path(tmp) / "results.json").exists())
            self.assertTrue((Path(tmp) / "metrics.json").exists())

    def test_metrics_includes_mean_cost(self):
        from calibrate.llm.run_tests import _write_test_results_outputs

        with tempfile.TemporaryDirectory() as tmp:
            results = [
                {
                    "metrics": {"passed": True, "judge_results": {}},
                    "test_case": {"evaluation": {"type": "tool_call", "tool_calls": []}},
                    "output": {"response": "", "tool_calls": [], "cost": 0.02},
                },
                {
                    "metrics": {"passed": True, "judge_results": {}},
                    "test_case": {"evaluation": {"type": "tool_call", "tool_calls": []}},
                    "output": {"response": "", "tool_calls": [], "cost": 0.04},
                },
            ]
            _write_test_results_outputs(results, tmp, {})
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())
            self.assertAlmostEqual(metrics["cost"]["mean"], 0.03)
            self.assertAlmostEqual(metrics["cost"]["min"], 0.02)
            self.assertAlmostEqual(metrics["cost"]["max"], 0.04)
            self.assertEqual(metrics["cost"]["count"], 2)

    def test_metrics_cost_omitted_without_costs(self):
        from calibrate.llm.run_tests import _write_test_results_outputs

        with tempfile.TemporaryDirectory() as tmp:
            results = [
                {
                    "metrics": {"passed": True, "judge_results": {}},
                    "test_case": {"evaluation": {"type": "tool_call", "tool_calls": []}},
                    "output": {"response": "", "tool_calls": []},
                },
            ]
            _write_test_results_outputs(results, tmp, {})
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())
            self.assertNotIn("cost", metrics)


class TestAggregateCost(unittest.TestCase):
    def test_no_results_returns_none(self):
        from calibrate.llm.run_tests import _aggregate_cost

        self.assertIsNone(_aggregate_cost([]))

    def test_no_costs_returns_none(self):
        from calibrate.llm.run_tests import _aggregate_cost

        results = [
            {"output": {"response": "hi", "tool_calls": []}},
            {"output": {"response": "yo", "tool_calls": [], "cost": None}},
            {"metrics": {"passed": True}},  # no output key at all
        ]
        self.assertIsNone(_aggregate_cost(results))

    def test_mean_across_costs(self):
        from calibrate.llm.run_tests import _aggregate_cost

        results = [
            {"output": {"cost": 0.01}},
            {"output": {"cost": 0.03}},
            {"output": {"cost": None}},  # ignored
            {"output": {}},  # ignored
        ]
        agg = _aggregate_cost(results)
        self.assertAlmostEqual(agg["mean"], 0.02)
        self.assertAlmostEqual(agg["min"], 0.01)
        self.assertAlmostEqual(agg["max"], 0.03)
        self.assertEqual(agg["count"], 2)

    def test_zero_cost_counted(self):
        from calibrate.llm.run_tests import _aggregate_cost

        results = [{"output": {"cost": 0.0}}, {"output": {"cost": 0.02}}]
        agg = _aggregate_cost(results)
        self.assertAlmostEqual(agg["mean"], 0.01)
        self.assertEqual(agg["count"], 2)


    def test_cost_read_only_from_output_cost(self):
        """Cost lives at ``output.cost`` for both paths; a nested-only metrics
        cost is not dug out here (run_test_external lifts it first)."""
        from calibrate.llm.run_tests import _aggregate_cost

        results = [
            {"output": {"cost": 0.02}},                       # counted
            {"output": {"metrics": {"cost": 0.04}}},          # not lifted → ignored
            {"output": {"response": "x", "tool_calls": []}},  # no cost
        ]
        agg = _aggregate_cost(results)
        self.assertAlmostEqual(agg["mean"], 0.02)
        self.assertEqual(agg["count"], 1)


class TestAggregateTotalTokens(unittest.TestCase):
    def test_no_results_returns_none(self):
        from calibrate.llm.run_tests import _aggregate_total_tokens

        self.assertIsNone(_aggregate_total_tokens([]))

    def test_no_tokens_returns_none(self):
        from calibrate.llm.run_tests import _aggregate_total_tokens

        results = [
            {"output": {"response": "hi", "tool_calls": []}},
            {"output": {"response": "yo", "tool_calls": [], "total_tokens": None}},
            {"metrics": {"passed": True}},  # no output key at all
        ]
        self.assertIsNone(_aggregate_total_tokens(results))

    def test_mean_across_token_counts(self):
        from calibrate.llm.run_tests import _aggregate_total_tokens

        results = [
            {"output": {"total_tokens": 100}},
            {"output": {"total_tokens": 201}},
            {"output": {"total_tokens": None}},  # ignored
            {"output": {}},  # ignored
        ]
        agg = _aggregate_total_tokens(results)
        self.assertEqual(agg["mean"], 150)  # round(301 / 2) — banker's rounding
        self.assertEqual(agg["min"], 100)
        self.assertEqual(agg["max"], 201)
        self.assertEqual(agg["count"], 2)

    def test_read_only_from_output_total_tokens(self):
        """Token usage lives at ``output.total_tokens`` for both paths; a
        nested-only metrics count is not dug out here (run_test_external lifts
        it first)."""
        from calibrate.llm.run_tests import _aggregate_total_tokens

        results = [
            {"output": {"total_tokens": 50}},                    # counted
            {"output": {"metrics": {"total_tokens": 999}}},      # not lifted → ignored
            {"output": {"response": "x", "tool_calls": []}},     # no tokens
        ]
        agg = _aggregate_total_tokens(results)
        self.assertEqual(agg["mean"], 50)
        self.assertEqual(agg["count"], 1)


class TestCostTrackingService(unittest.IsolatedAsyncioTestCase):
    def _service(self):
        from calibrate.llm.run_tests import CostTrackingOpenRouterLLMService

        return CostTrackingOpenRouterLLMService.__new__(
            CostTrackingOpenRouterLLMService
        )

    async def _drain(self, svc, chunks):
        async def fake_stream():
            for c in chunks:
                yield c

        out = [c async for c in svc._capture_cost(fake_stream())]
        return out

    async def test_captures_cost_from_usage_attr(self):
        svc = self._service()
        svc.last_cost = None
        usage = MagicMock(cost=0.00231)
        chunks = [MagicMock(usage=None), MagicMock(usage=usage)]
        out = await self._drain(svc, chunks)
        self.assertEqual(len(out), 2)  # stream passes through unchanged
        self.assertAlmostEqual(svc.last_cost, 0.00231)

    async def test_captures_cost_from_model_extra(self):
        svc = self._service()
        svc.last_cost = None
        usage = MagicMock(spec=["model_extra"])
        usage.model_extra = {"cost": 0.005}
        out = await self._drain(svc, [MagicMock(usage=usage)])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(svc.last_cost, 0.005)

    async def test_no_cost_leaves_none(self):
        svc = self._service()
        svc.last_cost = None
        usage = MagicMock(spec=["model_extra"])
        usage.model_extra = {}
        await self._drain(svc, [MagicMock(usage=usage), MagicMock(usage=None)])
        self.assertIsNone(svc.last_cost)

    async def test_non_numeric_cost_ignored(self):
        svc = self._service()
        svc.last_cost = None
        usage = MagicMock(cost="not-a-number")
        await self._drain(svc, [MagicMock(usage=usage)])
        self.assertIsNone(svc.last_cost)


class TestProcessorTokenCapture(unittest.IsolatedAsyncioTestCase):
    """The Processor accumulates token usage from LLM usage MetricsFrames."""

    def _usage_frame(self, prompt, completion, total=None):
        from pipecat.frames.frames import MetricsFrame
        from pipecat.metrics.metrics import LLMUsageMetricsData, LLMTokenUsage

        usage = LLMTokenUsage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
        )
        return MetricsFrame(
            data=[LLMUsageMetricsData(processor="llm", model="m", value=usage)]
        )

    async def _feed(self, proc, frame):
        from calibrate.llm import run_tests as RT

        proc._ready = True  # skip the initial queue_frames branch
        proc._task = None
        with patch.object(RT.FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(proc, "push_frame", AsyncMock()):
            await proc.process_frame(frame, RT.FrameDirection.DOWNSTREAM)

    async def test_accumulates_total_tokens_across_frames(self):
        from calibrate.llm.run_tests import Processor

        proc = Processor(chat_history=[])
        await self._feed(proc, self._usage_frame(1000, 200, total=1200))
        await self._feed(proc, self._usage_frame(300, 50, total=350))
        self.assertEqual(proc._total_tokens, 1550)


class TestRunInferenceWrapping(unittest.IsolatedAsyncioTestCase):
    async def test_run_inference_wraps(self):
        from calibrate.llm import run_tests as RT

        with patch.object(RT, "_run_inference_inner",
                          AsyncMock(return_value={"response": "ok", "tool_calls": []})):
            result = await RT.run_inference([], "sp", "m", "openrouter", [])
        self.assertEqual(result["response"], "ok")
        self.assertIn("captured_errors", result)


class TestRunTestErrorPath(unittest.IsolatedAsyncioTestCase):
    async def test_no_response_no_tool_calls_raises(self):
        from calibrate.llm import run_tests as RT

        with patch.object(RT, "run_inference",
                          AsyncMock(return_value={
                              "response": "", "tool_calls": [], "captured_errors": ["fail"],
                          })):
            with self.assertRaises(RT.LLMInferenceError):
                await RT.run_test(
                    chat_history=[],
                    evaluation={"type": "response"},
                    system_prompt="x", model="m", provider="openrouter",
                    tools=[], unique_id="u",
                )

    async def test_response_path(self):
        from calibrate.llm import run_tests as RT

        with patch.object(RT, "run_inference",
                          AsyncMock(return_value={
                              "response": "Hi", "tool_calls": [], "captured_errors": [],
                          })), \
             patch.object(RT, "test_response_llm_judge",
                          AsyncMock(return_value={"default": {"reasoning": "ok", "match": True}})):
            result = await RT.run_test(
                chat_history=[],
                evaluation={"type": "response", "criteria": "x"},
                system_prompt="x", model="m", provider="openrouter",
                tools=[], unique_id="u",
                evaluators=[_bin_ev("default")],
            )
        self.assertTrue(result["metrics"]["passed"])

    async def test_response_path_with_langfuse(self):
        from calibrate.llm import run_tests as RT

        fake_lf = MagicMock()
        with patch.object(RT, "run_inference",
                          AsyncMock(return_value={
                              "response": "Hi", "tool_calls": [], "captured_errors": [],
                          })), \
             patch.object(RT, "test_response_llm_judge",
                          AsyncMock(return_value={"default": {"reasoning": "ok", "match": True}})), \
             patch.object(RT, "langfuse_enabled", True), \
             patch.object(RT, "langfuse", fake_lf):
            await RT.run_test(
                chat_history=[],
                evaluation={"type": "response", "criteria": "x"},
                system_prompt="x", model="m", provider="openrouter",
                tools=[], unique_id="u",
                evaluators=[_bin_ev("default")],
            )
        fake_lf.update_current_trace.assert_called_once()


class TestRunModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_model_tests_smoke(self):
        from calibrate.llm import run_tests as RT

        fake_test = {
            "output": {"response": "Hi", "tool_calls": []},
            "metrics": {"passed": True, "judge_results": {}, "reasoning": "ok"},
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(RT, "run_test", AsyncMock(return_value=fake_test)):
            config = {
                "system_prompt": "sp",
                "tools": [],
                "evaluators": [],
                "test_cases": [{
                    "id": "tc1",
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "be polite"},
                }],
            }
            result = await RT.run_model_tests(
                model="m", provider="openrouter", config=config, output_dir=tmp,
            )
        self.assertEqual(result["metrics"]["passed"], 1)


class TestResolveTestParallel(unittest.TestCase):
    def test_cli_value_takes_precedence(self):
        from calibrate.llm.run_tests import _resolve_test_parallel

        with patch.dict("os.environ", {"CALIBRATE_TEST_PARALLEL": "7"}):
            self.assertEqual(_resolve_test_parallel(3), 3)

    def test_env_var_used_when_no_cli(self):
        from calibrate.llm.run_tests import _resolve_test_parallel

        with patch.dict("os.environ", {"CALIBRATE_TEST_PARALLEL": "7"}):
            self.assertEqual(_resolve_test_parallel(None), 7)

    def test_default_when_neither_set(self):
        from calibrate.llm.run_tests import _resolve_test_parallel, DEFAULT_TEST_PARALLEL

        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("CALIBRATE_TEST_PARALLEL", None)
            self.assertEqual(_resolve_test_parallel(None), DEFAULT_TEST_PARALLEL)

    def test_invalid_env_falls_back_to_default(self):
        from calibrate.llm.run_tests import _resolve_test_parallel, DEFAULT_TEST_PARALLEL

        with patch.dict("os.environ", {"CALIBRATE_TEST_PARALLEL": "abc"}):
            self.assertEqual(_resolve_test_parallel(None), DEFAULT_TEST_PARALLEL)

    def test_non_positive_values_ignored(self):
        from calibrate.llm.run_tests import _resolve_test_parallel, DEFAULT_TEST_PARALLEL

        with patch.dict("os.environ", {"CALIBRATE_TEST_PARALLEL": "0"}):
            # CLI 0 ignored, env 0 ignored -> default
            self.assertEqual(_resolve_test_parallel(0), DEFAULT_TEST_PARALLEL)


class TestRunModelTestsParallel(unittest.IsolatedAsyncioTestCase):
    async def test_test_cases_run_concurrently(self):
        from calibrate.llm import run_tests as RT

        concurrency = {"current": 0, "peak": 0}
        gate = asyncio.Event()
        started = 0
        n_cases = 4

        async def fake_run_test(**kwargs):
            nonlocal started
            concurrency["current"] += 1
            concurrency["peak"] = max(concurrency["peak"], concurrency["current"])
            started += 1
            if started >= n_cases:
                gate.set()
            await gate.wait()  # block until all have started -> proves overlap
            concurrency["current"] -= 1
            return {
                "output": {"response": "Hi", "tool_calls": []},
                "metrics": {"passed": True, "judge_results": {}, "reasoning": "ok"},
            }

        config = {
            "system_prompt": "sp",
            "tools": [],
            "evaluators": [],
            "test_cases": [
                {
                    "id": f"tc{i}",
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "response", "criteria": "be polite"},
                }
                for i in range(n_cases)
            ],
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(RT, "run_test", side_effect=fake_run_test):
            result = await asyncio.wait_for(
                RT.run_model_tests(
                    model="m", provider="openrouter", config=config,
                    output_dir=tmp, test_parallel=n_cases,
                ),
                timeout=5,
            )

        self.assertEqual(concurrency["peak"], n_cases)
        self.assertEqual(result["metrics"]["passed"], n_cases)

    async def test_results_preserve_test_case_order(self):
        from calibrate.llm import run_tests as RT

        async def fake_run_test(**kwargs):
            # Later test cases finish first to scramble completion order.
            content = kwargs["chat_history"][0]["content"]
            await asyncio.sleep(0.01 * (5 - int(content)))
            return {
                "output": {"response": content, "tool_calls": []},
                "metrics": {"passed": True, "judge_results": {}, "reasoning": "ok"},
            }

        config = {
            "system_prompt": "sp",
            "tools": [],
            "evaluators": [],
            "test_cases": [
                {
                    "id": f"tc{i}",
                    "history": [{"role": "user", "content": str(i)}],
                    "evaluation": {"type": "response", "criteria": "x"},
                }
                for i in range(5)
            ],
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(RT, "run_test", side_effect=fake_run_test):
            result = await RT.run_model_tests(
                model="m", provider="openrouter", config=config,
                output_dir=tmp, test_parallel=5,
            )
            with open(Path(tmp) / "m" / "results.json") as f:
                saved = json.load(f)

        self.assertEqual(
            [r["test_case_id"] for r in result["results"]],
            [f"tc{i}" for i in range(5)],
        )
        self.assertEqual([r["test_case_id"] for r in saved], [f"tc{i}" for i in range(5)])


class TestMainCLI(unittest.IsolatedAsyncioTestCase):
    async def test_main_eval_only(self):
        from calibrate.llm import run_tests as RT

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({"evaluators": []}))
            ds = Path(tmp) / "ds.json"
            ds.write_text(json.dumps([{
                "test_case": {
                    "history": [],
                    "evaluation": {"type": "response", "criteria": "x"},
                },
                "output": {"response": "Hi", "tool_calls": []},
            }]))
            argv = ["rt.py", "-c", str(cfg), "-o", tmp, "--eval-only", "--dataset", str(ds)]
            with patch.object(sys, "argv", argv), \
                 patch.object(RT, "test_response_llm_judge",
                              AsyncMock(return_value={"correctness": {"reasoning": "ok", "match": True}})):
                await RT.main()

    async def test_main_eval_only_missing_dataset(self):
        from calibrate.llm import run_tests as RT

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({"evaluators": []}))
            argv = ["rt.py", "-c", str(cfg), "-o", tmp, "--eval-only"]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await RT.main()

    async def test_main_eval_only_invalid_dataset_json(self):
        from calibrate.llm import run_tests as RT

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({"evaluators": []}))
            ds = Path(tmp) / "ds.json"
            ds.write_text("{not json")
            argv = ["rt.py", "-c", str(cfg), "-o", tmp, "--eval-only", "--dataset", str(ds)]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await RT.main()

    async def test_main_eval_only_invalid_dataset_shape(self):
        from calibrate.llm import run_tests as RT

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({"evaluators": []}))
            ds = Path(tmp) / "ds.json"
            ds.write_text(json.dumps({}))  # not a list
            argv = ["rt.py", "-c", str(cfg), "-o", tmp, "--eval-only", "--dataset", str(ds)]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await RT.main()

    async def test_main_missing_model(self):
        from calibrate.llm import run_tests as RT

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({"evaluators": [], "test_cases": []}))
            argv = ["rt.py", "-c", str(cfg), "-o", tmp]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    await RT.main()

    async def test_main_with_model(self):
        from calibrate.llm import run_tests as RT

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({
                "evaluators": [],
                "test_cases": [],
                "system_prompt": "sp",
                "tools": [],
            }))
            argv = ["rt.py", "-c", str(cfg), "-o", tmp, "-m", "model-x"]
            with patch.object(sys, "argv", argv), \
                 patch.object(RT, "run_model_tests",
                              AsyncMock(return_value={
                                  "model": "model-x", "provider": "openrouter",
                                  "metrics": {"passed": 0, "total": 0}, "results": [],
                              })):
                await RT.main()


class TestGetNameToEvaluatorDictNoDefault(unittest.TestCase):
    def test_excludes_implicit_default(self):
        from calibrate.llm.run_tests import _get_name_to_evaluator_dict
        from calibrate.judges import DEFAULT_LLM_TEST_EVALUATOR

        reg = _get_name_to_evaluator_dict(
            {"evaluators": [_bin_ev("tone")]}, include_default=False
        )
        self.assertIn("tone", reg)
        self.assertNotIn(DEFAULT_LLM_TEST_EVALUATOR["name"], reg)
        self.assertNotIn("default", reg)

    def test_default_included_by_default(self):
        from calibrate.llm.run_tests import _get_name_to_evaluator_dict
        from calibrate.judges import DEFAULT_LLM_TEST_EVALUATOR

        reg = _get_name_to_evaluator_dict({"evaluators": [_bin_ev("tone")]})
        self.assertIn(DEFAULT_LLM_TEST_EVALUATOR["name"], reg)
        self.assertIn("default", reg)
        self.assertIn("tone", reg)

    def test_empty_no_default(self):
        from calibrate.llm.run_tests import _get_name_to_evaluator_dict

        self.assertEqual(
            _get_name_to_evaluator_dict({"evaluators": []}, include_default=False), {}
        )

    def test_missing_fields_raise(self):
        from calibrate.llm.run_tests import _get_name_to_evaluator_dict

        with self.assertRaises(ValueError):
            _get_name_to_evaluator_dict(
                {"evaluators": [{"name": "x"}]}, include_default=False
            )


class TestEvaluateConversation(unittest.IsolatedAsyncioTestCase):
    _NO_REPLY = "no reply"

    async def test_binary_pass(self):
        from calibrate.llm import run_tests as RT

        with patch.object(RT, "evaluate_simuation",
                          AsyncMock(return_value={"tone": {"reasoning": "good", "match": True}})):
            metrics = await RT._evaluate_conversation(
                chat_history=[{"role": "user", "content": "hi"}],
                evaluators=[_bin_ev("tone")],
                output={"response": "hi", "tool_calls": []},
                no_response_reasoning_no_tool_calls=self._NO_REPLY,
            )
        self.assertTrue(metrics["passed"])
        self.assertEqual(metrics["reasoning"], "All evaluators passed")
        self.assertEqual(metrics["judge_results"]["tone"]["match"], True)

    async def test_binary_fail_uses_failing_reasoning(self):
        from calibrate.llm import run_tests as RT

        with patch.object(RT, "evaluate_simuation",
                          AsyncMock(return_value={"tone": {"reasoning": "rude", "match": False}})):
            metrics = await RT._evaluate_conversation(
                chat_history=[], evaluators=[_bin_ev("tone")],
                output={"response": "x", "tool_calls": []},
                no_response_reasoning_no_tool_calls=self._NO_REPLY,
            )
        self.assertFalse(metrics["passed"])
        self.assertEqual(metrics["reasoning"], "rude")

    async def test_rating_below_max_fails(self):
        from calibrate.llm import run_tests as RT

        with patch.object(RT, "evaluate_simuation",
                          AsyncMock(return_value={"qual": {"reasoning": "ok", "score": 3}})):
            metrics = await RT._evaluate_conversation(
                chat_history=[], evaluators=[_rate_ev("qual", 1, 5)],
                output={"response": "x", "tool_calls": []},
                no_response_reasoning_no_tool_calls=self._NO_REPLY,
            )
        self.assertFalse(metrics["passed"])

    async def test_rating_at_max_passes(self):
        from calibrate.llm import run_tests as RT

        with patch.object(RT, "evaluate_simuation",
                          AsyncMock(return_value={"qual": {"reasoning": "great", "score": 5}})):
            metrics = await RT._evaluate_conversation(
                chat_history=[], evaluators=[_rate_ev("qual", 1, 5)],
                output={"response": "x", "tool_calls": []},
                no_response_reasoning_no_tool_calls=self._NO_REPLY,
            )
        self.assertTrue(metrics["passed"])

    async def test_output_appended_before_judging(self):
        from calibrate.llm import run_tests as RT

        sim = AsyncMock(return_value={"tone": {"reasoning": "ok", "match": True}})
        with patch.object(RT, "evaluate_simuation", sim):
            await RT._evaluate_conversation(
                chat_history=[{"role": "user", "content": "hi"}],
                evaluators=[_bin_ev("tone")],
                output={"response": "hello", "tool_calls": []},
                no_response_reasoning_no_tool_calls=self._NO_REPLY,
            )
        judged = sim.await_args.kwargs["conversation"]
        self.assertEqual(judged[-1], {"role": "assistant", "content": "hello"})


class TestEvaluateTestCaseOutputConversation(unittest.IsolatedAsyncioTestCase):
    async def test_dispatches_to_simulation_judge(self):
        from calibrate.llm import run_tests as RT

        sim = AsyncMock(return_value={"tone": {"reasoning": "ok", "match": True}})
        with patch.object(RT, "evaluate_simuation", sim):
            metrics = await RT.evaluate_test_case_output(
                chat_history=[{"role": "user", "content": "hi"}],
                evaluation={"type": "conversation", "criteria": [{"name": "tone"}]},
                output={"response": "hi", "tool_calls": []},
                evaluators=[_bin_ev("tone")],
            )
        sim.assert_awaited_once()
        self.assertTrue(metrics["passed"])
        self.assertIn("judge_results", metrics)

    async def test_live_output_appended_via_evaluate(self):
        from calibrate.llm import run_tests as RT

        sim = AsyncMock(return_value={"tone": {"reasoning": "ok", "match": True}})
        with patch.object(RT, "evaluate_simuation", sim):
            await RT.evaluate_test_case_output(
                chat_history=[{"role": "user", "content": "hi"}],
                evaluation={"type": "conversation", "criteria": [{"name": "tone"}]},
                output={"response": "hello", "tool_calls": []},
                evaluators=[_bin_ev("tone")],
            )
        judged = sim.await_args.kwargs["conversation"]
        self.assertEqual(judged[-1], {"role": "assistant", "content": "hello"})


class TestRunTestConversation(unittest.IsolatedAsyncioTestCase):
    async def test_runs_inference_and_judges_full_conversation(self):
        from calibrate.llm import run_tests as RT

        # Live mode: the agent generates the next reply, which is appended and
        # the whole conversation is judged by the simulation judge.
        infer = AsyncMock(return_value={
            "response": "It ships tomorrow.", "tool_calls": [], "captured_errors": [],
        })
        sim = AsyncMock(return_value={"tone": {"reasoning": "ok", "match": True}})
        with patch.object(RT, "run_inference", infer), \
             patch.object(RT, "evaluate_simuation", sim):
            result = await RT.run_test(
                chat_history=[{"role": "user", "content": "when does my order ship?"}],
                evaluation={"type": "conversation", "criteria": [{"name": "tone"}]},
                system_prompt="x", model="m", provider="openrouter",
                tools=[], unique_id="u",
                evaluators=[_bin_ev("tone")],
            )
        infer.assert_awaited_once()
        # The generated reply is included as output and appended to the judged
        # conversation.
        self.assertEqual(result["output"]["response"], "It ships tomorrow.")
        judged = sim.await_args.kwargs["conversation"]
        self.assertEqual(judged[-1], {"role": "assistant", "content": "It ships tomorrow."})
        self.assertTrue(result["metrics"]["passed"])

    async def test_tool_calls_appended_in_function_shape(self):
        from calibrate.llm import run_tests as RT

        infer = AsyncMock(return_value={
            "response": "",
            "tool_calls": [{"tool": "check_order", "arguments": '{"id": "1"}'}],
            "captured_errors": [],
        })
        sim = AsyncMock(return_value={"tone": {"reasoning": "ok", "match": True}})
        with patch.object(RT, "run_inference", infer), \
             patch.object(RT, "evaluate_simuation", sim):
            await RT.run_test(
                chat_history=[{"role": "user", "content": "check order 1"}],
                evaluation={"type": "conversation", "criteria": [{"name": "tone"}]},
                system_prompt="x", model="m", provider="openrouter",
                tools=[], unique_id="u",
                evaluators=[_bin_ev("tone")],
            )
        judged = sim.await_args.kwargs["conversation"]
        self.assertEqual(
            judged[-1]["tool_calls"],
            [{"function": {"name": "check_order", "arguments": '{"id": "1"}'}}],
        )


class TestRunTestExternalConversation(unittest.IsolatedAsyncioTestCase):
    async def test_empty_reply_fails_without_judging(self):
        from calibrate.llm import run_tests as RT

        agent = MagicMock()
        agent.call = AsyncMock(return_value={"response": "", "tool_calls": []})
        sim = AsyncMock(return_value={"tone": {"reasoning": "ok", "match": True}})
        with patch.object(RT, "evaluate_simuation", sim):
            result = await RT.run_test_external(
                chat_history=[{"role": "user", "content": "hi"}],
                evaluation={"type": "conversation", "criteria": [{"name": "tone"}]},
                agent=agent,
                evaluators=[_bin_ev("tone")],
            )
        # A non-responsive agent fails outright; the judge is never consulted.
        sim.assert_not_called()
        self.assertFalse(result["metrics"]["passed"])
        self.assertFalse(result["metrics"]["judge_results"]["tone"]["match"])

    async def test_reply_judged_as_full_conversation(self):
        from calibrate.llm import run_tests as RT

        agent = MagicMock()
        agent.call = AsyncMock(return_value={"response": "It ships tomorrow.", "tool_calls": []})
        sim = AsyncMock(return_value={"tone": {"reasoning": "ok", "match": True}})
        with patch.object(RT, "evaluate_simuation", sim):
            result = await RT.run_test_external(
                chat_history=[{"role": "user", "content": "when does it ship?"}],
                evaluation={"type": "conversation", "criteria": [{"name": "tone"}]},
                agent=agent,
                evaluators=[_bin_ev("tone")],
            )
        judged = sim.await_args.kwargs["conversation"]
        self.assertEqual(judged[-1], {"role": "assistant", "content": "It ships tomorrow."})
        self.assertTrue(result["metrics"]["passed"])


class TestAggregateCriteriaConversation(unittest.TestCase):
    def test_conversation_cases_aggregated(self):
        from calibrate.llm.run_tests import _aggregate_criteria

        registry = {"tone": dict(_bin_ev("tone"), id="ev1")}
        results = [
            {
                "test_case": {"evaluation": {"type": "conversation", "criteria": [{"name": "tone"}]}},
                "metrics": {"judge_results": {"tone": {"match": True}}},
            },
            {
                "test_case": {"evaluation": {"type": "conversation", "criteria": [{"name": "tone"}]}},
                "metrics": {"judge_results": {"tone": {"match": False}}},
            },
        ]
        agg = _aggregate_criteria(results, registry)
        self.assertEqual(agg["tone"]["passed"], 1)
        self.assertEqual(agg["tone"]["total"], 2)


class TestValidateConversationEvalOnly(unittest.TestCase):
    def test_conversation_requires_output_like_response(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, _ = validate_llm_eval_only_dataset([
            {
                "test_case": {
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "conversation", "criteria": [{"name": "tone"}]},
                },
            }
        ])
        self.assertFalse(is_valid)

    def test_conversation_still_requires_history(self):
        from calibrate.llm.run_tests import validate_llm_eval_only_dataset

        is_valid, _ = validate_llm_eval_only_dataset([
            {"test_case": {"evaluation": {"type": "conversation"}}}
        ])
        self.assertFalse(is_valid)


class TestRunEvalOnlyConversation(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_eval_only(self):
        from calibrate.llm.run_tests import run_eval_only_tests
        from calibrate.llm import run_tests as RT

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(RT, "evaluate_simuation",
                          AsyncMock(return_value={"tone": {"reasoning": "ok", "match": True}})):
            result = await run_eval_only_tests(
                config={"evaluators": [_bin_ev("tone")]},
                dataset=[
                    {
                        "test_case": {
                            "id": "tc1",
                            "history": [{"role": "user", "content": "hi"}],
                            "evaluation": {
                                "type": "conversation",
                                "criteria": [{"name": "tone"}],
                            },
                        },
                        "output": {"response": "hello", "tool_calls": []},
                    },
                ],
                output_dir=tmp,
            )
            self.assertEqual(result["passed"], 1)
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["results"][0]["output"]["response"], "hello")
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())
            self.assertIn("tone", metrics["criteria"])


class TestConversationJudgeModel(unittest.IsolatedAsyncioTestCase):
    async def test_no_fallback_model_override(self):
        from calibrate.llm import run_tests as RT

        # _evaluate_conversation does not override the judge model — each
        # evaluator uses its own judge_model, otherwise the simulation judge's
        # own default applies.
        sim = AsyncMock(return_value={"tone": {"reasoning": "ok", "match": True}})
        with patch.object(RT, "evaluate_simuation", sim):
            await RT._evaluate_conversation(
                chat_history=[],
                evaluators=[_bin_ev("tone")],
                output={"response": "hi", "tool_calls": []},
                no_response_reasoning_no_tool_calls="",
            )
        self.assertNotIn("fallback_model", sim.await_args.kwargs)


class TestConversationEvalOnlyNotMutated(unittest.IsolatedAsyncioTestCase):
    async def test_captured_transcript_judged_as_is(self):
        from calibrate.llm.run_tests import run_eval_only_tests
        from calibrate.llm import run_tests as RT

        # Same preprocess as response eval-only: missing tool responses are
        # filled in before judging; existing ones are left intact.
        history = [
            {"role": "user", "content": "check my order"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "check_order", "arguments": "{}"}}
                ],
            },
        ]
        sim = AsyncMock(return_value={"tone": {"reasoning": "ok", "match": True}})
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(RT, "evaluate_simuation", sim):
            await run_eval_only_tests(
                config={"evaluators": [_bin_ev("tone")]},
                dataset=[{
                    "test_case": {
                        "id": "c1",
                        "history": history,
                        "evaluation": {"type": "conversation", "criteria": [{"name": "tone"}]},
                    },
                    "output": {"response": "It ships tomorrow.", "tool_calls": []},
                }],
                output_dir=tmp,
            )
        judged = sim.await_args.kwargs["conversation"]
        self.assertEqual(judged[2]["role"], "tool")
        self.assertEqual(judged[-1], {"role": "assistant", "content": "It ships tomorrow."})


class TestRunModelTestsConversationOnlyConfig(unittest.IsolatedAsyncioTestCase):
    async def test_no_tools_or_system_prompt_keys(self):
        from calibrate.llm import run_tests as RT

        # A conversation-only suite need not define tools or system_prompt; in
        # live mode the agent still runs to produce the next reply.
        infer = AsyncMock(return_value={
            "response": "hello", "tool_calls": [], "captured_errors": [],
        })
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(RT, "run_inference", infer), \
             patch.object(RT, "evaluate_simuation",
                          AsyncMock(return_value={"tone": {"reasoning": "ok", "match": True}})):
            config = {
                "evaluators": [_bin_ev("tone")],
                "test_cases": [{
                    "id": "c1",
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {"type": "conversation", "criteria": [{"name": "tone"}]},
                }],
            }
            result = await RT.run_model_tests(
                model="m", provider="openrouter", config=config, output_dir=tmp,
            )
        self.assertEqual(result["metrics"]["passed"], 1)


class TestAggregateLatency(unittest.TestCase):
    def test_none_when_no_latency_present(self):
        from calibrate.llm.run_tests import _aggregate_latency

        # Eval-only results carry no latency_ms.
        results = [{"metrics": {"passed": True}}, {"metrics": {"passed": False}}]
        self.assertIsNone(_aggregate_latency(results))

    def test_mean_min_max_count(self):
        from calibrate.llm.run_tests import _aggregate_latency

        results = [
            {"latency_ms": 100},
            {"latency_ms": 200},
            {"latency_ms": 600},
        ]
        self.assertEqual(
            _aggregate_latency(results),
            {"mean": 300, "min": 100, "max": 600, "count": 3},
        )

    def test_ignores_missing_and_non_numeric(self):
        from calibrate.llm.run_tests import _aggregate_latency

        # Mix of live-inference results (latency) and entries without it; bools
        # must not be treated as numbers.
        results = [
            {"latency_ms": 50},
            {"metrics": {}},
            {"latency_ms": None},
            {"latency_ms": True},
            {"latency_ms": 150},
        ]
        self.assertEqual(
            _aggregate_latency(results),
            {"mean": 100, "min": 50, "max": 150, "count": 2},
        )

    def test_mean_is_rounded(self):
        from calibrate.llm.run_tests import _aggregate_latency

        # mean 100.5 -> 100
        agg = _aggregate_latency([{"latency_ms": 100}, {"latency_ms": 101}])
        self.assertEqual(agg["mean"], 100)


class TestWriteOutputsLatency(unittest.TestCase):
    def test_metrics_includes_latency_when_present(self):
        from calibrate.llm.run_tests import _write_test_results_outputs

        results = [
            {
                "metrics": {"passed": True},
                "latency_ms": 120,
                "test_case": {"evaluation": {"type": "response"}},
            },
            {
                "metrics": {"passed": False},
                "latency_ms": 180,
                "test_case": {"evaluation": {"type": "response"}},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _write_test_results_outputs(results, tmp, {})
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())
        self.assertEqual(
            metrics["latency_ms"],
            {"mean": 150, "min": 120, "max": 180, "count": 2},
        )

    def test_metrics_omits_latency_when_absent(self):
        from calibrate.llm.run_tests import _write_test_results_outputs

        results = [
            {
                "metrics": {"passed": True},
                "test_case": {"evaluation": {"type": "response"}},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _write_test_results_outputs(results, tmp, {})
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())
        self.assertNotIn("latency_ms", metrics)


class TestWriteOutputsTotalTokens(unittest.TestCase):
    def test_metrics_includes_total_tokens_when_present(self):
        from calibrate.llm.run_tests import _write_test_results_outputs

        results = [
            {
                "metrics": {"passed": True},
                "output": {"total_tokens": 100},
                "test_case": {"evaluation": {"type": "response"}},
            },
            {
                "metrics": {"passed": False},
                "output": {"total_tokens": 200},
                "test_case": {"evaluation": {"type": "response"}},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _write_test_results_outputs(results, tmp, {})
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())
        self.assertEqual(
            metrics["total_tokens"],
            {"mean": 150, "min": 100, "max": 200, "count": 2},
        )

    def test_metrics_omits_total_tokens_when_absent(self):
        from calibrate.llm.run_tests import _write_test_results_outputs

        results = [
            {
                "metrics": {"passed": True},
                "output": {"response": "hi", "tool_calls": []},
                "test_case": {"evaluation": {"type": "response"}},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _write_test_results_outputs(results, tmp, {})
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())
        self.assertNotIn("total_tokens", metrics)


class TestRunTestsMainDebug(unittest.IsolatedAsyncioTestCase):
    async def test_eval_only_debug_truncates_dataset(self):
        from calibrate.llm import run_tests

        captured = {}

        async def fake_eval(**kwargs):
            captured["dataset"] = kwargs["dataset"]
            return {"passed": 1, "total": 1}

        dataset = [
            {
                "test_case": {"history": [], "evaluation": {}},
                "output": {"response": "x", "tool_calls": []},
            }
            for _ in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "c.json"
            cfg.write_text("{}")
            ds = Path(tmp) / "d.json"
            ds.write_text(json.dumps(dataset))
            with patch.object(sys, "argv", [
                "calibrate", "-c", str(cfg), "-o", tmp,
                "--eval-only", "--dataset", str(ds), "-d", "-dc", "2",
            ]), \
                patch("calibrate.llm.run_tests.run_eval_only_tests",
                      side_effect=fake_eval), \
                patch("calibrate.llm.run_tests.validate_llm_eval_only_dataset",
                      return_value=(True, "")):
                await run_tests.main()
        self.assertEqual(len(captured["dataset"]), 2)

    async def test_debug_truncates_config_test_cases(self):
        from calibrate.llm import run_tests

        captured = {}

        async def fake_run(**kwargs):
            captured["config"] = kwargs["config"]
            return {
                "metrics": {"passed": 1, "total": 1},
                "provider": "openrouter",
                "model": "gpt-4.1",
            }

        config = {
            "system_prompt": "sp",
            "tools": [],
            "test_cases": [{"id": str(i)} for i in range(10)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "c.json"
            cfg.write_text(json.dumps(config))
            with patch.object(sys, "argv", [
                "calibrate", "-c", str(cfg), "-o", tmp,
                "-m", "gpt-4.1", "-d", "-dc", "3",
            ]), \
                patch("calibrate.llm.run_tests.run_model_tests",
                      side_effect=fake_run):
                await run_tests.main()
        self.assertEqual(len(captured["config"]["test_cases"]), 3)


if __name__ == "__main__":
    unittest.main()
