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

    def test_strict_raises_on_existing_response(self):
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
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ]
        with self.assertRaises(ValueError):
            preprocess_conversation_history(history, tools, strict=True)

    def test_non_strict_keeps_existing(self):
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
        result = preprocess_conversation_history(history, tools, strict=False)
        # No injection — length stays at 2
        self.assertEqual(len(result), 2)

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

        result = evaluate_tool_calls([], [{"tool": "a"}])
        self.assertFalse(result["passed"])

    def test_evaluate_tool_calls_pass(self):
        from calibrate.llm.run_tests import evaluate_tool_calls

        result = evaluate_tool_calls(
            [{"tool": "a", "arguments": {}}],
            [{"tool": "a", "arguments": {}}],
        )
        self.assertTrue(result["passed"])

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

        # Same preprocess as response eval-only (strict=False): missing tool
        # responses are filled in before judging.
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


if __name__ == "__main__":
    unittest.main()
