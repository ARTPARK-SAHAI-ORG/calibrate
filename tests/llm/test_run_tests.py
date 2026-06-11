"""
Tests for multi-evaluator LLM test evaluation.

Covers:
- run_test_external with multi-evaluator returns per-evaluator judge_results
- "passed" is True iff every referenced evaluator passes (AND): binary evaluators
  must match and rating evaluators must reach ``scale_max``
- _aggregate_criteria aggregates pass rates / rating means correctly across test cases

Run with:
    python -m pytest tests/test_run_tests_multi_criteria.py -v
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock


def _make_httpx_response(body: dict, status: int = 200):
    import httpx
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = body
    mock.raise_for_status = MagicMock()
    if status >= 400:
        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status}", request=MagicMock(), response=mock
        )
    return mock


def _patch_httpx(response_body: dict, status: int = 200):
    mock_resp = _make_httpx_response(response_body, status)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return patch("httpx.AsyncClient", return_value=mock_client), mock_client


def _binary_ev(name: str) -> dict:
    return {"name": name, "system_prompt": f"eval {name}", "judge_model": "openai/gpt-4.1"}


def _rating_ev(name: str, lo: int = 1, hi: int = 5) -> dict:
    return {
        "name": name,
        "system_prompt": f"rate {name}",
        "judge_model": "openai/gpt-4.1",
        "type": "rating",
        "scale_min": lo,
        "scale_max": hi,
    }


# ---------------------------------------------------------------------------
# run_test_external with multi-evaluator
# ---------------------------------------------------------------------------


class TestRunTestExternalMultiCriteria(unittest.IsolatedAsyncioTestCase):

    async def _run(self, agent_response, evaluators, judge_result, criteria=None):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import run_test_external

        agent = TextAgentConnection(url="http://fake-agent/chat")
        fake_body = {"response": agent_response, "tool_calls": []}
        evaluation = {
            "type": "response",
            "criteria": criteria
            or [{"name": ev["name"]} for ev in evaluators],
        }

        mock_judge = AsyncMock(return_value=judge_result)
        ctx, _ = _patch_httpx(fake_body)
        with ctx, patch(
            "calibrate.llm.run_tests.test_response_llm_judge", mock_judge
        ):
            return await run_test_external(
                chat_history=[{"role": "user", "content": "Hi"}],
                evaluation=evaluation,
                agent=agent,
                evaluators=evaluators,
            )

    async def test_no_latency_without_agent_reported_metrics(self):
        # Calibrate no longer times external agents itself (a round-trip timer
        # would fold in network/proxy overhead, not true inference time). With no
        # agent-reported metrics, the result simply has no latency. The
        # agent-reported latency path is covered in tests/test_connections.py.
        result = await self._run(
            agent_response="Hello, how can I help?",
            evaluators=[_binary_ev("greeting")],
            judge_result={"greeting": {"match": True, "reasoning": "greeted"}},
        )
        self.assertNotIn("latency_ms", result)

    async def test_all_evaluators_match_passes(self):
        result = await self._run(
            agent_response="Hello, how can I help?",
            evaluators=[_binary_ev("greeting"), _binary_ev("helpful")],
            judge_result={
                "greeting": {"match": True, "reasoning": "greeted"},
                "helpful": {"match": True, "reasoning": "offered help"},
            },
        )
        self.assertTrue(result["metrics"]["passed"])
        self.assertIn("judge_results", result["metrics"])
        self.assertEqual(
            result["metrics"]["judge_results"]["greeting"]["match"], True
        )
        self.assertEqual(
            result["metrics"]["judge_results"]["helpful"]["match"], True
        )

    async def test_one_evaluator_fails_overall_fails(self):
        result = await self._run(
            agent_response="Hello.",
            evaluators=[_binary_ev("greeting"), _binary_ev("helpful")],
            judge_result={
                "greeting": {"match": True, "reasoning": "greeted"},
                "helpful": {"match": False, "reasoning": "did not offer help"},
            },
        )
        self.assertFalse(result["metrics"]["passed"])
        # Reasoning surfaces the first failing evaluator
        self.assertEqual(
            result["metrics"]["reasoning"], "did not offer help"
        )

    async def test_all_fail(self):
        result = await self._run(
            agent_response="go away",
            evaluators=[_binary_ev("greeting"), _binary_ev("helpful")],
            judge_result={
                "greeting": {"match": False, "reasoning": "no greeting"},
                "helpful": {"match": False, "reasoning": "not helpful"},
            },
        )
        self.assertFalse(result["metrics"]["passed"])

    async def test_all_pass_reasoning_message(self):
        result = await self._run(
            agent_response="Hello, how can I help?",
            evaluators=[_binary_ev("greeting")],
            judge_result={
                "greeting": {"match": True, "reasoning": "greeted"},
            },
        )
        self.assertEqual(
            result["metrics"]["reasoning"], "All evaluators passed"
        )

    async def test_rating_below_scale_max_fails_test_case(self):
        """A rating evaluator below scale_max fails the test case."""
        result = await self._run(
            agent_response="Hello!",
            evaluators=[_rating_ev("fluency", lo=1, hi=5)],
            judge_result={"fluency": {"score": 2, "reasoning": "meh"}},
        )
        self.assertFalse(result["metrics"]["passed"])
        self.assertEqual(result["metrics"]["reasoning"], "meh")
        self.assertEqual(
            result["metrics"]["judge_results"]["fluency"]["score"], 2
        )

    async def test_rating_at_scale_max_passes_test_case(self):
        """A rating evaluator equal to scale_max passes the test case."""
        result = await self._run(
            agent_response="Hello!",
            evaluators=[_rating_ev("fluency", lo=1, hi=5)],
            judge_result={"fluency": {"score": 5, "reasoning": "great"}},
        )
        self.assertTrue(result["metrics"]["passed"])
        self.assertEqual(
            result["metrics"]["judge_results"]["fluency"]["score"], 5
        )

    async def test_mixed_binary_fails_overrides_rating(self):
        """A failing binary evaluator fails the test even if rating is high."""
        result = await self._run(
            agent_response="wrong answer",
            evaluators=[_binary_ev("accuracy"), _rating_ev("fluency")],
            judge_result={
                "accuracy": {"match": False, "reasoning": "wrong"},
                "fluency": {"score": 5, "reasoning": "very fluent"},
            },
        )
        self.assertFalse(result["metrics"]["passed"])
        self.assertEqual(result["metrics"]["reasoning"], "wrong")


# ---------------------------------------------------------------------------
# _aggregate_criteria helper
# ---------------------------------------------------------------------------


class TestAggregateCriteria(unittest.TestCase):
    def _registry(self, *evaluators) -> dict:
        from calibrate.judges import DEFAULT_LLM_TEST_EVALUATOR
        reg = {DEFAULT_LLM_TEST_EVALUATOR["name"]: DEFAULT_LLM_TEST_EVALUATOR}
        for ev in evaluators:
            reg[ev["name"]] = ev
        return reg

    def test_empty_list_returns_empty_dict(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        self.assertEqual(_aggregate_criteria([], self._registry()), {})

    def test_tool_call_tests_excluded(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        results = [
            {
                "metrics": {"passed": True},
                "test_case": {"evaluation": {"type": "tool_call"}},
            },
            {
                "metrics": {"passed": False},
                "test_case": {"evaluation": {"type": "tool_call"}},
            },
        ]
        self.assertEqual(_aggregate_criteria(results, self._registry()), {})

    def test_string_criteria_aggregates_under_default_evaluator(self):
        """String criteria are normalized to the implicit ``correctness`` evaluator."""
        from calibrate.llm.run_tests import _aggregate_criteria
        results = [
            {
                "metrics": {
                    "passed": True,
                    "reasoning": "ok",
                    "judge_results": {
                        "correctness": {"match": True, "reasoning": "ok"},
                    },
                },
                "test_case": {"evaluation": {"type": "response", "criteria": "X"}},
            },
            {
                "metrics": {
                    "passed": False,
                    "reasoning": "bad",
                    "judge_results": {
                        "correctness": {"match": False, "reasoning": "bad"},
                    },
                },
                "test_case": {"evaluation": {"type": "response", "criteria": "Y"}},
            },
        ]
        agg = _aggregate_criteria(results, self._registry())
        self.assertIn("correctness", agg)
        self.assertEqual(agg["correctness"]["passed"], 1)
        self.assertEqual(agg["correctness"]["total"], 2)
        self.assertEqual(agg["correctness"]["pass_rate"], 50.0)

    def test_multi_evaluators_counted_independently(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        accuracy = _binary_ev("accuracy")
        tone = _binary_ev("tone")
        results = [
            {
                "metrics": {
                    "passed": False,
                    "reasoning": "x",
                    "judge_results": {
                        "accuracy": {"match": True, "reasoning": "ok"},
                        "tone": {"match": False, "reasoning": "rude"},
                    },
                },
                "test_case": {
                    "evaluation": {
                        "type": "response",
                        "criteria": [
                            {"name": "accuracy"},
                            {"name": "tone"},
                        ],
                    }
                },
            },
        ]
        agg = _aggregate_criteria(results, self._registry(accuracy, tone))
        self.assertEqual(
            agg["accuracy"],
            {"type": "binary", "passed": 1, "total": 1, "pass_rate": 100.0},
        )
        self.assertEqual(
            agg["tone"],
            {"type": "binary", "passed": 0, "total": 1, "pass_rate": 0.0},
        )

    def test_rating_evaluator_aggregates_mean(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        fluency = _rating_ev("fluency")
        results = [
            {
                "metrics": {
                    "passed": True,
                    "reasoning": "All evaluators passed",
                    "judge_results": {
                        "fluency": {"score": 4, "reasoning": "ok"},
                    },
                },
                "test_case": {
                    "evaluation": {
                        "type": "response",
                        "criteria": [{"name": "fluency"}],
                    }
                },
            },
            {
                "metrics": {
                    "passed": True,
                    "reasoning": "All evaluators passed",
                    "judge_results": {
                        "fluency": {"score": 2, "reasoning": "ok"},
                    },
                },
                "test_case": {
                    "evaluation": {
                        "type": "response",
                        "criteria": [{"name": "fluency"}],
                    }
                },
            },
        ]
        agg = _aggregate_criteria(results, self._registry(fluency))
        self.assertEqual(
            agg["fluency"],
            {
                "type": "rating",
                "mean": 3.0,
                "min": 2,
                "max": 4,
                "count": 2,
                "scale_min": 1,
                "scale_max": 5,
            },
        )

    def test_mixed_binary_and_rating_evaluators(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        accuracy = _binary_ev("accuracy")
        fluency = _rating_ev("fluency")
        results = [
            {
                "metrics": {
                    "passed": True,
                    "reasoning": "All evaluators passed",
                    "judge_results": {
                        "accuracy": {"match": True, "reasoning": "ok"},
                        "fluency": {"score": 5, "reasoning": "ok"},
                    },
                },
                "test_case": {
                    "evaluation": {
                        "type": "response",
                        "criteria": [
                            {"name": "accuracy"},
                            {"name": "fluency"},
                        ],
                    }
                },
            },
        ]
        agg = _aggregate_criteria(results, self._registry(accuracy, fluency))
        self.assertEqual(agg["accuracy"]["type"], "binary")
        self.assertEqual(agg["accuracy"]["pass_rate"], 100.0)
        self.assertEqual(agg["fluency"]["type"], "rating")
        self.assertEqual(agg["fluency"]["mean"], 5.0)
        self.assertEqual(agg["fluency"]["scale_min"], 1)
        self.assertEqual(agg["fluency"]["scale_max"], 5)

    def test_mixed_string_and_multi_evaluator_criteria(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        accuracy = _binary_ev("accuracy")
        tone = _binary_ev("tone")
        results = [
            # Response test — string criteria (passes)
            {
                "metrics": {
                    "passed": True,
                    "reasoning": "ok",
                    "judge_results": {
                        "correctness": {"match": True, "reasoning": "ok"},
                    },
                },
                "test_case": {"evaluation": {"type": "response", "criteria": "X"}},
            },
            # Response test — multi-evaluator (one passes, one fails)
            {
                "metrics": {
                    "passed": False,
                    "reasoning": "rude",
                    "judge_results": {
                        "accuracy": {"match": True, "reasoning": "ok"},
                        "tone": {"match": False, "reasoning": "rude"},
                    },
                },
                "test_case": {
                    "evaluation": {
                        "type": "response",
                        "criteria": [
                            {"name": "accuracy"},
                            {"name": "tone"},
                        ],
                    }
                },
            },
            # Tool call test — skipped
            {
                "metrics": {"passed": True},
                "test_case": {"evaluation": {"type": "tool_call"}},
            },
        ]
        agg = _aggregate_criteria(results, self._registry(accuracy, tone))
        self.assertEqual(set(agg.keys()), {"correctness", "accuracy", "tone"})
        self.assertEqual(agg["correctness"]["total"], 1)
        self.assertEqual(agg["accuracy"]["total"], 1)
        self.assertEqual(agg["tone"]["total"], 1)


# ---------------------------------------------------------------------------
# Legacy "default" alias in evaluators registry
# ---------------------------------------------------------------------------


class TestEvaluatorsRegistryLegacyDefaultAlias(unittest.TestCase):
    """Pre-rename, the implicit default evaluator was named ``"default"``.
    The registry keeps it as an alias so older user configs that reference
    ``{"name": "default"}`` continue to resolve to the implicit default.
    """

    def test_default_alias_resolves_to_implicit_default(self):
        from calibrate.llm.run_tests import _get_name_to_evaluator_dict
        from calibrate.judges import DEFAULT_LLM_TEST_EVALUATOR

        registry = _get_name_to_evaluator_dict({})
        self.assertIn("default", registry)
        self.assertIs(registry["default"], DEFAULT_LLM_TEST_EVALUATOR)
        # Canonical name still present.
        self.assertIn(DEFAULT_LLM_TEST_EVALUATOR["name"], registry)

    def test_user_evaluator_named_default_overrides_alias(self):
        from calibrate.llm.run_tests import _get_name_to_evaluator_dict

        custom = {
            "name": "default",
            "system_prompt": "user-defined override",
            "judge_model": "openai/gpt-4.1",
        }
        registry = _get_name_to_evaluator_dict({"evaluators": [custom]})
        # User override wins.
        self.assertIs(registry["default"], custom)

    def test_default_alias_can_be_referenced_in_test_case(self):
        """A test case's ``criteria`` referencing ``{"name": "default"}`` must
        still render successfully and produce the implicit default evaluator."""
        from calibrate.llm.run_tests import (
            _get_name_to_evaluator_dict,
            _resolve_evaluators_for_test_case,
        )
        from calibrate.judges import DEFAULT_LLM_TEST_EVALUATOR

        evaluation = {
            "type": "response",
            "criteria": [{"name": "default", "arguments": {"criteria": "be polite"}}],
        }
        rendered = _resolve_evaluators_for_test_case(
            evaluation, _get_name_to_evaluator_dict({"evaluators": []})
        )
        self.assertEqual(len(rendered), 1)
        # Resolved evaluator carries the canonical name (not the alias).
        self.assertEqual(rendered[0]["name"], DEFAULT_LLM_TEST_EVALUATOR["name"])
        self.assertIn("be polite", rendered[0]["system_prompt"])


if __name__ == "__main__":
    unittest.main()
