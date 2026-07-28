"""Tests for calibrate_agent/judge_cost.py — token heuristic, cost math, confirmation gate."""

import io
import os
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch


class TestEstimateTokens(unittest.TestCase):
    def test_ascii_text_uses_the_latin_rate(self):
        from calibrate_agent.judge_cost import estimate_tokens

        self.assertEqual(estimate_tokens("a" * 20), 5)

    def test_devanagari_text_uses_the_indic_rate(self):
        from calibrate_agent.judge_cost import estimate_tokens

        self.assertEqual(estimate_tokens("क" * 20), 14)

    def test_mixed_script_lands_between_the_pure_cases(self):
        from calibrate_agent.judge_cost import estimate_tokens

        ascii_rate = estimate_tokens("a" * 20) / 20
        indic_rate = estimate_tokens("क" * 20) / 20
        mixed_rate = estimate_tokens("a" * 10 + "क" * 10) / 20

        self.assertLess(ascii_rate, mixed_rate)
        self.assertLess(mixed_rate, indic_rate)

    def test_empty_and_whitespace_text_is_zero_tokens(self):
        from calibrate_agent.judge_cost import estimate_tokens

        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("   \n\t "), 0)

    def test_short_text_rounds_up_to_one_token(self):
        from calibrate_agent.judge_cost import estimate_tokens

        self.assertEqual(estimate_tokens("a"), 1)


class TestEstimateAudioSecondsFromText(unittest.TestCase):
    def test_derives_seconds_from_the_token_estimate(self):
        from calibrate_agent.judge_cost import (
            SPOKEN_TOKENS_PER_SECOND,
            estimate_audio_seconds_from_text,
            estimate_tokens,
        )

        text = "hello there, this is a spoken sentence"
        self.assertAlmostEqual(
            estimate_audio_seconds_from_text(text),
            estimate_tokens(text) / SPOKEN_TOKENS_PER_SECOND,
        )

    def test_empty_text_is_zero_seconds(self):
        from calibrate_agent.judge_cost import estimate_audio_seconds_from_text

        self.assertEqual(estimate_audio_seconds_from_text(""), 0.0)


class TestEstimateJudgeCost(unittest.TestCase):
    def test_prices_a_text_only_group(self):
        from calibrate_agent.judge_cost import JudgeCallGroup, estimate_judge_cost

        group = JudgeCallGroup(
            label="semantic WER",
            model="anthropic/claude-sonnet-4.5",
            calls=10,
            input_tokens_per_call=1000,
            output_tokens_per_call=200,
        )

        estimate = estimate_judge_cost([group])
        row = estimate["groups"][0]

        # (10_000 * $3.00 + 2_000 * $15.00) per million tokens.
        self.assertEqual(row["input_tokens"], 10_000)
        self.assertEqual(row["output_tokens"], 2_000)
        self.assertEqual(row["audio_tokens"], 0)
        self.assertTrue(row["priced"])
        self.assertAlmostEqual(row["cost_usd"], 0.06)
        self.assertAlmostEqual(estimate["total_usd"], 0.06)
        self.assertEqual(estimate["source"], "openrouter")
        self.assertEqual(estimate["unpriced"], [])

    def test_both_sources_price_the_same_group(self):
        from calibrate_agent.judge_cost import (
            JudgeCallGroup,
            estimate_judge_cost,
            estimate_judge_cost_all_sources,
        )

        group = JudgeCallGroup(
            label="correctness",
            model="openai/gpt-5.4-mini",
            calls=4,
            input_tokens_per_call=500,
            output_tokens_per_call=100,
        )

        direct = estimate_judge_cost([group], source="direct")
        self.assertEqual(direct["source"], "direct")
        # (2_000 * $0.75 + 400 * $4.50) per million tokens.
        self.assertAlmostEqual(direct["total_usd"], 0.0033)

        both = estimate_judge_cost_all_sources([group])
        self.assertEqual(sorted(both), ["direct", "openrouter"])
        self.assertAlmostEqual(
            both["openrouter"]["total_usd"], both["direct"]["total_usd"]
        )

    def test_unpriced_model_is_reported_without_breaking_the_total(self):
        from calibrate_agent.judge_cost import JudgeCallGroup, estimate_judge_cost

        priced = JudgeCallGroup(
            label="semantic WER",
            model="anthropic/claude-sonnet-4.5",
            calls=10,
            input_tokens_per_call=1000,
            output_tokens_per_call=200,
        )
        unpriced = JudgeCallGroup(
            label="custom judge",
            model="acme/does-not-exist",
            calls=5,
            input_tokens_per_call=1000,
            output_tokens_per_call=200,
        )

        estimate = estimate_judge_cost([priced, unpriced])
        unpriced_row = estimate["groups"][1]

        self.assertFalse(unpriced_row["priced"])
        self.assertEqual(unpriced_row["cost_usd"], 0.0)
        self.assertEqual(unpriced_row["input_tokens"], 5_000)
        self.assertEqual(estimate["unpriced"], ["acme/does-not-exist"])
        self.assertAlmostEqual(estimate["total_usd"], 0.06)

    def test_unpriced_models_are_deduped_and_sorted(self):
        from calibrate_agent.judge_cost import JudgeCallGroup, estimate_judge_cost

        groups = [
            JudgeCallGroup("judge b", "zeta/unknown", 1, 10, 10),
            JudgeCallGroup("judge a", "acme/unknown", 1, 10, 10),
            JudgeCallGroup("judge c", "zeta/unknown", 1, 10, 10),
        ]

        estimate = estimate_judge_cost(groups)

        self.assertEqual(estimate["unpriced"], ["acme/unknown", "zeta/unknown"])
        self.assertEqual(estimate["total_usd"], 0.0)

    def test_audio_seconds_bill_at_the_audio_rate(self):
        from calibrate_agent.judge_cost import JudgeCallGroup, estimate_judge_cost

        group = JudgeCallGroup(
            label="pronunciation",
            model="openai/gpt-audio",
            calls=2,
            input_tokens_per_call=100,
            output_tokens_per_call=50,
            audio_seconds_per_call=3.0,
        )

        row = estimate_judge_cost([group])["groups"][0]

        # 2 calls * 3s * 10 tokens/s.
        self.assertEqual(row["audio_tokens"], 60)
        # (200 * $2.50 + 100 * $10.00 + 60 * $32.00) per million tokens.
        self.assertAlmostEqual(row["cost_usd"], 0.00342)

        text_rate_cost = (200 * 2.5 + 100 * 10.0 + 60 * 2.5) / 1_000_000
        self.assertGreater(row["cost_usd"], text_rate_cost)

    def test_audio_falls_back_to_the_text_input_rate(self):
        from calibrate_agent.judge_cost import JudgeCallGroup, estimate_judge_cost

        group = JudgeCallGroup(
            label="text-only judge given audio",
            model="anthropic/claude-sonnet-4.5",
            calls=1,
            input_tokens_per_call=0,
            output_tokens_per_call=0,
            audio_seconds_per_call=2.0,
        )

        row = estimate_judge_cost([group])["groups"][0]

        self.assertEqual(row["audio_tokens"], 20)
        # 20 tokens * $3.00 per million, the model's text input rate.
        self.assertAlmostEqual(row["cost_usd"], 20 * 3.0 / 1_000_000)

    def test_reasoning_multiplier_applies_only_to_reasoning_billed_models(self):
        from calibrate_agent.judge_cost import (
            REASONING_TOKEN_MULTIPLIER,
            JudgeCallGroup,
            estimate_judge_cost,
        )

        gemini = JudgeCallGroup(
            label="LLM WER",
            model="google/gemini-2.5-flash",
            calls=1,
            input_tokens_per_call=1000,
            output_tokens_per_call=100,
        )
        mini = JudgeCallGroup(
            label="correctness",
            model="openai/gpt-5.4-mini",
            calls=1,
            input_tokens_per_call=1000,
            output_tokens_per_call=100,
        )

        gemini_cost = estimate_judge_cost([gemini])["total_usd"]
        mini_cost = estimate_judge_cost([mini])["total_usd"]

        # (1_000 * $0.30 + 100 * 3.0 * $2.50) per million tokens.
        self.assertAlmostEqual(gemini_cost, 0.00105)
        self.assertAlmostEqual(
            gemini_cost,
            (1000 * 0.3 + 100 * REASONING_TOKEN_MULTIPLIER * 2.5) / 1_000_000,
        )
        # (1_000 * $0.75 + 100 * $4.50) per million tokens, unscaled.
        self.assertAlmostEqual(mini_cost, 0.0012)

    def test_reported_output_tokens_exclude_the_reasoning_allowance(self):
        from calibrate_agent.judge_cost import JudgeCallGroup, estimate_judge_cost

        group = JudgeCallGroup("LLM WER", "google/gemini-2.5-flash", 3, 100, 40)

        row = estimate_judge_cost([group])["groups"][0]

        self.assertEqual(row["output_tokens"], 120)

    def test_empty_group_list(self):
        from calibrate_agent.judge_cost import estimate_judge_cost

        estimate = estimate_judge_cost([])

        self.assertEqual(estimate["groups"], [])
        self.assertEqual(estimate["total_usd"], 0.0)
        self.assertEqual(estimate["unpriced"], [])


class TestFormatCostEstimate(unittest.TestCase):
    def test_names_every_group_and_both_totals(self):
        from calibrate_agent.judge_cost import (
            JudgeCallGroup,
            estimate_judge_cost_all_sources,
            format_cost_estimate,
        )

        groups = [
            JudgeCallGroup(
                "semantic WER", "anthropic/claude-sonnet-4.5", 10, 1000, 200
            ),
            JudgeCallGroup("pronunciation", "openai/gpt-audio", 2, 100, 50, 3.0),
        ]

        text = format_cost_estimate(estimate_judge_cost_all_sources(groups))

        self.assertIn("semantic WER", text)
        self.assertIn("anthropic/claude-sonnet-4.5", text)
        self.assertIn("pronunciation", text)
        self.assertIn("openai/gpt-audio", text)
        self.assertIn("OpenRouter", text)
        self.assertIn("direct", text)
        self.assertIn("$0.0634", text)
        self.assertIn("audio", text)
        self.assertNotIn("unpriced", text)

    def test_names_unpriced_models(self):
        from calibrate_agent.judge_cost import (
            JudgeCallGroup,
            estimate_judge_cost_all_sources,
            format_cost_estimate,
        )

        groups = [JudgeCallGroup("custom judge", "acme/does-not-exist", 5, 100, 20)]

        text = format_cost_estimate(estimate_judge_cost_all_sources(groups))

        self.assertIn("acme/does-not-exist", text)
        self.assertIn("unpriced", text)

    def test_mentions_the_estimate_caveats(self):
        from calibrate_agent.judge_cost import (
            JudgeCallGroup,
            estimate_judge_cost_all_sources,
            format_cost_estimate,
        )

        groups = [JudgeCallGroup("correctness", "openai/gpt-5.4-mini", 1, 100, 20)]

        text = format_cost_estimate(estimate_judge_cost_all_sources(groups))

        self.assertIn("Estimated from a bundled rate table", text)
        self.assertIn("approximate", text)


class _FakeStdin:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty

    def fileno(self) -> int:
        return 0


class TestConfirmJudgeCost(unittest.TestCase):
    def setUp(self):
        from calibrate_agent.judge_cost import (
            JudgeCallGroup,
            estimate_judge_cost_all_sources,
        )

        group = JudgeCallGroup(
            "semantic WER", "anthropic/claude-sonnet-4.5", 10, 1000, 200
        )
        self.both = estimate_judge_cost_all_sources([group])

    def _invoke(
        self,
        assume_yes=False,
        tty=True,
        env_assume_yes=None,
        tcgetpgrp=None,
        answer="",
        answer_error=None,
    ):
        from calibrate_agent.judge_cost import confirm_judge_cost

        env = {k: v for k, v in os.environ.items() if k != "CALIBRATE_ASSUME_YES"}
        if env_assume_yes is not None:
            env["CALIBRATE_ASSUME_YES"] = env_assume_yes

        err_stream = io.StringIO()
        out_stream = io.StringIO()
        input_mock = MagicMock(return_value=answer, side_effect=answer_error)

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, env, clear=True))
            stack.enter_context(patch.object(sys, "stdin", _FakeStdin(tty)))
            stack.enter_context(patch.object(sys, "stdout", out_stream))
            stack.enter_context(patch("builtins.input", input_mock))
            if tcgetpgrp is not None:
                stack.enter_context(patch.object(os, "tcgetpgrp", tcgetpgrp))
                stack.enter_context(
                    patch.object(os, "getpgrp", MagicMock(return_value=4242))
                )
            decision = confirm_judge_cost(
                self.both, assume_yes=assume_yes, stream=err_stream
            )

        return decision, err_stream.getvalue(), out_stream.getvalue(), input_mock

    def assertEstimateWentToStreamOnly(self, err_text: str, out_text: str):
        self.assertIn("semantic WER", err_text)
        self.assertIn("$0.0600", err_text)
        self.assertEqual(out_text, "")

    def test_assume_yes_proceeds_without_asking(self):
        decision, err_text, out_text, input_mock = self._invoke(assume_yes=True)

        self.assertTrue(decision)
        input_mock.assert_not_called()
        self.assertEstimateWentToStreamOnly(err_text, out_text)

    def test_env_var_proceeds_without_asking(self):
        decision, err_text, out_text, input_mock = self._invoke(env_assume_yes="1")

        self.assertTrue(decision)
        input_mock.assert_not_called()
        self.assertEstimateWentToStreamOnly(err_text, out_text)

    def test_empty_env_var_still_asks(self):
        decision, _, _, input_mock = self._invoke(
            env_assume_yes="", tcgetpgrp=MagicMock(return_value=4242), answer="y"
        )

        self.assertTrue(decision)
        input_mock.assert_called_once()

    def test_non_tty_stdin_proceeds_without_asking(self):
        decision, err_text, out_text, input_mock = self._invoke(tty=False)

        self.assertTrue(decision)
        input_mock.assert_not_called()
        self.assertEstimateWentToStreamOnly(err_text, out_text)

    def test_background_process_group_proceeds_without_asking(self):
        decision, err_text, out_text, input_mock = self._invoke(
            tcgetpgrp=MagicMock(return_value=1)
        )

        self.assertTrue(decision)
        input_mock.assert_not_called()
        self.assertEstimateWentToStreamOnly(err_text, out_text)

    def test_foreground_process_group_asks(self):
        decision, err_text, out_text, input_mock = self._invoke(
            tcgetpgrp=MagicMock(return_value=4242), answer="y"
        )

        self.assertTrue(decision)
        input_mock.assert_called_once()
        self.assertEstimateWentToStreamOnly(err_text, out_text)

    def test_missing_tcgetpgrp_falls_back_to_isatty(self):
        decision, _, _, input_mock = self._invoke(
            tcgetpgrp=MagicMock(side_effect=AttributeError), answer="y"
        )
        self.assertTrue(decision)
        input_mock.assert_called_once()

        decision, err_text, out_text, input_mock = self._invoke(
            tty=False, tcgetpgrp=MagicMock(side_effect=AttributeError)
        )
        self.assertTrue(decision)
        input_mock.assert_not_called()
        self.assertEstimateWentToStreamOnly(err_text, out_text)

    def test_failing_tcgetpgrp_falls_back_to_isatty(self):
        decision, _, _, input_mock = self._invoke(
            tcgetpgrp=MagicMock(side_effect=OSError), answer="n"
        )

        self.assertFalse(decision)
        input_mock.assert_called_once()

    def test_yes_answers_proceed(self):
        for answer in ("y", "Y", "yes", "YES", " y "):
            with self.subTest(answer=answer):
                decision, _, _, input_mock = self._invoke(
                    tcgetpgrp=MagicMock(return_value=4242), answer=answer
                )
                self.assertTrue(decision)
                input_mock.assert_called_once()

    def test_no_answer_cancels(self):
        decision, err_text, out_text, input_mock = self._invoke(
            tcgetpgrp=MagicMock(return_value=4242), answer="n"
        )

        self.assertFalse(decision)
        input_mock.assert_called_once()
        self.assertEstimateWentToStreamOnly(err_text, out_text)
        self.assertIn("Proceed with the judge run?", err_text)

    def test_empty_answer_cancels(self):
        decision, _, _, _ = self._invoke(
            tcgetpgrp=MagicMock(return_value=4242), answer=""
        )

        self.assertFalse(decision)

    def test_end_of_input_cancels(self):
        decision, err_text, out_text, input_mock = self._invoke(
            tcgetpgrp=MagicMock(return_value=4242), answer_error=EOFError
        )

        self.assertFalse(decision)
        input_mock.assert_called_once()
        self.assertEstimateWentToStreamOnly(err_text, out_text)

    def test_interrupt_cancels(self):
        decision, _, _, _ = self._invoke(
            tcgetpgrp=MagicMock(return_value=4242), answer_error=KeyboardInterrupt
        )

        self.assertFalse(decision)

    def test_defaults_to_stderr(self):
        from calibrate_agent.judge_cost import confirm_judge_cost

        err_stream = io.StringIO()
        out_stream = io.StringIO()
        with patch.object(sys, "stderr", err_stream), patch.object(
            sys, "stdout", out_stream
        ):
            decision = confirm_judge_cost(self.both, assume_yes=True)

        self.assertTrue(decision)
        self.assertEstimateWentToStreamOnly(
            err_stream.getvalue(), out_stream.getvalue()
        )


class TestBuildSttJudgeGroups(unittest.TestCase):
    def setUp(self):
        self.references = ["hello world", "how are you today"]
        self.predictions = ["hello world", "how are you today"]

    def test_no_evaluators_and_no_llm_judges_yields_nothing(self):
        from calibrate_agent.judge_cost import build_stt_judge_groups

        groups = build_stt_judge_groups(
            self.references, self.predictions, evaluators=None, run_llm_judges=False
        )

        self.assertEqual(groups, [])

    def test_run_llm_judges_produces_the_three_built_in_judges(self):
        from calibrate_agent.judge_cost import build_stt_judge_groups

        groups = build_stt_judge_groups(
            self.references, self.predictions, evaluators=None, run_llm_judges=True
        )
        labels = [g.label for g in groups]

        self.assertIn("Sarvam intent/entity", labels)
        self.assertIn("Sarvam LLM-WER/CER", labels)
        self.assertIn("Semantic WER (reasoning)", labels)
        self.assertIn("Semantic WER (commit)", labels)
        # No evaluator group since none were supplied.
        self.assertFalse(any("evaluator" in label.lower() and "Sarvam" not in label and "Semantic" not in label for label in labels))

    def test_run_llm_judges_false_drops_the_three_built_in_judges(self):
        from calibrate_agent.judge_cost import build_stt_judge_groups

        evaluators = [
            {"name": "semantic_match", "system_prompt": "You are a judge."}
        ]
        groups = build_stt_judge_groups(
            self.references,
            self.predictions,
            evaluators=evaluators,
            run_llm_judges=False,
        )
        labels = [g.label for g in groups]

        self.assertEqual(len(groups), 1)
        self.assertIn("semantic_match", labels[0])

    def test_semantic_wer_issues_two_calls_per_row(self):
        from calibrate_agent.judge_cost import build_stt_judge_groups

        groups = build_stt_judge_groups(
            self.references, self.predictions, evaluators=None, run_llm_judges=True
        )
        by_label = {g.label: g for g in groups}

        n = len(self.references)
        self.assertEqual(by_label["Semantic WER (reasoning)"].calls, n)
        self.assertEqual(by_label["Semantic WER (commit)"].calls, n)
        self.assertEqual(by_label["Sarvam intent/entity"].calls, n)
        self.assertEqual(by_label["Sarvam LLM-WER/CER"].calls, n)

    def test_mixed_judge_models_produce_separate_evaluator_groups(self):
        from calibrate_agent.judge_cost import build_stt_judge_groups

        evaluators = [
            {
                "name": "correctness",
                "system_prompt": "Judge A",
                "judge_model": "openai/gpt-5.4-mini",
            },
            {
                "name": "fluency",
                "system_prompt": "Judge B",
                "judge_model": "anthropic/claude-sonnet-4.5",
            },
        ]
        groups = build_stt_judge_groups(
            self.references,
            self.predictions,
            evaluators=evaluators,
            run_llm_judges=False,
        )
        models = {g.model for g in groups}

        self.assertEqual(len(groups), 2)
        self.assertEqual(models, {"openai/gpt-5.4-mini", "anthropic/claude-sonnet-4.5"})

    def test_providers_multiplies_call_counts(self):
        from calibrate_agent.judge_cost import build_stt_judge_groups

        evaluators = [{"name": "correctness", "system_prompt": "Judge A"}]
        base = build_stt_judge_groups(
            self.references,
            self.predictions,
            evaluators=evaluators,
            run_llm_judges=True,
            providers=1,
        )
        tripled = build_stt_judge_groups(
            self.references,
            self.predictions,
            evaluators=evaluators,
            run_llm_judges=True,
            providers=3,
        )
        base_by_label = {g.label: g.calls for g in base}
        tripled_by_label = {g.label: g.calls for g in tripled}

        for label, calls in base_by_label.items():
            self.assertEqual(tripled_by_label[label], calls * 3)

    def test_predictions_none_still_produces_a_sane_estimate(self):
        from calibrate_agent.judge_cost import build_stt_judge_groups

        evaluators = [{"name": "correctness", "system_prompt": "Judge A"}]
        groups = build_stt_judge_groups(
            self.references,
            predictions=None,
            evaluators=evaluators,
            run_llm_judges=True,
        )

        self.assertTrue(groups)
        for g in groups:
            self.assertGreater(g.calls, 0)
            self.assertGreater(g.input_tokens_per_call, 0)

    def test_empty_references_yields_no_calls(self):
        from calibrate_agent.judge_cost import build_stt_judge_groups

        groups = build_stt_judge_groups([], evaluators=None, run_llm_judges=True)

        self.assertEqual(groups, [])


class TestBuildTtsJudgeGroups(unittest.TestCase):
    def test_default_evaluator_one_call_per_row(self):
        from calibrate_agent.judge_cost import build_tts_judge_groups

        texts = ["hello world", "this is a test", "another sentence here"]
        groups = build_tts_judge_groups(texts)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].calls, len(texts))

    def test_multiple_evaluators_one_call_per_row_each(self):
        from calibrate_agent.judge_cost import build_tts_judge_groups

        texts = ["hello world", "this is a test"]
        evaluators = [
            {"name": "pronunciation", "system_prompt": "Judge A"},
            {"name": "naturalness", "system_prompt": "Judge B"},
        ]
        groups = build_tts_judge_groups(texts, evaluators=evaluators)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].calls, len(texts) * len(evaluators))

    def test_mixed_judge_models_produce_separate_groups(self):
        from calibrate_agent.judge_cost import build_tts_judge_groups

        texts = ["hello world", "this is a test"]
        evaluators = [
            {
                "name": "pronunciation",
                "system_prompt": "Judge A",
                "judge_model": "openai/gpt-audio",
            },
            {
                "name": "naturalness",
                "system_prompt": "Judge B",
                "judge_model": "google/gemini-2.5-flash",
            },
        ]
        groups = build_tts_judge_groups(texts, evaluators=evaluators)
        models = {g.model for g in groups}

        self.assertEqual(len(groups), 2)
        self.assertEqual(models, {"openai/gpt-audio", "google/gemini-2.5-flash"})

    def test_audio_seconds_passed_through_when_given(self):
        from calibrate_agent.judge_cost import build_tts_judge_groups

        texts = ["hello world", "this is a test"]
        durations = [1.5, 4.5]
        groups = build_tts_judge_groups(texts, audio_seconds=durations)

        self.assertAlmostEqual(groups[0].audio_seconds_per_call, sum(durations) / len(durations))

    def test_audio_seconds_derived_from_text_when_missing(self):
        from calibrate_agent.judge_cost import (
            build_tts_judge_groups,
            estimate_audio_seconds_from_text,
        )

        texts = ["hello world", "this is a longer test sentence"]
        groups = build_tts_judge_groups(texts, audio_seconds=None)

        expected = sum(estimate_audio_seconds_from_text(t) for t in texts) / len(texts)
        self.assertAlmostEqual(groups[0].audio_seconds_per_call, expected)

    def test_providers_multiplies_call_counts(self):
        from calibrate_agent.judge_cost import build_tts_judge_groups

        texts = ["hello world", "this is a test"]
        base = build_tts_judge_groups(texts, providers=1)
        tripled = build_tts_judge_groups(texts, providers=3)

        self.assertEqual(tripled[0].calls, base[0].calls * 3)

    def test_empty_texts_yields_no_calls(self):
        from calibrate_agent.judge_cost import build_tts_judge_groups

        groups = build_tts_judge_groups([])

        self.assertEqual(groups, [])


class TestConfirmEstimatedJudgeCost(unittest.TestCase):
    def _group(self):
        from calibrate_agent.judge_cost import JudgeCallGroup

        return JudgeCallGroup(
            label="evaluators",
            model="openai/gpt-5.4-mini",
            calls=10,
            input_tokens_per_call=100,
            output_tokens_per_call=50,
        )

    def test_a_plan_that_raises_proceeds(self):
        from calibrate_agent.judge_cost import confirm_estimated_judge_cost

        def plan():
            raise FileNotFoundError("dataset is not there")

        stream = io.StringIO()
        self.assertTrue(confirm_estimated_judge_cost(plan, stream=stream))
        self.assertEqual(stream.getvalue(), "")

    def test_no_groups_proceeds_without_an_estimate(self):
        from calibrate_agent.judge_cost import confirm_estimated_judge_cost

        stream = io.StringIO()
        self.assertTrue(
            confirm_estimated_judge_cost(lambda: ([], 0), stream=stream)
        )
        self.assertEqual(stream.getvalue(), "")

    def test_estimate_is_shown_and_confirmed(self):
        from calibrate_agent.judge_cost import confirm_estimated_judge_cost

        stream = io.StringIO()
        result = confirm_estimated_judge_cost(
            lambda: ([self._group()], 0), assume_yes=True, stream=stream
        )

        self.assertTrue(result)
        self.assertIn("evaluators", stream.getvalue())

    def test_declining_is_reported(self):
        from calibrate_agent.judge_cost import confirm_estimated_judge_cost

        stream = io.StringIO()
        with patch("calibrate_agent.judge_cost.confirm_judge_cost", return_value=False):
            result = confirm_estimated_judge_cost(
                lambda: ([self._group()], 0), stream=stream
            )

        self.assertFalse(result)

    def test_cached_count_is_reported(self):
        from calibrate_agent.judge_cost import confirm_estimated_judge_cost

        stream = io.StringIO()
        with patch("builtins.print") as fake_print:
            confirm_estimated_judge_cost(
                lambda: ([self._group()], 7), assume_yes=True, stream=stream
            )

        printed = " ".join(str(call) for call in fake_print.call_args_list)
        self.assertIn("7", printed)


if __name__ == "__main__":
    unittest.main()
