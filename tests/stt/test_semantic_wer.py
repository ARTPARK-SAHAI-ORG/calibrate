"""Tests for the pipecat-style semantic WER flow (stt/semantic_wer + wiring).

The judge (one holistic LLM call per row) is mocked — no network. Covers the
pooled/per-row WER formula, the build_prompt template, and that
semantic WER threads through ``_score_and_write_results`` into metrics.json +
results.csv when the LLM-judge group is enabled (``run_llm_judges``).
"""

import json
import tempfile
import unicodedata
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd


class TestBuildPrompt(unittest.TestCase):
    def test_prompt_includes_pair_and_rules(self):
        from calibrate_agent.stt.semantic_wer import build_prompt

        p = build_prompt("transfer to savings", "transfer to checking")
        self.assertIn("transfer to savings", p)
        self.assertIn("transfer to checking", p)
        # Carries the pipecat-style methodology.
        self.assertIn("NORMALIZE", p)
        self.assertIn("SEMANTIC CHECK", p)
        # Prompt is pipecat's verbatim: the rules our paraphrase had dropped
        # and the full few-shot set must be present.
        self.assertIn("Compound words count as ONE error", p)
        self.assertIn("TRUNCATED/INCOMPLETE TEXT", p)
        self.assertIn("TRAILING FUNCTION WORDS AT TRUNCATION", p)
        self.assertEqual(p.count("### Example"), 8)


class TestGetSemanticWERScore(unittest.IsolatedAsyncioTestCase):
    async def test_pooled_and_per_row_formula(self):
        from calibrate_agent.stt import metrics as M

        # Row "a": 1 error / 10 ref words -> 0.1. Row "b": 0 errors / 5 -> 0.0.
        # Pooled = (1 + 0) / (10 + 5) = 0.0667. Keyed by input (gather runs the
        # mocked judges concurrently, so ordering isn't guaranteed).
        by_ref = {
            "a": {
                "substitutions": 1, "deletions": 0, "insertions": 0,
                "reference_words": 10, "normalized_reference": "r1",
                "normalized_hypothesis": "h1", "reasoning": "one sub",
            },
            "b": {
                "substitutions": 0, "deletions": 0, "insertions": 0,
                "reference_words": 5, "normalized_reference": "r2",
                "normalized_hypothesis": "h2", "reasoning": "clean",
            },
        }

        async def fake_judge(reference, prediction, model=None):
            return by_ref[reference]

        with patch(
            "calibrate_agent.stt.semantic_wer.semantic_wer_judge",
            AsyncMock(side_effect=fake_judge),
        ):
            out = await M.get_semantic_wer_score(["a", "b"], ["a", "b"])

        self.assertAlmostEqual(out["semantic_wer"], 1 / 15)
        self.assertAlmostEqual(out["per_row"][0]["semantic_wer"], 0.1)
        self.assertAlmostEqual(out["per_row"][1]["semantic_wer"], 0.0)

    async def test_empty_input(self):
        from calibrate_agent.stt import metrics as M

        out = await M.get_semantic_wer_score([], [])
        self.assertEqual(out["semantic_wer"], 0.0)
        self.assertEqual(out["per_row"], [])

    async def test_zero_reference_words_with_errors_is_inf(self):
        from calibrate_agent.stt import metrics as M

        async def fake_judge(reference, prediction, model=None):
            return {
                "substitutions": 0, "deletions": 0, "insertions": 2,
                "reference_words": 0, "normalized_reference": "",
                "normalized_hypothesis": "x y", "reasoning": "hallucinated",
            }

        with patch(
            "calibrate_agent.stt.semantic_wer.semantic_wer_judge",
            AsyncMock(side_effect=fake_judge),
        ):
            out = await M.get_semantic_wer_score([""], ["x y"])

        self.assertEqual(out["per_row"][0]["semantic_wer"], float("inf"))

    async def test_inf_rows_excluded_from_pooled(self):
        from calibrate_agent.stt import metrics as M

        # Row "ok": 1 error / 10 -> 0.1 (finite). Row "bad": 2 errors / 0 ->
        # inf. Pooled must be 1/10 = 0.1 (the inf row contributes neither its
        # 2 errors nor its 0 ref words), matching pipecat's compute_pooled_wer.
        by_ref = {
            "ok": {
                "substitutions": 1, "deletions": 0, "insertions": 0,
                "reference_words": 10, "normalized_reference": "r",
                "normalized_hypothesis": "h", "reasoning": "one sub",
            },
            "bad": {
                "substitutions": 0, "deletions": 0, "insertions": 2,
                "reference_words": 0, "normalized_reference": "",
                "normalized_hypothesis": "x y", "reasoning": "no reference",
            },
        }

        async def fake_judge(reference, prediction, model=None):
            return by_ref[reference]

        with patch(
            "calibrate_agent.stt.semantic_wer.semantic_wer_judge",
            AsyncMock(side_effect=fake_judge),
        ):
            out = await M.get_semantic_wer_score(["ok", "bad"], ["a", "b"])

        self.assertAlmostEqual(out["semantic_wer"], 0.1)
        self.assertEqual(out["per_row"][1]["semantic_wer"], float("inf"))


def _fake_completion(args: dict, content: str = "my reasoning"):
    """A minimal OpenAI-style chat completion with one calculate_wer tool call."""
    tc = SimpleNamespace(
        function=SimpleNamespace(name="calculate_wer", arguments=json.dumps(args))
    )
    msg = SimpleNamespace(content=content, tool_calls=[tc])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class TestJudgeToolLoop(unittest.IsolatedAsyncioTestCase):
    async def _run(self, ref, hyp, args, content="my reasoning"):
        from calibrate_agent.stt.semantic_wer import judge as J

        captured = {}

        async def fake_create(*a, **kw):
            captured["messages"] = kw["messages"]
            captured["tools"] = kw["tools"]
            return _fake_completion(args, content)

        fake_client = MagicMock()
        fake_client.chat.completions.create = fake_create
        with patch.object(J, "_build_openrouter_client", return_value=fake_client):
            result = await J.semantic_wer_judge(ref, hyp)
        return result, captured

    async def test_system_user_split_and_count_extraction(self):
        result, captured = await self._run(
            "transfer to savings",
            "transfer to checking",
            {
                "substitutions": 1,
                "deletions": 0,
                "insertions": 0,
                "reference_words": 3,
                "normalized_reference": "transfer to savings",
                "normalized_hypothesis": "transfer to checking",
            },
            content="savings->checking changes the account",
        )
        # System (rules) / user (pair) split — not one crammed message.
        self.assertEqual(captured["messages"][0]["role"], "system")
        self.assertEqual(captured["messages"][1]["role"], "user")
        self.assertIn("SEMANTIC CHECK", captured["messages"][0]["content"])
        self.assertIn("transfer to savings", captured["messages"][1]["content"])
        self.assertNotIn("transfer to savings", captured["messages"][0]["content"])
        # Tool schema is offered as an OpenAI function named calculate_wer.
        self.assertEqual(captured["tools"][0]["function"]["name"], "calculate_wer")
        # Counts parsed from the tool call; reasoning captured from the message.
        self.assertEqual(result["substitutions"], 1)
        self.assertEqual(result["reference_words"], 3)
        self.assertEqual(result["reasoning"], "savings->checking changes the account")

    async def test_reference_and_prediction_are_nfc_normalized(self):
        # Decomposed (NFD) input whose NFC form differs — the judge must send
        # the composed form to the model, not the decomposed input it received.
        nfd_ref = unicodedata.normalize("NFD", "café résumé")
        _, captured = await self._run(
            nfd_ref, nfd_ref,
            {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 3},
        )
        user_msg = captured["messages"][1]["content"]
        self.assertIn(unicodedata.normalize("NFC", nfd_ref), user_msg)
        self.assertNotIn(nfd_ref, user_msg)


class TestJudgeEmptyShortCircuit(unittest.IsolatedAsyncioTestCase):
    async def _run_no_llm(self, ref, hyp):
        """Run the judge asserting the model is never called."""
        from calibrate_agent.stt.semantic_wer import judge as J

        def _boom(*a, **kw):
            raise AssertionError("LLM must not be called for empty inputs")

        with patch.object(J, "_build_openrouter_client", side_effect=_boom):
            return await J.semantic_wer_judge(ref, hyp)

    async def test_both_empty(self):
        r = await self._run_no_llm("   ", "")
        self.assertEqual(
            (r["substitutions"], r["deletions"], r["insertions"], r["reference_words"]),
            (0, 0, 0, 0),
        )

    async def test_empty_reference_is_inf(self):
        # pipecat _no_reference_result: insertions = len(hyp words), ref_words 0.
        r = await self._run_no_llm("", "two extra words here")
        self.assertEqual(r["insertions"], 4)
        self.assertEqual(r["reference_words"], 0)
        # Downstream this row is inf (errors>0, ref_words==0).

    async def test_empty_hypothesis_is_wer_one(self):
        # pipecat _no_hypothesis_result: deletions = ref_words = len(ref words).
        r = await self._run_no_llm("three word reference", "  ")
        self.assertEqual(r["deletions"], 3)
        self.assertEqual(r["reference_words"], 3)  # -> per-row WER 3/3 = 1.0

    async def test_empty_rows_pool_like_pipecat(self):
        from calibrate_agent.stt import metrics as M

        by_ref = {
            "": {"substitutions": 0, "deletions": 0, "insertions": 2,
                 "reference_words": 0, "normalized_reference": "",
                 "normalized_hypothesis": "", "reasoning": "empty reference"},
            "hi there": {"substitutions": 1, "deletions": 0, "insertions": 0,
                         "reference_words": 2, "normalized_reference": "",
                         "normalized_hypothesis": "", "reasoning": "one sub"},
        }

        async def fake_judge(reference, prediction, model=None):
            return by_ref[reference]

        with patch(
            "calibrate_agent.stt.semantic_wer.semantic_wer_judge",
            AsyncMock(side_effect=fake_judge),
        ):
            out = await M.get_semantic_wer_score(["", "hi there"], ["x y", "hi bye"])

        # Empty-reference row is inf and excluded from pooling; pooled = 1/2.
        self.assertEqual(out["per_row"][0]["semantic_wer"], float("inf"))
        self.assertAlmostEqual(out["semantic_wer"], 0.5)


def _fake_judge():
    async def judge(refs, preds, evaluators=None, fallback_model=None):
        return {
            "scores": {"semantic_match": {"type": "binary", "mean": 1.0}},
            "score": 1.0,
            "per_row": [
                {"semantic_match": {"match": True, "reasoning": "ok"}} for _ in refs
            ],
        }

    return AsyncMock(side_effect=judge)


class TestScoreAndWriteSemanticWER(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_wer_in_metrics_and_results_when_enabled(self):
        from calibrate_agent.stt import eval as stt_eval

        async def fake_sem(references, predictions, model=None):
            return {
                "semantic_wer": 0.05,
                "per_row": [
                    {
                        "semantic_wer": 0.05, "substitutions": 1, "deletions": 0,
                        "insertions": 0, "reference_words": 20,
                        "normalized_reference": "r", "normalized_hypothesis": "h",
                        "reasoning": "x",
                    }
                    for _ in references
                ],
            }

        # run_llm_judges=True now also triggers the Sarvam judges — make them
        # raise so isolation skips them, keeping this test focused on semantic WER.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(
                stt_eval, "get_llm_judge_score", _fake_judge()
            ), patch.object(
                stt_eval, "get_semantic_wer_score", AsyncMock(side_effect=fake_sem)
            ), patch.object(
                stt_eval,
                "get_intent_entity_score",
                AsyncMock(side_effect=RuntimeError("skip")),
            ), patch.object(
                stt_eval,
                "get_llm_wer_cer_score",
                AsyncMock(side_effect=RuntimeError("skip")),
            ):
                metrics = await stt_eval._score_and_write_results(
                    ids=["a", "b"],
                    gt_transcripts=["hi", "there"],
                    pred_transcripts=["hi", "there"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    run_llm_judges=True,
                )

            self.assertEqual(metrics["semantic_wer"], 0.05)
            df = pd.read_csv(out / "results.csv")
            self.assertIn("semantic_wer", df.columns)
            self.assertIn("semantic_wer_metadata", df.columns)
            self.assertIn("semantic_wer_reasoning", df.columns)
            meta = json.loads(df["semantic_wer_metadata"].iloc[0])
            self.assertEqual(meta["substitutions"], 1)
            self.assertEqual(meta["reference_words"], 20)
            self.assertEqual(meta["normalized_reference"], "r")
            self.assertNotIn("reasoning", meta)

    async def test_omitted_when_disabled(self):
        from calibrate_agent.stt import eval as stt_eval

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(
                stt_eval, "get_llm_judge_score", _fake_judge()
            ), patch.object(
                stt_eval, "get_semantic_wer_score", AsyncMock()
            ) as sem_mock:
                metrics = await stt_eval._score_and_write_results(
                    ids=["a"],
                    gt_transcripts=["hi"],
                    pred_transcripts=["hi"],
                    output_dir=str(out),
                    evaluator_config_dir=str(out),
                    run_llm_judges=False,
                )

            sem_mock.assert_not_called()
            self.assertNotIn("semantic_wer", metrics)
            df = pd.read_csv(out / "results.csv")
            self.assertNotIn("semantic_wer", df.columns)


if __name__ == "__main__":
    unittest.main()
