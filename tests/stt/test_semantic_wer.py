"""Tests for the pipecat-style semantic WER flow (stt/semantic_wer + wiring).

The judge (one holistic LLM call per row) is mocked — no network. Covers the
pooled/per-row WER formula, the build_prompt template, and that
semantic WER threads through ``_score_and_write_results`` into metrics.json +
results.csv when the LLM-judge group is enabled (``run_llm_judges``).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
            self.assertIn("semantic_wer_substitutions", df.columns)
            self.assertIn("semantic_wer_reasoning", df.columns)

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
