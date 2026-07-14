"""
Tests for the LLM-WER/CER judge aggregation.

``get_llm_wer_cer_score`` lives in ``calibrate_agent/stt/metrics.py`` (the metric
root). It normalizes reference/prediction via the vendored ``IndicNormalizer``
(mocked here to avoid downloading a model), word-aligns each pair with
``difflib.SequenceMatcher``, judges each *differing* segment for equivalence via
``calibrate_agent/stt/sarvam_llm_wer/judge.py`` (mocked), forgives equivalent
segments, and re-scores WER/CER with calibrate_agent's own jiwer scorer.

Run with:
    python -m unittest tests.stt.test_llm_wer -v
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock


def _identity_normalizer():
    """Mock normalizer whose normalize_texts returns inputs unchanged."""
    inst = MagicMock()
    inst.normalize_texts.side_effect = lambda texts, langs, n_jobs=1: list(texts)
    return inst


class TestGetLlmWerCerScore(unittest.IsolatedAsyncioTestCase):
    async def test_forgiveness_lowers_wer(self):
        from calibrate_agent.stt import sarvam_llm_wer as slw
        from calibrate_agent.stt import metrics

        # ("doctor","daktar") is a phonetic match -> forgiven; ("world","word")
        # is a genuine error -> kept.
        verdicts = {
            ("doctor", "daktar"): True,
            ("world", "word"): False,
        }

        async def fake_judge(reference, prediction, model=None):
            return {
                "index": 0,
                "equivalent": verdicts[(reference, prediction)],
                "reasoning": "because",
            }

        with patch.object(metrics, "_get_indic_normalizer", return_value=_identity_normalizer()), \
             patch.object(slw, "equivalence_judge", AsyncMock(side_effect=fake_judge)):
            result = await metrics.get_llm_wer_cer_score(
                references=["doctor ne bola", "hello world"],
                predictions=["daktar ne bola", "hello word"],
            )

        # Row 0 fully forgiven (0 errors); row 1 keeps 1 substitution.
        # Pooled over 5 reference words -> 1/5.
        self.assertAlmostEqual(result["llm_wer"], 0.2, places=6)
        self.assertEqual(len(result["per_row"]), 2)
        self.assertAlmostEqual(result["per_row"][0]["llm_wer"], 0.0, places=6)
        self.assertAlmostEqual(result["per_row"][1]["llm_wer"], 0.5, places=6)
        # Row 0's differing segment was forgiven; row 1's was kept as an error.
        self.assertTrue(result["per_row"][0]["segments"][0]["equivalent"])
        self.assertFalse(result["per_row"][1]["segments"][0]["equivalent"])

    async def test_unique_segments_judged_once(self):
        from calibrate_agent.stt import sarvam_llm_wer as slw
        from calibrate_agent.stt import metrics

        seen = []

        async def fake_judge(reference, prediction, model=None):
            seen.append((reference, prediction))
            return {"index": 0, "equivalent": True, "reasoning": "r"}

        # The same differing segment ("a","b") appears in both rows -> judged once.
        with patch.object(metrics, "_get_indic_normalizer", return_value=_identity_normalizer()), \
             patch.object(slw, "equivalence_judge", AsyncMock(side_effect=fake_judge)):
            await metrics.get_llm_wer_cer_score(
                references=["a x", "a y"],
                predictions=["b x", "b y"],
            )

        self.assertEqual(seen.count(("a", "b")), 1)

    async def test_insertions_and_deletions_not_judged(self):
        from calibrate_agent.stt import sarvam_llm_wer as slw
        from calibrate_agent.stt import metrics

        judge = AsyncMock()
        # Pure deletion (pred shorter) and pure insertion (pred longer) — no
        # ``replace`` segment with both sides non-empty, so nothing to judge.
        with patch.object(metrics, "_get_indic_normalizer", return_value=_identity_normalizer()), \
             patch.object(slw, "equivalence_judge", judge):
            result = await metrics.get_llm_wer_cer_score(
                references=["one two three", "one"],
                predictions=["one two", "one two"],
            )

        judge.assert_not_called()
        # Nothing forgiven -> corrected pairs equal the originals: 1 deletion +
        # 1 insertion over 4 reference words.
        self.assertAlmostEqual(result["llm_wer"], 0.5, places=6)

    async def test_empty_inputs(self):
        from calibrate_agent.stt import sarvam_llm_wer as slw
        from calibrate_agent.stt import metrics

        with patch.object(metrics, "_get_indic_normalizer", return_value=_identity_normalizer()), \
             patch.object(slw, "equivalence_judge", AsyncMock()):
            result = await metrics.get_llm_wer_cer_score(references=[], predictions=[])

        self.assertEqual(result["llm_wer"], 0.0)
        self.assertEqual(result["llm_cer"], 0.0)
        self.assertEqual(result["per_row"], [])

    async def test_normalized_text_reaches_judge(self):
        from calibrate_agent.stt import sarvam_llm_wer as slw
        from calibrate_agent.stt import metrics

        norm_inst = MagicMock()
        norm_inst.normalize_texts.side_effect = lambda texts, langs, n_jobs=1: [
            t.lower() for t in texts
        ]

        seen = []

        async def fake_judge(reference, prediction, model=None):
            seen.append((reference, prediction))
            return {"index": 0, "equivalent": False, "reasoning": "r"}

        with patch.object(metrics, "_get_indic_normalizer", return_value=norm_inst), \
             patch.object(slw, "equivalence_judge", AsyncMock(side_effect=fake_judge)):
            await metrics.get_llm_wer_cer_score(
                references=["HELLO WORLD"],
                predictions=["HELLO WORD"],
            )

        self.assertEqual(seen, [("world", "word")])


class TestGetSegments(unittest.TestCase):
    def test_replace_segment_extracted(self):
        from calibrate_agent.stt.sarvam_llm_wer import get_segments

        segs = get_segments("doctor ne bola", "daktar ne bola", key=0)
        replaces = [s for s in segs if s["tag"] == "replace"]
        self.assertEqual(len(replaces), 1)
        self.assertEqual(replaces[0]["reference"], "doctor")
        self.assertEqual(replaces[0]["prediction"], "daktar")

    def test_both_empty_returns_no_segments(self):
        from calibrate_agent.stt.sarvam_llm_wer import get_segments

        self.assertEqual(get_segments("", "", key=0), [])


class TestLazyImports(unittest.TestCase):
    def test_importing_stt_does_not_load_llm_wer_stack(self):
        import subprocess
        import sys

        code = (
            "import sys\n"
            "import calibrate_agent.stt.benchmark\n"
            "import calibrate_agent.stt.eval\n"
            "import calibrate_agent.stt.metrics\n"
            "assert 'calibrate_agent.stt.sarvam_llm_wer' not in sys.modules, 'llm_wer pkg loaded'\n"
            "print('ok')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)


class TestEquivalenceJudge(unittest.IsolatedAsyncioTestCase):
    async def test_judge_builds_prompt_and_returns_model_dump(self):
        from calibrate_agent.stt.sarvam_llm_wer import judge as jw

        fake_result = {"index": 2, "equivalent": True, "reasoning": "phonetic"}
        fake_response = MagicMock()
        fake_response.model_dump.return_value = fake_result

        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

        inner = jw.equivalence_judge
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__

        with patch.object(jw, "_build_openrouter_client", return_value=MagicMock()), \
             patch.object(jw.instructor, "apatch", return_value=fake_client):
            result = await inner("doctor", "daktar", model="m")

        self.assertEqual(result, fake_result)
        _, kwargs = fake_client.chat.completions.create.call_args
        sent = kwargs["messages"][0]["content"]
        self.assertIn('"reference": "doctor"', sent)
        self.assertIn('"prediction": "daktar"', sent)
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(kwargs["response_model"], jw.LLMEquivalenceResponse)


if __name__ == "__main__":
    unittest.main()
