"""
Unit tests for calibrate_agent/judge_store.py — the append-only judge
checkpoint and its gather helpers.

Run with:
    python -m pytest tests/test_judge_store.py -v
"""

import asyncio
import json
import os
import tempfile
import unittest

from calibrate_agent.judge_store import (
    JudgeKey,
    JudgeStore,
    gather_evaluators_with_store,
    gather_with_store,
    make_fingerprint,
)


class TestMakeFingerprint(unittest.TestCase):
    def test_stable_for_equal_inputs(self):
        a = make_fingerprint("reference text", "prediction text", "prompt", "model")
        b = make_fingerprint("reference text", "prediction text", "prompt", "model")
        self.assertEqual(a, b)

    def test_differs_for_different_inputs(self):
        a = make_fingerprint("reference text", "prediction text", "prompt", "model")
        b = make_fingerprint("reference text", "different prediction", "prompt", "model")
        self.assertNotEqual(a, b)

    def test_differs_when_prompt_or_model_changes(self):
        base = make_fingerprint("input", "prompt v1", "model-a")
        different_prompt = make_fingerprint("input", "prompt v2", "model-a")
        different_model = make_fingerprint("input", "prompt v1", "model-b")
        self.assertNotEqual(base, different_prompt)
        self.assertNotEqual(base, different_model)

    def test_insensitive_to_dict_key_order(self):
        a = make_fingerprint({"reference": "r", "prediction": "p"})
        b = make_fingerprint({"prediction": "p", "reference": "r"})
        self.assertEqual(a, b)

    def test_returns_hex_sha256(self):
        digest = make_fingerprint("x")
        self.assertEqual(len(digest), 64)
        int(digest, 16)  # raises ValueError if not valid hex


class TestJudgeKey(unittest.TestCase):
    def test_row_id_coerced_to_str(self):
        key = JudgeKey(kind="evaluators", row_id=1, fingerprint="abc")
        self.assertEqual(key.row_id, "1")
        self.assertIsInstance(key.row_id, str)

    def test_int_and_str_row_id_produce_equal_keys(self):
        key_int = JudgeKey(kind="llm_wer", row_id=1, fingerprint="abc")
        key_str = JudgeKey(kind="llm_wer", row_id="1", fingerprint="abc")
        self.assertEqual(key_int, key_str)
        self.assertEqual(hash(key_int), hash(key_str))

    def test_frozen_and_hashable(self):
        key = JudgeKey(kind="llm_wer", row_id="1", fingerprint="abc")
        with self.assertRaises(Exception):
            key.row_id = "2"
        {key: "usable as dict key"}  # does not raise


class JudgeStoreTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.output_dir = self._tmpdir.name


class TestJudgeStorePersistence(JudgeStoreTestBase):
    async def test_put_then_get_round_trips(self):
        store = JudgeStore.load(self.output_dir)
        key = JudgeKey(kind="evaluators", row_id="row_a", fingerprint="fp1", evaluator="semantic_match")
        result = {"reasoning": "matches", "match": True}

        await store.put(key, result)

        self.assertEqual(store.get(key), result)

    async def test_load_from_fresh_instance_sees_prior_records(self):
        store = JudgeStore.load(self.output_dir)
        key = JudgeKey(kind="llm_wer", row_id="row_a", fingerprint="fp1")
        result = {"score": 0.5}
        await store.put(key, result)

        reloaded = JudgeStore.load(self.output_dir)

        self.assertEqual(reloaded.get(key), result)
        self.assertEqual(len(reloaded), 1)

    async def test_creates_output_dir_if_missing(self):
        nested = os.path.join(self.output_dir, "nested", "deeper")
        self.assertFalse(os.path.exists(nested))

        store = JudgeStore.load(nested)

        self.assertTrue(os.path.isdir(nested))
        self.assertEqual(store.path, os.path.join(nested, JudgeStore.FILENAME))

    async def test_row_id_int_and_str_resolve_same_key(self):
        store = JudgeStore.load(self.output_dir)
        write_key = JudgeKey(kind="llm_wer", row_id=1, fingerprint="fp1")
        await store.put(write_key, {"score": 1.0})

        lookup_key = JudgeKey(kind="llm_wer", row_id="1", fingerprint="fp1")

        self.assertEqual(store.get(lookup_key), {"score": 1.0})

    async def test_non_ascii_reasoning_survives_round_trip(self):
        store = JudgeStore.load(self.output_dir)
        key = JudgeKey(kind="evaluators", row_id="row_a", fingerprint="fp1", evaluator="semantic_match")
        result = {"reasoning": "यह सही उत्तर है, ಸರಿಯಾಗಿದೆ", "match": True}
        await store.put(key, result)

        with open(store.path, encoding="utf-8") as f:
            raw = f.read()
        self.assertIn("यह सही उत्तर है", raw)
        self.assertIn("ಸರಿಯಾಗಿದೆ", raw)
        self.assertNotIn("\\u", raw)

        reloaded = JudgeStore.load(self.output_dir)
        self.assertEqual(reloaded.get(key), result)

    async def test_truncated_final_line_is_skipped_not_fatal(self):
        store = JudgeStore.load(self.output_dir)
        good_key = JudgeKey(kind="llm_wer", row_id="row_a", fingerprint="fp1")
        await store.put(good_key, {"score": 0.1})

        with open(store.path, "a", encoding="utf-8") as f:
            f.write('{"kind": "llm_wer", "row_id": "row_b", "fingerpr')  # truncated, no newline

        reloaded = JudgeStore.load(self.output_dir)

        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded.get(good_key), {"score": 0.1})

    async def test_stale_fingerprint_never_returned_for_current_key(self):
        store = JudgeStore.load(self.output_dir)
        stale_key = JudgeKey(kind="llm_wer", row_id="row_a", fingerprint="old-fp")
        await store.put(stale_key, {"score": 0.9})

        current_key = JudgeKey(kind="llm_wer", row_id="row_a", fingerprint="new-fp")

        self.assertIsNone(store.get(current_key))

    async def test_pending_returns_only_uncached_keys_in_order(self):
        store = JudgeStore.load(self.output_dir)
        k1 = JudgeKey(kind="llm_wer", row_id="1", fingerprint="fp1")
        k2 = JudgeKey(kind="llm_wer", row_id="2", fingerprint="fp2")
        k3 = JudgeKey(kind="llm_wer", row_id="3", fingerprint="fp3")
        await store.put(k2, {"score": 0.0})

        result = store.pending([k1, k2, k3])

        self.assertEqual(result, [k1, k3])

    async def test_clear_removes_file_and_empties_store(self):
        store = JudgeStore.load(self.output_dir)
        key = JudgeKey(kind="llm_wer", row_id="1", fingerprint="fp1")
        await store.put(key, {"score": 0.0})
        self.assertTrue(os.path.exists(store.path))

        store.clear()

        self.assertFalse(os.path.exists(store.path))
        self.assertEqual(len(store), 0)
        self.assertIsNone(store.get(key))

    async def test_concurrent_put_produces_that_many_valid_lines(self):
        store = JudgeStore.load(self.output_dir)
        n = 25
        keys = [JudgeKey(kind="llm_wer", row_id=str(i), fingerprint=f"fp{i}") for i in range(n)]

        await asyncio.gather(*[store.put(k, {"score": i}) for i, k in enumerate(keys)])

        with open(store.path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line]

        self.assertEqual(len(lines), n)
        for line in lines:
            json.loads(line)  # each line parses independently
        self.assertEqual(len(store), n)


class TestGatherWithStore(JudgeStoreTestBase):
    async def test_results_returned_in_row_order(self):
        store = JudgeStore.load(self.output_dir)
        keys = [JudgeKey(kind="llm_wer", row_id=str(i), fingerprint=f"fp{i}") for i in range(3)]

        async def run_one(index):
            await asyncio.sleep(0.01 * (2 - index))  # complete out of order
            return {"score": index}

        results = await gather_with_store(keys, run_one, store, desc="test")

        self.assertEqual(results, [{"score": 0}, {"score": 1}, {"score": 2}])

    async def test_run_one_skipped_for_cached_rows(self):
        store = JudgeStore.load(self.output_dir)
        keys = [JudgeKey(kind="llm_wer", row_id=str(i), fingerprint=f"fp{i}") for i in range(3)]
        await store.put(keys[1], {"score": "cached"})

        calls = []

        async def run_one(index):
            calls.append(index)
            return {"score": f"fresh-{index}"}

        results = await gather_with_store(keys, run_one, store, desc="test")

        self.assertEqual(sorted(calls), [0, 2])
        self.assertEqual(results[1], {"score": "cached"})
        self.assertEqual(results[0], {"score": "fresh-0"})
        self.assertEqual(results[2], {"score": "fresh-2"})

    async def test_fresh_results_are_persisted_immediately(self):
        store = JudgeStore.load(self.output_dir)
        keys = [JudgeKey(kind="llm_wer", row_id="a", fingerprint="fp1")]

        async def run_one(index):
            return {"score": 1.0}

        await gather_with_store(keys, run_one, store, desc="test")

        reloaded = JudgeStore.load(self.output_dir)
        self.assertEqual(reloaded.get(keys[0]), {"score": 1.0})

    async def test_store_none_runs_every_row(self):
        keys = [JudgeKey(kind="llm_wer", row_id=str(i), fingerprint=f"fp{i}") for i in range(3)]
        calls = []

        async def run_one(index):
            calls.append(index)
            return {"score": index}

        results = await gather_with_store(keys, run_one, None, desc="test")

        self.assertEqual(sorted(calls), [0, 1, 2])
        self.assertEqual(results, [{"score": 0}, {"score": 1}, {"score": 2}])

    async def test_none_key_forces_row_to_run(self):
        store = JudgeStore.load(self.output_dir)
        keys = [JudgeKey(kind="llm_wer", row_id="0", fingerprint="fp0"), None]
        await store.put(keys[0], {"score": "cached"})
        calls = []

        async def run_one(index):
            calls.append(index)
            return {"score": f"fresh-{index}"}

        results = await gather_with_store(keys, run_one, store, desc="test")

        self.assertEqual(calls, [1])
        self.assertEqual(results, [{"score": "cached"}, {"score": "fresh-1"}])

    async def test_exception_propagates_but_stored_results_persist(self):
        store = JudgeStore.load(self.output_dir)
        keys = [JudgeKey(kind="llm_wer", row_id=str(i), fingerprint=f"fp{i}") for i in range(3)]

        async def run_one(index):
            if index == 1:
                await asyncio.sleep(0.05)
                raise ValueError("judge call failed")
            return {"score": index}

        with self.assertRaises(ValueError):
            await gather_with_store(keys, run_one, store, desc="test")

        # Give the still-running background tasks (if any) a chance to finish.
        await asyncio.sleep(0.1)

        reloaded = JudgeStore.load(self.output_dir)
        self.assertEqual(reloaded.get(keys[0]), {"score": 0})
        self.assertEqual(reloaded.get(keys[2]), {"score": 2})
        self.assertIsNone(reloaded.get(keys[1]))


class TestGatherEvaluatorsWithStore(JudgeStoreTestBase):
    async def test_fully_cached_row_skips_run_subset(self):
        store = JudgeStore.load(self.output_dir)
        row_keys = [
            {
                "semantic_match": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp-sm", evaluator="semantic_match"),
                "fluency": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp-fl", evaluator="fluency"),
            }
        ]
        await store.put(row_keys[0]["semantic_match"], {"match": True, "reasoning": "ok"})
        await store.put(row_keys[0]["fluency"], {"match": False, "reasoning": "meh"})

        async def run_subset(index, names):
            raise AssertionError("run_subset should not be called for a fully cached row")

        results = await gather_evaluators_with_store(row_keys, run_subset, store, desc="test")

        self.assertEqual(
            results,
            [{"semantic_match": {"match": True, "reasoning": "ok"}, "fluency": {"match": False, "reasoning": "meh"}}],
        )

    async def test_partially_cached_row_runs_only_missing_names(self):
        store = JudgeStore.load(self.output_dir)
        row_keys = [
            {
                "semantic_match": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp-sm", evaluator="semantic_match"),
                "fluency": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp-fl", evaluator="fluency"),
            }
        ]
        await store.put(row_keys[0]["semantic_match"], {"match": True, "reasoning": "cached"})
        calls = []

        async def run_subset(index, names):
            calls.append((index, list(names)))
            return {name: {"match": True, "reasoning": f"fresh-{name}"} for name in names}

        results = await gather_evaluators_with_store(row_keys, run_subset, store, desc="test")

        self.assertEqual(calls, [(0, ["fluency"])])
        self.assertEqual(
            results,
            [
                {
                    "semantic_match": {"match": True, "reasoning": "cached"},
                    "fluency": {"match": True, "reasoning": "fresh-fluency"},
                }
            ],
        )

    async def test_fresh_evaluator_results_are_persisted(self):
        store = JudgeStore.load(self.output_dir)
        row_keys = [
            {"semantic_match": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp-sm", evaluator="semantic_match")}
        ]

        async def run_subset(index, names):
            return {name: {"match": True, "reasoning": "fresh"} for name in names}

        await gather_evaluators_with_store(row_keys, run_subset, store, desc="test")

        reloaded = JudgeStore.load(self.output_dir)
        self.assertEqual(
            reloaded.get(row_keys[0]["semantic_match"]),
            {"match": True, "reasoning": "fresh"},
        )

    async def test_results_in_row_order(self):
        store = JudgeStore.load(self.output_dir)
        row_keys = [
            {"a": JudgeKey(kind="evaluators", row_id=str(i), fingerprint=f"fp{i}", evaluator="a")}
            for i in range(3)
        ]

        async def run_subset(index, names):
            await asyncio.sleep(0.01 * (2 - index))  # complete out of order
            return {name: {"match": True, "reasoning": f"row-{index}"} for name in names}

        results = await gather_evaluators_with_store(row_keys, run_subset, store, desc="test")

        self.assertEqual(
            [r["a"]["reasoning"] for r in results],
            ["row-0", "row-1", "row-2"],
        )

    async def test_new_evaluator_reruns_only_that_evaluator_across_rows(self):
        store = JudgeStore.load(self.output_dir)
        row_keys = [
            {"semantic_match": JudgeKey(kind="evaluators", row_id=str(i), fingerprint=f"fp{i}", evaluator="semantic_match")}
            for i in range(2)
        ]

        async def run_subset_first_pass(index, names):
            return {name: {"match": True, "reasoning": "pass1"} for name in names}

        await gather_evaluators_with_store(row_keys, run_subset_first_pass, store, desc="test")

        # Second pass adds a "fluency" evaluator to every row.
        row_keys_v2 = [
            {
                **row_keys[i],
                "fluency": JudgeKey(kind="evaluators", row_id=str(i), fingerprint=f"fp-fl-{i}", evaluator="fluency"),
            }
            for i in range(2)
        ]
        calls = []

        async def run_subset_second_pass(index, names):
            calls.append((index, list(names)))
            return {name: {"match": True, "reasoning": "pass2"} for name in names}

        results = await gather_evaluators_with_store(row_keys_v2, run_subset_second_pass, store, desc="test")

        self.assertEqual(sorted(calls), [(0, ["fluency"]), (1, ["fluency"])])
        for r in results:
            self.assertEqual(r["semantic_match"]["reasoning"], "pass1")
            self.assertEqual(r["fluency"]["reasoning"], "pass2")

    async def test_store_none_runs_every_evaluator_for_every_row(self):
        row_keys = [
            {"a": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp0", evaluator="a")}
        ]
        calls = []

        async def run_subset(index, names):
            calls.append((index, list(names)))
            return {name: {"match": True, "reasoning": "fresh"} for name in names}

        results = await gather_evaluators_with_store(row_keys, run_subset, None, desc="test")

        self.assertEqual(calls, [(0, ["a"])])
        self.assertEqual(results, [{"a": {"match": True, "reasoning": "fresh"}}])

    async def test_missing_name_in_run_subset_result_raises(self):
        store = JudgeStore.load(self.output_dir)
        row_keys = [
            {
                "semantic_match": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp-sm", evaluator="semantic_match"),
                "fluency": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp-fl", evaluator="fluency"),
            }
        ]

        async def run_subset(index, names):
            return {"semantic_match": {"match": True, "reasoning": "ok"}}  # missing "fluency"

        with self.assertRaises(KeyError):
            await gather_evaluators_with_store(row_keys, run_subset, store, desc="test")


if __name__ == "__main__":
    unittest.main()
