"""Tests for calibrate_agent/judge_store.py — the resumable judge checkpoint."""

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
    def test_deterministic_across_calls(self):
        first = make_fingerprint("reference", "prediction", "openai/gpt-4.1")
        second = make_fingerprint("reference", "prediction", "openai/gpt-4.1")
        self.assertEqual(first, second)

    def test_differs_when_any_part_differs(self):
        base = make_fingerprint("reference", "prediction", "openai/gpt-4.1")
        self.assertNotEqual(base, make_fingerprint("other", "prediction", "openai/gpt-4.1"))
        self.assertNotEqual(base, make_fingerprint("reference", "other", "openai/gpt-4.1"))
        self.assertNotEqual(base, make_fingerprint("reference", "prediction", "other-model"))

    def test_dict_part_order_independent(self):
        a = make_fingerprint("ref", {"scale_min": 1, "scale_max": 5, "name": "x"})
        b = make_fingerprint("ref", {"name": "x", "scale_max": 5, "scale_min": 1})
        self.assertEqual(a, b)

    def test_separator_prevents_split_collision(self):
        # Without a separator between parts, ("ab", "c") and ("a", "bc") would
        # encode to the same concatenated string.
        self.assertNotEqual(make_fingerprint("ab", "c"), make_fingerprint("a", "bc"))

    def test_non_json_serializable_uses_str_fallback(self):
        class Unserializable:
            def __str__(self):
                return "unserializable-repr"

        fingerprint = make_fingerprint(Unserializable())
        self.assertIsInstance(fingerprint, str)
        # default=str means the object's str() is what actually gets hashed.
        self.assertEqual(fingerprint, make_fingerprint("unserializable-repr"))

    def test_non_json_serializable_nested_in_dict(self):
        class Unserializable:
            def __str__(self):
                return "nested-repr"

        fingerprint = make_fingerprint({"value": Unserializable()})
        self.assertEqual(fingerprint, make_fingerprint({"value": "nested-repr"}))


class TestJudgeKey(unittest.TestCase):
    def test_int_and_str_row_id_are_equal_and_hash_equal(self):
        int_key = JudgeKey(kind="stt", row_id=1, fingerprint="fp")
        str_key = JudgeKey(kind="stt", row_id="1", fingerprint="fp")
        self.assertEqual(int_key, str_key)
        self.assertEqual(hash(int_key), hash(str_key))

    def test_row_id_coerced_to_str(self):
        key = JudgeKey(kind="stt", row_id=42, fingerprint="fp")
        self.assertEqual(key.row_id, "42")
        self.assertIsInstance(key.row_id, str)

    def test_keys_differing_only_in_evaluator_are_distinct(self):
        key_a = JudgeKey(kind="evaluators", row_id="1", fingerprint="fp", evaluator="a")
        key_b = JudgeKey(kind="evaluators", row_id="1", fingerprint="fp", evaluator="b")
        no_evaluator = JudgeKey(kind="evaluators", row_id="1", fingerprint="fp")
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(key_a, no_evaluator)

    def test_usable_as_dict_key(self):
        key = JudgeKey(kind="stt", row_id=1, fingerprint="fp")
        mapping = {key: "value"}
        self.assertEqual(mapping[JudgeKey(kind="stt", row_id="1", fingerprint="fp")], "value")

    def test_usable_in_set(self):
        keys = {
            JudgeKey(kind="stt", row_id=1, fingerprint="fp"),
            JudgeKey(kind="stt", row_id="1", fingerprint="fp"),
        }
        self.assertEqual(len(keys), 1)


class TestJudgeStoreLoad(unittest.TestCase):
    def test_load_creates_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "nested", "out")
            self.assertFalse(os.path.exists(target))

            store = JudgeStore.load(target)

            self.assertTrue(os.path.isdir(target))
            self.assertEqual(len(store), 0)

    def test_path_points_at_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            self.assertEqual(store.path, os.path.join(tmp, JudgeStore.FILENAME))

    def test_get_unknown_key_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            self.assertIsNone(store.get(JudgeKey(kind="stt", row_id="1", fingerprint="fp")))

    def test_load_skips_blank_missing_key_and_truncated_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, JudgeStore.FILENAME)
            valid_line = json.dumps({
                "kind": "stt",
                "row_id": "1",
                "evaluator": None,
                "fingerprint": "fp1",
                "result": {"score": 1},
            })
            missing_result_key = json.dumps({
                "kind": "stt",
                "row_id": "2",
                "evaluator": None,
                "fingerprint": "fp2",
            })
            with open(path, "w", encoding="utf-8") as f:
                f.write(valid_line + "\n")
                f.write("\n")
                f.write(missing_result_key + "\n")
                f.write('{"kind": "stt", "row_id": "3", "fingerp')  # truncated, no newline

            store = JudgeStore.load(tmp)

            self.assertEqual(len(store), 1)
            self.assertEqual(
                store.get(JudgeKey(kind="stt", row_id="1", fingerprint="fp1")),
                {"score": 1},
            )


class TestJudgeStorePutAndGet(unittest.IsolatedAsyncioTestCase):
    async def test_put_then_get_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            key = JudgeKey(kind="stt", row_id="1", fingerprint="fp")

            await store.put(key, {"match": True, "reasoning": "ok"})

            self.assertEqual(store.get(key), {"match": True, "reasoning": "ok"})
            self.assertEqual(len(store), 1)

    async def test_second_load_sees_results_from_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_run = JudgeStore.load(tmp)
            key = JudgeKey(kind="stt", row_id="1", fingerprint="fp")
            await first_run.put(key, {"score": 1})

            resumed_run = JudgeStore.load(tmp)

            self.assertEqual(resumed_run.get(key), {"score": 1})
            self.assertEqual(len(resumed_run), 1)

    async def test_non_ascii_content_survives_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            key = JudgeKey(kind="stt", row_id="1", fingerprint="fp")
            result = {"reasoning": "पूरी तरह सही उत्तर है", "match": True}

            await store.put(key, result)
            reloaded = JudgeStore.load(tmp)

            self.assertEqual(reloaded.get(key), result)
            with open(store.path, encoding="utf-8") as f:
                raw = f.read()
            self.assertIn("पूरी तरह सही उत्तर है", raw)

    async def test_changed_fingerprint_is_a_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            old_key = JudgeKey(kind="stt", row_id="1", fingerprint="fp-before-edit")
            await store.put(old_key, {"score": 1})

            new_key = JudgeKey(kind="stt", row_id="1", fingerprint="fp-after-edit")

            self.assertIsNone(store.get(new_key))
            self.assertEqual(store.pending([new_key]), [new_key])

    async def test_pending_returns_uncached_keys_in_input_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            key1 = JudgeKey(kind="stt", row_id="1", fingerprint="fp1")
            key2 = JudgeKey(kind="stt", row_id="2", fingerprint="fp2")
            key3 = JudgeKey(kind="stt", row_id="3", fingerprint="fp3")
            await store.put(key2, {"score": 2})

            self.assertEqual(store.pending([key3, key1, key2]), [key3, key1])

    async def test_clear_removes_file_and_empties_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            key = JudgeKey(kind="stt", row_id="1", fingerprint="fp")
            await store.put(key, {"score": 1})
            self.assertTrue(os.path.exists(store.path))

            store.clear()

            self.assertFalse(os.path.exists(store.path))
            self.assertEqual(len(store), 0)
            self.assertIsNone(store.get(key))

    async def test_clear_on_never_written_store_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            store.clear()
            self.assertEqual(len(store), 0)

    async def test_concurrent_puts_all_survive_reload_without_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            keys = [
                JudgeKey(kind="stt", row_id=str(i), fingerprint=f"fp{i}")
                for i in range(50)
            ]

            await asyncio.gather(*[store.put(k, {"score": i}) for i, k in enumerate(keys)])

            with open(store.path, encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
            self.assertEqual(len(lines), 50)
            for line in lines:
                json.loads(line)  # each line is exactly one complete JSON object

            reloaded = JudgeStore.load(tmp)
            self.assertEqual(len(reloaded), 50)
            for i, key in enumerate(keys):
                self.assertEqual(reloaded.get(key), {"score": i})


class TestGatherWithStore(unittest.IsolatedAsyncioTestCase):
    async def test_results_returned_in_row_order_despite_scrambled_completion(self):
        n = 5
        keys = [None] * n

        async def run_one(index):
            # Later rows finish first; the helper must still return in row order.
            await asyncio.sleep((n - index) * 0.01)
            return {"index": index}

        results = await gather_with_store(keys, run_one, None, desc="test")

        self.assertEqual([r["index"] for r in results], list(range(n)))

    async def test_fully_cached_run_never_calls_run_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            keys = [JudgeKey(kind="stt", row_id=str(i), fingerprint="fp") for i in range(3)]
            for i, key in enumerate(keys):
                await store.put(key, {"score": i})

            calls = []

            async def run_one(index):
                calls.append(index)
                return {"score": -1}

            results = await gather_with_store(keys, run_one, store, desc="test")

            self.assertEqual(calls, [])
            self.assertEqual([r["score"] for r in results], [0, 1, 2])

    async def test_partial_cache_runs_only_uncached_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            keys = [JudgeKey(kind="stt", row_id=str(i), fingerprint="fp") for i in range(3)]
            await store.put(keys[1], {"score": "cached"})

            calls = []

            async def run_one(index):
                calls.append(index)
                return {"score": "fresh"}

            results = await gather_with_store(keys, run_one, store, desc="test")

            self.assertEqual(sorted(calls), [0, 2])
            self.assertEqual(results[0]["score"], "fresh")
            self.assertEqual(results[1]["score"], "cached")
            self.assertEqual(results[2]["score"], "fresh")

    async def test_store_none_runs_every_row_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            keys = [JudgeKey(kind="stt", row_id=str(i), fingerprint="fp") for i in range(3)]
            calls = []

            async def run_one(index):
                calls.append(index)
                return {"score": index}

            results = await gather_with_store(keys, run_one, None, desc="test")

            self.assertEqual(sorted(calls), [0, 1, 2])
            self.assertEqual([r["score"] for r in results], [0, 1, 2])
            self.assertFalse(os.path.exists(os.path.join(tmp, JudgeStore.FILENAME)))

    async def test_none_key_at_index_always_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            cached_key = JudgeKey(kind="stt", row_id="0", fingerprint="fp")
            await store.put(cached_key, {"score": "cached"})
            keys = [cached_key, None]

            calls = []

            async def run_one(index):
                calls.append(index)
                return {"score": "fresh"}

            results = await gather_with_store(keys, run_one, store, desc="test")

            self.assertEqual(calls, [1])
            self.assertEqual(results[0]["score"], "cached")
            self.assertEqual(results[1]["score"], "fresh")

    async def test_exception_propagates_and_completed_rows_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            keys = [JudgeKey(kind="stt", row_id=str(i), fingerprint="fp") for i in range(3)]

            async def run_one(index):
                if index == 1:
                    # Delay the failure so rows 0 and 2 (no delay) finish and
                    # persist first. asyncio.gather(return_exceptions=False,
                    # the default) propagates the first exception without
                    # cancelling sibling tasks, so their work is not lost.
                    await asyncio.sleep(0.05)
                    raise ValueError("judge call failed")
                return {"score": index}

            with self.assertRaises(ValueError):
                await gather_with_store(keys, run_one, store, desc="test")

            reloaded = JudgeStore.load(tmp)
            self.assertEqual(reloaded.get(keys[0]), {"score": 0})
            self.assertEqual(reloaded.get(keys[2]), {"score": 2})
            self.assertIsNone(reloaded.get(keys[1]))


class TestGatherEvaluatorsWithStore(unittest.IsolatedAsyncioTestCase):
    async def test_returns_one_dict_per_row_in_row_and_key_order(self):
        row_keys = [
            {
                "b": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp", evaluator="b"),
                "a": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp", evaluator="a"),
            },
            {
                "a": JudgeKey(kind="evaluators", row_id="1", fingerprint="fp", evaluator="a"),
            },
        ]

        async def run_subset(index, names):
            return {name: {"score": f"{index}-{name}"} for name in names}

        results = await gather_evaluators_with_store(row_keys, run_subset, None, desc="test")

        self.assertEqual(len(results), 2)
        # Key order follows the input mapping's order ("b" before "a" for row 0).
        self.assertEqual(list(results[0].keys()), ["b", "a"])
        self.assertEqual(results[0]["a"], {"score": "0-a"})
        self.assertEqual(results[1], {"a": {"score": "1-a"}})

    async def test_fully_cached_row_never_calls_run_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            key_a = JudgeKey(kind="evaluators", row_id="0", fingerprint="fp", evaluator="a")
            key_b = JudgeKey(kind="evaluators", row_id="0", fingerprint="fp", evaluator="b")
            await store.put(key_a, {"score": "cached-a"})
            await store.put(key_b, {"score": "cached-b"})
            row_keys = [{"a": key_a, "b": key_b}]

            calls = []

            async def run_subset(index, names):
                calls.append(names)
                return {}

            results = await gather_evaluators_with_store(row_keys, run_subset, store, desc="test")

            self.assertEqual(calls, [])
            self.assertEqual(
                results[0], {"a": {"score": "cached-a"}, "b": {"score": "cached-b"}}
            )

    async def test_partial_hit_runs_only_missing_evaluators_and_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            key_a = JudgeKey(kind="evaluators", row_id="0", fingerprint="fp", evaluator="a")
            key_b = JudgeKey(kind="evaluators", row_id="0", fingerprint="fp", evaluator="b")
            await store.put(key_a, {"score": "cached-a"})
            row_keys = [{"a": key_a, "b": key_b}]

            calls = []

            async def run_subset(index, names):
                calls.append(list(names))
                return {name: {"score": f"fresh-{name}"} for name in names}

            results = await gather_evaluators_with_store(row_keys, run_subset, store, desc="test")

            # Only the missing evaluator is asked for, not the already-cached one.
            self.assertEqual(calls, [["b"]])
            self.assertEqual(results[0]["a"], {"score": "cached-a"})
            self.assertEqual(results[0]["b"], {"score": "fresh-b"})

    async def test_fresh_results_are_persisted_for_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JudgeStore.load(tmp)
            key_a = JudgeKey(kind="evaluators", row_id="0", fingerprint="fp", evaluator="a")
            row_keys = [{"a": key_a}]

            async def run_subset(index, names):
                return {name: {"score": "fresh"} for name in names}

            await gather_evaluators_with_store(row_keys, run_subset, store, desc="test")

            reloaded = JudgeStore.load(tmp)
            self.assertEqual(reloaded.get(key_a), {"score": "fresh"})

    async def test_run_subset_omitting_requested_evaluator_raises_key_error(self):
        row_keys = [{
            "a": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp", evaluator="a"),
            "b": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp", evaluator="b"),
        }]

        async def run_subset(index, names):
            return {"a": {"score": "only-a"}}  # "b" was requested but omitted

        with self.assertRaises(KeyError) as ctx:
            await gather_evaluators_with_store(row_keys, run_subset, None, desc="test")

        message = str(ctx.exception)
        self.assertIn("0", message)
        self.assertIn("b", message)

    async def test_store_none_runs_every_evaluator_for_every_row(self):
        row_keys = [
            {"a": JudgeKey(kind="evaluators", row_id="0", fingerprint="fp", evaluator="a")},
            {"a": JudgeKey(kind="evaluators", row_id="1", fingerprint="fp", evaluator="a")},
        ]
        calls = []

        async def run_subset(index, names):
            calls.append((index, tuple(names)))
            return {name: {"score": "x"} for name in names}

        results = await gather_evaluators_with_store(row_keys, run_subset, None, desc="test")

        self.assertEqual(sorted(calls), [(0, ("a",)), (1, ("a",))])
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
