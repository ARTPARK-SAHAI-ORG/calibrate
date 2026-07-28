"""
Resumable checkpoint for LLM-as-judge results.

STT and TTS evaluation runs grade every dataset row with an LLM judge. A
single run can issue hundreds of judge calls, each one a paid API call; if
the process is interrupted or one row's judge call fails, the rows already
graded should not have to be re-graded (and re-paid for) on the next run.

``JudgeStore`` is an append-only JSONL checkpoint: every judge result is
written to disk the moment it is computed, keyed by a :class:`JudgeKey` that
fingerprints the exact input that produced it. Re-running with the same
input, evaluator prompt, and judge model reuses the cached result; changing
any of those changes the fingerprint, so the row is graded again instead of
silently returning a stale grade.

Two gather helpers, :func:`gather_with_store` and
:func:`gather_evaluators_with_store`, are drop-in replacements for the
``tqdm_asyncio.gather`` calls in the STT/TTS metrics modules: they skip
already-cached rows (or, per evaluator, already-cached evaluator results)
and persist each fresh result as it lands rather than waiting for the whole
batch to finish.
"""

import asyncio
import hashlib
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from tqdm.asyncio import tqdm_asyncio


@dataclass(frozen=True)
class JudgeKey:
    """Identity of one cached judge result.

    ``row_id`` is coerced to ``str`` in ``__post_init__``: pandas reads a
    numeric-looking ``id`` column back as an ``int`` on reload, so a key
    built from the string ``"1"`` would otherwise fail to match one built
    from the int ``1`` for the same row.

    ``evaluator`` is set only for the per-evaluator kinds (``"evaluators"``,
    ``"tts_evaluators"``) where one row produces one result per evaluator
    name; the single-result-per-row kinds leave it ``None``.
    """

    kind: str
    row_id: str
    fingerprint: str
    evaluator: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_id", str(self.row_id))


def make_fingerprint(*parts: object) -> str:
    """Return a stable hex sha256 digest over ``parts``.

    Each part is encoded independently with
    ``json.dumps(part, sort_keys=True, ensure_ascii=False, default=str)`` so
    dict-valued parts fingerprint the same regardless of key order, then the
    encoded parts are joined with ``"\\x00"`` — a separator that cannot
    appear inside any of the JSON-encoded parts — before hashing.

    Callers pass the judge input (reference/prediction text, audio path,
    conversation, ...) together with the evaluator's system prompt and the
    judge model id, so that editing the prompt, switching models, or
    re-transcribing a row invalidates the cached result for that row.
    """
    encoded = [
        json.dumps(part, sort_keys=True, ensure_ascii=False, default=str)
        for part in parts
    ]
    joined = "\x00".join(encoded)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class JudgeStore:
    """Append-only JSONL checkpoint of per-row judge results.

    Use :meth:`load` to construct an instance — it creates ``output_dir`` if
    needed and reads any results already on disk from a prior run.
    """

    FILENAME = "judge_cache.jsonl"

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir
        self._path = os.path.join(output_dir, self.FILENAME)
        self._records: dict[JudgeKey, dict] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def load(cls, output_dir: str) -> "JudgeStore":
        """Create ``output_dir`` if needed and load any existing checkpoint file."""
        os.makedirs(output_dir, exist_ok=True)
        store = cls(output_dir)
        store._load_existing()
        return store

    def _load_existing(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # A truncated final line is the expected shape of a killed
                # run, not corruption — skip it rather than fail the load.
                try:
                    record = json.loads(line)
                    key = JudgeKey(
                        kind=record["kind"],
                        row_id=record["row_id"],
                        fingerprint=record["fingerprint"],
                        evaluator=record.get("evaluator"),
                    )
                    result = record["result"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                self._records[key] = result

    @property
    def path(self) -> str:
        """Absolute path to the JSONL checkpoint file."""
        return self._path

    def get(self, key: JudgeKey) -> dict | None:
        """Return the cached result for ``key``, or ``None`` if not cached."""
        return self._records.get(key)

    async def put(self, key: JudgeKey, result: dict) -> None:
        """Append one record for ``key``/``result`` and update the in-memory map.

        The write is flushed and fsynced before the lock is released, so a
        result is durable on disk before ``put`` returns — the property that
        makes an interrupted run lose at most the row in flight.
        """
        record = {
            "kind": key.kind,
            "row_id": key.row_id,
            "evaluator": key.evaluator,
            "fingerprint": key.fingerprint,
            "result": result,
        }
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._records[key] = result

    def pending(self, keys: Sequence[JudgeKey]) -> list[JudgeKey]:
        """Return the subset of ``keys`` with no cached result, in input order."""
        return [key for key in keys if key not in self._records]

    def clear(self) -> None:
        """Delete the checkpoint file and empty the in-memory map."""
        if os.path.exists(self._path):
            os.remove(self._path)
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)


async def gather_with_store(
    keys: Sequence[JudgeKey | None],
    run_one: Callable[[int], Awaitable[dict]],
    store: JudgeStore | None,
    desc: str,
) -> list[dict]:
    """Run one judge call per row, reusing cached results where possible.

    For judges with exactly one result per row (intent/entity, LLM WER,
    semantic WER). ``keys[i]`` identifies row ``i``'s judge input;
    ``run_one(i)`` computes and returns its result. A row whose key is
    cached in ``store`` skips ``run_one`` entirely; a row with no cache hit
    runs immediately and its result is persisted to ``store`` before moving
    on. When ``store`` is ``None``, or ``keys[i]`` is ``None``, row ``i``
    always runs.

    An exception raised by ``run_one`` propagates to the caller; any result
    already computed and stored for other rows before the failure remains
    on disk.

    Returns:
        Results in row order (index ``i`` is row ``i``'s result),
        regardless of completion order.
    """
    results: list[dict | None] = [None] * len(keys)

    async def resolve(index: int) -> None:
        key = keys[index]
        if store is not None and key is not None:
            cached = store.get(key)
            if cached is not None:
                results[index] = cached
                return
        result = await run_one(index)
        if store is not None and key is not None:
            await store.put(key, result)
        results[index] = result

    await tqdm_asyncio.gather(
        *[resolve(index) for index in range(len(keys))],
        desc=desc,
    )
    return results


async def gather_evaluators_with_store(
    row_keys: Sequence[Mapping[str, JudgeKey]],
    run_subset: Callable[[int, list[str]], Awaitable[Mapping[str, dict]]],
    store: JudgeStore | None,
    desc: str,
) -> list[dict]:
    """Run per-evaluator judge calls, reusing cached results per evaluator.

    For evaluator-based judges where one row produces one result per
    evaluator. ``row_keys[i]`` maps each evaluator name to its key for row
    ``i``. ``run_subset(i, names)`` runs only the named evaluators for row
    ``i`` and returns ``{evaluator_name: result}``.

    A row whose evaluators are all cached in ``store`` skips ``run_subset``
    entirely. A row with a partial cache hit calls ``run_subset`` with only
    the missing evaluator names and merges the cached results back in, so
    adding a new evaluator to a config only pays for that evaluator across
    rows, not the ones already graded. Each freshly computed result is
    persisted to ``store`` as it arrives. When ``store`` is ``None``, every
    evaluator for every row is run.

    Raises:
        KeyError: if ``run_subset`` returns a mapping missing one of the
            evaluator names it was asked to compute, naming the row and
            evaluator.

    Returns:
        A list of ``{evaluator_name: result}`` dicts, one per row, in row
        order — the same ``per_row`` shape the existing aggregation
        functions produce.
    """
    results: list[dict | None] = [None] * len(row_keys)

    async def resolve(index: int) -> None:
        keys_for_row = row_keys[index]
        cached: dict[str, dict] = {}
        missing: list[str] = []
        for name, key in keys_for_row.items():
            hit = store.get(key) if store is not None else None
            if hit is not None:
                cached[name] = hit
            else:
                missing.append(name)

        if missing:
            fresh = await run_subset(index, missing)
            for name in missing:
                if name not in fresh:
                    raise KeyError(
                        f"run_subset for row {index!r} did not return a "
                        f"result for evaluator {name!r}"
                    )
                result = fresh[name]
                if store is not None:
                    await store.put(keys_for_row[name], result)
                cached[name] = result

        results[index] = {name: cached[name] for name in keys_for_row}

    await tqdm_asyncio.gather(
        *[resolve(index) for index in range(len(row_keys))],
        desc=desc,
    )
    return results
