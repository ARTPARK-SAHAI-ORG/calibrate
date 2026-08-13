"""
TTS evaluation metrics.
"""

import hashlib
from typing import List, Optional, Sequence

import numpy as np
import backoff

from calibrate_agent.judges import (
    audio_judge,
    is_rating,
    evaluator_result_value,
    DEFAULT_AUDIO_JUDGE_MODEL,
    DEFAULT_TTS_EVALUATOR,
)
from calibrate_agent.judge_store import (
    JudgeKey,
    JudgeStore,
    gather_evaluators_with_store,
    make_fingerprint,
)
from calibrate_agent.langfuse import observe

# Re-export for existing imports
DEFAULT_TTS_JUDGE_MODEL = DEFAULT_AUDIO_JUDGE_MODEL


def _resolve_evaluators(evaluators: Optional[List[dict]]) -> List[dict]:
    """Return ``evaluators`` if non-empty, else the implicit default."""
    return list(evaluators) if evaluators else [DEFAULT_TTS_EVALUATOR]


def _audio_content_hash(audio_path: str) -> str:
    """Hash the bytes of ``audio_path`` for the judge cache fingerprint.

    A synthesized clip's audio file is overwritten in place when the row is
    re-synthesized, so a fingerprint built from the path alone would return a
    stale cached grade for audio that no longer exists at that path. Hashing
    the file's content ties the cache entry to the actual audio.

    A missing or unreadable file returns a sentinel string instead of raising:
    it never matches a real sha256 digest, so the fingerprint always misses
    and the read failure surfaces naturally from ``audio_judge`` opening the
    same file, rather than aborting the whole batch here.
    """
    try:
        with open(audio_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError as e:
        return f"unreadable:{audio_path}:{e}"


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(
    name="tts_llm_judge",
    capture_input=False,
    capture_output=False,
)
async def tts_llm_judge(
    audio_path: str,
    reference_text: str,
    evaluators: Optional[List[dict]] = None,
    fallback_model: str = DEFAULT_TTS_JUDGE_MODEL,
) -> dict:
    """Evaluate a TTS audio output against one or more evaluators.

    Args:
        audio_path: Path to the synthesized WAV audio file.
        reference_text: The text that should have been spoken.
        evaluators: List of evaluator dicts. If omitted, the implicit
            ``DEFAULT_TTS_EVALUATOR`` is used.
        fallback_model: Audio-capable model id used when an evaluator
            lacks ``judge_model``.

    Returns:
        Dict keyed by evaluator name. Binary entries are
        ``{"reasoning": str, "match": bool}``; rating entries are
        ``{"reasoning": str, "score": int}``.
    """
    evaluators = _resolve_evaluators(evaluators)

    return await audio_judge(
        evaluators=evaluators,
        audio_path=audio_path,
        reference_text=reference_text,
        fallback_model=fallback_model,
    )


async def get_tts_llm_judge_score(
    audio_paths: List[str],
    reference_texts: List[str],
    evaluators: Optional[List[dict]] = None,
    fallback_model: str = DEFAULT_TTS_JUDGE_MODEL,
    store: Optional[JudgeStore] = None,
    row_ids: Optional[Sequence] = None,
) -> dict:
    """Run TTS judge across all rows and aggregate per-evaluator scores.

    Args:
        audio_paths: Path to each row's synthesized WAV audio file.
        reference_texts: The text that should have been spoken for each row.
        evaluators: List of evaluator dicts. If omitted, the implicit
            ``DEFAULT_TTS_EVALUATOR`` is used.
        fallback_model: Audio-capable model id used when an evaluator lacks
            ``judge_model``.
        store: Optional ``JudgeStore`` checkpoint. When given, a
            (row, evaluator) pair whose fingerprint is already cached skips
            the judge call entirely; fresh results are persisted as they
            complete. The fingerprint hashes the audio file's bytes (not just
            its path), the reference text, the evaluator's ``system_prompt``,
            its resolved model id, its type, and its rating scale — so
            re-synthesizing a clip, editing a prompt, or switching models
            invalidates the cached grade. When ``None``, every row/evaluator
            runs unconditionally, matching the behavior with no checkpoint.
        row_ids: Optional per-row id used to key the checkpoint, aligned with
            ``audio_paths``/``reference_texts``. Defaults to the row index
            when omitted.

    Returns:
        {
            "scores": {"pronunciation": {"type": "binary", "mean": 0.83}, ...},
            "score": float,
            "per_row": [
                {"pronunciation": {"reasoning": ..., "match": ...}, ...},
                ...
            ]
        }

    Iteration order of ``scores`` and each ``per_row`` entry matches the
    order of the ``evaluators`` argument.
    """
    evaluators = _resolve_evaluators(evaluators)

    if row_ids is None:
        row_ids = range(len(audio_paths))

    audio_hashes = [_audio_content_hash(path) for path in audio_paths]

    row_keys = [
        {
            ev["name"]: JudgeKey(
                kind="tts_evaluators",
                row_id=row_id,
                fingerprint=make_fingerprint(
                    audio_hash,
                    reference_text,
                    ev.get("system_prompt", ""),
                    ev.get("judge_model") or fallback_model,
                    ev.get("type", "binary"),
                    *(
                        (ev.get("scale_min"), ev.get("scale_max"))
                        if is_rating(ev)
                        else ()
                    ),
                ),
                evaluator=ev["name"],
            )
            for ev in evaluators
        }
        for row_id, audio_hash, reference_text in zip(
            row_ids, audio_hashes, reference_texts
        )
    ]

    async def run_subset(index: int, names: list[str]) -> dict:
        subset = [ev for ev in evaluators if ev["name"] in names]
        return await tts_llm_judge(
            audio_paths[index],
            reference_texts[index],
            evaluators=subset,
            fallback_model=fallback_model,
        )

    results = await gather_evaluators_with_store(
        row_keys,
        run_subset,
        store,
        desc="Running TTS evaluators",
    )

    # Aggregate per-evaluator scores — binary: mean 0/1, rating: mean score
    scores: dict = {}
    for ev in evaluators:
        name = ev["name"]
        per_row_values = [evaluator_result_value(ev, row[name]) for row in results]
        if is_rating(ev):
            scores[name] = {
                "type": "rating",
                "mean": float(np.mean(per_row_values)),
                "scale_min": int(ev["scale_min"]),
                "scale_max": int(ev["scale_max"]),
            }
        else:
            scores[name] = {
                "type": "binary",
                "mean": float(np.mean(per_row_values)),
            }

    overall_score = float(np.mean([s["mean"] for s in scores.values()]))

    return {
        "scores": scores,
        "score": overall_score,
        "per_row": results,
    }
