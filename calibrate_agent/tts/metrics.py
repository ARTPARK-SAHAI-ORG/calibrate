"""
TTS evaluation metrics.
"""

from typing import Callable, List, Optional

import numpy as np
from tqdm.asyncio import tqdm_asyncio
import backoff

from calibrate_agent.judges import (
    audio_judge,
    is_rating,
    evaluator_result_value,
    with_row_callback,
    DEFAULT_AUDIO_JUDGE_MODEL,
    DEFAULT_TTS_EVALUATOR,
)
from calibrate_agent.langfuse import observe

# Re-export for existing imports
DEFAULT_TTS_JUDGE_MODEL = DEFAULT_AUDIO_JUDGE_MODEL


def _resolve_evaluators(evaluators: Optional[List[dict]]) -> List[dict]:
    """Return ``evaluators`` if non-empty, else the implicit default."""
    return list(evaluators) if evaluators else [DEFAULT_TTS_EVALUATOR]


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
    on_row: Optional[Callable] = None,
    known: Optional[List] = None,
) -> dict:
    """Run TTS judge across all rows and aggregate per-evaluator scores.

    ``on_row`` is called with ``(row_index, row_result)`` as each row's judge
    finishes, so a caller can persist results as they arrive.

    ``known`` holds one entry per row: the judge result a previous run already
    produced for it, or None when the row still has to be judged. Only the None
    rows cost a judge call; the rest are slotted back into dataset order so the
    aggregated means match a single run over every row.

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

    known = list(known) if known else [None] * len(audio_paths)
    pending = [index for index, row in enumerate(known) if row is None]

    coroutines = [
        tts_llm_judge(
            audio_paths[index],
            reference_texts[index],
            evaluators=evaluators,
            fallback_model=fallback_model,
        )
        for index in pending
    ]

    # ``with_row_callback`` counts the coroutines it is given; map that count
    # back to the row it came from so callers always see dataset positions.
    row_callback = (
        (lambda position, row: on_row(pending[position], row)) if on_row else None
    )

    judged = await tqdm_asyncio.gather(
        *with_row_callback(coroutines, row_callback),
        desc="Running TTS evaluators",
    )

    results = list(known)
    for index, row in zip(pending, judged):
        results[index] = row

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
