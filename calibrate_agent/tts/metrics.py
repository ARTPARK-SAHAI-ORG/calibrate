"""
TTS evaluation metrics.
"""

from typing import Callable, List, Optional

from tqdm.asyncio import tqdm_asyncio
import backoff

from calibrate_agent.judges import (
    aggregate_judge_scores,
    audio_judge,
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
) -> dict:
    """Run TTS judge across all rows and aggregate per-evaluator scores.

    ``on_row`` is called with ``(row_index, row_result)`` as each row's judge
    finishes, so a caller can persist results as they arrive.

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

    coroutines = [
        tts_llm_judge(
            audio_path,
            reference_text,
            evaluators=evaluators,
            fallback_model=fallback_model,
        )
        for audio_path, reference_text in zip(audio_paths, reference_texts)
    ]

    results = await tqdm_asyncio.gather(
        *with_row_callback(coroutines, on_row),
        desc="Running TTS evaluators",
    )

    return {**aggregate_judge_scores(evaluators, results), "per_row": results}
