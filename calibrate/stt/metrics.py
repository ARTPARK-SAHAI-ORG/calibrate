"""
STT evaluation metrics.
"""

from typing import List, Optional

import numpy as np
from evaluate import load
from tqdm.asyncio import tqdm_asyncio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import backoff

from calibrate.judges import (
    text_judge,
    is_rating,
    evaluator_result_value,
    DEFAULT_TEXT_JUDGE_MODEL,
    DEFAULT_STT_EVALUATOR,
)
from calibrate.langfuse import observe, langfuse, langfuse_enabled

normalizer = BasicTextNormalizer()

# Re-export for existing imports
DEFAULT_STT_JUDGE_MODEL = DEFAULT_TEXT_JUDGE_MODEL


def _resolve_evaluators(evaluators: Optional[List[dict]]) -> List[dict]:
    """Return ``evaluators`` if non-empty, else the implicit default."""
    return list(evaluators) if evaluators else [DEFAULT_STT_EVALUATOR]


def _edit_metric(name: str, references: List[str], predictions: List[str]) -> dict:
    """Compute a normalized per-row edit-distance metric from ``evaluate``.

    Shared by WER and CER: both normalize ref/pred with the Whisper
    ``BasicTextNormalizer``, score each row independently via the
    HuggingFace ``evaluate`` metric ``name`` (``"wer"`` / ``"cer"``), and
    return the macro-mean plus the per-row list.
    """
    metric = load(name)

    references = [normalizer(str(ref)) for ref in references]
    predictions = [
        normalizer(str(pred)) if isinstance(pred, str) else "" for pred in predictions
    ]

    per_row = [
        metric.compute(predictions=[p], references=[r])
        for p, r in zip(predictions, references)
    ]

    return {"score": np.mean(per_row), "per_row": per_row}


def get_wer_score(references: List[str], predictions: List[str]) -> dict:
    return _edit_metric("wer", references, predictions)


def get_cer_score(references: List[str], predictions: List[str]) -> dict:
    return _edit_metric("cer", references, predictions)


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(
    name="stt_llm_judge",
    capture_input=False,
)
async def stt_llm_judge(
    reference: str,
    prediction: str,
    evaluators: Optional[List[dict]] = None,
    fallback_model: str = DEFAULT_STT_JUDGE_MODEL,
) -> dict:
    """Evaluate an STT transcription against one or more evaluators.

    Args:
        reference: The source/ground-truth text.
        prediction: The STT transcription output.
        evaluators: List of evaluator dicts. If omitted, the implicit
            ``DEFAULT_STT_EVALUATOR`` is used.
        fallback_model: Model id used when an evaluator lacks ``judge_model``.

    Returns:
        Dict keyed by evaluator name. Binary entries are
        ``{"reasoning": str, "match": bool}``; rating entries are
        ``{"reasoning": str, "score": int}``.
    """
    evaluators = _resolve_evaluators(evaluators)

    user_prompt = f"Source: {reference}\nTranscription: {prediction}"

    result = await text_judge(
        evaluators=evaluators,
        user_prompt=user_prompt,
        fallback_model=fallback_model,
    )

    if langfuse_enabled and langfuse:
        langfuse.update_current_trace(
            input={"reference": reference, "prediction": prediction},
            metadata={
                "reference": reference,
                "prediction": prediction,
                "output": result,
            },
        )

    return result


async def get_llm_judge_score(
    references: List[str],
    predictions: List[str],
    evaluators: Optional[List[dict]] = None,
    fallback_model: str = DEFAULT_STT_JUDGE_MODEL,
) -> dict:
    """Run STT judge across all rows and aggregate per-evaluator scores.

    Returns:
        {
            "scores": {
                "semantic_match": {"type": "binary", "mean": 0.83, ...},
                ...
            },
            "score": float,                        # mean across evaluators
            "per_row": [
                {"semantic_match": {"reasoning": ..., "match": ...}, ...},
                ...
            ]
        }

    Iteration order of ``scores`` and each ``per_row`` entry matches the
    order of the ``evaluators`` argument (Python dicts preserve insertion
    order; ``asyncio.gather`` preserves coroutine order).
    """
    evaluators = _resolve_evaluators(evaluators)

    coroutines = [
        stt_llm_judge(
            str(reference),
            str(prediction),
            evaluators=evaluators,
            fallback_model=fallback_model,
        )
        for reference, prediction in zip(references, predictions)
    ]

    results = await tqdm_asyncio.gather(
        *coroutines,
        desc="Running STT evaluators",
    )

    # Aggregate per-evaluator scores — mean of 0/1 for binary, mean of scores for rating.
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
                "mean": float(np.mean(per_row_values)),  # pass-rate fraction 0.0–1.0
            }

    # Backward compat: top-level "score" = mean across evaluator means.
    overall_score = float(np.mean([s["mean"] for s in scores.values()]))

    return {
        "scores": scores,
        "score": overall_score,
        "per_row": results,
    }
