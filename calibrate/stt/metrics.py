"""
STT evaluation metrics.
"""

import asyncio
import unicodedata
from functools import lru_cache
from typing import List, Optional

import numpy as np
import jiwer
from tqdm.asyncio import tqdm_asyncio
import backoff

from calibrate.judges import (
    text_judge,
    is_rating,
    evaluator_result_value,
    DEFAULT_TEXT_JUDGE_MODEL,
    DEFAULT_STT_EVALUATOR,
)
from calibrate.langfuse import observe, langfuse, langfuse_enabled

# NOTE: ``calibrate.stt.sarvam_intent_entity`` is imported lazily inside the
# intent/entity functions below — importing it eagerly pulls in transformers,
# indic-nlp, and joblib, which we want to avoid unless intent/entity scoring is
# actually requested (it's opt-in via ``--sarvam-judges``).

# Re-export for existing imports
DEFAULT_STT_JUDGE_MODEL = DEFAULT_TEXT_JUDGE_MODEL

# jiwer preprocessing pipeline, following AI4Bharat's Vistaar Indic ASR
# benchmark (https://github.com/AI4Bharat/vistaar/blob/master/evaluation.py):
# collapse whitespace, strip, case-fold, and drop punctuation before scoring.
# Text normalization (NFC + language-specific ``IndicNormalizer``) is applied
# separately, upstream of these transforms — see ``_normalize_text``.
_EDIT_CLEANUP = [
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
]
_WER_TRANSFORM = jiwer.Compose(_EDIT_CLEANUP + [jiwer.ReduceToListOfListOfWords()])
_CER_TRANSFORM = jiwer.Compose(_EDIT_CLEANUP + [jiwer.ReduceToListOfListOfChars()])

# calibrate language name / ISO code -> indic-nlp-library normalizer code.
# Languages absent here (english, urdu, …) get NFC-only normalization: Vistaar
# skips the IndicNormalizer for Urdu, and indic-nlp has no English normalizer.
_INDIC_NLP_LANG_CODES = {
    "hindi": "hi",
    "marathi": "mr",
    "sanskrit": "sa",
    "nepali": "ne",
    "konkani": "kK",
    "bengali": "bn",
    "assamese": "as",
    "punjabi": "pa",
    "gujarati": "gu",
    "odia": "or",
    "oriya": "or",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
}


def _resolve_indic_code(language: Optional[str]) -> Optional[str]:
    """Map a language name or ISO code to an indic-nlp normalizer code."""
    key = (language or "").strip().lower()
    if key in _INDIC_NLP_LANG_CODES:
        return _INDIC_NLP_LANG_CODES[key]
    if key in set(_INDIC_NLP_LANG_CODES.values()):
        return key
    return None


@lru_cache(maxsize=None)
def _indic_normalizer_for_lang_code(lang_code: str):
    """Build (and cache) the indic-nlp normalizer for ``lang_code``, or None.

    ``lang_code`` is an indic-nlp-library language code (e.g. ``"hi"``,
    ``"ta"``) as produced by ``_resolve_indic_code``. This is the lightweight
    ``indic-nlp-library`` normalizer Vistaar uses — distinct from the heavy
    vendored Sarvam ``IndicNormalizer`` in ``_get_indic_normalizer`` (which
    loads a Whisper processor). Any failure (unsupported language, missing
    optional dep like ``urduhack`` for Urdu) falls back to None so scoring
    proceeds with NFC-only normalization.
    """
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

        return IndicNormalizerFactory().get_normalizer(lang_code)
    except Exception:
        return None


def _indic_normalizer(language: Optional[str]):
    """Return the indic-nlp normalizer for ``language``, or None if unsupported."""
    lang_code = _resolve_indic_code(language)
    return _indic_normalizer_for_lang_code(lang_code) if lang_code else None


def _normalize_text(text: str, normalizer) -> str:
    """NFC-normalize, then apply ``normalizer`` (indic-nlp) if provided.

    NFC folds composed vs. decomposed diacritics; the ``IndicNormalizer``
    additionally canonicalizes script variants (nukta forms, ZWJ/ZWNJ,
    alternate spellings) that NFC alone leaves distinct.

    Mirrors the per-utterance normalization in AI4Bharat's Vistaar:
    https://github.com/AI4Bharat/vistaar/blob/master/evaluation.py
    """
    text = unicodedata.normalize("NFC", str(text))
    if normalizer is not None:
        text = normalizer.normalize(text)
    return text


@lru_cache(maxsize=1)
def _get_indic_normalizer():
    """Build the vendored ``IndicNormalizer`` once and reuse it.

    ``IndicNormalizer.__init__`` loads the ``openai/whisper-small`` processor
    from disk, so constructing one per scoring call (and per provider in a
    multi-provider benchmark) reloads the model repeatedly. Caching keeps a
    single instance for the process lifetime. The import is deferred so the
    heavy transformers/indic-nlp stack only loads when scoring is requested.
    """
    from calibrate.stt.sarvam_intent_entity import IndicNormalizer

    return IndicNormalizer()


def _normalize_pairs(
    references: List[str], predictions: List[str], language: str
) -> tuple[List[str], List[str]]:
    """Normalize references/predictions with the cached normalizer.

    Runs in-process (``n_jobs=1`` — no joblib subprocess fork) so it can be
    offloaded to a worker thread without freezing the event loop.
    """
    normalizer = _get_indic_normalizer()
    ref_langs = [language] * len(references)
    pred_langs = [language] * len(predictions)
    norm_references = normalizer.normalize_texts(
        [str(r) for r in references], ref_langs, n_jobs=1
    )
    norm_predictions = normalizer.normalize_texts(
        [str(p) for p in predictions], pred_langs, n_jobs=1
    )
    return norm_references, norm_predictions


def _resolve_evaluators(evaluators: Optional[List[dict]]) -> List[dict]:
    """Return ``evaluators`` if non-empty, else the implicit default."""
    return list(evaluators) if evaluators else [DEFAULT_STT_EVALUATOR]


def _edit_metric(
    metric_fn,
    transform,
    references: List[str],
    predictions: List[str],
    language: str,
) -> dict:
    """Compute a jiwer edit-distance metric, dataset-level plus per-row.

    Shared by WER and CER, mirroring AI4Bharat's Vistaar benchmark.
    References/predictions are normalized (NFC + language-specific
    ``IndicNormalizer``), then ``jiwer`` applies ``transform`` (whitespace
    collapse, strip, case-fold, punctuation removal, tokenization) before
    scoring.

    ``score`` is the **dataset-level** rate — total substitutions, deletions,
    and insertions across all utterances divided by total reference length —
    matching the NIST definition Vistaar uses. This differs from a macro-mean
    of per-utterance rates, which over-weights short utterances. ``per_row``
    holds the per-utterance rates, kept for row-level reporting.

    Empty references need no placeholder: jiwer pools them correctly at the
    dataset level (an empty ref with an empty hypothesis contributes nothing;
    a hallucinated hypothesis contributes insertions). The only degenerate
    case is a dataset with *no* reference words at all — guarded below to
    avoid jiwer returning an unbounded count instead of a rate.
    """
    normalizer = _indic_normalizer(language)

    references = [_normalize_text(ref, normalizer) for ref in references]
    predictions = [
        _normalize_text(pred, normalizer) if isinstance(pred, str) else ""
        for pred in predictions
    ]

    # Per-clip rates — for results.csv only, never averaged into `score`.
    per_row = [
        metric_fn(
            reference=[r],
            hypothesis=[p],
            reference_transform=transform,
            hypothesis_transform=transform,
        )
        for r, p in zip(references, predictions)
    ]

    # Headline score — pooled over the whole dataset, not a per-row average.
    score = (
        metric_fn(
            reference=references,
            hypothesis=predictions,
            reference_transform=transform,
            hypothesis_transform=transform,
        )
        if any(transform([ref])[0] for ref in references)
        else 0.0
    )

    return {"score": float(score), "per_row": per_row}


def get_wer_score(
    references: List[str], predictions: List[str], language: str = "english"
) -> dict:
    return _edit_metric(jiwer.wer, _WER_TRANSFORM, references, predictions, language)


def get_cer_score(
    references: List[str], predictions: List[str], language: str = "english"
) -> dict:
    return _edit_metric(jiwer.cer, _CER_TRANSFORM, references, predictions, language)


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


async def get_intent_entity_score(
    references: List[str],
    predictions: List[str],
    language: str = "english",
    model: Optional[str] = None,
) -> dict:
    """Normalize, judge, and aggregate intent/entity preservation.

    Mirrors Sarvam's flow: reference and prediction are first run through the
    vendored ``IndicNormalizer``, then each normalized pair is scored by the
    judge in ``stt/sarvam_intent_entity/judge.py``. Aggregation uses Sarvam's
    ``calculate_intent_accuracy`` / ``calculate_entity_metrics``. This is the
    metric root invoked by the eval pipeline, mirroring ``get_wer_score`` /
    ``get_llm_judge_score``.

    Returns:
        {
            "intent": float,          # intent accuracy (mean of 0/1)
            "entity": float,          # mean entity-preservation fraction
            "per_row": [ {<IntentEntityResponse fields>}, ... ],
        }

    ``model`` defaults to ``DEFAULT_INTENT_ENTITY_MODEL`` when omitted.
    ``per_row`` order matches the input order (``asyncio.gather`` preserves
    coroutine order).
    """
    if not references:
        return {"intent": 0.0, "entity": 0.0, "per_row": []}

    # Deferred so the transformers/indic-nlp stack only loads when scoring runs.
    from calibrate.stt import sarvam_intent_entity
    from calibrate.stt.sarvam_intent_entity import (
        DEFAULT_INTENT_ENTITY_MODEL,
        calculate_intent_accuracy,
        calculate_entity_metrics,
    )

    if model is None:
        model = DEFAULT_INTENT_ENTITY_MODEL

    norm_references, norm_predictions = await asyncio.to_thread(
        _normalize_pairs, references, predictions, language
    )

    coroutines = [
        sarvam_intent_entity.intent_entity_judge(
            str(reference), str(prediction), model=model, index=i
        )
        for i, (reference, prediction) in enumerate(
            zip(norm_references, norm_predictions)
        )
    ]

    results = await tqdm_asyncio.gather(
        *coroutines,
        desc="Running intent/entity judge",
    )

    intent_scores = [int(row["intent_score"]) for row in results]
    entity_scores = [float(row["entity_score"]) for row in results]

    return {
        "intent": float(calculate_intent_accuracy(intent_scores)),
        "entity": float(calculate_entity_metrics(entity_scores)["mean"]),
        "per_row": results,
    }
