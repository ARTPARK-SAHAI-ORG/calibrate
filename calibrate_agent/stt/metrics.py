"""
STT evaluation metrics.
"""

import asyncio
import json
import os
import unicodedata
from functools import lru_cache
from typing import Callable, List, Optional

import numpy as np
import jiwer
from tqdm.asyncio import tqdm_asyncio
import backoff

from calibrate_agent.judges import (
    text_judge,
    is_rating,
    evaluator_result_value,
    DEFAULT_TEXT_JUDGE_MODEL,
    DEFAULT_STT_EVALUATOR,
)
from calibrate_agent.langfuse import observe, langfuse, langfuse_enabled

# NOTE: ``calibrate_agent.stt.sarvam_intent_entity`` is imported lazily inside the
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

# calibrate_agent language name / ISO code -> indic-nlp-library normalizer code.
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
    from calibrate_agent.stt.sarvam_intent_entity import IndicNormalizer

    return IndicNormalizer()


def _normalize_refs_preds(
    references: List[str], predictions: List[str], language: str
) -> tuple[List[str], List[str]]:
    """NFC + indic-nlp normalize references/predictions (Vistaar path #1).

    Shared by WER/CER and LLM-WER/CER so edit metrics compare the same
    normalized strings; jiwer cleanup (case-fold, punctuation, whitespace)
    is applied later at scoring time.
    """
    normalizer = _indic_normalizer(language)
    norm_references = [_normalize_text(ref, normalizer) for ref in references]
    norm_predictions = [
        _normalize_text(pred, normalizer) if isinstance(pred, str) else ""
        for pred in predictions
    ]
    return norm_references, norm_predictions


def _normalize_pairs(
    references: List[str], predictions: List[str], language: str
) -> tuple[List[str], List[str]]:
    """Normalize references/predictions with the vendored Sarvam normalizer.

    Used only by intent/entity scoring (path #2 — Whisper + up-front
    punctuation/case stripping). Runs in-process (``n_jobs=1``) so it can be
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


# --- Per-row judge checkpointing ------------------------------------------
#
# LLM judge calls are the expensive, re-billable part of scoring. Each root
# persists its per-row results to a JSONL cache so an interrupted run resumes
# without re-billing completed calls. Keys are plain row indices ("0", "1", …)
# in dataset order — same assumption as transcription resume on results.csv.


def _row_cache_key(index: int) -> str:
    """JSONL cache key for the ``index``-th row in the current dataset."""
    return str(index)


def _load_judge_cache(cache_path: Optional[str]) -> dict:
    """Load a JSONL judge cache into ``{key: value}``.

    Missing file → empty cache. A partially written trailing line (from a crash
    mid-append) is skipped rather than raising, so a resumed run still recovers
    every fully written entry.
    """
    cache: dict = {}
    if not cache_path or not os.path.exists(cache_path):
        return cache
    with open(cache_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "key" in entry:
                cache[entry["key"]] = entry["value"]
    return cache


async def _gather_cached(
    cache_path: Optional[str],
    keys: List[str],
    make_coro: Callable[[int], "asyncio.Future"],
    desc: str,
) -> list:
    """Run ``make_coro(i)`` for each item whose ``keys[i]`` isn't already cached.

    Results are returned aligned to ``keys`` order (cached + newly computed).
    Each new result is appended to ``cache_path`` (JSONL) as soon as it completes
    so an interrupted run preserves finished work; ``cache_path=None`` disables
    persistence (every item is computed, nothing is written). Identical keys are
    computed once and shared. If a coroutine raises (after its own retries), the
    successes that completed are still persisted before the first exception is
    re-raised.
    """
    cache = _load_judge_cache(cache_path)

    # One representative index per not-yet-cached unique key.
    todo: List[int] = []
    scheduled: set = set(cache.keys())
    for i, key in enumerate(keys):
        if key not in scheduled:
            scheduled.add(key)
            todo.append(i)

    if todo:

        async def _wrapped(idx: int):
            return idx, await make_coro(idx)

        writer = None
        first_exc: Optional[BaseException] = None
        try:
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                writer = open(cache_path, "a", encoding="utf-8")
            for coro in tqdm_asyncio.as_completed(
                [_wrapped(i) for i in todo], desc=desc
            ):
                try:
                    idx, result = await coro
                except Exception as exc:  # noqa: BLE001 — preserve partial work
                    if first_exc is None:
                        first_exc = exc
                    continue
                cache[keys[idx]] = result
                if writer:
                    writer.write(
                        json.dumps(
                            {"key": keys[idx], "value": result}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    writer.flush()
        finally:
            if writer:
                writer.close()
        if first_exc is not None:
            raise first_exc

    return [cache[key] for key in keys]


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
    references, predictions = _normalize_refs_preds(references, predictions, language)
    return _score_edit_metric(metric_fn, transform, references, predictions)


def _score_edit_metric(
    metric_fn,
    transform,
    references: List[str],
    predictions: List[str],
) -> dict:
    """Score already-normalized (reference, prediction) pairs with jiwer.

    The scoring half of :func:`_edit_metric` — no text normalization is applied,
    so callers must normalize (and coerce to ``str``) upstream. Shared by the
    WER/CER metrics (which normalize first) and the LLM-WER/CER root (whose
    corrected pairs are already Vistaar-normalized strings).
    ``score`` is the dataset-pooled rate; ``per_row`` holds the per-utterance
    rates for row-level reporting.
    """
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
    cache_path: Optional[str] = None,
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
    order; results are returned in input order).

    ``cache_path`` (optional) enables per-row checkpointing: completed rows are
    persisted to a JSONL file and skipped on a resumed run.
    """
    evaluators = _resolve_evaluators(evaluators)

    refs = [str(r) for r in references]
    preds = [str(p) for p in predictions]
    keys = [_row_cache_key(i) for i in range(len(refs))]

    def _make(i: int):
        return stt_llm_judge(
            refs[i],
            preds[i],
            evaluators=evaluators,
            fallback_model=fallback_model,
        )

    results = await _gather_cached(
        cache_path, keys, _make, desc="Running STT evaluators"
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
    cache_path: Optional[str] = None,
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
    ``per_row`` order matches the input order. ``cache_path`` (optional) enables
    per-row checkpointing: judged rows are persisted to a JSONL file and skipped
    on a resumed run.
    """
    if not references:
        return {"intent": 0.0, "entity": 0.0, "per_row": []}

    # Deferred so the transformers/indic-nlp stack only loads when scoring runs.
    from calibrate_agent.stt import sarvam_intent_entity
    from calibrate_agent.stt.sarvam_intent_entity import (
        DEFAULT_INTENT_ENTITY_MODEL,
        calculate_intent_accuracy,
        calculate_entity_metrics,
    )

    if model is None:
        model = DEFAULT_INTENT_ENTITY_MODEL

    norm_references, norm_predictions = await asyncio.to_thread(
        _normalize_pairs, references, predictions, language
    )

    norm_references = [str(r) for r in norm_references]
    norm_predictions = [str(p) for p in norm_predictions]
    keys = [_row_cache_key(i) for i in range(len(norm_references))]

    def _make(i: int):
        return sarvam_intent_entity.intent_entity_judge(
            norm_references[i], norm_predictions[i], model=model, index=i
        )

    results = await _gather_cached(
        cache_path, keys, _make, desc="Running intent/entity judge"
    )

    intent_scores = [int(row["intent_score"]) for row in results]
    entity_scores = [float(row["entity_score"]) for row in results]

    return {
        "intent": float(calculate_intent_accuracy(intent_scores)),
        "entity": float(calculate_entity_metrics(entity_scores)["mean"]),
        "per_row": results,
    }


async def get_llm_wer_cer_score(
    references: List[str],
    predictions: List[str],
    language: str = "english",
    model: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> dict:
    """Normalize, judge segment equivalence, forgive, and re-score WER/CER.

    Reference and prediction are first normalized with the same Vistaar path as
    ``get_wer_score``/``get_cer_score`` (NFC + lightweight indic-nlp for Indic
    languages), then word-aligned with ``difflib.SequenceMatcher``. Each
    *differing* segment (a ``replace`` opcode with words on both sides) is
    judged by ``sarvam_llm_wer.equivalence_judge`` for semantic/phonetic
    equivalence; equivalent segments are rewritten to the reference
    ("forgiven"). Insertions and deletions are never forgiven. The corrected
    pairs are then scored with calibrate_agent's own jiwer WER/CER
    (``_score_edit_metric``), so ``llm_wer``/``llm_cer`` are directly
    comparable to the top-level ``wer``/``cer`` metrics — the delta is purely
    the effect of equivalence forgiveness.

    Unique segment pairs are judged once and reused across rows (dedup).
    ``cache_path`` (optional) additionally persists each segment verdict to a
    JSONL file and skips already-judged segments on a resumed run — restoring
    upstream's on-disk segment cache.

    Returns:
        {
            "llm_wer": float,     # dataset-pooled WER after forgiveness
            "llm_cer": float,     # dataset-pooled CER after forgiveness
            "per_row": [
                {
                    "llm_wer": float,          # per-utterance corrected WER
                    "llm_cer": float,          # per-utterance corrected CER
                    "segments": [              # judged differing segments
                        {"reference": str, "prediction": str,
                         "equivalent": bool, "reasoning": str},
                        ...
                    ],
                },
                ...
            ],
        }

    ``model`` defaults to ``DEFAULT_LLM_WER_MODEL`` when omitted. ``per_row``
    order matches the input order.
    """
    if not references:
        return {"llm_wer": 0.0, "llm_cer": 0.0, "per_row": []}

    from calibrate_agent.stt import sarvam_llm_wer
    from calibrate_agent.stt.sarvam_llm_wer import (
        DEFAULT_LLM_WER_MODEL,
        get_segments,
    )

    if model is None:
        model = DEFAULT_LLM_WER_MODEL

    norm_references, norm_predictions = await asyncio.to_thread(
        _normalize_refs_preds, references, predictions, language
    )

    # Word-align every row; segment equivalence is judged (and cached) per row.
    row_segments: List[List[dict]] = []
    for i, (ref, pred) in enumerate(zip(norm_references, norm_predictions)):
        row_segments.append(get_segments(str(ref), str(pred), key=i))

    row_keys = [_row_cache_key(i) for i in range(len(row_segments))]

    async def _judge_row_segments(row_i: int) -> list:
        differing = [
            seg
            for seg in row_segments[row_i]
            if seg["tag"] != "equal"
            and seg["reference"].strip()
            and seg["prediction"].strip()
        ]
        # Judge a row's differing segments concurrently; order is preserved so
        # the cached per-row verdict list is deterministic.
        results = await asyncio.gather(
            *(
                sarvam_llm_wer.equivalence_judge(
                    seg["reference"], seg["prediction"], model=model
                )
                for seg in differing
            )
        )
        return [
            {
                "reference": seg["reference"],
                "prediction": seg["prediction"],
                "equivalent": bool(res["equivalent"]),
                "reasoning": res["reasoning"],
            }
            for seg, res in zip(differing, results)
        ]

    def _make(i: int):
        return _judge_row_segments(i)

    row_verdicts_list = await _gather_cached(
        cache_path, row_keys, _make, desc="Running LLM-WER equivalence judge"
    )

    verdicts: dict = {}
    for row_verdicts in row_verdicts_list:
        for item in row_verdicts:
            verdicts[(item["reference"], item["prediction"])] = item

    # Reconstruct corrected reference/prediction per row: forgive equivalent
    # segments by rewriting the prediction side to the reference.
    corrected_references: List[str] = []
    corrected_predictions: List[str] = []
    per_row_segments: List[List[dict]] = []
    for segments in row_segments:
        ref_parts: List[str] = []
        pred_parts: List[str] = []
        seg_log: List[dict] = []
        for seg in segments:
            pair = (seg["reference"], seg["prediction"])
            verdict = verdicts.get(pair)
            forgiven = seg["tag"] == "equal" or (
                verdict is not None and verdict["equivalent"]
            )
            ref_parts.append(seg["reference"])
            pred_parts.append(seg["reference"] if forgiven else seg["prediction"])
            if verdict is not None:
                seg_log.append(
                    {
                        "reference": seg["reference"],
                        "prediction": seg["prediction"],
                        "equivalent": verdict["equivalent"],
                        "reasoning": verdict["reasoning"],
                    }
                )
        corrected_references.append(" ".join(ref_parts).strip())
        corrected_predictions.append(" ".join(pred_parts).strip())
        per_row_segments.append(seg_log)

    wer_scored = _score_edit_metric(
        jiwer.wer, _WER_TRANSFORM, corrected_references, corrected_predictions
    )
    cer_scored = _score_edit_metric(
        jiwer.cer, _CER_TRANSFORM, corrected_references, corrected_predictions
    )

    per_row = [
        {
            "llm_wer": float(row_wer),
            "llm_cer": float(row_cer),
            "segments": seg_log,
        }
        for row_wer, row_cer, seg_log in zip(
            wer_scored["per_row"],
            cer_scored["per_row"],
            per_row_segments,
        )
    ]

    return {
        "llm_wer": wer_scored["score"],
        "llm_cer": cer_scored["score"],
        "per_row": per_row,
    }
