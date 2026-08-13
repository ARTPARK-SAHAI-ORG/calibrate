"""
Per-segment equivalence judge for LLM-WER/CER.

A thin wrapper over the vendored prompt/schema in this package: it builds the
prompt with ``build_prompt``, asks for the ``LLMEquivalenceResponse`` schema, and
routes the call through calibrate_agent's OpenRouter + ``instructor`` client. The
aggregation entry point used by the eval pipeline, ``get_llm_wer_cer_score``,
lives in ``stt/metrics.py`` alongside the other metric roots.

One judge call per *unique differing word segment* returns whether the segment
is semantically/phonetically equivalent (forgiven) or a genuine error.
"""

from __future__ import annotations

from typing import List

import backoff
import instructor

from calibrate_agent.judges import _build_openrouter_client
from calibrate_agent.langfuse import observe, langfuse, langfuse_enabled
from calibrate_agent.utils import log_judge_io
from calibrate_agent.stt.sarvam_llm_wer.main import LLMEquivalenceResponse, build_prompt

# Model used to grade segment equivalence. Matches Sarvam's llm_wer flow, which
# judges with google/gemini-2.5-flash (reached here via OpenRouter).
DEFAULT_LLM_WER_MODEL = "google/gemini-2.5-flash"

_MULTI_TOOL_CALL_MARKER = "multiple tool calls"


def _is_multiple_tool_calls_error(exc: BaseException) -> bool:
    """True when instructor rejected a response that contained >1 tool call."""
    return _MULTI_TOOL_CALL_MARKER in str(exc).lower()


async def _create_equivalence_response(client, *, model: str, prompt: str):
    """Ask instructor for one ``LLMEquivalenceResponse``.

    Some models (notably Gemini via OpenRouter) occasionally emit two tool
    calls for the same single-segment prompt. Instructor rejects that with
    ``multiple tool calls``. In that case we retry once asking for a list and
    keep the first element — both calls are usually duplicates of one verdict.
    """
    create = client.chat.completions.create
    shared = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=8192,
    )
    try:
        return await create(response_model=LLMEquivalenceResponse, **shared)
    except Exception as exc:
        if not _is_multiple_tool_calls_error(exc):
            raise
        responses = await create(
            response_model=List[LLMEquivalenceResponse], **shared
        )
        if not responses:
            raise
        return responses[0]


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(name="stt_llm_wer_judge", capture_input=False)
async def equivalence_judge(
    reference: str,
    prediction: str,
    model: str = DEFAULT_LLM_WER_MODEL,
) -> dict:
    """Judge whether a reference/prediction word segment is equivalent.

    Args:
        reference: Reference-side words of a differing segment (normalized).
        prediction: Prediction-side words of the same segment (normalized).
        model: OpenRouter model id used for grading.

    Returns:
        Dict matching :class:`LLMEquivalenceResponse` fields
        (``index``, ``equivalent``, ``reasoning``).
    """
    client = instructor.apatch(_build_openrouter_client())

    prompt = build_prompt({"reference": reference, "prediction": prediction})

    response = await _create_equivalence_response(
        client, model=model, prompt=prompt
    )

    result = response.model_dump()

    log_judge_io(
        evaluator="llm_wer",
        model=model,
        system_prompt="",
        user_input=prompt,
        output=result,
    )

    if langfuse_enabled and langfuse:
        langfuse.update_current_trace(
            input={"reference": reference, "prediction": prediction},
            output=result,
            metadata={
                "reference": reference,
                "prediction": prediction,
                "model": model,
            },
        )

    return result
