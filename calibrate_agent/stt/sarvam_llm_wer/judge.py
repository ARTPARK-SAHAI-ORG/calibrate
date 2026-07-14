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

import backoff
import instructor

from calibrate_agent.judges import _build_openrouter_client
from calibrate_agent.langfuse import observe, langfuse, langfuse_enabled
from calibrate_agent.utils import log_judge_io
from calibrate_agent.stt.sarvam_llm_wer.main import LLMEquivalenceResponse, build_prompt

# Model used to grade segment equivalence. Matches Sarvam's llm_wer flow, which
# judges with google/gemini-2.5-flash (reached here via OpenRouter).
DEFAULT_LLM_WER_MODEL = "google/gemini-2.5-flash"


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(name="stt_llm_wer_judge", capture_input=False)
async def equivalence_judge(
    reference: str,
    prediction: str,
    model: str = DEFAULT_LLM_WER_MODEL,
    index: int = 0,
) -> dict:
    """Judge whether a reference/prediction word segment is equivalent.

    Args:
        reference: Reference-side words of a differing segment (normalized).
        prediction: Prediction-side words of the same segment (normalized).
        model: OpenRouter model id used for grading.
        index: Row index echoed into the input/output JSON.

    Returns:
        Dict matching :class:`LLMEquivalenceResponse` fields
        (``index``, ``equivalent``, ``reasoning``).
    """
    client = instructor.apatch(_build_openrouter_client())

    prompt = build_prompt({"reference": reference, "prediction": prediction})

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_model=LLMEquivalenceResponse,
        temperature=0,
        max_completion_tokens=8192,
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
