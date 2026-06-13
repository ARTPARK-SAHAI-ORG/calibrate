"""
Per-row intent/entity judge.

A thin wrapper over the vendored prompt/schema in this package: it builds the
prompt with ``build_prompt``, asks for the ``IntentEntityResponse`` schema, and
routes the call through calibrate's OpenRouter + ``instructor`` client. The
aggregation entry point used by the eval pipeline, ``get_intent_entity_score``,
lives in ``stt/metrics.py`` alongside the other metric roots.

A single judge call per row returns both intent (0/1) and entity (0–1) scores.
"""

import backoff
import instructor

from calibrate.judges import _build_openrouter_client
from calibrate.langfuse import observe, langfuse, langfuse_enabled
from calibrate.utils import log_judge_io
from calibrate.stt.sarvam_intent_entity.main import IntentEntityResponse, build_prompt

# Model used to grade intent/entity. Matches Sarvam's llm_intent_entity flow,
# which judges with google/gemini-2.5-flash (reached here via OpenRouter).
DEFAULT_INTENT_ENTITY_MODEL = "google/gemini-2.5-flash"


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(name="stt_intent_entity_judge", capture_input=False)
async def intent_entity_judge(
    reference: str,
    prediction: str,
    model: str = DEFAULT_INTENT_ENTITY_MODEL,
    index: int = 0,
    context: str = "",
) -> dict:
    """Score intent (0/1) and entity (0–1) preservation for one transcription.

    Args:
        reference: Ground-truth text (already normalized by the caller).
        prediction: STT hypothesis (already normalized by the caller).
        model: OpenRouter model id used for grading.
        index: Row index echoed into the input/output JSON.
        context: Optional context passed through to the judge.

    Returns:
        Dict matching :class:`IntentEntityResponse` fields.
    """
    client = instructor.apatch(_build_openrouter_client())

    prompt = build_prompt(
        {
            "index": index,
            "hypothesis": prediction,
            "ground_truth": reference,
            "context": context,
        }
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_model=IntentEntityResponse,
        temperature=0,
        max_completion_tokens=8192,
    )

    result = response.model_dump()

    log_judge_io(
        evaluator="intent_entity",
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
