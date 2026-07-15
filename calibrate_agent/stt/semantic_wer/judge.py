"""Holistic semantic-WER judge — one LLM call per (reference, hypothesis) pair.

Mirrors pipecat's stt-benchmark semantic WER: the model normalizes, aligns,
applies the semantic error check, and returns S/D/I + reference word counts,
which ``get_semantic_wer_score`` turns into a WER. Routes through calibrate's
OpenRouter + ``instructor`` client (same plumbing as the Sarvam LLM-WER and text
judges).
"""

import unicodedata

import backoff
import instructor

from calibrate_agent.judges import _build_openrouter_client
from calibrate_agent.langfuse import observe, langfuse, langfuse_enabled
from calibrate_agent.utils import log_judge_io
from calibrate_agent.stt.semantic_wer.main import SemanticWERResponse, build_prompt

# pipecat's stt-benchmark judges with Claude; reached here via OpenRouter.
# Overridable per call / via get_semantic_wer_score(model=...).
DEFAULT_SEMANTIC_WER_MODEL = "anthropic/claude-sonnet-4.5"


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(name="stt_semantic_wer_judge", capture_input=False)
async def semantic_wer_judge(
    reference: str,
    prediction: str,
    model: str = DEFAULT_SEMANTIC_WER_MODEL,
) -> dict:
    """Score one (reference, hypothesis) pair with the semantic-WER rubric.

    Returns a dict matching :class:`SemanticWERResponse`
    (``substitutions``, ``deletions``, ``insertions``, ``reference_words``,
    ``normalized_reference``, ``normalized_hypothesis``, ``reasoning``).
    """
    # Canonicalize Unicode (NFC) before the judge sees the text. pipecat targets
    # English and applies no upstream normalization, but this repo runs Indic
    # scripts where visually identical strings can differ at the codepoint level
    # (NFC vs NFD, nukta composition) — without this the LLM may count those as
    # spurious substitutions. NFC only: lossless canonicalization, no case /
    # punctuation stripping (the judge's inline STEP 1 handles the rest).
    reference = unicodedata.normalize("NFC", reference)
    prediction = unicodedata.normalize("NFC", prediction)

    client = instructor.apatch(_build_openrouter_client())

    prompt = build_prompt(reference, prediction)

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_model=SemanticWERResponse,
        temperature=0,
        max_completion_tokens=8192,
    )

    result = response.model_dump()

    log_judge_io(
        evaluator="semantic_wer",
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
