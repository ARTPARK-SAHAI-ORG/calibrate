"""Semantic-WER judge — two-phase reason-then-commit over OpenRouter.

Same rubric and tool contract as pipecat's stt-benchmark
``SemanticWEREvaluator.evaluate`` (prompt + ``calculate_wer`` schema are
pipecat's verbatim; see ``main.py``), but structured for reliability as two
explicit calls per pair instead of pipecat's single auto tool-use loop:

  Phase 1 — with the rules in a **system** message and the (reference,
            hypothesis) pair in a **user** message, the model reasons freely.
            No tool is offered, so it can only produce text.
  Phase 2 — the phase-1 reasoning is replayed and the model is *required* to
            call ``calculate_wer`` (forced ``tool_choice``), committing the
            counts.

This guarantees termination in exactly two calls (no nudge loop, no
runaway-retry worst case) at the cost of one extra call per row versus
pipecat's single-call-usually loop. Transport is calibrate's OpenRouter
(OpenAI-compatible) client, not the native Anthropic SDK. Seams: ``@backoff`` /
``@observe`` / ``log_judge_io`` and a Unicode NFC pre-normalization of the
inputs (for Indic scripts; pipecat targets English and normalizes nothing
upstream).
"""

import json
import unicodedata

import backoff

from calibrate_agent.judges import _build_openrouter_client
from calibrate_agent.langfuse import observe, langfuse, langfuse_enabled
from calibrate_agent.utils import log_judge_io
from calibrate_agent.stt.semantic_wer.main import (
    SYSTEM_PROMPT,
    CALCULATE_WER_TOOL,
    build_user_prompt,
)

# pipecat's stt-benchmark judges with Claude; reached here via OpenRouter.
# Overridable per call / via get_semantic_wer_score(model=...).
DEFAULT_SEMANTIC_WER_MODEL = "anthropic/claude-sonnet-4.5"

# Phase 2 requires the model to call calculate_wer (no free-form escape).
_FORCE_WER_TOOL = {"type": "function", "function": {"name": "calculate_wer"}}


def _short_circuit(
    substitutions: int,
    deletions: int,
    insertions: int,
    reference_words: int,
    reasoning: str,
) -> dict:
    """Deterministic empty-input result (no LLM call), pipecat's shape."""
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_words": reference_words,
        "normalized_reference": "",
        "normalized_hypothesis": "",
        "reasoning": reasoning,
    }

# pipecat's Anthropic tool → OpenAI function-calling shape. The Anthropic
# ``input_schema`` doubles as the OpenAI ``parameters`` object unchanged.
_OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": CALCULATE_WER_TOOL["name"],
        "description": CALCULATE_WER_TOOL["description"],
        "parameters": CALCULATE_WER_TOOL["input_schema"],
    },
}


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(name="stt_semantic_wer_judge", capture_input=False)
async def semantic_wer_judge(
    reference: str,
    prediction: str,
    model: str = DEFAULT_SEMANTIC_WER_MODEL,
) -> dict:
    """Score one (reference, hypothesis) pair with pipecat's tool-use loop.

    Returns the ``calculate_wer`` counts plus reasoning:
    ``substitutions``, ``deletions``, ``insertions``, ``reference_words``,
    ``normalized_reference``, ``normalized_hypothesis``, ``reasoning``.
    """
    # Seam: NFC pre-normalization (see module docstring). Applied before the
    # judge sees the text so Indic codepoint variants aren't counted as errors.
    reference = unicodedata.normalize("NFC", reference)
    prediction = unicodedata.normalize("NFC", prediction)

    # pipecat evaluate(): empty-input short-circuits, ported verbatim. These are
    # deterministic — no LLM call — matching _empty_result / _no_reference_result
    # / _no_hypothesis_result. get_semantic_wer_score turns these counts into the
    # same per-row WER pipecat reports (0.0 / inf / 1.0 respectively).
    if not reference.strip() and not prediction.strip():
        return _short_circuit(0, 0, 0, 0, reasoning="empty reference and hypothesis")
    if not reference.strip():
        return _short_circuit(
            0, 0, len(prediction.split()), 0, reasoning="empty reference"
        )
    if not prediction.strip():
        words = len(reference.split())
        return _short_circuit(0, words, 0, words, reasoning="empty hypothesis")

    client = _build_openrouter_client()
    user_msg = build_user_prompt(reference, prediction)

    # Phase 1 — reason freely. No tool is offered, so the model can only write
    # its NORMALIZE → ALIGN → SEMANTIC CHECK working out as text.
    reason_resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=4096,
    )
    reasoning = reason_resp.choices[0].message.content or ""

    # Phase 2 — replay the reasoning and REQUIRE the calculate_wer call, so the
    # counts are always committed (no nudge loop, guaranteed to terminate).
    commit_resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": "Now call calculate_wer with your final counts."},
        ],
        tools=[_OPENAI_TOOL],
        tool_choice=_FORCE_WER_TOOL,
        temperature=0,
        max_tokens=1024,
    )
    message = commit_resp.choices[0].message
    wer_call = next(
        (tc for tc in (message.tool_calls or []) if tc.function.name == "calculate_wer"),
        None,
    )
    if wer_call is None:
        raise ValueError("semantic_wer_judge: forced calculate_wer call was not returned")
    tool_input = json.loads(wer_call.function.arguments)

    result = {
        "substitutions": int(tool_input.get("substitutions", 0)),
        "deletions": int(tool_input.get("deletions", 0)),
        "insertions": int(tool_input.get("insertions", 0)),
        # pipecat defaults a missing reference_words to 1 at its call site.
        "reference_words": int(tool_input.get("reference_words", 1)),
        "normalized_reference": tool_input.get("normalized_reference") or "",
        "normalized_hypothesis": tool_input.get("normalized_hypothesis") or "",
        "reasoning": reasoning,
    }

    log_judge_io(
        evaluator="semantic_wer",
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_input=user_msg,
        output=result,
    )

    if langfuse_enabled and langfuse:
        langfuse.update_current_trace(
            input={"reference": reference, "prediction": prediction},
            output=result,
            metadata={"reference": reference, "prediction": prediction, "model": model},
        )

    return result
