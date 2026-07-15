"""Semantic-WER judge — pipecat's reason-then-tool loop over OpenRouter.

Mirrors pipecat's stt-benchmark ``SemanticWEREvaluator.evaluate``: a multi-turn
tool-use loop where the model writes its free-form semantic reasoning and then
commits the counts via a ``calculate_wer`` tool call, with the rules in a
**system** message and the (reference, hypothesis) pair in a **user** message.
The prompt and tool schema are pipecat's verbatim (``main.py``).

The one transport difference from pipecat: calibrate stays on its OpenRouter
(OpenAI-compatible) client rather than the native Anthropic SDK, so the
Anthropic ``tool_use`` loop is expressed as OpenAI function-calling (same
reason→commit shape, same tool contract). Seams: ``@backoff`` / ``@observe`` /
``log_judge_io`` and a Unicode NFC pre-normalization of the inputs (for Indic
scripts; pipecat targets English and normalizes nothing upstream).
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

_MAX_TURNS = 10  # pipecat's safety limit


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

    # System = pipecat's rules; user = pipecat's per-pair message.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(reference, prediction)},
    ]

    tool_input = None
    reasoning = ""

    for _ in range(_MAX_TURNS):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[_OPENAI_TOOL],
            tool_choice="auto",
            temperature=0,
            max_tokens=4096,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        wer_call = next(
            (tc for tc in tool_calls if tc.function.name == "calculate_wer"), None
        )
        if wer_call is not None:
            # The assistant's free-form reasoning precedes the committed counts.
            reasoning = message.content or ""
            tool_input = json.loads(wer_call.function.arguments)
            break

        # Model reasoned without committing yet — feed its turn back and ask it
        # to finish with the tool call (pipecat's loop likewise keeps going).
        messages.append({"role": "assistant", "content": message.content or ""})
        messages.append(
            {"role": "user", "content": "Now call calculate_wer with your counts."}
        )

    if tool_input is None:
        raise ValueError(
            "semantic_wer_judge: model did not call calculate_wer within "
            f"{_MAX_TURNS} turns"
        )

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
        user_input=build_user_prompt(reference, prediction),
        output=result,
    )

    if langfuse_enabled and langfuse:
        langfuse.update_current_trace(
            input={"reference": reference, "prediction": prediction},
            output=result,
            metadata={"reference": reference, "prediction": prediction, "model": model},
        )

    return result
