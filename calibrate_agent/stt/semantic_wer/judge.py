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
            counts plus a concise ``summary`` verdict. That summary — not the
            verbose phase-1 CoT — becomes the row's public ``reasoning``; the raw
            CoT is kept only in the per-provider debug log.

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

# Cache the long, identical-every-row system prompt. Both phases and every row
# share this exact prefix, so within the cache window rows are billed as cache
# reads instead of full prompt tokens. OpenRouter forwards the Anthropic-style
# breakpoint; providers without prompt caching simply ignore it.
_CACHED_SYSTEM_MESSAGE = {
    "role": "system",
    "content": [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
    ],
}


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
    ``normalized_reference``, ``normalized_hypothesis``, ``reasoning`` (the
    concise public summary), and ``chain_of_thought`` (the full phase-1 CoT,
    kept for debug surfaces only — never persisted to the leaderboard).
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
            _CACHED_SYSTEM_MESSAGE,
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
            _CACHED_SYSTEM_MESSAGE,
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reasoning},
            {
                "role": "user",
                "content": (
                    "Now call calculate_wer with your final counts and a concise "
                    "`summary` (1-2 sentences, plain language) explaining the "
                    "semantic errors you counted, or that there were none. The "
                    "summary is shown publicly — do not restate your step-by-step "
                    "working."
                ),
            },
        ],
        tools=[_OPENAI_TOOL],
        tool_choice=_FORCE_WER_TOOL,
        temperature=0,
        # Headroom for the tool-call JSON: it carries normalized_reference +
        # normalized_hypothesis (the full texts) and an errors list, which for a
        # long utterance can exceed a tight cap. A truncated call is unparseable
        # and, at temperature 0, fails identically on every retry — so undersize
        # here would deterministically drop the whole run's semantic WER.
        max_tokens=4096,
    )
    message = commit_resp.choices[0].message
    wer_call = next(
        (
            tc
            for tc in (message.tool_calls or [])
            if tc.function.name == "calculate_wer"
        ),
        None,
    )
    if wer_call is None:
        raise ValueError(
            "semantic_wer_judge: forced calculate_wer call was not returned"
        )
    tool_input = json.loads(wer_call.function.arguments)

    # The publicly-shown reasoning is the model's concise phase-2 ``summary``, not
    # the phase-1 chain-of-thought — the CoT is verbose working-out that leaks
    # into the leaderboard UI. Fall back to the CoT only if the summary is missing
    # (older models / truncated calls), so a row is never left without reasoning.
    summary = (tool_input.get("summary") or "").strip()
    result = {
        "substitutions": int(tool_input.get("substitutions", 0)),
        "deletions": int(tool_input.get("deletions", 0)),
        "insertions": int(tool_input.get("insertions", 0)),
        # pipecat defaults a missing reference_words to 1 at its call site.
        "reference_words": int(tool_input.get("reference_words", 1)),
        "normalized_reference": tool_input.get("normalized_reference") or "",
        "normalized_hypothesis": tool_input.get("normalized_hypothesis") or "",
        "reasoning": summary or reasoning,
        # Full CoT rides along for the debug surfaces below (log + Langfuse). It
        # never reaches the leaderboard: get_semantic_wer_score and eval.py both
        # cherry-pick named keys, so this one is dropped. pipecat likewise keeps
        # the raw reasoning trace only for offline debugging.
        "chain_of_thought": reasoning,
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
