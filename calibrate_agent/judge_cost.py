"""Up-front cost estimate for LLM-as-judge grading.

A judge run issues one LLM call per dataset row per evaluator, so a large
dataset turns into thousands of paid calls. This module estimates what that
will cost before any call is made, and offers a confirmation gate so the run
can be abandoned while it is still free.

The workload is described as a list of :class:`JudgeCallGroup` — one per judge,
each carrying its model, its call count, and the per-call token sizes.
:func:`estimate_judge_cost_all_sources` prices those groups from the bundled
rate table, :func:`format_cost_estimate` renders the result, and
:func:`confirm_judge_cost` asks the user whether to continue.

Token counts are estimated from character counts, weighted by writing system —
there is no tokenizer dependency, so the figures are approximations.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, TextIO

from calibrate_agent._cli_args import DEFAULT_STT_LLM_JUDGES
from calibrate_agent.judges import (
    DEFAULT_AUDIO_JUDGE_MODEL,
    DEFAULT_TEXT_JUDGE_MODEL,
    DEFAULT_TTS_EVALUATOR,
)
from calibrate_agent.pricing import LLM_PRICING_SOURCES, resolve_llm_pricing

# Unicode ranges whose characters cost far more tokens per character than
# Latin text does: the Indic scripts (Devanagari through Malayalam) and Arabic
# (the script Sindhi is written in).
INDIC_ARABIC_RANGES = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
)

# Characters per token. The flat chars/4 rule of thumb holds for Latin text but
# undercounts Indic text by 2-3x, and the datasets are largely Hindi, Kannada
# and Telugu, so the two writing systems are counted at separate rates.
CHARS_PER_TOKEN_LATIN = 4.0
CHARS_PER_TOKEN_INDIC = 1.5

# Audio input tokens per second of audio, at OpenAI's tokenization rate. The
# audio judge model bills audio per token, not per minute, so measured duration
# has to be converted into tokens before it can be priced.
AUDIO_TOKENS_PER_SECOND = 10.0

# Tokens of text spoken per second of speech. Used to size audio that does not
# exist yet (a TTS estimate is made before anything is synthesized).
SPOKEN_TOKENS_PER_SECOND = 3.5

# Multiplier applied to the output-token estimate for models that bill hidden
# reasoning tokens at the output rate. Those tokens never appear in the
# response, so an estimate built from the visible answer alone runs low. The
# multiplier is deliberately generous: quoting too high is recoverable,
# quoting too low is not.
REASONING_TOKEN_MULTIPLIER = 3.0

# Output-token allowances for each judge's reply shape, used by the group
# builders below. Every judge caps its call far higher than these (noted per
# constant); each value is a deliberately modest guess at what a real reply
# actually contains, not the saturation point of the cap.

# STT/TTS evaluator judges (``_judge_one_text`` / ``_judge_one_audio`` in
# judges.py, capped at max_completion_tokens=8192): a short ``reasoning``
# string plus a one-field verdict (``match`` or ``score``).
EVALUATOR_OUTPUT_TOKENS = 300

# Sarvam intent/entity judge (``IntentEntityResponse``, capped at
# max_completion_tokens=8192): five text fields (two explanations plus three
# entity lists) rather than one, so it runs larger than a single evaluator
# verdict.
INTENT_ENTITY_OUTPUT_TOKENS = 600

# Sarvam LLM-WER/CER equivalence judge (``LLMEquivalenceResponse``, capped at
# max_completion_tokens=8192): a bool plus one short ``reasoning`` string —
# the smallest reply of the structured judges.
LLM_WER_OUTPUT_TOKENS = 200

# Semantic WER phase 1 — the free-form NORMALIZE/ALIGN/SEMANTIC-CHECK
# reasoning (capped at max_tokens=4096). It has no fixed schema, so this is a
# rough middle ground rather than a reply-shape derivation.
SEMANTIC_WER_REASONING_OUTPUT_TOKENS = 500

# Semantic WER phase 2 — the forced ``calculate_wer`` tool call (capped at
# max_tokens=4096). Besides small counts/summary fields, its arguments echo
# back the row's *entire* normalized reference and hypothesis, so its size
# scales with the row's own text; callers add the row's own token count on top
# of this fixed allowance for the rest of the call.
SEMANTIC_WER_TOOL_CALL_OVERHEAD_TOKENS = 150

_ENV_ASSUME_YES = "CALIBRATE_ASSUME_YES"


def _is_indic_or_arabic(char: str) -> bool:
    code_point = ord(char)
    return any(start <= code_point <= end for start, end in INDIC_ARABIC_RANGES)


def estimate_tokens(text: str) -> int:
    """Estimate how many tokens ``text`` will be billed as.

    Each character is classified by Unicode block and counted at its script's
    characters-per-token rate, so mixed-script text lands between the pure
    cases. Unclassified characters count at the Latin rate. Empty or
    whitespace-only text is 0 tokens; anything else is at least 1.
    """
    if not text or not text.strip():
        return 0

    indic_chars = sum(1 for char in text if _is_indic_or_arabic(char))
    latin_chars = len(text) - indic_chars
    tokens = (
        indic_chars / CHARS_PER_TOKEN_INDIC + latin_chars / CHARS_PER_TOKEN_LATIN
    )
    return max(1, math.ceil(tokens))


def estimate_audio_seconds_from_text(text: str) -> float:
    """Estimate how many seconds of speech ``text`` becomes when synthesized.

    Derived from the token estimate rather than the character count, which
    keeps the implied speaking rate roughly script-invariant: a Devanagari
    character carries more speech than a Latin one.
    """
    return estimate_tokens(text) / SPOKEN_TOKENS_PER_SECOND


@dataclass
class JudgeCallGroup:
    """One judge's workload: the same model and call shape repeated ``calls`` times."""

    label: str
    model: str
    calls: int
    input_tokens_per_call: int
    output_tokens_per_call: int
    audio_seconds_per_call: float = 0.0


def _mean_tokens(values) -> int:
    """Return the rounded mean of an iterable of token counts, or 0 if empty."""
    values = list(values)
    if not values:
        return 0
    return int(round(sum(values) / len(values)))


def build_stt_judge_groups(
    references: Sequence[str],
    predictions: Optional[Sequence[str]] = None,
    evaluators: Optional[Sequence[dict]] = None,
    llm_judges: frozenset[str] | None = None,
    providers: int = 1,
) -> list[JudgeCallGroup]:
    """Build the judge workload for an STT run, mirroring ``_score_and_write_results``.

    One group per judge that will actually run: the user ``evaluators`` (via
    ``get_llm_judge_score``), grouped by resolved model so a config mixing
    models prices correctly; and each enabled built-in judge from
    ``llm_judges`` (``intent``, ``llm_wer``, ``semantic_wer``). ``llm_judges``
    of ``None`` means all three; an empty frozenset means none. Semantic WER
    splits into two call groups per row (free-form reasoning, then a forced
    tool-call commit). The evaluators group only appears when ``evaluators``
    is non-empty. Returns ``[]`` when neither would run, or when
    ``references`` is empty.

    The Sarvam LLM-WER/CER judge actually calls its equivalence judge once per
    *unique differing word segment* after alignment, not once per row — but
    the segments don't exist until WER is computed, so this approximates one
    call per row using each row's full (reference, prediction) text as a
    stand-in segment (over-counting short/matching rows, under-counting rows
    with several differing segments).

    ``predictions`` is the STT hypothesis for each reference. When omitted (a
    benchmark estimate runs before any transcription happens), each row's
    prediction is assumed to be the same size as its reference, since the real
    text doesn't exist yet.

    ``providers`` multiplies every group's call count: a multi-provider
    benchmark runs every judge once per provider's transcriptions.
    """
    n = len(references)
    if n == 0:
        return []
    preds = list(predictions) if predictions is not None else list(references)
    enabled = DEFAULT_STT_LLM_JUDGES if llm_judges is None else llm_judges

    groups: list[JudgeCallGroup] = []

    if evaluators:
        by_model: dict[str, list[dict]] = {}
        for ev in evaluators:
            model = ev.get("judge_model") or DEFAULT_TEXT_JUDGE_MODEL
            by_model.setdefault(model, []).append(ev)

        # The STT evaluator user prompt (``stt_llm_judge`` in stt/metrics.py)
        # doesn't depend on the evaluator, so its average is computed once and
        # added to each model group's own average system-prompt size.
        avg_user_tokens = _mean_tokens(
            estimate_tokens(f"Source: {ref}\nTranscription: {pred}")
            for ref, pred in zip(references, preds)
        )

        for model, evs in by_model.items():
            avg_system_tokens = _mean_tokens(
                estimate_tokens(ev.get("system_prompt", "")) for ev in evs
            )
            groups.append(
                JudgeCallGroup(
                    label=f"STT evaluators ({', '.join(ev['name'] for ev in evs)})",
                    model=model,
                    calls=n * len(evs) * providers,
                    input_tokens_per_call=avg_system_tokens + avg_user_tokens,
                    output_tokens_per_call=EVALUATOR_OUTPUT_TOKENS,
                )
            )

    if not enabled:
        return groups

    # Deferred: importing sarvam_intent_entity eagerly pulls in
    # transformers/indic-nlp/joblib (see stt/metrics.py), which should only
    # happen when these judges are actually going to run.
    if "intent" in enabled:
        from calibrate_agent.stt.sarvam_intent_entity import (
            DEFAULT_INTENT_ENTITY_MODEL,
            build_prompt as build_intent_entity_prompt,
        )

        intent_entity_tokens = _mean_tokens(
            estimate_tokens(
                build_intent_entity_prompt(
                    {
                        "index": i,
                        "hypothesis": pred,
                        "ground_truth": ref,
                        "context": "",
                    }
                )
            )
            for i, (ref, pred) in enumerate(zip(references, preds))
        )
        groups.append(
            JudgeCallGroup(
                label="Sarvam intent/entity",
                model=DEFAULT_INTENT_ENTITY_MODEL,
                calls=n * providers,
                input_tokens_per_call=intent_entity_tokens,
                output_tokens_per_call=INTENT_ENTITY_OUTPUT_TOKENS,
            )
        )

    if "llm_wer" in enabled:
        from calibrate_agent.stt.sarvam_llm_wer import (
            DEFAULT_LLM_WER_MODEL,
            build_prompt as build_llm_wer_prompt,
        )

        llm_wer_tokens = _mean_tokens(
            estimate_tokens(build_llm_wer_prompt({"reference": ref, "prediction": pred}))
            for ref, pred in zip(references, preds)
        )
        groups.append(
            JudgeCallGroup(
                label="Sarvam LLM-WER/CER",
                model=DEFAULT_LLM_WER_MODEL,
                calls=n * providers,
                input_tokens_per_call=llm_wer_tokens,
                output_tokens_per_call=LLM_WER_OUTPUT_TOKENS,
            )
        )

    if "semantic_wer" in enabled:
        from calibrate_agent.stt.semantic_wer import (
            DEFAULT_SEMANTIC_WER_MODEL,
            SYSTEM_PROMPT as SEMANTIC_WER_SYSTEM_PROMPT,
            build_user_prompt as build_semantic_wer_user_prompt,
        )

        semantic_wer_system_tokens = estimate_tokens(SEMANTIC_WER_SYSTEM_PROMPT)
        semantic_wer_user_tokens = _mean_tokens(
            estimate_tokens(build_semantic_wer_user_prompt(ref, pred))
            for ref, pred in zip(references, preds)
        )
        semantic_wer_row_text_tokens = _mean_tokens(
            estimate_tokens(ref) + estimate_tokens(pred)
            for ref, pred in zip(references, preds)
        )
        groups.append(
            JudgeCallGroup(
                label="Semantic WER (reasoning)",
                model=DEFAULT_SEMANTIC_WER_MODEL,
                calls=n * providers,
                input_tokens_per_call=semantic_wer_system_tokens
                + semantic_wer_user_tokens,
                output_tokens_per_call=SEMANTIC_WER_REASONING_OUTPUT_TOKENS,
            )
        )
        groups.append(
            JudgeCallGroup(
                label="Semantic WER (commit)",
                model=DEFAULT_SEMANTIC_WER_MODEL,
                calls=n * providers,
                # Phase 2 replays phase 1's prompt plus its reasoning plus a
                # short nudge, so it costs strictly more input than phase 1;
                # approximated by adding phase 1's own output-token estimate.
                input_tokens_per_call=(
                    semantic_wer_system_tokens
                    + semantic_wer_user_tokens
                    + SEMANTIC_WER_REASONING_OUTPUT_TOKENS
                ),
                output_tokens_per_call=semantic_wer_row_text_tokens
                + SEMANTIC_WER_TOOL_CALL_OVERHEAD_TOKENS,
            )
        )

    return groups


def build_tts_judge_groups(
    texts: Sequence[str],
    audio_seconds: Optional[Sequence[float]] = None,
    evaluators: Optional[Sequence[dict]] = None,
    providers: int = 1,
) -> list[JudgeCallGroup]:
    """Build the judge workload for a TTS run, mirroring ``tts_llm_judge``.

    One group per resolved evaluator model (evaluators are grouped by model,
    like :func:`build_stt_judge_groups`); every call carries the row's audio.
    ``audio_seconds[i]`` is used when known (a prior synthesis run's measured
    duration); a row with no known duration (or when ``audio_seconds`` is
    omitted entirely — a benchmark estimate runs before anything is
    synthesized) falls back to :func:`estimate_audio_seconds_from_text`.
    Returns ``[]`` when ``texts`` is empty.

    ``providers`` multiplies every group's call count: a multi-provider
    benchmark judges every provider's synthesized audio separately.
    """
    n = len(texts)
    if n == 0:
        return []
    _evaluators = list(evaluators) if evaluators else [DEFAULT_TTS_EVALUATOR]

    durations = list(audio_seconds) if audio_seconds is not None else [None] * n
    if len(durations) < n:
        durations = durations + [None] * (n - len(durations))
    resolved_durations = [
        d if d is not None else estimate_audio_seconds_from_text(str(t))
        for d, t in zip(durations, texts)
    ]
    avg_duration = sum(resolved_durations) / n

    # The audio judge's text preamble (``audio_judge`` in judges.py) doesn't
    # depend on the evaluator, so its average is computed once and added to
    # each model group's own average system-prompt size.
    avg_preamble_tokens = _mean_tokens(
        estimate_tokens(f"Reference text: {t}\n\nAudio:") for t in texts
    )

    by_model: dict[str, list[dict]] = {}
    for ev in _evaluators:
        model = ev.get("judge_model") or DEFAULT_AUDIO_JUDGE_MODEL
        by_model.setdefault(model, []).append(ev)

    groups: list[JudgeCallGroup] = []
    for model, evs in by_model.items():
        avg_system_tokens = _mean_tokens(
            estimate_tokens(ev.get("system_prompt", "")) for ev in evs
        )
        groups.append(
            JudgeCallGroup(
                label=f"TTS evaluators ({', '.join(ev['name'] for ev in evs)})",
                model=model,
                calls=n * len(evs) * providers,
                input_tokens_per_call=avg_system_tokens + avg_preamble_tokens,
                output_tokens_per_call=EVALUATOR_OUTPUT_TOKENS,
                audio_seconds_per_call=avg_duration,
            )
        )
    return groups


def estimate_judge_cost(
    groups: Sequence[JudgeCallGroup],
    source: str = "openrouter",
) -> dict:
    """Price ``groups`` against the ``source`` rate table.

    Text input tokens bill at the input rate and output tokens at the output
    rate, scaled by :data:`REASONING_TOKEN_MULTIPLIER` where the model bills
    reasoning as output. Audio seconds convert to tokens at
    :data:`AUDIO_TOKENS_PER_SECOND` and bill at the model's audio input rate,
    falling back to its text input rate when it prices audio as text.

    Returns the per-group breakdown, ``total_usd`` over the priced groups, and
    the sorted list of models with no rates in ``unpriced``. A group whose
    model is unpriced is reported with ``priced: False`` and contributes
    nothing to the total, so an unknown model does not block the estimate.
    """
    group_rows: list[dict] = []
    unpriced: set[str] = set()
    total_usd = 0.0

    for group in groups:
        input_tokens = group.input_tokens_per_call * group.calls
        output_tokens = group.output_tokens_per_call * group.calls
        audio_tokens = int(
            round(group.audio_seconds_per_call * AUDIO_TOKENS_PER_SECOND * group.calls)
        )
        row = {
            "label": group.label,
            "model": group.model,
            "calls": group.calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "audio_tokens": audio_tokens,
            "cost_usd": 0.0,
            "priced": False,
        }

        pricing = resolve_llm_pricing(group.model, source=source)
        if pricing is None:
            unpriced.add(group.model)
            group_rows.append(row)
            continue

        input_rate = pricing["input_price_per_million_tokens_usd"]
        output_rate = pricing["output_price_per_million_tokens_usd"]
        audio_rate = pricing.get(
            "audio_input_price_per_million_tokens_usd", input_rate
        )
        billed_output_tokens = output_tokens * (
            REASONING_TOKEN_MULTIPLIER
            if pricing.get("reasoning_billed_as_output")
            else 1.0
        )
        cost_usd = (
            input_tokens * input_rate
            + billed_output_tokens * output_rate
            + audio_tokens * audio_rate
        ) / 1_000_000

        row["cost_usd"] = cost_usd
        row["priced"] = True
        total_usd += cost_usd
        group_rows.append(row)

    return {
        "source": source,
        "groups": group_rows,
        "total_usd": total_usd,
        "unpriced": sorted(unpriced),
    }


def estimate_judge_cost_all_sources(groups: Sequence[JudgeCallGroup]) -> dict:
    """Price ``groups`` under every billing source, keyed by source name."""
    return {
        source: estimate_judge_cost(groups, source=source)
        for source in LLM_PRICING_SOURCES
    }


def format_cost_estimate(both: dict) -> str:
    """Render the output of :func:`estimate_judge_cost_all_sources` as a text block.

    One line per judge group, then a total for each billing source, then the
    caveats. OpenRouter charges provider list price on every model in the rate
    table, so the two totals normally match; both are reported so a divergence
    becomes visible if one appears.
    """
    openrouter = both.get("openrouter", {})
    direct = both.get("direct", {})
    groups = openrouter.get("groups", [])

    lines = ["Estimated LLM-as-judge cost for this run:", ""]
    for group in groups:
        usage = (
            f"{group['input_tokens']:,} input + {group['output_tokens']:,} output"
        )
        if group["audio_tokens"]:
            usage += f" + {group['audio_tokens']:,} audio"
        if group["priced"]:
            amount = f"${group['cost_usd']:.4f}"
        else:
            amount = "cost unknown (unpriced)"
        lines.append(
            f"  {group['label']} ({group['model']}): {group['calls']:,} calls, "
            f"{usage} tokens = {amount}"
        )

    openrouter_total = openrouter.get("total_usd", 0.0)
    direct_total = direct.get("total_usd", 0.0)
    lines.append("")
    lines.append(f"  Total via OpenRouter: ${openrouter_total:.4f}")
    lines.append(f"  Total via direct provider APIs: ${direct_total:.4f}")

    unpriced = openrouter.get("unpriced") or direct.get("unpriced") or []
    if unpriced:
        lines.append("")
        lines.append(
            "  No rates for: "
            + ", ".join(unpriced)
            + " — those judges are excluded from the totals."
        )

    lines.append("")
    lines.append(
        "  Estimated from a bundled rate table and a character-based token "
        "heuristic, so"
    )
    lines.append(
        "  the actual cost will differ: token counts are approximate and "
        "provider billing"
    )
    lines.append("  quirks (minimums, rounding, taxes, discounts) are not modeled.")
    return "\n".join(lines)


def _in_foreground() -> bool:
    try:
        return os.tcgetpgrp(sys.stdin.fileno()) == os.getpgrp()
    except (AttributeError, OSError, ValueError):
        return True


def confirm_judge_cost(
    both: dict,
    assume_yes: bool = False,
    stream: TextIO | None = None,
) -> bool:
    """Show the estimate and return whether the judge run should go ahead.

    The estimate and the question are written to ``stream`` (``sys.stderr`` by
    default, resolved per call). stderr is used because stdout is commonly
    redirected — ``calibrate-agent stt ... > run.log`` should still put the
    question on the screen the answer will be typed at, and stderr survives the
    stream tee the benchmarks install.

    Proceeds without asking when ``assume_yes`` is set, when the
    ``CALIBRATE_ASSUME_YES`` environment variable is non-empty, when stdin is
    not a terminal, or when the process is not in the terminal's foreground
    process group. That last case covers ``calibrate-agent stt ... &``: stdin is
    still the terminal, so it reads as one, but a background process group that
    reads it is stopped by the kernel with SIGTTIN — asking there would suspend
    the run instead of prompting.

    An answer of ``y`` or ``yes`` (any case) proceeds; anything else, including
    an empty line or an interrupted read, cancels.
    """
    out = sys.stderr if stream is None else stream
    out.write(format_cost_estimate(both) + "\n")
    out.flush()

    if assume_yes or os.getenv(_ENV_ASSUME_YES):
        return True
    if not sys.stdin.isatty() or not _in_foreground():
        return True

    out.write("\nProceed with the judge run? [y/N]: ")
    out.flush()
    try:
        answer = input()
    except (EOFError, KeyboardInterrupt):
        out.write("\n")
        out.flush()
        return False
    return answer.strip().lower() in ("y", "yes")


def confirm_estimated_judge_cost(
    plan: Callable[[], tuple[Sequence[JudgeCallGroup], int]],
    assume_yes: bool = False,
    stream: TextIO | None = None,
) -> bool:
    """Estimate the cost of the workload ``plan`` describes and confirm it.

    ``plan`` is called to produce ``(groups, cached_count)``, where
    ``cached_count`` is how many judge results a prior run already checkpointed.
    It is called inside a guard: an estimate is advisory, so a workload that
    cannot be sized — an unreadable dataset, a missing audio file, a column that
    is not there — proceeds unasked rather than stopping a run that would
    otherwise succeed. The paths that genuinely require that data validate it
    themselves and report it with their own message.

    Returns whether the judge run should go ahead. A workload with no judge
    calls in it needs no confirmation.
    """
    try:
        groups, cached_count = plan()
    except Exception:
        return True

    if not groups:
        return True

    if cached_count:
        print(
            f"Note: {cached_count} judge result(s) are already checkpointed from "
            "a prior run and will be reused. The estimate below covers the whole "
            "dataset, so it overstates what this run will spend."
        )

    return confirm_judge_cost(
        estimate_judge_cost_all_sources(groups),
        assume_yes=assume_yes,
        stream=stream,
    )
