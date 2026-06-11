"""
Unified LLM Judge module.

An *evaluator* is the unit of grading. It is a dict with the shape::

    {
        "id": str,                    # optional unique id
        "name": str,
        "system_prompt": str,         # may contain {{var}} placeholders
        "judge_model": str,           # OpenRouter model id
        "type": "binary" | "rating",  # default: "binary"
        "scale_min": int,             # only when type == "rating"
        "scale_max": int,             # only when type == "rating"
    }

Two judge entry points:

- ``text_judge``  — runs every evaluator in parallel against a text-only
  user prompt and returns ``{evaluator_name: result}``.
- ``audio_judge`` — same shape, but also attaches a base64 audio block
  to each call (used by the TTS pipeline).

Each evaluator becomes one independent LLM call: its ``system_prompt`` is
sent verbatim as the system message and the per-row context is sent as
the user message. If multiple evaluators are supplied they are issued
concurrently; results are stitched back into a single dict keyed by
``evaluator["name"]``. If an evaluator includes an optional ``id``, that value
is echoed as ``evaluator_id`` in the individual result payload.

Default evaluators are exposed for callers that want to preserve the
pre-evaluators-API behavior:

- ``DEFAULT_LLM_TEST_EVALUATOR`` (name: ``correctness``)
- ``DEFAULT_STT_EVALUATOR``      (name: ``semantic_match``)
- ``DEFAULT_TTS_EVALUATOR``      (name: ``pronunciation``)

Simulation has no implicit default — callers must supply evaluators.
"""

import asyncio
import base64
import json
import os
import re
from typing import Literal, Optional

import instructor
from pydantic import BaseModel, Field, create_model

from calibrate.langfuse import AsyncOpenAI, observe, langfuse, langfuse_enabled
from calibrate.utils import log_judge_io


# ── OpenRouter configuration ────────────────────────────────────────────────
# All judges (text + audio) route through OpenRouter so users only need to
# configure a single API key (OPENROUTER_API_KEY) and gain access to the full
# model catalog. OpenRouter is drop-in compatible with the OpenAI Chat
# Completions API.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build_openrouter_client() -> "AsyncOpenAI":
    """Return an AsyncOpenAI client pointed at OpenRouter."""
    return AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=_OPENROUTER_BASE_URL,
    )


# ── Default models (OpenRouter format: <provider>/<model>) ──────────────────
DEFAULT_TEXT_JUDGE_MODEL = "openai/gpt-5.4-mini"
# Simulation uses a stronger model by default for grading multi-turn conversations
DEFAULT_SIMULATION_JUDGE_MODEL = "openai/gpt-5.4-mini"
# OpenRouter's audio-capable OpenAI model. Override per-evaluator when needed.
DEFAULT_AUDIO_JUDGE_MODEL = "openai/gpt-audio"


# ── Evaluator type tags ─────────────────────────────────────────────────────

BINARY = "binary"
RATING = "rating"


# ── Default evaluators ──────────────────────────────────────────────────────
# These are auto-injected by the LLM-test / STT / TTS pipelines when the user
# does not declare any top-level ``evaluators`` list (or doesn't override the
# default evaluator's name). Simulation has no implicit default.

DEFAULT_LLM_TEST_EVALUATOR = {
    "name": "correctness",
    "system_prompt": (
        "You are a highly accurate evaluator evaluating the response of an agent to a "
        "user's message.\n\n"
        "You will be given a conversation between a user and an agent "
        "along with the response of the agent to the final user message.\n\n"
        "You need to evaluate if the response adheres to the evaluation "
        "criteria:\n\n{{criteria}}"
    ),
    "judge_model": DEFAULT_TEXT_JUDGE_MODEL,
    "type": BINARY,
}

DEFAULT_TOOL_CALL_PARAM_EVALUATOR = {
    "name": "tool_call_parameter",
    "system_prompt": (
        "You are a highly accurate evaluator checking whether the value an "
        "agent produced for a single tool-call argument satisfies a given "
        "criteria.\n\n"
        "You will be given the tool name, the argument name, and the actual "
        "value the agent produced for that argument.\n\n"
        "Mark `match` true only if the actual value satisfies the following "
        "criteria, and false otherwise:\n\n{{criteria}}"
    ),
    "judge_model": DEFAULT_TEXT_JUDGE_MODEL,
    "type": BINARY,
}

DEFAULT_GENERAL_TASK_EVALUATOR = {
    "name": "task_quality",
    "system_prompt": (
        "You are a highly accurate evaluator assessing the output produced for "
        "a task.\n\n"
        "You will be given the task input (when one is provided) and the output "
        "produced for it. Judge the output on its own merits — do not assume the "
        "input is a conversation or that the output is a reply to a user.\n\n"
        "Mark `match` true only if the output satisfies the following criteria, "
        "and false otherwise:\n\n{{criteria}}"
    ),
    "judge_model": DEFAULT_TEXT_JUDGE_MODEL,
    "type": BINARY,
}

DEFAULT_STT_EVALUATOR = {
    "name": "semantic_match",
    "system_prompt": (
        "You are a highly accurate evaluator evaluating the transcription "
        "output of an STT model.\n\n"
        "You will be given two strings - one is the source string used to "
        "produce an audio and the other is the transcription of that audio.\n\n"
        "You need to evaluate if the two strings are the same.\n\n"
        "# Important Instructions:\n"
        "- Check whether the values represented by both the strings match. "
        'E.g. if one string says 1,2,3 but the other string says "one, two, '
        'three" or "one, 2, three", they should be considered the same as '
        "their underlying value is the same. However, if the actual values "
        "itself are different, e.g. for the name of a person or address or "
        "the value of any other key detail - that difference should be noted.\n"
        "- Ignore differences like a word being split up into more than 1 "
        "word by spaces. Look at whether the values mean the same in both "
        "the strings.\n"
        "- Minor differences in values of entities (e.g. proper nouns, numbers) matter and should be considered an error.\n"
        '- If all the "values" for the strings match, mark it as True. Else, '
        "False."
    ),
    "judge_model": DEFAULT_TEXT_JUDGE_MODEL,
    "type": BINARY,
}

DEFAULT_TTS_EVALUATOR = {
    "name": "pronunciation",
    "system_prompt": (
        "You are a highly accurate evaluator evaluating the audio output of "
        "a TTS model.\n\n"
        "You will be given the audio and the text that should have been "
        "spoken in the audio.\n\n"
        "You need to evaluate if the text is easily understandable from the "
        "audio. Check whether the spoken words match the reference text and "
        "the audio is clear enough to convey the intended message."
    ),
    "judge_model": DEFAULT_AUDIO_JUDGE_MODEL,
    "type": BINARY,
}


# ── Type-introspection helpers ──────────────────────────────────────────────


def is_rating(evaluator: dict) -> bool:
    """Return True when the evaluator (or per-evaluator score dict) is rating-typed."""
    return evaluator.get("type") == RATING


def evaluator_result_value(evaluator: dict, result: dict) -> float:
    """Extract the numeric value from a single evaluator's judge result.

    - binary  → ``{"match": bool, "reasoning": str}``  → 0.0 / 1.0
    - rating  → ``{"score": int,  "reasoning": str}``  → score as float

    Used by downstream aggregation (CSV columns, metrics.json means).
    """
    if is_rating(evaluator):
        return float(result["score"])
    return float(int(bool(result["match"])))


def attach_evaluator_id(evaluator: dict, result: dict) -> dict:
    """Echo an optional unique id into a result payload."""
    if "id" not in evaluator or not isinstance(result, dict):
        return result
    result = dict(result)
    result["evaluator_id"] = evaluator["id"]
    return result


def evaluator_config_payload(evaluators: list[dict], extra: Optional[dict] = None) -> dict:
    """Build the evaluator config artifact written alongside run outputs."""
    payload = dict(extra or {})
    raw_evaluators = [dict(ev) for ev in evaluators or [] if isinstance(ev, dict)]
    payload["evaluators"] = raw_evaluators
    payload["evaluators_map"] = {
        str(ev["id"]): ev["name"]
        for ev in raw_evaluators
        if ev.get("id") is not None and ev.get("name")
    }
    return payload


def write_evaluator_config(
    output_dir: str, evaluators: list[dict], extra: Optional[dict] = None
) -> None:
    """Write raw evaluators and id-to-name mapping to config.json."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(evaluator_config_payload(evaluators, extra=extra), f, indent=4)


# Back-compat alias (older imports). Kept to avoid breaking SDK consumers.
criterion_result_value = evaluator_result_value


def format_evaluation_result_lines(eval_row: dict) -> list[str]:
    """Format a single evaluation_results row as one or two CLI/log lines.

    ``eval_row`` matches the per-evaluator dict written into
    ``evaluation_results.csv`` (and the simulation ``evaluation_results``
    list): ``{"name", "type", "value", "reasoning", evaluator_id?, scale_min?, scale_max?}``.

    Returns a header line plus an indented "Reason:" line when reasoning is
    present, so callers can ``log_and_print`` each line individually:

        [name] ✅ Pass        # binary, value == 1.0
        [name] ❌ Fail        # binary, value == 0.0
        [name] 4/5            # rating
          Reason: ...
    """
    name = eval_row.get("name", "evaluator")
    ev_type = eval_row.get("type", "binary")
    value = eval_row.get("value")

    if ev_type == "rating":
        scale_max = eval_row.get("scale_max")
        score_str = f"{int(value)}/{scale_max}" if scale_max is not None else str(value)
        header = f"[{name}] {score_str}"
    else:
        passed = bool(value)
        header = f"[{name}] {'✅ Pass' if passed else '❌ Fail'}"

    lines = [header]
    reasoning = eval_row.get("reasoning")
    if reasoning:
        lines.append(f"  Reason: {reasoning}")
    return lines


def _rating_range(evaluator: dict) -> list[int]:
    """Return the list of allowed score values for a rating evaluator."""
    lo = int(evaluator["scale_min"])
    hi = int(evaluator["scale_max"])
    if hi < lo:
        raise ValueError(
            f"Rating evaluator '{evaluator.get('name', '?')}' has scale_max "
            f"({hi}) less than scale_min ({lo})."
        )
    return list(range(lo, hi + 1))


# ── Tool / schema names (Azure OpenAI: ^[a-zA-Z0-9_.-]+$) ────────────────────


def _sanitize_evaluator_for_tool_model(name: str) -> str:
    """Return a suffix safe for Pydantic ``create_model`` ``__name__`` / Instructor tool names.

    Providers (e.g. Azure) reject tool names outside ``^[a-zA-Z0-9_.-]+$``.
    Human-readable evaluator titles often contain spaces or punctuation; map
    those to underscores and ensure a non-empty, identifier-like string.
    """
    raw = (name or "").strip() or "evaluator"
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw)
    s = re.sub(r"_+", "_", s).strip("_.-") or "evaluator"
    s = re.sub(r"[.-]", "_", s)
    s = re.sub(r"_+", "_", s)
    if s[0].isdigit():
        s = f"E_{s}"
    return s


def _normalize_judge_api_result(result: dict, model_cls_name: str) -> dict:
    """If structured output nests fields under the schema/model name, unwrap to a flat dict.

    Callers key the outer result dict by the original evaluator ``name``; this
    only normalizes the inner payload so downstream always sees
    ``reasoning`` / ``match`` or ``reasoning`` / ``score``.
    """
    if not isinstance(result, dict):
        return result
    inner = result.get(model_cls_name)
    if isinstance(inner, dict) and any(
        k in inner for k in ("reasoning", "match", "score")
    ):
        return dict(inner)
    return result


# ── Pydantic result models ──────────────────────────────────────────────────


class CriterionResult(BaseModel):
    """Binary result for a single evaluator."""

    reasoning: str = Field(
        ...,
        description="Step-by-step analysis of whether the criterion is met; be concise.",
    )
    match: bool = Field(
        ...,
        description="True if the criterion is met. False if it is not.",
    )


def _build_rating_result_model(evaluator: dict) -> type[BaseModel]:
    """Dynamically build a Pydantic model for a rating evaluator with a Literal-constrained score."""
    values = _rating_range(evaluator)
    # Literal[tuple(...)] expands to Literal[1, 2, 3, ...] — safe across Python 3.11+
    ScoreType = Literal[tuple(values)]  # type: ignore[valid-type]

    scale_min = evaluator["scale_min"]
    scale_max = evaluator["scale_max"]
    suffix = _sanitize_evaluator_for_tool_model(str(evaluator.get("name", "evaluator")))
    model_name = f"RatingResult_{suffix}"

    return create_model(
        model_name,
        reasoning=(
            str,
            Field(
                ...,
                description=(
                    "Step-by-step analysis of how the context measures against "
                    "the criterion; be concise."
                ),
            ),
        ),
        score=(
            ScoreType,
            Field(
                ...,
                description=(
                    f"Integer rating score from {scale_min} (lowest) to "
                    f"{scale_max} (highest) as defined in the system prompt."
                ),
            ),
        ),
    )


def _result_model_for_evaluator(evaluator: dict) -> type[BaseModel]:
    """Return the Pydantic result model for a single evaluator, based on its type."""
    if is_rating(evaluator):
        return _build_rating_result_model(evaluator)
    return CriterionResult


# ── Evaluator helpers ───────────────────────────────────────────────────────


def render_template(template: str, arguments: dict) -> str:
    """Substitute ``{{var}}`` placeholders in ``template`` with values from ``arguments``.

    Plain text replacement only — no escaping, conditionals, or loops. Missing
    placeholders are left intact (so a follow-up render or a self-aware
    judge LLM can still pick them up).
    """
    out = template
    for key, value in (arguments or {}).items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


def tool_call_param_evaluator(judge_model: Optional[str] = None) -> dict:
    """Return a copy of the default tool-call parameter evaluator.

    Used by the LLM tool-call test runner to judge a single tool-call argument
    value against a free-text criteria. ``judge_model``, when provided,
    overrides the evaluator's default ``judge_model``.
    """
    evaluator = dict(DEFAULT_TOOL_CALL_PARAM_EVALUATOR)
    if judge_model:
        evaluator["judge_model"] = judge_model
    return evaluator


def render_evaluator(evaluator: dict, arguments: Optional[dict] = None) -> dict:
    """Return a copy of ``evaluator`` with its ``system_prompt`` placeholders filled in."""
    rendered = dict(evaluator)
    rendered["system_prompt"] = render_template(
        evaluator.get("system_prompt", ""), arguments or {}
    )
    return rendered


def _model_for(evaluator: dict, fallback: str) -> str:
    """Return the model id for ``evaluator``, falling back when none is set."""
    return evaluator.get("judge_model") or fallback


# ── Single-evaluator judge calls ────────────────────────────────────────────


@observe(name="evaluator_call", capture_input=False)
async def _judge_one_text(
    evaluator: dict,
    user_prompt: str,
    fallback_model: str,
) -> dict:
    """Issue one chat-completion call for a single text evaluator."""
    client = instructor.apatch(_build_openrouter_client())
    Output = _result_model_for_evaluator(evaluator)
    model = _model_for(evaluator, fallback_model)
    system_prompt = evaluator.get("system_prompt", "")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=Output,
        temperature=0,
        max_completion_tokens=8192,
    )

    result = _normalize_judge_api_result(response.model_dump(), Output.__name__)

    log_judge_io(
        evaluator=evaluator.get("name", ""),
        model=model,
        system_prompt=system_prompt,
        user_input=user_prompt,
        output=result,
    )

    if langfuse_enabled and langfuse:
        langfuse.update_current_span(
            metadata={
                "evaluator": evaluator.get("name"),
                "model": model,
                "system_prompt": system_prompt,
                "input": user_prompt,
                "output": result,
                "output_schema": Output.model_json_schema(),
            }
        )

    return result


@observe(name="evaluator_call_audio", capture_input=False, capture_output=False)
async def _judge_one_audio(
    evaluator: dict,
    reference_text: str,
    audio_path: str,
    audio_b64: str,
    fallback_model: str,
) -> dict:
    """Issue one chat-completion call for a single audio evaluator."""
    client = instructor.apatch(_build_openrouter_client())
    Output = _result_model_for_evaluator(evaluator)
    model = _model_for(evaluator, fallback_model)
    system_prompt = evaluator.get("system_prompt", "")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Reference text: {reference_text}\n\nAudio:",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    },
                ],
            },
        ],
        response_model=Output,
        temperature=0,
        max_completion_tokens=8192,
    )

    result = _normalize_judge_api_result(response.model_dump(), Output.__name__)

    log_judge_io(
        evaluator=evaluator.get("name", ""),
        model=model,
        system_prompt=system_prompt,
        user_input=f"Reference text: {reference_text}\n[audio omitted from log]",
        output=result,
    )

    if langfuse_enabled and langfuse:
        from calibrate.langfuse import create_langfuse_audio_media

        audio_media = create_langfuse_audio_media(audio_path)
        langfuse.update_current_trace(
            input={"audio": audio_media, "reference_text": reference_text},
            output=result,
            metadata={
                "evaluator": evaluator.get("name"),
                "model": model,
                "system_prompt": system_prompt,
                "input": f"Reference text: {reference_text}",
                "output": result,
            },
        )

    return result


# ── Public judge entry points ───────────────────────────────────────────────


@observe(name="text_judge", capture_input=False)
async def text_judge(
    evaluators: list[dict],
    user_prompt: str,
    fallback_model: str = DEFAULT_TEXT_JUDGE_MODEL,
) -> dict:
    """Run every evaluator in parallel against ``user_prompt``.

    Each evaluator becomes one chat-completion call (system message =
    ``evaluator["system_prompt"]``, user message = ``user_prompt``). All
    calls are issued concurrently via ``asyncio.gather``.

    Args:
        evaluators: List of evaluator dicts. ``system_prompt`` should already
            be rendered (placeholders substituted) by the caller.
        user_prompt: The per-row context (transcription pair, conversation,
            etc.) shared across every evaluator call.
        fallback_model: Model id used when an evaluator has no ``judge_model``.

    Returns:
        Dict keyed by ``evaluator["name"]``. Binary evaluators give
        ``{"reasoning": str, "match": bool}``; rating evaluators give
        ``{"reasoning": str, "score": int}``. If the evaluator has ``id``,
        the payload also includes ``evaluator_id``.
    """
    if not evaluators:
        return {}

    coros = [_judge_one_text(ev, user_prompt, fallback_model) for ev in evaluators]
    results = await asyncio.gather(*coros)
    return {
        ev["name"]: attach_evaluator_id(ev, r)
        for ev, r in zip(evaluators, results)
    }


@observe(name="audio_judge", capture_input=False, capture_output=False)
async def audio_judge(
    evaluators: list[dict],
    audio_path: str,
    reference_text: str,
    fallback_model: str = DEFAULT_AUDIO_JUDGE_MODEL,
) -> dict:
    """Run every evaluator in parallel against an audio + reference-text pair.

    Each evaluator gets one chat-completion call carrying the same audio
    as a base64 ``input_audio`` block (encoded once and reused).

    Args:
        evaluators: List of evaluator dicts (already rendered).
        audio_path: Path to the WAV audio file to evaluate.
        reference_text: The text that should have been spoken.
        fallback_model: Model id used when an evaluator has no ``judge_model``;
            should be an audio-capable model.

    Returns:
        Dict keyed by ``evaluator["name"]``. If the evaluator has ``id``, the
        payload also includes ``evaluator_id``.
    """
    if not evaluators:
        return {}

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    coros = [
        _judge_one_audio(ev, reference_text, audio_path, audio_b64, fallback_model)
        for ev in evaluators
    ]
    results = await asyncio.gather(*coros)
    return {
        ev["name"]: attach_evaluator_id(ev, r)
        for ev, r in zip(evaluators, results)
    }


# ── Conversation-transcript helper ──────────────────────────────────────────


def format_conversation(conversation: list[dict]) -> str:
    """Format a chat history list into a flat ``role: content`` transcript.

    Tool calls are inlined as ``[Tool Call] name(args)`` lines so the judge
    LLM can see them alongside textual messages. Used by the simulation and
    LLM-test runners to build the user prompt that gets passed to text_judge.
    """
    lines: list[str] = []
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if content:
            lines.append(f"{role}: {content}")

        for tc in tool_calls or []:
            func = tc.get("function", {})
            func_name = func.get("name", "unknown")
            func_args = func.get("arguments", "{}")
            lines.append(f"[Tool Call] {func_name}({func_args})")

    return "\n".join(lines)


def require_unique_evaluator_names(evaluators: object) -> None:
    """Raise ``ValueError`` if ``evaluators`` contains duplicate ``name`` values.

    Downstream code (judge runners, metrics, leaderboard, UI) keys results by
    ``evaluator["name"]``; duplicate names would silently collapse and only
    the last entry's results would survive. Reject early with a clear error.

    Accepts a non-list / empty / ``None`` input as a no-op so callers can
    invoke this without first checking shape.
    """
    if not isinstance(evaluators, list):
        return
    seen: set[str] = set()
    duplicates: list[str] = []
    for ev in evaluators:
        if not isinstance(ev, dict):
            continue
        name = ev.get("name")
        if not isinstance(name, str):
            continue
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            "Duplicate evaluator name(s) found: "
            + ", ".join(repr(n) for n in duplicates)
            + ". Each evaluator must have a unique `name`."
        )


def require_simulation_evaluators(evaluators: object) -> None:
    """Raise ``ValueError`` unless ``evaluators`` is a non-empty list with
    unique names.

    Text and voice simulations do not inject implicit judges; config and SDK
    callers must provide at least one evaluator.
    """

    if not isinstance(evaluators, list) or len(evaluators) == 0:
        raise ValueError(
            "Simulation config must define a non-empty top-level `evaluators` "
            "(simulations have no implicit default)."
        )
    require_unique_evaluator_names(evaluators)


# ── General task judge (thin wrapper over text_judge) ───────────────────────


def format_task_io(output: str, input_text: Optional[str] = None) -> str:
    """Build a neutral ``input``/``output`` user prompt for a general task.

    Unlike :func:`format_conversation` this does not assume a chat transcript:
    the text is framed as a task ``Input`` and ``Output`` pair. ``input_text``
    is optional — when it is ``None`` or blank only the ``Output`` section is
    emitted, so a caller can judge an output on its own (e.g. "is this valid
    JSON", "is this summary faithful") without inventing a fake input.
    """
    sections: list[str] = []
    if input_text is not None and str(input_text).strip():
        sections.append(f"`Input`:\n\n{input_text}")
    sections.append(f"`Output`:\n\n{output if output is not None else ''}")
    return "\n\n".join(sections)


@observe(name="general_task_judge", capture_input=False)
async def general_task_judge(
    evaluators: list[dict],
    output: str,
    input_text: Optional[str] = None,
    fallback_model: str = DEFAULT_TEXT_JUDGE_MODEL,
) -> dict:
    """Evaluate a task output (with optional input) against a list of evaluators.

    A general-purpose, non-conversational counterpart to
    :func:`simulation_judge`. The ``output`` (and ``input_text`` when given) are
    framed as a plain task ``Input``/``Output`` pair rather than a chat
    transcript, so this fits arbitrary single-shot LLM tasks — summarization,
    extraction, classification, rewriting, code generation, etc.

    There is no implicit default evaluator. If ``evaluators`` is empty the
    function returns ``{}`` and no judge calls are made.

    Args:
        evaluators: List of evaluator dicts (already rendered). Use
            :data:`DEFAULT_GENERAL_TASK_EVALUATOR` for a generic criteria-driven
            default.
        output: The text output to evaluate.
        input_text: Optional task input the output was produced for. Omitted from
            the prompt when ``None`` or blank.
        fallback_model: Model id used when an evaluator has no ``judge_model``.

    Returns:
        Dict keyed by ``evaluator["name"]`` — same shape as :func:`text_judge`.
    """
    if not evaluators:
        return {}

    user_prompt = format_task_io(output, input_text)
    return await text_judge(
        evaluators=evaluators,
        user_prompt=user_prompt,
        fallback_model=fallback_model,
    )


# ── Simulation judge (thin wrapper over text_judge) ─────────────────────────


@observe(name="simulation_judge")
async def simulation_judge(
    conversation: list[dict],
    evaluators: list[dict],
    fallback_model: str = DEFAULT_SIMULATION_JUDGE_MODEL,
) -> dict:
    """Evaluate a conversation transcript against a list of evaluators.

    Simulation has no implicit default evaluator. If ``evaluators`` is empty
    the function returns ``{}`` and no judge calls are made.

    Args:
        conversation: List of message dicts (role/content, may include tool_calls).
        evaluators: List of evaluator dicts (already rendered).
        fallback_model: Model id used when an evaluator has no ``judge_model``.

    Returns:
        Dict keyed by ``evaluator["name"]``.
    """
    if not evaluators:
        return {}

    user_prompt = f"`Chat history`:\n\n{format_conversation(conversation)}"
    return await text_judge(
        evaluators=evaluators,
        user_prompt=user_prompt,
        fallback_model=fallback_model,
    )
