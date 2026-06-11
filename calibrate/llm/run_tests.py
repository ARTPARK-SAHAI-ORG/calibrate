import asyncio
import argparse
import re
import sys
import time
import uuid
from collections import defaultdict
from typing import Any, Awaitable, Callable, List, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from calibrate.connections import TextAgentConnection  # noqa: F401
import os
from os.path import join, exists
import json
from pathlib import Path
from calibrate.utils import (
    configure_print_logger,
    log_and_print,
    build_tools_schema,
    provider_log_file,
)
from pipecat.frames.frames import (
    TranscriptionFrame,
    LLMRunFrame,
    EndFrame,
    EndTaskFrame,
    LLMFullResponseEndFrame,
    CancelFrame,
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    TextFrame,
    FunctionCallInProgressFrame,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from calibrate.llm.metrics import test_response_llm_judge, evaluate_simuation
from calibrate.llm._metrics_utils import _numeric_or_none
from calibrate.judges import (
    DEFAULT_LLM_TEST_EVALUATOR,
    attach_evaluator_id,
    is_rating,
    render_evaluator,
    require_unique_evaluator_names,
    text_judge,
    tool_call_param_evaluator,
    write_evaluator_config,
)

from calibrate.langfuse import observe, langfuse, langfuse_enabled


# ── Evaluator resolution helpers (LLM-test-specific) ────────────────────────


def _normalize_criteria_refs(criteria) -> list[dict]:
    """Coerce a test case's ``evaluation.criteria`` into a list of evaluator refs.

    Accepted shapes:
    - ``str`` → ``[{"name": <DEFAULT_LLM_TEST_EVALUATOR.name>, "arguments": {"criteria": <str>}}]``
    - ``list[{name, arguments}]`` → returned as-is

    Any other shape raises ``ValueError``.
    """
    if isinstance(criteria, str):
        return [
            {
                "name": DEFAULT_LLM_TEST_EVALUATOR["name"],
                "arguments": {"criteria": criteria},
            }
        ]
    if isinstance(criteria, list):
        for entry in criteria:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError(
                    "evaluation.criteria entries must be dicts with a 'name' key "
                    "(and optional 'arguments'); got: " + repr(entry)
                )
        return criteria
    raise ValueError(
        f"evaluation.criteria must be a string or list of evaluator refs; got {type(criteria).__name__}"
    )


def _get_name_to_evaluator_dict(config: dict, include_default: bool = True) -> dict:
    """Return ``{evaluator_name: evaluator_dict}`` from top-level ``config.evaluators``.

    When ``include_default`` is True (``response`` tests), the implicit default
    LLM-test evaluator is included under its canonical name
    (``DEFAULT_LLM_TEST_EVALUATOR["name"]``) and the legacy alias ``"default"``
    so older configs that reference ``{"name": "default"}`` keep working;
    config evaluators with the same name override either entry.

    When ``include_default`` is False (``conversation`` tests, which have no
    implicit default), only evaluators listed in the config are included.
    """
    user_defined_evaluators = config.get("evaluators") or []
    require_unique_evaluator_names(user_defined_evaluators)
    name_to_evaluator: dict = {}
    if include_default:
        # ``"default"`` is a legacy alias for the canonical default evaluator
        # (``DEFAULT_LLM_TEST_EVALUATOR["name"]``). Defining both in the same
        # config would map two names to the same logical evaluator — reject early.
        _default_name = DEFAULT_LLM_TEST_EVALUATOR["name"]
        _user_defined_evaluator_names = {
            ev.get("name") for ev in user_defined_evaluators if isinstance(ev, dict)
        }
        if (
            "default" in _user_defined_evaluator_names
            and _default_name in _user_defined_evaluator_names
        ):
            raise ValueError(
                f"config.evaluators defines both 'default' and '{_default_name}', "
                f"which are aliases for the same default LLM-test evaluator. "
                f"Define only one."
            )
        name_to_evaluator[_default_name] = DEFAULT_LLM_TEST_EVALUATOR
        name_to_evaluator["default"] = DEFAULT_LLM_TEST_EVALUATOR
    for ev in user_defined_evaluators:
        if "name" not in ev or "system_prompt" not in ev:
            raise ValueError(
                "Each evaluator in config.evaluators must include 'name' and "
                "'system_prompt' (got: " + repr(ev) + ")"
            )
        name_to_evaluator[ev["name"]] = ev
    return name_to_evaluator


def _evaluators_for_config_output(config: dict) -> list[dict]:
    """Return raw evaluators for the output config artifact."""
    user_defined_evaluators = config.get("evaluators") or []
    return (
        list(user_defined_evaluators)
        if user_defined_evaluators
        else [DEFAULT_LLM_TEST_EVALUATOR]
    )


def _resolve_evaluators_for_test_case(
    evaluation: dict, name_to_evaluator: dict
) -> list[dict]:
    """Resolve a test case's ``evaluation.criteria`` to rendered evaluator dicts."""
    refs = _normalize_criteria_refs(evaluation.get("criteria"))
    rendered: list[dict] = []
    for ref in refs:
        name = ref["name"]
        if name not in name_to_evaluator:
            raise ValueError(
                f"Unknown evaluator '{name}' referenced in test case. Define "
                f"it under config.evaluators (or use "
                f"'{DEFAULT_LLM_TEST_EVALUATOR['name']}')."
            )
        rendered.append(render_evaluator(name_to_evaluator[name], ref.get("arguments")))
    return rendered


class Processor(FrameProcessor):
    """Processor that captures LLM text output."""

    def __init__(
        self,
        chat_history: List[dict[str, str]],
    ):
        super().__init__(enable_direct_mode=True, name="Processor")
        self._current_response = ""
        self._collecting_response = False
        self._tool_calls = []
        self._ready = False
        self._chat_history = chat_history

    def set_task(self, task: "PipelineTask"):
        """Set the task reference after task creation."""
        self._task = task

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        logger.info(f"text output processor frame: {frame}")

        if not self._ready:
            self._ready = True
            if self._task:
                await self._task.queue_frames(
                    [
                        LLMMessagesAppendFrame(self._chat_history, run_llm=True),
                    ]
                )

        # Capture text frames from LLM
        if isinstance(frame, TextFrame):
            text = frame.text
            if text:
                self._collecting_response = True
                self._current_response += text
                logger.info(f"Received text chunk: {text}")

        if isinstance(frame, FunctionCallInProgressFrame):
            log_and_print(f"Function call in progress: {frame.function_name}")
            log_and_print(f"Arguments: {frame.arguments}")
            self._tool_calls.append(
                {
                    "tool": frame.function_name,
                    "arguments": frame.arguments,
                }
            )

        # When we get an EndFrame after collecting text, save the complete response
        if isinstance(frame, LLMFullResponseEndFrame):
            if self._task:
                await self._task.queue_frames([EndFrame()])

        await self.push_frame(frame, direction)


class CostTrackingOpenRouterLLMService(OpenRouterLLMService):
    """OpenRouter LLM service that records the authoritative per-request USD cost.

    OpenRouter returns the actual charged cost inline in the final usage chunk of
    every streaming response (``usage.cost`` — already net of cache discounts,
    fallback routing, and BYOK upstream pricing). The base pipecat service reads
    only token counts from that chunk and discards the cost, so we wrap the
    completion stream to capture it without reimplementing the chunk-processing
    loop. This keeps cost tracking robust to new models — the price comes from
    OpenRouter itself, not a local price table that can go stale.

    After a run, ``last_cost`` holds the most recent request's cost (USD) or
    ``None`` if OpenRouter did not report one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_cost: Optional[float] = None

    async def get_chat_completions(self, params_from_context):
        stream = await super().get_chat_completions(params_from_context)
        return self._capture_cost(stream)

    async def _capture_cost(self, stream):
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                cost = getattr(usage, "cost", None)
                if cost is None:
                    extra = getattr(usage, "model_extra", None) or {}
                    cost = extra.get("cost")
                if cost is not None:
                    try:
                        self.last_cost = float(cost)
                    except (TypeError, ValueError):
                        pass
            yield chunk


class LLMInferenceError(Exception):
    """Raised when LLM inference fails due to system error (API error, invalid model, etc.)"""

    pass


# Lock to protect logger add/remove operations from race conditions when running in parallel
import threading

_logger_lock = threading.Lock()

# Default number of test cases to evaluate concurrently within a single model.
DEFAULT_TEST_PARALLEL = 4


def _resolve_test_parallel(cli_value: Optional[int] = None) -> int:
    """Resolve max test-case concurrency: CLI flag > CALIBRATE_TEST_PARALLEL > default."""
    if cli_value is not None and cli_value > 0:
        return cli_value
    env_value = os.getenv("CALIBRATE_TEST_PARALLEL")
    if env_value:
        try:
            parsed = int(env_value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return DEFAULT_TEST_PARALLEL


async def _run_items_parallel(
    items: List[Any],
    process: "Callable[[int, Any], Awaitable[dict]]",
    results_file_path: str,
    test_parallel: Optional[int] = None,
) -> List[dict]:
    """Run ``process(index, item)`` over ``items`` with bounded concurrency.

    Concurrency is capped by ``_resolve_test_parallel(test_parallel)``. Results
    are collected in input order regardless of completion order, and the
    partial (in-order) ``results.json`` is rewritten after each item completes.
    Returns the ordered list of results.
    """
    results: List[Optional[dict]] = [None] * len(items)
    semaphore = asyncio.Semaphore(_resolve_test_parallel(test_parallel))
    write_lock = asyncio.Lock()

    async def run_one(index: int, item: Any) -> None:
        async with semaphore:
            results[index] = await process(index, item)
            # Save intermediate results as each item completes (in order).
            async with write_lock:
                with open(results_file_path, "w") as f:
                    json.dump([r for r in results if r is not None], f, indent=4)

    await asyncio.gather(*[run_one(i, item) for i, item in enumerate(items)])
    return results


# Strip ANSI color codes when mirroring terminal output to results.log
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def display_label(provider: str, model: str) -> str:
    """Format a ``provider/model`` display label, dropping the redundant
    ``openrouter/`` prefix when the provider is OpenRouter.

    OpenRouter model ids are themselves namespaced ``<actual_provider>/<model>``
    (e.g. ``openai/gpt-4.1``), so prefixing them with ``openrouter/`` produces
    awkward chains like ``openrouter/openai/gpt-4.1``. For any other provider
    (e.g. ``openai``) the label is the standard ``provider/model``.
    """
    if provider == "openrouter":
        return model
    return f"{provider}/{model}"


def _print_and_log(text: str, log_path: Optional[str]) -> None:
    """Print ``text`` to stdout and append the ANSI-stripped form to ``log_path``.

    Used by :func:`run_model_tests` so the per-model ``results.log`` mirrors
    everything that's shown live in the terminal (model header, per-test-case
    pass/fail lines with provider/model prefix, and the final summary banner)
    instead of being a separately-formatted, stripped-down recap.

    Models run in parallel but each writes to its own ``results.log`` path,
    so plain append-mode-per-call is safe; tests within a model run
    sequentially so writes within a file are ordered.
    """
    print(text, flush=True)
    if log_path:
        with open(log_path, "a") as f:
            f.write(_ANSI_RE.sub("", text) + "\n")


async def run_inference(
    chat_history: List[dict[str, str]],
    system_prompt: str,
    model: str,
    provider: str,
    tools: List[dict[str, str]],
) -> dict:
    """Runs a text-only bot that processes text inputs through an LLM and returns text outputs.

    Returns dict with 'response', 'tool_calls', and 'captured_errors' keys.
    """
    # Capture ERROR-level logs to surface pipecat internal errors
    captured_errors: list[str] = []

    def error_capture_sink(message):
        record = message.record
        if record["level"].name in ("ERROR", "CRITICAL"):
            captured_errors.append(record["message"])

    # Use lock to protect logger operations from race conditions in parallel execution
    with _logger_lock:
        error_sink_id = logger.add(error_capture_sink, level="ERROR")

    try:
        result = await _run_inference_inner(
            chat_history=chat_history,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            tools=tools,
        )
        result["captured_errors"] = captured_errors
        return result
    finally:
        with _logger_lock:
            try:
                logger.remove(error_sink_id)
            except ValueError:
                # Handler may have already been removed in race condition
                pass


async def _run_inference_inner(
    chat_history: List[dict[str, str]],
    system_prompt: str,
    model: str,
    provider: str,
    tools: List[dict[str, str]],
) -> dict:
    """Inner implementation of run_inference."""
    # Create LLM service
    if provider == "openrouter":
        llm = CostTrackingOpenRouterLLMService(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=model,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
        )

    # Create context with system prompt
    messages = [{"role": "system", "content": system_prompt}]

    end_call_tool = FunctionSchema(
        name="end_call",
        description="End the current call when the conversation is complete.",
        properties={
            "reason": {
                "type": "string",
                "description": "Optional explanation for why the call should end.",
            }
        },
        required=[],
    )

    # Build tool schemas using common utility
    tool_schemas, webhook_tool_configs = build_tools_schema(tools)
    standard_tools = [end_call_tool] + tool_schemas

    tools_schema = ToolsSchema(standard_tools=standard_tools)

    async def generic_tool_call(params: FunctionCallParams):
        logger.info(f"tool call: {params}")
        await params.result_callback(
            None, properties=FunctionCallResultProperties(run_llm=False)
        )
        return

    def create_webhook_tool_call(webhook_config: dict):
        async def webhook_tool_call(params: FunctionCallParams):
            logger.info(
                f"webhook tool call: {params.function_name}\n"
                f"  method: {webhook_config['method']}\n"
                f"  url: {webhook_config['url']}\n"
                f"  headers: {webhook_config['headers']}\n"
                f"  query: {params.arguments.get('query', {})}\n"
                f"  body: {params.arguments.get('body', {})}"
            )
            await params.result_callback(
                None, properties=FunctionCallResultProperties(run_llm=False)
            )
            return

        return webhook_tool_call

    # Register appropriate handler for each tool
    for tool_schema in standard_tools:
        if tool_schema.name in webhook_tool_configs:
            llm.register_function(
                tool_schema.name,
                create_webhook_tool_call(webhook_tool_configs[tool_schema.name]),
            )
        else:
            llm.register_function(tool_schema.name, generic_tool_call)

    context = LLMContext(messages, tools=tools_schema)
    context_aggregator = LLMContextAggregatorPair(context)

    # Create processors (text_output needs to be created first for reference)
    processor = Processor(chat_history)
    # text_input = TextInputProcessor(text_inputs, text_output)

    # Build pipeline with all processors
    pipeline = Pipeline(
        [
            # text_input,
            context_aggregator.user(),
            llm,
            processor,
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[LLMLogObserver()],
        # idle_timeout_secs=5,
    )

    # Set task reference for text_input processor
    processor.set_task(task)

    runner = PipelineRunner(handle_sigint=False)

    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

    return {
        "response": processor._current_response,
        "tool_calls": processor._tool_calls,
        "cost": getattr(llm, "last_cost", None),
    }


def sort_tool_calls(tool_calls):
    return sorted(tool_calls, key=lambda val: val["tool"])


def get_webhook_tool_names(tools: List[dict]) -> set:
    """
    Extract names of webhook tools from the tools configuration.

    Args:
        tools: List of tool definition dicts

    Returns:
        Set of webhook tool names
    """
    return {tool["name"] for tool in tools if tool.get("type") == "webhook"}


def preprocess_conversation_history(
    chat_history: List[dict], tools: List[dict]
) -> List[dict]:
    """
    Preprocess conversation history to add tool responses for non-webhook tools.

    For non-webhook tools that have tool calls but no corresponding tool response,
    this function inserts a default tool response with {"status": "received"}.

    If a non-webhook tool already has a tool response in the conversation, it is
    left in place and no injection occurs for that tool call.

    Args:
        chat_history: The conversation history to preprocess
        tools: List of tool definition dicts

    Returns:
        Preprocessed conversation history with tool responses inserted
    """
    webhook_tool_names = get_webhook_tool_names(tools)

    # Build a set of existing tool response IDs
    existing_tool_response_ids = set()
    for message in chat_history:
        if message.get("role") == "tool" and "tool_call_id" in message:
            existing_tool_response_ids.add(message["tool_call_id"])

    # Process conversation history and collect tool calls that need responses
    processed_history = []
    for message in chat_history:
        processed_history.append(message)

        # Check for assistant messages with tool calls
        if message.get("role") == "assistant" and "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                tool_call_id = tool_call.get("id")
                tool_name = tool_call.get("function", {}).get("name")

                # Skip webhook tools - they handle their own responses
                if tool_name in webhook_tool_names:
                    continue

                if tool_call_id in existing_tool_response_ids:
                    continue

                # Insert a default tool response for non-webhook tools
                tool_response = {
                    "role": "tool",
                    "content": '{"status": "received"}',
                    "tool_call_id": tool_call_id,
                }
                processed_history.append(tool_response)
                logger.info(
                    f"Inserted tool response for structured output tool '{tool_name}' "
                    f"(tool_call_id: {tool_call_id})"
                )

    return processed_history


def _sorted_union_dict_keys(left: dict, right: dict) -> List[Any]:
    """Keys from both sides, ordered so mixed types (illegal in JSON) cannot break ``sorted``."""
    return sorted(set(left) | set(right), key=lambda k: (type(k).__name__, repr(k)))


def _value_mismatch_detail(expected, actual) -> str:
    """The reason (no ``path`` prefix) two leaf values differ.

    Returned standalone so it can be both stored as a record's ``reasoning`` and
    composed into a display line, without re-parsing the rendered line.
    """
    exp_t = type(expected).__name__
    act_t = type(actual).__name__
    if type(expected) is not type(actual):
        same_as_str = str(expected) == str(actual)
        note = " (same string form)" if same_as_str else ""
        return (
            f"type mismatch — expected {exp_t} {expected!r}, "
            f"got {act_t} {actual!r}{note}"
        )
    return f"value mismatch — expected {expected!r}, got {actual!r}"


def _tool_call_argument_value_mismatch_line(key_path: str, expected, actual) -> str:
    """One human-readable line explaining why ``expected`` and ``actual`` differ."""
    return f"  {key_path}: {_value_mismatch_detail(expected, actual)}"


def _tool_call_arguments_diff_lines(
    expected: dict, actual: dict, prefix: str = ""
) -> List[str]:
    """List per-field mismatch lines for two argument dicts (recursive).

    Criteria-agnostic: every value is compared literally. This is the
    exact-match view used by the aggregation fallback and by ``exact`` specs
    (whose value must be matched verbatim). It is a thin wrapper over
    :func:`_collect_arg_diffs` with criteria interpretation disabled.
    """
    lines: List[str] = []
    _collect_arg_diffs(expected, actual, prefix, lines, [], criteria_aware=False)
    return lines


def _tool_call_arguments_mismatch_message(expected, actual) -> str:
    """Build a multi-line tool-call arguments mismatch reason."""
    header = "Tool call arguments mismatch:"
    if not isinstance(expected, dict):
        return (
            f"{header}\n"
            f"  arguments: cannot diff — expected non-dict {type(expected).__name__} "
            f"{expected!r}, got {type(actual).__name__} {actual!r}"
        )
    if not isinstance(actual, dict):
        return (
            f"{header}\n"
            f"  arguments: expected dict {expected!r}, "
            f"got {type(actual).__name__} {actual!r}"
        )
    detail_lines = _tool_call_arguments_diff_lines(expected, actual)
    if not detail_lines:
        return (
            f"{header}\n"
            f"  expected arguments: {expected!r}\n"
            f"  actual arguments: {actual!r}"
        )
    return header + "\n" + "\n".join(detail_lines)


def _tool_call_pair_mismatch(
    output_tool_call: dict, evaluation_tool_call: dict
) -> Optional[str]:
    """Return a failure reason string if the pair does not match, else ``None``."""
    if output_tool_call["tool"] != evaluation_tool_call["tool"]:
        return (
            f"Tool call mismatch - expected tool call: {evaluation_tool_call['tool']} "
            f"but got: {output_tool_call['tool']}"
        )
    if "arguments" not in evaluation_tool_call:
        return None
    exp_args = evaluation_tool_call.get("arguments")
    if exp_args is None:
        return None
    out_args = output_tool_call.get("arguments")
    if out_args == exp_args:
        return None
    return _tool_call_arguments_mismatch_message(exp_args, out_args)


# ── Per-parameter criteria specs (exact match vs. LLM judge) ─────────────────


def _param_criteria_spec(value, key: str) -> Optional[dict]:
    """Interpret an expected-argument value as a per-parameter matching spec.

    A spec is a dict carrying a ``match_type`` key, letting a test case opt a
    single tool-call parameter out of exact matching::

        {"match_type": "llm_judge", "criteria": "...", "judge_model": "..."}
        {"match_type": "exact", "value": <literal>}

    Returns the normalized spec, or ``None`` when ``value`` is an ordinary
    literal (the default, exact-match behavior). Raises ``ValueError`` for a
    malformed spec.
    """
    if not isinstance(value, dict) or "match_type" not in value:
        return None
    match_type = value["match_type"]
    if match_type == "llm_judge":
        criteria = value.get("criteria")
        if not isinstance(criteria, str) or not criteria.strip():
            raise ValueError(
                f"Tool-call parameter '{key}': an 'llm_judge' match_type "
                f"requires a non-empty 'criteria' string."
            )
        spec = {"match_type": "llm_judge", "criteria": criteria}
        if value.get("judge_model"):
            spec["judge_model"] = value["judge_model"]
        return spec
    if match_type == "exact":
        if "value" not in value:
            raise ValueError(
                f"Tool-call parameter '{key}': an 'exact' match_type requires "
                f"a 'value' field holding the literal to match."
            )
        return {"match_type": "exact", "value": value["value"]}
    raise ValueError(
        f"Tool-call parameter '{key}': match_type must be 'exact' or "
        f"'llm_judge' (got {match_type!r})."
    )


async def _judge_tool_call_parameter(
    tool_name: str, param_name: str, spec: dict, actual_value
) -> dict:
    """Judge a single tool-call argument value against a criteria via an LLM."""
    evaluator = render_evaluator(
        tool_call_param_evaluator(spec.get("judge_model")),
        {"criteria": spec["criteria"]},
    )
    try:
        value_repr = json.dumps(actual_value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        value_repr = repr(actual_value)
    user_prompt = (
        f"Tool name: {tool_name}\n"
        f"Argument name: {param_name}\n"
        f"Actual value produced by the agent: {value_repr}"
    )
    results = await text_judge([evaluator], user_prompt)
    return results.get(
        evaluator["name"], {"match": False, "reasoning": "Judge returned no result"}
    )


def _record_failure(
    path: str,
    detail: str,
    lines: List[str],
    records: Optional[List[dict]],
    *,
    match_type: str = "exact",
    criteria: Optional[str] = None,
    missing: bool = False,
) -> None:
    """Append a failing parameter's display line and (when collecting) its record.

    The shared shape behind the non-judged failure branches of
    :func:`_collect_arg_diffs`: each renders ``  path: detail`` and, when
    ``records`` is being collected, stores a ``match=False`` record carrying the
    same ``detail`` as its reasoning (plus optional ``criteria`` / ``missing``).
    """
    lines.append(f"  {path}: {detail}")
    if records is None:
        return
    record: dict = {
        "param": path,
        "match_type": match_type,
        "match": False,
        "reasoning": detail,
    }
    if criteria is not None:
        record["criteria"] = criteria
    if missing:
        record["missing"] = True
    records.append(record)


def _collect_arg_diffs(
    expected: dict,
    actual: dict,
    prefix: str,
    lines: List[str],
    judge_jobs: List[tuple],
    *,
    criteria_aware: bool = True,
    records: Optional[List[dict]] = None,
) -> None:
    """Recursively diff expected vs. actual argument dicts.

    The single walk behind both exact and criteria-aware matching:

    - ``criteria_aware=True`` (default): a value may be a criteria spec — an
      ``llm_judge`` field is queued in ``judge_jobs`` (judged after the walk; a
      missing value adds a line), an ``exact`` field is unwrapped to its literal
      value, and a plain nested object is recursed into so its sub-parameters
      can themselves be specs.
    - ``criteria_aware=False``: every value is compared literally; specs are not
      interpreted. Used by the aggregation fallback and by ``exact`` values
      (matched verbatim — specs inside them are not re-interpreted).

    Mismatch lines (with dotted ``path`` prefixes) are appended to ``lines``.
    Synchronous: judging runs afterwards so all calls can be issued at once.

    When ``records`` is provided (criteria-aware path only) a structured record
    is appended for **every** leaf parameter — matched or not, exact or judged —
    so the caller can report on all of them, not just the failures. Each record
    is ``{"param", "match_type", "match", ...}``; for a present ``llm_judge``
    parameter a placeholder record (no ``match``/``reasoning`` yet) is appended
    and also linked into ``judge_jobs`` so the verdict can be filled in after the
    concurrent judging completes. Nested objects contribute their leaves' records
    (the container itself gets none).
    """
    for key in _sorted_union_dict_keys(expected, actual):
        path = f"{prefix}.{key}" if prefix else key
        in_act = key in actual
        if key not in expected:
            _record_failure(
                path,
                f"unexpected key in actual output (value={actual[key]!r})",
                lines,
                records,
            )
            continue

        spec = _param_criteria_spec(expected[key], path) if criteria_aware else None
        if spec is not None and spec["match_type"] == "llm_judge":
            if not in_act:
                _record_failure(
                    path,
                    f"missing in actual output (criteria: {spec['criteria']})",
                    lines,
                    records,
                    match_type="llm_judge",
                    criteria=spec["criteria"],
                    missing=True,
                )
            else:
                record = None
                if records is not None:
                    record = {
                        "param": path,
                        "match_type": "llm_judge",
                        "criteria": spec["criteria"],
                    }
                    records.append(record)
                judge_jobs.append((path, spec, actual[key], record))
            continue

        ev = spec["value"] if spec is not None else expected[key]
        if not in_act:
            _record_failure(
                path,
                f"missing in actual output (expected {ev!r})",
                lines,
                records,
                missing=True,
            )
            continue
        av = actual[key]
        if ev == av:
            if records is not None:
                records.append(
                    {"param": path, "match_type": "exact", "match": True}
                )
            continue
        if isinstance(ev, dict) and isinstance(av, dict):
            if spec is not None:
                # exact spec → compare its value literally (specs inside an
                # exact value are not re-interpreted)
                sub_lines = _tool_call_arguments_diff_lines(ev, av, path)
                lines.extend(sub_lines)
                if records is not None:
                    records.append(
                        {
                            "param": path,
                            "match_type": "exact",
                            "match": False,
                            "reasoning": "; ".join(
                                line.strip() for line in sub_lines
                            ),
                        }
                    )
            else:
                _collect_arg_diffs(
                    ev,
                    av,
                    path,
                    lines,
                    judge_jobs,
                    criteria_aware=criteria_aware,
                    records=records,
                )
            continue
        _record_failure(path, _value_mismatch_detail(ev, av), lines, records)


def _param_path(record: dict, tool: Optional[str]) -> str:
    """The parameter's display path, prefixed ``tool.param`` when ``tool`` is set.

    The single place the tool-name prefix is applied, so every reasoning line
    qualifies parameter names the same way.
    """
    return f"{tool}.{record['param']}" if tool else record["param"]


def _render_failing_param_line(record: dict, *, tool: Optional[str] = None) -> str:
    """One ``  ❌ path: detail`` line for a failing parameter record."""
    path = _param_path(record, tool)
    if record.get("missing"):
        return f"  ❌ {path}: {record['reasoning']}"
    if record["match_type"] == "llm_judge":
        return f"  ❌ {path}: criteria not met — {record.get('reasoning', '')}"
    return f"  ❌ {path}: {record.get('reasoning', '')}"


def _render_passing_param_line(record: dict, *, tool: Optional[str] = None) -> str:
    """One ``  ✅ path: detail`` line for a passing parameter record."""
    path = _param_path(record, tool)
    if record["match_type"] == "llm_judge":
        return f"  ✅ {path}: criteria met — {record.get('reasoning', '')}"
    return f"  ✅ {path}: value matches the expected value"


def _detailed_call_lines(records: List[dict], *, tool: Optional[str] = None) -> List[str]:
    """Full per-parameter breakdown for a call's arguments.

    Failures (``❌``) are rendered first so the things that need attention are
    surfaced at the top, followed by passes (``✅``). Used for both the
    all-passed reasoning and the per-call mismatch message so the agent's
    output is reported the same way whether the call passed or failed.
    """
    lines: List[str] = []
    for r in records:
        if not r.get("match"):
            lines.append(_render_failing_param_line(r, tool=tool))
    for r in records:
        if r.get("match"):
            lines.append(_render_passing_param_line(r, tool=tool))
    return lines


async def _tool_call_arguments_eval_async(
    tool_name: str, expected, actual
) -> dict:
    """Evaluate one tool call's arguments, returning a structured result.

    Returns ``{"message", "records", "had_llm"}``:

    - ``message`` — a multi-line mismatch reason, or ``None`` when the arguments
      match. Built from the per-parameter records: when any ``llm_judge`` param
      is involved the full breakdown (matched exacts + judged verdicts) is shown;
      otherwise only the failing lines are listed (the original exact-only form).
    - ``records`` — one entry per leaf parameter (exact and judged, matched and
      failed); see :func:`_collect_arg_diffs`. Empty for the non-dict edge cases.
    - ``had_llm`` — whether any parameter was graded by an LLM judge.
    """
    header = "Tool call arguments mismatch:"
    if not isinstance(expected, dict):
        return {
            "message": (
                f"{header}\n"
                f"  arguments: cannot diff — expected non-dict "
                f"{type(expected).__name__} {expected!r}, got "
                f"{type(actual).__name__} {actual!r}"
            ),
            "records": [],
            "had_llm": False,
        }
    if not isinstance(actual, dict):
        return {
            "message": (
                f"{header}\n"
                f"  arguments: expected dict {expected!r}, "
                f"got {type(actual).__name__} {actual!r}"
            ),
            "records": [],
            "had_llm": False,
        }

    lines: List[str] = []
    judge_jobs: List[tuple] = []
    records: List[dict] = []
    _collect_arg_diffs(expected, actual, "", lines, judge_jobs, records=records)

    if judge_jobs:
        judge_results = await asyncio.gather(
            *[
                _judge_tool_call_parameter(tool_name, path, spec, value)
                for path, spec, value, _record in judge_jobs
            ]
        )
        for (_path, _spec, _value, record), result in zip(judge_jobs, judge_results):
            if record is not None:
                record["match"] = bool(result.get("match"))
                record["reasoning"] = result.get("reasoning", "")

    had_llm = any(r["match_type"] == "llm_judge" for r in records)
    failures = [r for r in records if not r.get("match")]
    if not failures:
        return {"message": None, "records": records, "had_llm": had_llm}

    body = _detailed_call_lines(records)
    return {
        "message": header + "\n" + "\n".join(body),
        "records": records,
        "had_llm": had_llm,
    }


async def _tool_call_pair_mismatch_async(
    output_tool_call: dict, evaluation_tool_call: dict
) -> dict:
    """Async pair matcher supporting per-parameter ``llm_judge`` criteria.

    Returns ``{"mismatch", "records", "had_llm"}`` — ``mismatch`` is a failure
    reason string or ``None`` when the pair matches; ``records`` is the
    per-parameter breakdown (exact + judged); ``had_llm`` flags whether any
    parameter was judged. Equivalent to :func:`_tool_call_pair_mismatch` when no
    parameter carries a criteria spec.
    """
    if output_tool_call["tool"] != evaluation_tool_call["tool"]:
        return {
            "mismatch": (
                f"Tool call mismatch - expected tool call: "
                f"{evaluation_tool_call['tool']} but got: {output_tool_call['tool']}"
            ),
            "records": [],
            "had_llm": False,
        }
    if "arguments" not in evaluation_tool_call:
        return {"mismatch": None, "records": [], "had_llm": False}
    exp_args = evaluation_tool_call.get("arguments")
    if exp_args is None:
        return {"mismatch": None, "records": [], "had_llm": False}
    out_args = output_tool_call.get("arguments")
    if out_args == exp_args:
        return {"mismatch": None, "records": [], "had_llm": False}
    result = await _tool_call_arguments_eval_async(
        evaluation_tool_call["tool"], exp_args, out_args
    )
    return {
        "mismatch": result["message"],
        "records": result["records"],
        "had_llm": result["had_llm"],
    }


def _consolidated_pass_reasoning(pass_blocks: List[tuple], *, multi: bool) -> str:
    """Build the all-passed reasoning string for a ``tool_call`` evaluation.

    ``pass_blocks`` is a list of ``(tool_name, records)`` for every passing tool
    call that involved at least one ``llm_judge`` parameter. When it is empty —
    i.e. every parameter was matched exactly — the flat success message is
    returned. Otherwise each call contributes a line naming its exact-matched
    parameters plus one ``criteria met`` line per judged parameter, so both the
    exact and the LLM-based matches are consolidated into the overall output.

    When ``multi`` (more than one expected tool call) each parameter name is
    prefixed with its tool so the lines stay unambiguous.
    """
    base = "The agent's tools calls matches the expected tool calls"
    if not pass_blocks:
        return base
    lines = [base + ":"]
    for tool_name, records in pass_blocks:
        lines.extend(_detailed_call_lines(records, tool=tool_name if multi else None))
    return "\n".join(lines)


async def evaluate_tool_calls(output_tool_calls, evaluation_tool_calls):
    """Evaluate produced tool calls against expected ones.

    Each expected argument is matched exactly by default; a parameter whose
    expected value is a criteria spec (see :func:`_param_criteria_spec`) is
    instead graded by an LLM judge.

    Returns ``passed`` / ``reasoning`` plus ``tool_call_results`` — a per-slot
    ``[{"tool": name, "passed": bool}]`` list (sorted order, missing output
    slots marked failed) consumed by :func:`_aggregate_tool_calls`. A slot whose
    call involved at least one ``llm_judge`` parameter additionally carries a
    ``param_judgments`` list — one record per leaf parameter, exact *and* judged,
    matched *and* failed (``{"param", "match_type", "match", ...}``) — so the
    full per-parameter verdict survives into ``results.json``.

    Reasoning rules:
    - Every parameter matched exactly (no judge involved) and all passed → the
      flat ``"The agent's tools calls matches the expected tool calls"``.
    - Any ``llm_judge`` parameter involved → the verdict is spelled out: each
      judged parameter's met/not-met reasoning plus a line stating that the
      exact-matched parameters' values match the expected value (see
      :func:`_consolidated_pass_reasoning` / :func:`_detailed_call_lines`).
    """
    evaluation_sorted = sort_tool_calls(evaluation_tool_calls or [])

    if not output_tool_calls:
        return {
            "passed": False,
            "reasoning": "No tool calls were generated by the LLM",
            "tool_call_results": [
                {"tool": tc["tool"], "passed": False}
                for tc in evaluation_sorted
                if tc.get("tool")
            ],
        }

    output_sorted = sort_tool_calls(output_tool_calls)
    multi = len(evaluation_sorted) > 1

    tool_call_results: List[dict] = []
    first_failure: Optional[str] = None
    pass_blocks: List[tuple] = []
    for i, evaluation_tool_call in enumerate(evaluation_sorted):
        name = evaluation_tool_call.get("tool")
        if i >= len(output_sorted):
            if name:
                tool_call_results.append({"tool": name, "passed": False})
            continue
        evaluated = await _tool_call_pair_mismatch_async(
            output_sorted[i], evaluation_tool_call
        )
        mismatch = evaluated["mismatch"]
        if name:
            slot: dict = {"tool": name, "passed": mismatch is None}
            if evaluated["had_llm"] and evaluated["records"]:
                slot["param_judgments"] = evaluated["records"]
            tool_call_results.append(slot)
        if mismatch and first_failure is None:
            first_failure = mismatch
        if mismatch is None and evaluated["had_llm"] and name:
            pass_blocks.append((name, evaluated["records"]))

    if first_failure is not None:
        return {
            "passed": False,
            "reasoning": first_failure,
            "tool_call_results": tool_call_results,
        }

    return {
        "passed": True,
        "reasoning": _consolidated_pass_reasoning(pass_blocks, multi=multi),
        "tool_call_results": tool_call_results,
    }


def _per_slot_tool_passes(
    output_tool_calls: list, evaluation_tool_calls: Optional[list]
) -> List[tuple]:
    """For each expected tool slot (sorted), whether output matches at that index.

    Aligns with :func:`evaluate_tool_calls` ordering and pairwise rules. Used for
    per-tool leaderboard stats so a failing slot does not mark other tools wrong.
    """
    evaluation_tool_calls = sort_tool_calls(evaluation_tool_calls or [])
    if not evaluation_tool_calls:
        return []
    if not output_tool_calls:
        return [(tc["tool"], False) for tc in evaluation_tool_calls if tc.get("tool")]

    output_tool_calls = sort_tool_calls(output_tool_calls)
    out: List[tuple] = []
    for i, evaluation_tool_call in enumerate(evaluation_tool_calls):
        name = evaluation_tool_call.get("tool")
        if not name:
            continue
        if i >= len(output_tool_calls):
            out.append((name, False))
        else:
            out.append(
                (
                    name,
                    _tool_call_pair_mismatch(output_tool_calls[i], evaluation_tool_call)
                    is None,
                )
            )
    return out


def _no_response_judge_results(evaluators: List[dict], reasoning: str) -> dict:
    """Build judge_results entries for the case where no response was produced.

    Each response-type evaluator that was supposed to grade the reply is
    recorded as ``match=False`` for binary evaluators, or as the evaluator's
    ``scale_min`` (the lowest valid rating) for rating evaluators. Anchoring
    rating fallbacks at ``scale_min`` keeps the value inside the evaluator's
    declared range so downstream mean/min aggregates aren't skewed below the
    scale (e.g. dragging a 1-5 rating's mean toward 0).
    """
    judge_results: dict = {}
    for ev in evaluators or []:
        name = ev.get("name")
        if not name:
            continue
        if is_rating(ev):
            try:
                fallback_score = int(ev["scale_min"])
            except (KeyError, TypeError, ValueError):
                fallback_score = 0
            judge_results[name] = attach_evaluator_id(
                ev,
                {
                    "reasoning": reasoning,
                    "score": fallback_score,
                },
            )
        else:
            judge_results[name] = attach_evaluator_id(
                ev, {"reasoning": reasoning, "match": False}
            )
    return judge_results


def _evaluator_passed(evaluator: dict, ev_result: dict) -> bool:
    """Return whether a single evaluator's result passes.

    - binary  → ``match is True``
    - rating  → ``score == scale_max`` (anything below the top of the scale fails)

    Used to compute the test-case-level ``passed`` flag as the AND of every
    referenced evaluator: any binary mismatch *or* any rating below its scale
    max fails the whole case.
    """
    if is_rating(evaluator):
        return int(ev_result["score"]) == int(evaluator["scale_max"])
    return bool(ev_result["match"])


def _metrics_from_judge_results(evaluators: List[dict], result: dict) -> dict:
    failing = [ev for ev in evaluators if not _evaluator_passed(ev, result[ev["name"]])]
    return {
        "passed": not failing,
        "reasoning": (
            result[failing[0]["name"]]["reasoning"]
            if failing
            else "All evaluators passed"
        ),
        "judge_results": result,
    }


async def _evaluate_response(
    chat_history: List[dict],
    response: str,
    tool_calls: list,
    evaluators: Optional[List[dict]],
    no_response_reasoning_with_tool_calls: str,
    no_response_reasoning_no_tool_calls: str,
) -> dict:
    """Evaluate a ``response``-type test case and build its ``metrics`` dict.

    Shared by :func:`run_test` (internal LLM) and :func:`run_test_external`
    (external agent) so the binary/rating pass logic, failing-evaluator
    reasoning pickup, and empty-response fallback are defined in one place.
    Each caller supplies its own no-response reasoning strings so user-facing
    messages remain caller-specific (e.g. "the LLM" vs. "the external agent").

    The test case passes only when every referenced evaluator passes (AND):
    binary evaluators must match and rating evaluators must reach
    ``scale_max``. See :func:`_evaluator_passed`.

    Returns a dict with ``passed``, ``reasoning``, and ``judge_results``.
    """
    metrics: dict = {"passed": False}
    if response:
        evaluators = evaluators or []
        result = await test_response_llm_judge(
            conversation=chat_history,
            response=response,
            evaluators=evaluators,
        )
        metrics.update(_metrics_from_judge_results(evaluators, result))
    else:
        if tool_calls:
            metrics["reasoning"] = no_response_reasoning_with_tool_calls
        else:
            metrics["reasoning"] = no_response_reasoning_no_tool_calls

        metrics["judge_results"] = _no_response_judge_results(
            evaluators or [], metrics["reasoning"]
        )
    return metrics


async def _evaluate_conversation(
    chat_history: List[dict],
    evaluators: List[dict],
    output: dict,
    no_response_reasoning_no_tool_calls: str,
) -> dict:
    """Append the agent's last turn to ``chat_history`` and judge the transcript."""
    response = output.get("response")
    tool_calls = output.get("tool_calls") or []
    if not response and not tool_calls:
        return {
            "passed": False,
            "reasoning": no_response_reasoning_no_tool_calls,
            "judge_results": _no_response_judge_results(
                evaluators or [], no_response_reasoning_no_tool_calls
            ),
        }
    turn: dict = {"role": "assistant", "content": response or ""}
    if tool_calls:
        turn["tool_calls"] = [
            {"function": {"name": tc.get("tool"), "arguments": tc.get("arguments")}}
            for tc in tool_calls
        ]
    result = await evaluate_simuation(
        conversation=list(chat_history) + [turn],
        evaluators=evaluators,
    )
    return _metrics_from_judge_results(evaluators, result)


async def evaluate_test_case_output(
    chat_history: List[dict],
    evaluation: dict,
    output: dict,
    evaluators: Optional[List[dict]] = None,
    no_response_reasoning_with_tool_calls: Optional[str] = None,
    no_response_reasoning_no_tool_calls: Optional[str] = None,
) -> dict:
    """Compute metrics for a test case given its (already produced) output.

    Shared between live inference (``run_test`` / ``run_test_external``) and
    eval-only mode where ``output`` is loaded from disk instead of generated.

    ``output`` must contain ``response`` (str) and ``tool_calls`` (list).
    """
    if evaluation["type"] == "conversation":
        return await _evaluate_conversation(
            chat_history=chat_history,
            evaluators=evaluators or [],
            output=output,
            no_response_reasoning_no_tool_calls=(
                no_response_reasoning_no_tool_calls or "No reply was returned"
            ),
        )
    if evaluation["type"] == "tool_call":
        return await evaluate_tool_calls(output["tool_calls"], evaluation["tool_calls"])
    if evaluation["type"] == "response":
        tool_calls = output["tool_calls"]
        return await _evaluate_response(
            chat_history=chat_history,
            response=output["response"],
            tool_calls=tool_calls,
            evaluators=evaluators,
            no_response_reasoning_with_tool_calls=(
                no_response_reasoning_with_tool_calls
                or f"Tool calls were generated: {tool_calls}, but no reply was returned"
            ),
            no_response_reasoning_no_tool_calls=(
                no_response_reasoning_no_tool_calls or "No reply was returned"
            ),
        )
    raise ValueError(f"Invalid evaluation type: {evaluation['type']}")


@observe(name="llm_test", capture_input=False, capture_output=False)
async def run_test(
    chat_history: List[dict[str, str]],
    evaluation: dict[str, str],
    system_prompt: str,
    model: str,
    provider: str,
    tools: List[dict[str, str]],
    unique_id: str,
    evaluators: Optional[List[dict]] = None,
):
    # Time only response generation; the LLM-judge evaluation that follows is
    # deliberately excluded so the reported latency reflects the agent, not the
    # judge.
    inference_start = time.time()
    output = await run_inference(
        chat_history=chat_history,
        system_prompt=system_prompt,
        model=model,
        provider=provider,
        tools=tools,
    )
    latency_ms = round((time.time() - inference_start) * 1000)

    # Check for system-level failures: if both response and tool_calls are empty,
    # LLM inference failed (API error, invalid model, auth failure, etc.)
    if not output["response"] and not output["tool_calls"]:
        error_details = ""
        if output.get("captured_errors"):
            error_details = f"{'; '.join(output['captured_errors'])}"
        raise LLMInferenceError(
            f"LLM inference failed - no response or tool calls returned. {error_details}"
        )

    metrics = await evaluate_test_case_output(
        chat_history=chat_history,
        evaluation=evaluation,
        output=output,
        evaluators=evaluators,
        no_response_reasoning_with_tool_calls=(
            f"The LLM generated tool calls: {output['tool_calls']}, but no reply was generated"
        ),
        no_response_reasoning_no_tool_calls="No reply was generated by the LLM",
    )

    if langfuse_enabled and langfuse:
        langfuse.update_current_trace(
            input={
                "chat_history": chat_history,
                "evaluation": evaluation,
                "model": model,
                "provider": provider,
            },
            output={"output": output, "metrics": metrics},
            metadata={
                "model": model,
                "provider": provider,
                "tools": tools,
                "system_prompt": system_prompt,
                "input": f"Chat history: {chat_history}\nEvaluation: {evaluation}",
                "output": f"Output: {output}\n\nMetrics: {metrics}",
            },
            session_id=unique_id,
        )

    return {
        "output": output,
        "metrics": metrics,
        "latency_ms": latency_ms,
    }


async def run_test_external(
    chat_history: List[dict],
    evaluation: dict,
    agent,
    model: Optional[str] = None,
    evaluators: Optional[List[dict]] = None,
) -> dict:
    """Run a single LLM test case against an external text agent.

    Sends ``chat_history`` to the agent and evaluates the response using the
    same logic as the internal :func:`run_test`.

    The agent must return ``{"response": ..., "tool_calls": [...]}`` — see
    :meth:`~calibrate.connections.TextAgentConnection.call` for details. It may
    optionally include a ``metrics`` dict (e.g. ``{"cost": 0.0021,
    "latency_ms": 850}``); when present and well-formed it is preserved on the
    output and its ``cost`` / ``latency_ms`` feed the same per-model aggregates
    as internal LLM tests. Anything malformed is ignored so the agent contract
    stays backwards-compatible.

    Unlike the internal path, latency is **not** measured here: a wall-clock
    timer around the HTTP call would fold in network, proxy, and queueing
    overhead rather than the agent's true inference time. We rely solely on the
    agent's self-reported ``metrics.latency_ms``; if it doesn't report one, the
    run simply has no latency.

    Args:
        chat_history: Conversation history (role/content dicts, no system message).
        evaluation: Evaluation dict with ``type`` and criteria.
        agent: A :class:`~calibrate.connections.TextAgentConnection`.
        model: Optional model name included in the request body (for benchmarking).

    Returns:
        dict with ``output`` and ``metrics`` keys.
    """
    output = await agent.call(chat_history, model=model)
    response = output.get("response")
    tool_calls = output.get("tool_calls", [])
    agent_metrics = output.get("metrics")
    metrics = await evaluate_test_case_output(
        chat_history=chat_history,
        evaluation=evaluation,
        output={"response": response, "tool_calls": tool_calls},
        evaluators=evaluators,
        no_response_reasoning_with_tool_calls=(
            f"The agent made tool calls {tool_calls} but returned no text response"
        ),
        no_response_reasoning_no_tool_calls="No reply was returned by the external agent",
    )

    result_output: dict = {"response": response, "tool_calls": tool_calls}
    if isinstance(agent_metrics, dict):
        result_output["metrics"] = agent_metrics

    # Lift the agent's self-reported cost and latency out of its metrics block
    # into the same canonical fields the internal path populates, so both are
    # extracted the same way (cost → output.cost, latency → result.latency_ms)
    # instead of one being dug back out of the nested metrics during aggregation.
    # Latency uses only the agent's number (clean inference time), never a
    # network-contaminated round-trip measured from here.
    metrics_block = agent_metrics if isinstance(agent_metrics, dict) else {}
    agent_cost = _numeric_or_none(metrics_block.get("cost"))
    if agent_cost is not None:
        result_output["cost"] = agent_cost

    result: dict = {"output": result_output, "metrics": metrics}
    agent_latency = _numeric_or_none(metrics_block.get("latency_ms"))
    if agent_latency is not None:
        result["latency_ms"] = agent_latency
    return result


def _aggregate_criteria(results: List[dict], name_to_evaluator: dict) -> dict:
    """Aggregate per-evaluator metrics across test case results.

    Each ``response`` / ``conversation`` test case contributes to the totals for
    the evaluators referenced in its ``evaluation.criteria``.
    ``name_to_evaluator`` maps evaluator name to its config dict (type/scale).

    Per-evaluator output shape depends on the evaluator's type:
    - binary: ``{"type": "binary", "passed": int, "total": int, "pass_rate": float}``
    - rating: ``{"type": "rating", "mean": float, "min": int, "max": int,
                  "count": int, "scale_min": int, "scale_max": int}``
    """
    binary_totals: defaultdict = defaultdict(lambda: {"passed": 0, "total": 0})
    rating_scores: defaultdict = defaultdict(list)
    rating_scale: dict = {}

    for result in results:
        metrics = result.get("metrics", {})
        evaluation = result.get("test_case", {}).get("evaluation", {})

        if evaluation.get("type") not in ("response", "conversation"):
            continue

        judge_results = metrics.get("judge_results")
        if not judge_results:
            continue

        refs = _normalize_criteria_refs(evaluation.get("criteria"))
        for ref in refs:
            name = ref["name"]
            ev = name_to_evaluator.get(name)
            if ev is None:
                continue
            ev_data = judge_results.get(name, {})
            if is_rating(ev):
                if "score" in ev_data:
                    rating_scores[name].append(int(ev_data["score"]))
                    rating_scale[name] = (
                        int(ev["scale_min"]),
                        int(ev["scale_max"]),
                    )
            else:
                binary_totals[name]["total"] += 1
                if ev_data.get("match"):
                    binary_totals[name]["passed"] += 1

    aggregated: dict = {}
    for name, c in binary_totals.items():
        aggregated[name] = {
            "type": "binary",
            "passed": c["passed"],
            "total": c["total"],
            "pass_rate": (c["passed"] / c["total"]) * 100 if c["total"] else 0.0,
        }
        ev = name_to_evaluator.get(name)
        if ev and "id" in ev:
            aggregated[name]["evaluator_id"] = ev["id"]
    for name, scores in rating_scores.items():
        lo, hi = rating_scale[name]
        aggregated[name] = {
            "type": "rating",
            "mean": float(sum(scores) / len(scores)) if scores else 0.0,
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "count": len(scores),
            "scale_min": lo,
            "scale_max": hi,
        }
        ev = name_to_evaluator.get(name)
        if ev and "id" in ev:
            aggregated[name]["evaluator_id"] = ev["id"]
    return aggregated


def _aggregate_cost(results: List[dict]) -> Optional[dict]:
    """Aggregate per-test-case LLM cost (USD) across results that report one.

    Cost lives in ``output.cost`` for every path: internal LLM tests get it from
    OpenRouter (see :class:`CostTrackingOpenRouterLLMService`) and external
    agents have it lifted there from their self-reported ``metrics`` dict in
    :func:`run_test_external`. Test cases without a cost (the OpenAI provider,
    agents that don't report one, or older results) are ignored. Returns
    ``None`` when no result carries a cost so the metric is absent rather than a
    misleading ``0.0``.

    Output shape mirrors :func:`_aggregate_latency`:
    ``{"mean", "min", "max", "count"}`` with USD floats.
    """
    costs = [
        c
        for r in results
        if (c := _numeric_or_none((r.get("output") or {}).get("cost"))) is not None
    ]
    if not costs:
        return None
    return {
        "mean": sum(costs) / len(costs),
        "min": min(costs),
        "max": max(costs),
        "count": len(costs),
    }


def _aggregate_tool_calls(results: List[dict]) -> dict:
    """Aggregate per-tool pass rates across tool_call-type test case results.

    Each expected tool slot (same sort order as :func:`evaluate_tool_calls`) adds
    one to that tool's ``total``; ``passed`` increments only when the output
    matches at that index (tool name and optional arguments), not from the
    case-level ``metrics.passed`` flag.

    Per-slot results are read from ``metrics.tool_call_results`` (produced by
    :func:`evaluate_tool_calls`) so LLM-judged parameters are not re-graded
    here. Results that predate that field fall back to a synchronous,
    exact-match recompute from the stored output.

    Per-tool output shape: ``{"passed": int, "total": int, "pass_rate": float}``.
    """
    totals: defaultdict = defaultdict(lambda: {"passed": 0, "total": 0})

    for result in results:
        evaluation = result.get("test_case", {}).get("evaluation", {})

        if evaluation.get("type") != "tool_call":
            continue

        stored = (result.get("metrics") or {}).get("tool_call_results")
        if stored is not None:
            slot_passes = [
                (slot.get("tool"), bool(slot.get("passed")))
                for slot in stored
                if slot.get("tool")
            ]
        else:
            output = result.get("output") or {}
            slot_passes = _per_slot_tool_passes(
                output.get("tool_calls") or [], evaluation.get("tool_calls")
            )

        for name, slot_passed in slot_passes:
            totals[name]["total"] += 1
            if slot_passed:
                totals[name]["passed"] += 1

    aggregated: dict = {}
    for name, c in totals.items():
        aggregated[name] = {
            "passed": c["passed"],
            "total": c["total"],
            "pass_rate": (c["passed"] / c["total"]) * 100 if c["total"] else 0.0,
        }
    return aggregated


async def run_model_tests(
    model: str,
    provider: str,
    config: dict,
    output_dir: str,
    test_parallel: Optional[int] = None,
) -> dict:
    """Run tests for a single model and return results.

    Args:
        model: Model name to evaluate
        provider: LLM provider (openai or openrouter)
        config: Test configuration dict
        output_dir: Base output directory - results saved to output_dir/model_name/
        test_parallel: Max test cases to evaluate concurrently.
    """
    # Build model folder name: for openai provider, prefix with provider name
    save_folder_name = f"{provider}/{model}" if provider == "openai" else f"{model}"
    save_folder_name = save_folder_name.replace("/", "__")
    model_output_dir = join(output_dir, save_folder_name)

    if not exists(model_output_dir):
        os.makedirs(model_output_dir)

    log_save_path = join(model_output_dir, "logs")
    if exists(log_save_path):
        os.remove(log_save_path)

    # Add file sink for pipecat logs (use lock to avoid race conditions in parallel runs)
    with _logger_lock:
        log_sink_id = logger.add(log_save_path, level="DEBUG")

    # Route judge LLM input/output into this model's logs file (per-context).
    judge_log_token = provider_log_file.set(log_save_path)

    print_log_save_path = join(model_output_dir, "results.log")
    if exists(print_log_save_path):
        os.remove(print_log_save_path)

    label = display_label(provider, model)

    # Print model header (mirrored to results.log)
    _print_and_log(f"\n\033[94m{'='*60}\033[0m", print_log_save_path)
    _print_and_log(f"\033[94mModel: {label}\033[0m", print_log_save_path)
    _print_and_log(f"\033[94m{'='*60}\033[0m\n", print_log_save_path)

    results_file_path = join(model_output_dir, "results.json")

    unique_id = str(uuid.uuid4())

    name_to_evaluator = _get_name_to_evaluator_dict(config)
    write_evaluator_config(output_dir, _evaluators_for_config_output(config))

    tools = config.get("tools") or []
    system_prompt = config.get("system_prompt", "")

    async def process(test_case_index: int, test_case: dict) -> dict:
        evaluation = test_case["evaluation"]
        preprocessed_history = preprocess_conversation_history(
            test_case["history"], tools
        )
        resolved_evaluators = (
            _resolve_evaluators_for_test_case(
                evaluation,
                _get_name_to_evaluator_dict(
                    config,
                    include_default=(evaluation.get("type") == "response"),
                ),
            )
            if evaluation.get("type") in ("response", "conversation")
            else None
        )

        result = await run_test(
            chat_history=preprocessed_history,
            evaluation=evaluation,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            tools=tools,
            unique_id=unique_id,
            evaluators=resolved_evaluators,
        )

        if result["metrics"]["passed"]:
            _print_and_log(
                f"[{label}] ✅ Test case {test_case_index + 1} passed",
                print_log_save_path,
            )
        else:
            _print_and_log(
                f"[{label}] ❌ Test case {test_case_index + 1} failed",
                print_log_save_path,
            )
        if "reasoning" in result["metrics"]:
            _print_and_log(
                f"  Reason: {result['metrics']['reasoning']}",
                print_log_save_path,
            )

        if "id" in test_case:
            result["test_case_id"] = test_case["id"]
        result["test_case"] = test_case
        return result

    results = await _run_items_parallel(
        config["test_cases"], process, results_file_path, test_parallel
    )

    total_passed = sum(1 for result in results if result["metrics"]["passed"])
    total_tests = len(results)
    passed_count = total_passed
    failed_count = total_tests - total_passed

    # Print summary for this model (mirrored to results.log)
    if passed_count == total_tests:
        _print_and_log(f"[{label}] 🎉 All tests passed!", print_log_save_path)
    elif failed_count == total_tests:
        _print_and_log(f"[{label}] ❌ All tests failed!", print_log_save_path)
    else:
        _print_and_log(
            f"[{label}] ✅ Total Passed: {passed_count}/{total_tests} ({(passed_count/total_tests)*100:.1f}%)",
            print_log_save_path,
        )

    _write_test_results_outputs(results, model_output_dir, name_to_evaluator)

    # Remove pipecat log file sink
    with _logger_lock:
        logger.remove(log_sink_id)

    provider_log_file.reset(judge_log_token)

    return {
        "model": model,
        "provider": provider,
        "metrics": {"passed": passed_count, "total": total_tests},
        "results": results,
    }


def _aggregate_latency(results: List[dict]) -> Optional[dict]:
    """Aggregate per-test-case response-generation latency across results.

    Only results that carry a numeric ``latency_ms`` (those produced by live
    inference — ``run_test`` / ``run_test_external``) contribute; eval-only
    results, which skip inference, have none. Returns ``None`` when no result
    carries a latency so eval-only ``metrics.json`` stays free of an empty block.

    Output shape: ``{"mean", "min", "max", "count"}`` with millisecond ints.
    """
    values = [
        r["latency_ms"]
        for r in results
        if isinstance(r.get("latency_ms"), (int, float))
        and not isinstance(r.get("latency_ms"), bool)
    ]
    if not values:
        return None
    return {
        "mean": round(sum(values) / len(values)),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def _write_test_results_outputs(
    results: List[dict],
    output_dir: str,
    name_to_evaluator: dict,
) -> tuple[int, int]:
    """Write results.json + metrics.json for an LLM test run.

    Returns ``(passed, total)``.
    """
    total = len(results)
    passed = sum(1 for r in results if r["metrics"]["passed"])

    with open(join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    metrics = {
        "total": total,
        "passed": passed,
        "criteria": _aggregate_criteria(results, name_to_evaluator),
        "tool_calls": _aggregate_tool_calls(results),
    }
    cost = _aggregate_cost(results)
    if cost is not None:
        metrics["cost"] = cost
    latency = _aggregate_latency(results)
    if latency is not None:
        metrics["latency_ms"] = latency
    with open(join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return passed, total


def validate_llm_eval_only_dataset(
    dataset: object,
) -> tuple[bool, str]:
    """Validate the shape of an LLM eval-only dataset.

    Each item must be ``{"test_case": {history, evaluation}, "output":
    {response, tool_calls}}``. Returns ``(is_valid, error_message)``; the
    caller is expected to surface the message and exit non-zero on failure.

    Each entry in ``output.tool_calls`` may optionally carry an ``output``
    field (the tool's own result). It is accepted but not validated — it is
    preserved for display/review only and never affects evaluation.
    """
    if not isinstance(dataset, list):
        return False, "Dataset must be a JSON list of {test_case, output} items"

    for i, item in enumerate(dataset):
        if not isinstance(item, dict):
            return False, f"Item {i}: must be an object"
        if "test_case" not in item or "output" not in item:
            return (
                False,
                f"Item {i}: missing required keys 'test_case' and/or 'output'",
            )
        tc = item["test_case"]
        out = item["output"]
        if not isinstance(tc, dict):
            return False, f"Item {i}: 'test_case' must be an object"
        if not isinstance(out, dict):
            return False, f"Item {i}: 'output' must be an object"
        if "history" not in tc or "evaluation" not in tc:
            return (
                False,
                f"Item {i}: 'test_case' missing required fields 'history' and/or 'evaluation'",
            )
        if not isinstance(tc["history"], list):
            return False, f"Item {i}: 'test_case.history' must be a list"
        if not isinstance(tc["evaluation"], dict):
            return False, f"Item {i}: 'test_case.evaluation' must be an object"
        ev_type = tc["evaluation"].get("type")
        if ev_type not in ("response", "tool_call", "conversation"):
            return (
                False,
                f"Item {i}: 'test_case.evaluation.type' must be 'response', "
                f"'tool_call', or 'conversation' (got {ev_type!r})",
            )
        if "response" not in out or "tool_calls" not in out:
            return (
                False,
                f"Item {i}: 'output' must include 'response' (str) and 'tool_calls' (list)",
            )
        if not isinstance(out.get("tool_calls", []), list):
            return False, f"Item {i}: 'output.tool_calls' must be a list"

    return True, ""


async def run_eval_only_tests(
    config: dict,
    dataset: list[dict],
    output_dir: str,
    test_parallel: Optional[int] = None,
) -> dict:
    """Run evaluators on a pre-existing dataset of (test_case, output) pairs.

    Skips LLM inference. ``config`` supplies the evaluators registry; each
    dataset item must have a ``test_case`` (with ``history`` and
    ``evaluation``) and an ``output`` (with ``response`` and ``tool_calls``).

    Writes ``results.json`` and ``metrics.json`` to ``output_dir``.
    """
    os.makedirs(output_dir, exist_ok=True)

    name_to_evaluator = _get_name_to_evaluator_dict(config)
    write_evaluator_config(output_dir, _evaluators_for_config_output(config))

    log_save_path = join(output_dir, "logs")
    if exists(log_save_path):
        os.remove(log_save_path)
    # Route judge LLM input/output into the eval-only logs file.
    judge_log_token = provider_log_file.set(log_save_path)

    print_log_save_path = join(output_dir, "results.log")
    if exists(print_log_save_path):
        os.remove(print_log_save_path)

    _print_and_log("\n\033[94mEval-only\033[0m\n", print_log_save_path)

    results_file_path = join(output_dir, "results.json")

    tools = config.get("tools", []) or []

    async def process(i: int, item: dict) -> dict:
        test_case = item["test_case"]
        evaluation = test_case["evaluation"]
        resolved_evaluators = (
            _resolve_evaluators_for_test_case(
                evaluation,
                _get_name_to_evaluator_dict(
                    config,
                    include_default=(evaluation.get("type") == "response"),
                ),
            )
            if evaluation.get("type") in ("response", "conversation")
            else None
        )

        preprocessed_history = preprocess_conversation_history(
            test_case["history"], tools
        )
        output = item["output"]
        metrics = await evaluate_test_case_output(
            chat_history=preprocessed_history,
            evaluation=evaluation,
            output=output,
            evaluators=resolved_evaluators,
            no_response_reasoning_with_tool_calls=(
                f"Tool calls present: {output.get('tool_calls')}, but no reply provided"
            ),
            no_response_reasoning_no_tool_calls="No reply provided",
        )

        if metrics["passed"]:
            _print_and_log(f"✅ Test case {i + 1} passed", print_log_save_path)
        else:
            _print_and_log(f"❌ Test case {i + 1} failed", print_log_save_path)
        if "reasoning" in metrics:
            _print_and_log(f"  Reason: {metrics['reasoning']}", print_log_save_path)

        result = {"output": output, "metrics": metrics, "test_case": test_case}
        if "id" in test_case:
            result["test_case_id"] = test_case["id"]
        return result

    results = await _run_items_parallel(
        dataset, process, results_file_path, test_parallel
    )

    passed, total = _write_test_results_outputs(results, output_dir, name_to_evaluator)
    pct = (passed / total * 100) if total else 0.0
    _print_and_log(
        f"\n✅ Total Passed: {passed}/{total} ({pct:.1f}%)", print_log_save_path
    )

    provider_log_file.reset(judge_log_token)

    return {"passed": passed, "total": total, "results": results}


async def main():
    """CLI entry point for single-model LLM test evaluation.

    Used by the Ink UI which spawns individual model processes.
    For multi-model benchmark, use benchmark.py via `calibrate llm -m model1 model2 ...`
    """
    parser = argparse.ArgumentParser(
        description="Single-model LLM test evaluation (used by Ink UI)"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file for the tests",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./out",
        help="Path to the output directory to save the results",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model to use for evaluation. Not required with --eval-only.",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=["openai", "openrouter"],
        default="openrouter",
        help="LLM provider to use (openai or openrouter)",
    )
    parser.add_argument(
        "-n",
        "--parallel",
        type=int,
        default=None,
        help="Number of test cases to evaluate in parallel",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip LLM inference and run evaluators on a dataset of (test_case, output) pairs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON for --eval-only (list of {test_case, output} items)",
    )

    args = parser.parse_args()

    config = json.load(open(args.config))

    if args.eval_only:
        if not args.dataset:
            print("\033[31mError: --dataset is required with --eval-only\033[0m")
            sys.exit(1)

        try:
            with open(args.dataset) as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"\033[31mError: failed to read dataset {args.dataset}: {e}\033[0m")
            sys.exit(1)

        is_valid, err = validate_llm_eval_only_dataset(dataset)
        if not is_valid:
            print(f"\033[31mDataset validation error: {err}\033[0m")
            sys.exit(1)

        print("\n\033[91mLLM Eval-Only\033[0m\n")
        print(f"Config: {args.config}")
        print(f"Dataset: {args.dataset}")
        print("")

        os.makedirs(args.output_dir, exist_ok=True)
        result = await run_eval_only_tests(
            config=config,
            dataset=dataset,
            output_dir=args.output_dir,
            test_parallel=args.parallel,
        )
        passed = result["passed"]
        total = result["total"]
        pct = (passed / total * 100) if total else 0.0
        print(f"\n\033[92m{'='*60}\033[0m")
        print(f"\033[92mSummary\033[0m")
        print(f"\033[92m{'='*60}\033[0m\n")
        print(f"  eval-only: {passed}/{total} ({pct:.1f}%)")
        return

    if not args.model:
        print("\033[31mError: --model is required (omit only with --eval-only)\033[0m")
        sys.exit(1)

    model = args.model

    print("\n\033[91mLLM Tests\033[0m\n")
    print(f"Config: {args.config}")
    print(f"Model: {display_label(args.provider, model)}")
    print(f"Provider: {args.provider}")
    print("")

    # Run tests for single model - results saved to output_dir/model_name/
    result = await run_model_tests(
        model=model,
        provider=args.provider,
        config=config,
        output_dir=args.output_dir,
        test_parallel=args.parallel,
    )

    # Print summary
    print(f"\n\033[92m{'='*60}\033[0m")
    print(f"\033[92mSummary\033[0m")
    print(f"\033[92m{'='*60}\033[0m\n")

    passed = result["metrics"]["passed"]
    total = result["metrics"]["total"]
    pct = (passed / total * 100) if total > 0 else 0
    print(f"  {result['provider']}/{result['model']}: {passed}/{total} ({pct:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
