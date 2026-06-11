"""General task evaluation metrics.

Thin, non-conversational counterpart to ``calibrate.stt.metrics`` /
``calibrate.llm.metrics``: judge a list of ``(input, output)`` task pairs
against a list of evaluators and aggregate per-evaluator scores.
"""

from typing import List, Optional

import numpy as np
import backoff
from tqdm.asyncio import tqdm_asyncio

from calibrate.judges import (
    general_task_judge,
    is_rating,
    evaluator_result_value,
    ensure_known_evaluator_names,
    render_evaluator,
    require_unique_evaluator_names,
    DEFAULT_TEXT_JUDGE_MODEL,
)
from calibrate.langfuse import observe, langfuse, langfuse_enabled

# Re-export for symmetry with the other metrics modules.
DEFAULT_GENERAL_JUDGE_MODEL = DEFAULT_TEXT_JUDGE_MODEL


def _require_evaluators(evaluators: Optional[List[dict]]) -> List[dict]:
    """Return ``evaluators`` if it is a non-empty list with unique names.

    The general task judge has no implicit default — there is no universal
    criteria to grade against — so callers must supply at least one evaluator
    (each carrying its own ``system_prompt``). Mirrors
    :func:`calibrate.judges.require_simulation_evaluators`.
    """
    if not isinstance(evaluators, list) or len(evaluators) == 0:
        raise ValueError(
            "General task evaluation requires a non-empty `evaluators` list "
            "(there is no implicit default). Each evaluator must define a "
            "`name` and `system_prompt`."
        )
    require_unique_evaluator_names(evaluators)
    return list(evaluators)


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(
    name="general_llm_judge",
    capture_input=False,
)
async def general_judge(
    input_text: Optional[str],
    output: str,
    evaluators: List[dict],
    fallback_model: str = DEFAULT_GENERAL_JUDGE_MODEL,
) -> dict:
    """Evaluate a single task output (with optional input) against evaluators.

    Args:
        input_text: The task input the output was produced for (optional).
        output: The output text to evaluate.
        evaluators: List of evaluator dicts (already rendered).
        fallback_model: Model id used when an evaluator lacks ``judge_model``.

    Returns:
        Dict keyed by evaluator name — same shape as
        :func:`calibrate.judges.text_judge`.
    """
    result = await general_task_judge(
        evaluators=evaluators,
        output=output,
        input_text=input_text,
        fallback_model=fallback_model,
    )

    if langfuse_enabled and langfuse:
        langfuse.update_current_trace(
            input={"input": input_text, "output": output},
            metadata={
                "input": input_text,
                "output": output,
                "result": result,
            },
        )

    return result


async def get_general_judge_score(
    inputs: List[Optional[str]],
    outputs: List[str],
    evaluators: List[dict],
    fallback_model: str = DEFAULT_GENERAL_JUDGE_MODEL,
    arguments_list: Optional[List[Optional[dict]]] = None,
) -> dict:
    """Run the general judge across all rows and aggregate per-evaluator scores.

    ``inputs`` and ``outputs`` are positionally paired; ``inputs[i]`` may be
    ``None`` to judge ``outputs[i]`` on its own.

    ``arguments_list`` optionally supplies per-row, per-evaluator template
    variables. When provided it must have the same length as
    ``inputs``/``outputs``. Each entry is a dict keyed by evaluator ``name`` →
    that evaluator's argument dict (mirroring how ``llm`` test cases attach
    ``arguments`` to each ``criteria`` ref). For row ``i`` and evaluator ``ev``,
    ``ev``'s ``system_prompt`` is rendered against ``arguments_list[i][ev["name"]]``
    (via :func:`calibrate.judges.render_evaluator`) before being passed to the
    judge; evaluators with no entry — and rows with ``None``/empty arguments —
    are left unchanged. An ``arguments`` key that names no known evaluator
    raises ``ValueError`` (mirroring ``llm``, which rejects unknown ``criteria``
    references rather than silently ignoring them). Rendering only changes
    ``system_prompt`` — ``name``/``type``/``scale_*`` are untouched, so
    aggregation remains keyed off the base ``evaluators`` list.

    Returns:
        {
            "scores": {
                "<evaluator>": {"type": "binary", "mean": 0.83},
                "<evaluator>": {"type": "rating", "mean": 4.0,
                                "scale_min": 1, "scale_max": 5},
                ...
            },
            "score": float,        # mean across evaluator means (legacy top-level)
            "per_row": [
                {"<evaluator>": {"reasoning": ..., "match": ...}, ...},
                ...
            ],
        }

    Iteration order of ``scores`` and each ``per_row`` entry matches the order
    of the ``evaluators`` argument.
    """
    evaluators = _require_evaluators(evaluators)

    if len(inputs) != len(outputs):
        raise ValueError(
            f"inputs and outputs must be the same length "
            f"(got {len(inputs)} inputs, {len(outputs)} outputs)."
        )

    if arguments_list is not None and len(arguments_list) != len(inputs):
        raise ValueError(
            f"arguments_list must be the same length as inputs "
            f"(got {len(arguments_list)} arguments, {len(inputs)} inputs)."
        )

    evaluator_names = {ev["name"] for ev in evaluators}

    coroutines = []
    for i, (input_text, output) in enumerate(zip(inputs, outputs)):
        row_arguments = arguments_list[i] if arguments_list is not None else None
        if row_arguments:
            ensure_known_evaluator_names(
                row_arguments, evaluator_names, context=f"Row {i} arguments"
            )
            row_evaluators = [
                render_evaluator(ev, row_arguments.get(ev["name"]))
                for ev in evaluators
            ]
        else:
            row_evaluators = evaluators
        coroutines.append(
            general_judge(
                None if input_text is None else str(input_text),
                str(output),
                evaluators=row_evaluators,
                fallback_model=fallback_model,
            )
        )

    results = await tqdm_asyncio.gather(
        *coroutines,
        desc="Running general evaluators",
    )

    scores: dict = {}
    for ev in evaluators:
        name = ev["name"]
        per_row_values = [evaluator_result_value(ev, row[name]) for row in results]
        if is_rating(ev):
            scores[name] = {
                "type": "rating",
                "mean": float(np.mean(per_row_values)),
                "scale_min": int(ev["scale_min"]),
                "scale_max": int(ev["scale_max"]),
            }
        else:
            scores[name] = {
                "type": "binary",
                "mean": float(np.mean(per_row_values)),  # pass-rate fraction 0.0–1.0
            }

    overall_score = float(np.mean([s["mean"] for s in scores.values()]))

    return {
        "scores": scores,
        "score": overall_score,
        "per_row": results,
    }
