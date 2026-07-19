"""
LLM evaluation metrics.

Thin wrappers around calibrate_agent.judges that take pre-rendered evaluators.
"""

from typing import List

from calibrate_agent.judges import (
    text_judge,
    simulation_judge,
    DEFAULT_TEXT_JUDGE_MODEL,
    DEFAULT_SIMULATION_JUDGE_MODEL,
)
from calibrate_agent.langfuse import observe

# Re-export defaults for existing imports
DEFAULT_JUDGE_MODEL = DEFAULT_TEXT_JUDGE_MODEL


@observe(
    name="llm_test_llm_judge",
    capture_input=False,
)
async def test_response_llm_judge(
    conversation: List[dict],
    response: str,
    evaluators: List[dict],
    fallback_model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    """Evaluate an LLM response against a list of pre-rendered evaluators.

    Args:
        conversation: Chat history (list of role/content dicts).
        response: The LLM's response text to evaluate.
        evaluators: List of evaluator dicts with their ``system_prompt``
            already rendered (placeholders substituted) by the caller.
        fallback_model: Model id for evaluators that don't set ``judge_model``.

    Returns:
        Dict keyed by evaluator name. Binary entries are
        ``{"reasoning": str, "match": bool}``; rating entries are
        ``{"reasoning": str, "score": int}``.
    """
    conversation_as_prompt = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in conversation if "content" in msg]
    )

    user_prompt = (
        f"`Chat history`:\n\n{conversation_as_prompt}\n\n"
        f"`Response to evaluate`:\n\n{response}"
    )

    return await text_judge(
        evaluators=evaluators,
        user_prompt=user_prompt,
        fallback_model=fallback_model,
    )


async def evaluate_simuation(
    conversation: List[dict],
    evaluators: List[dict],
    fallback_model: str = DEFAULT_SIMULATION_JUDGE_MODEL,
    **kwargs,
) -> dict:
    """Evaluate a simulation transcript against a list of pre-rendered evaluators.

    Simulation has no implicit default evaluator. If ``evaluators`` is empty,
    no judge calls are made and ``{}`` is returned.

    Args:
        conversation: Full conversation transcript (list of role/content dicts).
        evaluators: List of evaluator dicts (already rendered).
        fallback_model: Model id for evaluators that don't set ``judge_model``.
    """
    return await simulation_judge(
        conversation=conversation,
        evaluators=evaluators,
        fallback_model=fallback_model,
    )
