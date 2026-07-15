"""Semantic WER — pipecat stt-benchmark methodology.

An LLM (Claude by default, via OpenRouter) computes a semantically-forgiving WER
per (reference, hypothesis) pair via pipecat's reason-then-tool loop: with the
rules in a system message and the pair in a user message, it normalizes, aligns,
applies a "would an LLM agent respond differently?" error check, writes its
reasoning, then commits substitution/deletion/insertion counts through a
``calculate_wer`` tool call. WER is computed in Python from those counts. The
prompt and tool schema are pipecat's verbatim; only the transport (OpenRouter's
OpenAI-compatible tool calling vs pipecat's native Anthropic SDK) differs.

Distinct from ``stt/sarvam_llm_wer`` (deterministic difflib alignment + per-
segment equivalence forgiveness). Runs by default as part of the LLM-judge
group; disable the group with ``--skip-llm-judges``.

Reference: https://github.com/pipecat-ai/stt-benchmark
"""

from .main import (
    SYSTEM_PROMPT,
    PROMPT_TEMPLATE,
    CALCULATE_WER_TOOL,
    build_user_prompt,
    build_prompt,
)
from .judge import semantic_wer_judge, DEFAULT_SEMANTIC_WER_MODEL

__all__ = [
    "SYSTEM_PROMPT",
    "PROMPT_TEMPLATE",
    "CALCULATE_WER_TOOL",
    "build_user_prompt",
    "build_prompt",
    "semantic_wer_judge",
    "DEFAULT_SEMANTIC_WER_MODEL",
]
