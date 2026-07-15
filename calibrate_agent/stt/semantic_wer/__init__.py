"""Semantic WER — pipecat stt-benchmark methodology.

An LLM (Claude by default, via OpenRouter) computes a semantically-forgiving WER
in a single holistic call per (reference, hypothesis) pair: it normalizes, aligns,
applies a "would an LLM agent respond differently?" error check, and reports
substitution/deletion/insertion counts, from which WER is computed in Python.

Distinct from ``stt/sarvam_llm_wer`` (deterministic difflib alignment + per-
segment equivalence forgiveness). Runs by default as part of the LLM-judge
group; disable the group with ``--skip-llm-judges``.

Reference: https://github.com/pipecat-ai/stt-benchmark
"""

from .main import SemanticWERResponse, build_prompt, PROMPT_TEMPLATE
from .judge import semantic_wer_judge, DEFAULT_SEMANTIC_WER_MODEL

__all__ = [
    "SemanticWERResponse",
    "build_prompt",
    "PROMPT_TEMPLATE",
    "semantic_wer_judge",
    "DEFAULT_SEMANTIC_WER_MODEL",
]
