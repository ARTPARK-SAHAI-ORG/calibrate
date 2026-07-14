"""
Vendored from Sarvam AI's ``llm_wer`` repo (no LICENSE published):
https://github.com/sarvamai/llm_wer

Kept as close to the upstream source as practical so the LLM-WER/CER flow —
prompt, response schema, ``build_prompt``, and the ``get_segments`` word
alignment — matches how Sarvam computes these scores. Only the parts the
calibrate_agent STT pipeline uses are included; the upstream Vertex AI client,
Google Sheets export, custom WER/CER, and CLI orchestration are omitted
(calibrate_agent reuses its own jiwer scorer and the ``IndicNormalizer`` vendored
under ``sarvam_intent_entity``).

See also: https://www.sarvam.ai/blogs/evaluating-indian-language-asr
"""

from .main import (
    LLMEquivalenceResponse,
    build_prompt,
    get_segments,
    PROMPT_TEMPLATE,
)
from .judge import equivalence_judge, DEFAULT_LLM_WER_MODEL

__all__ = [
    "LLMEquivalenceResponse",
    "build_prompt",
    "get_segments",
    "PROMPT_TEMPLATE",
    "equivalence_judge",
    "DEFAULT_LLM_WER_MODEL",
]
