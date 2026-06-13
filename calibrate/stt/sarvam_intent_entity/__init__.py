"""
Vendored from Sarvam AI's ``llm_intent_entity`` repo (no LICENSE published):
https://github.com/sarvamai/llm_intent_entity

Kept as close to the upstream source as practical so the intent/entity judge
flow — prompt, response schema, ``build_prompt``, and the ``IndicNormalizer``
text normalization — matches how Sarvam computes these scores. Only the parts
the calibrate STT pipeline uses are included; the upstream Vertex AI client,
Google Sheets export, and CLI orchestration are omitted.

See also: https://www.sarvam.ai/blogs/evaluating-indian-language-asr
"""

from .main import IntentEntityResponse, build_prompt, PROMPT_TEMPLATE
from .judge import intent_entity_judge, DEFAULT_INTENT_ENTITY_MODEL
from .utilities import (
    IndicNormalizer,
    calculate_intent_accuracy,
    calculate_entity_metrics,
    lang_to_code,
    indic_langs,
)

__all__ = [
    "IntentEntityResponse",
    "build_prompt",
    "PROMPT_TEMPLATE",
    "intent_entity_judge",
    "DEFAULT_INTENT_ENTITY_MODEL",
    "IndicNormalizer",
    "calculate_intent_accuracy",
    "calculate_entity_metrics",
    "lang_to_code",
    "indic_langs",
]
