"""
Prompt, response schema, and word-alignment segmentation for the LLM-WER/CER
judge.

Vendored from Sarvam AI's ``llm_wer`` (main.py):
https://github.com/sarvamai/llm_wer

The prompt is read from the sibling ``prompt_template.txt`` rather than the
upstream project root. Only the pieces the calibrate_agent STT pipeline uses are
kept: the ``LLMEquivalenceResponse`` schema, ``get_segments`` (word-level
alignment via ``difflib.SequenceMatcher``), and ``build_prompt``. The upstream
Vertex AI client, Google Sheets export, custom WER/CER, and CLI orchestration
are omitted — calibrate_agent reuses its own jiwer scorer and normalizer.

See also: https://www.sarvam.ai/blogs/evaluating-indian-language-asr
"""

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

PROMPT_PATH = Path(__file__).parent / "prompt_template.txt"

try:
    PROMPT_TEMPLATE = PROMPT_PATH.read_text()
except FileNotFoundError:
    raise FileNotFoundError(
        f"prompt_template.txt not found at {PROMPT_PATH}. "
        "Please ensure it exists alongside main.py."
    )


class LLMEquivalenceResponse(BaseModel):
    index: int
    equivalent: bool
    reasoning: str


def get_segments(
    reference_string: str, predicted_string: str, key: Any
) -> List[Dict[str, Any]]:
    """Word-align ``reference_string`` against ``predicted_string``.

    Returns one dict per ``difflib`` opcode. ``tag`` is one of ``equal``,
    ``replace``, ``insert``, ``delete``; ``reference`` / ``prediction`` hold the
    words on each side of that opcode. Only ``replace`` segments (both sides
    non-empty) are candidates for equivalence forgiveness downstream.
    """
    reference_words = reference_string.strip().split()
    predicted_words = predicted_string.strip().split()
    if not reference_words and not predicted_words:
        return []

    matcher = SequenceMatcher(None, reference_words, predicted_words)
    return [
        {
            "reference": " ".join(reference_words[i1:i2]),
            "prediction": " ".join(predicted_words[j1:j2]),
            "tag": tag,
            "key": key,
            "segment_idx": segment_idx,
        }
        for segment_idx, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes())
    ]


def build_prompt(segment_pair: Dict[str, str]) -> str:
    """Build the equivalence prompt for a single (reference, prediction) segment."""
    prompt = PROMPT_TEMPLATE + "\n\n**INPUT:**\n"
    json_objects = [
        {
            "index": 0,
            "reference": segment_pair["reference"],
            "prediction": segment_pair["prediction"],
        }
    ]
    prompt += json.dumps(json_objects, indent=2, ensure_ascii=False)
    return prompt
