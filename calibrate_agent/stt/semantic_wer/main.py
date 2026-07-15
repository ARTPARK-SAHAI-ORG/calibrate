"""Prompt + response schema for the semantic WER judge.

Implements pipecat's stt-benchmark "semantic WER" methodology
(https://github.com/pipecat-ai/stt-benchmark): a single holistic LLM call per
(reference, hypothesis) pair where the model normalizes, aligns, applies a
semantic error check ("would an LLM agent respond differently?"), and reports the
substitution / deletion / insertion counts plus the reference word count. The WER
itself is computed in Python from those counts (see ``get_semantic_wer_score`` in
``stt/metrics.py``), mirroring pipecat's ``_calculate_wer``.

This is distinct from ``stt/sarvam_llm_wer`` (Sarvam's approach), which aligns
deterministically with difflib and only asks the LLM to forgive per-segment
equivalence — here the LLM does the whole count in one call.
"""

from pathlib import Path

from pydantic import BaseModel, Field

PROMPT_PATH = Path(__file__).parent / "prompt_template.txt"

try:
    PROMPT_TEMPLATE = PROMPT_PATH.read_text()
except FileNotFoundError:
    raise FileNotFoundError(
        f"prompt_template.txt not found at {PROMPT_PATH}. "
        "Please ensure it exists alongside main.py."
    )


class SemanticWERResponse(BaseModel):
    """Structured output the judge returns (equivalent to pipecat's
    ``calculate_wer`` tool call)."""

    normalized_reference: str = Field(
        default="", description="Reference after applying the normalization rules"
    )
    normalized_hypothesis: str = Field(
        default="", description="Hypothesis after applying the normalization rules"
    )
    substitutions: int = Field(description="Number of meaning-changing word substitutions")
    deletions: int = Field(description="Number of meaning-changing word deletions")
    insertions: int = Field(description="Number of meaning-changing word insertions")
    reference_words: int = Field(description="Total word count in the normalized reference")
    reasoning: str = Field(
        default="", description="Brief semantic-check reasoning for the counted errors"
    )


def build_prompt(reference: str, prediction: str) -> str:
    """Build the semantic-WER prompt for a single (reference, hypothesis) pair."""
    return (
        f"{PROMPT_TEMPLATE}\n\n"
        "=== EVALUATE ===\n"
        f'REFERENCE: "{reference}"\n'
        f'HYPOTHESIS: "{prediction}"\n'
    )
