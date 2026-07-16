"""Prompt + tool schema for the semantic WER judge.

Implements pipecat's stt-benchmark "semantic WER" methodology
(https://github.com/pipecat-ai/stt-benchmark): the model normalizes, aligns,
applies a semantic error check ("would an LLM agent respond differently?"), and
reports substitution / deletion / insertion counts plus the reference word count
via a ``calculate_wer`` tool call. The WER itself is computed in Python from
those counts (see ``get_semantic_wer_score`` in ``stt/metrics.py``), mirroring
pipecat's ``_calculate_wer``.

``SYSTEM_PROMPT`` (``prompt_template.txt``) is pipecat's verbatim, and
``CALCULATE_WER_TOOL`` is pipecat's plus one required ``summary`` field (a
concise, publicly-showable verdict — see below), so the judge runs the same
rules + tool contract pipecat does. The judge (``judge.py``) drives them through
a tool-calling loop with a system/user split, matching pipecat's shape; only the
transport (OpenRouter's OpenAI-compatible API vs pipecat's native Anthropic SDK)
and the ``summary`` capture differ.

Distinct from ``stt/sarvam_llm_wer`` (Sarvam's approach), which aligns
deterministically with difflib and only asks the LLM to forgive per-segment
equivalence.

Source: pipecat-ai/stt-benchmark @ 41b34a49a754bf43c99e2ce50a10724be4866941
"""

from pathlib import Path

PROMPT_PATH = Path(__file__).parent / "prompt_template.txt"

try:
    # pipecat's SEMANTIC_WER_SYSTEM_PROMPT, verbatim.
    SYSTEM_PROMPT = PROMPT_PATH.read_text()
except FileNotFoundError:
    raise FileNotFoundError(
        f"prompt_template.txt not found at {PROMPT_PATH}. "
        "Please ensure it exists alongside main.py."
    )

# pipecat's CALCULATE_WER_TOOL (Anthropic tool schema). The judge converts this
# to the OpenAI function-calling shape for OpenRouter; the ``input_schema``
# doubles as the OpenAI ``parameters`` object unchanged.
#
# Calibrate addition (not in pipecat): a required ``summary`` field. pipecat
# keeps the model's full chain-of-thought only for offline debugging and never
# surfaces it; calibrate shows per-row judge reasoning in the leaderboard UI, so
# we ask the model to commit a short, publicly-showable verdict alongside the
# counts instead of leaking the raw CoT. See ``semantic_wer_judge``.
CALCULATE_WER_TOOL = {
    "name": "calculate_wer",
    "description": "Calculate Word Error Rate from error counts. Call this ONCE after you have normalized, aligned, and verified the texts. WER = (substitutions + deletions + insertions) / reference_words",
    "input_schema": {
        "type": "object",
        "properties": {
            "substitutions": {
                "type": "integer",
                "description": "Number of word substitutions (different words at same position)",
            },
            "deletions": {
                "type": "integer",
                "description": "Number of word deletions (words in reference missing from hypothesis)",
            },
            "insertions": {
                "type": "integer",
                "description": "Number of word insertions (extra words in hypothesis not in reference)",
            },
            "reference_words": {
                "type": "integer",
                "description": "Total word count in normalized reference text",
            },
            "normalized_reference": {
                "type": "string",
                "description": "The normalized reference text (for verification)",
            },
            "normalized_hypothesis": {
                "type": "string",
                "description": "The normalized hypothesis text (for verification)",
            },
            "summary": {
                "type": "string",
                "description": (
                    "A concise, publicly-showable explanation (1-2 sentences, "
                    "plain language) of the semantic errors you counted, or a "
                    "brief statement that there were none. Summarize only the "
                    "verdict and the errors that mattered — do NOT include "
                    "step-by-step working, normalization steps, or alignment "
                    "tables."
                ),
            },
            "errors": {
                "type": "array",
                "description": "List of identified errors",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["substitution", "deletion", "insertion"],
                        },
                        "reference": {
                            "type": "string",
                            "description": "Reference word (null for insertion)",
                        },
                        "hypothesis": {
                            "type": "string",
                            "description": "Hypothesis word (null for deletion)",
                        },
                        "position": {
                            "type": "integer",
                            "description": "Position in alignment",
                        },
                    },
                },
            },
        },
        "required": [
            "substitutions",
            "deletions",
            "insertions",
            "reference_words",
            "summary",
        ],
    },
}


def build_user_prompt(reference: str, prediction: str) -> str:
    """pipecat's per-pair user message, verbatim (from ``evaluate``)."""
    return (
        "Please calculate the Word Error Rate (WER) for this ASR transcription.\n\n"
        "**Reference (ground truth):**\n"
        f"{reference}\n\n"
        "**Hypothesis (ASR transcription):**\n"
        f"{prediction}\n\n"
        "Follow the process: NORMALIZE → ALIGN → COUNT → VERIFY → CALCULATE\n\n"
        "Show your work clearly, then call calculate_wer with your verified counts."
    )
