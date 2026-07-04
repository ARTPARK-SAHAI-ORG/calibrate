"""
Prompt + response schema for the intent/entity judge.

Vendored from Sarvam AI's ``llm_intent_entity`` (src/llm_intent_entity/main.py).
The prompt is read from the sibling ``prompt_template.txt`` rather than the
upstream project root.
"""

import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

PROMPT_PATH = Path(__file__).parent / "prompt_template.txt"

try:
    PROMPT_TEMPLATE = PROMPT_PATH.read_text()
except FileNotFoundError:
    raise FileNotFoundError(
        f"prompt_template.txt not found at {PROMPT_PATH}. "
        "Please ensure it exists alongside main.py."
    )


class IntentEntityResponse(BaseModel):
    index: int
    intent_score: int
    intent_explanation: str
    entity_score: float
    ground_truth_entities: str
    preserved_entities: str
    missing_entities: str
    entity_explanation: str


def build_prompt(item: Dict[str, Any]) -> str:
    """Build prompt for intent entity evaluation"""
    prompt = PROMPT_TEMPLATE + "\n\n**INPUT:**\n"
    json_object = {
        "index": item["index"],
        "hypothesis": item["hypothesis"],
        "ground_truth": item["ground_truth"],
        "context": item.get("context", ""),
    }
    prompt += json.dumps(json_object, indent=2, ensure_ascii=False)
    return prompt
