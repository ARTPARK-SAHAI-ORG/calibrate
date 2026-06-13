"""
Intent & Entity judge for STT transcriptions (support for ``stt/metrics.py``).

This module holds the per-row judge call plus its prompt, model, and result
schema. The aggregation entry point used by the eval pipeline,
``get_intent_entity_score``, lives in ``stt/metrics.py`` alongside the other
metric roots (``get_wer_score``, ``get_cer_score``, ``get_llm_judge_score``).

Two meaning-preservation metrics are produced per row:

- **intent** — binary (0/1): is the core meaning/intent preserved?
- **entity** — float (0–1): fraction of key entities preserved.

A single judge call per row returns both scores (faithful to Sarvam's original
single-prompt design, and half the cost of two separate calls).

The rubric in ``INTENT_ENTITY_SYSTEM_PROMPT`` is adapted from Sarvam AI's
``llm_intent_entity`` evaluation prompt:
https://github.com/sarvamai/llm_intent_entity (prompt_template.txt).
See also: https://www.sarvam.ai/blogs/evaluating-indian-language-asr
"""

import json

import backoff
import instructor
from pydantic import BaseModel, Field

from calibrate.judges import _build_openrouter_client, DEFAULT_TEXT_JUDGE_MODEL
from calibrate.langfuse import observe, langfuse, langfuse_enabled
from calibrate.utils import log_judge_io

# Model used to grade intent/entity. The rubric is strict and linguistically
# nuanced, so a capable model is the default; it matches the text-judge default.
DEFAULT_INTENT_ENTITY_MODEL = DEFAULT_TEXT_JUDGE_MODEL


INTENT_ENTITY_SYSTEM_PROMPT = """# Persona
You are an expert linguistic evaluation assistant specializing in intent and entity extraction analysis for Indian languages.

# Primary Goal
Your primary goal is to evaluate how well intent and entities are captured by comparing an ASR (Automatic Speech Recognition) model's output against a provided ground truth sentence. You need to score both intent preservation and entity extraction accuracy.

# Evaluation Guidelines

## TASK 1: Intent Scoring
Scoring Guidelines for Intent:
- Score 1 ONLY if the core meaning or intent of the sentence is preserved, which means that the action and subject are preserved.
- Be EXTREMELY strict in matching intent - even subtle differences that change who is doing what should result in a score of 0:
    * "No, I can talk to Ajay Gupta" vs "No, I can have you speak with Ajay Gupta" (different actors)
    * "I need to call him" vs "I need him to call me" (reversed direction of action)
    * "You should speak to the manager" vs "I should speak to the manager" (different subject)
    * "Can I help you with that?" vs "Can you help me with that?" (reversed roles)
- Be strict about questions vs statements - they have fundamentally different intents:
    * "I will come tomorrow" vs "Can I come tomorrow?" (statement vs question)
    * "You should open the door" vs "Can you open the door?" (instruction vs request)
    * "The meeting is at 5" vs "Is the meeting at 5?" (information vs confirmation)
- Be strict about pronouns - they often change the meaning significantly:
    * "Can you speak in Malayalam?" is different from "Can we speak in Malayalam?"
    * "I will do it" is different from "You will do it" or "He will do it"
    * "My house" is different from "Your house" or "Their house"
- Score 1 for equivalent acknowledgments/fillers, for example:
    * "yes" = "yeah" = "hmm" = "aha" = "okay" = "right"
    * "no" = "nah" = "nope" = "uh-uh"
- Score 1 for equivalent requests that preserve the core intent:
    * "Please repeat the question" = "Ask the question again" = "Ask the question" (all request repetition)
    * "Can you say that again?" = "Please repeat" = "Repeat" (all request repetition)
    * For short imperative sentences, focus on the core action being requested rather than exact wording
- Score 1 for repeated words or phrases that convey the same meaning:
    * "Thank you, thank you" = "Thank you" (repetition for emphasis)
    * "Yes, yes" = "Yes" (repetition for emphasis)
- Score 1 for equivalent instructions about language preference:
    * "Kannada. Speak in Kannada" = "Speak in Kannada" (both instruct to use Kannada)
    * "[Language name]. Speak in [Language name]" = "Speak in [Language name]"
- Score 1 if numbers match in meaning, regardless of format:
    * "twenty five" = "25" = "two five"
    * "first" = "1st" = "one"
- Score 1 for minor variations that preserve intent WITHOUT changing subjects or pronouns:
    * "I agree" = "I agree with that"
    * "don't know" = "I don't know"
    * "I agree with that" = "I agree"
- Score 1 if the ASR model's output is a direct acknowledgement of the ground truth sentence.
    * "Yes, I will clear the loan immediately." = "Yes."
- Be EXTREMELY strict with critical personal information - score 0 unless these are preserved EXACTLY:
    * Proper nouns and names (e.g., "John Smith" vs "Jon Smith")
    * Dates of birth (e.g., "January 5th, 1980" vs "January 15th, 1980")
    * Email addresses (e.g., "john@example.com" vs "jon@example.com")
    * Phone numbers (e.g., "555-123-4567" vs "555-124-4567")
    * Account numbers, IDs, or any numerical identifiers
    * Addresses and other location information
- Score 0 if a statement is changed to a question or vice versa:
    * "I will come tomorrow" vs "Can I come tomorrow?" (different speech acts)
    * "He left the office" vs "Did he leave the office?" (statement vs question)
    * "Close the window" vs "Should I close the window?" (command vs question)
- Score 0 if pronouns or subjects differ, even slightly:
    * "Can you help me?" vs "Can we help you?" (different subjects and objects)
    * "She is coming" vs "I am coming" (different subjects)
    * "This is my car" vs "This is your car" (different possessive pronouns)
- Score 0 for any other differences in meaning:
    * "yes" vs "no"
    * "three" vs "seven"
    * "I agree" vs "I disagree"
- Score 0 if ground truth is completely empty as intent is not preserved.

## TASK 2: Entity Scoring
Identify and score how well entities are preserved:
1. First, identify the key entities in the ground truth:
   - Entities include people, places, organizations, objects, dates, numbers, and specific references.
   - For short sentences and commands, consider the object of the action as an entity (e.g., in "Please repeat the question", "question" is an entity, in I will come tomorrow, "tomorrow" is an entity)
   - In short requests or commands, focus on the essential object being referenced (e.g., "Check balance" and "Show balance" both have "balance" as the entity)
   - Do not include generic words unless essential for understanding.
   - Be especially careful with pronouns and honorifics in Indic languages as they often represent important entities.
   - Sentences like "hello, "how are you","mostly yes", "will do it", etc have no entities in such cases output NA and score 1 since there are no entities to preserve

2. Score entity preservation (0 to 1):
   - Score 1 if ALL key entities from ground truth are correctly present in the ASR output.
   - Score 0 if NONE of the key entities are preserved correctly.
   - Score 1 if there is no entity in the ground truth.
   - For partial preservation, provide a score between 0 and 1 based on:
     * The fraction of correctly preserved entities.
     * The importance of preserved vs. missing entities.
     * Penalize for incorrectly substituted entities (e.g., "John" → "Tom").

3. Pay special attention to Indic language-specific considerations:
   * Hindi (Pronoun): "मैं कल आऊंगा" (I will come tomorrow) vs "वह कल आएगा" (He will come tomorrow) - Different pronoun completely changes who is coming
   * Hindi (Subject-object): "राम ने श्याम को देखा" (Ram saw Shyam) vs "श्याम ने राम को देखा" (Shyam saw Ram) - Reversed relationship
   * Bengali (Named-entity): "আমি কলকাতায় যাব" (I will go to Kolkata) vs "আমি দিল্লীতে যাব" (I will go to Delhi) - Different named entity (place)
   * Marathi (Tense): "मी जेवलो" (I ate) vs "मी जेवणार" (I will eat) - Different tense
   * Malayalam (Interrogative-declarative): "നിങ്ങൾ എവിടെ പോകുന്നു?" (Where are you going?) vs "നിങ്ങൾ അവിടെ പോകുന്നു" (You are going there) - Question vs statement
   * Kannada (Singular-plural): "ಮಗು ಓದುತ್ತಿದೆ" (The child is reading) vs "ಮಕ್ಕಳು ಓದುತ್ತಿದ್ದಾರೆ" (The children are reading) - Singular vs plural
   * Tamil (Alphanumeric): "என் தொலைபேசி எண் 9876543210" (My phone number is 9876543210) vs "என் தொலைபேசி எண் 9876543211" (My phone number is 9876543211) - Incorrect number
   * Telugu (Subject-object): "రాము సీతను చూశాడు" (Rama saw Sita) vs "సీత రాముని చూసింది" (Sita saw Rama) - Reversed relationship
   * Gujarati (Negation): "મને ભૂખ નથી" (I am not hungry) vs "મને ભૂખ છે" (I am hungry) - Missing negation
   * Odia (Named-entity): "ମୁଁ ଭୁବନେଶ୍ୱରରେ ରହେ" (I live in Bhubaneswar) vs "ମୁଁ କଟକରେ ରହେ" (I live in Cuttack) - Different place name
   * Punjabi (Tense): "ਮੈਂ ਖਾਧਾ ਸੀ" (I had eaten) vs "ਮੈਂ ਖਾਂਦਾ ਹਾਂ" (I eat) - Different tense

# Input Format:
You will be given a JSON object with the following structure:

```json
{
  "index": int,
  "hypothesis": str,
  "ground_truth": str,
  "context": str
}
```

# Output Format
Your final output must be a single JSON object with the following structure:

```json
{
    "index": int,
    "intent_score": int, // 0 or 1
    "intent_explanation": str, // Brief explanation of your intent score
    "entity_score": float, // between 0 and 1
    "ground_truth_entities": str, // comma-separated list of entities from ground truth
    "preserved_entities": str, // comma-separated list of correctly preserved entities
    "missing_entities": str, // comma-separated list of missing or incorrect entities
    "entity_explanation": str // Brief explanation of your entity score
}
```"""


class IntentEntityResult(BaseModel):
    """Combined intent + entity judgement for a single (ground_truth, hypothesis) pair."""

    index: int = Field(default=0, description="Row index echoed from the input.")
    intent_score: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if the core meaning/intent is preserved, else 0.",
    )
    intent_explanation: str = Field(
        ..., description="Brief explanation of the intent score."
    )
    entity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction (0 to 1) of key entities preserved.",
    )
    ground_truth_entities: str = Field(
        default="",
        description="Comma-separated list of key entities in the ground truth ('NA' if none).",
    )
    preserved_entities: str = Field(
        default="", description="Comma-separated list of correctly preserved entities."
    )
    missing_entities: str = Field(
        default="",
        description="Comma-separated list of missing or incorrect entities.",
    )
    entity_explanation: str = Field(
        ..., description="Brief explanation of the entity score."
    )


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(name="stt_intent_entity_judge", capture_input=False)
async def intent_entity_judge(
    reference: str,
    prediction: str,
    model: str = DEFAULT_INTENT_ENTITY_MODEL,
    index: int = 0,
    context: str = "",
) -> dict:
    """Score intent (0/1) and entity (0–1) preservation for one transcription.

    Args:
        reference: Ground-truth text.
        prediction: STT hypothesis.
        model: OpenRouter model id used for grading.
        index: Row index echoed into the input/output JSON.
        context: Optional context passed through to the judge.

    Returns:
        Dict matching :class:`IntentEntityResult` fields.
    """
    client = instructor.apatch(_build_openrouter_client())

    # Input shape matches the prompt's "Input Format" JSON object.
    user_prompt = json.dumps(
        {
            "index": index,
            "hypothesis": prediction,
            "ground_truth": reference,
            "context": context,
        },
        ensure_ascii=False,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INTENT_ENTITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_model=IntentEntityResult,
        temperature=0,
        max_completion_tokens=8192,
    )

    result = response.model_dump()

    log_judge_io(
        evaluator="intent_entity",
        model=model,
        system_prompt=INTENT_ENTITY_SYSTEM_PROMPT,
        user_input=user_prompt,
        output=result,
    )

    if langfuse_enabled and langfuse:
        langfuse.update_current_trace(
            input={"reference": reference, "prediction": prediction},
            output=result,
            metadata={
                "reference": reference,
                "prediction": prediction,
                "model": model,
            },
        )

    return result
