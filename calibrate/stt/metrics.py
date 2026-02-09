from evaluate import load
from typing import List
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import numpy as np
from tqdm.asyncio import tqdm_asyncio
import difflib
from calibrate.langfuse import AsyncOpenAI, observe, langfuse, langfuse_enabled
from pydantic import BaseModel, Field
import backoff

normalizer = BasicTextNormalizer()


def get_wer_score(references: List[str], predictions: List[str]) -> float:
    wer_metric = load("wer")

    references = [normalizer(ref) for ref in references]
    predictions = [
        normalizer(pred) if isinstance(pred, str) else "" for pred in predictions
    ]

    per_row_wer = [
        wer_metric.compute(predictions=[p], references=[r])
        for p, r in zip(predictions, references)
    ]

    return {"score": np.mean(per_row_wer), "per_row": per_row_wer}


def get_string_similarity(references: List[str], predictions: List[str]) -> float:
    similarities = []

    # Use edit distance (Levenshtein distance) to compute similarity between strings
    for reference, prediction in zip(references, predictions):
        seq = difflib.SequenceMatcher(
            None,
            normalizer(reference),
            normalizer(prediction) if isinstance(prediction, str) else "",
        )
        similarities.append(seq.ratio())  # value between 0 and 1

    return {
        "score": np.mean(similarities),
        "per_row": similarities,
    }


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(
    name="stt_llm_judge",
    capture_input=False,
)
async def stt_llm_judge(reference: str, prediction: str) -> float:
    client = AsyncOpenAI()

    class Output(BaseModel):
        reasoning: str = Field(
            ...,
            description="Analyse the inputs on whether they match or not given the guidelines",
        )
        match: bool = Field(
            ..., description="True if the two strings match, otherwise false."
        )

    system_prompt = """You are a highly accurate evaluator evaluating the transcription output of an STT model.

You will be given two strings - one is the source string used to produce an audio and the other is the transcription of that audio.

You need to evaluate if the two strings are the same.

# Important Instructions:
- Check whether the values represented by both the strings match. E.g. if one string says 1,2,3 but the other string says "one, two, three" or "one, 2, three", they should be considered the same as their underlying value is the same. However, if the actual values itself are different, e.g. for the name of a person or address or the value of any other key detail - that difference should be noted.
- Ignore differences like a word being split up into more than 1 word by spaces. Look at whether the values mean the same in both the strings.
- If all the "values" for the strings match, mark it as True. Else, False."""

    user_prompt = f"""Source: {reference}\nTranscription: {prediction}"""

    response = await client.responses.parse(
        model="gpt-4.1-2025-04-14",
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        text_format=Output,
        temperature=0,
        max_output_tokens=2048,
        store=True,
    )

    response = response.output_parsed.model_dump()

    if langfuse_enabled and langfuse:
        langfuse.update_current_trace(
            input={"reference": reference, "prediction": prediction},
            metadata={
                "reference": reference,
                "prediction": prediction,
                "output": response,
            },
        )

    return response


async def get_llm_judge_score(references: List[str], predictions: List[str]) -> float:
    coroutines = []

    for reference, prediction in zip(references, predictions):
        coroutines.append(stt_llm_judge(reference, prediction))

    results = await tqdm_asyncio.gather(
        *coroutines,
        desc="Running STT LLM Judge",
    )

    return {
        "score": np.mean([int(result["match"]) for result in results]),
        "per_row": results,
    }
