from evaluate import load
from typing import List
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import numpy as np
from tqdm.asyncio import tqdm_asyncio
import difflib
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import backoff

load_dotenv(".env", override=True)

normalizer = BasicTextNormalizer()


def get_wer_score(references: List[str], predictions: List[str]) -> float:
    wer_metric = load("wer")

    references = [normalizer(ref) for ref in references]
    predictions = [normalizer(pred) for pred in predictions]

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
            None, normalizer(reference), normalizer(prediction)
        )
        similarities.append(seq.ratio())  # value between 0 and 1

    return {
        "score": np.mean(similarities),
        "per_row": similarities,
    }


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
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

    response = await client.responses.parse(
        model="gpt-4.1-2025-04-14",
        prompt={
            "id": "pmpt_6911a8348998819081b6c12f5c3025f909e4b4db654ec487",
            "version": "1",
            "variables": {
                "source": reference,
                "transcription": prediction,
            },
        },
        text_format=Output,
        temperature=0,
        max_output_tokens=2048,
        store=True,
    )

    return response.output_parsed.model_dump()


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
