from evaluate import load
from typing import List
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import numpy as np
import instructor
from tqdm.asyncio import tqdm_asyncio
from pydantic import BaseModel, Field
import base64
import backoff
from calibrate.langfuse import (
    AsyncOpenAI,
    observe,
    langfuse,
    langfuse_enabled,
    create_langfuse_audio_media,
)

normalizer = BasicTextNormalizer()


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(
    name="tts_llm_judge",
    capture_input=False,
    capture_output=False,
)
async def tts_llm_judge(audio_path: str, reference_text: str) -> float:
    client = instructor.apatch(AsyncOpenAI())
    # client = instructor.from_provider("openai/gpt-4o-audio-preview", async_client=True)

    class Output(BaseModel):
        reasoning: str = Field(
            ...,
            description="Step-by-step analysis of what is said in the audio and how it compares with the given text.",
        )
        match: bool = Field(
            ..., description="Indicates whether the audio matches the provided text."
        )

    system_prompt = """You are a highly accurate evaluator evaluating the audio output of a TTs model.\n\nYou will be given the audio and the text that should have been spoken in the audio.\n\nYou need to evaluate if the text is easily understandable from the audio."""

    response = await client.chat.completions.create(
        model="gpt-audio-2025-08-28",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Reference text: {reference_text}\n\nAudio:",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(
                                open(audio_path, "rb").read()
                            ).decode("utf-8"),
                            "format": "wav",
                        },
                    },
                ],
            },
        ],
        response_model=Output,
        modalities=["text"],
        temperature=0,
        max_completion_tokens=8192,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        store=True,
    )

    response = response.model_dump()

    if langfuse_enabled and langfuse:
        audio_media = create_langfuse_audio_media(audio_path)
        langfuse.update_current_trace(
            input={"audio": audio_media, "reference_text": reference_text},
            output=response,
            metadata={
                "input": f"Reference text: {reference_text}",
                "output": response,
                "system_prompt": system_prompt,
                "output_schema": Output.model_json_schema(),
            },
        )

    return response


async def get_tts_llm_judge_score(
    audio_paths: List[str], reference_texts: List[str]
) -> float:
    coroutines = []

    for audio_path, reference_text in zip(audio_paths, reference_texts):
        coroutines.append(tts_llm_judge(audio_path, reference_text))

    results = await tqdm_asyncio.gather(
        *coroutines,
        desc="Running TTS LLM Judge",
    )

    return {
        "score": np.mean([int(result["match"]) for result in results]),
        "per_row": results,
    }
