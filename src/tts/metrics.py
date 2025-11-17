from evaluate import load
from typing import List
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import numpy as np
from tqdm.asyncio import tqdm_asyncio
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import base64
import backoff

load_dotenv(".env", override=True)

normalizer = BasicTextNormalizer()


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
async def tts_llm_judge(audio_path: str, reference_text: str) -> float:
    client = AsyncOpenAI()

    response = await client.chat.completions.create(
        model="gpt-audio-2025-08-28",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a highly accurate evaluator evaluating the audio output of a TTs model.\n\nYou will be given the audio and the text that should have been spoken in the audio.\n\nYou need to evaluate if the text is easily understandable from the audio. ",
                    }
                ],
            },
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
        modalities=["text"],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "evaluation_result",
                    "description": "Provides a step-by-step analysis of the audio's content and determines if it matches the given text.",
                    "parameters": {
                        "type": "object",
                        "required": ["reasoning", "match"],
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "Step-by-step analysis of what is said in the audio and how it compares with the given text.",
                            },
                            "match": {
                                "type": "boolean",
                                "description": "Indicates whether the audio matches the provided text.",
                            },
                        },
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
        temperature=0,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        store=True,
    )

    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)


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
