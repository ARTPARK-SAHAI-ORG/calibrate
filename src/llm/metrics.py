from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()


async def test_response_llm_judge(
    conversation: list[dict], response: str, criteria: str
) -> float:
    client = AsyncOpenAI()

    conversation_as_prompt = "\n".join(
        [f'{msg["role"]}: {msg["content"]}' for msg in conversation]
    )

    class Output(BaseModel):
        reasoning: str = Field(
            ...,
            description="Analyse the response and give step by step breakdown of whether it matches the criteria given; don't repeat the conversation or the response to be evaluated; just give your analysis",
        )
        match: bool = Field(
            ...,
            description="True if the response passes the criteria. False if it does not.",
        )

    response = await client.responses.parse(
        model="gpt-4.1-2025-04-14",
        prompt={
            "id": "pmpt_69171459e94c8190ba2b33ff842f642e0061d85591099c58",
            "version": "3",
            "variables": {
                "conversation": conversation_as_prompt,
                "response": response,
                "criteria": criteria,
            },
        },
        text_format=Output,
        temperature=0,
        max_output_tokens=2048,
        store=True,
    )

    return response.output_parsed.model_dump()
