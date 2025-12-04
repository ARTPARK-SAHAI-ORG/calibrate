from openai import AsyncOpenAI
from pyarrow import system_memory_pool
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field, create_model
import json

load_dotenv()


async def test_response_llm_judge(
    conversation: list[dict], response: str, criteria: str
) -> float:
    client = AsyncOpenAI()

    conversation_as_prompt = "\n".join(
        [f'{msg["role"]}: {msg["content"]}' for msg in conversation if "content" in msg]
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

    system_prompt = """You are a highly accurate evaluator evaluating the response to a conversation.

You will be given a conversation between a user and a human agent along with the response of the human agent to the final user message and an evaluation criteria to use for evaluating the agent's final response. 

You need to evaluate if the response adheres to the evaluation criteria."""

    user_prompt = f"""Chat history: {conversation_as_prompt}\nResponse to evaluation: {response}\nEvaluation criteria: {criteria}"""

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

    return response.output_parsed.model_dump()


def convert_evaluation_criteria_to_prompt(evaluation_criteria: list[dict]) -> str:
    return "\n\n".join(
        [
            f"""**{criterion['name']}**: {criterion['description']}"""
            for criterion in evaluation_criteria
        ]
    )


async def evaluate_simuation(conversation: List[str], evaluation_criteria: list[dict]):
    client = AsyncOpenAI()

    conversation_as_prompt = "\n".join(
        [f'{msg["role"]}: {msg["content"]}' for msg in conversation if "content" in msg]
    )

    class CriterionOutput(BaseModel):
        reasoning: str = Field(
            ...,
            description="Analyse the chat history and give step by step breakdown of whether the assistant's behavior matches the evaluation criterion given; don't repeat the conversation; just give your analysis",
        )
        match: bool = Field(
            ...,
            description="True if the response passes the criterion. False if it does not.",
        )

    def make_output_model(fields: list[str]) -> type[BaseModel]:
        """
        Dynamically create a Pydantic model with fields from a list of evaluation criteria.
        """
        # build dictionary for create_model
        field_definitions: dict[str, tuple[type, any]] = {
            field: (CriterionOutput, ...) for field in fields
        }
        return create_model("Output", **field_definitions)

    Output = make_output_model([criterion["name"] for criterion in evaluation_criteria])

    system_prompt = """You are a highly accurate grader. 

You will be given a conversation between a user and an agent along with an evaluation criteria to use for evaluating the agent's behaviour. 

You need to evaluate if the agent's behaviour adheres to the evaluation criteria."""

    user_prompt = f"""Chat history: {conversation_as_prompt}\nEvaluation criteria: {convert_evaluation_criteria_to_prompt(evaluation_criteria)}"""

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

    return response.output_parsed.model_dump()
