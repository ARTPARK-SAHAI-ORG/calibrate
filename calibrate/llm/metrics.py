from pydantic import BaseModel, Field, create_model
import os
from calibrate.langfuse import AsyncOpenAI, observe, langfuse, langfuse_enabled


@observe(
    name="llm_test_llm_judge",
    capture_input=False,
)
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

    system_prompt = """You are a highly accurate evaluator evaluating the response to a conversation.\n\nYou will be given a conversation between a user and a human agent along with the response of the human agent to the final user message and an evaluation criteria to use for evaluating the agent's final response.\n\nYou need to evaluate if the response adheres to the evaluation criteria."""

    user_prompt = f"""`Chat history`:\n\n{conversation_as_prompt}\n\n`Response to evaluation`:\n\n{response}\n\n`Evaluation criteria`:\n\n{criteria}"""

    judge_response = await client.responses.parse(
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

    judge_response = judge_response.output_parsed.model_dump()

    if langfuse_enabled and langfuse:
        langfuse.update_current_span(
            input=user_prompt,
            metadata={
                "input": user_prompt,
                "output": judge_response,
                "system_prompt": system_prompt,
                "output_schema": Output.model_json_schema(),
            },
        )

    return judge_response


def convert_evaluation_criteria_to_prompt(evaluation_criteria: list[dict]) -> str:
    return "\n\n".join(
        [
            f"""**{criterion['name']}**: {criterion['description']}"""
            for criterion in evaluation_criteria
        ]
    )


def format_conversation_with_tool_calls(conversation: list[dict]) -> str:
    """Format conversation including tool calls in a readable format."""
    lines = []
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if content:
            lines.append(f"{role}: {content}")

        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                func_name = func.get("name", "unknown")
                func_args = func.get("arguments", "{}")
                lines.append(f"[Tool Call] {func_name}({func_args})")

    return "\n".join(lines)


@observe(name="simulation_llm_judge")
async def evaluate_simuation(
    conversation: list[dict],
    evaluation_criteria: list[dict],
    agent_system_prompt: str = "",
):
    client = AsyncOpenAI()

    conversation_as_prompt = format_conversation_with_tool_calls(conversation)

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

    agent_instructions_section = ""
    if agent_system_prompt:
        agent_instructions_section = f"""\n\nThe agent was given the following instructions:\n\n<agent_instructions>\n\n{agent_system_prompt}\n\n</agent_instructions>\n\nUse these instructions to understand what the agent was supposed to do and evaluate if the agent followed its instructions correctly."""

    system_prompt = f"""You are a highly accurate grader.\n\nYou will be given a conversation between a user and an agent along with an evaluation criteria to use for evaluating the agent's behaviour.{agent_instructions_section}\n\nYou need to evaluate if the agent's behaviour adheres to the evaluation criteria. Always give your reasoning in english irrespective of the language of the conversation."""

    evaluation_criteria_as_prompt = convert_evaluation_criteria_to_prompt(
        evaluation_criteria
    )

    user_prompt = f"""`Chat history`:\n\n{conversation_as_prompt}\n\n`Evaluation criteria`:\n\n{evaluation_criteria_as_prompt}"""

    response = await client.responses.parse(
        model="gpt-5.2-2025-12-11",
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
        reasoning={"effort": "medium"},
        max_output_tokens=8192,
        store=True,
    )

    response = response.output_parsed.model_dump()

    if langfuse_enabled and langfuse:
        langfuse.update_current_span(
            metadata={
                "input": f"Conversation: {conversation_as_prompt}\nEvaluation criteria: {evaluation_criteria_as_prompt}",
                "output": response,
                "system_prompt": system_prompt,
                "output_schema": Output.model_json_schema(),
            }
        )

    return response
