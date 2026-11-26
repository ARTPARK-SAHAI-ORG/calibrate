"""One reply from an agent, with its tool calls run, from a plain chat model."""

import asyncio
import json
from typing import Literal

from openai import AsyncOpenAI

from calibrate_agent.limits import MAX_REPLY_TOKENS
from calibrate_agent.tools import (
    build_tool_specs,
    fill_missing_tool_results,
    make_webhook_call,
)

END_CALL_SPEC = {
    "name": "end_call",
    "description": "End the current call when the conversation is complete.",
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional explanation for why the call should end.",
            }
        },
        "required": [],
    },
}


def _parse_arguments(raw: str | None) -> dict:
    try:
        parsed = json.loads(raw or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _cost(usage) -> float | None:
    cost = getattr(usage, "cost", None)
    if cost is None:
        cost = (getattr(usage, "model_extra", None) or {}).get("cost")
    return cost


async def agent_reply(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict],
    model: str,
    provider: Literal["openrouter", "openai"],
    api_key: str | None,
    *,
    max_tool_rounds: int = 5,
    client: AsyncOpenAI | None = None,
) -> dict:
    """
    Ask the model for the agent's next reply, running any tools it calls and
    asking again with their results, until it answers in text, calls end_call,
    or has used max_tool_rounds rounds of tool calls.

    `messages` is the conversation so far in OpenAI format, without the system
    message. Provider errors are raised to the caller.
    """
    # AsyncOpenAI falls back to OPENAI_API_KEY when given no key, which would
    # spend the server's key, or send it to OpenRouter.
    if client is None and not api_key:
        raise ValueError("agent_reply needs an api_key")
    specs, webhook_configs = build_tool_specs(tools)
    api_tools = [{"type": "function", "function": spec} for spec in [END_CALL_SPEC] + specs]

    owns_client = client is None
    if owns_client:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1" if provider == "openrouter" else None,
        )

    history = [{"role": "system", "content": system_prompt}] + fill_missing_tool_results(
        messages, tools
    )
    new_messages: list[dict] = []
    tool_calls_out: list[dict] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost: float | None = None
    reply = ""

    def result(stopped: str, ended: bool = False) -> dict:
        return {
            "reply": reply,
            "ended": ended,
            "tool_calls": tool_calls_out,
            "new_messages": new_messages,
            "cost": cost,
            **totals,
            "stopped": stopped,
        }

    async def run_tool(name: str, arguments: dict) -> dict:
        if name in webhook_configs:
            return await make_webhook_call(webhook_configs[name], arguments)
        return {"status": "received"}

    try:
        for _ in range(max_tool_rounds):
            response = await client.chat.completions.create(
                model=model,
                messages=history + new_messages,
                tools=api_tools,
                max_completion_tokens=MAX_REPLY_TOKENS,
            )
            usage = response.usage
            if usage is not None:
                for key in totals:
                    totals[key] += getattr(usage, key, 0) or 0
                round_cost = _cost(usage) if provider == "openrouter" else None
                if round_cost is not None:
                    cost = (cost or 0) + round_cost

            message = response.choices[0].message
            reply = message.content or ""
            calls = message.tool_calls or []
            if not calls:
                new_messages.append({"role": "assistant", "content": reply})
                return result("reply")

            new_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in calls
                    ],
                }
            )
            arguments = [_parse_arguments(call.function.arguments) for call in calls]
            outputs = await asyncio.gather(
                *(run_tool(call.function.name, args) for call, args in zip(calls, arguments))
            )
            for call, args, output in zip(calls, arguments, outputs):
                new_messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(output)}
                )
                tool_calls_out.append(
                    {"id": call.id, "tool": call.function.name, "arguments": args, "output": output}
                )

            if any(call.function.name == "end_call" for call in calls):
                return result("end_call", ended=True)

        return result("max_tool_rounds")
    finally:
        if owns_client:
            await client.close()
