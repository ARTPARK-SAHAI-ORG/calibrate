"""Tool definitions and webhook calls, with no pipecat dependency."""

import asyncio
from typing import Any

import aiohttp
from loguru import logger


def _build_param_property(param: dict) -> dict:
    """Build a property dict for a single parameter."""
    prop = {
        "type": param["type"],
        "description": param["description"],
    }
    if "items" in param:
        prop["items"] = param["items"]
    if "enum" in param:
        prop["enum"] = param["enum"]
    return prop


def build_tool_specs(tools: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """
    Build OpenAI-style function specs from tool definitions.

    Supports two tool types:
    - structured_output (default): Parameters at top level in 'parameters' array
    - webhook: Parameters in nested 'webhook.queryParameters' and 'webhook.body.parameters'

    Returns:
        tuple of (list of {"name", "description", "parameters"} dicts,
        dict of webhook configs keyed by tool name)

    Raises:
        ValueError: If a webhook tool is missing its url or method
    """
    specs = []
    webhook_configs = {}

    for tool in tools:
        properties = {}
        required = []

        if tool.get("type") == "webhook":
            webhook = tool.get("webhook", {})

            for field in ["url", "method"]:
                if field not in webhook:
                    raise ValueError(
                        f"Webhook tool '{tool['name']}' is missing required '{field}' field"
                    )

            webhook_configs[tool["name"]] = {
                "url": webhook["url"],
                "method": webhook["method"],
                "headers": webhook.get("headers", []),
                "timeout": webhook.get("timeout", 20),
            }

            query_properties = {}
            query_required = []
            for param in webhook.get("queryParameters", []):
                query_properties[param["id"]] = _build_param_property(param)
                if param.get("required"):
                    query_required.append(param["id"])

            body_properties = {}
            body_required = []
            if "body" in webhook and "parameters" in webhook["body"]:
                for param in webhook["body"]["parameters"]:
                    body_properties[param["id"]] = _build_param_property(param)
                    if param.get("required"):
                        body_required.append(param["id"])

            if query_properties:
                properties["query"] = {
                    "type": "object",
                    "description": "Query parameters for the webhook request",
                    "properties": query_properties,
                    "required": query_required,
                }
                if query_required:
                    required.append("query")
            if body_properties:
                properties["body"] = {
                    "type": "object",
                    "description": webhook.get("body", {}).get(
                        "description", "Request body parameters"
                    ),
                    "properties": body_properties,
                    "required": body_required,
                }
                if body_required:
                    required.append("body")
        else:
            for parameter in tool.get("parameters", []):
                if parameter.get("required"):
                    required.append(parameter["id"])
                properties[parameter["id"]] = _build_param_property(parameter)

        specs.append(
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        )

    return specs, webhook_configs


def get_webhook_tool_names(tools: list[dict]) -> set:
    """Names of the webhook tools in the tools configuration."""
    return {tool["name"] for tool in tools if tool.get("type") == "webhook"}


def fill_missing_tool_results(chat_history: list[dict], tools: list[dict]) -> list[dict]:
    """
    Add a {"status": "received"} tool response after every non-webhook tool call
    that has no response in the conversation. Webhook tool calls are left alone.
    """
    webhook_tool_names = get_webhook_tool_names(tools)

    existing_tool_response_ids = {
        message["tool_call_id"]
        for message in chat_history
        if message.get("role") == "tool" and "tool_call_id" in message
    }

    processed_history = []
    for message in chat_history:
        processed_history.append(message)

        if message.get("role") == "assistant" and "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                tool_call_id = tool_call.get("id")
                tool_name = tool_call.get("function", {}).get("name")

                if tool_name in webhook_tool_names:
                    continue

                if tool_call_id in existing_tool_response_ids:
                    continue

                processed_history.append(
                    {
                        "role": "tool",
                        "content": '{"status": "received"}',
                        "tool_call_id": tool_call_id,
                    }
                )
                logger.info(
                    f"Inserted tool response for structured output tool '{tool_name}' "
                    f"(tool_call_id: {tool_call_id})"
                )

    return processed_history


async def make_webhook_call(
    webhook_config: dict,
    arguments: dict,
) -> dict[str, Any]:
    """
    Make an HTTP webhook call with the provided configuration and arguments.

    Args:
        webhook_config: Dict containing url, method, headers, and timeout
        arguments: Dict containing 'query' and/or 'body' parameters from LLM

    Returns:
        Dict with 'status', 'status_code', and 'response' (or 'error')
    """
    url = webhook_config["url"]
    method = webhook_config["method"].upper()
    headers_list = webhook_config["headers"]
    timeout = webhook_config.get("timeout", 20)

    headers = {header["name"]: header["value"] for header in headers_list}

    query_params = arguments.get("query", {})
    body = arguments.get("body", {})

    # Header values can hold API keys, so only their names are logged.
    logger.info(
        f"Making webhook call:\n"
        f"  method: {method}\n"
        f"  url: {url}\n"
        f"  headers: {list(headers)}\n"
        f"  query: {query_params}\n"
        f"  body: {body}\n"
        f"  timeout: {timeout}s"
    )

    try:
        async with aiohttp.ClientSession() as session:
            request_kwargs = {
                "url": url,
                "headers": headers,
                "params": query_params if query_params else None,
                "timeout": aiohttp.ClientTimeout(total=timeout),
            }

            if method in ["POST", "PUT", "PATCH"] and body:
                request_kwargs["json"] = body

            async with session.request(method, **request_kwargs) as response:
                status_code = response.status
                try:
                    response_data = await response.json()
                except Exception:
                    response_data = await response.text()

                logger.info(
                    f"Webhook response: status={status_code}, response={response_data}"
                )

                return {
                    "type": "webhook_response",
                    "status": "success" if 200 <= status_code < 300 else "error",
                    "status_code": status_code,
                    "response": response_data,
                }

    except asyncio.TimeoutError:
        logger.error(f"Webhook call timed out after {timeout}s")
        return {
            "type": "webhook_response",
            "status": "error",
            "error": f"Request timed out after {timeout}s",
        }
    except aiohttp.ClientError as e:
        logger.error(f"Webhook call failed: {e}")
        return {
            "type": "webhook_response",
            "status": "error",
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Unexpected error during webhook call: {e}")
        return {
            "type": "webhook_response",
            "status": "error",
            "error": str(e),
        }
