# calibrate.connections module
"""
Connection types for injecting external agents into calibrate evaluations.

Usage:
    from calibrate.connections import TextAgentConnection

    agent = TextAgentConnection(
        url="https://your-agent.com/chat",
        headers={"Authorization": "Bearer sk-..."},
    )

    # Verify the connection before running evals
    result = asyncio.run(agent.verify())
    result = asyncio.run(agent.verify(messages=[{"role": "user", "content": "Hello"}]))

    # Run LLM tests
    result = asyncio.run(tests.run(agent=agent, test_cases=[...]))

    # Run LLM simulations
    result = asyncio.run(simulations.run(agent=agent, personas=[...], ...))
"""

from dataclasses import dataclass, field
from typing import Optional
import httpx


# Default messages used by verify() when no custom input is provided
_DEFAULT_VERIFY_MESSAGES = [{"role": "user", "content": "Hello, are you there?"}]


@dataclass
class TextAgentConnection:
    """
    Connect to an external text agent via HTTP POST.

    Calibrate sends a fixed request and expects a fixed response format.

    ── Request (POST to ``url``) ────────────────────────────────────────────
        {
            "messages": [
                {"role": "user",      "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user",      "content": "What can you do?"}
            ]
        }

    ── Response (agent must return) ────────────────────────────────────────
        {
            "response":   "The agent's reply text",
            "tool_calls": [{"tool": "function_name", "arguments": {"key": "value"}}]
        }

        Both keys are optional — include whichever applies:
        • Text reply only  → ``{"response": "...", "tool_calls": []}``
        • Tool call only   → ``{"response": null,  "tool_calls": [{...}]}``
        • Both             → ``{"response": "...", "tool_calls": [{...}]}``

        Each tool call may optionally carry an ``output`` field — the result the
        tool actually returned when the agent executed it. It is any JSON value
        and is preserved for display/review only; it never affects evaluation::

            {"tool": "get_weather", "arguments": {"city": "NYC"},
             "output": {"temp": 72, "condition": "sunny"}}

    Use :meth:`verify` to confirm the endpoint is reachable and returns the
    expected format before running a full evaluation.

    Example:
        >>> import asyncio
        >>> from calibrate.connections import TextAgentConnection
        >>> agent = TextAgentConnection(
        ...     url="https://your-agent.com/chat",
        ...     headers={"Authorization": "Bearer sk-..."},
        ... )
        >>> asyncio.run(agent.verify())
    """

    url: str
    """HTTP(S) endpoint to POST the messages array to."""

    headers: Optional[dict] = field(default=None)
    """Optional HTTP headers, e.g. ``{"Authorization": "Bearer sk-..."}``. Default: none."""

    async def verify(
        self,
        messages: Optional[list] = None,
        model: Optional[str] = None,
    ) -> dict:
        """Check the endpoint is reachable and returns the expected format.

        Sends ``messages`` (or a built-in greeting if omitted) to the endpoint
        and validates the response structure.

        Args:
            messages: Custom messages to send, e.g.
                ``[{"role": "user", "content": "Hello"}]``.
                Defaults to a simple greeting when not provided.
            model: Optional model name to include in the request (for verifying
                benchmark mode, e.g. ``"gemma-4-26b-a4b-it"``).

        Returns:
            ``{"ok": True, "sample_output": {"response": "...", "tool_calls": [...]}}``
            on success, or
            ``{"ok": False, "error": "<reason>", "sample_output": ...}`` on failure.
            ``sample_output`` contains whatever the agent returned (may be absent
            if the request never completed).

        Example:
            >>> result = asyncio.run(agent.verify())
            >>> result = asyncio.run(agent.verify(
            ...     messages=[{"role": "user", "content": "What is 2+2?"}]
            ... ))
            >>> # Benchmark verify — checks agent accepts model param
            >>> result = asyncio.run(agent.verify(model="gemma-4-26b-a4b-it"))
        """
        input_messages = messages if messages is not None else _DEFAULT_VERIFY_MESSAGES

        # ── 1. POST to endpoint ──────────────────────────────────────────
        try:
            req_headers = {"Content-Type": "application/json"}
            if self.headers:
                req_headers.update(self.headers)

            body: dict = {"messages": input_messages}
            if model is not None:
                body["model"] = model

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    self.url,
                    json=body,
                    headers=req_headers,
                )
        except httpx.ConnectError as e:
            return {"ok": False, "error": f"Could not connect to endpoint: {e}"}
        except httpx.TimeoutException:
            return {"ok": False, "error": "Request timed out (30s)"}
        except Exception as e:
            return {"ok": False, "error": f"Unexpected error during request: {e}"}

        # ── 2. HTTP status ────────────────────────────────────────────────
        if resp.status_code != 200:
            return {
                "ok": False,
                "error": f"Endpoint returned HTTP {resp.status_code}: {resp.text[:500]}",
            }

        # ── 3. Valid JSON ─────────────────────────────────────────────────
        try:
            data = resp.json()
        except Exception:
            return {
                "ok": False,
                "error": "Response is not valid JSON",
            }

        if not isinstance(data, dict):
            return {
                "ok": False,
                "error": f"Response must be a JSON object, got {type(data).__name__}",
                "sample_output": data,
            }

        # ── 4. At least one expected key ──────────────────────────────────
        has_response = "response" in data
        has_tool_calls = "tool_calls" in data

        if not has_response and not has_tool_calls:
            return {
                "ok": False,
                "error": 'Response JSON must contain "response" and/or "tool_calls"',
                "sample_output": data,
            }

        # ── 5. Type checks ────────────────────────────────────────────────
        if has_response and data["response"] is not None:
            if not isinstance(data["response"], str):
                return {
                    "ok": False,
                    "error": f'"response" must be a string or null, got {type(data["response"]).__name__}',
                    "sample_output": data,
                }

        if has_tool_calls:
            if not isinstance(data["tool_calls"], list):
                return {
                    "ok": False,
                    "error": f'"tool_calls" must be a list, got {type(data["tool_calls"]).__name__}',
                    "sample_output": data,
                }
            for i, tc in enumerate(data["tool_calls"]):
                if not isinstance(tc, dict):
                    return {
                        "ok": False,
                        "error": f'"tool_calls[{i}]" must be an object, got {type(tc).__name__}',
                        "sample_output": data,
                    }
                if "tool" not in tc:
                    return {
                        "ok": False,
                        "error": f'"tool_calls[{i}]" is missing required key "tool"',
                        "sample_output": data,
                    }
                if "arguments" not in tc:
                    return {
                        "ok": False,
                        "error": f'"tool_calls[{i}]" is missing required key "arguments"',
                        "sample_output": data,
                    }
                if not isinstance(tc["arguments"], dict):
                    return {
                        "ok": False,
                        "error": f'"tool_calls[{i}].arguments" must be an object, got {type(tc["arguments"]).__name__}',
                        "sample_output": data,
                    }

        return {
            "ok": True,
            "sample_output": {
                "response": data.get("response"),
                "tool_calls": data.get("tool_calls", []),
            },
        }

    async def call(
        self,
        messages: list,
        model: "Optional[str]" = None,
    ) -> dict:
        """POST a messages array to the agent endpoint and return its output.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            model: Optional model name to include in the request body (for
                benchmarking, e.g. ``"gemma-4-26b-a4b-it"``).

        Returns:
            dict with ``response`` (str | None) and ``tool_calls`` (list) keys.
            Each tool call dict is passed through verbatim, so an optional
            ``output`` field (the tool's own result) is preserved for review.

        Raises:
            RuntimeError: On connection error, timeout, non-200 status, or
                invalid JSON response.
        """
        req_headers = {"Content-Type": "application/json"}
        if self.headers:
            req_headers.update(self.headers)

        body: dict = {"messages": messages}
        if model is not None:
            body["model"] = model

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    self.url,
                    json=body,
                    headers=req_headers,
                )
        except httpx.ConnectError as e:
            raise RuntimeError(f"Could not connect to agent at {self.url}: {e}") from e
        except httpx.TimeoutException:
            raise RuntimeError(f"Agent request timed out (60s): {self.url}") from None
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error calling agent at {self.url}: {e}"
            ) from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"Agent returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(
                f"Agent response is not valid JSON: {resp.text[:500]}"
            ) from None

        return {
            "response": data.get("response"),
            "tool_calls": data.get("tool_calls", []),
        }


__all__ = ["TextAgentConnection"]
