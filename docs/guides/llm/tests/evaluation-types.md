# LLM Test Evaluation Types

Two ways to evaluate LLM behavior.

## Tool Call Evaluation

Verifies that the LLM calls tools correctly:

```json
{
  "type": "tool_call",
  "tool_calls": [
    {
      "tool": "get_weather",
      "arguments": {"location": "San Francisco"}
    }
  ]
}
```

## Response Evaluation

Uses an LLM judge to evaluate response quality:

```json
{
  "type": "response",
  "criteria": "The assistant should be helpful and concise"
}
```

## Next Steps

[Generate leaderboards](leaderboard.md).
