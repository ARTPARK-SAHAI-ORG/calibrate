# Creating LLM Tests

Set up test configurations for LLM evaluation.

## Test Configuration Structure

```json
{
  "system_prompt": "You are a helpful assistant.",
  "tools": [...],
  "test_cases": [
    {
      "history": [...],
      "evaluation": {
        "type": "tool_call",
        "tool_calls": [...]
      }
    }
  ]
}
```

## Evaluation Types

- **tool_call**: Verify specific tool calls
- **response**: Evaluate response quality with LLM judge

## Next Steps

[Run your tests](running-tests.md).
