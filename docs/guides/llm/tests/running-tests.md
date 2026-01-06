# Running LLM Tests

Execute LLM test suites.

## Basic Command

```bash
pense llm tests run \
  -c <config.json> \
  -o <output-dir> \
  -m <model>
```

## Example

```bash
pense llm tests run \
  -c ./configs/test-suite.json \
  -o ./out/llm-tests \
  -m gpt-4.1
```

## Next Steps

[Understand evaluation types](evaluation-types.md).
