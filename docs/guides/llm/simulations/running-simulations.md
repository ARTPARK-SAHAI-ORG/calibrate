# Running LLM Simulations

Execute LLM simulations.

## Basic Command

```bash
pense llm simulations run \
  -c <config.json> \
  -o <output-dir> \
  -m <model>
```

## Example

```bash
pense llm simulations run \
  -c ./configs/simulation.json \
  -o ./out/simulations \
  -m gpt-4.1
```

## Next Steps

[Generate leaderboards](leaderboard.md).
