# Understanding Agent Outputs

Interpret agent simulation results.

## Output Structure

```
output_dir/
├── simulation_persona_1_scenario_1/
│   ├── audios/
│   ├── transcript.json
│   ├── tool_calls.json
│   ├── metrics.json
│   └── logs/
```

## Key Files

- **transcript.json**: Full conversation
- **tool_calls.json**: Tool invocations
- **metrics.json**: Latency and performance metrics
- **audios/**: Audio files for each turn

## Next Steps

- Explore [examples](../../examples/index.md)
- Check [reference docs](../../reference/cli-reference.md)
