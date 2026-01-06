# Output Structure

Understanding Pense's output files.

## Common Output Structure

All evaluations create output directories with:

```
output_dir/
├── results.csv          # Detailed results
├── metrics.json         # Aggregated metrics
├── logs/                # Execution logs
└── [component-specific files]
```

## File Types

### CSV Files

- **results.csv**: Row per test case with metrics
- **leaderboard.csv**: Comparison across providers/runs

### JSON Files

- **metrics.json**: Aggregated statistics
- **transcripts.json**: Conversation transcripts (agent simulations)
- **tool_calls.json**: Tool call history (agent simulations)

### Logs

- **logs/**: Full execution logs for debugging
- **results.log**: Terminal output summary

## Component-Specific Outputs

### STT/TTS

- Audio files (for TTS)
- Transcription results

### Agent Simulations

- `audios/`: Audio files for each turn
- `transcripts.json`: Full conversation
- `tool_calls.json`: Tool invocations
- `stt_outputs.json`: STT results per turn

## Next Steps

- Explore [guides](../guides/stt/index.md) for component-specific details
- Check [examples](../examples/index.md) to see outputs in action
