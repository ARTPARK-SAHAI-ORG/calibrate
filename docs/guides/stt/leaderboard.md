# STT Leaderboards

Compare STT providers side-by-side.

## Generate Leaderboard

After running multiple provider evaluations:

```bash
pense stt leaderboard \
  -o ./out \
  -s ./leaderboards
```

## Output

Creates:
- **stt_leaderboard.xlsx**: Excel file with all metrics
- **all_metrics_by_run.png**: Visual comparison chart

## Next Steps

- Try [TTS evaluation](../tts/index.md)
- Explore [LLM testing](../llm/index.md)
