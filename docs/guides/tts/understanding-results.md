# Understanding TTS Results

Interpret TTS evaluation outputs.

## Output Files

### results.csv

```csv
text,audio_path
hello world,./out/audios/1.wav
```

### metrics.json

Latency metrics:

```json
[
  {
    "metric_name": "ttfb",
    "mean": 0.35,
    "std": 0.03
  }
]
```

## Next Steps

[Generate leaderboards](leaderboard.md).
