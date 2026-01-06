# Understanding STT Results

Interpret STT evaluation outputs.

## Output Files

### results.csv

Contains per-file results:

```csv
id,gt,pred,wer,string_similarity,llm_judge_score,llm_judge_reasoning
audio_1,"Hi","Hi",0.0,1.0,True,"Perfect match"
```

- **wer**: Word Error Rate (lower is better)
- **string_similarity**: Similarity score 0-1 (higher is better)
- **llm_judge_score**: Pass/fail evaluation

### metrics.json

Aggregated statistics:

```json
{
  "wer": 0.13,
  "string_similarity": 0.88,
  "llm_judge_score": 1.0,
  "metrics": [...]
}
```

## Interpreting Metrics

- **WER < 0.1**: Excellent accuracy
- **WER 0.1-0.2**: Good accuracy
- **WER > 0.2**: May need improvement

## Next Steps

[Generate leaderboards](leaderboard.md) to compare providers.
