# Running TTS Evaluations

Execute TTS evaluations with Pense.

## Basic Command

```bash
pense tts eval \
  -p <provider> \
  -l <language> \
  -i <input.csv> \
  -o <output-dir>
```

## Example

```bash
pense tts eval \
  -p elevenlabs \
  -l english \
  -i ./data/tts-input.csv \
  -o ./out/elevenlabs-results
```

## Next Steps

[Understand your results](understanding-results.md).
