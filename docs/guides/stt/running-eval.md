# Running STT Evaluations

Execute STT evaluations with Pense.

## Basic Command

```bash
pense stt eval \
  -p <provider> \
  -l <language> \
  -i <input-dir> \
  -o <output-dir>
```

## Parameters

- **`-p, --provider`**: Provider name (deepgram, openai, google, sarvam, etc.)
- **`-l, --language`**: Language (english, hindi)
- **`-i, --input-dir`**: Path to input directory
- **`-o, --output-dir`**: Path to output directory
- **`-d, --debug`**: Debug mode (process first 5 files only)

## Example

```bash
pense stt eval \
  -p deepgram \
  -l english \
  -i ./data/stt-test \
  -o ./out/deepgram-results \
  -d
```

## Next Steps

[Understand your results](understanding-results.md).
