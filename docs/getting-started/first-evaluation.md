# Your First Evaluation

Run your first STT evaluation to see Pense in action.

## Quick Start: STT Evaluation

Let's run a simple STT evaluation using the sample data provided with Pense.

### Step 1: Prepare Sample Data

Pense includes sample data you can use for testing. It's located at:

```
pense/stt/examples/sample_input/
```

### Step 2: Run the Evaluation

Run an STT evaluation with a provider of your choice:

```bash
pense stt eval \
  -p deepgram \
  -l english \
  -i pense/stt/examples/sample_input \
  -o ./out/stt-test \
  -d
```

**Parameters explained:**
- `-p deepgram`: Provider to use (deepgram, openai, google, sarvam, etc.)
- `-l english`: Language (english or hindi)
- `-i`: Input directory containing audio files and CSV
- `-o`: Output directory for results
- `-d`: Debug mode (only processes first 5 files)

### Step 3: View Results

After the evaluation completes, check the output directory:

```bash
ls -la ./out/stt-test/
```

You'll find:
- `results.csv` - Detailed results for each audio file
- `metrics.json` - Aggregated metrics and latency statistics
- `logs/` - Full execution logs

### Step 4: Understand the Results

Open `results.csv` to see:
- **WER** (Word Error Rate): Lower is better
- **String Similarity**: Higher is better (0-1 scale)
- **LLM Judge Score**: Pass/fail evaluation

Check `metrics.json` for latency metrics:
- **TTFB** (Time to First Byte): How quickly the provider responds
- **Processing Time**: Total processing time

## What's Next?

Now that you've run your first evaluation:

- [Learn about STT Evaluation](../guides/stt/index.md) - Deep dive into STT testing
- [Explore Concepts](../concepts/index.md) - Understand how Pense works
- [See More Examples](../examples/index.md) - Try other evaluation types

## Troubleshooting

If you encounter issues:

- **API Key Errors**: Check your `.env` file configuration
- **File Not Found**: Verify the input directory path
- **Provider Errors**: Ensure your API key is valid and has credits

For more help, see the [Troubleshooting](../troubleshooting/index.md) section.
