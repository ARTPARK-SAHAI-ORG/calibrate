# Metrics

Understanding the metrics Pense calculates.

## STT Metrics

- **WER (Word Error Rate)**: Percentage of words incorrectly transcribed (lower is better)
- **String Similarity**: Similarity score between ground truth and transcription (0-1, higher is better)
- **LLM Judge Score**: Pass/fail evaluation by LLM judge
- **Latency**: Time to first byte (TTFB) and processing time

## TTS Metrics

- **Latency**: Time to first byte (TTFB) and processing time
- **Audio Quality**: Subjective evaluation (when available)

## LLM Metrics

- **Pass Rate**: Percentage of tests that passed
- **Tool Call Accuracy**: Correctness of tool calls
- **Response Quality**: LLM judge evaluation scores

## Understanding Metrics

Metrics help you:
- Compare providers objectively
- Identify performance bottlenecks
- Make informed decisions about which providers to use

## Next Steps

- Learn about [providers](providers.md) and their capabilities
- Understand [output structure](output-structure.md)
