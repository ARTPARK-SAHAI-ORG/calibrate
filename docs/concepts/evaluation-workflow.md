# Evaluation Workflow

How evaluations work in Pense.

## Overview

Pense follows a simple workflow: **Input → Process → Output**

## Workflow Steps

1. **Prepare Input Data**: Organize your test data (audio files, CSV files, or JSON configs)
2. **Run Evaluation**: Execute the evaluation command
3. **Analyze Results**: Review metrics, logs, and output files

## Component-Specific Workflows

### STT/TTS Evaluations

- Input: Audio files + ground truth data
- Process: Send audio to provider, get transcription/synthesis
- Output: Accuracy metrics, latency data, comparison results

### LLM Evaluations

- Input: Test cases or simulation configs
- Process: Run LLM with test inputs, evaluate responses
- Output: Pass/fail results, evaluation scores, detailed logs

### Agent Simulations

- Input: Personas, scenarios, agent configuration
- Process: Simulate conversations between agent and personas
- Output: Transcripts, tool calls, evaluation results

## Next Steps

Learn about the [metrics](metrics.md) Pense calculates.
