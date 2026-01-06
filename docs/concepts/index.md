# Core Concepts

Understand the fundamental concepts behind Pense.

## Overview

Pense is designed around the principle of **component isolation** - testing each part of your voice agent pipeline independently before testing the whole system.

## Key Concepts

- [Evaluation Workflow](evaluation-workflow.md) - How evaluations work in Pense
- [Metrics](metrics.md) - Understanding the metrics Pense calculates
- [Providers](providers.md) - Supported providers and their capabilities
- [Output Structure](output-structure.md) - Understanding Pense's output files

## Component Isolation

Voice agents typically consist of three main components:

1. **STT (Speech-to-Text)**: Converts audio to text
2. **LLM (Large Language Model)**: Processes text and generates responses
3. **TTS (Text-to-Speech)**: Converts text back to audio

Pense allows you to test each component independently, making it easier to identify issues and optimize performance.

## Next Steps

- Learn about the [evaluation workflow](evaluation-workflow.md)
- Understand [metrics](metrics.md) and how to interpret them
- Explore [provider options](providers.md)
