# Pense

An open-source simulation and testing framework for voice agents.

With Pense, you can move from a slow, manual testing to a fast, automated, and repeatable testing process:

- **Evaluate different components** of your voice agents across multiple vendors in isolation
- **Create comprehensive test suites** that verify both conversational responses and tool usage
- **Simulate entire conversations** spanning thousands of scenarios across multiple user personas

Pense is built on top of [pipecat](https://github.com/pipecat-ai/pipecat), a framework for building voice agents.

## What Pense Does

Pense helps you test and evaluate voice agents by breaking down the pipeline into testable components:

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│   STT   │ --> │   LLM   │ --> │   TTS   │
│(Speech) │     │(Language│     │(Speech) │
│  to     │     │ Model)  │     │  to     │
│  Text)  │     │         │     │  Text)  │
└─────────┘     └─────────┘     └─────────┘
```

You can evaluate each component independently or test the entire pipeline together.

## Main Use Cases

### 1. **Compare STT/TTS Providers**
Evaluate different speech-to-text and text-to-speech providers to find the best fit for your use case. Compare accuracy, latency, and cost across multiple vendors.

### 2. **Test LLM Behavior**
Create comprehensive test suites to verify that your LLM responds correctly, calls tools appropriately, and handles edge cases as expected.

### 3. **Simulate Voice Agent Conversations**
Run automated simulations with different user personas and scenarios to test your voice agent at scale before deployment.

## Quick Start

Ready to get started? Follow our quick start guide:

[Get Started :material-arrow-right:](getting-started/index.md){ .md-button .md-button--primary }

## Key Features

- **Component Isolation**: Test STT, TTS, and LLM components independently
- **Multi-Provider Support**: Evaluate across multiple vendors (Deepgram, OpenAI, Google, ElevenLabs, and more)
- **Automated Testing**: Run thousands of test scenarios automatically
- **Comprehensive Metrics**: Get detailed metrics including accuracy, latency, and quality scores
- **Leaderboards**: Compare providers side-by-side with visual charts and reports

## Learn More

- [Installation Guide](getting-started/installation.md) - Set up Pense in minutes
- [Concepts](concepts/index.md) - Understand how Pense works
- [Examples](examples/index.md) - See Pense in action
- [Reference](reference/cli-reference.md) - Complete API reference

---

**Built with ❤️ for the voice agent community**
