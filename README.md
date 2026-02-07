# Calibrate

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

An open-source evaluation framework for voice agents.

[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289da?logo=discord&logoColor=white)](https://discord.gg/9dQB4AngK2)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-Join%20Community-25D366?logo=whatsapp&logoColor=white)](https://chat.whatsapp.com/JygDNcZ943a3VmZDXYMg5Z)

With [Calibrate](https://calibrate.artpark.ai), you can move from slow, manual testing to a fast, automated, and repeatable testing process for your entire voice agent stack:

- `Speech to Text (STT)`: Benchmark multiple providers (Google, Sarvam, ElevenLabs and more) on your dataset across 10+ indic languages using metrics optimised for agentic use cases
- `Text to Speech (TTS)`: Benchmark generated speech by multiple providers automatically using an Audio LLM Judge across 10+ indic languages
- `Text to Text (LLMs)`: Evaluate the response quality and tool calling of your LLMs for multi-turn conversations and find the find LLM for your agent
- `Simulations`: Simulate realistic conversations using realistic user personas and scenarios to test failure modes for your agent including interruptions

Calibrate is built on top of [pipecat](https://github.com/pipecat-ai/pipecat), a framework for building voice agents.

## Installation

```bash
pip install calibrate-agent
```

## Usage

```bash
calibrate              # Interactive main menu
calibrate stt          # Benchmark STT providers
calibrate tts          # Benchmark TTS providers
calibrate llm          # Interactive LLM evaluation
calibrate simulations  # Interactive text or voice simulations
```

- [CLI Documentation](https://calibrate.artpark.ai/docs/cli/overview)

## License

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
