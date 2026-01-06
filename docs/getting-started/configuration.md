# Configuration

Set up API keys and configure Pense to work with your providers.

## Environment Setup

Pense uses a single `.env` file for all API keys. This file is shared across all modules (STT, TTS, LLM, Agent).

## Create .env File

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Open `.env` in your editor and add your API keys.

## Required API Keys

Add API keys for the providers you want to use:

### STT Providers

```bash
# Deepgram
DEEPGRAM_API_KEY=your_deepgram_key

# OpenAI
OPENAI_API_KEY=your_openai_key

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Sarvam
SARVAM_API_KEY=your_sarvam_key
```

### TTS Providers

```bash
# ElevenLabs
ELEVENLABS_API_KEY=your_elevenlabs_key

# Cartesia
CARTESIA_API_KEY=your_cartesia_key

# OpenAI (same as STT)
OPENAI_API_KEY=your_openai_key

# Google Cloud (same as STT)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

### LLM Providers

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# OpenRouter
OPENROUTER_API_KEY=your_openrouter_key
```

## Security Best Practices

!!! warning "Never commit .env files"
    The `.env` file contains sensitive API keys. Make sure it's in your `.gitignore` file and never commit it to version control.

## Verify Configuration

Test your configuration by running a simple command:

```bash
# This will show available providers
pense stt eval --help
```

## Next Steps

Now that you're configured, run your first evaluation:

[Run Your First Evaluation :material-arrow-right:](first-evaluation.md){ .md-button }
