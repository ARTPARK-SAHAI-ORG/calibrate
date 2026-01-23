---
description: "Description of the project"
alwaysApply: true
---

# Pense

**Open-Source Voice Agent Simulation and Testing Framework**

---

## What is Pense?

Pense is an open-source Python framework for building, testing, and evaluating **voice-based AI agents**. It provides comprehensive tools to move from slow, manual testing to fast, automated, and repeatable testing processes.

The framework enables:

- **Component-level testing** - Evaluate STT, TTS, and LLM providers in isolation
- **LLM unit tests** - Verify agent behavior with deterministic test cases
- **End-to-end simulations** - Run automated conversations with simulated users
- **Benchmarking** - Compare performance across different AI providers

Pense is built on top of [pipecat](https://github.com/pipecat-ai/pipecat), a framework for building voice agents.

---

## Project Structure

```
/
├── pense/                    # Main Python package
│   ├── __init__.py
│   ├── cli.py               # CLI entry point
│   ├── utils.py             # Shared utilities
│   ├── stt/                 # Speech-to-Text evaluation module
│   │   ├── __init__.py      # Public API: eval(), leaderboard()
│   │   ├── eval.py          # STT evaluation implementation
│   │   ├── leaderboard.py   # STT leaderboard generation
│   │   ├── metrics.py       # STT metrics (WER, string similarity, LLM judge)
│   │   └── examples/        # Sample inputs/outputs
│   ├── tts/                 # Text-to-Speech evaluation module
│   │   ├── __init__.py      # Public API: eval(), leaderboard()
│   │   ├── eval.py          # TTS evaluation implementation
│   │   ├── leaderboard.py   # TTS leaderboard generation
│   │   ├── metrics.py       # TTS metrics (LLM judge, TTFB, processing time)
│   │   └── examples/        # Sample inputs/outputs
│   ├── llm/                 # LLM evaluation module
│   │   ├── __init__.py      # Public API: tests, simulations
│   │   ├── run_tests.py     # LLM test runner
│   │   ├── run_simulation.py # LLM simulation runner
│   │   ├── tests_leaderboard.py
│   │   ├── simulation_leaderboard.py
│   │   ├── metrics.py       # LLM evaluation metrics
│   │   └── examples/        # Sample configs and outputs
│   ├── agent/               # Voice agent simulation module
│   │   ├── __init__.py      # Public API: simulation, STTConfig, TTSConfig, LLMConfig
│   │   ├── bot.py           # Voice agent pipeline
│   │   ├── test.py          # Interactive agent testing
│   │   ├── run_simulation.py # Voice simulation runner
│   │   └── examples/        # Sample configs and outputs
│   └── integrations/        # Third-party provider integrations
│       └── smallest/        # Smallest AI STT/TTS integration
├── docs/                    # Mintlify documentation
│   ├── docs.json           # Navigation and theme config
│   ├── getting-started/
│   ├── quickstart/
│   ├── core-concepts/
│   ├── python-sdk/
│   ├── cli/
│   └── integrations/
├── pyproject.toml          # Package configuration
├── requirements-docs.txt   # Documentation dependencies
├── uv.lock                 # Dependency lockfile
└── README.md               # Project documentation
```

---

## Architecture Overview

### Module Design

Pense is organized into four main modules, each providing both a Python API and CLI commands:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PENSE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│   │   pense.stt     │  │   pense.tts     │  │   pense.llm     │        │
│   │                 │  │                 │  │                 │        │
│   │  eval()         │  │  eval()         │  │  tests.run()    │        │
│   │  leaderboard()  │  │  leaderboard()  │  │  tests.leaderboard()    │
│   │                 │  │                 │  │  simulations.run()       │
│   │                 │  │                 │  │  simulations.leaderboard()│
│   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘        │
│            │                    │                    │                  │
│            └──────────────────┬─┴────────────────────┘                  │
│                               │                                         │
│                    ┌──────────▼──────────┐                              │
│                    │   pense.agent       │                              │
│                    │                     │                              │
│                    │  simulation.run()   │  Full STT → LLM → TTS        │
│                    │  simulation.run_single()  pipeline testing         │
│                    └─────────────────────┘                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Voice Agent Pipeline

A voice agent processes conversations through this pipeline:

```
User Speech → [STT] → Text → [LLM] → Response Text → [TTS] → Agent Speech
                              ↓
                        [Tool Calls]
                              ↓
                       External APIs
```

Pense allows testing and benchmarking each component individually or end-to-end.

---

## Key Concepts

### 1. Speech-to-Text (STT) Evaluation

Evaluates STT providers by transcribing audio files and comparing against ground truth.

**Supported Providers:** deepgram, openai, cartesia, google, gemini, sarvam, elevenlabs, smallest, groq

**Metrics:**
- **WER (Word Error Rate):** Measures transcription accuracy
- **String Similarity:** Character-level similarity score
- **LLM Judge:** AI-based evaluation of semantic accuracy

**Input Structure:**
```
input_dir/
├── stt.csv          # id,text pairs
└── audios/
    ├── audio_1.wav
    └── audio_2.wav
```

**Output Structure:**
```
output_dir/provider/
├── results.csv      # Per-audio results with metrics
├── metrics.json     # Aggregated metrics
└── results.log      # Terminal output
```

### 2. Text-to-Speech (TTS) Evaluation

Evaluates TTS providers by synthesizing speech and measuring quality.

**Supported Providers:** cartesia, openai, groq, google, elevenlabs, sarvam

**Metrics:**
- **LLM Judge:** AI evaluation of pronunciation accuracy
- **TTFB (Time to First Byte):** Latency measurement
- **Processing Time:** Total synthesis time

**Input:** CSV file with `id,text` columns

**Output Structure:**
```
output_dir/provider/
├── audios/          # Generated audio files
├── results.csv      # Per-text results
├── metrics.json     # Aggregated metrics
└── results.log      # Terminal output
```

### 3. LLM Tests

Unit tests for LLM behavior verification.

**Test Types:**
- **Tool Call Tests:** Verify the LLM calls the correct tools with correct arguments
- **Response Tests:** Verify the LLM response meets criteria (via LLM judge)

**Test Case Structure:**
```python
{
    "history": [
        {"role": "assistant", "content": "Hello! What is your name?"},
        {"role": "user", "content": "Aman Dalmia"}
    ],
    "evaluation": {
        "type": "tool_call",  # or "response"
        "tool_calls": [{"tool": "plan_next_question", "arguments": {...}}]
        # or "criteria": "The assistant should allow skipping"
    },
    "settings": {"language": "english"}  # optional
}
```

**Supported Providers:** openai, openrouter

### 4. LLM Simulations

Automated text-only conversations between two LLMs (agent + simulated user).

**Key Components:**
- **Personas:** Define simulated user characteristics (age, personality, speaking style)
- **Scenarios:** Define conversation goals/tasks
- **Evaluation Criteria:** Define success metrics

**Output per Simulation:**
```
simulation_persona_N_scenario_M/
├── transcript.json          # Full conversation
├── evaluation_results.csv   # Per-criterion results
├── config.json             # Persona + scenario used
└── logs/                   # Pipeline logs
```

### 5. Voice Agent Simulations

Full end-to-end voice pipeline testing with STT, LLM, and TTS components.

**Additional Features:**
- **Interruption Sensitivity:** Simulate users interrupting mid-sentence (none/low/medium/high)
- **Audio Recording:** All turns saved as WAV files
- **STT Evaluation:** Compare transcribed speech against intended user messages

**Output per Simulation:**
```
simulation_persona_N_scenario_M/
├── audios/
│   ├── 0_user.wav
│   ├── 1_bot.wav
│   └── ...
├── transcript.json
├── evaluation_results.csv   # Includes latency metrics + STT judge
├── stt_results.csv         # Per-turn STT evaluation
├── metrics.json            # Latency traces
├── tool_calls.json         # Chronological tool calls
├── config.json
├── conversation.wav        # Combined full conversation
└── logs/
```

---

## Usage Patterns

### Python SDK

```python
import asyncio
from pense.stt import eval as stt_eval, leaderboard as stt_leaderboard
from pense.tts import eval as tts_eval, leaderboard as tts_leaderboard
from pense.llm import tests, simulations
from pense.agent import simulation, STTConfig, TTSConfig, LLMConfig

# STT Evaluation
asyncio.run(stt_eval(
    provider="deepgram",
    language="english",
    input_dir="./data",
    output_dir="./out"
))
stt_leaderboard(output_dir="./out", save_dir="./leaderboard")

# TTS Evaluation
asyncio.run(tts_eval(
    provider="google",
    language="english",
    input="./data/texts.csv",
    output_dir="./out"
))
tts_leaderboard(output_dir="./out", save_dir="./leaderboard")

# LLM Tests
asyncio.run(tests.run(
    system_prompt="You are a helpful assistant...",
    tools=[...],
    test_cases=[...],
    model="openai/gpt-4.1",
    provider="openrouter",
    output_dir="./out"
))
tests.leaderboard(output_dir="./out", save_dir="./leaderboard")

# LLM Simulations
asyncio.run(simulations.run(
    system_prompt="You are a helpful nurse...",
    tools=[...],
    personas=[{"characteristics": "...", "gender": "female", "language": "english"}],
    scenarios=[{"description": "User completes the form"}],
    evaluation_criteria=[{"name": "completeness", "description": "..."}],
    model="openai/gpt-4.1",
    provider="openrouter",
    output_dir="./out"
))
simulations.leaderboard(output_dir="./out", save_dir="./leaderboard")

# Voice Agent Simulations
asyncio.run(simulation.run(
    system_prompt="You are a helpful nurse...",
    tools=[...],
    personas=[{
        "characteristics": "...",
        "gender": "female",
        "language": "english",
        "interruption_sensitivity": "medium"
    }],
    scenarios=[{"description": "..."}],
    evaluation_criteria=[{"name": "completeness", "description": "..."}],
    stt=STTConfig(provider="google"),
    tts=TTSConfig(provider="google"),
    llm=LLMConfig(provider="openrouter", model="openai/gpt-4.1"),
    output_dir="./out"
))
```

### CLI

```bash
# STT Evaluation
pense stt eval -p deepgram -l english -i ./data -o ./out
pense stt leaderboard -o ./out -s ./leaderboard

# TTS Evaluation
pense tts eval -p google -l english -i ./data/texts.csv -o ./out
pense tts leaderboard -o ./out -s ./leaderboard

# LLM Tests
pense llm tests run -c ./config.json -o ./out -m gpt-4.1 -p openrouter
pense llm tests leaderboard -o ./out -s ./leaderboard

# LLM Simulations
pense llm simulations run -c ./config.json -o ./out -m gpt-4.1 -p openrouter -n 1
pense llm simulations leaderboard -o ./out -s ./leaderboard

# Voice Agent Simulations
pense agent simulation -c ./config.json -o ./out --port 8765

# Interactive Agent Testing
pense agent test -c ./config.json -o ./out
```

---

## Configuration Files

### Tool Definition Format

```json
{
    "type": "client",
    "name": "plan_next_question",
    "description": "Plan the next question to ask",
    "parameters": [
        {
            "id": "next_unanswered_question_index",
            "type": "integer",
            "description": "Index of next question",
            "required": true
        },
        {
            "id": "questions_answered",
            "type": "array",
            "description": "List of answered question indices",
            "items": {"type": "integer"},
            "required": true
        }
    ]
}
```

### Persona Definition Format

```json
{
    "characteristics": "A shy mother named Geeta, 39 years old, gives short answers",
    "gender": "female",
    "language": "english",
    "interruption_sensitivity": "medium"  // none, low, medium, high
}
```

**Interruption Sensitivity Mapping:**
- `none`: 0% probability
- `low`: 25% probability
- `medium`: 50% probability
- `high`: 80% probability

### Scenario Definition Format

```json
{
    "description": "User completes the form without any issues"
}
```

### Evaluation Criteria Format

```json
{
    "name": "question_completeness",
    "description": "Whether all the questions in the form were covered"
}
```

### Voice Agent Config Format

```json
{
    "system_prompt": "You are a helpful assistant.",
    "language": "english",
    "stt": {"provider": "deepgram"},
    "tts": {"provider": "cartesia", "voice_id": "YOUR_VOICE_ID"},
    "llm": {"provider": "openrouter", "model": "openai/gpt-4.1"},
    "tools": [...],
    "personas": [...],
    "scenarios": [...],
    "evaluation_criteria": [...],
    "settings": {"agent_speaks_first": true},
    "max_turns": 50
}
```

---

## Tech Stack

- **Language:** Python 3.10+
- **Package Manager:** uv
- **Core Framework:** pipecat-ai
- **Key Dependencies:**
  - `pipecat-ai` - Voice pipeline framework
  - `instructor` - Structured LLM outputs
  - `jiwer` - WER calculation
  - `numpy`, `pandas` - Data processing
  - `matplotlib` - Visualization
  - `openpyxl` - Excel exports

---

## Environment Variables

```bash
# Required based on providers used

# STT Providers
DEEPGRAM_API_KEY=your_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
OPENAI_API_KEY=your_key
CARTESIA_API_KEY=your_key
ELEVENLABS_API_KEY=your_key
SARVAM_API_KEY=your_key

# LLM Providers
OPENAI_API_KEY=your_key
OPENROUTER_API_KEY=your_key

# TTS Providers (same keys as above where applicable)
```

All modules read from a single `.env` file in the project root.

---

## Output Formats

### Metrics JSON (STT)

```json
{
    "wer": 0.129,
    "string_similarity": 0.879,
    "llm_judge_score": 1.0
}
```

### Metrics JSON (TTS)

```json
[
    {"llm_judge_score": 1.0},
    {"metric_name": "ttfb", "processor": "GoogleTTSService#0", "mean": 0.354, "std": 0.027, "values": [...]},
    {"metric_name": "processing_time", "processor": "GoogleTTSService#0", "mean": 0.0002, "std": 0.00003, "values": [...]}
]
```

### Results JSON (LLM Tests)

```json
[
    {
        "output": {
            "response": "...",
            "tool_calls": [{"tool": "...", "arguments": {...}}]
        },
        "metrics": {"passed": true},
        "test_case": {...}
    }
]
```

### Aggregated Metrics JSON (Simulations)

```json
{
    "question_completeness": {"mean": 1.0, "std": 0.0, "values": [1.0, 1.0, 1.0]},
    "assistant_behavior": {"mean": 0.67, "std": 0.58, "values": [1.0, 0.0, 1.0]},
    "stt_llm_judge": {"mean": 0.95, "std": 0.03, "values": [0.95, 0.92, 0.98]}
}
```

---

## Leaderboard Generation

All modules support leaderboard generation that:
1. Scans output directories for provider results
2. Aggregates metrics across runs
3. Generates comparison Excel files and PNG charts

**Output Files:**
- `stt_leaderboard.xlsx` / `tts_leaderboard.xlsx` / `llm_leaderboard.csv`
- `all_metrics_by_run.png` / `llm_leaderboard.png`

---

## Documentation

Documentation is built using [Mintlify](https://mintlify.com) with configuration in `docs/docs.json`.

**Documentation Tabs:**
- Getting Started
- Integrations (STT, LLM, TTS providers)
- Python SDK
- CLI

**Local Preview:**
```bash
uv pip install -r requirements-docs.txt
# Follow Mintlify docs for local preview
```

---

## Coding Standards

1. **Async/Await:** All evaluation functions are async
2. **Type Hints:** Use `Literal` for constrained string parameters
3. **Module Structure:** Each module exposes a clean public API via `__init__.py`
4. **Output Organization:** Consistent directory structure across all modules
5. **Logging:** Dual logging to terminal and log files
6. **Error Handling:** Graceful failure with detailed error messages
7. **Checkpointing:** Resume from interruption using existing results

---

## Gotchas & Edge Cases

### Audio Files
- All audio must be WAV format
- STT input audio should match the file names in `stt.csv`
- Voice simulation audio uses 1-based indexing: `1_user.wav`, `1_bot.wav`, etc.

### Provider-Specific
- Different providers may have different language support
- Some providers require specific voice IDs for TTS
- OpenRouter model names use `provider/model` format (e.g., `openai/gpt-4.1`)

### Simulations
- `max_turns` limits assistant turns, not total turns
- Conversation ends gracefully after max turns (final assistant message recorded)
- Transcript includes `end_reason` message when max turns reached

### Interactive Testing
- Use headphones to avoid audio feedback
- Opens browser UI at `http://localhost:7860/client/`
- Requires explicit `pense agent test` CLI command (no Python API for interactive mode)
