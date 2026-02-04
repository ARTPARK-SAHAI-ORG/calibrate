---
description: "Description of the project"
alwaysApply: true
---

# Calibrate

**Open-Source Voice Agent Simulation and Testing Framework**

---

## What is Calibrate?

Calibrate is an open-source Python framework for building, testing, and evaluating **voice-based AI agents**. It provides comprehensive tools to move from slow, manual testing to fast, automated, and repeatable testing processes.

The framework enables:

- **Component-level testing** - Evaluate STT, TTS, and LLM providers in isolation
- **LLM unit tests** - Verify agent behavior with deterministic test cases
- **End-to-end simulations** - Run automated conversations with simulated users
- **Benchmarking** - Compare performance across different AI providers

Calibrate uses direct API calls for STT and TTS provider evaluations, and [pipecat](https://github.com/pipecat-ai/pipecat) for voice agent simulations.

---

## Project Structure

```
/
├── calibrate/                    # Main Python package
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
│   ├── integrations/
│   └── examples/           # Use cases and recipes
├── pyproject.toml          # Package configuration
├── requirements-docs.txt   # Documentation dependencies
├── uv.lock                 # Dependency lockfile
└── README.md               # Project documentation
```

---

## Architecture Overview

### Module Design

Calibrate is organized into four main modules, each providing both a Python API and CLI commands:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CALIBRATE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│   │   calibrate.stt     │  │   calibrate.tts     │  │   calibrate.llm     │        │
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
│                    │   calibrate.agent       │                              │
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

Calibrate allows testing and benchmarking each component individually or end-to-end.

---

## Key Concepts

### 1. Speech-to-Text (STT) Evaluation

Evaluates STT providers by transcribing audio files and comparing against ground truth.

**Supported Providers:** deepgram, openai, cartesia, google, gemini, sarvam, elevenlabs, smallest, groq

**Supported Languages:** english, hindi, kannada, bengali, malayalam, marathi, odia, punjabi, tamil, telugu, gujarati, sindhi (Indian languages)

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

**Supported Providers:** cartesia, openai, groq, google, elevenlabs, sarvam, smallest

**Supported Languages:** english, hindi, kannada, bengali, malayalam, marathi, odia, punjabi, tamil, telugu, gujarati, sindhi (Indian languages)

**Metrics:**

- **LLM Judge:** AI evaluation of pronunciation accuracy using an audio-capable model (`gpt-audio`). Directly compares raw audio against input text — does NOT convert speech to text first.
- **TTFB (Time to First Byte):** Latency measurement (time to receive first audio chunk)

**Input:** CSV file with `id,text` columns

**Output Structure:**

```
output_dir/provider/
├── audios/          # Generated audio files (named after id: row_1.wav, row_2.wav, etc.)
├── results.csv      # Per-text results (id, text, audio_path, ttfb, llm_judge_score, llm_judge_reasoning)
├── metrics.json     # Aggregated metrics (llm_judge_score, ttfb with mean/std/values)
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

**Conversation History Preprocessing:**

Before running a test, the conversation history is preprocessed to handle tool calls:

- **Webhook tools:** Left as-is. Webhook tools are expected to have their own `role: "tool"` responses in the history since they interact with external APIs and need realistic response data.
- **Non-webhook tools (structured_output/client):** Tool responses are auto-inserted. For any assistant message with `tool_calls` where the tool is NOT a webhook:
  - If a `role: "tool"` response already exists for that `tool_call_id` → **Error** (test fails with validation error)
  - If no tool response exists → Auto-insert: `{"role": "tool", "content": "{\"status\": \"received\"}", "tool_call_id": "<id>"}`

This preprocessing is handled by `preprocess_conversation_history()` in `calibrate/llm/run_tests.py`.

**Example history with tool calls:**

```python
# Input history (non-webhook tool call without response)
{
    "history": [
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Aman"},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "abc123",
                "function": {"name": "plan_next_question", "arguments": "..."},
                "type": "function"
            }]
        },
        {"role": "user", "content": "Continue"}
    ]
}

# Preprocessed history (tool response auto-inserted)
[
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Aman"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "{\"status\": \"received\"}", "tool_call_id": "abc123"},  # <-- auto-inserted
    {"role": "user", "content": "Continue"}
]
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
from calibrate.stt import eval as stt_eval, leaderboard as stt_leaderboard
from calibrate.tts import eval as tts_eval, leaderboard as tts_leaderboard
from calibrate.llm import tests, simulations
from calibrate.agent import simulation, STTConfig, TTSConfig, LLMConfig

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
calibrate stt eval -p deepgram -l english -i ./data -o ./out
calibrate stt leaderboard -o ./out -s ./leaderboard

# TTS Evaluation
calibrate tts eval -p google -l english -i ./data/texts.csv -o ./out
calibrate tts leaderboard -o ./out -s ./leaderboard

# LLM Tests
calibrate llm tests run -c ./config.json -o ./out -m gpt-4.1 -p openrouter
calibrate llm tests leaderboard -o ./out -s ./leaderboard

# LLM Simulations
calibrate llm simulations run -c ./config.json -o ./out -m gpt-4.1 -p openrouter -n 1
calibrate llm simulations leaderboard -o ./out -s ./leaderboard

# Voice Agent Simulations
calibrate agent simulation -c ./config.json -o ./out --port 8765

# Interactive Agent Testing
calibrate agent test -c ./config.json -o ./out
```

---

## Configuration Files

### Tool Definition Format

Tools can be defined in two formats: `structured_output` (default) and `webhook`.

**Structured Output Format (default):**

Parameters are defined at the top level in a `parameters` array:

```json
{
  "type": "structured_output",
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
      "items": { "type": "integer" },
      "required": true
    }
  ]
}
```

**Webhook Format:**

Parameters are extracted from `webhook.queryParameters` and `webhook.body.parameters`.

**Required fields** in `webhook` object (raises `ValueError` if missing):
- `url` - The webhook endpoint URL
- `method` - HTTP method (GET, POST, PUT, etc.)
- `headers` - Array of header objects (can be empty `[]`)

```json
{
  "type": "webhook",
  "name": "submit_form",
  "description": "Submit form data to external API",
  "parameters": [],
  "webhook": {
    "method": "POST",
    "url": "https://api.example.com/submit",
    "timeout": 20,
    "headers": [
      {"name": "Authorization", "value": "Bearer X"}
    ],
    "queryParameters": [
      {"id": "key", "type": "string", "description": "API key", "required": true}
    ],
    "body": {
      "description": "Request body",
      "parameters": [
        {"id": "data", "type": "string", "description": "Form data", "required": true}
      ]
    }
  }
}
```

**How tool parameters are processed:**

Tool schema building is centralized in `calibrate/utils.py` via the `build_tools_schema(tools)` function, which returns `tuple[list[FunctionSchema], dict[str, dict]]` (schemas and webhook configs).

This function is used by all files that handle tools:
- `calibrate/llm/run_tests.py` - Uses webhook configs to log webhook details (no actual HTTP call)
- `calibrate/agent/bot.py` - Uses webhook configs to make actual HTTP calls
- `calibrate/llm/run_simulation.py` - Uses webhook configs to make actual HTTP calls
- `calibrate/agent/test.py` - Uses webhook configs to make actual HTTP calls

**For `structured_output` type (or when `type` is not specified):**
- Parameters from `tool["parameters"]` are added as flat properties to the FunctionSchema
- `required` list contains parameter IDs where `"required": true`

**For `webhook` type:**
- Parameters are structured as nested `query` and `body` objects in the FunctionSchema:
  - `query`: object containing properties from `webhook.queryParameters` with its own `required` list
  - `body`: object containing properties from `webhook.body.parameters` with its own `required` list
- Top-level `required` list contains `"query"` and/or `"body"` if they have any required params
- Webhook configs (url, method, headers, timeout) are returned separately for handler registration

**Webhook HTTP calls:**

The `make_webhook_call(webhook_config, arguments)` utility function in `calibrate/utils.py` makes actual HTTP requests:
- Uses `aiohttp` for async HTTP calls
- Converts headers list to dict format
- Extracts query params from `arguments["query"]` and body from `arguments["body"]`
- Supports GET, POST, PUT, PATCH, DELETE methods (body only sent for POST/PUT/PATCH)
- Returns `{status, status_code, response}` on success or `{status: "error", error: "..."}` on failure
- Handles timeouts (configurable, defaults to 20s) and client errors gracefully

**Handler registration:**

| File | structured_output | webhook |
|------|-------------------|---------|
| `calibrate/llm/run_tests.py` | `generic_tool_call` (logs only) | `webhook_tool_call` (logs only, no HTTP) |
| `calibrate/agent/bot.py` | `generic_function_call` | `webhook_function_call` (makes HTTP call in "run" mode) |
| `calibrate/llm/run_simulation.py` | `generic_function_call` | `webhook_function_call` (makes HTTP call) |
| `calibrate/agent/test.py` | `generic_function_call` | `webhook_function_call` (makes HTTP call) |
| `calibrate/agent/run_simulation.py` | `RTVIFunctionCallResponder` returns `{"status": "received"}` | `RTVIFunctionCallResponder` makes HTTP call |

**Note on LLM tests tool handling:** In `calibrate/llm/run_tests.py`, conversation history with tool calls is preprocessed before passing to the LLM. For non-webhook tools, tool responses (`role: "tool"`) are auto-inserted with `{"status": "received"}`. For webhook tools, manual tool responses are expected in the history. See "Conversation History Preprocessing" in the LLM Tests section.

**Voice agent simulation tool handling (`calibrate/agent/run_simulation.py`):**

The `RTVIFunctionCallResponder` class handles function calls received via RTVI protocol in voice agent simulations:
- Accepts `webhook_configs` parameter built from tools via `build_tools_schema`
- `end_call`: Returns `{"acknowledged": true}` and triggers end call callback
- Webhook tools: Makes actual HTTP call via `make_webhook_call` and returns response
- Non-webhook tools: Returns `{"status": "received"}`
- Tools are passed from `run_single_simulation_task` → `run_simulation` → `RTVIFunctionCallResponder`

**Example LLM arguments for webhook tool:**
```json
{
  "query": {"key": "abc123"},
  "body": {"data": "form data"}
}
```

### Persona Definition Format

```json
{
  "characteristics": "A shy mother named Geeta, 39 years old, gives short answers",
  "gender": "female",
  "language": "english",
  "interruption_sensitivity": "medium" // none, low, medium, high
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
    "settings": {"agent_speaks_first": true, "max_turns": 50}
}
```

---

## Tech Stack

- **Language:** Python 3.10+
- **Package Manager:** uv
- **Key Dependencies:**
  - `pipecat-ai` - Voice pipeline framework (used for voice agent simulations only). **MUST be pinned to exact version** (e.g., `==0.0.98`) in `pyproject.toml` because the library is distributed as a wheel and loose constraints (`>=`) will install latest version on servers, causing version mismatches and breaking changes.
  - `pipecat-ai-small-webrtc-prebuilt` - WebRTC support for pipecat. **MUST also be pinned** to exact version.
  - `aiohttp` - Async HTTP client for webhook calls
  - `openai` - OpenAI STT/TTS API
  - `google-cloud-speech`, `google-cloud-texttospeech` - Google Cloud APIs
  - `elevenlabs` - ElevenLabs STT/TTS API
  - `cartesia` - Cartesia STT/TTS API
  - `sarvamai` - Sarvam STT/TTS API
  - `groq` - Groq STT/TTS API
  - `deepgram-sdk` - Deepgram STT API
  - `instructor` - Structured LLM outputs
  - `jiwer` - WER calculation
  - `pydub` - Audio format conversion (MP3 to WAV, etc.)
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
{
  "llm_judge_score": 1.0,
  "ttfb": {
    "mean": 0.354,
    "std": 0.027,
    "values": [0.38, 0.33]
  }
}
```

**Note:** TTS metrics.json uses a flat dict with metric names as keys. The `ttfb` metric is a nested dict with mean, std, and values array.

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
  "question_completeness": {
    "mean": 1.0,
    "std": 0.0,
    "values": [1.0, 1.0, 1.0]
  },
  "assistant_behavior": {
    "mean": 0.67,
    "std": 0.58,
    "values": [1.0, 0.0, 1.0]
  },
  "stt_llm_judge": { "mean": 0.95, "std": 0.03, "values": [0.95, 0.92, 0.98] }
}
```

---

## Leaderboard Generation

All modules support leaderboard generation that:

1. Scans output directories for provider results
2. Aggregates metrics across runs
3. Generates comparison Excel files and per-metric PNG charts

**Output Files:**

- `stt_leaderboard.xlsx` / `tts_leaderboard.xlsx` / `llm_leaderboard.csv`
- Individual metric charts - one chart per metric for easy comparison:
  - **STT:** `wer.png`, `string_similarity.png`, `llm_judge_score.png`
  - **TTS:** `llm_judge_score.png`, `ttfb.png`

**Chart Features:**

- Each chart shows all providers as bars on the x-axis
- Value labels displayed on top of each bar (integers shown as integers, decimals with 4 decimal places)
- Metrics with all NaN values are automatically skipped
- Charts saved at 300 DPI for high quality

---

## Documentation

Documentation is built using [Mintlify](https://mintlify.com) with configuration in `docs/docs.json`.

**Documentation Tabs:**

- Getting Started
- Integrations (STT, LLM, TTS providers)
- Python SDK
- CLI
- Use cases (examples and recipes for common evaluation workflows)

**Core Concepts Pages (in order):**

- speech-to-text
- text-to-speech
- agents
- tools
- text-to-text
- personas
- scenarios
- simulations

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
5. **Logging:** Dual logging to terminal and log files with parallel-safe per-simulation loggers
6. **Error Handling:** Let errors propagate (no silent catching) so Sentry captures them automatically
7. **Checkpointing:** Resume from interruption using existing results

---

## Gotchas & Edge Cases

### Audio Files

- All audio must be WAV format
- STT input audio should match the file names in `stt.csv`
- Voice simulation audio uses 1-based indexing: `1_user.wav`, `1_bot.wav`, etc.

### Provider-Specific

- **Separate STT/TTS language codes:** STT and TTS providers often support different languages. Language codes are managed separately in `calibrate/utils.py`:
  - `get_stt_language_code(language, provider)` - For STT providers
  - `get_tts_language_code(language, provider)` - For TTS providers
  - `get_language_code()` - Deprecated, defaults to STT codes for backwards compatibility
- **STT-specific dictionaries:** `DEEPGRAM_STT_LANGUAGE_CODES`, `OPENAI_STT_LANGUAGE_CODES`, `GOOGLE_STT_LANGUAGE_CODES`, `CARTESIA_STT_LANGUAGE_CODES`, `ELEVENLABS_STT_LANGUAGE_CODES`, `SMALLEST_STT_LANGUAGE_CODES`, `GROQ_STT_LANGUAGE_CODES`
- **TTS-specific dictionaries:** `GOOGLE_TTS_LANGUAGE_CODES`, `CARTESIA_TTS_LANGUAGE_CODES`, `ELEVENLABS_TTS_LANGUAGE_CODES`, `GROQ_TTS_LANGUAGE_CODES`, `OPENAI_TTS_LANGUAGE_CODES`, `SMALLEST_TTS_LANGUAGE_CODES`
- **Shared dictionaries:** `SARVAM_LANGUAGE_CODES` (same for both STT and TTS)
- **Key differences between STT and TTS language support:**
  - Groq TTS only supports English (Orpheus model), while Groq STT supports 50+ languages (Whisper)
  - Cartesia TTS supports ~42 languages, STT supports 100+ languages
  - Google TTS supports ~47 languages, STT supports 70+ languages
  - ElevenLabs TTS supports ~29 languages, STT supports 90+ languages
- **Language code formats vary by provider:**
  - Sarvam uses BCP-47 Indian codes: `hi-IN`, `kn-IN`, `bn-IN`, etc.
  - Google uses BCP-47 codes: `en-US`, `hi-IN`, etc.
  - ElevenLabs uses ISO 639-3 codes: `eng`, `hin`, etc.
  - Most others use ISO 639-1 codes: `en`, `hi`, `kn`, etc.
- Not all providers support all 12 languages - Sarvam has the most comprehensive support for Indian languages
- **Sindhi language special handling:**
  - **STT:** For Google STT, Sindhi requires a different model (`chirp_2`) and region (`asia-southeast1`) compared to the default (`chirp_3` model, `us` region). This is handled automatically in `calibrate/stt/eval.py` via `transcribe_google()`. Sindhi is supported by Google, Cartesia, and ElevenLabs for STT.
  - **TTS:** Sindhi TTS requires special handling for both Google and ElevenLabs:
    - **Google:** Uses streaming API with `gemini-2.5-flash-lite-preview-tts` model. Key difference: voice name is just "Charon" (not locale-prefixed like `sd-IN-Chirp3-HD-Charon`) and requires `model_name` parameter in `VoiceSelectionParams`. See [Google Gemini-TTS docs](https://cloud.google.com/text-to-speech/docs/gemini-tts).
    - **ElevenLabs:** Uses `eleven_v3` model with `text_to_dialogue` API instead of the standard `text_to_speech` API
- Some providers require specific voice IDs for TTS
- OpenRouter model names use `provider/model` format (e.g., `openai/gpt-4.1`)

### LLM Tests

- **Tool response auto-insertion:** For non-webhook tools, `role: "tool"` responses are automatically inserted with `{"status": "received"}`. Do NOT manually add tool responses for non-webhook tools in test history - this will cause a validation error.
- **Webhook tools need manual responses:** Webhook tools expect realistic response data, so you must provide `role: "tool"` messages in the history for webhook tool calls.
- **Tool call ID matching:** The auto-inserted tool response uses the `id` from the `tool_calls` array in the assistant message. Ensure tool call IDs are unique.
- **Preprocessing happens before each test:** The `preprocess_conversation_history()` function runs before each test case, not once for all tests.

### Simulations

- `max_turns` is configured in `settings` (e.g., `"settings": {"max_turns": 50}`)
- `max_turns` limits assistant turns, not total turns
- Conversation ends gracefully after max turns (final assistant message recorded)
- Transcript includes `end_reason` message when max turns reached

### Parallel Simulation Logging

When multiple simulations run in parallel (e.g., via `asyncio.gather`), each simulation needs its own isolated logging to prevent logs from mixing. There are two logging systems that need isolation.

**Important:** Logger identification uses UUIDs, not simulation folder names. Multiple parallel runs can have the same folder name pattern (e.g., `simulation_persona_1_scenario_2` from different batches), so a UUID ensures each run has a unique logger identity.

#### 1. Loguru (`logs` file) - Uses `logger.contextualize()`

Loguru's `contextualize()` context manager binds extra data to ALL log calls within its scope, including logs from libraries like pipecat. Combined with a strict filter, this ensures each simulation's `logs` file only contains its own logs.

```python
# Generate unique ID for this simulation run (NOT the folder name)
simulation_run_id = str(uuid.uuid4())

# Create sink with strict filter (only accepts logs from this simulation)
def simulation_filter(record):
    sim_id = record["extra"].get("simulation")
    return sim_id == simulation_run_id

log_file_id = logger.add(logs_file_path, filter=simulation_filter, ...)

# Wrap ALL simulation code in contextualize using the UUID
with logger.contextualize(simulation=simulation_run_id):
    # All logger calls here (including from pipecat) have simulation in extra
    await run_simulation(...)

# Cleanup
logger.remove(log_file_id)
```

#### 2. Print Logger (`results.log` file) - Uses per-simulation loggers

- **`calibrate/utils.py` logging utilities:**
  - `_simulation_print_loggers: dict[str, logging.Logger]` - Stores per-simulation print loggers (keyed by UUID)
  - `current_simulation_name: ContextVar[str]` - Context variable to track active simulation (stores UUID)
  - `configure_print_logger(log_path, simulation_name="")` - Creates unique logger per simulation (pass UUID)
  - `cleanup_print_logger(simulation_name)` - Closes file handlers and removes logger from dict (pass UUID)
  - `log_and_print(message)` - Uses context variable to find correct print logger

```python
# Generate unique ID for this simulation run
simulation_run_id = str(uuid.uuid4())

# Setup - use UUID for logger identification, but log_path uses folder name
configure_print_logger(print_log_save_path, simulation_name=simulation_run_id)
current_simulation_name.set(simulation_run_id)

# During simulation - log_and_print uses context var automatically
log_and_print("message")  # Goes to correct simulation's results.log

# Cleanup (in finally block) - use UUID
cleanup_print_logger(simulation_run_id)
```

- **Gotchas:**
  - Call `logger.remove()` once at the start of `main()` to remove the default stderr handler - this prevents all loguru logs from appearing on terminal
  - The `logger.contextualize()` block MUST wrap all simulation code including the `run_simulation()` call
  - The filter must be strict (`return sim_id == simulation_run_id`) - do NOT use `or sim_id is None` fallback
  - Always call `cleanup_print_logger` in a `finally` block to avoid resource leaks
  - Always call `logger.remove(log_file_id)` in a `finally` block
  - The global `_print_logger` is used for backwards compatibility when `simulation_name` is not provided
  - Only `log_and_print` output appears on terminal (via `print()`); loguru logs go only to file sinks
  - **Use UUIDs for logger IDs, folder names for file paths** - `simulation_name` (folder) is for display and file paths, `simulation_run_id` (UUID) is for logger isolation

### Interactive Testing

- Use headphones to avoid audio feedback
- Opens browser UI at `http://localhost:7860/client/`
- Requires explicit `calibrate agent test` CLI command (no Python API for interactive mode)

### STT/TTS Evaluation Architecture

- **Direct API calls:** Both STT and TTS evaluations use direct provider SDK/API calls (not pipecat)
- **Streaming TTFB:** Most TTS providers use streaming APIs and measure true TTFB (time to first audio chunk): OpenAI, ElevenLabs, Cartesia, Google, Sarvam, Smallest
- **Non-streaming:** Groq does not support streaming and does not return TTFB
- **Pipecat usage:** Only voice agent simulations use pipecat for the full STT→LLM→TTS pipeline
- **TTS LLM Judge accuracy:** The audio-capable model (`gpt-audio`) may have reduced accuracy for low-resource languages like Sindhi due to limited training data
- **Language validation:** `validate_stt_language()` and `validate_tts_language()` in `calibrate/utils.py` check if a language is supported by a provider before evaluation starts. Each function uses the appropriate STT or TTS language dictionaries. If invalid, the run stops with an error listing all supported languages for that provider.
- **TTS audio saving patterns:** All TTS synthesize functions accept `audio_path` and save audio:
  - Streaming providers (OpenAI, Cartesia) write chunks directly to file as they arrive
  - Streaming providers (Google, Sarvam, Smallest) collect chunks then save combined audio (Google uses PCM encoding)
  - ElevenLabs streams MP3, then converts to WAV using `convert_mp3_to_wav()` helper function (uses pydub)
- **Google TTS voice naming patterns:**
  - Default (Chirp3-HD): locale-prefixed name like `"{lang_code}-Chirp3-HD-Charon"` (e.g., `en-US-Chirp3-HD-Charon`)
  - Gemini-TTS (for Sindhi): just the voice name `"Charon"` with `model_name` parameter set
- **Optional TTFB:** TTS synthesize functions may return an empty dict `{}` if TTFB cannot be measured (e.g., Groq). The evaluation script handles this gracefully:
  - Missing TTFB values are stored as `None` in results.csv
  - Only valid TTFB values are included in metrics.json aggregation

### Metrics JSON Format

- **Consistent structure:** Both STT and TTS use flat dict format with metric names as keys
- **Simple metrics:** Stored as direct float values (e.g., `"wer": 0.129`, `"llm_judge_score": 1.0`)
- **Latency metrics:** Stored as nested dicts with `mean`, `std`, and `values` (e.g., `"ttfb": {"mean": 0.35, "std": 0.03, "values": [...]}`)
- **Backwards compatibility:** Leaderboard readers support both new dict format and legacy list-of-dicts format for older results
