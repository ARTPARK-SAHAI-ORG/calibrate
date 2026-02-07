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
│   │   ├── __init__.py      # Public API: run() (multi-provider benchmark), run_single(), generate_leaderboard()
│   │   ├── eval.py          # Single-provider STT evaluation (run_single_provider_eval)
│   │   ├── benchmark.py     # Multi-provider benchmark with parallelization + leaderboard (run)
│   │   ├── leaderboard.py   # Leaderboard generation (generate_leaderboard)
│   │   └── metrics.py       # STT metrics (WER, string similarity, LLM judge)
│   ├── tts/                 # Text-to-Speech evaluation module
│   │   ├── __init__.py      # Public API: run() (multi-provider benchmark), run_single(), generate_leaderboard()
│   │   ├── eval.py          # Single-provider TTS evaluation (run_single_provider_eval)
│   │   ├── benchmark.py     # Multi-provider benchmark with parallelization + leaderboard (run)
│   │   ├── leaderboard.py   # Leaderboard generation (generate_leaderboard)
│   │   └── metrics.py       # TTS metrics (LLM judge, TTFB, processing time)
│   ├── llm/                 # LLM evaluation module
│   │   ├── __init__.py      # Public API: tests.run(), tests.run_single(), tests.leaderboard(), simulations.run(), simulations.run_single(), simulations.leaderboard()
│   │   ├── run_tests.py     # Single-model LLM test runner (run_model_tests)
│   │   ├── benchmark.py     # Multi-model benchmark with parallelization + leaderboard (run)
│   │   ├── run_simulation.py # LLM simulation runner
│   │   ├── tests_leaderboard.py  # LLM tests leaderboard generation
│   │   ├── simulation_leaderboard.py
│   │   └── metrics.py       # LLM evaluation metrics
│   ├── agent/               # Voice agent simulation module
│   │   ├── __init__.py      # Public API: simulation, STTConfig, TTSConfig, LLMConfig
│   │   ├── bot.py           # Voice agent pipeline
│   │   ├── test.py          # Interactive agent testing
│   │   └── run_simulation.py # Voice simulation runner
│   ├── ui/                  # Bundled interactive CLI (ships with package)
│   │   ├── __init__.py
│   │   └── cli.bundle.mjs   # esbuild output — single file, all deps included
│   └── integrations/        # Third-party provider integrations
│       └── smallest/        # Smallest AI STT/TTS integration
├── examples/                # Sample inputs, configs, and outputs (not shipped with package)
│   ├── stt/                # STT sample input/output/leaderboard
│   ├── tts/                # TTS sample CSV/output/leaderboard
│   ├── llm/                # LLM tests and simulation configs/output
│   └── agent/              # Voice agent simulation/test configs/output
├── docs/                    # Mintlify documentation
│   ├── docs.json           # Navigation and theme config
│   ├── getting-started/
│   ├── quickstart/
│   ├── core-concepts/
│   ├── cli/
│   └── integrations/
├── ui/                      # Interactive Ink-based CLI (Node.js/TypeScript)
│   ├── package.json         # Dependencies (ink, react)
│   ├── tsconfig.json        # TypeScript config
│   └── source/
│       ├── cli.tsx          # Entry point — reads mode from argv, defaults to "menu"
│       ├── app.tsx          # Main menu + routing + STT/TTS eval flow components
│       ├── shared.ts        # Shared types (CalibrateCmd, AppMode) and utilities (findCalibrateBin, stripAnsi, findAvailablePort)
│       ├── llm-app.tsx      # LLM tests interactive flow (config path, provider, model, run, results)
│       ├── sim-app.tsx      # Simulations interactive flow (type, config path, provider, model, run, results)
│       ├── resource-app.tsx # Resource CRUD UIs (agents, tools, personas, scenarios, metrics) — not in menu
│       ├── storage.ts       # Persistent resource storage in ~/.calibrate/ (agents, tools, personas, etc.)
│       ├── components.tsx   # Reusable UI components (Spinner, TextInput, TextArea, MultiSelect, etc.)
│       ├── providers.ts     # TTS + STT provider definitions with per-language support
│       └── credentials.ts   # Secure API key storage (~/.calibrate/credentials.json)
├── .github/workflows/
│   ├── docs.yml            # Mintlify docs deployment
│   └── publish.yml         # PyPI publish on GitHub release
├── pyproject.toml          # Package configuration
├── LICENSE.md              # CC BY-SA 4.0 license
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
│   │  run()          │  │  run()          │  │  tests.run()    │        │
│   │  (eval+leader-  │  │  (eval+leader-  │  │  (multi-model,  │        │
│   │   board combo)  │  │   board combo)  │  │   parallel, +lb)│        │
│   │                 │  │                 │  │  simulations.run()       │
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

**Key Parameters:**

- `overwrite` (bool, default: False): Overwrite existing results instead of resuming from last checkpoint. When False, the evaluation script loads existing results and skips already processed audio files, allowing graceful recovery from interruptions.
- `port` (int, default: 8765): WebSocket port for STT bot communication during evaluation.

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

**Key Parameters:**

- `overwrite` (bool, default: False): Overwrite existing results instead of resuming from last checkpoint. When False, the evaluation script loads existing results and skips already processed texts, allowing graceful recovery from interruptions.

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

**Parallelism:** Multiple models can be tested in parallel (space-separated via `-m`, e.g., `-m gpt-4.1 gpt-4o`). Max 2 models run concurrently to avoid rate limits.

**Output Structure:**

Results are saved directly to `output_dir/model_name/` (no config name in path):

```
output_dir/
├── openai__gpt-4.1/          # Model folder (slashes replaced with __)
│   ├── results.json          # Per-test results with metrics
│   ├── metrics.json          # Aggregated metrics {total, passed}
│   └── results.log           # Terminal output
├── openai__gpt-4.1-mini/
│   ├── results.json
│   ├── metrics.json
│   └── results.log
└── leaderboard/
    ├── llm_leaderboard.csv   # Model comparison
    └── llm_leaderboard.png   # Bar chart

```

**Model folder naming:** For `openai` provider, model folder is `{provider}__{model}` (e.g., `openai__gpt-4.1`). For `openrouter` provider, model folder is just `{model}` with slashes replaced (e.g., `openai/gpt-4.1` → `openai__gpt-4.1`).

**Leaderboard:** Generated automatically by `benchmark.py` after all models complete. In the Ink UI, leaderboard is generated via `python -m calibrate.llm.tests_leaderboard` after all model evaluations finish.

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
from calibrate.stt import run as stt_run, run_single as stt_run_single, generate_leaderboard as stt_leaderboard
from calibrate.tts import run as tts_run, run_single as tts_run_single, generate_leaderboard as tts_leaderboard
from calibrate.llm import tests, simulations
from calibrate.agent import simulation, STTConfig, TTSConfig, LLMConfig

# STT Benchmark (runs multiple providers in parallel + generates leaderboard)
asyncio.run(stt_run(
    providers=["deepgram", "google", "sarvam"],
    language="english",
    input_dir="./data",
    output_dir="./out"
))

# STT Single Provider Evaluation (no leaderboard)
asyncio.run(stt_run_single(
    provider="deepgram",
    language="english",
    input_dir="./data",
    output_dir="./out"
))

# Generate STT leaderboard separately
stt_leaderboard(output_dir="./out", save_dir="./out/leaderboard")

# TTS Benchmark (runs multiple providers in parallel + generates leaderboard)
asyncio.run(tts_run(
    providers=["google", "openai", "elevenlabs"],
    language="english",
    input="./data/texts.csv",
    output_dir="./out"
))

# TTS Single Provider Evaluation (no leaderboard)
asyncio.run(tts_run_single(
    provider="google",
    language="english",
    input_file="./data/texts.csv",
    output_dir="./out"
))

# Generate TTS leaderboard separately
tts_leaderboard(output_dir="./out", save_dir="./out/leaderboard")

# LLM Tests Benchmark (runs multiple models in parallel + generates leaderboard)
asyncio.run(tests.run(
    system_prompt="You are a helpful assistant...",
    tools=[...],
    test_cases=[...],
    models=["gpt-4.1", "claude-3.5-sonnet", "gemini-2.0-flash"],
    provider="openrouter",
    output_dir="./out"
))

# LLM Tests Single Model (no leaderboard)
asyncio.run(tests.run_single(
    system_prompt="You are a helpful assistant...",
    tools=[...],
    test_cases=[...],
    model="gpt-4.1",
    provider="openrouter",
    output_dir="./out"
))

# Generate LLM tests leaderboard separately
tests.leaderboard(output_dir="./out", save_dir="./leaderboard")

# LLM Simulations Benchmark (runs multiple models in parallel + generates leaderboard)
asyncio.run(simulations.run(
    system_prompt="You are a helpful nurse...",
    tools=[...],
    personas=[{"characteristics": "...", "gender": "female", "language": "english"}],
    scenarios=[{"description": "User completes the form"}],
    evaluation_criteria=[{"name": "completeness", "description": "..."}],
    models=["gpt-4.1", "claude-3.5-sonnet"],
    provider="openrouter",
    output_dir="./out"
))

# LLM Simulations Single Model (no leaderboard)
asyncio.run(simulations.run_single(
    system_prompt="You are a helpful nurse...",
    tools=[...],
    personas=[{"characteristics": "...", "gender": "female", "language": "english"}],
    scenarios=[{"description": "User completes the form"}],
    evaluation_criteria=[{"name": "completeness", "description": "..."}],
    model="gpt-4.1",
    provider="openrouter",
    output_dir="./out"
))

# Generate LLM simulations leaderboard separately
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
# Main menu (interactive — requires Node.js)
calibrate

# STT Evaluation (interactive Ink UI — guided setup with validation)
calibrate stt

# TTS Evaluation (interactive Ink UI — guided setup with validation)
calibrate tts

# STT Evaluation - single provider (uses eval.py)
calibrate stt -p deepgram -i ./data -l english -o ./out

# STT Evaluation - single provider with leaderboard (Ink UI uses this for last provider)
calibrate stt -p deepgram -i ./data -l english -o ./out --leaderboard

# STT Benchmark - multiple providers (uses benchmark.py, auto-generates leaderboard)
calibrate stt -p deepgram google sarvam -i ./data -l english -o ./out

# TTS Evaluation - single provider (uses eval.py)
calibrate tts -p google -i ./data/texts.csv -l english -o ./out

# TTS Evaluation - single provider with leaderboard (Ink UI uses this for last provider)
calibrate tts -p google -i ./data/texts.csv -l english -o ./out --leaderboard

# TTS Benchmark - multiple providers (uses benchmark.py, auto-generates leaderboard)
calibrate tts -p google openai elevenlabs -i ./data/texts.csv -l english -o ./out

# LLM Tests - single model (uses run_tests.py, no leaderboard)
calibrate llm -c config.json -m gpt-4.1 -p openrouter -o ./out

# LLM Tests Benchmark - multiple models (uses benchmark.py, auto-generates leaderboard)
calibrate llm -c config.json -m gpt-4.1 claude-3.5-sonnet gemini-2.0-flash -p openrouter -o ./out

# Interactive Agent Testing (voice, opens browser)
calibrate agent test -c ./config.json -o ./out
```

**Hidden CLI commands (not shown in `--help` or main menu, but fully functional internally):**

The following commands are registered in argparse but hidden from `--help` output and the Ink UI main menu. They still work if invoked directly and are used internally by the Ink UI when spawning child processes:

- `calibrate llm` — LLM tests interactive mode
- `calibrate simulations` — Simulations interactive mode
- `calibrate agents`, `calibrate tools`, `calibrate personas`, `calibrate scenarios`, `calibrate metrics` — Resource management
- `calibrate agent simulation`, `calibrate llm tests run`, `calibrate llm simulations run`, etc. — Internal commands spawned by the Ink UI

**How commands are hidden:**

- `argparse` subparsers are registered without a `help=` argument, which prevents them from appearing in `--help` descriptions
- The `parser` has an explicit `usage=` string that only lists `{stt,tts}`
- The `subparsers` has `metavar="{stt,tts}"` to control what appears in the positional arguments section
- The Ink UI main menu in `app.tsx` has the LLM, simulations, and resource management items commented out

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
    "headers": [{ "name": "Authorization", "value": "Bearer X" }],
    "queryParameters": [
      {
        "id": "key",
        "type": "string",
        "description": "API key",
        "required": true
      }
    ],
    "body": {
      "description": "Request body",
      "parameters": [
        {
          "id": "data",
          "type": "string",
          "description": "Form data",
          "required": true
        }
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

| File                                | structured_output                                            | webhook                                                 |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| `calibrate/llm/run_tests.py`        | `generic_tool_call` (logs only)                              | `webhook_tool_call` (logs only, no HTTP)                |
| `calibrate/agent/bot.py`            | `generic_function_call`                                      | `webhook_function_call` (makes HTTP call in "run" mode) |
| `calibrate/llm/run_simulation.py`   | `generic_function_call`                                      | `webhook_function_call` (makes HTTP call)               |
| `calibrate/agent/test.py`           | `generic_function_call`                                      | `webhook_function_call` (makes HTTP call)               |
| `calibrate/agent/run_simulation.py` | `RTVIFunctionCallResponder` returns `{"status": "received"}` | `RTVIFunctionCallResponder` makes HTTP call             |

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
  "query": { "key": "abc123" },
  "body": { "data": "form data" }
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
            "tool_calls": [{"tool": "...", "arguments": {...}}],
            "captured_errors": []
        },
        "metrics": {
            "passed": true,
            "reasoning": "Tool call matches expected: plan_next_question with correct arguments"
        },
        "test_case": {
            "history": [
                {"role": "assistant", "content": "Hello! What is your name?"},
                {"role": "user", "content": "My name is John"}
            ],
            "evaluation": {
                "type": "tool_call",
                "tool_calls": [{"tool": "plan_next_question", "arguments": {...}}]
            }
        }
    }
]
```

**Results JSON fields:**

- `output.response`: The text response generated by the LLM (empty string if only tool calls)
- `output.tool_calls`: Array of tool calls made by the LLM (`[{tool, arguments}]`)
- `output.captured_errors`: Any errors captured during inference
- `metrics.passed`: Boolean indicating if the test passed
- `metrics.reasoning`: Explanation of why the test passed or failed
- `test_case.history`: Conversation history as `[{role, content}]` messages
- `test_case.evaluation.type`: Either `"response"` or `"tool_call"`
- `test_case.evaluation.criteria`: (for response type) Text criteria for LLM judge
- `test_case.evaluation.tool_calls`: (for tool_call type) Expected tool calls

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

**STT Leaderboard:**

- Leaderboard generation is in a separate `leaderboard.py` file
- When using `benchmark.py` (via `calibrate stt -p provider1 provider2 ...` or Python SDK `run()`), leaderboard is auto-generated after all providers complete
- Optionally specify save directory with `-s/--save-dir` (defaults to `{output_dir}/leaderboard`)
- The `generate_leaderboard(output_dir, save_dir)` function is exported from `calibrate.stt`
- In the Ink UI, the last provider's eval includes `--leaderboard` flag to trigger leaderboard generation

**TTS Leaderboard:**

- Leaderboard generation is in a separate `leaderboard.py` file (same architecture as STT)
- When using `benchmark.py` (via `calibrate tts -p provider1 provider2 ...` or Python SDK `run()`), leaderboard is auto-generated after all providers complete
- Optionally specify save directory with `-s/--save-dir` (defaults to `{output_dir}/leaderboard`)
- The `generate_leaderboard(output_dir, save_dir)` function is exported from `calibrate.tts`
- In the Ink UI, the last provider's eval includes `--leaderboard` flag to trigger leaderboard generation

**LLM Tests Leaderboard:**

- Leaderboard generation is in `tests_leaderboard.py`
- When using `benchmark.py` (via `calibrate llm -m model1 model2 ...` or Python SDK `tests.run()` with `models` list), leaderboard is auto-generated after all models complete
- The `tests.leaderboard(output_dir, save_dir)` method can be called separately
- In the Ink UI, leaderboard is generated via `python -m calibrate.llm.tests_leaderboard -o <output_dir> -s <output_dir>/leaderboard` after all model evaluations complete
- `tests_leaderboard.py` expects a flat structure: `output_dir/model_name/metrics.json` (no scenario/config_name nesting)

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
- CLI
- Use cases (examples and recipes for common evaluation workflows)

Note: Python SDK documentation was removed. The Python SDK still exists and works, but all user-facing documentation focuses on the CLI. The Python API patterns are documented in this file for internal reference.

**Core Concepts Pages (in order):**

- speech-to-text
- text-to-speech
- agents
- tools
- text-to-text
- personas
- scenarios
- simulations

**CLI Documentation Structure:**

- overview
- speech-to-text
- text-to-speech
- text-to-text
- simulations (consolidated page for both text and voice simulations)

**Local Preview:**

```bash
npm i -g mintlify
cd docs
mintlify dev
```

**README.md** is intentionally minimal — it provides installation, links to the CLI docs on the docs site, and local docs setup instructions. All detailed usage documentation lives in `docs/` and on [docs.calibrate.dev](https://docs.calibrate.dev).

---

## Interactive CLI (`ui/`)

An Ink-based (React for CLIs) interactive terminal UI. Source lives in `ui/`, bundled output ships inside the Python package at `calibrate/ui/cli.bundle.mjs`.

**Run:** `calibrate` (main menu), or directly: `calibrate stt`, `calibrate tts` (requires Node.js). Other modes (`llm`, `simulations`, `agents`, `tools`, `personas`, `scenarios`, `metrics`) are functional but hidden from `--help` and the main menu until ready.

**Dev mode:** `cd ui && npx tsx source/cli.tsx` (menu) or `cd ui && npx tsx source/cli.tsx stt` etc.

**Startup log suppression:** The CLI entry point (`calibrate/cli.py`) suppresses noisy startup logs from pipecat (loguru INFO) and transformers before any imports occur. This is done by:

- Calling `logger.remove()` then `logger.add(sys.stderr, level="WARNING")` at the very top of the file
- Setting `TRANSFORMERS_VERBOSITY=error` environment variable

**Mode selection:** The `App` component accepts a `mode` prop of type `AppMode`. `cli.tsx` reads `process.argv[2]` (passed by `calibrate/cli.py` via `_launch_ink_ui(mode)`) and passes it as the mode. Defaults to `"menu"` if no argument provided.

**Available modes (`AppMode` in `shared.ts`):** `menu`, `stt`, `tts`, `llm`, `simulations`, `agents`, `tools`, `personas`, `scenarios`, `metrics`

### Main Menu (`mode = "menu"`)

Shows four options: STT Evaluation, TTS Evaluation, LLM Tests, and Simulations. Resource management items (Agents, Tools, Personas, Scenarios, Metrics) are removed from the menu. Selecting an option routes to that mode's flow. Launched by running `calibrate` with no arguments.

### STT/TTS Evaluation Flow (`mode = "stt" | "tts"`)

**Two execution modes:**

1. **Ink UI mode** (`calibrate stt` / `calibrate tts` without arguments): Launches Node.js-based interactive UI with full workflow (language → provider → input → output → API keys → run → leaderboard). The Ink UI handles parallelization via `RunStep` component, running max 2 providers concurrently via separate `child_process.spawn` calls. Each call is `calibrate stt -p <single_provider>` which routes to `eval.py`. The last provider's call includes `--leaderboard` to generate leaderboard.
2. **Direct CLI mode**:
   - **Single provider** (`calibrate stt -p provider -i ...`): Routes to `eval.py` for single-provider evaluation. Add `--leaderboard` to generate leaderboard after.
   - **Multiple providers** (`calibrate stt -p provider1 provider2 ... -i ...`): Routes to `benchmark.py` which runs providers in parallel (max 2 concurrent) and auto-generates leaderboard.

Provider and language choices are defined as module-level constants (`STT_PROVIDERS`, `STT_LANGUAGES`, `TTS_PROVIDERS`, `TTS_LANGUAGES`) in the eval files.

**Input validation (in Ink UI):**

The UI validates inputs in `ConfigInputStep` (`app.tsx`) before proceeding:

- **STT input validation** (`validateSttInput()`):

  - Checks input directory exists
  - Checks CSV file exists (default: `stt.csv`)
  - Validates CSV has required columns: `id`, `text`
  - Checks `audios/` subdirectory exists
  - Verifies all audio files referenced in CSV exist in `audios/` as `{id}.wav`
  - Shows error and lets user re-enter path if validation fails

- **TTS input validation** (`validateTtsInput()`):
  - Checks input file exists and is a `.csv` file
  - Validates CSV has required columns: `id`, `text`
  - Checks CSV is not empty
  - Shows error and lets user re-enter path if validation fails

**Input validation (in Python CLI):**

The eval scripts also validate inputs but exit with error instead of prompting:

- `validate_stt_input_dir(input_dir, input_file_name)` in `calibrate/stt/eval.py`
- `validate_tts_input_file(input_path)` in `calibrate/tts/eval.py`
- These are exported from `calibrate.stt` and `calibrate.tts` for programmatic use

**Output directory validation (in Ink UI):**

The `ConfigOutputStep` component checks if output directories exist for selected providers:

- Lists provider directories that already contain data
- Shows confirmation prompt: "Do you want to overwrite existing results?"
- If confirmed, sets `overwrite: true` in `EvalConfig` and passes `--overwrite` flag to CLI
- If declined, lets user enter a different output directory
- The Python CLI handles the actual overwriting (not the UI)

**Results CSV validation (in Python CLI):**

When resuming an evaluation (not using `--overwrite`), the eval scripts validate that any existing `results.csv` has the expected column structure:

- **STT expected columns:** `id`, `gt`, `pred`, `wer`, `string_similarity`, `llm_judge_score`, `llm_judge_reasoning`
- **TTS expected columns:** `id`, `text`, `audio_path`, `ttfb`, `llm_judge_score`, `llm_judge_reasoning`
- `validate_existing_results_csv()` function in both `stt/eval.py` and `tts/eval.py`
- If columns are missing or incompatible, exits with error suggesting `--overwrite` or manual deletion
- Empty files (headers only) are considered valid

**Ink UI flow** (handled by the `EvalApp` component in `app.tsx`):

1. **Language selection** — user picks a language first
2. **Provider selection** — only providers that support the chosen language are shown (filtered using `languages` in `providers.ts`, derived from the STT/TTS language dictionaries in `calibrate/utils.py`). Uses `STT_PROVIDERS` or `TTS_PROVIDERS` based on mode.
3. **Input path** — for TTS: path to `id,text` CSV file; for STT: path to directory containing `stt.csv` and `audios/`. **Full validation** runs before proceeding (CSV columns, audio file existence). A documentation link is displayed below the input prompt pointing to the relevant docs page (`https://calibrate.artpark.ai/cli/speech-to-text` or `https://calibrate.artpark.ai/cli/text-to-speech`).
4. **Output directory** — optional, defaults to `./out`. **Prompts for overwrite confirmation** if provider directories already contain data. If confirmed, `--overwrite` is passed to the eval CLI.
5. **API key setup** — checks `~/.calibrate/credentials.json` and env vars; only prompts for missing keys; always requires `OPENAI_API_KEY` for the LLM judge
6. **Evaluation** — The Ink UI's `RunStep` component spawns `calibrate <mode> -p <provider> ...` for each provider individually via `child_process.spawn`. **Max 2 providers run in parallel** to balance speed vs resource usage. Port availability is checked before starting each provider. The last provider's eval includes `--leaderboard` flag to generate the leaderboard. **Parallel provider logs are displayed side-by-side in vertical columns** (not stacked rows) for easy comparison. Real-time log streaming is enabled via `PYTHONUNBUFFERED=1` environment variable.
7. **Results** — displays leaderboard table + charts after all providers complete. The `LeaderboardStep` component now supports two views:
   - **Leaderboard view** (default): Shows comparison table, bar charts for each metric, and output file paths. Below the charts, a "View Provider Details" menu allows selecting a provider to inspect.
   - **Provider detail view**: Shows row-by-row results from the provider's `results.csv` in a scrollable table:
     - **STT**: ID, Ground Truth, Prediction, WER, String Similarity, LLM Judge (Pass/Fail)
     - **TTS**: Play button, ID, Text, TTFB, LLM Judge (Pass/Fail) — with **audio playback** support
     - **LLM Judge Reasoning**: Below the table, shows the full reasoning text for each visible row, color-coded (green for Pass, red for Fail)
     - Supports scrolling with ↑↓ keys (max 10 rows visible at a time)
     - Press `q` or `Esc` to return to leaderboard view
     - Uses `parseCSVLine()` helper function to properly handle quoted CSV fields containing commas (e.g., `"Madam, my name is Geeta"`)
   - **TTS Audio Playback**: In the TTS provider detail view, users can play generated audio files:
     - Navigate rows with ↑↓ arrow keys (selected row highlighted in cyan)
     - Press `Enter` or `p` to play/stop audio for the selected row
     - Press `s` to stop currently playing audio
     - Playing audio shows "▶ Stop" in green; idle rows show "Play"
     - Uses `afplay` on macOS, `aplay` on Linux to play WAV files from `{outputDir}/{provider}/audios/{id}.wav`
     - Audio playback state is tracked via `playingAudio` (current row ID) and `audioProcessRef` (child process)
     - Audio automatically stops when navigating back to leaderboard or exiting
   - Select "Exit" from the provider menu to quit the application

### LLM Tests Flow (`mode = "llm"`)

Handled by `LlmTestsApp` in `llm-app.tsx`. Flow with free-form model entry and parallel execution:

1. **Config file path** — path to a JSON config file containing system prompt, tools, and test cases
2. **Provider** — OpenRouter or OpenAI (asked first to determine model examples)
3. **Model entry** — **Free-form text input** (one model at a time) with confirmation loop:
   - User enters a model name using `TextInput`
   - Provider-specific examples are shown (not a fixed list):
     - **OpenAI examples**: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`, `o1`, `o1-mini`, `o3-mini`
     - **OpenRouter examples**: `openai/gpt-4.1`, `anthropic/claude-sonnet-4`, `google/gemini-2.0-flash-001`
   - Platform hints shown: "Enter model name exactly as it appears on OpenAI (platform.openai.com)" or "...OpenRouter (openrouter.ai/models)"
   - After entering a model, user is asked "Add another model?" with Yes/No options
   - Duplicate models are rejected with an error message
   - Pressing Enter with no input: uses default model if no models selected, otherwise goes to confirmation
   - Default model: `gpt-4.1` for OpenAI, `openai/gpt-4.1` for OpenRouter
4. **Output directory** — where results will be saved (default: `./out`). **Prompts for overwrite confirmation** if output directory already contains model directories or other data. If confirmed, sets `overwrite: true` in config. If declined, lets user enter a different path.
5. **API key setup** — prompts only for missing keys (OPENAI_API_KEY, OPENROUTER_API_KEY)
6. **Run** — Spawns `calibrate llm -c <path> -o <dir> -m <model> -p <provider>` **for each model individually** via `child_process.spawn`. **Max 2 models run in parallel** (same pattern as STT/TTS). After all models complete, leaderboard is generated via `python -m calibrate.llm.tests_leaderboard`. **Parallel model logs are displayed side-by-side in vertical columns** for easy comparison. Real-time log streaming is enabled via `PYTHONUNBUFFERED=1` environment variable. Each model shows:
   - Status indicator (spinner for running, + for done, x for error, - for waiting)
   - Model name
   - Pass/fail count after completion (e.g., "5/8 passed")
7. **Results** — shows full leaderboard display with two views (same pattern as STT/TTS):
   - **Leaderboard view** (default):
     - Comparison table with model, passed, failed, total, pass_rate
     - Bar charts for "Pass Rate" and "Overall Score" (if available)
     - "View Model Details" menu to select a model to inspect
     - Output file paths
   - **Model detail view**:
     - Shows per-test results in bordered boxes (green border for Pass, red for Fail)
     - Each test box displays:
       - **Header**: Test ID + pass/fail status (✓ Pass / ✗ Fail)
       - **History**: Conversation history as `role: content` pairs (e.g., "assistant: Hello!", "user: My name is John")
       - **Criteria**: Evaluation type and expected output/behavior (for `response` type: text criteria; for `tool_call` type: expected tool calls as `tool_name(args)`)
       - **Actual Output**: What the LLM actually produced (response text or tool calls as `tool_name(args)`)
       - **Reasoning**: The pass/fail reasoning, color-coded (green for Pass, red for Fail)
     - Supports scrolling with ↑↓ keys (max 10 rows visible at a time)
     - Press `q` or `Esc` to return to leaderboard view
   - Output file paths:
     - Results: `<output>/<model>/results.json`
     - Logs: `<output>/<model>/results.log`
     - Leaderboard: `<output>/leaderboard/llm_leaderboard.xlsx`
     - Charts: `<output>/leaderboard/`

### Simulations Flow (`mode = "simulations"`)

Handled by `SimulationsApp` in `sim-app.tsx`. Flow mirrors STT/TTS pattern with multi-select for models (text simulations) and parallel execution:

1. **Simulation type** — text (LLM-only) or voice (full STT → LLM → TTS)
2. **Config file path** — path to a JSON config containing system prompt, personas, scenarios, evaluation criteria
3. **Provider** — OpenRouter or OpenAI (text simulations only)
4. **Model selection** — **Multi-select** (text simulations only, like provider selection in STT/TTS). Uses `MultiSelect` component with provider-specific model options:
   - **OpenAI**: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`, `o1`, `o1-mini`, `o3-mini`
   - **OpenRouter**: `openai/gpt-4.1`, `openai/gpt-4.1-mini`, `openai/gpt-4o`, `anthropic/claude-sonnet-4`, `anthropic/claude-3.5-sonnet`, `google/gemini-2.0-flash-001`, `google/gemini-2.5-pro-preview`
   - Space to toggle selection, Enter to confirm
   - Each selected model will run all persona × scenario combinations
5. **Parallel count** — number of concurrent simulations per model (text only, default: 1)
6. **Output directory** — where results will be saved (default: `./out`). **Prompts for overwrite confirmation** if output directory already contains simulation folders, `metrics.json`, or `results.csv`. If confirmed, sets `overwrite: true` in config. If declined, lets user enter a different path.
7. **API key setup** — prompts only for missing keys
8. **Run** — For text simulations: spawns `calibrate simulations --type text -c <path> -m <model> ...` **for each model individually** via `child_process.spawn`. **Max 2 models run in parallel** (same pattern as STT/TTS). For voice simulations: single process. **Parallel model logs are displayed side-by-side in vertical columns**. After all simulations complete, spawns `calibrate simulations leaderboard` to aggregate results.
9. **Results** — shows full leaderboard display with two views (same pattern as STT/TTS):
   - **Leaderboard view** (default):
     - Comparison table with dynamic metric columns from `metrics.json`
     - Bar charts for top metrics
     - "View Model Details" menu to select a model to inspect (text simulations)
     - Output file paths
   - **Model detail view**:
     - Shows per-simulation results in a scrollable table (Persona, Scenario, metric columns)
     - Supports scrolling with ↑↓ keys (max 10 rows visible at a time)
     - Press `q` or `Esc` to return to leaderboard view
   - Output directory path

### Resource Management (not in menu — code retained in `resource-app.tsx`)

Handled by `ResourceListScreen` in `resource-app.tsx`. Not currently accessible from the main menu or CLI. The code is retained for future use. Three-mode view:

1. **List view** — Items shown as a selectable list with summaries (e.g. persona shows name + language + gender, agent shows name + tool count). Select an item to view details, or pick "Create new" / "Back to menu".
2. **Detail view** — Shows all fields of the selected resource with current values (truncated at 80 chars). Offers per-field "Edit" options for editable fields, plus "Delete" and "Back to list". Complex fields like tools and parameters are read-only in the detail view.
3. **Edit mode** — Field-type-aware editors:
   - Simple text (`name`) → `TextInput` pre-filled with current value
   - Long text (`system_prompt`, `characteristics`, `description`) → `TextArea` pre-filled
   - Select fields (`gender`, `language`, `interruption_sensitivity`, `type`, `agent_speaks_first`) → `SelectInput` with appropriate options
   - Numeric (`max_turns`) → `TextInput` with integer parsing

Field definitions per resource type are in the `getFieldDefs()` helper. The `updaters` map routes to the correct `update*()` function in `storage.ts`.

### Resource Creation Wizards

All creation flows in `resource-app.tsx` are retained for future use. Currently not accessible from the UI:

- **CreateToolFlow** — type (structured output/webhook) → name → description → parameters loop (name, type, description, required) → webhook details (method, URL)
- **CreatePersonaFlow** — label → characteristics (TextArea) → gender → language → interruption sensitivity
- **CreateScenarioFlow** — label → description (TextArea)
- **CreateMetricFlow** — name → evaluation instructions (TextArea)
- **CreateAgentFlow** — name → system prompt (TextArea) → select/create tools → who speaks first → max turns

### Persistent Resource Storage (`storage.ts`)

Resources are stored as JSON files in `~/.calibrate/`:

- `agents.json` — agents with system prompt, tools (inline copies), and settings
- `tools.json` — tool definitions (structured output or webhook)
- `personas.json` — persona definitions (characteristics, gender, language, interruption sensitivity)
- `scenarios.json` — scenario definitions (name, description)
- `metrics.json` — evaluation criteria (name, description)

Each item has a UUID-based `id` field. Tools stored inside agents are full inline copies (not references), so editing a tool in the tools menu doesn't affect existing agents.

CRUD operations: `list*()`, `add*()`, `update*()`, `remove*()` for each resource type. `update*()` takes an `id` and a `Partial<Omit<T, 'id'>>` and merges updates into the existing item.

Config builder functions (`buildTestsConfig`, `buildTextSimConfig`, `buildVoiceSimConfig`) convert stored resources into the JSON format expected by the Python CLI.

### Bundling & Distribution

- `cd ui && npm run bundle` uses esbuild to compile the entire Ink app into a single self-contained `calibrate/ui/cli.bundle.mjs` (~2MB)
- `pyproject.toml` includes `cli.bundle.mjs` via `[tool.setuptools.package-data]` so it ships with the Python package
- `calibrate/cli.py` dispatches all interactive commands via the shared `_launch_ink_ui(mode)` helper, which checks for `node` then runs `node calibrate/ui/cli.bundle.mjs <mode>`
- STT and TTS commands accept arguments directly (no `eval` subcommand): `calibrate stt -p deepgram -i ./data` runs evaluation, `calibrate stt` without args launches interactive UI
- Hidden internal CLI subcommands (`llm leaderboard`, `simulations leaderboard`, `agent test`) still exist in argparse because the Ink UI spawns them as child processes
- The `--leaderboard` flag on STT/TTS commands triggers leaderboard generation after evaluation completes
- If Node.js is not installed, the user gets a clear error with instructions

### Key Patterns

- **Calibrate binary resolution** (`findCalibrateBin()` in `shared.ts`): Checks PATH → `.venv/bin/calibrate` → `uv run calibrate`, in that order. Shared across all flow files via import.
- **Credential storage** (`credentials.ts`): Keys saved to `~/.calibrate/credentials.json` with `0o600` permissions. Reads from stored file first, then falls back to env vars.
- **Language-based provider filtering** (`providers.ts`): Each `ProviderInfo` has a `languages: Set<string>`. Mode-aware helpers `getProvidersForLanguage(language, mode)` and `getProviderById(id, mode)` select from the correct provider list.
- **Config file based flows**: LLM tests and simulations flows now collect a user-provided config file path and pass it directly to the Python CLI commands. No temp files are created by the Ink UI for these flows.
- **TextArea component** (`components.tsx`): Multi-line text input where Enter adds a newline and Escape submits. Used for system prompts, persona characteristics, scenario descriptions, and evaluation criteria.
- **Step navigation with Escape key**: All interactive flows (STT, TTS, LLM) support going back to the previous step by pressing Escape. Each step component accepts an `onBack` prop and uses `useInput` to listen for `key.escape`. The "running" and "done" steps don't support going back. A "Press Esc to go back" hint is shown on each step.
- **Parallel log display** (`RunStep` in `app.tsx`, `LlmTestsApp` in `llm-app.tsx`, `SimulationsApp` in `sim-app.tsx`): When multiple providers/models/simulations run in parallel, their logs are displayed side-by-side in vertical columns (using `flexDirection="row"` with percentage-based widths). Each column shows the last 8-10 log lines, truncated to 45-50 characters. Log buffer keeps 15-20 lines per slot. STT/TTS and LLM tests parse terminal output to route logs to slots. **Simulations use a different approach**: they poll the output directory for `simulation_persona_X_scenario_Y` folders and read `results.log` files directly from each folder. Columns are labeled "Persona X Scenario Y" (not generic "Slot N"). Polling interval is 500ms.
- **Explicit keyboard instructions**: All interactive UI components show clear keyboard hints to guide users:
  - **SelectInput** (`components.tsx`): Shows "↑↓ navigate Enter select" below the options list
  - **MultiSelect** (`components.tsx`): Shows "Space toggle a all Enter confirm" below the options list (no up/down hint since cursor already indicates navigation)
  - **TextInput**: Steps using TextInput show "Enter to submit" hints in dimColor text (e.g., "Enter to submit, Esc to go back" or "Enter to submit (default: ./out)")
  - **TextArea** (`components.tsx`): Shows "enter: new line esc: done" when active
  - All hints use `<Text dimColor>` for consistent styling

### Dependencies

`ink`, `react`, `react-devtools-core`, `esbuild` (dev), `tsx` (dev), `typescript` (dev). End users only need Node.js — all JS deps are bundled.

### Gotchas

- Ink requires a real TTY for interactive input (won't work in piped/non-interactive shells)
- After changing UI source code, run `cd ui && npm run bundle` to regenerate `calibrate/ui/cli.bundle.mjs` before releasing
- `languages` sets in `providers.ts` must be kept in sync with the language dictionaries in `calibrate/utils.py`
- The bundle file (`cli.bundle.mjs`) should be committed to the repo so it ships with `pip install`
- Hidden CLI subcommands (`llm leaderboard`, `simulations leaderboard`) must NOT be removed — the Ink UI spawns them as child processes. They are hidden from `--help` by registering subparsers without `help=` and using `metavar=""` on the parent subparser group
- **STT/TTS/LLM architecture split**: Each module follows the same pattern:
  - `eval.py` / `run_tests.py` — handles single-provider/model evaluation
  - `benchmark.py` — handles multi-provider/model parallel execution + auto-leaderboard
  - `leaderboard.py` / `tests_leaderboard.py` — handles leaderboard generation
  - The CLI routes based on provider/model count: single → `eval.py` / `run_tests.py`, multiple → `benchmark.py`
- **Ink UI vs CLI parallelization**: The Ink UI (`calibrate stt/tts/llm` interactive mode) handles parallelization in TypeScript via `RunStep` component (spawning individual single-provider/model commands that route to `eval.py` / `run_tests.py`). The direct CLI with multiple providers/models handles parallelization in Python via `benchmark.py`.
- **Leaderboard generation timing**:
  - **STT/TTS Ink UI**: The last provider's call includes `--leaderboard` flag which triggers leaderboard generation after eval completes.
  - **LLM Ink UI**: After all model evaluations complete, leaderboard is generated via `python -m calibrate.llm.tests_leaderboard` (no `--leaderboard` flag on individual model runs).
  - **Direct CLI with multiple providers/models**: `benchmark.py` always generates leaderboard after all complete.
- Tools stored inside agents in `~/.calibrate/agents.json` are inline copies. Updating a tool via `calibrate tools` does not update agents that already contain a copy of that tool
- The `App` component's `Mode` type is re-exported as `AppMode` from `shared.ts` with values: `menu`, `stt`, `tts`, `llm`, `simulations`. The internal `EvalMode` type (`"tts" | "stt"`) is separate and only used within `app.tsx` for STT/TTS-specific components
- **Real-time log streaming**: The Ink UI sets `PYTHONUNBUFFERED=1` when spawning Python subprocesses to ensure logs appear immediately. The `log_and_print()` function in `calibrate/utils.py` uses `print(..., flush=True)` for the same reason. Without these, Python buffers stdout when piped to a subprocess, causing logs to appear in batches instead of real-time.
- **Loguru logger thread safety in LLM tests**: The `run_inference()` function in `calibrate/llm/run_tests.py` uses a threading lock (`_logger_lock`) to protect `logger.add()` and `logger.remove()` operations. When running multiple models in parallel via `benchmark.py`, concurrent logger handler manipulation can cause "There is no existing handler with id X" errors. The lock prevents this race condition.

---

## PyPI Release

**Package name:** `calibrate-agent` (on PyPI)

**Publishing workflow** (`.github/workflows/publish.yml`):

1. Triggered by a GitHub release being published
2. Checks out with `fetch-depth: 0` (full git history required by `setuptools_scm`)
3. Builds sdist + wheel using `python -m build` — `setuptools_scm` reads the version from the git tag automatically
4. Publishes to PyPI using trusted publishing (`pypa/gh-action-pypi-publish`) with OIDC `id-token` — no API key needed
5. Uses a `pypi` GitHub environment for deployment approval control

**`pyproject.toml` key details:**

- `dynamic = ["version"]` — version is not hardcoded; derived by `setuptools_scm` from git tags
- `license = "CC-BY-SA-4.0"` — plain SPDX string (not a table; the table format `{text = "..."}` is deprecated by setuptools)
- `license-files = ["LICENSE.md"]` — explicitly declares the license file
- `[tool.setuptools.packages.find]` excludes `ui*`, `examples*`, `docs*`, `images*` to prevent non-package directories from leaking into the wheel
- `[tool.setuptools.package-data]` includes `calibrate/ui/cli.bundle.mjs` so the bundled Ink UI ships with the package
- `pipecat-ai` and `pipecat-ai-small-webrtc-prebuilt` are pinned to exact versions (see Tech Stack section)
- `[tool.setuptools_scm]` with `local_scheme = "no-local-version"` and `fallback_version = "0.0.0-dev"`
- Build requires `setuptools>=64` and `setuptools_scm>=8`

**Versioning (via `setuptools_scm`):**

- **No hardcoded version anywhere** — `setuptools_scm` reads git tags at build time
- Tag `v0.1.0` on the current commit → version `0.1.0`
- 3 commits past `v0.1.0` → version `0.1.1.dev3` (dev version)
- No tags at all → `fallback_version` of `0.0.0-dev`
- **`calibrate/__init__.py`** reads the version dynamically via `importlib.metadata.version("calibrate-agent")`. Falls back to `"0.0.0-dev"` if the package isn't installed (e.g., during local development without `pip install -e .`)
- **`ui/package.json`** version is independent — the UI is bundled into the Python package and end users never interact with the npm package directly
- **`uv.lock`** tracks dependency versions, not the project version — only needs updating when dependencies change (`uv lock`)
- **Release tags must use `v` prefix** (e.g., `v0.1.0`, `v1.0.0`)
- **CI checkout needs `fetch-depth: 0`** — `setuptools_scm` requires full git history to find tags

**Pre-release checklist:**

1. Rebuild the Ink UI bundle if UI changed: `cd ui && npm run bundle`
2. Verify build locally: `python -m build` — check no stray files in the wheel
3. Create a GitHub release with a `v`-prefixed tag (e.g., `v0.1.0`) — `setuptools_scm` reads the tag, builds the correct version, and publishes automatically

---

## Coding Standards

1. **Async/Await:** All evaluation functions are async
2. **Type Hints:** Use `Literal` for constrained string parameters
3. **Module Structure:** Each module exposes a clean public API via `__init__.py`
4. **Output Organization:** Consistent directory structure across all modules
5. **Logging:** Dual logging to terminal and log files with parallel-safe per-simulation loggers
6. **Error Handling:** Let errors propagate (no silent catching) so Sentry captures them automatically
7. **Checkpointing:** Resume from interruption using existing results
8. **Results CSV Validation:** Before resuming, validate that existing `results.csv` has expected columns

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
- **System error handling:** If LLM inference returns no response and no tool calls, the script raises `LLMInferenceError` and exits with code 1. Error details are captured via loguru's error sink pattern (same as `run_simulation.py`) and included in the exception message. The `run_inference` function wraps `_run_inference_inner` to capture ERROR/CRITICAL level logs, which are then surfaced in the `LLMInferenceError` message (e.g., "Captured errors: gpt-4.3902 is not a valid model ID").

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
