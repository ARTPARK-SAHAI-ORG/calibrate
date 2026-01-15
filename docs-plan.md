# Pense Documentation Structure Plan

## Overview

This plan outlines a comprehensive, user-focused documentation structure for Pense using MkDocs with Material theme. The structure is designed to help developers quickly understand and use the framework effectively.

---

## Documentation Structure

```
docs/
├── index.md                    # Landing page - What is Pense?
├── getting-started/
│   ├── index.md               # Quick start guide
│   ├── installation.md        # Installation and setup
│   ├── configuration.md       # Environment setup and API keys
│   └── first-evaluation.md    # Your first evaluation (5-minute tutorial)
├── concepts/
│   ├── index.md               # Core concepts overview
│   ├── evaluation-workflow.md # How evaluations work
│   ├── metrics.md             # Understanding metrics (WER, latency, etc.)
│   ├── providers.md           # Supported providers overview
│   └── output-structure.md    # Understanding output files
├── guides/
│   ├── stt/
│   │   ├── index.md           # STT evaluation overview
│   │   ├── preparing-data.md  # How to prepare STT test data
│   │   ├── running-eval.md    # Running STT evaluations
│   │   ├── understanding-results.md # Interpreting STT results
│   │   └── leaderboard.md     # Generating STT leaderboards
│   ├── tts/
│   │   ├── index.md           # TTS evaluation overview
│   │   ├── preparing-data.md  # How to prepare TTS test data
│   │   ├── running-eval.md    # Running TTS evaluations
│   │   ├── understanding-results.md # Interpreting TTS results
│   │   └── leaderboard.md     # Generating TTS leaderboards
│   ├── llm/
│   │   ├── index.md           # LLM evaluation overview
│   │   ├── tests/
│   │   │   ├── index.md       # LLM tests overview
│   │   │   ├── creating-tests.md # Creating test configurations
│   │   │   ├── running-tests.md # Running LLM tests
│   │   │   ├── evaluation-types.md # Tool call vs response evaluation
│   │   │   └── leaderboard.md # Generating test leaderboards
│   │   └── simulations/
│   │       ├── index.md       # LLM simulations overview
│   │       ├── creating-config.md # Creating simulation configs
│   │       ├── personas-scenarios.md # Defining personas and scenarios
│   │       ├── evaluation-criteria.md # Setting up evaluation criteria
│   │       ├── running-simulations.md # Running simulations
│   │       └── leaderboard.md # Generating simulation leaderboards
│   └── agent/
│       ├── index.md           # Agent evaluation overview
│       ├── interactive-testing.md # Using the interactive agent test
│       ├── agent-simulations.md # Running automated agent simulations
│       ├── config-reference.md # Agent configuration reference
│       └── understanding-outputs.md # Understanding agent simulation outputs
├── reference/
│   ├── cli-reference.md       # Complete CLI command reference
│   ├── config-reference.md    # Configuration file schemas
│   ├── providers.md           # Detailed provider information
│   └── metrics-reference.md   # Detailed metrics documentation
├── examples/
│   ├── index.md               # Examples overview
│   ├── stt-quick-eval.md      # Quick STT evaluation example
│   ├── tts-quick-eval.md      # Quick TTS evaluation example
│   ├── llm-test-example.md    # LLM test example
│   ├── llm-simulation-example.md # LLM simulation example
│   └── agent-simulation-example.md # Agent simulation example
├── troubleshooting/
│   ├── index.md               # Common issues and solutions
│   ├── api-keys.md            # API key setup issues
│   ├── data-format.md         # Data format errors
│   └── performance.md         # Performance optimization tips
└── faq.md                     # Frequently asked questions
```

---

## Detailed Content Plan

### 1. **index.md** (Landing Page)

**Purpose**: First impression - what Pense is and why use it

**Content**:

- Brief introduction: "An open-source simulation and testing framework for voice agents"
- Key value propositions:
  - Move from manual to automated testing
  - Evaluate components in isolation
  - Test across multiple vendors
  - Simulate thousands of scenarios
- Quick visual: What Pense does (diagram showing STT → LLM → TTS pipeline)
- Main use cases:
  - Comparing STT/TTS providers
  - Testing LLM behavior
  - Simulating voice agent conversations
- Quick links to getting started
- Link to GitHub

---

### 2. **Getting Started Section**

#### **getting-started/index.md**

- Overview of the getting started journey
- Prerequisites (Python 3.10+, uv)
- What you'll learn
- Next steps navigation

#### **getting-started/installation.md**

- Installing uv (if needed)
- Installing Pense (`uv sync --frozen`)
- Verifying installation (`pense --help`)
- Common installation issues

#### **getting-started/configuration.md**

- Setting up `.env` file
- Required API keys by component
- Provider-specific setup:
  - STT providers (Deepgram, Google, OpenAI, etc.)
  - TTS providers (ElevenLabs, Cartesia, Google, etc.)
  - LLM providers (OpenAI, OpenRouter)
- Testing API keys
- Security best practices

#### **getting-started/first-evaluation.md**

- Step-by-step tutorial: Run your first STT evaluation
- Using sample data
- Understanding the output
- Next steps

---

### 3. **Concepts Section**

#### **concepts/index.md**

- Core concepts overview
- How Pense fits into voice agent development
- Component isolation philosophy
- Evaluation vs simulation distinction

#### **concepts/evaluation-workflow.md**

- How evaluations work (input → processing → output)
- Batch processing concept
- Metrics collection
- Output file structure

#### **concepts/metrics.md**

- STT metrics: WER, string similarity, LLM judge score, latency
- TTS metrics: Latency (TTFB, processing time)
- LLM metrics: Pass/fail rates, evaluation scores
- What each metric means
- How to interpret them

#### **concepts/providers.md**

- What is a provider?
- Supported providers by component
- Provider capabilities and limitations
- Choosing the right provider

#### **concepts/output-structure.md**

- Common output structure
- File types: CSV, JSON, logs, audio files
- Understanding directory organization
- Finding specific results

---

### 4. **Guides Section**

### **STT Guides**

#### **guides/stt/index.md**

- What is STT evaluation?
- When to use it
- Quick overview of the workflow
- Links to detailed guides

#### **guides/stt/preparing-data.md**

- Required directory structure
- CSV format (`id, text`)
- Audio formats (WAV, PCM16)
- Sample data structure
- Tips for creating good test data
- Common mistakes

#### **guides/stt/running-eval.md**

- Command syntax: `pense stt eval`
- All parameters explained:
  - `-p, --provider`: Provider selection
  - `-l, --language`: Language options
  - `-i, --input-dir`: Input directory
  - `-o, --output-dir`: Output directory
  - `-d, --debug`: Debug mode
- Example commands
- What happens during execution
- Monitoring progress

#### **guides/stt/understanding-results.md**

- Output directory structure
- `results.csv`: Columns explained
- `metrics.json`: Structure and meaning
- `logs/`: How to use logs
- Interpreting WER scores
- Understanding latency metrics
- Common result patterns

#### **guides/stt/leaderboard.md**

- What is a leaderboard?
- When to use it
- Command: `pense stt leaderboard`
- Output files (Excel, PNG charts)
- Reading leaderboard results
- Comparing providers

### **TTS Guides** (Similar structure to STT)

#### **guides/tts/index.md**

- What is TTS evaluation?
- When to use it
- Quick overview

#### **guides/tts/preparing-data.md**

- CSV format (single `text` column)
- Sample data examples
- Best practices

#### **guides/tts/running-eval.md**

- Command syntax and parameters
- Examples
- Execution flow

#### **guides/tts/understanding-results.md**

- Output structure
- Audio files location
- Metrics interpretation
- Latency analysis

#### **guides/tts/leaderboard.md**

- Generating leaderboards
- Comparing TTS providers

### **LLM Guides**

#### **guides/llm/index.md**

- LLM evaluation overview
- Tests vs Simulations distinction
- When to use each

#### **guides/llm/tests/index.md**

- What are LLM tests?
- Use cases: verifying tool calls, response quality
- Quick overview

#### **guides/llm/tests/creating-tests.md**

- Test configuration structure
- System prompts
- Tools definition
- Test cases structure
- History format
- Evaluation types (tool_call vs response)
- Variables and templating
- Complete example

#### **guides/llm/tests/running-tests.md**

- Command: `pense llm tests run`
- Parameters explained
- Example commands
- Understanding output

#### **guides/llm/tests/evaluation-types.md**

- Tool call evaluation: How it works, examples
- Response evaluation: LLM judge, criteria, examples
- Choosing the right evaluation type

#### **guides/llm/tests/leaderboard.md**

- Generating test leaderboards
- Comparing models/scenarios

#### **guides/llm/simulations/index.md**

- What are LLM simulations?
- Use cases: end-to-end conversation testing
- Quick overview

#### **guides/llm/simulations/creating-config.md**

- Configuration structure
- Agent system prompt
- Tools definition
- Personas and scenarios
- Evaluation criteria

#### **guides/llm/simulations/personas-scenarios.md**

- What are personas?
- Creating effective personas
- What are scenarios?
- Creating scenarios
- Persona-scenario combinations
- Examples

#### **guides/llm/simulations/evaluation-criteria.md**

- What are evaluation criteria?
- Defining good criteria
- LLM judge evaluation
- Examples

#### **guides/llm/simulations/running-simulations.md**

- Command: `pense llm simulations run`
- Parameters
- Execution flow
- Understanding output

#### **guides/llm/simulations/leaderboard.md**

- Generating simulation leaderboards
- Aggregating results

### **Agent Guides**

#### **guides/agent/index.md**

- Agent evaluation overview
- Interactive testing vs simulations
- When to use each

#### **guides/agent/interactive-testing.md**

- What is interactive testing?
- Use case: Manual testing with real audio
- Command: `pense agent test`
- Configuration file
- Starting the server
- Using the web UI
- Understanding metrics tab
- Output files
- Tips for effective testing

#### **guides/agent/agent-simulations.md**

- What are agent simulations?
- Use case: Automated voice agent testing
- Command: `pense agent simulation`
- Configuration structure
- Personas and scenarios
- Execution flow
- Understanding output

#### **guides/agent/config-reference.md**

- Complete configuration schema
- System prompt guidelines
- Tools configuration
- Provider configuration
- Language settings
- Advanced options

#### **guides/agent/understanding-outputs.md**

- Output directory structure
- Audio files (`audios/`)
- Transcripts (`transcript.json`)
- Tool calls (`tool_calls.json`)
- STT outputs (`stt_outputs.json`)
- Metrics (`metrics.json`)
- Logs
- How to analyze results

---

### 5. **Reference Section**

#### **reference/cli-reference.md**

- Complete CLI command reference
- All commands and subcommands
- All flags and options
- Examples for each command
- Organized by component

#### **reference/config-reference.md**

- JSON schema for all config types
- Required vs optional fields
- Type definitions
- Validation rules
- Examples

#### **reference/providers.md**

- Detailed provider information
- STT providers: capabilities, setup, limitations
- TTS providers: capabilities, setup, limitations
- LLM providers: models, setup
- Provider comparison tables

#### **reference/metrics-reference.md**

- Complete metrics documentation
- Calculation methods
- Interpretation guidelines
- Normal ranges
- Troubleshooting metrics

---

### 6. **Examples Section**

#### **examples/index.md**

- Examples overview
- Quick links to all examples
- Prerequisites

#### **examples/stt-quick-eval.md**

- Complete walkthrough
- Sample data setup
- Running evaluation
- Interpreting results

#### **examples/tts-quick-eval.md**

- Complete walkthrough
- Sample data setup
- Running evaluation
- Interpreting results

#### **examples/llm-test-example.md**

- Complete test configuration
- Running tests
- Understanding results

#### **examples/llm-simulation-example.md**

- Complete simulation configuration
- Running simulation
- Understanding results

#### **examples/agent-simulation-example.md**

- Complete agent simulation configuration
- Running simulation
- Understanding results

---

### 7. **Troubleshooting Section**

#### **troubleshooting/index.md**

- Common issues overview
- Quick links
- How to get help

#### **troubleshooting/api-keys.md**

- API key not found errors
- Invalid API key errors
- Provider-specific issues
- Environment variable setup

#### **troubleshooting/data-format.md**

- CSV format errors
- Audio format issues
- Directory structure problems
- Common mistakes and fixes

#### **troubleshooting/performance.md**

- Slow evaluations
- Memory issues
- Debug mode usage
- Optimization tips

---

### 8. **FAQ**

- Frequently asked questions
- Common misconceptions
- Quick answers
- Links to detailed docs

---

## MkDocs Configuration Structure

```yaml
# mkdocs.yml structure
site_name: Pense Documentation
site_description: Documentation for Pense - Voice Agent Testing Framework
site_url: https://your-site-url.com
repo_url: https://github.com/your-repo/pense
repo_name: pense

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.annotate
    - content.code.copy

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - admonition
  - pymdownx.details
  - pymdownx.tabbed:
      alternate_style: true
  - tables
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Getting Started:
      - getting-started/index.md
      - getting-started/installation.md
      - getting-started/configuration.md
      - getting-started/first-evaluation.md
  - Concepts:
      - concepts/index.md
      - concepts/evaluation-workflow.md
      - concepts/metrics.md
      - concepts/providers.md
      - concepts/output-structure.md
  - Guides:
      - STT Evaluation:
          - guides/stt/index.md
          - guides/stt/preparing-data.md
          - guides/stt/running-eval.md
          - guides/stt/understanding-results.md
          - guides/stt/leaderboard.md
      - TTS Evaluation:
          - guides/tts/index.md
          - guides/tts/preparing-data.md
          - guides/tts/running-eval.md
          - guides/tts/understanding-results.md
          - guides/tts/leaderboard.md
      - LLM Evaluation:
          - guides/llm/index.md
          - LLM Tests:
              - guides/llm/tests/index.md
              - guides/llm/tests/creating-tests.md
              - guides/llm/tests/running-tests.md
              - guides/llm/tests/evaluation-types.md
              - guides/llm/tests/leaderboard.md
          - LLM Simulations:
              - guides/llm/simulations/index.md
              - guides/llm/simulations/creating-config.md
              - guides/llm/simulations/personas-scenarios.md
              - guides/llm/simulations/evaluation-criteria.md
              - guides/llm/simulations/running-simulations.md
              - guides/llm/simulations/leaderboard.md
      - Agent Evaluation:
          - guides/agent/index.md
          - guides/agent/interactive-testing.md
          - guides/agent/agent-simulations.md
          - guides/agent/config-reference.md
          - guides/agent/understanding-outputs.md
  - Reference:
      - reference/cli-reference.md
      - reference/config-reference.md
      - reference/providers.md
      - reference/metrics-reference.md
  - Examples:
      - examples/index.md
      - examples/stt-quick-eval.md
      - examples/tts-quick-eval.md
      - examples/llm-test-example.md
      - examples/llm-simulation-example.md
      - examples/agent-simulation-example.md
  - Troubleshooting:
      - troubleshooting/index.md
      - troubleshooting/api-keys.md
      - troubleshooting/data-format.md
      - troubleshooting/performance.md
  - FAQ: faq.md
```

---

## Key Documentation Principles

1. **User-Focused**: Every page answers "How do I use this?" not "How is this implemented?"
2. **Progressive Disclosure**: Start simple, add complexity gradually
3. **Examples First**: Show examples before explaining concepts
4. **Visual Aids**: Use diagrams, code blocks, and screenshots liberally
5. **Quick Reference**: Make it easy to find specific information
6. **Consistent Structure**: Similar sections follow the same pattern
7. **Action-Oriented**: Focus on what users can do, not just what exists

---

## Content Style Guidelines

- **Tone**: Friendly, helpful, professional
- **Voice**: Second person ("you") or imperative ("Run the command...")
- **Code**: Always show complete, runnable examples
- **Structure**: Use clear headings, bullet points, and callouts
- **Cross-references**: Link liberally to related topics
- **Warnings**: Use admonitions for important notes and warnings

---

## Next Steps

1. Set up MkDocs with Material theme
2. Create the directory structure
3. Write content starting with Getting Started section
4. Add examples and reference documentation
5. Review and iterate based on user feedback
