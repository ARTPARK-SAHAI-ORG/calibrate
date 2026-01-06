# Configuration Reference

Reference for all configuration file formats.

## STT/TTS Config

No separate config file - uses command-line arguments.

## LLM Test Config

```json
{
  "system_prompt": "...",
  "tools": [...],
  "test_cases": [...]
}
```

## LLM Simulation Config

```json
{
  "agent_system_prompt": "...",
  "personas": [...],
  "scenarios": [...],
  "evaluation_criteria": [...]
}
```

## Agent Config

```json
{
  "agent_system_prompt": "...",
  "language": "english",
  "tools": [...],
  "personas": [...],
  "scenarios": [...]
}
```
