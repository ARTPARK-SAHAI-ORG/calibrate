# CLI Reference

Complete command-line reference for Pense.

## STT Commands

### Evaluate STT

```bash
pense stt eval -p <provider> -l <language> -i <input-dir> -o <output-dir>
```

### Generate Leaderboard

```bash
pense stt leaderboard -o <output-dir> -s <save-dir>
```

## TTS Commands

### Evaluate TTS

```bash
pense tts eval -p <provider> -l <language> -i <input.csv> -o <output-dir>
```

### Generate Leaderboard

```bash
pense tts leaderboard -o <output-dir> -s <save-dir>
```

## LLM Commands

### Run Tests

```bash
pense llm tests run -c <config.json> -o <output-dir> -m <model>
```

### Run Simulations

```bash
pense llm simulations run -c <config.json> -o <output-dir> -m <model>
```

## Agent Commands

### Interactive Test

```bash
pense agent test -c <config.json> -o <output-dir>
```

### Simulation

```bash
pense agent simulation -c <config.json> -o <output-dir>
```
