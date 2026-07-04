# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

**Calibrate** (`calibrate-agent` on PyPI) is an open-source evaluation framework
for voice agents. It benchmarks LLMs, STT providers, TTS providers, and runs
agent simulations — all from a single CLI / Python library.

- Website / docs: https://calibrate.artpark.ai
- Built on top of [pipecat](https://github.com/pipecat-ai/pipecat).
- The CLI entry point is `calibrate` (defined in `pyproject.toml:scripts` →
  `calibrate.cli:main`).

The repo also ships an **Ink (React) terminal UI** in `ui/` that is bundled into
the Python package and launched from the CLI.

## Before making any change

For **every** change, new feature, or modification — no matter how small —
follow this process before writing code:

1. **Review the existing code thoroughly.** Search the codebase for code or
   functionality that already does (or partly does) what's being asked. Don't
   assume something doesn't exist — confirm it.
2. **Plan the work as parallel, independent agent runs.** Decompose the change
   into chunks that can be executed by multiple independent weak agents running
   in parallel (touching independent files / sections), and lay that plan out.
3. **Identify reusable code.** Call out the parts of the existing code that can
   be reused directly, or repackaged / extracted into reusable functions, to
   support what needs to be built — prefer that over duplicating logic.
4. **Prefer well-known, reliable libraries.** If a requirement is already solved
   by a mature, trustworthy library, bias toward using it instead of
   re-implementing it here — unless the requirements genuinely demand a custom
   approach.
5. **Share the plan and surface the choices — don't assume.** Present the plan
   and explicitly raise any decisions to be made, along with their tradeoffs,
   and ask the user instead of silently picking an option.

## Repository layout

```
calibrate/                 # Python package (the importable library + CLI)
├── cli.py                 # Top-level CLI entry — wires subcommands to UI/SDK
├── connections.py         # TextAgentConnection — HTTP client for external agents
├── judges.py              # text_judge / audio_judge / simulation_judge — LLM-as-judge core
├── langfuse.py            # Optional Langfuse tracing wrappers (@observe)
├── status.py              # Run-status reporting helpers
├── utils.py               # Provider language code maps, logging, validation
├── stt/
│   ├── eval.py            # Per-provider transcribe_* + transcribe_audio router
│   ├── metrics.py         # WER + LLM-judge aggregation (get_llm_judge_score)
│   ├── benchmark.py       # Multi-provider parallel runner + leaderboard
│   └── leaderboard.py     # Excel workbook generator
├── tts/
│   ├── eval.py            # Per-provider synthesize_* + synthesize_speech router
│   ├── metrics.py         # Audio LLM-judge aggregation (get_tts_llm_judge_score)
│   ├── benchmark.py       # Multi-provider parallel runner + leaderboard
│   └── leaderboard.py
├── llm/
│   ├── run_tests.py       # Tool-call / response evaluation across test cases
│   ├── run_simulation.py  # Multi-turn user-simulator conversations
│   ├── benchmark.py       # Multi-model parallel runner + leaderboard
│   ├── tests_leaderboard.py
│   ├── simulation_leaderboard.py
│   ├── metrics.py
│   └── _output.py         # Shared print_benchmark_summary
└── agent/
    ├── bot.py             # Pipecat bot bootstrap
    ├── run_simulation.py  # Voice-agent simulation driver
    └── test.py            # Voice-agent tests

tests/                     # Test suite — mirrors the calibrate/ structure
├── stt/        test_eval.py, test_metrics.py, test_leaderboard.py
├── tts/        test_eval.py, test_metrics.py, test_leaderboard.py
├── llm/        test_benchmark.py, test_run_tests.py, test_run_simulation.py,
│               test_run_simulation_integration.py, test_output.py,
│               test_tests_leaderboard.py
├── test_connections.py, test_cli.py, test_judges.py,
│   test_sdk_judge_regressions.py

ui/                        # Ink (React + TypeScript) terminal UI
├── source/                # *.tsx entry points (app, llm-app, sim-app, etc.)
├── tests/                 # vitest tests
└── package.json           # Bundled into calibrate/ui/cli.bundle.mjs

docs/                      # Mintlify docs site (.mdx)
examples/                  # Example datasets + scripts users can run
.github/workflows/         # tests.yml, publish.yml, claude.yml, claude-code-review.yml
.githooks/pre-commit       # Runs pytest before commits to main
```

## Conventions in this codebase

### Evaluator dicts everywhere
Every LLM/audio judge in the codebase takes a list of **evaluator** dicts of
this shape:

```python
{
  "name": "semantic_match",
  "system_prompt": "...",
  "judge_model": "openai/gpt-4.1",   # routed through OpenRouter
  "type": "binary" | "rating",       # binary is the default if absent
  "scale_min": 1, "scale_max": 5,    # only for rating
}
```

Helpers in `calibrate/judges.py`:
- `is_rating(evaluator)` — True if `type == "rating"`
- `evaluator_result_value(ev, row)` — pulls the score/match value out of a per-row result
- `DEFAULT_STT_EVALUATOR`, `DEFAULT_TTS_EVALUATOR`, `DEFAULT_LLM_TEST_EVALUATOR`

Result shape returned by `text_judge`/`audio_judge`:
```python
{
  "evaluator_name": {"reasoning": str, "match": bool}   # binary
  "evaluator_name": {"reasoning": str, "score": int}    # rating
}
```

### Aggregation shape
`get_llm_judge_score` / `get_tts_llm_judge_score` return:
```python
{
  "scores": {
    "name": {"type": "binary", "mean": 0.85}                            # binary
    "name": {"type": "rating", "mean": 4.0, "scale_min": 1, "scale_max": 5}  # rating
  },
  "score": float,        # mean across evaluator means (legacy top-level)
  "per_row": [ ... ],    # list of per-row dicts, same shape as text_judge output
}
```

Leaderboards detect evaluators in `metrics.json` by looking for dict values
with a `type` field — that's the marker. `wer` and `ttfb` are top-level floats
and dicts respectively.

### Routing pattern
Both `transcribe_audio` (STT) and `synthesize_speech` (TTS) are dispatch
routers wrapped in `@backoff.on_exception(...)` + `@observe(...)`. They look up
the per-provider implementation in a dict and `await` it. For unit testing,
call `router.__wrapped__(...)` to skip the decorators (the `@backoff` retry
would otherwise mask `ValueError`s).

### Keeping live-agent and eval defaults in sync
There are **two** places that talk to each STT/TTS provider, and they MUST use
the same per-provider defaults:

- **The benchmarks** — the `transcribe_*` (`stt/eval.py`) and `synthesize_*`
  (`tts/eval.py`) functions, which produce the leaderboards.
- **The live agent** — `create_stt_service` / `create_tts_service` in
  `calibrate/utils.py`, which the voice-agent simulation / tests actually run
  (`agent/bot.py`, `agent/test.py`).

If these drift, a leaderboard stops reflecting the config the agent deploys —
you'd benchmark one voice/model and ship another. **Any time you change a
provider's model, voice, endpoint, or other default on one side, change the
other side too.**

To make this structural rather than a discipline you have to remember, the
**model names and TTS voices are single-sourced**: `STT_PROVIDER_MODELS`,
`TTS_PROVIDER_MODELS`, `TTS_PROVIDER_VOICES`, and `GOOGLE_TTS_VOICE_FAMILY` in
`calibrate/utils.py` are the one place each provider's default model/voice is
defined, and **both** the eval functions and the `create_*_service` factories
read from them. Change a value in those dicts and both sides move together. Do
**not** re-introduce a hardcoded model or voice string in either place —
reference the constant. (Google's TTS voice name embeds the language code, so
both sides build it as `{lang_code}-{GOOGLE_TTS_VOICE_FAMILY}`; other
per-provider params still need manual mirroring.)

`tests/test_utils_factories.py` guards this: it asserts every provider's model
and voice in `create_*_service` comes from the shared constants. If you add a
provider or change a default, update both sides
and that test. **Sarvam STT** uses `saaras:v3` with `mode="transcribe"` (passed
to pipecat's `SarvamSTTService`), which under pipecat 1.0.0 routes to the plain
STT streaming endpoint — matching the benchmark's `transcribe_sarvam`. This was
impossible under pipecat 0.0.98 (its wrapper forced `saaras` onto the *translate*
endpoint with no `mode` param), so Sarvam STT was excluded there; the 1.0.0
upgrade removed that limitation and it's now single-sourced like every other
provider. Sarvam **TTS** is likewise in `TTS_PROVIDER_MODELS` (`bulbul:v3`).

### Resumability
`run_stt_eval` / `run_tts_eval` write `results.csv` row-by-row and skip already
processed `id`s on retry. Use `--overwrite` to force a clean run. Beware:
pandas coerces numeric `id` values to int on read — if your dataset uses
string-looking ids like `"1"`, they round-trip as `1` and string comparisons
break. Tests use non-numeric ids (`"row_a"`) for this reason.

### Logging
`provider_log` (alias `_log` in eval modules) writes to both stdout and a
per-provider `logs` file under the output dir. Set `to_terminal=False` to
suppress stdout. The active log file is held in a `contextvars.ContextVar`
(`provider_log_file`) so concurrent benchmarks don't cross-write.

### Langfuse
All judge / transcribe / synthesize functions are decorated with `@observe`.
If `LANGFUSE_PUBLIC_KEY` is set, traces flow to Langfuse; otherwise the
decorator is a no-op. Don't remove these decorators casually — production
runs rely on them.

## Workflows

### Running the test suite

```bash
uv sync --extra dev                  # one-time
uv run pytest tests/stt              # subset (prefer this)
uv run pytest tests/stt/test_eval.py::TestSTTValidateInputDir -v
uv run pytest tests/                 # full suite (slow — avoid unless needed)
```

**Run only the tests relevant to your change, not the whole suite.** The full
suite is slow; scope your run to the mirrored test file(s) for the modules you
touched (e.g. a change to `calibrate/llm/run_tests.py` →
`uv run pytest tests/llm/test_run_tests.py tests/llm/test_run_tests_extra.py`).
CI runs the whole suite on the PR — let it be the backstop for the full run
rather than running everything locally on every change.

Tests are pure unit tests — **no real API calls** are ever made:
- All provider SDK clients are patched with `AsyncMock`/`MagicMock`.
- `instructor.apatch` and `_build_openrouter_client` are mocked in judge tests.
- HTTP-dependent tests use `pytest_httpserver` (in-process).
- A few tests stick dummy values into `os.environ` (e.g. `"sk-fake"`) just to
  pass the "is the key set?" guard before the mocked code path runs.

The suite runs in ~10s locally and contributes coverage to Codecov on CI.

### Git hooks
`.githooks/pre-commit` runs `uv run --extra dev pytest tests/` **only when
HEAD is on `main`**. Other branches commit instantly. Activated per-clone
with `git config core.hooksPath .githooks` (also in the README).

### CI
- **`.github/workflows/tests.yml`** — runs on push to `main` and on every PR
  targeting `main`. Installs `libasound2-dev` (needed for `simpleaudio`),
  syncs `--extra dev`, runs pytest with coverage, uploads to Codecov, and
  emails `aman.dalmia@artpark.in` on failure via `dawidd6/action-send-mail`.
- **`.github/workflows/publish.yml`** — release-triggered. Has a `test` job
  that `build` `needs:` — if tests fail, the PyPI publish is blocked.
- Secrets required: `MAIL_SERVER`, `MAIL_PORT`, `MAIL_USERNAME`,
  `MAIL_PASSWORD` (SMTP for failure emails) and `CODECOV_TOKEN`.

### Versioning + publish
Version is `dynamic` via `setuptools_scm` (`fallback_version = "0.0.0-dev"`).
A GitHub release tag becomes the package version. `publish.yml` builds the
sdist+wheel and pushes to PyPI via OIDC (no API token).

## Testing discipline

For any function block you add or modify:

1. **Write unit tests covering the change** — happy path plus the edge cases
   that motivated the change (empty inputs, missing keys, error branches,
   boundary values, concurrent / resume paths if applicable). Put them in the
   mirrored test file under `tests/` (e.g. a change to
   `calibrate/stt/eval.py` goes in `tests/stt/test_eval.py`).
2. **Run only the scoped tests for what you changed** — the mirrored test
   file(s) for the modules you touched (e.g. a change to
   `calibrate/llm/run_tests.py` → run `tests/llm/test_run_tests.py` and
   `tests/llm/test_run_tests_extra.py`), not the whole `tests/` suite, which is
   slow. Confirm they pass. Don't rely on the type checker or "it looks right"
   — the test must actually exercise the new path. CI runs the full suite on
   the PR, so let that be the backstop for the complete run.
3. **Only after tests pass** should you report the task as done. If a change
   is genuinely untestable (e.g. a CLI flag wired through to a third-party
   SDK), say so explicitly in the response rather than implying coverage.

This is not optional — every PR is gated by the test suite in CI and by the
pre-commit hook on `main`.

## Things to keep in mind

- **Default branch is `main`**, not `master`. Some early conversations used
  "master" but the repo and all CI configs use `main`.
- **Don't add comments unless the why is non-obvious.** The codebase follows
  the rule from the global guidelines: comments explain *why*, not *what*.
- **Prefer editing existing files** over creating new ones — especially in
  `stt/`, `tts/`, and `llm/`, where the structure is mirrored 1-to-1 in
  `tests/`.
- The `out/` folder appears inside several module dirs (e.g. `calibrate/stt/out`).
  These are gitignored runtime artifacts from local runs — don't commit them.
- `pipecat-ai` is pinned to `1.0.0` because the API surface changes between
  versions; bump deliberately and re-test the agent simulation paths.

## Useful pointers when debugging

- Failing tests in `tests/llm/test_run_simulation_integration.py` or
  `tests/test_cli.py` usually mean `pytest-httpserver` isn't installed
  (it's in the `dev` extra).
- `simpleaudio` build failures on Linux → missing `libasound2-dev`.
- Pandas mangling string ids in `results.csv` resume logic → cast to str
  explicitly or use non-numeric ids.
- Backoff retries swallowing a `ValueError` from an unknown provider →
  call `router.__wrapped__()` to bypass `@backoff` in tests.
