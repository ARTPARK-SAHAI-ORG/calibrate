# CLAUDE.md

Guidance for Claude Code when working in this repository.

> **⚠️ Explain in plain, consistent words.** When answering the user, pick one
> term per concept and keep it for the whole explanation — never swap synonyms
> for the same thing (e.g. don't call one thing a "model name", then an
> "engine", then an "instruction"). Avoid jargon and vague filler verbs
> ("carry", "hold", "surface", "spread", "wire up") — say plainly what actually
> happens. Before sending an explanation, re-read it and make sure a
> non-engineer would follow it and that no term shifts meaning midway.

> **⚠️ Start every new task in a new worktree when you're on `main`.** Before
> beginning any new piece of work, check the current branch. If HEAD is on
> `main`, create a dedicated worktree (and branch) for the task and do the work
> there — never work or commit directly on `main`. If you're already on a
> non-`main` branch/worktree scoped to the task, continue there.

## What this project is

**Calibrate** (`calibrate-agent` on PyPI) is an open-source evaluation framework
for voice agents. It benchmarks LLMs, STT providers, TTS providers, and runs
agent simulations — all from a single CLI / Python library.

- Website / docs: https://calibrate.artpark.ai
- Built on top of [pipecat](https://github.com/pipecat-ai/pipecat).
- The CLI entry point is `calibrate-agent` (defined in `pyproject.toml:scripts` →
  `calibrate_agent.cli:main`).

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
6. **Implement through the `/parallelize` skill.** Once the plan is settled and
   it's time to write code, invoke the `parallelize` skill and let it run the
   independent chunks as parallel agents. This applies to every implementation,
   including single-file ones.

## Repository layout

```
calibrate_agent/                 # Python package (the importable library + CLI)
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

tests/                     # Test suite — mirrors the calibrate_agent/ structure
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
└── package.json           # Bundled into calibrate_agent/ui/cli.bundle.mjs

docs/                      # Mintlify docs site (.mdx)
examples/                  # Example datasets + scripts users can run
.github/workflows/         # tests.yml, publish.yml, claude.yml, claude-code-review.yml
.githooks/pre-commit       # Runs pytest before commits to main
```

## Conventions in this codebase

### Docs (Mintlify) writing style
The docs under `docs/` use **sentence case** for all headings (`##`/`###`),
`Card`/`Step` titles, and frontmatter `title` fields — capitalize only the first
word, e.g. `## Next steps`, `## Get started`, `### Launch run`. Do **not** use
Title Case (`## Next Steps`). The only words that stay capitalized mid-heading
are acronyms (API, LLM, STT, TTS, WER, TTFB, PR, HTML, URL) and proper nouns
(Calibrate, GitHub, Mintlify, Markdown). Match this when adding or editing any
`.mdx` page.

Write docs standalone — a reader never saw the change history or our chat. State
what the feature *is*, never what it changed from, what it's "not", or what was
requested (see "Never leak the conversation into long-lived text" under Things
to keep in mind).

Some pages are **generated** from templates in `docs/templates/` by
`scripts/fetch_public_openapi.py` (which single-sources the backend host from
`PUBLIC_API_BASE_URL` — never hardcode it). Edit the template, not the generated
output, then re-run the script. Currently templated: `api-reference/introduction.mdx`,
`reference/api-keys.mdx`, `reference/github-actions.mdx`.

The **cloud `calibrate` CLI** pages under `docs/cli/calibrate/**` are also
generated — from the Cobra markdown that the separate `dalmia/calibrate-cli`
repo ships in its `docs/` folder. `scripts/generate_cli_docs.py` (parser in
`scripts/cli_reference.py`) produces one **Guides** page per command
(`agents`, `agent-tests`, `auth`, …), with each subcommand rendered as a `##`
section — usage block, an `Option | Type | Default | Description` table parsed
from the Cobra `### Options` dump, and examples. Pages are kept minimal
(Coval-style): the "options inherited from parent commands" block is dropped and
there is **no per-page global-flags callout** (global flags are documented once
under Getting started); option tables drop `-h, --help` and placeholder-only
flags (e.g. Cobra's `--x-api-key string  string value`). The two **Getting
started** pages (`overview`, `agent-mode`) are hand-written templates under
`docs/templates/cli/calibrate/`. The generator patches the "CLI" tab
into `docs.json` (inserted after "Home") as two groups — Getting started, then
Guides. Only **API-backed** command groups get a
Guides page: a command whose name is an OpenAPI tag (`agents`,
`agent-tests`) is documented; local/utility commands (`auth`, `configure`,
`version`, …) are skipped and covered under Getting started instead. There is
**no hand-maintained map**: the sidebar is derived from the CLI source itself
(like the SDK docs derive theirs from the backend OpenAPI/Fern overrides) —
resource titles come from `api_group_from_tag` (shared with
`generate_sdk_docs.py` — `agent-tests` → "Agent tests"), and resource +
subcommand order follow each command's Cobra `SEE ALSO` list. Never edit the
generated `docs/cli/calibrate/*.mdx` (except the two Getting-started templates)
— edit the source markdown in `calibrate-cli` or the templates, then re-run.
`fetch_public_openapi.py` calls the generator when `CLI_DOCS_PATH` points at a
`calibrate-cli/docs` checkout (the `sync-api-spec.yml` workflow checks out that
repo and sets it). The cloud CLI is the "CLI" tab; don't confuse it with
the offline `calibrate-agent` eval CLI, which is the separate "Calibrate Local"
tab (`docs/cli/overview.mdx` etc. — a static, hand-maintained tab in
`docs.json`, positioned after "API reference"). The CLI-docs tests use synthetic
Cobra samples (`tests/cli_doc_samples.py`), **not** a snapshot of the real CLI —
so they don't drift; the live CLI is exercised by the sync workflow instead.
`generate_cli_docs.py` and `generate_sdk_docs.py` share `scripts/docs_nav.py`
for inserting their tab into `docs.json`.

The **MCP docs** under `docs/mcp/**` are likewise generated — from the
Speakeasy-generated MCP server the separate `dalmia/calibrate-mcp` repo ships
under `src/mcp-server/tools/*.ts`. `scripts/generate_mcp_docs.py` (parser in
`scripts/mcp_reference.py`) produces the "MCP" tab (inserted after "Python SDK"),
whose flow mirrors [Coval's MCP docs](https://docs.coval.ai/mcp/overview): a
single **"MCP server"** sidebar group with five pages in a fixed order
(`NAV_ORDER`) — **Overview**, **Installation**, **Tools**, **Beginner's guide**,
**Troubleshooting**. All four conceptual pages are hand-written templates under
`docs/templates/mcp/` (`GUIDE_PAGES`); only the **Tools** page
(`docs/mcp/tools.mdx`) is generated — tools grouped by API resource
(`## Agents`, `## Agent tests`, `## Tests`), each tool a `###` section with its
description, a scope / access line, a parameters table, and a cross-link to the
matching API-reference operation. The overview's "Available tools" category table
is injected into the template at the `{/* AVAILABLE_TOOLS */}` token (like the
CLI overview's global-flags token), so it always lists the live tool set with
links to the Tools-page anchors. Add or reorder a page by editing `NAV_ORDER` (and
`GUIDE_PAGES` for a new hand-written page + its template).

Like the SDK/CLI docs the structure is **single-sourced, not hand-maintained**:
each Speakeasy tool imports one request model whose file stem equals the OpenAPI
`operationId` with non-alphanumerics stripped plus `op`
(`mcp_reference.operation_key`), so tools match operations deterministically; the
operation's tag drives grouping (`api_group_from_tag`, shared with the SDK/CLI
generators) and its parameters + request body (unwrapping `$ref` and optional
`anyOf: [Ref, null]` bodies) drive the parameters table. Tools sort
read-before-write within a resource (`_VERB_ORDER`: list, get, …, create,
update), mirroring Coval. Multi-line OpenAPI property descriptions collapse to
their first line in the table — the cross-link carries the full nested schema. A
tool that matches **no** operation aborts the run (stale spec / renamed op)
before any file is touched. Never edit the generated `docs/mcp/tools.mdx` (or the
generated overview table) — edit the tool definitions in `calibrate-mcp` or the
Getting-started templates, then re-run. `fetch_public_openapi.py` calls the
generator when `MCP_DOCS_PATH` points at a `calibrate-mcp/src/mcp-server/tools`
checkout (the `sync-api-spec.yml` workflow checks out that repo and sets it).
Tests use synthetic Speakeasy tool samples (`tests/mcp_tool_samples.py`), **not**
a live snapshot. All three generators share `scripts/docs_nav.py` (tab insertion)
and `scripts/docs_mdx.py` (MDX escaping / frontmatter / table-cell helpers).

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

Helpers in `calibrate_agent/judges.py`:
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
  `calibrate_agent/utils.py`, which the voice-agent simulation / tests actually run
  (`agent/bot.py`, `agent/test.py`).

If these drift, a leaderboard stops reflecting the config the agent deploys —
you'd benchmark one voice/model and ship another. **Any time you change a
provider's model, voice, endpoint, or other default on one side, change the
other side too.**

To make this structural rather than a discipline you have to remember, the
**model names and TTS voices are single-sourced** in `calibrate_agent/utils.py`:
`STT_PROVIDER_MODELS` / `TTS_PROVIDER_MODELS` define each provider's default
model, and TTS voices resolve through `get_tts_voice(provider, language)` —
which returns the per-language override in `TTS_PROVIDER_VOICES_BY_LANGUAGE` if
one exists, else the provider default in `TTS_PROVIDER_VOICES` (Google builds
`{lang_code}-{GOOGLE_TTS_VOICE_FAMILY}`). **Both** the eval functions and the
`create_*_service` factories read from these. Change a value and both sides move
together. Do **not** re-introduce a hardcoded model or voice string in either
place — reference the constant / call `get_tts_voice`. (Voices are per-language:
e.g. Cartesia and Smallest ship distinct Hindi/Kannada voices; other
per-provider params still need manual mirroring.)

`tests/test_utils_factories.py` guards this: it asserts every provider's model
and voice in `create_*_service` matches the shared source for every language. If
you add a provider or change a default, update both sides
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

`results.csv` belongs to that resume logic: the full run owns the file and
reads it back on retry. Only the `--eval-only` paths pass `stream_rows=True` to
`_score_and_write_results`, which appends each row as its judge result arrives
(`PartialResultsWriter` in `judges.py`). Turning that on for a full run would
replace the transcription/synthesis rows mid-judge, and a judged row carries
neither `ttfb` nor a transcription of its own: STT resumes off the surviving
`id`s and re-transcribes everything else, while TTS fails
`validate_existing_results_csv` (no `ttfb` column) and stops the next run until
`--overwrite`, which re-synthesizes every row. `general` has no resume file, so
it always streams.

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
touched (e.g. a change to `calibrate_agent/llm/run_tests.py` →
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

### Committing and pushing
**Once a change is complete and its scoped tests pass, commit and push it
without waiting for approval** — don't stop to ask first. This overrides the
default "commit/push only when asked" behavior for this repo. Applies to normal
feature/fix work on a branch:

- Commit with a clear message; push the branch to its remote (`git push`, or
  `git push -u origin HEAD` the first time).
- If the branch already has an open PR whose commits were rebased/amended, use
  `git push --force-with-lease` to update it.
- If you're on `main`, branch first — never commit straight to `main`.
- Still pause to ask before genuinely destructive or irreversible actions
  (e.g. `git reset --hard` over uncommitted work, deleting remote branches,
  history rewrites beyond your own un-merged branch, publishing a release).

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
   `calibrate_agent/stt/eval.py` goes in `tests/stt/test_eval.py`).
2. **Run only the scoped tests for what you changed** — the mirrored test
   file(s) for the modules you touched (e.g. a change to
   `calibrate_agent/llm/run_tests.py` → run `tests/llm/test_run_tests.py` and
   `tests/llm/test_run_tests_extra.py`), not the whole `tests/` suite, which is
   slow. Confirm they pass. Don't rely on the type checker or "it looks right"
   — the test must actually exercise the new path. CI runs the full suite on
   the PR, so let that be the backstop for the complete run.
3. **Only after tests pass** should you report the task as done. If a change
   is genuinely untestable (e.g. a CLI flag wired through to a third-party
   SDK), say so explicitly in the response rather than implying coverage.

This is not optional — every PR is gated by the test suite in CI and by the
pre-commit hook on `main`.

### When the user asks "how do I test this?"

Answer with the **single specific command they should run locally** for the
change at hand — the exact `uv run …` / `calibrate-agent …` invocation, with the
real flags, dataset path, and provider filled in (use `examples/` datasets when
one fits). Do **not** give a tiered "here are the options" answer or a menu of
fast-vs-thorough choices. One concrete, copy-pasteable command (a second line
only if a follow-up command is genuinely needed, e.g. inspecting the output
file). If a real run needs API keys or a dataset the user must supply, say so in
one line, but still give the exact command.

## Things to keep in mind

- **Default branch is `main`**, not `master`. Some early conversations used
  "master" but the repo and all CI configs use `main`.
- **A question is not a change request.** When the user asks "why…", "what is
  …", "isn't this…", "where is…", answer it and stop. Do not edit, commit, or
  push code off a question — answer, *offer* the change ("want me to…?"), and
  wait for an explicit yes. This holds even when the change looks obviously
  correct (dead-data cleanup, a rename, a fix): making it unasked removes the
  user's decision and creates churn. Especially during review/Q&A, the user is
  deciding what to do, not delegating it.
- **Don't add comments unless the why is non-obvious.** The codebase follows
  the rule from the global guidelines: comments explain *why*, not *what*.
- **Never leak the conversation into long-lived text.** Code comments, docs,
  commit-persisted files, and CLAUDE.md are read cold by someone who never saw
  our chat. Write them as if authored fresh: state what *is*, not what changed,
  what it's "not", what was "removed", or what someone asked for. Ban words like
  "now", "previously", "still", "no longer", "as requested", "instead of X",
  "note that we don't…" — anything that only makes sense relative to a prior
  version or a request. If a distinction matters on its own (a real limitation,
  a gotcha), state it plainly without the reactive framing. This applies to
  every file you write, not just docs.
- **Prefer editing existing files** over creating new ones — especially in
  `stt/`, `tts/`, and `llm/`, where the structure is mirrored 1-to-1 in
  `tests/`.
- The `out/` folder appears inside several module dirs (e.g. `calibrate_agent/stt/out`).
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
