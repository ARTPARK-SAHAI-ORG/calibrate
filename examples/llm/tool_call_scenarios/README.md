# Tool-call matching scenarios

Test fixtures that exercise per-parameter tool-call matching.

## Files

- **`config.json`** — clinic-assistant system prompt + tools, plus a few
  `tool_call` test cases for a live run against a real model.
- **`eval_only_dataset.json`** — 12 `(test_case, output)` pairs that pin the
  agent's output, so you see the exact pass/fail + reasoning each scenario
  produces. The `_scenario` field on each item just labels what it demonstrates.

## Run it (eval-only — recommended)

Eval-only skips model inference and runs the evaluator on the pinned outputs.
Exact-only scenarios need no API; the `llm_judge` scenarios call the judge model
(needs `OPENROUTER_API_KEY`).

```bash
python -m calibrate.llm.run_tests --eval-only \
  --config  examples/llm/tool_call_scenarios/config.json \
  --dataset examples/llm/tool_call_scenarios/eval_only_dataset.json \
  -o ./out/tool_call_scenarios
```

## Run it (live — against a model)

```bash
python -m calibrate.llm.run_tests \
  --config examples/llm/tool_call_scenarios/config.json \
  -m openai/gpt-4.1 -p openrouter \
  -o ./out/tool_call_scenarios_live
```

## Where to look

- `out/.../results.json` — per-case `output` + `metrics`. Tool calls that
  involved an `llm_judge` parameter carry a `param_judgments` list (one record
  per parameter, exact and judged, with the judge's reasoning).
- `out/.../metrics.json` — per-tool pass rates.
- `out/.../results.log` — the pass/fail + reasoning lines, mirrored from stdout.

## What each scenario shows

| id                     | What it demonstrates                     | Outcome                            |
| ---------------------- | ---------------------------------------- | ---------------------------------- |
| `exact-all-pass`       | all exact, all match                     | pass (flat message)                |
| `exact-value-mismatch` | one exact value wrong                    | fail                               |
| `exact-unexpected-key` | agent sent an extra arg                  | fail (`value=True`)                |
| `exact-missing-key`    | required arg missing                     | fail                               |
| `judge-pass`           | one `llm_judge` param, satisfied         | pass                               |
| `judge-fail`           | one `llm_judge` param, not satisfied     | fail                               |
| `mixed-pass`           | exact + judge, both ok                   | pass (names exact + judge verdict) |
| `mixed-exact-fail`     | exact wrong, judge ok                    | fail (judge still captured)        |
| `exact-spec-wrap`      | `match_type: exact` verbatim compare     | pass                               |
| `nested-object`        | exact + judged sub-fields (dotted paths) | pass                               |
| `wrong-tool`           | agent called a different tool            | fail                               |
| `multi-call`           | two tool calls, each judged              | pass (tool-prefixed lines)         |
