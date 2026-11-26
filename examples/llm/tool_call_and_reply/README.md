# A reply and its tool calls, judged by different evaluators

Three rows where the agent made tool calls, each judged by two evaluators:

- **Tool call correctness** (`"tool_call": true` in `config.json`) judges the tool calls. It reads the conversation, the tool calls and the expected tool calls, never the reply.
- **Faithfulness** (no `tool_call` key) judges the reply. It reads the conversation, the reply, and the tool calls with what each returned, never the expected tool calls.

## Run it

Needs `OPENROUTER_API_KEY`, since a real judge model is called.

```bash
calibrate-agent llm --eval-only \
  --config  examples/llm/tool_call_and_reply/config.json \
  --dataset examples/llm/tool_call_and_reply/eval_only_dataset.json \
  -o ./out/tool_call_and_reply
```

## What you should see

In `out/tool_call_and_reply/results.json`, each row's `metrics.judge_results`:

| Row | Tool call correctness | Faithfulness |
| --- | --- | --- |
| `weather_contradicts` | passes | fails, because the reply says sunny and the tool said heavy rain |
| `booking_correct` | passes | passes |
| `missed_call` | fails, because no booking call was made | passes |

If Tool call correctness mentions the reply, or Faithfulness passes `weather_contradicts`, each evaluator is no longer being shown only its own part.
