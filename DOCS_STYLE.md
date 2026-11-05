# Docs writing guidelines

These rules apply to every page under `docs/`.

Two checks enforce them, on the lines a change adds:

- `scripts/check_docs_style.py` runs in under a second and blocks the commit. It
  carries only the rules a regular expression settles with no false positives:
  rule 8, and the banned list in rule 3.
- A Claude review reads this file and judges every rule, including the ones a
  regular expression cannot. It prints what it finds at commit time without
  blocking, and fails the check on a pull request.

Generated pages are exempt, since their wording comes from the backend spec, the
cloud CLI, or the MCP server: `docs/api-reference/`, `docs/sdk/`, `docs/mcp/`,
`docs/cli/calibrate/`, `docs/reference/api-keys.mdx`, and
`docs/reference/github-actions.mdx`.

A reader lands on a page cold, from a search result, and is not an engineer.
Write for that person.

## 1. Show the shape, never describe it

When a page says what Calibrate sends, or what an endpoint returns, print the
body. A sentence that characterises a shape in words says nothing the reader can
act on.

Wrong:

> A `general` agent takes one instruction per call, so it receives only the
> latest user text.

Right:

> ```json
> { "input": "Check order ORD-12345" }
> ```

## 2. Name the setting value beside every shape

A reader seeing two bodies must be able to tell which setting produces which. A
JSON block with no setting value named beside it is a violation. When there is
more than one shape, use a table with one row per value.

Right:

> | `agent_type`             | Request body                                                 |
> | ------------------------ | ------------------------------------------------------------ |
> | `conversation` (default) | `{"messages": [{"role": "user", "content": "<the input>"}]}` |
> | `general`                | `{"input": "<the input>"}`                                    |

## 3. No invented phrases

Use the name the codebase, the product, or the tool already uses. Never coin a
label for an idea and then reuse it as if the reader knows it. Never put quotes
or capitals around a phrase to turn it into a term.

These phrases have no reading the rules allow, so the script fails the commit on
them:

- "one instruction per call"
- "plain instruction"
- "under the hood"
- "out of the box"
- "as requested"
- "previously"

Words that are filler in one sentence and correct in the next are left to the
review, which reads the sentence around them: "simply", "just", "on its own",
"per call", "now", "still", "no longer", "instead of", "note that".

## 4. Plain words

No technical term the page can do without. Say what happens, and who does it to
what, in the order it happens. If a sentence would sound strange read aloud,
rewrite it.

## 5. One idea per sentence

A sentence carrying a "which was" or a "rather than" is doing two jobs. Split it
or cut it.

## 6. Write standalone

State what the feature is. Never what it changed from, what it is not, or what
someone asked for. The script fails on these words, which only make sense next
to an older version:

- now, previously, no longer, still, as requested, instead of, note that

## 7. Sentence case headings

Capitalise the first word only: `## Next steps`, not `## Next Steps`. Acronyms
(API, LLM, STT, TTS, URL, HTML, PR, WER, TTFB) and proper nouns (Calibrate,
GitHub, Mintlify, Markdown) keep their capitals.

## 8. No em-dashes

Use a comma, a period, or a colon.

## 9. No step that assumes earlier setup

Every instruction works for a reader who arrived at this page and nowhere else.
A command that only runs after an `export` on another page needs that step
repeated here, or a link placed before it.

## 10. Write only what the reader must do

Cut every sentence about something the reader does not have to do or know.
"There is nothing to configure", "this does not affect you", "no setup needed".
If nothing is needed from them, the sentence is not needed either.
