---
name: code-review
description: Project code-review style for this repo. Review the current branch's changes in two passes (correctness + justify-every-line), tracing any bug claim to ground before reporting.
---

Review only the changes made in this branch, not the whole repository.

Write everything as if you're explaining it to a product manager, not an engineer:
simple, jargon-free language. Keep it easy to scan — short lines, no long dense
paragraphs.

## Method — do both passes before reporting

Do not stop after the first interesting finding. A partial review that looks
thorough is worse than none. Make **two explicit passes** over every changed
function, then report once.

### Pass 1 — Correctness
Hunt for real bugs: wrong behaviour, crashes, data loss, security/access holes,
drift from a single-sourced value (see the repo's CLAUDE.md — model/voice maps,
eval-vs-live defaults).

**Trace every bug claim to ground before you write it down.** Follow the actual
data flow one step at a time — which function writes the value, what shape it
has, who reads it — instead of reasoning about it abstractly. If you can't point
to the concrete line where it goes wrong, it isn't a finding yet. (A confident
bug that dissolves on the first follow-up question means the trace was skipped.)

### Pass 2 — Justify every new line
For **each new symbol** the diff adds — every function, constant, map, helper,
loop, argument — ask out loud: *why is this here, and is this the simplest form?*
This pass catches what Pass 1 never will. Concrete prompts to run on each:

- **Is this map/helper necessary?** Could the values be set correctly at the
  source instead of translated later? (e.g. a rename map that only exists
  because two sides picked different names.)
- **Is this duplicated?** Does the same rule/value already live somewhere else,
  now copied? Copies drift — flag them.
- **Is this hardcoded when it needn't be?** A package/file/provider name that
  could be derived (`__package__`, an existing constant).
- **Is this defending against a case that can't happen?** A length check, a
  case-insensitive fallback, a branch — trace whether any real path hits it.
- **Is this read/written in the same pattern as its siblings?** An odd-one-out
  loop or index dance usually simplifies to match the rest.
- **Is this number/output in a sane form?** Raw floats, unrounded values,
  mislabeled fields.

These aren't style nitpicks — each must be a concrete redundancy, duplication,
dead abstraction, or drift risk you can name. Skip pure formatting/preference.

## Response structure

### 1. High-level flow
Open with 2-3 plain sentences: what this PR does and how the pieces fit
together. No code, no jargon.

### 2. Issues, grouped by severity
Bucket the issues under severity headings, highest first. Only include headings
that have issues:

- **Critical** — breaks a core feature, crashes, data loss, or a security/access hole.
- **High** — wrong behaviour or a likely failure that users will hit, but not catastrophic.
- **Medium** — works for now but fragile, or a smaller correctness gap.
- **Low** — duplication, redundant abstraction, hardcoding, drift-prone code, cleanup.

For each issue, lead with the file + method as a bold header on its own line
(name only — point to where it needs handling, don't paste code). Under it, put
three lines, each starting with a **bold label**. Keep each line to one short
sentence:

**`file.py` → `method_name()`**
- **Issue:** what's wrong, in 1 plain sentence.
- **Severity:** the bucket + one short reason it lands there.
- **Likelihood:** Almost certain / Likely / Occasional / Rare — plus a few words on why.

Always bold the labels. Put a blank line between issues so each reads as its own
block, never a wall of text.

If there are no real issues, say so in one line.

Stop after the issue list — wait for me to ask before going deeper on any of them.
