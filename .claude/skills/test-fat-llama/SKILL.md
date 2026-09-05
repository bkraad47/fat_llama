---
name: test-fat-llama
description: Runs the code-tester-reviewer and audio-quality-checker subagents against fat_llama's code and tests, then reports combined structured JSON results (test outcomes, PEP8/code-quality feedback, audio quality). Use when asked to check, verify, or review fat_llama's code, tests, coverage, style, or audio output quality. Does not fix anything itself — see the generate-code skill for that.
allowed-tools: Agent
disable-model-invocation: false
model: sonnet
---

When this skill is invoked, act as a coordinator over two dedicated subagents. Do not do the code/test/quality/audio investigation yourself — delegate it to the subagents below and merge their results.

Before doing anything else, read `.claude/skills/rules/test-fat-llama.md` (merge contract and relaying policy) and `.claude/rules/scope-and-safety.md` (filesystem write scope and safety boundaries every skill/agent follows) — both may be updated over time without this file changing.

**This skill only tests and reviews.** Neither subagent it dispatches modifies source code — `code-tester-reviewer` is read-only (it runs the suite and reviews style, it does not edit anything), and `audio-quality-checker` only ever touches test *assertions*, never production code. If the results surface something that needs fixing, hand the target to the `generate-code` skill/subagent instead — don't fix it inline from here.

## Logging

Before Step 1, open this run's log file per `.claude/rules/logging.md` (name: `test-fat-llama-<time>-<user>.log`). Append one entry per step below, including each subagent dispatch and its own log filename (from its JSON report's `"log"` field).

## Steps

1. Launch the `code-tester-reviewer` subagent via the Agent tool with `subagent_type: "code-tester-reviewer"` and `run_in_background: false` (wait for it — the caller needs the result in this turn). Give it a self-contained prompt: it should run fat_llama's test suite and review code quality/PEP8 compliance per its own instructions and rules file (`.claude/agents/rules/code-quality.md`); no extra context is needed, the agent definition already covers scope and paths.

2. Launch the `audio-quality-checker` subagent the same way (`subagent_type: "audio-quality-checker"`, `run_in_background: false`). Since `code-tester-reviewer` no longer edits any files, there's no shared-file race anymore — you may launch both subagents in parallel (single message, two Agent tool calls) rather than sequentially.

3. Each subagent's final message is JSON (see `.claude/agents/rules/code-quality.md` and `.claude/agents/rules/audio-quality.md` for the exact schemas):
   - `code-tester-reviewer` → `{"tests": [...], "quality": [...], "log": "..."}`
   - `audio-quality-checker` → `{"tests": [...], "log": "..."}`

   Merge them per the merge contract in `.claude/skills/rules/test-fat-llama.md`.

4. Output the merged JSON as your final result, per the relaying policy in `.claude/skills/rules/test-fat-llama.md`.
