---
name: generate-code
description: Hypothesis-driven fix/change workflow — consults the review-current-state factblock, forms an explicit guess at the root cause and fix, then updates code and tests to verify it scientifically. Use when asked to fix a bug, change behavior, or implement something against a specific target, with a documented, hypothesis-driven process rather than an ad hoc edit.
allowed-tools: Skill, Agent
disable-model-invocation: false
model: fable
---

When this skill is invoked, act as a thin coordinator over the `generate-code` subagent — do not do the investigation or editing yourself.

Before doing anything else, read `.claude/skills/rules/generate-code.md` (target-handling and relaying policy), `.claude/rules/scope-and-safety.md` (filesystem write scope and safety boundaries every skill/agent follows), and `.claude/rules/project-mission.md` (what fat_llama actually is and how it works) — all three may be updated over time without this file changing.

Also read `README.md` at the repo root in full, right now, before Step 1 — it's fat_llama's own description of its purpose and method (iterative soft thresholding of FFT data to upscale compressed audio across supported formats, tested and built primarily against the MP3→FLAC outcome, CUDA-only, deliberately without AI/ML-based upscaling). This skill's job is to make sure the subagent it dispatches never drifts toward an out-of-scope "fix" (e.g. reaching for a trained/learned model or a CPU fallback path), so carry this context into the subagent's prompt in Step 2.

`args` names the target: what to look at and what to fix or change. If `args` is empty, ask the user what target to work on before proceeding.

## Logging

Before Step 1, open this run's log file per `.claude/rules/logging.md` (name: `generate-code-<time>-<user>.log`). Append one entry per step below, including the subagent dispatch in Step 2 and its own log filename (from the subagent's report).

## Steps

1. Check whether `docs/CURRENT_STATE.md` exists.
   - If it's missing entirely, invoke the `review-current-state` skill first (via the Skill tool) to generate it — `generate-code` depends on that factblock as its starting map.
   - If it already exists, don't regenerate it automatically; the subagent itself will flag if it looks stale for the files it needs and read source directly in that case.

2. Launch the `generate-code` subagent via the Agent tool with `subagent_type: "generate-code"` and `run_in_background: false` (the caller needs the result in this turn). Give it a self-contained prompt containing the exact target text from `args`, plus the condensed mission context from `.claude/rules/project-mission.md` (mp3→flac is the primary tested outcome among the formats fat_llama supports, via IST on FFT data, CUDA-only, no AI/ML-based upscaling) — it also reads its own rules file, the mission-context file, and the factblock, so no further context is required beyond that.

3. The subagent's final message is JSON per the contract in `.claude/agents/rules/scientific-coding.md`. Relay it per the relaying policy in `.claude/skills/rules/generate-code.md`.

## Note for coordinator agents

A coordinator does not need this skill at all — it can call the `generate-code` subagent directly via the Agent tool with `subagent_type: "generate-code"`, passing the target in the prompt. This skill exists as the interactive `/generate-code <target>` entry point; the subagent is the reusable unit.
