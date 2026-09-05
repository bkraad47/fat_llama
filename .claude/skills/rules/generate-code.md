# Rules — generate-code (skill)

These are the tunable rules the `generate-code` skill follows as a coordinator. Edit this file to change its policy without touching `SKILL.md`'s step flow. The subagent it dispatches has its own separate rules at `.claude/agents/rules/scientific-coding.md` — don't duplicate those here.

## Mission context

Per `.claude/rules/project-mission.md`: this skill reads `README.md` in full at the start of the run (before Step 1) and passes its condensed context — mp3→flac is the primary tested outcome among the formats fat_llama supports, via iterative soft thresholding (IST) of FFT data, CUDA-only, deliberately no AI/ML-based upscaling — into the subagent's prompt in Step 2, alongside the target. This applies whether the skill was invoked interactively or dispatched by `iterate-fat-llama` (which already carries the same context and should still be treated as the source of truth for it in that case, rather than re-deriving it).

## Model

This skill runs on **Fable (or a higher-tier model if one supersedes it)** — set in its frontmatter — rather than the project's Sonnet default, since it's the coordinator for the code-generation path. Keep this in sync with `.claude/agents/generate-code.md`'s own model setting; they should always match.

## Write scope

Per `.claude/rules/scope-and-safety.md`: this skill writes nothing itself. It may trigger `review-current-state` (which writes only `docs/CURRENT_STATE.md`) and dispatches the `generate-code` subagent (which is the only role permitted to edit `fat_llama/**` source and tests). If you find yourself about to edit a file directly from this skill rather than through the subagent, stop — that's out of scope for the coordinator.

## Target handling

- `args` names the target: what to look at and what to fix or change.
- If `args` is empty, ask the user what target to work on before proceeding — never guess a target or invent scope on your own.

## Relaying the subagent's report

- Interactively: give a short human summary first (the hypothesis tested, what changed, verification result), then the JSON.
- Non-interactively (e.g. invoked via `claude -p ... --output-format json`, such as from a coordinator agent or CI): output **only** the JSON — no summary, no commentary, no markdown fencing.

## Open items

Fill in over time: any additional target-validation rules (e.g. rejecting targets that look like they'd touch out-of-scope files) found necessary in practice.
