# Rules — generate-code (skill)

These are the tunable rules the `generate-code` skill follows as a coordinator. Edit this file to change its policy without touching `SKILL.md`'s step flow. The subagent it dispatches has its own separate rules at `.claude/agents/rules/scientific-coding.md` — don't duplicate those here.

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
