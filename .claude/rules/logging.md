# Logging — all skills and subagents

Every skill and subagent in this project writes one append-only log file per invocation, so a run can be reconstructed afterward without re-running anything. This file is the single source of truth for the format — each skill/agent just links here rather than repeating it, so edit this file (not each skill/agent) to change logging project-wide.

## Where

```
.claude/log/<name>-<invocation-time>-<user>.log
```

- `<name>` — the skill or subagent's own name, exactly as it appears in its own frontmatter (`review-current-state`, `generate-code`, `code-tester-reviewer`, `audio-quality-checker`, `test-fat-llama`, `iterate-fat-llama`, ...).
- `<invocation-time>` — UTC timestamp captured once, at the very start of the run: `YYYYMMDD-HHMMSS`.
- `<user>` — `git config user.name`, with anything outside `[A-Za-z0-9_.-]` replaced by `-`; fall back to the OS username if git isn't configured.
- Create `.claude/log/` first if it doesn't exist yet.
- If that exact filename already exists (two invocations landed in the same second), append `-2`, `-3`, ... until the name is free. Never overwrite or append to a previous run's log file.

## When

Compute the filename once, at the very start of the run, before doing anything else described in the skill/agent's own steps. Every step from then on appends to that same file.

## Entry format

One entry per step (not per individual tool call — group everything a step does under its one entry), appended in order as the step actually happens, not reconstructed from memory at the end:

```
### [HH:MM:SS UTC] Step <n/name> — <short step name>
Input: <what this step received — args, file paths, upstream JSON/results, etc.>
Action: <what was actually done — commands run, files read/written, subagents/skills dispatched>
Reasoning: <why, in 1-2 sentences — the decision or hypothesis behind the action>
Output: <what came back — result, pass/fail, file produced, score, etc.>
```

- Be factual and brief — this is a run record, not a report to the user. When a step's actual output is structured data (e.g. a subagent's JSON result), include it verbatim in `Output:` rather than paraphrasing it into prose.
- When a skill dispatches a subagent or another skill, its own log entry for that step records what was dispatched and with what prompt/inputs, plus the dispatched subagent's own log filename — don't copy the subagent's internal step-by-step log into the coordinator's log; that's what the subagent's own log file is for.
- End every log with one final line: `### [HH:MM:SS UTC] DONE — <one-line outcome summary>`.

## Failures

If a step fails or a dispatched subagent errors, log it the same way (`Output: FAILED — <what happened>`) and continue or abort per that skill/agent's own instructions — a failure still gets logged, it isn't skipped just because there's nothing successful to report.
