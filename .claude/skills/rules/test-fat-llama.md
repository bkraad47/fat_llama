# Rules — test-fat-llama

These are the tunable rules the `test-fat-llama` skill follows as a coordinator. Edit this file to change its policy without touching `SKILL.md`'s step flow.

## Write scope

Per `.claude/rules/scope-and-safety.md`: this skill writes nothing itself, ever. It only dispatches `code-tester-reviewer` and `audio-quality-checker`, both of which have their own scope restrictions (see `.claude/agents/rules/code-quality.md` and `.claude/agents/rules/audio-quality.md`). If a result suggests something needs fixing, hand it to the `generate-code` skill/subagent instead of touching anything here.

## Merge contract

```json
{
  "tests": [ /* entries from code-tester-reviewer */ ],
  "quality": [ /* entries from audio-quality-checker */ ],
  "logs": [ "<code-tester-reviewer log path>", "<audio-quality-checker log path>", "<this coordinator's own log path>" ]
}
```

## Relaying the result

- Interactively: a one-line human summary (e.g. "3 tests passed, 1 failed; 4 PEP8 issues found; audio output coherent") may precede the JSON.
- Non-interactively (e.g. invoked via `claude -p ... --output-format json`, such as from a GitHub Actions workflow or a coordinator agent): output **only** the JSON — no summary, no commentary, no markdown fencing.

## Open items

Fill in over time: any additional subagents this coordinator should dispatch as the project grows.
