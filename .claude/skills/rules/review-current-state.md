# Rules — review-current-state

These are the tunable rules `review-current-state` follows. Edit this file to change its scope/output policy without touching the skill's step-by-step flow in `SKILL.md`.

## Write scope

Per `.claude/rules/scope-and-safety.md`: this skill writes only `docs/CURRENT_STATE.md`, or a path explicitly given in `args`. It never touches source, tests, config, or anything under `.claude/`/`.github/workflows/`. It is a read/snapshot skill — reading widely across the repo is expected and fine; writing is not.

## Scope of review

- Default scope is the whole repository.
- If the user names a path/module in `args`, scope the review to that path only and note the narrowed scope in the output.
- Exclude: `.git`, `.venv`, `venv`, `build`, `dist`, `*.egg-info`, `__pycache__`, `.pytest_cache`, `node_modules`, and any other virtualenv/artifact directories found. Only tracked source is factblocked, never generated/vendored files.

## Output path

Default: `docs/CURRENT_STATE.md` at the repo root (create `docs/` if missing), overwriting any previous version — this file is meant to be regenerated on demand, not hand-edited. Use a different path only when `args` explicitly names one.

## Open items

Fill in over time: any additional exclusion directories found in practice, whether non-Python source should ever be in scope.
