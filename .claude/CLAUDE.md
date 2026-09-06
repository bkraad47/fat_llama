# CLAUDE.md

Guidance for Claude Code — and any skill/subagent it dispatches — working in this repository.

## Project overview

fat_llama upscales compressed audio (primarily MP3 → FLAC) via iterative soft thresholding (IST) on FFT data — a CUDA/CuPy-only DSP pipeline, deliberately without AI/ML-based upscaling. See `README.md` for the full description and `.claude/rules/project-mission.md` for the condensed version every skill/agent works from.

This project's own skills, subagents, and their tunable rules live under `.claude/` — start with `.claude/rules/scope-and-safety.md`, the project-wide source of truth for write-scope and safety boundaries every skill/agent follows. `docs/CURRENT_STATE.md` is a regenerable factblock snapshot of the codebase (see the `review-current-state` skill) — read it for an up-to-date map of functions/classes rather than re-deriving one from scratch.

An MCP server (`fllm-mcp-server`) can run the real upscale pipeline on an on-demand remote GPU; its own repo is `bkraad47/fat-llama-mcp`. `.github/workflows/issue-branch-resolve.yml` runs `iterate-fat-llama` non-interactively when the repo owner creates an issue-linked branch.

## Git branch discipline (strict)

**Never create a new branch, rename the current branch, or switch to a different branch unless the user explicitly asks you to.** Work directly on whichever branch is already checked out — commit there, push there, and open any PR straight from that same branch.

This applies to every skill and agent here, not only `iterate-fat-llama` (which carries its own version of this rule in `.claude/skills/rules/iterate-fat-llama.md`, added after a real incident). A past run created a `iterate-fat-llama/<timestamp>` working branch, then renamed it again to a `v-<version>` branch — three branch identities for one piece of work. That produced exactly the kind of mess this rule exists to prevent: a PR merged from the wrong branch, missing the actual version bump/changelog because those commits ended up on a different, orphaned branch nobody looked at again. Whatever branch you're handed — an issue-linked branch, a feature branch, whatever the user is currently on — is already the right place to do the work and open the PR from.

The one narrow exception: genuinely risky, multi-cycle, rollback-capable work (e.g. `iterate-fat-llama`, if ever run with `main` itself as the starting branch) should isolate onto a new branch first rather than experimenting directly on `main`. This is the rare exception, not the default — and even then, name the branch clearly and clean it up (merge or delete it) rather than leaving it dangling once the work lands.

## Remote actions

Pushing, force-pushing, merging, and opening/closing PRs are all real, visible actions on a shared repository. Confirm with the user before any of these unless they've already explicitly asked for it in the current request.
