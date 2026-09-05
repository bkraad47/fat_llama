---
name: iterate-fat-llama
description: End-to-end iterate-and-ship loop for fat_llama — refreshes the review-current-state factblock, runs test-fat-llama, and dispatches generate-code to fix what it finds, repeating up to 5 cycles or until audio quality is fully satisfactory; rolls back to whichever cycle scored best if none was fully satisfactory; then regenerates docs, bumps the version, and opens a PR for review. Use when asked to autonomously iterate on fat_llama's code/tests/audio quality and ship the result as a reviewable PR, not just check it once.
allowed-tools: Bash, Skill, Agent
disable-model-invocation: true
model: sonnet
---

Before doing anything else, read `.claude/skills/rules/iterate-fat-llama.md` in full — it defines the cycle cap, score formula, naming conventions, version-bump policy, and remote-action confirmation requirement referenced throughout the steps below, and may be updated over time without this file changing. Also read `.claude/rules/scope-and-safety.md` for the filesystem write scope and safety boundaries every skill/agent follows, and `.claude/rules/project-mission.md` for what fat_llama actually is and how it works.

Also read `README.md` at the repo root in full, right now, before Step 0 — it's fat_llama's own description of its purpose and method (iterative soft thresholding of FFT data to upscale compressed audio across supported formats, tested and built primarily against the MP3→FLAC outcome, CUDA-only, deliberately without AI/ML-based upscaling). Carry that context — condensed per `.claude/rules/project-mission.md` — into every `generate-code` dispatch in step (d) below and into the docs/changelog/PR steps that follow, so nothing drifts toward an out-of-scope "fix".

`args` is optional free-text directives on what to fix or not fix (e.g. "focus on bitrate accuracy, don't touch the CLI"). Handle it per the DIRECTIVES rule in `.claude/skills/rules/iterate-fat-llama.md`.

This skill coordinates other skills/subagents — it does not review, test, or fix anything itself. It also touches git branches, tags, and (at the very end, with confirmation) the GitHub remote — read Step 0 and Step 7 carefully.

## Logging

Before Step 0, open this run's log file per `.claude/rules/logging.md` (name: `iterate-fat-llama-<time>-<user>.log`). This is the top-level log for the whole run — append one entry per step, including per-cycle entries for (a)-(e) in the loop below (log each cycle's `score_i` and the dispatch/log-filename of every skill or subagent it calls), the rollback decision in (f) if it happens, and the final PR outcome in Step 7.

## Step 0 — setup

1. Read `README.md` in full (if not already done per the note above) and log a one-line entry recording that the run's mission context is fat_llama's own README plus `.claude/rules/project-mission.md`.
2. `git status`. If there are uncommitted changes that aren't something you just made in this run, stop and ask the user before doing anything else — don't stash or discard in-progress work.
3. Record the starting branch as `BASE_BRANCH` and current HEAD as `BASE_COMMIT`.
4. Create a new branch per the naming convention in `.claude/skills/rules/iterate-fat-llama.md` and switch to it. Tell the user you've switched branches — `BASE_BRANCH` is left untouched for the rest of this run.
5. Tag the untouched starting state per that same naming convention (`iter-0`). This is candidate `C_0`.

## Steps 1–4 — the iteration loop (up to MAX_CYCLES)

For `i = 1..MAX_CYCLES` (see the cycle policy in `.claude/skills/rules/iterate-fat-llama.md` for the current cap and score formula):

**a. Review current state.** Run the `review-current-state` skill (Skill tool) to refresh `docs/CURRENT_STATE.md` against the working tree as it stands right now — that's candidate `C_{i-1}` (the original code when `i=1`, or the previous cycle's fix when `i>1`).

**b. Test.** Run the `test-fat-llama` skill (Skill tool) to get its merged JSON (`{"tests": [...], "quality": [...]}`) for `C_{i-1}`. From `tests`, take the entries with `"type": "audio_quality"` and compute:
   - `total_i` = how many there are
   - `passed_i` = how many have `"status": "pass"`
   - `score_i = passed_i / total_i` (use `0` if `total_i` is `0` — no audio evidence isn't satisfactory)

   `score_i` is the satisfaction of candidate `C_{i-1}` — keep a running table of `(i, score_i)` for the rollback decision in step (f).

**c. Check for early stop.** If `score_i == 1.0` (every audio-quality check passed), `C_{i-1}` is already satisfactory — stop the loop immediately without running a fix this cycle, and go straight to Step 5 with the working tree exactly as it is.

**d. Fix.** Otherwise, launch the `generate-code` subagent (Agent tool, `subagent_type: "generate-code"`, `run_in_background: false`) with a prompt containing: the full `test-fat-llama` JSON from (b), `DIRECTIVES` verbatim if non-empty, and the condensed mission context from `.claude/rules/project-mission.md` (mp3→flac via IST on FFT data; no AI/ML-based upscaling) — ask it to address the reported test/quality/audio findings within those directives and that constraint.

**e. Checkpoint.** Commit whatever `generate-code` changed (`git add -A && git commit -m "iterate-fat-llama: cycle <i> fix"`) and tag the result per the naming convention (`<branch>/iter-<i>`) — this is candidate `C_i`. If `generate-code` reported no changes were needed, still tag current HEAD as `<branch>/iter-<i>` (so `C_i == C_{i-1}`, which will simply score the same next cycle).

If all MAX_CYCLES cycles complete without the early stop in (c):

**f. Roll back to the best cycle.** Apply the rollback rule from `.claude/skills/rules/iterate-fat-llama.md`: find `i* = argmax(score_i)`, tie-broken toward the smaller `i`, and reset the working tree to candidate `C_{i*-1}`: `git reset --hard <branch>/iter-<i*-1>` (use `<branch>/iter-0` when `i*==1`). This discards later cycles that made things worse — nothing is actually lost, every cycle's code is still reachable via its tag. Tell the user which cycle was kept and its score versus the others.

## Step 5 — docs and changelog

1. Run `review-current-state` once more against the final chosen working tree, so `docs/CURRENT_STATE.md` matches what's about to ship.
2. Create or update `CHANGELOG.md` at the repo root with a new entry (version number filled in during Step 6): date, and one bullet per *kept* fix — pull `hypothesis` / `changes` / `notes` from each kept cycle's `generate-code` JSON report. Drop any cycles that were rolled back in step (f); they never happened as far as the shipped result is concerned.

## Step 6 — version bump

1. Read the current `version` in [setup.py](../../../setup.py).
2. Bump it per the version-bump policy in `.claude/skills/rules/iterate-fat-llama.md` (patch by default, minor only if `DIRECTIVES` explicitly asked for new functionality). State which you chose and why.
3. Update `setup.py`'s `version` field and fill the version number into the `CHANGELOG.md` entry from Step 5.

## Step 7 — version-specific PR

Follow the "Remote actions" policy in `.claude/skills/rules/iterate-fat-llama.md` — confirmation before anything remote:

1. Show the user a summary before doing anything remote: the version bump, files changed, the changelog entry, and which cycle's state was kept (or that the loop stopped early because it was already satisfactory). **Ask for explicit confirmation before pushing or opening a PR.**
2. On confirmation: commit the docs/changelog/version-bump changes.
3. Rename the local branch to the version-branch naming convention: `git branch -m v-<new_version>`.
4. Push it: `git push -u origin v-<new_version>`.
5. Open a PR against `main`: `gh pr create --base main --title "v-<new_version>" --body "<changelog entry>"`. If `gh` isn't installed or authenticated, push the branch anyway and give the user the compare link (`https://github.com/bkraad47/fat_llama/compare/main...v-<new_version>`) so they can open the PR by hand.
6. Report the PR URL (or the compare link) as the final result.
