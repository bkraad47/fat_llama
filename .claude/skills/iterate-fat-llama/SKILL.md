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

This skill also runs non-interactively from `.github/workflows/issue-branch-resolve.yml` when `ITERATE_FAT_LLAMA_CI_MODE=1` is set in the environment — see `.claude/skills/rules/iterate-fat-llama.md`'s "CI / GitHub Actions mode" section for the exact behavior differences (Steps 0.2, 0.4, and 7 change) before proceeding if that variable is present.

## Logging

Before Step 0, open this run's log file per `.claude/rules/logging.md` (name: `iterate-fat-llama-<time>-<user>.log`). This is the top-level log for the whole run — append one entry per step, including per-cycle entries for (a)-(e) in the loop below (log each cycle's `score_i` and the dispatch/log-filename of every skill or subagent it calls), the rollback decision in (f) if it happens, and the final PR outcome in Step 7.

## Step 0 — setup

1. Read `README.md` in full (if not already done per the note above) and log a one-line entry recording that the run's mission context is fat_llama's own README plus `.claude/rules/project-mission.md`.
2. `git status`. If there are uncommitted changes that aren't something you just made in this run, stop and ask the user before doing anything else — don't stash or discard in-progress work.
3. Record the starting branch as `BASE_BRANCH` and current HEAD as `BASE_COMMIT`.
4. Decide whether to isolate, per the Naming conventions rule: if `BASE_BRANCH` is `main` (or another shared/long-lived branch), create a new working branch and switch to it — tell the user you've switched branches, `BASE_BRANCH` is left untouched for the rest of this run. Otherwise (the common case — an already single-purpose, disposable branch, e.g. anything CI checks out) stay on `BASE_BRANCH` directly; don't create a second branch just to abandon it later.
5. Tag the untouched starting state per that same naming convention (`iter-0`), on whichever branch you're now operating on. This is candidate `C_0`.

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
3. If the changelog entry is empty (no kept cycles), still create a dated entry with a single bullet: "No changes were needed; all audio quality tests passed." This is a valid outcome and should be documented.
4. Review the document holistcally and edit for clarity, grammar, and style. Make sure it reads like a human-written changelog entry, not a raw machine dump of JSON fields for github.

## Step 6 — version bump

1. Read the current `version` in [setup.py](../../../setup.py).
2. Bump it per the version-bump policy in `.claude/skills/rules/iterate-fat-llama.md` (patch by default, minor only if `DIRECTIVES` explicitly asked for new functionality). State which you chose and why.
3. Update `setup.py`'s `version` field and fill the version number into the `CHANGELOG.md` entry from Step 5.

## Step 7 — ship the result

Follow the "Remote actions" policy in `.claude/skills/rules/iterate-fat-llama.md` — confirmation before anything remote:

1. Show the user a summary before doing anything remote: the version bump, files changed, the changelog entry, and which cycle's state was kept (or that the loop stopped early because it was already satisfactory). **Ask for explicit confirmation before pushing or opening a PR.**
2. On confirmation: commit the docs/changelog/version-bump changes.
3. **If you created an isolated working branch in Step 0.4** (`BASE_BRANCH` was `main`/shared): rename it to the version-branch naming convention (`git branch -m v-<new_version>`) and push it (`git push -u origin v-<new_version>`).
   **If you stayed on `BASE_BRANCH` directly** (the common case): no rename — push `BASE_BRANCH` itself with `git push --force-with-lease origin BASE_BRANCH` (not a plain push; see Remote actions in the rules file for why force-with-lease is needed here specifically).
4. Open a PR against `main` from whichever branch you just pushed: `gh pr create --base main --head <branch> --title "v-<new_version>" --body "<changelog entry>"`. If `gh` isn't installed or authenticated, push anyway and give the user the compare link (`https://github.com/bkraad47/fat_llama/compare/main...<branch>`) so they can open the PR by hand.
5. Report the PR URL (or the compare link) as the final result, and end with the `ITERATE_FAT_LLAMA_DONE_JSON:` block per the Success reporting contract in `.claude/skills/rules/iterate-fat-llama.md` — this is what proves the run actually reached here, not just exited cleanly.
