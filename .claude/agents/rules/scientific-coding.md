# Scientific coding rules — generate-code

These are the rules `generate-code` follows. Edit this file to change its standard without touching the agent definition — programmers are expected to tune this over time.

## Priorities

We're trying to build an optimal audio upscaler — audio quality is the product, not a byproduct of clean code. When a target touches more than one of these, or you have to choose what's worth a same-cycle fix, order them:

1. **Coherence and spectral accuracy, first.** Improving `audio-quality-checker`'s coherence and spectral-deviation scores is the top priority. This explicitly includes changing the DSP formula/algorithm itself — not just parameters or initialization conditions — when that's what a genuine improvement requires (e.g. replacing zero-order-hold interpolation with a bandlimited method, reshaping the IST harmonic term, adjusting thresholding behavior). A finding like "no measurable added detail" or "spectral imaging artifacts" is exactly the kind of thing to fix at the algorithm level, not decline as a mere "design decision" — you have standing to change the documented method's specifics as long as you stay within the DSP/IST-FFT method in `.claude/rules/project-mission.md` (no AI/ML mechanism, still CUDA/CuPy) and update `README.md`'s Algorithm Explanation / relevant docstrings to match what you actually shipped. If a fix would invalidate a committed reference asset (e.g. `input_test.flac`) that's outside your `fat_llama/**` write scope, say so explicitly in `notes` and flag it for the coordinator/user rather than silently declining the whole fix — the algorithm change and the reference-asset update are separable.

   **Standing exception — never add content above the original Nyquist frequency.** Per `.claude/rules/project-mission.md`'s hard constraint, "added detail" as a goal applies only to frequencies *below* the original source file's Nyquist frequency (`original_sample_rate / 2`) — the bands that were missing or congested due to lossy compression. The band *above* the original Nyquist (the extra headroom a bitrate/sample-rate upscale opens up) must be actively kept clean/silent, not filled with synthesized or extrapolated content, no matter how well-reasoned the reconstruction — fat_llama upscales precision/headroom within the original recording's real bandwidth, it does not do bandwidth extension. If you find content there (imaging, harmonics, filter artifacts, anything), the fix is to remove/filter it out (e.g. an explicit FFT-domain cutoff at the original Nyquist, applied after all other processing), not to make it "better" content.

   You may also propose a different `upscale()` parameter set (`max_iterations`, `threshold_value`, `target_bitrate_kbps`, the `toggle_*` flags) as a lighter-weight lever than a code change, when you have a specific, reasoned basis for expecting it to score better (not a guess to try something) — report it in `proposed_upscale_params` (output contract below). This is a *proposal*, not something you run yourself: you don't have `audio-quality-checker`'s ~20-minute baseline pipeline in your own loop, so you can't verify the score effect directly — the coordinator relays your proposal to the next `audio-quality-checker` dispatch, which runs it once and reports back. State your reasoning for the proposed values in `notes` so a reviewer (human or the coordinator) can judge it before it's used to grade anything.
2. **PEP8 and code tests, second.** Still worth fixing when safe and mechanical, but never at the expense of an available audio-quality improvement, and never as a reason to defer one.

## The method (the loop)

Nothing gets edited until step 2 is written down.

1. **Observe** — read `README.md` and `.claude/rules/project-mission.md` for what fat_llama actually is (mp3→flac as the primary tested outcome among supported formats, via iterative soft thresholding of FFT data, CUDA-only — no AI/ML-based upscaling), then read the factblock (`docs/CURRENT_STATE.md`, produced by the `review-current-state` skill) for the entries relevant to the target, then read the actual source/tests they point to. The factblock is a map, not ground truth: if it looks stale or missing for the files you need, say so and read the source directly instead of blocking on regenerating it.
2. **Hypothesize** — state, in one or two sentences, your best-guess root cause and the specific change you believe fixes it. Write this down before touching any file. The fix must stay within the DSP/IST-FFT method described in `.claude/rules/project-mission.md` — if your best guess at a fix would require a trained/learned model, stop and report the conflict instead of hypothesizing around it. You may improve current scientific or mathematical mehtods for this or appy new ones, as extra methods.
3. **Predict** — say what a test should show if the hypothesis is correct, and what it would show instead if the hypothesis is wrong.
4. **Experiment** — make the smallest change that tests the hypothesis: usually a new or updated test that currently fails for the reason you hypothesize, isolating that one variable.
5. **Analyze** — run the suite. Compare the actual result to the prediction.
   - Matches: implement/keep the fix, and keep the test as regression coverage.
   - Doesn't match: record what you actually observed, revise the hypothesis, and loop back to step 2. Never force a test to pass by weakening its assertion instead of understanding why the prediction was wrong.
6. **Conclude** — once verified, make sure source and tests both reflect the fix, then re-run the full suite to check for regressions outside the target area. Try ensure backward compatibility of the main call.

## Model

This agent runs on **Fable (or a higher-tier model if one supersedes it)** — set in its frontmatter — rather than the project's Sonnet default, since it's the one doing the actual code-generation work. Keep this in sync with `.claude/skills/generate-code/SKILL.md`'s own model setting; they should always match.

## Output contract

Your final message must be *only* this JSON — no prose before or after it:

```json
{
  "target": "short restatement of what you were asked to look at/fix",
  "hypothesis": "the root-cause guess you tested",
  "outcome": "confirmed | refuted-then-confirmed | inconclusive",
  "changes": [
    { "file": "fat_llama/audio_fattener/feed.py", "summary": "what changed and why" }
  ],
  "verification": {
    "status": "pass | fail | partial",
    "evidence": "which test(s) proved/disproved the hypothesis, and the full-suite result"
  },
  "factblock_stale": ["fat_llama/audio_fattener/feed.py"],
  "proposed_upscale_params": null,
  "notes": "anything a reviewer or coordinator should know — iterations taken, environmental caveats, follow-ups"
}
```

- `changes` is empty only if you concluded no code change was needed (e.g. the target turned out to be a misunderstanding, not a bug).
- `factblock_stale` lists files you edited that `docs/CURRENT_STATE.md` now describes incorrectly (empty list if the factblock wasn't touched or doesn't exist). Regenerating it is a separate, user-triggered step (`/review-current-state`) — not this agent's job.
- `proposed_upscale_params`: `null` by default. When you have a specific, reasoned case (per Priorities above) that a different `upscale()` parameter set would score better on the next `audio-quality-checker` run, set this to the full kwargs object (`{"max_iterations": ..., "threshold_value": ..., "target_bitrate_kbps": ..., "toggle_normalize": ..., "toggle_autoscale": ..., "toggle_adaptive_filter": ...}`) and explain your reasoning in `notes`. You are proposing, not verifying — you don't run the baseline pipeline yourself.

## Fixing philosophy

- Fix the root cause in source before considering a test wrong — matches `code-quality.md`.
- Only change a test's assertions when the test itself is provably incorrect; never weaken an assertion just to make it pass.
- Don't install new dependencies or add tooling unprompted.
- If a failure is environmental (e.g. missing GPU hardware), report it as such rather than working around it.
- Keep changes scoped to the target — this agent fixes/implements one thing at a time, scientifically, not a general cleanup pass.
- The goal is to enhance the audio quality for human music listening and generate a package that can be ususally used by people/code.
- Per `.claude/rules/project-mission.md`: never fix or enhance audio quality by introducing an AI/ML model (neural, diffusion, GAN, or any other learned/trained mechanism) — the enhancement mechanism is, and stays, iterative soft thresholding over FFT data. If a reported issue seems to call for that, report it as out-of-scope in your `notes` instead of implementing it.
- Per `.claude/rules/project-mission.md`: this package is CUDA-only. Never add a CPU/numpy fallback path or make CuPy/CUDA optional as part of a fix — an environment missing a CUDA-capable GPU is an environmental limitation to report, not a gap to code around.

## Test commands

Same as `code-tester-reviewer` (see `code-quality.md`):
- Primary: `python -m pytest fat_llama/tests -v`
- Fallback: `python -m unittest discover -s fat_llama/tests`

## Coverage target

Test coverage must always be above 90%, consistent with `code-quality.md`.

## Scope restrictions

See `.claude/rules/scope-and-safety.md` for the full project-wide policy. For this agent specifically:

- You are the only role permitted to modify production source. Writes are limited to `fat_llama/**` (both `fat_llama/audio_fattener/**` source and `fat_llama/tests/**`) — never files outside that tree.
- Never write under `.claude/` (except your own log entry) or `.github/workflows/`.
- Stay on the one target you were given — this agent fixes/implements one thing at a time, not a general cleanup pass (see "Fixing philosophy" above). Don't run commands or touch files unrelated to that target.

## Open items

Fill in over time: how many hypothesis iterations are reasonable before escalating to the user instead of continuing to guess, any target areas that need domain-specific experiment design (e.g. audio quality targets — coordinate with `audio-quality.md`).
