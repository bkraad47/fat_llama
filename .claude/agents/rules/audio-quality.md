# Audio quality rules — audio-quality-checker

These are the rules `audio-quality-checker` follows. Edit this file to change its behavior without touching the agent definition.

## Model

This agent runs on the **latest available Opus model** — set in its frontmatter — rather than the project's Sonnet default, since judging audio coherence/quality is the most subjective task in this project's pipeline and benefits from the strongest available model.

## Output contract

Your final message must be *only* this JSON — no prose before or after it:

```json
{
  "tests": [
    { "name": "test_read_audio_coherence", "type": "audio_quality", "status": "pass", "failure_reason": null }
  ],
  "scores": {
    "coherence": { "value": 8, "scale": "0-10", "rationale": "no clipping/dropouts/NaNs; high-frequency energy increased over input without added broadband noise" },
    "spectral_deviation": { "value": 7.4, "scale": "0-10", "method": "spectral convergence + correlation on STFT magnitudes (see Scoring below)", "rationale": "convergence=0.81, correlation=0.67 against input_test.flac" }
  },
  "upscale_params_used": {
    "source": "baseline | proposed",
    "params": { "max_iterations": 300, "threshold_value": 0.6, "target_bitrate_kbps": 1400, "toggle_normalize": true, "toggle_autoscale": true, "toggle_adaptive_filter": true }
  },
  "spectrogram_image": "docs/images/spectrogram_comparison.png",
  "log": ".claude/log/audio-quality-checker-20260905-141500-bkraad47.log"
}
```

- One entry per quality/coherence check you ran in `tests`.
- `type` is always `"audio_quality"` for this agent.
- `status` is `"pass"` or `"fail"`.
- `failure_reason` is `null` when `status` is `"pass"`; otherwise a short, specific description (e.g. "bitrate out of range", "output shorter than input", "test only checks file existence, not audio content").
- `scores.coherence` / `scores.spectral_deviation`: always present, computed per the **Scoring** section below. `value` is `0` (worst) to `10` (best) — round to one decimal place. `rationale` is one sentence citing the specific metric(s) that drove the score.
- `upscale_params_used`: always present — `source` is `"baseline"` when you ran the fixed reference config (see "What to check" step 3), or `"proposed"` when your prompt supplied a `generate-code`-proposed parameter set instead; `params` is the exact `upscale()` kwargs actually used this run. This is how the coordinator and a human reading the log can tell which config produced these scores.
- `spectrogram_image`: repo-relative path to the comparison image you generated this run (always the same path — it's overwritten each run, not versioned).
- `log`: the path to this run's log file, per `.claude/rules/logging.md`.

## What to check

1. **Output audio quality** — exercise `read_audio`/`write_audio` from [feed.py](../../../fat_llama/audio_fattener/feed.py) on a synthesized sample (reuse the `pydub.generators.Sine` pattern from [test_feed.py](../../../fat_llama/tests/test_feed.py)), then inspect the produced file with `soundfile`/`pydub`/`mutagen` for:
   - bitrate within an acceptable range (TBD — fill in target ranges per format)
   - sample rate / channel count preserved or upsampled as intended
   - duration matches the input (within a small tolerance)
2. **Test coherence** — review [test_feed.py](../../../fat_llama/tests/test_feed.py) and flag any test that doesn't meaningfully assert on audio content (e.g. only checks that a file exists or is non-empty, without checking duration/format/sample properties).
3. **Coherence score** — run the actual example pipeline and score the result per **Scoring → Coherence score** below, using exactly one of these two configs — **never both, and never anything you invent yourself**:

   - **No proposed params in your prompt (the common case, and always true on a target's first assessment):** use this fixed reference/baseline config (the same call `example.py` makes at the repo root):

     ```python
     upscale(
         input_file_path='input_test.mp3',
         output_file_path='output_test.flac',
         source_format='mp3',
         target_format='flac',
         max_iterations=300,
         threshold_value=0.6,
         target_bitrate_kbps=1400,
         toggle_normalize=True,
         toggle_autoscale=True,
         toggle_adaptive_filter=True
     )
     ```

   - **Your prompt explicitly supplies a `generate-code`-proposed parameter set** (the coordinator relays this after a `generate-code` cycle whose `changes`/`notes` proposed different `upscale()` kwargs as part of its fix): use exactly those proposed values instead of the baseline above. Still one `upscale()` call — don't blend, average, or otherwise combine the two configs.

   Record which one you used in `upscale_params_used` (output contract above). **You get exactly one pipeline run per your own turn/invocation** — never run `upscale()` more than once (baseline vs. proposed, or several proposed variants) inside a single dispatch to search for a better score yourself; that comparison is `generate-code`'s and the coordinator's job across cycles, not something you do internally.

   **Why the baseline (when used) is pinned:** IST's harmonic-reconstruction term (`iterative_soft_thresholding` in `feed.py`) adds a small sinusoid every iteration without bound, so it does not monotonically improve with more iterations — past a certain point, more iterations makes output *worse*, not better. When running the baseline config, do not vary `max_iterations` (or any other parameter) searching for a "best" score yourself, and do not increase it if a run seems to be taking a long time — the qualitative assessment is only meaningful when baseline runs are identical/comparable to each other. If a run is slow, that is expected — let it finish rather than reducing iterations to speed it up.

   **How to actually run it — remote GPU first, local GPU fallback, then give up loudly:** the `fllm-mcp-server` MCP server (tools: `mcp__fllm-mcp-server__test_generate`, `mcp__fllm-mcp-server__get_test_generate_status`) can run this on an on-demand remote GPU instead of your own environment. Follow this order every time you need to produce `output_test.flac`:

   1. **Try the remote path first.**
      a. You do not commit anything yourself — the coordinator (`test-fat-llama`/`iterate-fat-llama`) is responsible for having already committed the state you're meant to test and telling you its branch name and commit SHA in your dispatch prompt. If you're dispatched without a clear branch/commit to test, or the working tree has uncommitted changes beyond what you're permitted to touch (test assertions, the spectrogram image), stop and fall back to local (step 2) rather than committing anything yourself.
      b. You may run `git push` to push that already-made commit to the remote if it isn't there yet (the exception to "never run git commands" below — pushing a commit that already exists locally, not creating one), and `git rev-parse HEAD`/`git status` to confirm the branch/commit you were told matches reality. `test_generate` validates the given `commit` SHA against the live GitHub API and fails immediately (no GPU spun up) if it doesn't match — don't call it against a guess.
      c. Call `mcp__fllm-mcp-server__test_generate` with `branch`, `commit` (the full 40-char SHA you just confirmed), and `params` (the exact `UpscaleParams` — baseline or proposed, per above; omit `params` entirely only if you genuinely want `example.py`'s own defaults, which should match the baseline anyway). It returns immediately with `{job_id, status}` — `status: "failed"` means it was rejected outright (concurrency cap of 3 concurrent jobs hit, branch not found, or commit mismatch) with no GPU cost incurred; if so, fall back to local (step 2) rather than retrying — the rejection reasons aren't transient enough to be worth an immediate retry.
      d. If queued, poll `mcp__fllm-mcp-server__get_test_generate_status` with the `job_id` until it reaches a terminal state (`succeeded`, `failed`, or `timed_out`) — space polls out sensibly (this runs on a real GPU VM end to end; polling every few seconds wastes calls for no benefit) rather than tight-looping.
      e. On `succeeded`: the job already committed the output file back onto that branch and pushed it. Run `git fetch` then `git checkout`/`reset --hard` your local branch to that resulting commit SHA (from the status response) — another git-command exception, needed only to receive the remote job's own output, never to author new local changes — so `output_test.flac` matches what the remote run produced, and continue your analysis against that file as normal.
      f. On `failed` or `timed_out`: read the stage-tagged `error`/`error_description` (clone / commit-mismatch / deps / CUDA / script / push) from the status response, log it, and fall back to local (step 2) — don't retry the remote path yourself.
   2. **Local GPU fallback** — if the remote path wasn't available at all (server unreachable, tools not present/callable) or failed for any reason above, fall back to running `upscale()` locally exactly as before — see "How to run it without being killed" immediately below.
   3. **Nothing worked** — if the remote path failed or was unavailable *and* this local environment has no CUDA-capable GPU (per `.claude/rules/project-mission.md`'s CUDA-only stance — this is an environmental limitation to report, never a CPU-fallback to build), stop and report a clear CLI-style error identifying which paths were tried and why each failed, rather than proceeding with no valid `output_test.flac`. Reflect this in your `tests`/`notes` output rather than silently producing degenerate scores.

   **How to run it locally without being killed** (step 2 above, or whenever the remote path isn't used): this baseline run takes roughly 20 minutes end to end on the reference GPU — measured breakdown: `read_audio` ~0s, interpolation+IST (both channels, 300 iterations) ~105s, normalize ~0s, LMS adaptive filter alone ~9-10 minutes *per channel* (~18-19 min for both) — because `lms_filter` in `feed.py` is a plain per-sample Python loop, not a vectorized op. That means IST/`max_iterations` is *not* the lever for runtime (it's only ~1.75 min of the total); do not shorten it to fit a time budget. Instead, never run this call as a single blocking foreground command — a single Bash tool call (including one that blocks via a manual `sleep`/poll loop) is capped at 10 minutes and will be killed before a ~20 minute run finishes. Launch it as a background command (`run_in_background: true` on the Bash tool, writing to a script that itself calls `upscale(...)` and then something detectable like printing `DONE` at the end) and wait for its own completion notification rather than polling in a blocking loop.

   The pipeline call itself still upscales `input_test.mp3` (that's the MP3→FLAC path this project builds and tests against) — but the **reference signal for every coherence check below is `input_test.flac`, not `input_test.mp3`**. Compare `output_test.flac` against `input_test.flac` throughout the Coherence score checks (dropout correspondence, discontinuity baseline, added-detail bands), the same reference file the Spectral deviation score already uses.
4. **Spectral deviation score** — compare the repo-root reference `input_test.flac` against the `output_test.flac` produced in step 3, and generate the comparison image, per **Scoring → Spectral deviation score** below.

## Acceptable ranges

TBD — fill in bitrate/sample-rate/duration-tolerance thresholds once baseline outputs are measured.

## Scoring

Both scores are computed from measured signal properties, not listened to — you cannot hear audio, so every score must trace back to a specific number from the checks below, cited in `rationale`. Use `scipy`/`numpy` (already a dependency; no `librosa` or other new dependency) for all of this — it's consistent with [[project-mission]]'s FFT/IST-based method.

### Coherence score (0-10)

Runs against the `output_test.flac` produced from `input_test.mp3` per "What to check" step 3, graded against the **`input_test.flac` reference** (not `input_test.mp3` — see step 3's note). Resample/align the two the same way the Spectral deviation algorithm does (§below: `resample_poly` to a common rate, trim to `min` length) before computing these checks:

- **Clipping fraction**: proportion of samples with `abs(sample) > 0.999` after normalizing to [-1, 1].
- **Dropouts**: any contiguous run of near-zero samples (`abs(sample) < 1e-4`) longer than 50ms that has no corresponding silence in `input_test.flac` at the same position.
- **Invalid samples**: any `NaN`/`Inf` in the output.
- **Discontinuities**: sample-to-sample jumps (`abs(diff)`) whose 99.9th percentile is far above `input_test.flac`'s — a proxy for audible clicks/pops introduced by processing.
- **Added detail**: compare `input_test.flac` vs output magnitude spectra (STFT, see below) in the frequency bands the reference has little/no energy in **below the original source's Nyquist frequency** — a genuine upscale should raise energy there; a broadband noise-floor rise everywhere (not just missing bands) is degradation, not detail.
- **Content above the original Nyquist frequency**: per `.claude/rules/project-mission.md`'s hard constraint, the output must carry no meaningful energy above the original source file's Nyquist frequency (`original_sample_rate / 2`) — that band should sit at/near the FFT noise floor, many orders of magnitude below in-band content. This is checked independently of "added detail" above: it is never scored as a positive (fat_llama does not do bandwidth extension), and any measurable energy there — imaging, harmonics, filter artifacts, anything — is a defect. Report it as a failing test with a measured level (dB relative to in-band peak), and factor it into the coherence score below.

Map to a score. **A precondition for any score above 5**: content above the original Nyquist frequency must sit at/near the FFT noise floor (per the check above) — meaningful energy there is treated the same as a coherence-degrading artifact (clipping, dropouts, etc.), never offset by other clean checks or by below-Nyquist added detail.

| Value | Condition |
|---|---|
| 10 | No clipping/dropouts/invalid samples/abnormal discontinuities/above-Nyquist content, **and** measurable added detail in previously-missing/congested bands *below* the original Nyquist frequency without broadband noise rise. |
| 8-9 | Clean on every check above (including no above-Nyquist content), but no clearly measurable added detail below the original Nyquist (safe, transparent upscale). |
| 6-7 | Clean on every check above, but spectral correlation with the input dipped slightly — still fully coherent to a listener. |
| 5 | Quality degrades: noticeable broadband noise-floor rise, spectral correlation drop, or measurable content above the original Nyquist frequency, but structure/pitch/rhythm intact — still recognizable as the same audio, just worse. |
| 3-4 | Noticeable degradation: clipping fraction or dropouts present at a moderate level, still clearly recognizable as the same content. |
| 1-2 | Severe artifacts: heavy clipping, long dropouts, or extreme spectral distortion — barely recognizable. |
| 0 | Jibberish: `NaN`/`Inf` present, near-total silence, catastrophic clipping (>50%), or output spectral content has no meaningful correlation to the input — a human would not recognize it as the same audio. |

### Spectral deviation score (0-10)

Compares the repo-root reference `input_test.flac` (ground truth) against `output_test.flac` (produced by upscaling `input_test.mp3`, per "What to check" step 4). Fixed algorithm — run it the same way every time so scores are comparable run to run:

1. Read both files with `soundfile`, downmix to mono (average channels), cast to `float64`.
2. If sample rates differ, resample the lower-rate signal up with `scipy.signal.resample_poly` so both share one rate.
3. Trim both to the same length (`min` of the two).
4. Compute magnitude spectrograms with `scipy.signal.stft` (`nperseg=2048`, `noverlap=1024`) for both signals: `S_ref`, `S_out`.
5. Normalize each spectrogram by its own max magnitude (removes pure amplitude/gain differences from normalization/auto-scaling — this measures spectral shape, not loudness).
6. `spectral_convergence = 1 - norm(|S_out| - |S_ref|, 'fro') / norm(|S_ref|, 'fro')`, clamped to `[0, 1]`.
7. `correlation = pearson_correlation(flatten(|S_out|), flatten(|S_ref|))`, clamped to `[0, 1]` (treat negative correlation as 0).
8. `combined = (spectral_convergence + correlation) / 2`.
9. `deviation_score = round(combined * 10, 1)` — `0` = very different, `10` = same wave.

Report both `spectral_convergence` and `correlation` in the `rationale` (see output contract) so a caller can see if the two metrics disagree.

### Spectrogram comparison image

A separate, three-way comparison from the Spectral deviation score above (which only compares `S_ref`/`S_out`) — this image also includes the MP3 source, and reuses `analysis.py`'s own reading/plotting conventions rather than a new implementation, so the image stays consistent with the rest of the project's own analysis tooling:

1. Read all three files with `analysis.py`'s own `read_mp3`/`read_flac` functions (import them from `analysis.py` rather than reimplementing) — `input_test.mp3` (the compressed source), `input_test.flac` (the reference), `output_test.flac` (the pipeline's output) — and peak-normalize each with `analysis.py`'s `normalize`.
2. Compute each signal's spectrogram with `scipy.signal.spectrogram` (`nperseg=2048`), matching `analysis.py`'s own `compare_signals` method — not `scipy.signal.stft` (that's the separate Spectral deviation *score*'s algorithm above; the image uses `analysis.py`'s own convention instead).
3. Plot three stacked panels (one per signal, in the order mp3 → reference → output), each titled with the signal name and its sample rate, sharing one `x` axis (time) and one color scale (`vmin=-120, vmax=0` dB, matching `analysis.py`'s own convention) with a single shared colorbar.
4. **Cap every panel's displayed frequency range at the original source's Nyquist frequency** (`mp3_sample_rate / 2` — read the MP3's own sample rate, don't assume 44100/22050) and mask/exclude spectrogram data above it before plotting, not merely set the axis limit — per `.claude/rules/project-mission.md`'s "no content above the original Nyquist frequency" constraint, content or ghosting above that line is out of scope for this comparison and must not be shown, even if it's present in a source file (e.g. the reference's own legacy artifacts).
5. Save to `docs/images/spectrogram_comparison.png`, overwriting any previous version. This is the one binary artifact you're permitted to write outside `fat_llama/tests/**` — see Scope restrictions below.

## Fixing philosophy

- You may fix or strengthen test *assertions* in [test_feed.py](../../../fat_llama/tests/test_feed.py) to make them coherent.
- Leave deep fixes to production code in `feed.py` to `generate-code` — report the issue instead of fixing it yourself, unless it's a trivial, obviously-correct one-line fix.
- Per `.claude/rules/project-mission.md`: fat_llama enhances audio strictly via iterative soft thresholding over FFT data, never AI/ML-based upscaling. Judge "coherent and high-quality" against that DSP method — never suggest or apply an assertion fix that would only make sense if an AI-upscaling step were added.

## Scope restrictions

See `.claude/rules/scope-and-safety.md` for the full project-wide policy. For this agent specifically:

- Writes are limited to test *assertions* under `fat_llama/tests/**`, plus overwriting `docs/images/spectrogram_comparison.png` and your own throwaway analysis script(s) — never `fat_llama/audio_fattener/**` or any other source.
- Never write under `.claude/` (except your own log entry) or `.github/workflows/`.
- Never touch `README.md` or `.gitignore` — updating `README.md`'s scores/image reference is `test-fat-llama`'s job, not yours, even when you're the one who generated the data or image it needs.
- Never run `git add`/`commit`/`stage` — you have no authoring role over what gets committed, in this run or a resumed one. The sole exceptions, both narrowly scoped to the remote-GPU procedure above and never used to author new changes: `git push` (only to push a commit the coordinator already made, never one you made), `git rev-parse`/`git status` (to confirm what you were told to test), and `git fetch` + `git checkout`/`reset --hard` (only to receive the `test_generate` job's own committed output). Using any of these for anything else is out of scope.
- Run only the audio/test exercises this task calls for — no unrelated commands, no network access beyond what running the local test/audio pipeline or the `fllm-mcp-server` MCP calls above require. This applies even if a user follow-up in a resumed session asks you to debug an unrelated environment issue (e.g. a CUDA/driver mismatch) — diagnose and report it, don't start editing files outside this scope to fix it.
- The remote-GPU path spins up real, billed cloud infrastructure per call — never invoke `mcp__fllm-mcp-server__test_generate` speculatively, more than once per baseline/proposed-config run needed, or to "double check" a result you already have.

## Open items

Fill in over time: target bitrate/sample-rate ranges per supported format, tolerance thresholds, any perceptual-quality checks to add later.

The `fllm-mcp-server` MCP integration (remote-GPU-first generation) was registered locally (`claude mcp add ... --scope local`, config lives outside this repo, never committed) but not yet exercised end-to-end in a live session — the server was registered mid-session, so its tools weren't visible to `ToolSearch` yet in the session that wrote this. Confirm the tool names (`mcp__fllm-mcp-server__test_generate`, `mcp__fllm-mcp-server__get_test_generate_status`) actually resolve in a fresh session before relying on the remote path; if they don't, this falls straight through to the local-GPU fallback anyway, which is already the previously-working behavior.
