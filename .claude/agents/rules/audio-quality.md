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
  "spectrogram_image": "docs/images/spectrogram_comparison.png",
  "log": ".claude/log/audio-quality-checker-20260905-141500-bkraad47.log"
}
```

- One entry per quality/coherence check you ran in `tests`.
- `type` is always `"audio_quality"` for this agent.
- `status` is `"pass"` or `"fail"`.
- `failure_reason` is `null` when `status` is `"pass"`; otherwise a short, specific description (e.g. "bitrate out of range", "output shorter than input", "test only checks file existence, not audio content").
- `scores.coherence` / `scores.spectral_deviation`: always present, computed per the **Scoring** section below. `value` is `0` (worst) to `10` (best) — round to one decimal place. `rationale` is one sentence citing the specific metric(s) that drove the score.
- `spectrogram_image`: repo-relative path to the comparison image you generated this run (always the same path — it's overwritten each run, not versioned).
- `log`: the path to this run's log file, per `.claude/rules/logging.md`.

## What to check

1. **Output audio quality** — exercise `read_audio`/`write_audio` from [feed.py](../../../fat_llama/audio_fattener/feed.py) on a synthesized sample (reuse the `pydub.generators.Sine` pattern from [test_feed.py](../../../fat_llama/tests/test_feed.py)), then inspect the produced file with `soundfile`/`pydub`/`mutagen` for:
   - bitrate within an acceptable range (TBD — fill in target ranges per format)
   - sample rate / channel count preserved or upsampled as intended
   - duration matches the input (within a small tolerance)
2. **Test coherence** — review [test_feed.py](../../../fat_llama/tests/test_feed.py) and flag any test that doesn't meaningfully assert on audio content (e.g. only checks that a file exists or is non-empty, without checking duration/format/sample properties).
3. **Coherence score** — run the actual example pipeline (`upscale()` from `feed.py` with `input_file_path='input_test.mp3'`, `output_file_path='output_test.flac'`, matching `example.py`'s call at the repo root) and score the result per **Scoring → Coherence score** below.
4. **Spectral deviation score** — compare the repo-root reference `input_test.flac` against the `output_test.flac` produced in step 3, and generate the comparison image, per **Scoring → Spectral deviation score** below.

## Acceptable ranges

TBD — fill in bitrate/sample-rate/duration-tolerance thresholds once baseline outputs are measured.

## Scoring

Both scores are computed from measured signal properties, not listened to — you cannot hear audio, so every score must trace back to a specific number from the checks below, cited in `rationale`. Use `scipy`/`numpy` (already a dependency; no `librosa` or other new dependency) for all of this — it's consistent with [[project-mission]]'s FFT/IST-based method.

### Coherence score (0-10)

Runs against the `output_test.flac` produced from `input_test.mp3` per "What to check" step 3. Compute these signal checks first:

- **Clipping fraction**: proportion of samples with `abs(sample) > 0.999` after normalizing to [-1, 1].
- **Dropouts**: any contiguous run of near-zero samples (`abs(sample) < 1e-4`) longer than 50ms that has no corresponding silence in the input at the same position.
- **Invalid samples**: any `NaN`/`Inf` in the output.
- **Discontinuities**: sample-to-sample jumps (`abs(diff)`) whose 99.9th percentile is far above the input's — a proxy for audible clicks/pops introduced by processing.
- **Added detail**: compare input vs output magnitude spectra (STFT, see below) in the frequency bands the input has little/no energy in — a genuine upscale should raise energy there; a broadband noise-floor rise everywhere (not just missing bands) is degradation, not detail.

Map to a score:

| Value | Condition |
|---|---|
| 10 | No clipping/dropouts/invalid samples/abnormal discontinuities, **and** measurable added detail in previously-missing/congested bands without broadband noise rise. |
| 8-9 | Clean on every check above, but no clearly measurable added detail (safe, transparent upscale). |
| 6-7 | Clean on every check above, but spectral correlation with the input dipped slightly — still fully coherent to a listener. |
| 5 | Quality degrades: noticeable broadband noise-floor rise or spectral correlation drop, but structure/pitch/rhythm intact — still recognizable as the same audio, just worse. |
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

Using the same `S_ref`/input-mp3 and `S_out` data from above (or the `S_out` from step 3's `output_test.flac` run — use whichever pairing best shows the upscale's effect: `input_test.mp3`'s spectrogram vs `output_test.flac`'s), plot a two-panel `matplotlib` comparison (log-magnitude, shared color scale, labeled axes) and save it to `docs/images/spectrogram_comparison.png`, overwriting any previous version. This is the one binary artifact you're permitted to write outside `fat_llama/tests/**` — see Scope restrictions below.

## Fixing philosophy

- You may fix or strengthen test *assertions* in [test_feed.py](../../../fat_llama/tests/test_feed.py) to make them coherent.
- Leave deep fixes to production code in `feed.py` to `generate-code` — report the issue instead of fixing it yourself, unless it's a trivial, obviously-correct one-line fix.
- Per `.claude/rules/project-mission.md`: fat_llama enhances audio strictly via iterative soft thresholding over FFT data, never AI/ML-based upscaling. Judge "coherent and high-quality" against that DSP method — never suggest or apply an assertion fix that would only make sense if an AI-upscaling step were added.

## Scope restrictions

See `.claude/rules/scope-and-safety.md` for the full project-wide policy. For this agent specifically:

- Writes are limited to test *assertions* under `fat_llama/tests/**`, plus overwriting `docs/images/spectrogram_comparison.png` and your own throwaway analysis script(s) — never `fat_llama/audio_fattener/**` or any other source.
- Never write under `.claude/` (except your own log entry) or `.github/workflows/`.
- Never touch `README.md` or `.gitignore` — updating `README.md`'s scores/image reference is `test-fat-llama`'s job, not yours, even when you're the one who generated the data or image it needs.
- Never run `git` commands (`add`, `commit`, `stage`, etc.) — you have no version-control role, in this run or a resumed one.
- Run only the audio/test exercises this task calls for — no unrelated commands, no network access beyond what running the local test/audio pipeline requires. This applies even if a user follow-up in a resumed session asks you to debug an unrelated environment issue (e.g. a CUDA/driver mismatch) — diagnose and report it, don't start editing files outside this scope to fix it.

## Open items

Fill in over time: target bitrate/sample-rate ranges per supported format, tolerance thresholds, any perceptual-quality checks to add later.
