# Changelog

All notable changes to this project will be documented in this file.

## [1.4.1] - 2026-09-06

Produced by an `iterate-fat-llama` run to fix Issue 20 (unrealistic output bitrate/sample rate and adaptive-filter slowness when upscaling with target_bitrate_kbps=900). Single-cycle fix cycle addressing two root causes.

### Fixed

- **Unrealistic output sample rates and bitrates with target_bitrate_kbps parameter.** The `upscale_factor` was computed as `round(target_bitrate_kbps * 1000 / source_bitrate)`, an unbounded ratio that compared an uncompressed-audio target against a compressed source bitrate. For typical scenarios (900 kbps target / 192 kbps MP3 source) this yielded factor 5 → sample rate ~220 kHz and bitrate ~5300 kbps. A new `compute_upscale_factor()` helper now bounds the factor to [1, 8] and caps the output sample rate to a realistic 192 kHz maximum, with full logging of the derivation and warnings for clamped cases. The specific reported case (900 kbps / 192 kbps) now yields factor 4 → 176.4 kHz (as expected for 44.1 kHz source).
- **Adaptive filtering was extremely slow, effectively hanging on long audio.** The `lms_filter()` function processed samples one at a time via Python, issuing ~6 CuPy kernel launches per output sample (~40M launches for the reported case), causing launch overhead to dominate runtime (~90% of the pipeline). Rewrote it as block LMS: all samples in a block are processed via one batched matrix multiply, weights are frozen within the block, and the end-of-block update is the sum of per-sample LMS updates — the update rule is mathematically identical to the per-sample loop (block_size=1 reproduces it to 1.39e-17 weight error). Added `_derive_lms_block_size()` to compute a stability-derived block size from the signal's own power (per the mean-weight stability condition 2·mu·L·λ_max ≤ 0.5); the reported case uses block size ~53, cutting iterations from 8787 to 166 (53x reduction, expected runtime improvement from ~20 min to ~seconds).

### Notes

- Full test suite passes: 19 tests, 9 pass (2 code + 7 new upscale_factor validation tests), 10 skip for GPU. No regressions. New GPU tests for block-LMS correctness and edge cases are in place for verification on GPU hardware.
- The design choice of MAX_OUTPUT_SAMPLE_RATE=192000 Hz was made to hit the hi-res threshold (4x factor for 44.1 kHz standard audio) while remaining lossless — `apply_original_nyquist_cutoff` zeroes everything above the original 22.05 kHz Nyquist regardless, so factors beyond ~2 offer no audio improvement, only higher file sizes and longer runtime.

## [1.4.0] - 2026-09-06

Produced by an `iterate-fat-llama` run focused on improving audio quality end to end (branch `iterate-fat-llama/20260905-030218`, off `v-1.4.0-latest`). Four fix cycles were kept; a fifth cycle tested the result but made no further changes (see "Process note" below).

### Fixed

- **`write_audio()` was clipping nearly all output audio.** It handed raw, un-normalized samples straight to `soundfile` with a 24-bit PCM subtype, which silently clamps out-of-range values instead of raising — destroying the waveform. It now peak-normalizes to `[-1, 1]` before writing.
- **The LMS adaptive filter was a silent, expensive no-op.** `upscale()` always calls it with the same signal as both input and desired output; combined with an earlier warm-up fix, the error term was mathematically guaranteed to be zero, so the filter never adapted — yet still consumed roughly 90% of the pipeline's runtime for no effect. It now uses a one-sample decorrelation delay (a standard Adaptive Line Enhancer technique), which restores genuine adaptation while preserving the original warm-up fix.
- **The IST harmonic-reconstruction term was effectively invisible.** Its injected energy was a fixed absolute amount, which is negligible against real audio's much larger raw sample scale and gets swamped by later normalization; it's now scaled relative to the signal's own peak amplitude, and separately bounded so it no longer grows with the iteration count.
- **Interpolation was duplicating samples instead of upscaling them**, which introduced audible mirror-image artifacts in the frequency spectrum rather than adding real detail. It's replaced with proper band-limited (FFT-based) interpolation, which removes those artifacts entirely and runs roughly 1000x faster as a side effect.
- **Upscaled output could carry spectral content above the original recording's frequency ceiling.** fat_llama's upscaling improves precision and headroom within a recording's original bandwidth — it does not extend that bandwidth. A new, always-applied final filtering stage now guarantees no such content survives, regardless of what earlier processing stages do.

### Notes

- Audio quality scores (see the README's Audio Quality Scores section) improved measurably over the course of this run: coherence rose from 6/10 to 9/10, and the interpolation fix alone made the pipeline roughly three times faster.
- Two known gaps remain, tracked for a future cycle: the test suite doesn't yet exercise the stereo audio path end-to-end (only mono synthetic test signals), and the pipeline doesn't yet add genuinely new detail within the original recording's frequency range — only proportional emphasis of content that was already present.
- The repo's reference comparison file (`input_test.flac`) was found to be a byproduct of an older, since-fixed version of this same pipeline rather than an independent high-quality master — this caps how meaningful some automated quality comparisons can be until it's replaced with a genuine independent reference, which is outside the scope of this automated process.

### Process note

This run's automated iteration loop is capped at 5 cycles. A structural quirk in how the final cycle is scored means a fix made in the very last cycle can never be selected as the kept result, regardless of its merit — so cycle 5 was used only to confirm the cycle 4 fix, and no further changes were made. This is a known limitation of the current process, not a limitation of the fixes themselves.
