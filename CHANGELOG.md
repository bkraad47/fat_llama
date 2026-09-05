# Changelog

All notable changes to this project will be documented in this file.

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
