# Changelog

All notable changes to this project will be documented in this file.

## [1.4.2] - 2026-09-06

Produced by an `iterate-fat-llama` run resolving [GitHub issue #20](https://github.com/bkraad47/fat_llama/issues/20) (branch `iterate-fat-llama/20260906-044742`, off `Issue-no-20-unrealistic-final-bitrate-fixing`). Four fix cycles were kept; a fifth confirmed the result on real GPU hardware and made no further changes, having already met this process's bar for a satisfactory result.

### Fixed

- **`upscale()` could produce wildly unrealistic output sample rates and bitrates.** The upscale factor was derived by comparing the requested `target_bitrate_kbps` directly against the source file's own *compressed* bitrate (e.g. an mp3 at 128–192 kbps) with no ceiling — for realistic inputs this routinely landed at a 5–7x factor, driving output sample rates past 250–300 kHz (and effective bitrates over 5000 kbps) for no real informational gain, since the pipeline's own Nyquist-cutoff stage guaranteed the vast majority of that extra bandwidth was silence. The factor is now clamped so the output sample rate never exceeds a realistic consumer/professional playback ceiling (192 kHz) — confirmed end to end on real hardware: a typical mp3 source that used to produce a ~308,700 Hz / ~2391 kbps output now produces 176,400 Hz / ~1876 kbps.
- **Enabling the adaptive filter made `upscale()` impractically slow (30+ minutes for a 15-second clip).** The LMS adaptive filter updated its tap weights one sample at a time in a plain Python loop, so its runtime scaled directly with the (often inflated, see above) sample count. It's now a block-adaptive LMS filter — weights update once per block of samples instead of once per sample, cutting the number of sequential loop iterations by roughly two orders of magnitude while remaining a genuinely adaptive, sequential filter. Confirmed on real hardware: a full baseline run with the adaptive filter enabled now completes in well under 3 minutes total (was ~27.5 minutes for that stage alone).
- **IST's harmonic-reconstruction term never contributed genuine audible detail.** This one took three attempts across the run to actually resolve, documented here for the full picture: it originally spanned exactly one sine cycle across the *entire* buffer regardless of length, landing as an inaudible ~0.066 Hz subsonic artifact; an attempted fix derived its frequency from the signal's own dominant retained frequency instead, which correctly moved it into the audible range but — because it was computed from a whole-multi-second-buffer FFT with a single global dominant peak — turned out to be a constant, static tone (measured at 98 Hz) rather than time-varying detail, which measurably collapsed dynamic range in quiet passages (57.1 dB down to 25.4 dB). The term has been removed entirely rather than revised a fourth time; `iterative_soft_thresholding` now performs the plain FFT/threshold/IFFT round trip with nothing synthetic added on top. Confirmed on real hardware to fully resolve the dynamic-range collapse with no tradeoff — both of this project's own quality scores improved together (coherence 7→9, spectral deviation 9.3→9.9).

### Added

- Two non-GPU-gated regression tests for the sample-rate/bitrate fix (`compute_upscale_factor`'s realistic ceiling, the block-partitioning logic behind the adaptive-filter fix) that run without a CUDA GPU, unlike most of this project's test suite.
- An end-to-end `upscale()` test with the adaptive filter enabled — previously untested at the pipeline level, since it was impractical to run at all before the runtime fix.
- Tests validating the adaptive-filter fix's own claims: that its fast-path setting reproduces the exact prior per-sample behavior, and that it actually cuts the number of sequential iterations as documented.

### Notes

- Audio quality scores (see the README's Audio Quality Scores section) ended the run where they started (coherence 9/10, having dipped to 7/10 mid-run during the harmonic-term investigation before recovering) but spectral deviation rose from 9.0/10 to 9.9/10 — net progress, not a wash: the sample-rate/bitrate and adaptive-filter-runtime fixes are real, measured improvements that don't show up in these two scores at all (they're not what coherence/spectral-deviation measure), and the harmonic-term investigation, despite the mid-run dip, ended by removing a defect (a static audible drone) that existed before this run even started.
- A pre-existing, previously-flagged issue was newly measured and documented rather than fixed this run: `upscale_channels` adds IST's output on top of the original signal rather than replacing it, which — combined with a long-documented threshold-scale issue (an absolute threshold applied to raw-PCM-scale FFT magnitudes barely masks anything) — makes IST's contribution close to a redundant second copy of the signal. Left for a future cycle, along with the threshold-scale issue itself.
- Known gaps remain, carried over from before this run: the test suite doesn't yet exercise `upscale()` end-to-end with both the adaptive filter enabled and a stereo source together, and the repo's reference comparison file (`input_test.flac`) is itself a byproduct of an older pipeline version rather than an independent high-quality master, which caps how meaningful some automated quality comparisons can be.

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
