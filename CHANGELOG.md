# Changelog

All notable changes to this project will be documented in this file.

## [1.4.3] - 2026-09-07

Produced by an `iterate-fat-llama` run resolving [GitHub issue #18](https://github.com/bkraad47/fat_llama/issues/18) (branch `Issue-no-18-suggestions-floating-point-dequantize-fixes`). The issue raised a set of technical suggestions for improving MP3→FLAC upscaling fidelity; one fix cycle was kept, and a second confirmed the result with every audio-quality check passing, meeting this process's bar for an early, satisfactory stop.

### Fixed

- **`toggle_normalize=False` had no real effect on the output's level.** `upscale()`'s `toggle_normalize` parameter looked like it controlled whether the output was normalized, but `write_audio()` — the function actually responsible for the final on-disk levels — force-normalized to full scale (0 dBFS) regardless of that flag, so the toggle was effectively decorative. `write_audio()` now takes its own `normalize`/`reference_amplitude` arguments, and `upscale()` wires `toggle_normalize` and the source's own bit-depth full-scale amplitude straight through to them; `toggle_normalize=False` now genuinely preserves the original recording's relative signal level instead of always being rescaled to touch full scale.
- **WAV output was quantized to 24-bit integer PCM despite the pipeline computing internally in 64-bit float.** Every stage of the DSP pipeline (interpolation, IST, adaptive filtering, the Nyquist cutoff) already runs in float64/complex128, but the final write step discarded that precision by writing both FLAC and WAV as 24-bit integer PCM. WAV output now uses libsndfile's 64-bit float (`DOUBLE`) subtype, storing the exact computed values losslessly with no quantization or clamping. FLAC output is unchanged — libsndfile has no float/double FLAC subtype, so 24-bit PCM was already, and remains, FLAC's real ceiling.

### Investigated, no change needed

- FFmpeg's `-drc_scale 0` flag (to avoid dynamic-range compression on MP3 decode) was confirmed already applied in `read_audio()` since v1.1.0, predating this issue — no change was needed.
- Internal computation was confirmed to already be float64/complex128 throughout the pipeline prior to this run; the one real precision gap was the final write step (see Fixed above).

### Notes

- The issue also suggested investigating audio dequantization methods/academic literature to enhance the IST/FFT approach, and studying commercial tools (iZotope Spectral Recovery, Stereotool Delossifier) for inspiration. Neither produced a concrete, verifiable change this run; the project's existing, previously-documented lead — `iterative_soft_thresholding`'s threshold being an absolute cutoff that barely triggers at real audio's amplitude scale — remains the strongest concrete direction for a future cycle. Comment suggestions for ML-based (GAN/transformer) enhancement were out of scope per this project's no-AI/ML-mechanism constraint and were not implemented.
- Audio quality scores held steady across both cycles (coherence 9/10, spectral deviation 9.9/10) — expected, since the baseline scoring config (`toggle_normalize=True`, `target_format='flac'`) doesn't exercise either changed code path (`toggle_normalize=False`, or `target_format='wav'`); both fixes were verified directly via dedicated new tests instead.
- Two coverage gaps were newly identified (not fixed this run): `write_audio`'s integer-subtype clipping safety net on the `normalize=False` path is documented but unasserted, and no test covers `write_audio(audio_format='wav', normalize=True)`. Also still open from before this run: no pipeline-level test exercises the stereo path or non-mp3 source formats end to end, and `input_test.flac` remains a byproduct of an older pipeline version rather than an independent high-quality master.

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

### [1.4.0] - 2026-09-06

#### Fixed

- `write_audio()` was clipping nearly all output audio (missing normalization before writing PCM).
- The LMS "adaptive filter" was a silent no-op that burned most of the pipeline's runtime for zero effect.
- IST's harmonic-reconstruction term was swamped to invisibility at real audio scale.
- Interpolation used naive sample duplication, causing audible imaging artifacts; replaced with proper band-limited (FFT-based) interpolation — also roughly 1000x faster.
- Added an unconditional final filter guaranteeing no output content exceeds the original recording's frequency ceiling — upscaling improves precision/headroom within the original bandwidth, it does not extend it.
- Fixed CI: GitHub's hosted runners have no GPU, so CUDA-dependent tests now skip cleanly there instead of crashing; fixed a stale `cupy-cuda12x`/`cupy-cuda13x` version mismatch in the test workflow.

See [CHANGELOG.md](CHANGELOG.md) for full details, including known gaps and measured audio-quality improvements.

### [1.1.0] - 2024-08-01

#### Chanaged

- Moved adaptive filtering to after normalization and auto-scaling steps.
- Reduced step size for LMS adaptive filter for improved stability.
- Ensured all processing uses CuPy for GPU acceleration.
- Added detailed comments and logging for better traceability.

### [1.0.2] - 2024-07-26

#### Changed

- Remove `logging` from requirements to fix pip bug.

### [1.0.1] - 2024-07-26

#### Changed

- Updated `analytics.py` analysis and spectorgram results.
- Updated `README.md` details.

### [1.0.0] - 2024-07-25

#### Added

- Added support for reading 'ogg', 'flac', and 'wav' file formats and calculating their bitrates correctly.

#### Changed

- Renamed `upscale_mp3_to_flac` method to `upscale` to support multiple source formats.
- Simplified the workflow to focus on 'mp3' to 'flac' conversion with essential steps only.

#### Removed

- Dropped support for 'ape' and 'alac' target formats.

### [0.1.8] - 2024-07-24

#### Added

- Introduced toggle flags for normalization, equalization, amplitude scaling, and gain reduction.
- Enhanced auto-scaling of amplitude based on the original MP3 file when `toggle_scale_amplitude` is `False`.
- Logging for each step of the processing to provide better traceability and debugging.

#### Changed

- Default values for parameters are now set at the function call.
- Refined the upscaling algorithm to ensure better handling of amplitude and gain.
- Renamed the flags for consistency (`toggle_wiener_filter`, `toggle_normalize`, `toggle_equalize`, `toggle_scale_amplitude`, `toggle_gain_reduction`).

#### Fixed

- Fixed issues related to numpy and cupy array conversions.
- Improved error handling for invalid target bitrate values.
- Addressed the issue where the amplitude of the produced signal was significantly weaker than the original.

### [0.1.7] - 2024-07-22

#### Added

- Added methods for MP3 to FLAC conversion with optional processing using CuPy for GPU acceleration.
- Initial version of `upscale_mp3_to_flac` method with parameters for iterative soft thresholding (IST), gain reduction, and equalization.

### [0.1.0] to [0.1.6] - 2024-07-20

#### Added

- Basic functionality for reading MP3 files and writing FLAC files.
- Initial implementation of the new interpolation algorithm and IST for audio processing.