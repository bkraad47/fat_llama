# fat_llama — Current State

Regenerable snapshot produced by the `review-current-state` skill. Scope: the whole repository (tracked files only, per `git ls-files`, excluding `.claude/` and `.github/`). Do not hand-edit — rerun the skill to refresh.

## File tree

```
fat_llama/
├── .gitignore
├── .mcp.json
├── CHANGELOG.md
├── LICENSE
├── Manifest.in
├── README.md
├── analysis.py
├── example.py
├── requirements.txt
├── setup.py
├── input_test.mp3
├── input_test.flac
├── output_test.flac
├── docs/
│   ├── CURRENT_STATE.md
│   └── images/
│       ├── logo.jpg
│       ├── spectrogram_comparison.png
│       └── theory.png
└── fat_llama/
    ├── __init__.py
    ├── audio_fattener/
    │   ├── __init__.py
    │   └── feed.py
    └── tests/
        ├── __init__.py
        └── test_feed.py
```

## fat_llama/audio_fattener/feed.py

Module-level constants: `MAX_REALISTIC_SAMPLE_RATE_HZ = 192000` — the realistic consumer-playback sample-rate ceiling used to bound `compute_upscale_factor`'s output; see that function's factblock. `IST_BLOCK_SIZE = 8192` — default windowed-overlap-add (WOLA) block size (samples, ~186 ms at 44.1 kHz) for `iterative_soft_thresholding`'s local-block processing; see that function's factblock.

### `read_audio(file_path, audio_format) -> (int, np.ndarray, float|None, AudioSegment)`
**File:** fat_llama/audio_fattener/feed.py:27
**Kind:** function
**Description:** Reads an audio file via `pydub.AudioSegment` (with `-drc_scale 0` passed to ffmpeg), extracts raw PCM samples as `float64`, the sample rate, and — via format-specific `mutagen` readers (mp3/flac/ogg/wav) or a duration-based estimate otherwise — the source bitrate. Reshapes to `(-1, 2)` for 2-channel audio.
**Parameters:**
- `file_path` (`str`): path to the input audio file; raises `FileNotFoundError` if missing.
- `audio_format` (`str`): one of `'mp3'`, `'flac'`, `'ogg'`, `'wav'` (others fall back to a duration-derived bitrate estimate).
**Returns:** `(sample_rate, samples, bitrate, audio)` — `sample_rate` (int), `samples` (`np.ndarray`, raw-PCM-scale float64, mono 1-D or stereo `(N, 2)`), `bitrate` (float or `None`), `audio` (the `pydub.AudioSegment`, used downstream for `audio.channels` and `audio.max_possible_amplitude`).
**Usage:**
```python
sample_rate, samples, bitrate, audio = read_audio('input_test.mp3', audio_format='mp3')
```

### `write_audio(file_path, sample_rate, data, audio_format, normalize=True, reference_amplitude=None) -> None`
**File:** fat_llama/audio_fattener/feed.py:81
**Kind:** function
**Description:** Writes `data` to a FLAC (`PCM_24`, libsndfile's real ceiling for FLAC) or WAV (`DOUBLE`, true 64-bit float, lossless) file via `soundfile`. When `normalize=True` (default), peak-normalizes to full scale (clipping non-float subtypes to `[-1, 1]`). When `False`, divides by a fixed `reference_amplitude` (typically the source's own bit-depth full-scale value) instead of the buffer's own peak — a lossless domain conversion for float subtypes, clipped for integer subtypes — so the written level genuinely reflects the source's relative loudness instead of always being stretched to 0 dBFS.
**Parameters:**
- `file_path` (`str`): output path.
- `sample_rate` (`int`): output sample rate.
- `data` (`np.ndarray`): audio samples to write (raw-PCM-scale or otherwise, depending on `normalize`).
- `audio_format` (`str`): `'flac'` or `'wav'` (else raises `ValueError`).
- `normalize` (`bool`): force full-scale peak-normalize before writing. Default `True`.
- `reference_amplitude` (`float|None`): fixed divisor used when `normalize=False`; falls back to peak-based scaling if not given.
**Returns:** `None` (writes the file as a side effect).
**Usage:**
```python
write_audio('output_test.flac', 44100, samples, audio_format='flac')
```

### `new_interpolation_algorithm(data, upscale_factor) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:200
**Kind:** function
**Description:** Upsamples a single-channel real signal via bandlimited FFT-domain interpolation: `cp.fft.rfft` the input, zero-pad the one-sided spectrum to the upscaled length's rfft size (introducing no new spectral content, unlike zero-order-hold's imaging), `cp.fft.irfft` back to the longer signal, rescaled by `upscale_factor` to correct `irfft`'s own normalization. `upscale_factor == 1` is a no-op passthrough.
**Parameters:**
- `data` (`cp.ndarray`): single-channel input audio data.
- `upscale_factor` (`int`): integer factor to upsample by.
**Returns:** `cp.ndarray` — upscaled data, length `len(data) * upscale_factor`, band-limited to the original Nyquist frequency.
**Usage:**
```python
expanded_channel = new_interpolation_algorithm(channel, upscale_factor=7)
```

### `initialize_ist(data, threshold) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:270
**Kind:** function
**Description:** Initializes IST by hard-thresholding `data` in the time domain: keeps samples whose absolute value exceeds `threshold * cp.max(cp.abs(data))` (a **peak-relative** fraction, 0-1, as of cycle 5 — fixed from a prior absolute-cutoff comparison, which barely masked anything at real audio's raw-PCM scale), zeroing the rest. Handles an empty `data` array as a no-op passthrough.
**Parameters:**
- `data` (`cp.ndarray`): input audio data.
- `threshold` (`float`): peak-relative threshold fraction (0-1).
**Returns:** `cp.ndarray` — thresholded data, same shape as `data`.
**Usage:**
```python
data_thres = initialize_ist(data, threshold=0.6)
```

### `_ist_chain(data, max_iter, threshold, convergence_tol) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:296
**Kind:** function
**Description:** The actual IST fixed-point iteration (new in cycle 5, factored out of `iterative_soft_thresholding`): initializes via `initialize_ist`, then repeats FFT → peak-relative frequency-domain threshold (with the DC bin, index 0, always excluded to prevent an asymmetric transient from making it the dominant/sole surviving bin) → IFFT, breaking early once a pass's largest per-sample change falls below `convergence_tol` times the initial post-threshold peak magnitude (hard-threshold IST is an exact fixed-point projection, so further passes past convergence recompute an identical result). Operates on whichever buffer it's given — the whole signal for short inputs, or one windowed block for long inputs (see `iterative_soft_thresholding`/`_wola_block_process`).
**Parameters:**
- `data` (`cp.ndarray`): buffer to run IST on (whole signal or one analysis block).
- `max_iter` (`int`): maximum passes before giving up on convergence.
- `threshold` (`float`): peak-relative threshold fraction (0-1).
- `convergence_tol` (`float`): relative early-exit tolerance.
**Returns:** `cp.ndarray` — IST-processed buffer, same length as `data`.
**Usage:**
```python
result = _ist_chain(data, max_iter=300, threshold=0.6, convergence_tol=1e-6)
```

### `_periodic_hann_window(n) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:358
**Kind:** function
**Description:** DFT-even ("periodic") Hann window (`sin^2(pi*k/n)` form), distinct from the symmetric/endpoint-zero Hann window — satisfies the constant-overlap-add (COLA) property at 50% hop, needed for `_wola_block_process`'s reconstruction.
**Parameters:**
- `n` (`int`): window length in samples.
**Returns:** `cp.ndarray` — length-`n` window, `float64`.
**Usage:**
```python
window = _periodic_hann_window(8192)
```

### `_sqrt_hann_window(n) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:378
**Kind:** function
**Description:** Square root of `_periodic_hann_window`, used as both the analysis and synthesis window in `_wola_block_process` — applying it twice is equivalent to applying the COLA-compliant periodic Hann window once.
**Parameters:**
- `n` (`int`): window length in samples.
**Returns:** `cp.ndarray` — length-`n` window, `float64`.
**Usage:**
```python
window = _sqrt_hann_window(8192)
```

### `_wola_block_process(data, block_size, frame_processor) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:398
**Kind:** function
**Description:** Applies `frame_processor` independently to overlapping (50% hop), sqrt-Hann-windowed blocks of `data` (analysis), then reconstructs via weighted overlap-add (WOLA): the same sqrt-Hann window is applied again on the synthesis side, and the sum is divided by the real accumulated window-squared weight at each sample (not a flat scalar), including at the zero-padded edges. New in cycle 5, backing `iterative_soft_thresholding`'s block/local processing.
**Parameters:**
- `data` (`cp.ndarray`): full-length signal to process.
- `block_size` (`int`): analysis/synthesis window length in samples (must be even; hop is `block_size // 2`).
- `frame_processor` (`callable`): `cp.ndarray -> cp.ndarray` of the same length, applied to each windowed block independently.
**Returns:** `cp.ndarray` — reconstructed signal, same length as `data`.
**Usage:**
```python
result = _wola_block_process(data, 8192, lambda frame: _ist_chain(frame, 300, 0.6, 1e-6))
```

### `iterative_soft_thresholding(data, max_iter, threshold, convergence_tol=1e-6, block_size=IST_BLOCK_SIZE) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:456
**Kind:** function
**Description:** Public IST entry point; as of cycle 5, a dispatcher over `_ist_chain`. Signals no longer than `block_size` run `_ist_chain` directly on the whole buffer (unchanged fast path, matches every short synthetic-buffer unit test). Longer signals are processed via `_wola_block_process` — each 50%-overlap, sqrt-Hann-windowed ~186ms block runs its own independent, converging `_ist_chain` (so each block's peak-relative threshold reflects only that block's local content), then WOLA-reconstructed. The per-block approach was added this cycle after finding that a single whole-buffer-FFT threshold lets one loud passage's globally-retained content leak (via `ifft`'s whole-buffer basis) into temporally distant quiet passages once peak-relative thresholding (see `initialize_ist`) actually engages — measured as a 57.8 dB quiet-segment RMS rise before this fix, brought back within the original ~12 dB bound after it. A prior per-iteration synthetic harmonic-reconstruction term was tried across cycles 2-3 and removed in cycle 4 after being found to inject a constant, non-source tone; this function does not reintroduce it.
**Parameters:**
- `data` (`cp.ndarray`): input audio data (typically the interpolated, pre-IST channel).
- `max_iter` (`int`): ceiling on IST passes (per block, if blocked); actual count may be lower due to convergence early-exit.
- `threshold` (`float`): peak-relative threshold fraction (0-1), applied each iteration as `threshold * max(abs(current))`.
- `convergence_tol` (`float`): relative early-exit tolerance. Default `1e-6`.
- `block_size` (`int`): WOLA block size in samples. Default `IST_BLOCK_SIZE` (8192).
**Returns:** `cp.ndarray` — the IST-processed data (added onto the interpolated signal by `upscale_channels`, not used standalone).
**Usage:**
```python
ist_changes = iterative_soft_thresholding(expanded_channel, max_iter=300, threshold=0.6)
```

### `_lms_block_ranges(start, n, block_size) -> Generator[(int, int)]`
**File:** fat_llama/audio_fattener/feed.py:360
**Kind:** function
**Description:** Pure-Python (no CuPy) generator partitioning `[start, n)` into consecutive, non-overlapping chunks of at most `block_size` samples, covering the range exactly once in order. Extracted standalone so `lms_filter`'s block-partitioning logic can be unit-tested without a GPU.
**Parameters:**
- `start` (`int`): first index to include.
- `n` (`int`): one past the last index to include.
- `block_size` (`int`): maximum chunk length (`>= 1`).
**Returns:** generator of `(block_start, block_end)` pairs, `block_end` exclusive.
**Usage:**
```python
list(_lms_block_ranges(33, 1000, 256))
```

### `lms_filter(signal, desired, mu=0.001, num_taps=32, delay=1, block_size=256, return_weights=False) -> cp.ndarray | (cp.ndarray, cp.ndarray)`
**File:** fat_llama/audio_fattener/feed.py:389
**Kind:** function
**Description:** Block-adaptive LMS Adaptive Line Enhancer (ALE): predicts `desired[i]` from `signal` lagged by `delay` samples using `num_taps` filter taps, held fixed within each block of up to `block_size` samples and updated once per block via the block-averaged instantaneous gradient (`block_size=1` reproduces the exact original per-sample update). `delay >= 1` makes the filter a genuine (non-degenerate) estimation problem even when `signal is desired` (as `upscale()` calls it). Tap weights are near-identity-initialized (`w[0] = 1.0`) to avoid a warm-up dropout, and clipped to `[-1e10, 1e10]` for numerical safety.
**Parameters:**
- `signal` (`cp.ndarray`): input signal (the filter's predictor source, lagged by `delay`).
- `desired` (`cp.ndarray`): desired/target output signal.
- `mu` (`float`): LMS step size. Default `0.001`.
- `num_taps` (`int`): number of filter taps. Default `32`.
- `delay` (`int`): ALE decorrelation lag in samples; must be `>= 1` for the self-referential case to be non-degenerate. Default `1`.
- `block_size` (`int`): samples per block-adaptive weight update. Default `256`.
- `return_weights` (`bool`): if `True`, also return the final tap-weight vector. Default `False`.
**Returns:** `cp.ndarray` (filtered signal), or `(filtered_signal, w)` if `return_weights=True`.
**Usage:**
```python
filtered_channel = lms_filter(normalized_channel, normalized_channel)
```

### `_cap_ist_changes_to_baseline_peak(expanded_channel, ist_changes, max_rounds=20, dominant_band_ratio=0.002, safety_margin=1.2) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:619
**Kind:** function
**Description:** Fixes the same peak-inflation regression as before (IST's peak-relative boost — usually of low-frequency content — inflates a channel's peak, which `upscale()`'s later autoscale/normalize stages then divide the *whole* channel down by, attenuating even untouched bands). Cycle 6's original fix used one uniform per-channel scalar, which closed the attenuation regression but was found (audio-quality-checker, real audio) to also suppress IST's own contribution almost everywhere else — the small scalar needed to tame the dominant band applied to every other frequency too, including legitimately-added quiet high-frequency detail. Cycle 7's frequency-selective rewrite: splits `ist_changes`'s FFT spectrum (`cp.fft.rfft`) into a "dominant" component (bins `>= dominant_band_ratio` of the spectrum's own peak magnitude) and a "residual" component; iteratively shrinks (up to `max_rounds` rounds) only the dominant component toward the baseline peak, leaving the residual untouched; falls back to one additional whole-signal uniform shrink (cycle 6's original method) only if the frequency-selective result's own peak still exceeds `baseline_peak * safety_margin`. Verified (own adversarial synthetic case: loud 100Hz + quiet 8000Hz): combined-peak excess reduced to ~8% of uncapped (vs cycle 6's own ~15% bound) while the quiet component's survival rose from ~5.9% (cycle 6) to ~100%. As with cycle 6, no closed-form guarantee the combined peak never exceeds baseline (only `scale=0` on the dominant component would guarantee that when content is exactly in-phase, which would reopen "IST adds nothing") — this approaches, not guarantees, the baseline peak.
**Parameters:**
- `expanded_channel` (`cp.ndarray`): pre-IST, interpolated channel (the baseline peak).
- `ist_changes` (`cp.ndarray`): IST's output for this channel, about to be added onto `expanded_channel`.
- `max_rounds` (`int`): maximum iterative shrink rounds (dominant-band pass and safety-net fallback). Default `20`.
- `dominant_band_ratio` (`float`): a bin is "dominant" (subject to shrinking) if its magnitude is `>=` this fraction of the spectrum's own peak; else "residual" (untouched). Default `0.002`.
- `safety_margin` (`float`): the whole-signal safety-net fallback only engages if the frequency-selective candidate's peak still exceeds `baseline_peak * safety_margin`. Default `1.2`.
**Returns:** `cp.ndarray` — `ist_changes` reshaped in the FFT domain per the above (unchanged if no capping needed).
**Usage:**
```python
ist_changes = _cap_ist_changes_to_baseline_peak(expanded_channel, ist_changes)
```

### `upscale_channels(channels, upscale_factor, max_iter, threshold) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:915
**Kind:** function
**Description:** Runs `new_interpolation_algorithm`, then `iterative_soft_thresholding`, then (as of cycle 6) `_cap_ist_changes_to_baseline_peak` (added onto the interpolated result) sequentially over each channel in `channels.T`, then stacks the processed channels back into a single array.
**Parameters:**
- `channels` (`cp.ndarray`): input audio channels, shape `(N, C)`.
- `upscale_factor` (`int`): interpolation factor.
- `max_iter` (`int`): IST iteration count.
- `threshold` (`float`): IST threshold.
**Returns:** `cp.ndarray` — processed/upscaled channels, shape `(N * upscale_factor, C)`.
**Usage:**
```python
upscaled_channels = upscale_channels(channels, upscale_factor=7, max_iter=300, threshold=0.6)
```

### `normalize_signal(signal) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:576
**Kind:** function
**Description:** Peak-normalizes a signal to `[-1, 1]` by dividing by its own maximum absolute value.
**Parameters:**
- `signal` (`cp.ndarray`): input audio signal.
**Returns:** `cp.ndarray` — normalized signal.
**Usage:**
```python
normalized = normalize_signal(scaled_upscaled_channels[:, 0])
```

### `apply_original_nyquist_cutoff(signal, original_sample_rate, new_sample_rate) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:589
**Kind:** function
**Description:** Unconditional final safety stage guaranteeing no spectral content survives above the *original* source's Nyquist frequency (`original_sample_rate / 2`), regardless of what earlier stages did — the hard constraint in `.claude/rules/project-mission.md`. Implemented via `cp.fft.rfft` → zero all bins above the original Nyquist → `cp.fft.irfft`, entirely on CuPy/CUDA.
**Parameters:**
- `signal` (`cp.ndarray`): fully processed single-channel signal, sampled at `new_sample_rate`.
- `original_sample_rate` (`int`): the source's original sample rate; the cutoff is `original_sample_rate / 2`.
- `new_sample_rate` (`int|float`): the actual sample rate of `signal`.
**Returns:** `cp.ndarray` — `signal` with all content above the original Nyquist frequency removed, same length.
**Usage:**
```python
cutoff_channel = apply_original_nyquist_cutoff(filtered_channel, sample_rate, new_sample_rate)
```

### `compute_upscale_factor(sample_rate, source_bitrate_bps, target_bitrate_kbps) -> int`
**File:** fat_llama/audio_fattener/feed.py:645
**Kind:** function
**Description:** Derives an integer upscale factor as `round(target_bitrate_kbps * 1000 / source_bitrate_bps)` (falling back to `4` if `source_bitrate_bps` is falsy), floored at `1` (never downscales) and clamped so `sample_rate * factor` never exceeds `MAX_REALISTIC_SAMPLE_RATE_HZ` (192 kHz) — closing a gap where unbounded ratios could drive unrealistic (250-300+ kHz) output sample rates.
**Parameters:**
- `sample_rate` (`int`): source sample rate, Hz.
- `source_bitrate_bps` (`float|None`): source's own bitrate in bits/sec, or `None`.
- `target_bitrate_kbps` (`int`): caller-requested target bitrate in kbps (pre-validated by the caller).
**Returns:** `int` — upscale factor, `>= 1`, such that `sample_rate * factor <= MAX_REALISTIC_SAMPLE_RATE_HZ` whenever `sample_rate` itself is within that ceiling.
**Usage:**
```python
upscale_factor = compute_upscale_factor(44100, bitrate, 1400)
```

### `upscale(input_file_path, output_file_path, source_format, target_format='flac', max_iterations=300, threshold_value=0.6, target_bitrate_kbps=1411, toggle_normalize=True, toggle_autoscale=True, toggle_adaptive_filter=True) -> None`
**File:** fat_llama/audio_fattener/feed.py:703
**Kind:** function
**Description:** Top-level entry point. Validates `target_bitrate_kbps` against the target format's range, reads the source (`read_audio`), derives `upscale_factor` (`compute_upscale_factor`), runs `upscale_channels` (interpolation + IST), optionally autoscales each channel back to its original peak, optionally normalizes, optionally applies `lms_filter` adaptive filtering, then unconditionally applies `apply_original_nyquist_cutoff` per channel before writing via `write_audio` (with `toggle_normalize` wired through, and `audio.max_possible_amplitude` as the reference amplitude for the `normalize=False` path).
**Parameters:**
- `input_file_path` (`str`): path to the input audio file.
- `output_file_path` (`str`): path to the output file.
- `source_format` (`str`): `'mp3'`, `'wav'`, `'ogg'`, or `'flac'`.
- `target_format` (`str`): `'flac'` (default) or `'wav'`.
- `max_iterations` (`int`): IST iteration count. Default `300`.
- `threshold_value` (`float`): IST threshold. Default `0.6`.
- `target_bitrate_kbps` (`int`): drives `upscale_factor`; must be in `(800, 1411)` for flac or `(800, 6444)` for wav. Default `1411`.
- `toggle_normalize` (`bool`): controls both the in-pipeline normalize stage and `write_audio`'s own final scaling. Default `True`.
- `toggle_autoscale` (`bool`): rescale each channel to its original peak after IST. Default `True`.
- `toggle_adaptive_filter` (`bool`): apply `lms_filter` after normalization. Default `True`.
**Returns:** `None` — writes the upscaled file to `output_file_path` as a side effect.
**Usage:**
```python
from fat_llama.audio_fattener.feed import upscale

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

## fat_llama/tests/test_feed.py

### `_cuda_gpu_available() -> bool`
**File:** fat_llama/tests/test_feed.py:21
**Kind:** function
**Description:** Checks for a functional CUDA-capable GPU via `cp.cuda.runtime.getDeviceCount() > 0`, returning `False` on any exception. Used to skip GPU-dependent tests cleanly on CPU-only CI runners rather than crashing, per the project's CUDA-only-with-no-fallback design.
**Returns:** `bool`.
**Usage:**
```python
GPU_AVAILABLE = _cuda_gpu_available()
```

### `class TestAudioFattener(unittest.TestCase)`
**File:** fat_llama/tests/test_feed.py:44
**Kind:** class
**Description:** The project's test suite for `fat_llama.audio_fattener.feed`. `setUp`/`tearDown` create and remove a synthetic 1-second 440 Hz sine-wave MP3 fixture. Non-GPU tests (`test_read_audio`, `test_write_audio`, `test_write_audio_normalize_false_preserves_relative_level`, `test_write_audio_wav_uses_64bit_float_and_is_lossless`, `test_compute_upscale_factor_bounds_realistic_sample_rate`, `test_lms_block_ranges_partitions_range_exactly`) run unconditionally; everything else is decorated `@requires_gpu` and exercises `lms_filter`, `iterative_soft_thresholding`/`_ist_chain`/`initialize_ist`, `new_interpolation_algorithm`, `apply_original_nyquist_cutoff`, and end-to-end `upscale()` behavior (Nyquist cutoff, adaptive filter wiring, `toggle_normalize`, `target_bitrate_kbps`-driven factor bounds). Cycle 5 (fftw-optimization port) added `test_initialize_ist_threshold_is_peak_relative`, `test_iterative_soft_thresholding_excludes_dc_bin`, `test_iterative_soft_thresholding_converges_before_max_iter`, and `test_iterative_soft_thresholding_block_processing_shape_and_finite`, and adjusted `test_ist_no_static_floor_in_quiet_segment`'s measurement window to start one `IST_BLOCK_SIZE` past the loud/quiet transition (a documented, bounded one-block-width transition-edge artifact, not a reintroduction of the persistent whole-track floor the test guards against). Cycle 6 (fixing a real regression cycle 5 introduced) added `test_uncapped_ist_inflates_combined_channel_peak`, `test_cap_ist_changes_to_baseline_peak_bounds_peak_without_zeroing_detail`, `test_cap_ist_changes_to_baseline_peak_is_noop_when_not_needed`, and `test_upscale_channels_combined_peak_never_exceeds_interpolation_baseline`. Cycle 7 (frequency-selective capping rewrite) updated `test_cap_ist_changes_to_baseline_peak_bounds_peak_without_zeroing_detail` to assert near-full (>90%) survival of the quiet 8000Hz component's own magnitude (previously just checked a uniform-scalar relationship) and added `test_cap_ist_changes_to_baseline_peak_safety_net_matches_uniform_cap` (exercises the safety-net fallback via a pathological `dominant_band_ratio` and confirms it still meets cycle 6's own bound). No test currently asserts that the *upscaled* output content resembles the *source* content beyond dominant-frequency checks (no decimate-and-correlate coherence test, unlike the fftw sibling package's test suite; also flagged: no stereo-channel end-to-end coverage, no non-mp3 source-format coverage).
**Usage:**
```python
python -m unittest fat_llama.tests.test_feed
```

## analysis.py

Standalone (not imported by `fat_llama`) spectrogram/waveform comparison script; imports `cupy` for its GPU cross-correlation/FFT steps.

### `read_mp3(file_path) -> (np.ndarray, int)`
**File:** analysis.py:8
**Kind:** function
**Description:** Reads an MP3 via `pydub`, converts to mono by averaging channels if stereo.
**Parameters:**
- `file_path` (`str`): path to the MP3 file.
**Returns:** `(data, frame_rate)`.
**Usage:**
```python
mp3, sample_rate_mp3 = read_mp3('input_test.mp3')
```

### `read_flac(file_path) -> (np.ndarray, int)`
**File:** analysis.py:16
**Kind:** function
**Description:** Reads a FLAC via `soundfile`, converts to mono by averaging channels if stereo.
**Parameters:**
- `file_path` (`str`): path to the FLAC file.
**Returns:** `(data, sample_rate)`.
**Usage:**
```python
flac, sample_rate_flac = read_flac('output_test.flac')
```

### `normalize(signal) -> np.ndarray`
**File:** analysis.py:22
**Kind:** function
**Description:** Peak-normalizes a signal to `[-1, 1]`.
**Parameters:**
- `signal` (`np.ndarray`): input signal.
**Returns:** `np.ndarray`.
**Usage:**
```python
normalized = normalize(mp3)
```

### `compare_signals(mp3, flac, sample_rate) -> None`
**File:** analysis.py:25
**Kind:** function
**Description:** Normalizes and length-matches `mp3`/`flac`, then plots (via `matplotlib`) a waveform comparison, difference signal, prints MSE, plots spectrograms (via `scipy.signal.spectrogram`), a GPU cross-correlation (`cp.correlate`), and a GPU FFT frequency-domain comparison (`cp.fft.fft`).
**Parameters:**
- `mp3` (`np.ndarray`): decoded MP3 samples.
- `flac` (`np.ndarray`): decoded FLAC samples.
- `sample_rate` (`int`): shared sample rate.
**Returns:** `None` (side effect: displays plots, prints stats).
**Usage:**
```python
compare_signals(mp3, flac, sample_rate)  # illustrative
```

## example.py

Top-level script (no functions/classes) — calls `fat_llama.audio_fattener.feed.upscale` with `input_file_path='input_test.mp3'`, `output_file_path='output_test.flac'`, `source_format='mp3'`, `target_format='flac'`, `max_iterations=300`, `threshold_value=0.6`, `target_bitrate_kbps=1400`, and all three toggles `True`.

## fat_llama/__init__.py, fat_llama/audio_fattener/__init__.py, fat_llama/tests/__init__.py

Empty package marker files — no functions/classes.

## Other tracked files (not factblocked — non-source)

`.gitignore`, `.mcp.json`, `CHANGELOG.md`, `LICENSE`, `Manifest.in`, `README.md`, `requirements.txt`, `setup.py` (current `version='2.0.0'`), `input_test.mp3`, `input_test.flac`, `output_test.flac`, `docs/images/logo.jpg`, `docs/images/spectrogram_comparison.png`, `docs/images/theory.png`.
