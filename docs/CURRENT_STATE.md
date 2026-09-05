# fat_llama — Current State

Snapshot produced by the `review-current-state` skill. Scope: whole repository (default). This file is regenerated on demand — do not hand-edit it.

## File tree

```
fat_llama/
├── .github/
│   └── workflows/
│       ├── deploy.yml
│       └── tests.yml
├── docs/
│   ├── CURRENT_STATE.md
│   └── images/
│       └── spectrogram_comparison.png
├── fat_llama/
│   ├── __init__.py
│   ├── audio_fattener/
│   │   ├── __init__.py
│   │   └── feed.py
│   └── tests/
│       ├── __init__.py
│       └── test_feed.py
├── analysis.py
├── example.py
├── LICENSE
├── Manifest.in
├── README.md
├── requirements.txt
├── setup.py
├── input_test.mp3
├── input_test.flac
└── output_test.flac
```

## analysis.py

Standalone comparison/analysis script (not part of the installed `fat_llama` package) — loads an MP3/FLAC pair and produces waveform, difference, MSE, spectrogram, cross-correlation, and frequency-domain comparisons. Depends on `cupy` (GPU) unconditionally at import time.

### `read_mp3(file_path) -> Tuple[np.ndarray, int]`
**File:** analysis.py:8
**Kind:** function
**Description:** Loads an MP3 file via `pydub.AudioSegment`, converts stereo to mono by averaging channels, and returns the sample data alongside the frame rate. No docstring; inferred from body.
**Parameters:**
- `file_path` (`str`): path to the MP3 file to read.
**Returns:** `(data, frame_rate)` — mono sample array and the file's frame rate (Hz).
**Usage:**
```python
data, sample_rate = read_mp3('input_test.mp3')  # illustrative
```

### `read_flac(file_path) -> Tuple[np.ndarray, int]`
**File:** analysis.py:16
**Kind:** function
**Description:** Loads a FLAC file via `soundfile.read`, averaging stereo channels down to mono. No docstring; inferred from body.
**Parameters:**
- `file_path` (`str`): path to the FLAC file to read.
**Returns:** `(data, sample_rate)` — mono sample array and sample rate (Hz).
**Usage:**
```python
data, sample_rate = read_flac('output_test.flac')  # illustrative
```

### `normalize(signal) -> np.ndarray`
**File:** analysis.py:22
**Kind:** function
**Description:** Scales a signal by its peak absolute amplitude so the result falls within [-1, 1]. No docstring; inferred from body.
**Parameters:**
- `signal` (`np.ndarray`): the signal to normalize.
**Returns:** `np.ndarray` — the peak-normalized signal.
**Usage:**
```python
normed = normalize(mp3_samples)  # illustrative
```

### `compare_signals(mp3, flac, sample_rate) -> None`
**File:** analysis.py:25
**Kind:** function
**Description:** Normalizes and length-aligns two signals, then renders/prints a battery of comparisons: waveform plot, difference-signal plot, mean squared error, spectrograms (via `scipy.signal.spectrogram`), a GPU cross-correlation (`cupy`), and a GPU FFT-based frequency-domain comparison. Purely diagnostic — produces plots via `matplotlib.pyplot.show()` and prints to stdout; returns nothing. No docstring; inferred from body.
**Parameters:**
- `mp3` (`np.ndarray`): MP3-decoded signal.
- `flac` (`np.ndarray`): FLAC-decoded signal.
- `sample_rate` (`int`): sample rate shared by both signals (post-resampling).
**Returns:** `None` — side effects only (plots, stdout).
**Usage:**
```python
# From analysis.py's own __main__ block:
mp3, sample_rate_mp3 = read_mp3('input_test.mp3')
flac, sample_rate_flac = read_flac('output_test.flac')
compare_signals(mp3, flac, sample_rate_mp3)
```

## example.py

Minimal usage example for the package's public `upscale` entry point; also wired as the `example` console-script entry point in `setup.py` (though `setup.py` points it at `example:main`, and this module defines no `main` function — see factblock_stale note below).

No functions/classes defined — the module body is a single top-level call:
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

## setup.py

Package metadata for `fat_llama` (PyPI distribution). No functions/classes — a single `setuptools.setup(...)` call. Current `version`: `1.1.0`. `install_requires`: `numpy`, `cupy-cuda13x`, `pydub`, `soundfile`, `mutagen`, `scipy`. Declares console-script entry point `example=example:main`.

## fat_llama/__init__.py

Empty — no exports.

## fat_llama/audio_fattener/__init__.py

Empty — no exports.

## fat_llama/audio_fattener/feed.py

The package's core module: reads/writes audio files and implements the GPU-accelerated (CuPy) upscaling pipeline (interpolation, iterative soft-thresholding "IST", LMS adaptive filtering).

### `read_audio(file_path, audio_format) -> Tuple[int, np.ndarray, Optional[float], AudioSegment]`
**File:** fat_llama/audio_fattener/feed.py:18
**Kind:** function
**Description:** Reads an audio file via `pydub.AudioSegment.from_file` (with `-drc_scale 0` passed to ffmpeg), converts samples to a `float64` NumPy array, and looks up the file's bitrate via `mutagen` for mp3/flac/ogg/wav (falling back to a byte-rate estimate for other formats). Reshapes samples to `(-1, 2)` for stereo input. Raises `FileNotFoundError` if the input path doesn't exist. Parameter was renamed from `format` to `audio_format` (cycle 1 fix) to stop shadowing the `format()` builtin.
**Parameters:**
- `file_path` (`str`): path to the input audio file.
- `audio_format` (`str`): input format (`'mp3'`, `'flac'`, `'ogg'`, `'wav'`, or other ffmpeg-supported format).
**Returns:** `(sample_rate, samples, bitrate, audio)` — sample rate in Hz, sample data (`float64`, shape `(n,)` or `(n, 2)`), bitrate in bits/sec (`None` if undeterminable), and the underlying `AudioSegment`.
**Usage:**
```python
# From fat_llama/tests/test_feed.py:
sample_rate, samples, bitrate, audio = read_audio(self.test_mp3_file, audio_format='mp3')
```

### `write_audio(file_path, sample_rate, data, audio_format) -> None`
**File:** fat_llama/audio_fattener/feed.py:72
**Kind:** function
**Description:** Writes sample data to disk via `soundfile.write`. As of the cycle 1 fix, peak-normalizes `data` to `[-1, 1]` (leaving near-silent input as-is to avoid a divide-by-zero) and clips as a numerical safety net before writing — `soundfile` silently clamps out-of-range float input when writing an integer PCM subtype instead of raising, and callers (e.g. `read_audio`'s own output, or the pipeline's intermediate stages) hand back data on the raw PCM/processing scale, not `[-1, 1]`; without this normalization nearly every sample was clamped to full scale. Supports `'flac'` and `'wav'` only (both written as 24-bit PCM); raises `ValueError` for any other target format. Parameter renamed from `format` to `audio_format` (cycle 1 fix).
**Parameters:**
- `file_path` (`str`): output file path.
- `sample_rate` (`int`): sample rate in Hz.
- `data` (`np.ndarray`): audio sample data (any scale — normalized internally before writing).
- `audio_format` (`str`): output format, `'flac'` or `'wav'`.
**Returns:** `None`.
**Usage:**
```python
# From fat_llama/tests/test_feed.py:
write_audio(output_file, sample_rate, samples, audio_format='flac')
```

### `new_interpolation_algorithm(data, upscale_factor) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:109
**Kind:** function
**Description:** Upsamples a 1-D real signal via FFT-domain zero-padding (bandlimited/sinc interpolation): `cp.fft.rfft` the input, zero-pad the spectrum with additional (all-zero) high-frequency bins, `cp.fft.irfft` back to a longer time-domain signal, then rescale by `upscale_factor` to correct for `irfft`'s output-length normalization. As of the **cycle 3 fix**, this replaces the prior zero-order-hold duplication (each sample repeated `upscale_factor` times) that cycles 1-2 had investigated and deliberately left unfixed — measured to inject strong mirrored spectral images at multiples of the original sample rate rather than genuine added detail, and to leave `iterative_soft_thresholding` little headroom to add real content. Post-fix: energy above the original Nyquist frequency measured at ~1e-8 relative magnitude (FFT round-off) instead of dominating the extended band; entirely `cp.fft` (CuPy/CUDA), no scipy/numpy CPU dependency; also ~1000x faster than the old per-sample Python loop as a side effect. `upscale_factor == 1` short-circuits to a copy.
**Parameters:**
- `data` (`cp.ndarray`): input audio data (single channel).
- `upscale_factor` (`int`): the factor by which to upscale the audio data.
**Returns:** `cp.ndarray` — band-limited upscaled data, length `len(data) * upscale_factor`.
**Usage:**
```python
# From fat_llama/tests/test_feed.py:
expanded = new_interpolation_algorithm(tone, upscale_factor)
```

### `initialize_ist(data, threshold) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:170
**Kind:** function
**Description:** Zeroes out samples whose absolute value is at or below `threshold`, keeping only samples above it — the initialization step for iterative soft-thresholding.
**Parameters:**
- `data` (`cp.ndarray`): input audio data.
- `threshold` (`float`): magnitude threshold below which samples are zeroed.
**Returns:** `cp.ndarray` — thresholded data, same shape as input.
**Usage:**
```python
thresholded = initialize_ist(data, 0.6)  # illustrative
```

### `iterative_soft_thresholding(data, max_iter, threshold) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:186
**Kind:** function
**Description:** Runs `max_iter` rounds of: FFT the thresholded signal, zero out FFT bins at/below `threshold`, inverse-FFT back to the time domain, then add a sinusoidal harmonic term each iteration — used to reconstruct missing high-frequency content after upscaling. The harmonic term's *total* amplitude across all iterations is `0.1 * peak / max_iter` (cycle 2 bounded it to `0.1/max_iter` so it doesn't grow with `max_iter`; **cycle 3** additionally scaled that total by `data`'s own peak amplitude, since `data` is raw-PCM-scale here and a fixed absolute `0.1` was measured to be an ~1e-5 relative contribution — unmeasurable after the pipeline's later autoscale/normalize steps, which only apply a global gain). **Known issue (investigated, not fixed as of cycle 3):** `threshold` is applied as an absolute cutoff to both raw-PCM-scale time-domain samples and raw FFT-bin magnitudes, but real audio's FFT-bin magnitudes are order ~1e4-1e5 — many orders of magnitude above the conventional default `threshold=0.6` — so the "keep significant frequencies, discard noise" masking barely triggers at real audio scale, leaving IST to mostly perform a near-lossless FFT/IFFT round trip beyond the (now-fixed) harmonic term. Flagged as a strong next-cycle candidate (convert to a peak-relative fraction) rather than folded into cycle 3.
**Parameters:**
- `data` (`cp.ndarray`): input (already interpolated) audio data.
- `max_iter` (`int`): number of IST iterations to run.
- `threshold` (`float`): magnitude threshold applied in both time and frequency domain.
**Returns:** `cp.ndarray` — the IST-processed signal after `max_iter` iterations.
**Usage:**
```python
ist_result = iterative_soft_thresholding(expanded_channel, max_iter=300, threshold=0.6)  # illustrative
```

### `lms_filter(signal, desired, mu=0.001, num_taps=32, delay=1, return_weights=False) -> cp.ndarray | Tuple[cp.ndarray, cp.ndarray]`
**File:** fat_llama/audio_fattener/feed.py:263
**Kind:** function
**Description:** Applies a sample-by-sample LMS adaptive filter: for each sample past warm-up, computes a filtered output from `num_taps` input samples lagged `delay` behind the sample being predicted, compares it to `desired`, and adjusts filter weights by the LMS update rule (weights clipped to ±1e10). Cycle 1 fixed a warm-up-dropout bug (tap-weight vector `w` initialized to `[1, 0, ..., 0]` instead of all-zero, `filtered_signal[:start]` seeded from `signal` directly). **Cycle 3 finding and fix:** `upscale()` always calls this as `lms_filter(channel, channel)` (signal and desired are the *same* array); with the old `delay=0` behavior, tap 0 of the input vector was `signal[i]` itself, so combined with the cycle 1 identity init, `y == desired[i]` exactly on every sample — the error term was identically zero forever, `w` never moved, and the filter was a bit-exact no-op consuming ~18-19 of the pipeline's ~20 minute runtime for zero effect. The new `delay` parameter (default `1`, an Adaptive Line Enhancer / ALE pattern) draws the predictor's taps from `signal` lagged by `delay` samples instead of `signal[i]` itself, making even the self-referential case a genuine (if small) estimation problem — `w` now measurably evolves — while the near-identity init still avoids reintroducing the warm-up dropout, since real audio is highly autocorrelated at lag 1.
**Parameters:**
- `signal` (`cp.ndarray`): input signal to filter.
- `desired` (`cp.ndarray`): desired/reference signal used to compute the error term.
- `mu` (`float`): LMS step size (default `0.001`).
- `num_taps` (`int`): number of filter taps (default `32`).
- `delay` (`int`): ALE decorrelation lag in samples between the predictor's taps and the predicted sample (default `1`); must be `>= 1` for the self-referential `lms_filter(x, x)` case to be non-degenerate — `0` reproduces the cycle-3-fixed no-op behavior.
- `return_weights` (`bool`): if `True`, return `(filtered_signal, w)` — the final tap weights alongside the filtered signal — instead of just `filtered_signal` (default `False`, preserves the original call signature for existing callers).
**Returns:** `cp.ndarray` — the filtered signal, same length as `signal` (or `(filtered_signal, w)` if `return_weights=True`).
**Usage:**
```python
# From fat_llama/tests/test_feed.py:
filtered, w_final = lms_filter(
    signal, signal, mu=0.001, num_taps=num_taps, return_weights=True
)
```

### `upscale_channels(channels, upscale_factor, max_iter, threshold) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:358
**Kind:** function
**Description:** Per-channel pipeline stage: for each channel in `channels`, runs `new_interpolation_algorithm` then `iterative_soft_thresholding`, adds the IST result back onto the interpolated signal, and stacks all processed channels back together.
**Parameters:**
- `channels` (`cp.ndarray`): input audio, shape `(n_samples, n_channels)`.
- `upscale_factor` (`int`): interpolation factor passed through to `new_interpolation_algorithm`.
- `max_iter` (`int`): IST iteration count passed through to `iterative_soft_thresholding`.
- `threshold` (`float`): IST threshold passed through to `iterative_soft_thresholding`.
**Returns:** `cp.ndarray` — upscaled/processed channels, shape `(n_samples * upscale_factor, n_channels)`.
**Usage:**
```python
processed = upscale_channels(channels, upscale_factor=4, max_iter=300, threshold=0.6)  # illustrative
```

### `normalize_signal(signal) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:390
**Kind:** function
**Description:** Peak-normalizes a signal to the range [-1, 1] (CuPy equivalent of `analysis.py`'s `normalize`).
**Parameters:**
- `signal` (`cp.ndarray`): input signal.
**Returns:** `cp.ndarray` — peak-normalized signal.
**Usage:**
```python
normed = normalize_signal(channel)  # illustrative
```

### `apply_original_nyquist_cutoff(signal, original_sample_rate, new_sample_rate) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:403
**Kind:** function
**Description:** Added in **cycle 4** as an unconditional final safety stage. Zeroes all spectral content above the *original* source's Nyquist frequency (`original_sample_rate / 2`) via `cp.fft.rfft` → mask bins by `cp.fft.rfftfreq` → `cp.fft.irfft`, entirely CuPy/CUDA. Per `.claude/rules/project-mission.md`'s "no content above the original Nyquist frequency" hard constraint, fat_llama upscales precision/headroom within the original recording's real bandwidth and does not do bandwidth extension — the band an upsample opens up above the original Nyquist must be actively guaranteed silent, not left as an emergent property of whichever earlier stages (interpolation, IST's harmonic term, autoscale, normalize, LMS) happen to behave well. As of cycle 3's bandlimited interpolation, that band already measures ~-136dB for a real run, so this stage is close to a no-op today — its purpose is to make that a structural guarantee that survives future changes to earlier stages, not to fix a currently-observed defect.
**Parameters:**
- `signal` (`cp.ndarray`): the fully processed signal (single channel), sampled at `new_sample_rate`.
- `original_sample_rate` (`int`): the original source's sample rate before upscaling; the cutoff is `original_sample_rate / 2`.
- `new_sample_rate` (`int` or `float`): the sample rate `signal` is actually sampled at (`original_sample_rate * upscale_factor`).
**Returns:** `cp.ndarray` — `signal` with all content above `original_sample_rate / 2` removed, same length.
**Usage:**
```python
# From fat_llama/tests/test_feed.py:
cutoff_signal = apply_original_nyquist_cutoff(
    signal, original_sample_rate, new_sample_rate
)
```

### `upscale(input_file_path, output_file_path, source_format, target_format='flac', max_iterations=300, threshold_value=0.6, target_bitrate_kbps=1411, toggle_normalize=True, toggle_autoscale=True, toggle_adaptive_filter=True) -> None`
**File:** fat_llama/audio_fattener/feed.py:459
**Kind:** function
**Description:** The package's main public entry point. Validates `target_bitrate_kbps` against per-format ranges (`flac`: 800–1411 kbps, `wav`: 800–6444 kbps), reads the input via `read_audio`, computes an integer upscale factor from the ratio of target to original bitrate (defaulting to `4` if the original bitrate is unknown), runs the channel pipeline (`upscale_channels`), optionally autoscales each channel back to its original peak amplitude, optionally normalizes, optionally applies `lms_filter` per channel (using each channel as its own `desired` signal), applies `apply_original_nyquist_cutoff` per channel unconditionally (cycle 4, no toggle — always the final processing step), and writes the result via `write_audio` at `sample_rate * upscale_factor`.
**Parameters:**
- `input_file_path` (`str`): path to the input audio file.
- `output_file_path` (`str`): path to write the processed output.
- `source_format` (`str`): input format passed to `read_audio` (e.g. `'mp3'`, `'wav'`, `'ogg'`, `'flac'`).
- `target_format` (`str`): output format, `'flac'` (default) or `'wav'`.
- `max_iterations` (`int`): IST iteration count (default `300`).
- `threshold_value` (`float`): IST/LMS threshold (default `0.6`).
- `target_bitrate_kbps` (`int`): used only to derive `upscale_factor` relative to the source file's own bitrate (default `1411`); must itself fall within the valid range for `target_format` (a sanity bound on this parameter, not a promise about the output's real bitrate). As clarified in the cycle 2 fix: the output is always uncompressed PCM at an upsampled rate, so its real bitrate is, by design, substantially higher than `target_bitrate_kbps` once `upscale_factor > 1` (measured: target 1400 kbps vs. ~7822 kbps effective output).
- `toggle_normalize` (`bool`): whether to peak-normalize the output (default `True`).
- `toggle_autoscale` (`bool`): whether to rescale output amplitude to match the original (default `True`).
- `toggle_adaptive_filter` (`bool`): whether to apply `lms_filter` (default `True`).
**Returns:** `None` — writes the output file as a side effect.
**Usage:**
```python
# From example.py:
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

## fat_llama/tests/__init__.py

Empty — no exports.

## fat_llama/tests/test_feed.py

`unittest`-based test module for `feed.py`, covering `read_audio`, `write_audio`, `lms_filter`'s warm-up behavior (cycle 1), `iterative_soft_thresholding`'s bounded harmonic injection and `upscale`'s `target_bitrate_kbps` contract (cycle 2), `lms_filter`'s genuine self-referential adaptation, `iterative_soft_thresholding`'s peak-relative harmonic scaling, and `new_interpolation_algorithm`'s bandlimited-ness (cycle 3), and (cycle 4) `apply_original_nyquist_cutoff`'s above-Nyquist suppression, both in isolation and wired into `upscale()`.

### `TestAudioFattener`
**File:** fat_llama/tests/test_feed.py:14
**Kind:** class
**Description:** `unittest.TestCase` subclass exercising `read_audio`/`write_audio`. `setUp` generates a 1-second 440 Hz sine wave as `test_input.mp3` via `pydub.generators.Sine`; `tearDown` removes the generated MP3 and any leftover `output_processed.flac`.
**Usage:**
```python
python -m pytest fat_llama/tests/test_feed.py -v
```

#### `TestAudioFattener.setUp(self) -> None`
**File:** fat_llama/tests/test_feed.py:16
**Kind:** method
**Description:** Creates a fresh test MP3 (`test_input.mp3`) before each test via `create_test_mp3`. Inferred from body (no docstring).
**Returns:** `None`.

#### `TestAudioFattener.tearDown(self) -> None`
**File:** fat_llama/tests/test_feed.py:21
**Kind:** method
**Description:** Deletes `test_input.mp3` and `output_processed.flac` if present, after each test. Inferred from body (no docstring).
**Returns:** `None`.

#### `TestAudioFattener.create_test_mp3(self, filename) -> None`
**File:** fat_llama/tests/test_feed.py:28
**Kind:** method
**Description:** Synthesizes a 1-second 440 Hz sine wave with `pydub.generators.Sine` and exports it as an MP3 to `filename`, then explicitly closes the file handle `export()` returns (cycle 2 fix — `pydub` does not close it, previously leaking an open file descriptor per test).
**Parameters:**
- `filename` (`str`): output path for the generated test MP3.
**Returns:** `None`.

#### `TestAudioFattener.test_read_audio(self) -> None`
**File:** fat_llama/tests/test_feed.py:38
**Kind:** method
**Description:** Asserts `read_audio` on the generated sine-wave MP3 returns a 44100 Hz sample rate and 44100 samples (1 second); that a mono source comes back as a flat 1-D array (`audio.channels == 1`, `samples.ndim == 1`) rather than reshaped to `(N, 2)`; that duration is 1000 ms; that the mp3-reported bitrate falls within the encoder's default CBR band (32000–320000, not a single hard-coded value); and that the samples actually carry signal — non-silent, all-finite, and (via windowed FFT) a dominant spectral peak within 5 Hz of 440 Hz.
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_write_audio(self) -> None`
**File:** fat_llama/tests/test_feed.py:68
**Kind:** method
**Description:** Reads the generated sine-wave MP3, writes it out as FLAC via `write_audio`, then asserts real coherence of the round-trip: output file exists; `soundfile.info` reports the same sample rate/channel count and a duration matching the ~1 s input within 0.05 s tolerance; the re-read written data is non-silent and all-finite; the written waveform correlates >0.999 with the peak-normalized input (guards against a test that would pass for any arbitrary non-silent signal, not necessarily the true written waveform); the dominant spectral peak is still within 5 Hz of 440 Hz; and fewer than 5% of written samples sit at full-scale clipping (`> 0.999`), guarding against `write_audio` handing raw (non-normalized, PCM-scale) samples straight to an integer `soundfile` subtype. As of the cycle 1 fix to `write_audio`, this test passes. Cleans up `test_output.flac` in a `finally` block.
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_lms_filter_no_extended_warmup_dropout(self) -> None`
**File:** fat_llama/tests/test_feed.py:141
**Kind:** method
**Description:** Added in cycle 1 as a regression test for `lms_filter`'s warm-up fix. Builds a 50ms two-tone synthetic signal (300 Hz + 900 Hz), runs `lms_filter(signal, signal, mu=0.001, num_taps=32)`, and checks the RMS of the filtered output over the 50 samples immediately following the first `num_taps` against the RMS of the input signal over that same window — asserting the ratio exceeds 0.5. Guards against `lms_filter` ramping up from a zero-initialized state instead of tracking the signal from (near) the first sample; production symptom before the fix was a ~200ms, -82 dBFS dropout at the head of upscaled audio with no corresponding silence in the source.
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_ist_harmonic_injection_bounded_across_iterations(self) -> None`
**File:** fat_llama/tests/test_feed.py:179
**Kind:** method
**Description:** Added in cycle 2 as a regression test for `iterative_soft_thresholding`'s bounded-harmonic fix. Builds a synthetic two-tone signal (300 Hz + 700 Hz, n=2000), runs `iterative_soft_thresholding` at `max_iter=5` and again at `max_iter=150`, and asserts the 150-iteration run's peak magnitude is less than 2x the 5-iteration run's — guarding against the harmonic-injection term accumulating roughly linearly with `max_iter` instead of staying bounded (production symptom before the fix: a measured +6.12 dB broadband noise-floor rise).
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_lms_filter_self_referential_call_genuinely_adapts(self) -> None`
**File:** fat_llama/tests/test_feed.py:214
**Kind:** method
**Description:** Added in cycle 3 as a regression test for `lms_filter`'s decorrelation-delay fix. Runs `lms_filter(signal, signal, mu=0.001, num_taps=32, return_weights=True)` on a 200ms two-tone signal (300 Hz + 900 Hz) and asserts three things: the final tap weights differ from the `[1,0,...,0]` identity init (proves adaptation happened), the filtered output is not bit-identical to the input over the post-warm-up region (the direct symptom of the cycle 3 no-op bug), and the warm-up RMS ratio still exceeds 0.5 (proves the delay fix didn't reintroduce the cycle 1 dropout).
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_ist_harmonic_amplitude_scales_with_signal_peak(self) -> None`
**File:** fat_llama/tests/test_feed.py:282
**Kind:** method
**Description:** Added in cycle 3 as a regression test for `iterative_soft_thresholding`'s peak-relative harmonic scaling. Runs IST (with a negligible threshold so the FFT/IFFT round trip is near-identity and the harmonic term dominates the change) on the same waveform shape at two absolute scales (peak 1 vs. peak 10000) and asserts the *relative* added contribution (added magnitude / signal peak) stays comparable across that 10000x scale change — guarding against a fixed-absolute harmonic amplitude that would collapse to an unmeasurable relative contribution at real (raw-PCM-scale) audio amplitudes.
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_new_interpolation_algorithm_is_bandlimited(self) -> None`
**File:** fat_llama/tests/test_feed.py:341
**Kind:** method
**Description:** Added in cycle 3 as a regression test for `new_interpolation_algorithm`'s bandlimited-interpolation fix. Upsamples a synthetic 300 Hz tone (0.1s @ 44100 Hz) by `upscale_factor=7` and asserts: output length is `n * upscale_factor` and all-finite; peak spectral energy above the original Nyquist frequency is less than 1e-4x the below-Nyquist peak (guards against zero-order-hold imaging, which would put comparable energy at mirrored image frequencies); and the dominant below-Nyquist frequency is still within 5 Hz of the source's 300 Hz.
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_apply_original_nyquist_cutoff_removes_above_nyquist_content(self) -> None`
**File:** fat_llama/tests/test_feed.py:395
**Kind:** method
**Description:** Added in cycle 4 as a regression test for `apply_original_nyquist_cutoff`. Builds a synthetic 50ms signal with a 300 Hz in-band tone and a 30000 Hz tone above the original 22050 Hz Nyquist (simulating artifact energy a future upstream stage might reintroduce), sanity-checks the synthetic signal genuinely carries comparable energy in both bands before the cutoff, then asserts: the above-Nyquist peak drops to below `1e-6` of the pre-cutoff in-band peak; the in-band tone survives within 5% of its original amplitude; the output length and finiteness are preserved; and the dominant frequency is still the 300 Hz in-band tone.
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_upscale_no_content_above_original_nyquist_frequency(self) -> None`
**File:** fat_llama/tests/test_feed.py:477
**Kind:** method
**Description:** Added in cycle 4 to confirm `apply_original_nyquist_cutoff` is actually wired into `upscale()`'s public entry point, not just correct in isolation. Runs a full `upscale()` call at two different `target_bitrate_kbps` values (800 and 1400, giving two different `upscale_factor`s) with `max_iterations=2` and `toggle_adaptive_filter=False` to stay fast, using `target_format='wav'` (this source's bitrate/target combination can drive an `upscale_factor` that pushes FLAC's output sample rate past libsndfile's ~655350 Hz format ceiling — a pre-existing, unrelated limitation, not a defect in this fix). For each run, asserts the peak spectral energy above the original 22050 Hz Nyquist is below `1e-4` of the in-band peak (skipped if `upscale_factor` came out to 1, i.e. no upsampling headroom to check).
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_target_bitrate_kbps_drives_upscale_factor_not_output_bitrate(self) -> None`
**File:** fat_llama/tests/test_feed.py:542
**Kind:** method
**Description:** Added in cycle 2 to document `target_bitrate_kbps`'s intended contract; strengthened by `audio-quality-checker` with real audio-content assertions. Runs a full `upscale()` call (`max_iterations=2`, `toggle_adaptive_filter=False` to stay fast, `target_bitrate_kbps=800`) and asserts: the output sample rate matches `source_sample_rate * round(target_bitrate_kbps * 1000 / source_bitrate)`; duration is preserved (~1s, since both sample count and rate scale by `upscale_factor`); the output is mono, all-finite, non-silent (RMS > 1e-3), has fewer than 5% full-scale-clipped samples, and its dominant spectral peak is still within 5 Hz of the source's 440 Hz tone; and the real output bitrate (computed from file size / duration) is more than 2x `target_bitrate_kbps` — confirming the parameter only derives `upscale_factor` and does not constrain the real (uncompressed PCM) output bitrate, which is structurally much higher by design.
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

## Notes

- `factblock_stale` / inconsistencies observed while reviewing (not fixed by this skill — report only):
  - `setup.py`'s `console_scripts` entry point (`example=example:main`) references `example.main`, but `example.py` defines no `main` function — it only has top-level script code. This entry point would fail if invoked as an installed console script. Flagged by code-tester-reviewer and generate-code in iterate-fat-llama cycle 1; left unfixed both times as out of scope (setup.py sits outside `fat_llama/**`).
  - `setup.py`'s `version` (`1.1.0`) does not yet reflect this iterate-fat-llama run's in-progress cycles (branch `iterate-fat-llama/20260905-030218`, off `v-1.4.0-latest`) — the version bump happens in that skill's own Step 6, after cycling completes.
  - `analysis.py` imports `cupy` unconditionally at module load, same as `feed.py` — both modules require a CUDA-capable GPU/environment to import successfully at all, not just to run GPU-specific code paths.
  - `new_interpolation_algorithm`'s zero-order-hold interpolation issue (investigated and left unfixed in cycles 1-2) was **fixed in cycle 3** — replaced with FFT-domain bandlimited interpolation (see its factblock above).
  - `iterative_soft_thresholding`'s `threshold` parameter is an absolute cutoff applied to raw-PCM-scale/raw-FFT-magnitude data, which real audio's actual scale (~1e4-1e5) dwarfs — masking barely triggers at the conventional default `threshold=0.6`, so IST's "keep significant frequencies" mechanism is mostly a near-lossless FFT/IFFT round trip beyond the harmonic term (see its factblock above). Investigated and documented in cycle 3, deliberately not fixed — flagged as a strong next-cycle candidate (convert to a peak-relative fraction).
  - `iterative_soft_thresholding`'s harmonic-reconstruction term (`cp.sin(cp.linspace(0, 2*pi, len(data_thres)))`) spans exactly one sine cycle across the *entire* buffer regardless of sample rate — for a real ~15s buffer that's a ~0.066 Hz subsonic oscillation, not audible-band content (found in cycle 4 by `audio-quality-checker`: 4.87% of peak amplitude concentrated there, sub-20Hz energy at 3.32% of total output energy vs. 0.018% in the reference). An initial cycle 4 attempt to fix this by extrapolating real content into the band above the original Nyquist was **redirected by the user mid-run** — the project's design goal changed: that band must be kept silent (see the new project-mission.md hard constraint and `apply_original_nyquist_cutoff` above), not filled with reconstructed detail. The harmonic term's subsonic-frequency defect itself (independent of the above-Nyquist question — it also isn't landing anywhere useful *below* the original Nyquist either) remains open as of cycle 4's actual shipped fix (`apply_original_nyquist_cutoff`) and is a candidate for a future cycle if IST is meant to add detail within the original band.
  - The full baseline `upscale()` pipeline (max_iterations=300, real ~15s input_test.mp3, both channels) took roughly 20-21 minutes end to end pre-cycle-3, ~91% of it in `lms_filter`'s per-sample Python loop; `new_interpolation_algorithm`'s cycle 3 rewrite made interpolation itself ~1000x faster (was the second-largest cost), but `lms_filter` is still an unvectorized per-sample Python loop and remains the dominant runtime cost.
  - No test exercises `upscale()` end-to-end with `toggle_adaptive_filter=True` or a stereo source — the only full-pipeline tests use `toggle_adaptive_filter=False` and/or mono. Flagged repeatedly (cycles 3-4) as a good next-cycle candidate; likely needs `lms_filter` vectorization to be practically testable given its per-sample loop.
  - The repo-root reference `input_test.flac` (used by `audio-quality-checker`'s spectral-deviation and — as of the coherence-methodology fix — coherence scoring) is itself a zero-order-hold-duplicated derivative of `input_test.mp3` (confirmed independently in cycles 2, 3, and 4), not an independent ground-truth master — flagged repeatedly as structurally unfixable by `generate-code` (the asset sits outside `fat_llama/**`), a ceiling on both scores' meaningfulness until a human replaces the reference asset with a genuine independent high-resolution master of the same source. Cycle 4's spectrogram comparison visibly shows this reference's own legacy zero-order-hold imaging (~44.1/88.2/132.3 kHz mirror bands) — the new pipeline output itself no longer has this artifact.
