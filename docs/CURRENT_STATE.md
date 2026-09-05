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
**Description:** Upsamples a 1-D signal by repeating (duplicating) each sample `upscale_factor` times — a zero-order-hold interpolation, not band-limited resampling.
**Parameters:**
- `data` (`cp.ndarray`): input audio data (single channel).
- `upscale_factor` (`int`): number of times each sample is duplicated.
**Returns:** `cp.ndarray` — expanded data of length `len(data) * upscale_factor`.
**Usage:**
```python
# From fat_llama/tests/test_feed.py (imported, not directly called in current tests):
from fat_llama.audio_fattener.feed import new_interpolation_algorithm
```

**Known issue (investigated, not fixed as of cycle 1):** this is zero-order-hold duplication with no anti-imaging lowpass. `audio-quality-checker` measured that it produces mirrored spectral images (near 44/88/132 kHz for a 7x upscale) rather than genuine added high-frequency detail, and the audible-band noise floor rises ~1.43x. `generate-code` investigated a bandlimited (FFT-based zero-padding) replacement in cycle 1 but deliberately left it unfixed — it would replace README.md's documented interpolation method wholesale (a design decision, not a bug fix) and could invalidate the committed `input_test.flac` reference used by the spectral-deviation test. Flagged as a candidate for an explicit next-cycle target.

### `initialize_ist(data, threshold) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:134
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
**File:** fat_llama/audio_fattener/feed.py:150
**Kind:** function
**Description:** Runs `max_iter` rounds of: FFT the thresholded signal, zero out FFT bins at/below `threshold`, inverse-FFT back to the time domain, then add a small sinusoidal harmonic term each iteration — used to reconstruct missing high-frequency content after upscaling. As of the cycle 2 fix, the harmonic term's amplitude is `0.1 / max_iter` (was a fixed `0.1` every iteration) so total injected harmonic energy stays constant regardless of iteration count — the fixed amplitude previously accumulated ~linearly with `max_iter` (measured max|output| ~0.88 at `max_iter=1` vs. ~30.8 at `max_iter=300`), causing a measurable broadband noise-floor rise in produced audio.
**Parameters:**
- `data` (`cp.ndarray`): input (already interpolated) audio data.
- `max_iter` (`int`): number of IST iterations to run.
- `threshold` (`float`): magnitude threshold applied in both time and frequency domain.
**Returns:** `cp.ndarray` — the IST-processed signal after `max_iter` iterations.
**Usage:**
```python
ist_result = iterative_soft_thresholding(expanded_channel, max_iter=300, threshold=0.6)  # illustrative
```

### `lms_filter(signal, desired, mu=0.001, num_taps=32) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:176
**Kind:** function
**Description:** Applies a sample-by-sample LMS adaptive filter: for each sample past the first `num_taps`, computes a filtered output from the last `num_taps` input samples, compares it to `desired`, and adjusts filter weights by the LMS update rule (weights clipped to ±1e10 for numerical stability). As of the cycle 1 fix, the tap-weight vector `w` is initialized to `[1, 0, ..., 0]` (identity pass-through on the most recent sample) instead of all-zero, and `filtered_signal[:num_taps]` is seeded with `signal[:num_taps]` directly instead of left at zero — previously the all-zero initialization made early output ~0 regardless of true input level, producing an audible silent ramp-up (measured as ~204ms at -82 dBFS in production use) at the start of every filtered channel; since `upscale()` always calls this with `signal == desired`, the identity start is also the filter's eventual convergence point for that self-referential case, so this removes the warm-up dropout without changing the LMS update rule itself.
**Parameters:**
- `signal` (`cp.ndarray`): input signal to filter.
- `desired` (`cp.ndarray`): desired/reference signal used to compute the error term.
- `mu` (`float`): LMS step size (default `0.001`).
- `num_taps` (`int`): number of filter taps (default `32`).
**Returns:** `cp.ndarray` — the filtered signal, same length as `signal`.
**Usage:**
```python
# From fat_llama/tests/test_feed.py:
filtered = lms_filter(signal, signal, mu=0.001, num_taps=num_taps)
```

### `upscale_channels(channels, upscale_factor, max_iter, threshold) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:228
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
**File:** fat_llama/audio_fattener/feed.py:260
**Kind:** function
**Description:** Peak-normalizes a signal to the range [-1, 1] (CuPy equivalent of `analysis.py`'s `normalize`).
**Parameters:**
- `signal` (`cp.ndarray`): input signal.
**Returns:** `cp.ndarray` — peak-normalized signal.
**Usage:**
```python
normed = normalize_signal(channel)  # illustrative
```

### `upscale(input_file_path, output_file_path, source_format, target_format='flac', max_iterations=300, threshold_value=0.6, target_bitrate_kbps=1411, toggle_normalize=True, toggle_autoscale=True, toggle_adaptive_filter=True) -> None`
**File:** fat_llama/audio_fattener/feed.py:273
**Kind:** function
**Description:** The package's main public entry point. Validates `target_bitrate_kbps` against per-format ranges (`flac`: 800–1411 kbps, `wav`: 800–6444 kbps), reads the input via `read_audio`, computes an integer upscale factor from the ratio of target to original bitrate (defaulting to `4` if the original bitrate is unknown), runs the channel pipeline (`upscale_channels`), optionally autoscales each channel back to its original peak amplitude, optionally normalizes, optionally applies `lms_filter` per channel (using each channel as its own `desired` signal), and writes the result via `write_audio` at `sample_rate * upscale_factor`.
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

`unittest`-based test module for `feed.py`, covering `read_audio`, `write_audio`, `lms_filter`'s warm-up behavior (cycle 1), and (cycle 2) `iterative_soft_thresholding`'s bounded harmonic injection and `upscale`'s `target_bitrate_kbps` contract. As of cycle 2, imports are all used (stdlib/third-party/local grouped); the prior unused `patch`/`MagicMock`/`new_interpolation_algorithm` imports were removed.

### `TestAudioFattener`
**File:** fat_llama/tests/test_feed.py:11
**Kind:** class
**Description:** `unittest.TestCase` subclass exercising `read_audio`/`write_audio`. `setUp` generates a 1-second 440 Hz sine wave as `test_input.mp3` via `pydub.generators.Sine`; `tearDown` removes the generated MP3 and any leftover `output_processed.flac`.
**Usage:**
```python
python -m pytest fat_llama/tests/test_feed.py -v
```

#### `TestAudioFattener.setUp(self) -> None`
**File:** fat_llama/tests/test_feed.py:13
**Kind:** method
**Description:** Creates a fresh test MP3 (`test_input.mp3`) before each test via `create_test_mp3`. Inferred from body (no docstring).
**Returns:** `None`.

#### `TestAudioFattener.tearDown(self) -> None`
**File:** fat_llama/tests/test_feed.py:20
**Kind:** method
**Description:** Deletes `test_input.mp3` and `output_processed.flac` if present, after each test. Inferred from body (no docstring).
**Returns:** `None`.

#### `TestAudioFattener.create_test_mp3(self, filename) -> None`
**File:** fat_llama/tests/test_feed.py:27
**Kind:** method
**Description:** Synthesizes a 1-second 440 Hz sine wave with `pydub.generators.Sine` and exports it as an MP3 to `filename`, then explicitly closes the file handle `export()` returns (cycle 2 fix — `pydub` does not close it, previously leaking an open file descriptor per test).
**Parameters:**
- `filename` (`str`): output path for the generated test MP3.
**Returns:** `None`.

#### `TestAudioFattener.test_read_audio(self) -> None`
**File:** fat_llama/tests/test_feed.py:37
**Kind:** method
**Description:** Asserts `read_audio` on the generated sine-wave MP3 returns a 44100 Hz sample rate and 44100 samples (1 second); that a mono source comes back as a flat 1-D array (`audio.channels == 1`, `samples.ndim == 1`) rather than reshaped to `(N, 2)`; that duration is 1000 ms; that the mp3-reported bitrate falls within the encoder's default CBR band (32000–320000, not a single hard-coded value); and that the samples actually carry signal — non-silent, all-finite, and (via windowed FFT) a dominant spectral peak within 5 Hz of 440 Hz.
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_write_audio(self) -> None`
**File:** fat_llama/tests/test_feed.py:67
**Kind:** method
**Description:** Reads the generated sine-wave MP3, writes it out as FLAC via `write_audio`, then asserts real coherence of the round-trip: output file exists; `soundfile.info` reports the same sample rate/channel count and a duration matching the ~1 s input within 0.05 s tolerance; the re-read written data is non-silent and all-finite; the written waveform correlates >0.999 with the peak-normalized input (guards against a test that would pass for any arbitrary non-silent signal, not necessarily the true written waveform); the dominant spectral peak is still within 5 Hz of 440 Hz; and fewer than 5% of written samples sit at full-scale clipping (`> 0.999`), guarding against `write_audio` handing raw (non-normalized, PCM-scale) samples straight to an integer `soundfile` subtype. As of the cycle 1 fix to `write_audio`, this test passes. Cleans up `test_output.flac` in a `finally` block.
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_lms_filter_no_extended_warmup_dropout(self) -> None`
**File:** fat_llama/tests/test_feed.py:140
**Kind:** method
**Description:** Added in cycle 1 as a regression test for `lms_filter`'s warm-up fix. Builds a 50ms two-tone synthetic signal (300 Hz + 900 Hz), runs `lms_filter(signal, signal, mu=0.001, num_taps=32)`, and checks the RMS of the filtered output over the 50 samples immediately following the first `num_taps` against the RMS of the input signal over that same window — asserting the ratio exceeds 0.5. Guards against `lms_filter` ramping up from a zero-initialized state instead of tracking the signal from (near) the first sample; production symptom before the fix was a ~200ms, -82 dBFS dropout at the head of upscaled audio with no corresponding silence in the source.
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_ist_harmonic_injection_bounded_across_iterations(self) -> None`
**File:** fat_llama/tests/test_feed.py:178
**Kind:** method
**Description:** Added in cycle 2 as a regression test for `iterative_soft_thresholding`'s bounded-harmonic fix. Builds a synthetic two-tone signal (300 Hz + 700 Hz, n=2000), runs `iterative_soft_thresholding` at `max_iter=5` and again at `max_iter=150`, and asserts the 150-iteration run's peak magnitude is less than 2x the 5-iteration run's — guarding against the harmonic-injection term accumulating roughly linearly with `max_iter` instead of staying bounded (production symptom before the fix: a measured +6.12 dB broadband noise-floor rise).
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_target_bitrate_kbps_drives_upscale_factor_not_output_bitrate(self) -> None`
**File:** fat_llama/tests/test_feed.py:213
**Kind:** method
**Description:** Added in cycle 2 to document `target_bitrate_kbps`'s intended contract; strengthened by `audio-quality-checker` with real audio-content assertions. Runs a full `upscale()` call (`max_iterations=2`, `toggle_adaptive_filter=False` to stay fast, `target_bitrate_kbps=800`) and asserts: the output sample rate matches `source_sample_rate * round(target_bitrate_kbps * 1000 / source_bitrate)`; duration is preserved (~1s, since both sample count and rate scale by `upscale_factor`); the output is mono, all-finite, non-silent (RMS > 1e-3), has fewer than 5% full-scale-clipped samples, and its dominant spectral peak is still within 5 Hz of the source's 440 Hz tone; and the real output bitrate (computed from file size / duration) is more than 2x `target_bitrate_kbps` — confirming the parameter only derives `upscale_factor` and does not constrain the real (uncompressed PCM) output bitrate, which is structurally much higher by design.
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

## Notes

- `factblock_stale` / inconsistencies observed while reviewing (not fixed by this skill — report only):
  - `setup.py`'s `console_scripts` entry point (`example=example:main`) references `example.main`, but `example.py` defines no `main` function — it only has top-level script code. This entry point would fail if invoked as an installed console script. Flagged by code-tester-reviewer and generate-code in iterate-fat-llama cycle 1; left unfixed both times as out of scope (setup.py sits outside `fat_llama/**`).
  - `setup.py`'s `version` (`1.1.0`) does not yet reflect this iterate-fat-llama run's in-progress cycles (branch `iterate-fat-llama/20260905-030218`, off `v-1.4.0-latest`) — the version bump happens in that skill's own Step 6, after cycling completes.
  - `analysis.py` imports `cupy` unconditionally at module load, same as `feed.py` — both modules require a CUDA-capable GPU/environment to import successfully at all, not just to run GPU-specific code paths.
  - `new_interpolation_algorithm`'s zero-order-hold interpolation produces spectral imaging artifacts with no genuine added high-frequency detail (see its factblock above) — investigated in cycles 1 and 2, deliberately left unfixed both times pending an explicit decision on replacing the documented interpolation method.
  - The full baseline `upscale()` pipeline (max_iterations=300, real ~15s input_test.mp3, both channels) takes roughly 20-21 minutes end to end, ~91% of it in `lms_filter`'s per-sample Python loop (not `iterative_soft_thresholding`, which is vectorized and only ~105s for both channels) — a known performance characteristic, not (yet) treated as a bug.
  - The repo-root reference `input_test.flac` (used by `audio-quality-checker`'s spectral-deviation scoring) is itself a prior output of this same `upscale()` pipeline (waveform correlates r=0.9999986 with a fresh `output_test.flac` run), not an independent ground-truth master — flagged in iterate-fat-llama cycle 2 as structurally unfixable by `generate-code` (the asset sits outside `fat_llama/**`), a likely ceiling on the spectral-deviation score's meaningfulness until a human replaces the reference asset.
