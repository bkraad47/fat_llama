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
│   └── CURRENT_STATE.md
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

Package metadata for `fat_llama` (PyPI distribution). No functions/classes — a single `setuptools.setup(...)` call. Current `version`: `1.1.0`. `install_requires`: `numpy`, `cupy-cuda12x`, `pydub`, `soundfile`, `mutagen`, `scipy`. Declares console-script entry point `example=example:main`.

## fat_llama/__init__.py

Empty — no exports.

## fat_llama/audio_fattener/__init__.py

Empty — no exports.

## fat_llama/audio_fattener/feed.py

The package's core module: reads/writes audio files and implements the GPU-accelerated (CuPy) upscaling pipeline (interpolation, iterative soft-thresholding "IST", LMS adaptive filtering).

### `read_audio(file_path, format) -> Tuple[int, np.ndarray, Optional[float], AudioSegment]`
**File:** fat_llama/audio_fattener/feed.py:16
**Kind:** function
**Description:** Reads an audio file via `pydub.AudioSegment.from_file` (with `-drc_scale 0` passed to ffmpeg), converts samples to a `float64` NumPy array, and looks up the file's bitrate via `mutagen` for mp3/flac/ogg/wav (falling back to a byte-rate estimate for other formats). Reshapes samples to `(-1, 2)` for stereo input. Raises `FileNotFoundError` if the input path doesn't exist.
**Parameters:**
- `file_path` (`str`): path to the input audio file.
- `format` (`str`): input format (`'mp3'`, `'flac'`, `'ogg'`, `'wav'`, or other ffmpeg-supported format).
**Returns:** `(sample_rate, samples, bitrate, audio)` — sample rate in Hz, sample data (`float64`, shape `(n,)` or `(n, 2)`), bitrate in bits/sec (`None` if undeterminable), and the underlying `AudioSegment`.
**Usage:**
```python
# From fat_llama/tests/test_feed.py:
sample_rate, samples, bitrate, audio = read_audio(self.test_mp3_file, format='mp3')
```

### `write_audio(file_path, sample_rate, data, format) -> None`
**File:** fat_llama/audio_fattener/feed.py:66
**Kind:** function
**Description:** Writes sample data to disk via `soundfile.write`. Supports `'flac'` and `'wav'` only (both written as 24-bit PCM); raises `ValueError` for any other target format.
**Parameters:**
- `file_path` (`str`): output file path.
- `sample_rate` (`int`): sample rate in Hz.
- `data` (`np.ndarray`): audio sample data.
- `format` (`str`): output format, `'flac'` or `'wav'`.
**Returns:** `None`.
**Usage:**
```python
# From fat_llama/tests/test_feed.py:
write_audio(output_file, sample_rate, samples, format='flac')
```

### `new_interpolation_algorithm(data, upscale_factor) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:83
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

### `initialize_ist(data, threshold) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:107
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
**File:** fat_llama/audio_fattener/feed.py:122
**Kind:** function
**Description:** Runs `max_iter` rounds of: FFT the thresholded signal, zero out FFT bins at/below `threshold`, inverse-FFT back to the time domain, then add a small sinusoidal harmonic term (`0.1 * sin(linspace(0, 2π, len))`) each iteration — used to reconstruct missing high-frequency content after upscaling.
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
**File:** fat_llama/audio_fattener/feed.py:147
**Kind:** function
**Description:** Applies a sample-by-sample LMS adaptive filter: for each sample past the first `num_taps`, computes a filtered output from the last `num_taps` input samples, compares it to `desired`, and adjusts filter weights by the LMS update rule (weights clipped to ±1e10 for numerical stability).
**Parameters:**
- `signal` (`cp.ndarray`): input signal to filter.
- `desired` (`cp.ndarray`): desired/reference signal used to compute the error term.
- `mu` (`float`): LMS step size (default `0.001`).
- `num_taps` (`int`): number of filter taps (default `32`).
**Returns:** `cp.ndarray` — the filtered signal, same length as `signal`.
**Usage:**
```python
filtered = lms_filter(channel, channel, mu=0.001, num_taps=32)  # illustrative — called in upscale() with signal == desired
```

### `upscale_channels(channels, upscale_factor, max_iter, threshold) -> cp.ndarray`
**File:** fat_llama/audio_fattener/feed.py:185
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
**File:** fat_llama/audio_fattener/feed.py:211
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
**File:** fat_llama/audio_fattener/feed.py:223
**Kind:** function
**Description:** The package's main public entry point. Validates `target_bitrate_kbps` against per-format ranges (`flac`: 800–1411 kbps, `wav`: 800–6444 kbps), reads the input via `read_audio`, computes an integer upscale factor from the ratio of target to original bitrate (defaulting to `4` if the original bitrate is unknown), runs the channel pipeline (`upscale_channels`), optionally autoscales each channel back to its original peak amplitude, optionally normalizes, optionally applies `lms_filter` per channel (using each channel as its own `desired` signal), and writes the result via `write_audio` at `sample_rate * upscale_factor`.
**Parameters:**
- `input_file_path` (`str`): path to the input audio file.
- `output_file_path` (`str`): path to write the processed output.
- `source_format` (`str`): input format passed to `read_audio` (e.g. `'mp3'`, `'wav'`, `'ogg'`, `'flac'`).
- `target_format` (`str`): output format, `'flac'` (default) or `'wav'`.
- `max_iterations` (`int`): IST iteration count (default `300`).
- `threshold_value` (`float`): IST/LMS threshold (default `0.6`).
- `target_bitrate_kbps` (`int`): desired output bitrate in kbps (default `1411`); must fall within the valid range for `target_format`.
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

`unittest`-based test module for `feed.py`, covering `read_audio` and `write_audio` against a synthesized 440 Hz sine-wave MP3.

### `TestAudioFattener`
**File:** fat_llama/tests/test_feed.py:9
**Kind:** class
**Description:** `unittest.TestCase` subclass exercising `read_audio`/`write_audio`. `setUp` generates a 1-second 440 Hz sine wave as `test_input.mp3` via `pydub.generators.Sine`; `tearDown` removes the generated MP3 and any leftover `output_processed.flac`.
**Usage:**
```python
python -m pytest fat_llama/tests/test_feed.py -v
```

#### `TestAudioFattener.setUp(self) -> None`
**File:** fat_llama/tests/test_feed.py:11
**Kind:** method
**Description:** Creates a fresh test MP3 (`test_input.mp3`) before each test via `create_test_mp3`. Inferred from body (no docstring).
**Returns:** `None`.

#### `TestAudioFattener.tearDown(self) -> None`
**File:** fat_llama/tests/test_feed.py:16
**Kind:** method
**Description:** Deletes `test_input.mp3` and `output_processed.flac` if present, after each test. Inferred from body (no docstring).
**Returns:** `None`.

#### `TestAudioFattener.create_test_mp3(self, filename) -> None`
**File:** fat_llama/tests/test_feed.py:23
**Kind:** method
**Description:** Synthesizes a 1-second 440 Hz sine wave with `pydub.generators.Sine` and exports it as an MP3 to `filename`. Inferred from body (no docstring).
**Parameters:**
- `filename` (`str`): output path for the generated test MP3.
**Returns:** `None`.

#### `TestAudioFattener.test_read_audio(self) -> None`
**File:** fat_llama/tests/test_feed.py:28
**Kind:** method
**Description:** Asserts that `read_audio` on the generated sine-wave MP3 returns a 44100 Hz sample rate, 44100 samples (1 second), and a bitrate of exactly `63999`.
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

#### `TestAudioFattener.test_write_audio(self) -> None`
**File:** fat_llama/tests/test_feed.py:34
**Kind:** method
**Description:** Reads the generated sine-wave MP3, writes it out as FLAC via `write_audio`, and asserts the output file exists on disk (existence-only check — does not verify audio content/properties of the written file).
**Returns:** `None` — raises `AssertionError` on failure via `unittest` assertions.

## Notes

- `factblock_stale` / inconsistencies observed while reviewing (not fixed by this skill — report only):
  - `setup.py`'s `console_scripts` entry point (`example=example:main`) references `example.main`, but `example.py` defines no `main` function — it only has top-level script code. This entry point would fail if invoked as an installed console script.
  - `setup.py`'s `version` (`1.1.0`) does not match the current git branch name (`1.4.0-beta`) — version bump bookkeeping for this branch has not yet touched `setup.py`.
  - `analysis.py` imports `cupy` unconditionally at module load, same as `feed.py` — both modules require a CUDA-capable GPU/environment to import successfully at all, not just to run GPU-specific code paths.
