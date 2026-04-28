![Fat Llama Logo](https://drive.google.com/uc?export=view&id=1BHe352g43zAdDYLDusBrBPdNvRIvlQwp)

# Fat Llama ![build - status](https://github.com/bkraad47/fat_llama/actions/workflows/tests.yml/badge.svg) ![PyPI](https://img.shields.io/pypi/v/fat-llama?label=pypi%20package) ![PyPI - Downloads](https://img.shields.io/pypi/dm/fat-llama)
fat_llama is a Python package for upscaling audio files to FLAC or WAV formats using advanced audio processing techniques. It utilizes CUDA-accelerated calculations to enhance audio quality by upsampling and adding missing frequencies through FFT (Fast Fourier Transform), resulting in richer and more detailed audio.

## Features

- Upscale MP3/OGG/WAV/FLAC files to high-quality FLAC or WAV.
- Bandlimited FFT (sinc) interpolation — no spectral imaging artifacts.
- Iterative soft thresholding (IST) for enhanced spectral reconstruction.
- GPU-accelerated LMS adaptive filtering via a block-parallel CUDA kernel.
- Auto-scaling amplitude adjustment and normalization.
- ML-based bandwidth extension via [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) to synthesize content above the source codec's lowpass (e.g. >16 kHz for MP3).
- GPU-accelerated processing with CuPy.

## Requirements

- CUDA capable GPU

**(Note: For cpu verison please look at https://pypi.org/project/fat-llama-fftw/)**

## Installation

Install via pip:
```
pip install fat-llama
```
Note: This version works with CUDA 12.

Further need CUDA & CuPy properly installed: https://docs.cupy.dev/en/stable/install.html

Also, requires ffmpeg: https://support.audacityteam.org/basics/installing-ffmpeg

### AudioSR (ML-based super-resolution)

`audiosr` and `torch` are installed automatically as dependencies. The first
time `toggle_audiosr=True` is used, AudioSR downloads its model weights
(a few hundred MB) on demand.

**Note to install on older versions of CUDA and CuPy. You will need to download specific versions and install locally.**

- cupy version - https://github.com/bkraad47/fat_llama/tree/v-0.1.3---cupy
- cupy-cuda11x version - https://github.com/bkraad47/fat_llama/tree/v-0.1.3---cupy-cuda11x

To install locally:
```
git clone <target_url>
cd fat_llama
pip install .
```

## Usage

### Example Usage

You can run the example provided in example.py:

```
from fat_llama.audio_fattener.feed import upscale

upscale(
    input_file_path='input_test.mp3',
    output_file_path='output_test.flac',
    source_format='mp3',
    target_format='flac',
    max_iterations=100,
    threshold_value=0.6,
    target_bitrate_kbps=1400,
    toggle_normalize=True,
    toggle_autoscale=True,
    toggle_adaptive_filter=True,

    toggle_audiosr=False,
    audiosr_model='basic',
    audiosr_ddim_steps=50,
    audiosr_guidance_scale=3.5,
    audiosr_seed=42,
    audiosr_device=None,
)
```
### Function Parameters

- `input_file_path (str)`: Path to the input audio file. Mandatory.
- `output_file_path (str)`: Path to the output processed audio file. Mandatory.
- `source_format (str)`: Format of the input audio file (e.g., 'mp3', 'wav', 'ogg', 'flac').
- `target_format (str)`: Format of the output audio file (e.g., 'flac', 'wav'). Default is 'flac'.
- `max_iterations (int)`: Maximum number of iterations for IST. Default is 300.
- `threshold_value (float)`: Threshold value for IST. Default is 0.6.
- `target_bitrate_kbps (int)`: Target bitrate in kbps. Default is 1411.
- `toggle_normalize (bool)`: Whether to normalize the audio. Default True.
- `toggle_autoscale (bool)`: Whether to autoscale the audio based on the original audio. Default True.
- `toggle_adaptive_filter (bool)`: Apply LMS adaptive filtering (block-parallel CUDA kernel). Default True.
- `toggle_audiosr (bool)`: Run pretrained AudioSR diffusion super-resolution before the sinc/IST stage to synthesize content above the source's Nyquist. Default False.
- `audiosr_model (str)`: AudioSR variant, `'basic'` or `'speech'`. Default `'basic'`.
- `audiosr_ddim_steps (int)`: Diffusion sampling steps. Default 50.
- `audiosr_guidance_scale (float)`: Classifier-free guidance scale. Default 3.5.
- `audiosr_seed (int)`: RNG seed for AudioSR. Default 42.
- `audiosr_device (str | None)`: Torch device override, e.g. `'cuda'` or `'cpu'`. Default None (auto).

## Running the Example

To run the example, execute the following command:
```
python example.py
```
This will upscale the MP3 file specified in the example and produce a FLAC file with full processing.

## On signal-altering options

The pipeline preserves the source as much as possible by default in terms of internal precision (fp64) and FFmpeg decoding (`-drc_scale 0`). The toggles `toggle_normalize`, `toggle_autoscale`, and `toggle_adaptive_filter` *do* alter the signal — set them to `False` if you want the least intrusive result, or to `True` if you want the fuller processing chain. They are independent and can be combined freely.

## Spectrogram Results

![Spectrogram Results](https://drive.google.com/uc?export=view&id=1nPGMHuR8hEeoo3rl8zWFuREf35uQF1IA)

## How it works

![How it Works](https://drive.google.com/uc?export=view&id=1rzIGzghlRUMTrqKSst_FdZk-WhpznVX1)

## Algorithm Explanation

The upscaling process involves the following stages:

1. **Reading Audio File**: The audio file is read; samples, sample rate, and bitrate are extracted.
2. **(Optional) AudioSR Super-Resolution**: If `toggle_audiosr=True`, a pretrained latent-diffusion model synthesizes plausible high-frequency content above the source codec's lowpass (e.g. >16 kHz for MP3). Output is at 48 kHz.
3. **Calculating Upscale Factor**: The integer upscale factor is derived from the target bitrate.
4. **Bandlimited Upscaling**: Channels are upsampled by zero-padding the rFFT spectrum and inverse-transforming. This is mathematically equivalent to sinc interpolation and produces no spectral images (avoids the "ghosting" artifact zero-order hold creates above the original Nyquist).
5. **Iterative Soft Thresholding (IST)**: FFT → threshold → IFFT loop reinforces significant frequency components and suppresses noise.
6. **Auto-Scaling**: Per-channel amplitude is restored to match the original.
7. **Normalization**: Audio is scaled to [-1, 1].
8. **LMS Adaptive Filtering**: Block-parallel LMS — the signal is split into independent chunks processed sequentially per CUDA thread and in parallel across threads, eliminating the previous single-thread bottleneck.
9. **Writing Output**: The processed audio is written as 24-bit FLAC or WAV.

## Why FFT and IST?

FFT (Fast Fourier Transform) is used to transform the audio signal into the frequency domain. This allows for the identification and manipulation of specific frequency components. By applying a threshold in the frequency domain, we can keep significant frequencies and discard noise and add it to our upscaling data to add detail to upscaling frequencies.

The report titled "Fast Sparse Fourier Transformations for NMR Spectroscopy" by Badruddin Kamal, supervised by Thomas Huber and Alastair Rendall, 2015, provides a comprehensive understanding of sparse representations and their applications in signal processing. IST leverages the concepts from this report to add missing frequencies and enhance the audio quality by making it more detailed and rich. This is particularly useful in upscaling audio where some frequencies might be missing or congested.

### Test Audio Source

ericzo - beyond link(https://soundcloud.com/ericzomusic/free-electro-trap-anthem-beyond)

## Changelog

All notable changes to this project will be documented in this file.

### [1.2.0.1] - 2026-04-28

#### Fixed

- Corrected the `upscale_factor` derivation. It previously divided the requested
  target *bitrate* by the source *bitrate* and used that ratio as a sample-rate
  multiplier, which produced absurd output sample rates (e.g. ~250–308 kHz and
  ~5300 kbps reported FLAC bitrate from a 128 kbps MP3 source). The factor is
  now derived from a sane target sample rate picked from the requested bitrate
  tier and clamped to `[1, 4]` (typical 44.1 kHz MP3 sources now upscale to
  88.2/176.4 kHz instead of 250 kHz+).
- Replaced the single-thread LMS CUDA kernel with a block-parallel kernel.
  The previous implementation ran the entire sequential LMS recurrence on one
  CUDA thread, which made `toggle_adaptive_filter=True` hang for tens of
  minutes on laptop GPUs (e.g. RTX 3060 Mobile) for short clips. The new
  kernel splits the signal into independent chunks processed in parallel
  across threads (block-LMS).

#### Changed

- `setup.py` and `requirements.txt`: `audiosr` and `torch` are now mandatory
  dependencies and are automatically installed via `pip install fat-llama`.

### [1.2.0] - 2026-04-28

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