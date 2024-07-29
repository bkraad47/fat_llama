![Fat Llama Logo](https://drive.google.com/uc?export=view&id=1BHe352g43zAdDYLDusBrBPdNvRIvlQwp)

# Fat Llama ![build - status](https://github.com/bkraad47/fat_llama/actions/workflows/tests.yml/badge.svg) ![PyPI](https://img.shields.io/pypi/v/fat-llama?label=pypi%20package) ![PyPI - Downloads](https://img.shields.io/pypi/dm/fat-llama)
fat_llama is a Python package for upscaling audio files to FLAC or WAV formats using advanced audio processing techniques. It utilizes CUDA-accelerated calculations to enhance audio quality by upsampling and adding missing frequencies through FFT (Fast Fourier Transform), resulting in richer and more detailed audio.

## Features

- Upscale MP3 files to high-quality FLAC format.
- Iterative soft thresholding (IST) for enhanced audio processing.
- Auto-scaling amplitude adjustment and normalization.
- Supports GPU-accelerated processing with CuPy.

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

# Example call to the method
upscale(
    input_file_path='input_test.mp3',
    output_file_path='output_test.flac',
    source_format='mp3',
    target_format='flac',
    max_iterations=1000,
    threshold_value=0.6,
    target_bitrate_kbps=1400
)
```
### Function Parameters

- `input_file_path (str)`: Path to the input audio file. Mandatory.
- `output_file_path (str)`: Path to the output processed audio file. Mandatory.
- `source_format (str)`: Format of the input audio file (e.g., 'mp3', 'wav', 'ogg', 'flac').
- `target_format (str)`: Format of the output audio file (e.g., 'flac', 'wav'). Default is 'flac'.
- `max_iterations (int)`: Maximum number of iterations for IST. Default is 800.
- `threshold_value (float)`: Threshold value for IST. Default is 0.6.
- `target_bitrate_kbps (int)`: Target bitrate in kbps. Default is 1411.

## Running the Example

To run the example, execute the following command:
```
python example.py
```
This will upscale the MP3 file specified in the example and produce a FLAC file with full processing.

## Spectrogram Results

![Spectrogram Results](https://drive.google.com/uc?export=view&id=1_QgMQ8FR1Rryyw22bBQa0EAEGIjw9eS_)

## How it works

![How it Works](https://drive.google.com/uc?export=view&id=1rzIGzghlRUMTrqKSst_FdZk-WhpznVX1)

## Algorithm Explanation

The upscaling process involves several steps:

1. **Reading Audio File**: The audio file is read, and the audio samples are extracted along with the sample rate and bitrate.
2. **Calculating Upscale Factor**: The upscale factor is calculated to achieve the target bitrate.
3. **Upscaling Channels**: The audio channels are upscaled using an interpolation algorithm. Each sample is repeated multiple times to increase the resolution.
4. **Iterative Soft Thresholding (IST)**: IST is applied to enhance the audio by adding missing frequencies. This process uses FFT to transform the signal into the frequency domain, apply a threshold to keep significant frequencies, and then inverse transform back to the time domain.
5. **Scaling Amplitude**: The amplitude of the upscaled audio is scaled to match the original.
6. **Normalizing Audio**: The audio is normalized to the range -1 to 1.
7. **Writing FLAC File**: The processed audio is written to a FLAC file.

## Why FFT and IST?

FFT (Fast Fourier Transform) is used to transform the audio signal into the frequency domain. This allows for the identification and manipulation of specific frequency components. By applying a threshold in the frequency domain, we can keep significant frequencies and discard noise and add it to our upscaling data to add detail to upscaling frequencies.

The report titled "Fast Sparse Fourier Transformations for NMR Spectroscopy" by Badruddin Kamal, supervised by Thomas Huber and Alastair Rendall, 2015, provides a comprehensive understanding of sparse representations and their applications in signal processing. IST leverages the concepts from this report to add missing frequencies and enhance the audio quality by making it more detailed and rich. This is particularly useful in upscaling audio where some frequencies might be missing or congested.

### Test Audio Source

ericzo - beyond link(https://soundcloud.com/ericzomusic/free-electro-trap-anthem-beyond)

## Changelog

All notable changes to this project will be documented in this file.

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