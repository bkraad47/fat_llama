![Fat Llama Logo](logo.png)
# Fat Llama

fat_llama is a Python package for upscaling MP3 files to FLAC format using advanced audio processing techniques. It utilizes GPU-accelerated calculations to enhance audio quality by upsampling and adding missing frequencies, resulting in richer and more detailed audio.

## Features

- Upscale MP3 files to high-quality FLAC format.
- Optional iterative soft thresholding (IST) for enhanced audio processing.
- Gain adjustment, equalization, and optional Wiener filtering.
- Supports GPU-accelerated processing with CuPy.

## Requirements
- Cuda capable GPU
## Installation
Install via pip:
```
pip install fat_llama
```
Note: This version works with cuda 12.

Further need CUDA & CuPy properly installed: https://docs.cupy.dev/en/stable/install.html

Also, requires ffmpeg for windows: https://support.audacityteam.org/basics/installing-ffmpeg

**Note to install on older versions of cuda and cupy. You will need to download specific version and install locally.**
- cupy version - https://github.com/bkraad47/fat_llama/tree/v-0.1.3---cupy 
- cupy-cuda11x version - https://github.com/bkraad47/fat_llama/tree/v-0.1.3---cupy-cuda11x

To install locally
```
git clone <target_url>
cd fat_llama
pip install .
```
## Usage
### Example Usage
You can run the example provided in example.py:

```
from fat_llama.audio_fattener.feed import upscale_mp3_to_flac 

# Example call to the method
upscale_mp3_to_flac(
    input_file_path='input_test.mp3',
    output_file_path_processed='output_test.flac',
    max_iterations=1000,
    threshold_value=0.6,
    gain_factor=32.8,
    reduction_profile=[
        (5, 140, -38.4),
        (1000, 10000, 36.4),
    ],
    lowcut=5.0,
    highcut=150000.0,
    target_bitrate_kbps=1400,
    output_file_path_no_processing=None,
    use_wiener_filter=False
)
```
### Function Parameters
input_file_path (str): Path to the input MP3 file. Mandatory.
output_file_path_processed (str): Path to the output processed FLAC file. Mandatory.
max_iterations (int): Number of iterations for IST. Default is 400.
threshold_value (float): Threshold value for IST. Default is 0.4.
gain_factor (float): Gain factor for scaling amplitude. Default is 22.8.
reduction_profile (list): Profile for gain reduction. Default is [(5, 140, -28.4), (1000, 10000, 26.4)].
lowcut (float): Low cut frequency for equalizer. Default is 5.0.
highcut (float): High cut frequency for equalizer. Default is 150000.0.
target_bitrate_kbps (int): Target bitrate in kbps. Default is 1400.
output_file_path_no_processing (str): Path to the output upscaled (no processing) FLAC file. Default is None.
use_wiener_filter (bool): Flag to use Wiener filter. Default is False.

## Running the Example
To run the example, execute the following command:
```
python example.py
```
This will upscale the MP3 file specified in the example and produce two FLAC files: one with just upscaling and one with full processing.

## Algorithm Explanation
The upscaling process involves several steps:

1. Reading MP3 File: The MP3 file is read, and the audio samples are extracted along with the sample rate and bitrate.
2. Calculating Upscale Factor: The upscale factor is calculated to achieve the target bitrate.
3. Upscaling Channels: The audio channels are upscaled using an interpolation algorithm. Each sample is repeated multiple times to increase the resolution.
4. Iterative Soft Thresholding (IST): IST is applied to enhance the audio by adding missing frequencies. This process uses FFT to transform the signal into the frequency domain, apply a threshold to keep significant frequencies, and then inverse transform back to the time domain.
5. Scaling Amplitude: The amplitude of the upscaled audio is scaled to match the original.
6. Applying Gain Reduction: Frequency-specific gain reduction is applied based on a given profile.
7. Equalization: A bandpass filter is applied to the audio to equalize it.
8. Optional Wiener Filtering: Wiener filtering is applied to reduce noise if specified.
9. Writing FLAC File: The processed audio is written to a FLAC file.

## Why FFT and IST?
FFT (Fast Fourier Transform) is used to transform the audio signal into the frequency domain. This allows for the identification and manipulation of specific frequency components. By applying a threshold in the frequency domain, we can keep significant frequencies and discard noise and add it to our upscaling data to add detail to upscaling frequencies.

The report titled "Fast Sparse Fourier Transformations for NMR Spectroscopy" by Badruddin Kamal, supervised by Thomas Huber and Alastair Rendall, 2015, provides a comprehensive understanding of sparse representations and their applications in signal processing. IST leverages the concepts from this report to add missing frequencies and enhance the audio quality by making it more detailed and rich. This is particularly useful in upscaling audio where some frequencies might be missing or congested.

### Test Audio Source, ericzo - beyond link(https://soundcloud.com/ericzomusic/free-electro-trap-anthem-beyond)

## Run Tests
Clone the repository and install the requirements.
Test the packaged files:
```
python -m unittest discover -s fat_llama/tests
```
