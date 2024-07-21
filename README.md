# fat_llama

fat_llama is a Python package for upscaling MP3 files to FLAC format using advanced audio processing techniques. It utilizes GPU-accelerated calculations to enhance audio quality by upsampling and adding missing frequencies, resulting in richer and more detailed audio.

## Features

- Upscale MP3 files to high-quality FLAC format.
- Optional iterative soft thresholding (IST) for enhanced audio processing.
- Gain adjustment, equalization, and optional Wiener filtering.
- Supports GPU-accelerated processing with CuPy.

## Installation

Clone the repository:

```
sh
git clone https://github.com/yourusername/fat_llama.git
cd fat_llama 
```
Install the required packages:

```
pip install -r requirements.txt
```
Further need CuPy properly installed: https://docs.cupy.dev/en/stable/install.html
Also, requires ffmpeg for windows: https://support.audacityteam.org/basics/installing-ffmpeg
## Usage
### Example Usage
You can run the example provided in example.py:

```
from fat_llama.audio_fattener import upscale_mp3_to_flac

# Example call to the method
upscale_mp3_to_flac(
    input_file_path='input_short.mp3',
    output_file_path_processed='output_processed.flac',
    max_iterations=400,
    threshold_value=0.4,
    gain_factor=22.8,
    reduction_profile=[
        (5, 140, -28.4),
        (1000, 10000, 26.4),
    ],
    lowcut=5.0,
    highcut=150000.0,
    target_bitrate_kbps=1400,
    output_file_path_no_processing='output_upscaled_no_processing.flac',
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

### Reading MP3 File: The MP3 file is read, and the audio samples are extracted along with the sample rate and bitrate.
### Calculating Upscale Factor: The upscale factor is calculated to achieve the target bitrate.
### Upscaling Channels: The audio channels are upscaled using an interpolation algorithm. Each sample is repeated multiple times to increase the resolution.
### Iterative Soft Thresholding (IST): Optional IST is applied to enhance the audio by adding missing frequencies. This process uses FFT to transform the signal into the frequency domain, apply a threshold to keep significant frequencies, and then inverse transform back to the time domain.
### Scaling Amplitude: The amplitude of the upscaled audio is scaled to match the original.
### Applying Gain Reduction: Frequency-specific gain reduction is applied based on a given profile.
### Equalization: A bandpass filter is applied to the audio to equalize it.
### Optional Wiener Filtering: Wiener filtering is applied to reduce noise if specified.
### Writing FLAC File: The processed audio is written to a FLAC file.

## Why FFT and IST?
FFT (Fast Fourier Transform) is used to transform the audio signal into the frequency domain. This allows for the identification and manipulation of specific frequency components. By applying a threshold in the frequency domain, we can keep significant frequencies and discard noise.

The report titled "Fast Sparse Fourier Transformations for NMR Spectroscopy" by Badruddin Kamal, supervised by Thomas Huber and Alastair Rendall, 2015, provides a comprehensive understanding of sparse representations and their applications in signal processing. IST leverages the concepts from this report to add missing frequencies and enhance the audio quality by making it more detailed and rich. This is particularly useful in upscaling audio where some frequencies might be missing or congested.

### Test Audio Source, ericzo - beyond link(https://soundcloud.com/ericzomusic/free-electro-trap-anthem-beyond)
