import numpy as np
import cupy as cp
from pydub import AudioSegment
import soundfile as sf
import os
import logging
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_audio(file_path, format):
    """ 
    Read an audio file and return the sample rate and data as a NumPy array 
    
    Parameters:
    file_path (str): The path to the input audio file.
    format (str): The format of the input audio file.
    
    Returns:
    int: Sample rate of the audio.
    numpy.ndarray: Audio samples.
    int: Bitrate of the audio file.
    AudioSegment: The audio segment object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    audio = AudioSegment.from_file(file_path, format=format)
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    bitrate = None
    
    if format == 'mp3':
        mp3_info = MP3(file_path)
        bitrate = mp3_info.info.bitrate
    elif format == 'flac':
        flac_info = FLAC(file_path)
        bitrate = flac_info.info.bitrate
    elif format == 'ogg':
        ogg_info = OggVorbis(file_path)
        bitrate = ogg_info.info.bitrate
    elif format == 'wav':
        wav_info = WAVE(file_path)
        bitrate = wav_info.info.bitrate
    else:
        duration_seconds = len(audio) / 1000.0
        bitrate = (len(samples) * 8) / duration_seconds  # calculate bitrate for other formats
    
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    
    return sample_rate, samples, bitrate, audio

def write_audio(file_path, sample_rate, data, format):
    """ 
    Write data to an audio file 
    
    Parameters:
    file_path (str): The path to the output audio file.
    sample_rate (int): The sample rate of the audio.
    data (numpy.ndarray): The audio data to write.
    format (str): The format of the output audio file.
    """
    if format == 'flac':
        sf.write(file_path, data.astype(np.float32), sample_rate, format='FLAC', subtype='PCM_24')
    elif format == 'wav':
        sf.write(file_path, data.astype(np.float32), sample_rate, format='WAV', subtype='PCM_24')
    else:
        raise ValueError(f"Unsupported target format: {format}")

def new_interpolation_algorithm(data, upscale_factor):
    """ 
    Interpolate data with scaled changes around center points 
    
    Parameters:
    data (numpy.ndarray): The input audio data.
    upscale_factor (int): The factor by which to upscale the audio data.
    
    Returns:
    numpy.ndarray: The upscaled audio data.
    """
    original_length = len(data)
    expanded_length = original_length * upscale_factor
    expanded_data = np.zeros(expanded_length, dtype=np.float32)
    
    for i in range(original_length):
        center_point = data[i]
        for j in range(upscale_factor):
            index = i * upscale_factor + j
            expanded_data[index] = center_point
    
    return expanded_data

def initialize_ist(data, threshold):
    """ 
    Initialize IST variables 
    
    Parameters:
    data (cupy.ndarray): The input audio data.
    threshold (float): The threshold value for IST.
    
    Returns:
    cupy.ndarray: The thresholded audio data.
    """
    mask = cp.abs(data) > threshold
    data_thres = cp.where(mask, data, 0)
    return data_thres

def iterative_soft_thresholding(data, max_iter, threshold):
    """ 
    Perform IST on data using CuPy and cuFFT 
    
    Parameters:
    data (numpy.ndarray): The input audio data.
    max_iter (int): The maximum number of iterations for IST.
    threshold (float): The threshold value for IST.
    
    Returns:
    numpy.ndarray: The processed audio data after IST.
    """
    data_cp = cp.array(data, dtype=cp.float32)
    data_thres = initialize_ist(data_cp, threshold)
    for _ in range(max_iter):
        data_fft = cp.fft.fft(data_thres)
        mask = cp.abs(data_fft) > threshold
        data_fft_thres = cp.where(mask, data_fft, 0)
        data_thres = cp.fft.ifft(data_fft_thres).real
    return cp.asnumpy(data_thres)

def upscale_channels(channels, upscale_factor, max_iter, threshold):
    """ 
    Process and upscale channels using the new interpolation and IST algorithms 
    
    Parameters:
    channels (numpy.ndarray): The input audio channels.
    upscale_factor (int): The factor by which to upscale the audio data.
    max_iter (int): The maximum number of iterations for IST.
    threshold (float): The threshold value for IST.
    
    Returns:
    numpy.ndarray: The upscaled (and processed, if apply_ist=True) audio data.
    """
    processed_channels = []
    for channel in channels.T:
        logger.info("Interpolating data...")
        expanded_channel = new_interpolation_algorithm(channel, upscale_factor)
        
        logger.info("Performing IST...")
        ist_changes = iterative_soft_thresholding(expanded_channel, max_iter, threshold)
        expanded_channel = expanded_channel.astype(np.float32) + ist_changes
        
        processed_channels.append(expanded_channel)
    
    return np.column_stack(processed_channels)

def normalize_signal(signal):
    """ 
    Normalize signal to the range -1 to 1 
    
    Parameters:
    signal (numpy.ndarray): The input audio signal.
    
    Returns:
    numpy.ndarray: The normalized audio signal.
    """
    return signal / np.max(np.abs(signal))

def upscale(
        input_file_path,
        output_file_path,
        source_format,
        target_format='flac',
        max_iterations=800,
        threshold_value=0.6,
        target_bitrate_kbps=1411
    ):
    """
    Main function to upscale an audio file to a specified format with optional processing.
    
    Parameters:
    input_file_path (str): Path to the input audio file.
    output_file_path (str): Path to the output processed audio file.
    source_format (str): Format of the input audio file (e.g., 'mp3', 'wav', 'ogg', 'flac').
    target_format (str): Format of the output audio file (e.g., 'flac', 'wav').
    max_iterations (int): Maximum number of iterations for IST.
    threshold_value (float): Threshold value for IST.
    target_bitrate_kbps (int): Target bitrate in kbps (must be within valid range for the target format).
    """
    valid_bitrate_ranges = {
        'flac': (800, 1411),
        'wav': (800, 6444),
    }
    
    if target_format not in valid_bitrate_ranges:
        raise ValueError(f"Unsupported target format: {target_format}")
    
    min_bitrate, max_bitrate = valid_bitrate_ranges[target_format]
    
    if not (min_bitrate <= target_bitrate_kbps <= max_bitrate):
        raise ValueError(f"{target_format.upper()} bitrate out of range. Please provide a value between {min_bitrate} and {max_bitrate} kbps.")
    
    logger.info(f"Loading {source_format.upper()} file...")
    sample_rate, samples, bitrate, audio = read_audio(input_file_path, format=source_format)
    if bitrate:
        logger.info(f"Original {source_format.upper()} bitrate: {bitrate / 1000:.2f} kbps")
    
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    
    target_bitrate = target_bitrate_kbps * 1000
    upscale_factor = round(target_bitrate / bitrate) if bitrate else 4
    logger.info(f"Upscale factor set to: {upscale_factor}")

    if samples.ndim == 1:
        logger.info("Mono channel detected.")
        channels = samples[:, np.newaxis]
    else:
        logger.info("Stereo channels detected.")
        channels = samples

    logger.info("Upscaling and processing channels...")
    upscaled_channels = upscale_channels(
        channels,
        upscale_factor=upscale_factor,
        max_iter=max_iterations,
        threshold=threshold_value
    )
    
    logger.info("Auto-scaling amplitudes based on original audio...")
    scaled_upscaled_channels = []
    for i, channel in enumerate(channels.T):
        scaled_channel = normalize_signal(upscaled_channels[:, i]) * np.max(np.abs(channel))
        scaled_upscaled_channels.append(scaled_channel)
    scaled_upscaled_channels = np.column_stack(scaled_upscaled_channels)

    logger.info("Normalizing audio...")
    normalized_upscaled_channels = []
    for i in range(scaled_upscaled_channels.shape[1]):
        normalized_channel = normalize_signal(scaled_upscaled_channels[:, i])
        normalized_upscaled_channels.append(normalized_channel)
    normalized_upscaled_channels = np.column_stack(normalized_upscaled_channels)

    new_sample_rate = sample_rate * upscale_factor
    write_audio(output_file_path, new_sample_rate, normalized_upscaled_channels, target_format)
    logger.info(f"Saved processed {target_format.upper()} file at {output_file_path}")
