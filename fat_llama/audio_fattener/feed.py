import numpy as np
import cupy as cp
from pydub import AudioSegment
import soundfile as sf
import os
import logging
from mutagen.mp3 import MP3
from scipy.signal import butter, lfilter, sosfilt, wiener
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_mp3(file_path):
    """ 
    Read an MP3 file and return the sample rate and data as a NumPy array 
    
    Parameters:
    file_path (str): The path to the input MP3 file.
    
    Returns:
    int: Sample rate of the audio.
    numpy.ndarray: Audio samples.
    int: Bitrate of the MP3 file.
    AudioSegment: The audio segment object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    audio = AudioSegment.from_mp3(file_path)
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    mp3_info = MP3(file_path)
    bitrate = mp3_info.info.bitrate
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    return sample_rate, samples, bitrate, audio

def write_flac(file_path, sample_rate, data):
    """ 
    Write data to a FLAC file 
    
    Parameters:
    file_path (str): The path to the output FLAC file.
    sample_rate (int): The sample rate of the audio.
    data (numpy.ndarray): The audio data to write.
    """
    sf.write(file_path, data.astype(np.float32), sample_rate, format='FLAC', subtype='PCM_24')

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

def upscale_channels(channels, upscale_factor, max_iter, threshold, apply_ist=True):
    """ 
    Process and upscale channels using the new interpolation and IST algorithms 
    
    Parameters:
    channels (numpy.ndarray): The input audio channels.
    upscale_factor (int): The factor by which to upscale the audio data.
    max_iter (int): The maximum number of iterations for IST.
    threshold (float): The threshold value for IST.
    apply_ist (bool): Flag to apply IST or not.
    
    Returns:
    numpy.ndarray: The upscaled (and processed, if apply_ist=True) audio data.
    """
    processed_channels = []
    for channel in channels.T:
        logger.info("Interpolating data...")
        expanded_channel = new_interpolation_algorithm(channel, upscale_factor)
        
        if apply_ist:
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

def scale_amplitude(original, upscaled, gain_factor):
    """ 
    Scale the amplitude of the upscaled channels to match the original and apply gain factor 
    
    Parameters:
    original (numpy.ndarray): The original audio data.
    upscaled (numpy.ndarray): The upscaled audio data.
    gain_factor (float): The gain factor to apply.
    
    Returns:
    numpy.ndarray: The scaled upscaled audio data.
    """
    normalized_original = normalize_signal(original)
    normalized_upscaled = normalize_signal(upscaled)
    
    scale_factor = np.max(np.abs(normalized_original)) / np.max(np.abs(normalized_upscaled))
    scaled_upscaled = normalized_upscaled * scale_factor * gain_factor
    
    return scaled_upscaled

def apply_gain_reduction(data, sample_rate, reduction_profile):
    """ 
    Apply gain reduction based on the given reduction profile 
    
    Parameters:
    data (cupy.ndarray): The input audio data.
    sample_rate (int): The sample rate of the audio.
    reduction_profile (list of tuple): The reduction profile as a list of (lowcut, highcut, reduction_factor) tuples.
    
    Returns:
    numpy.ndarray: The gain-reduced audio data.
    """
    nyquist = 0.5 * sample_rate
    reduced_data = cp.asnumpy(data)

    for (lowcut, highcut, reduction_factor) in reduction_profile:
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(2, [low, high], btype='band', output='sos')
        reduced_data = reduced_data + sosfilt(sos, reduced_data) * reduction_factor
    
    return reduced_data

def equalize_audio(data, sample_rate, lowcut, highcut):
    """ 
    Apply a bandpass filter to the audio data 
    
    Parameters:
    data (numpy.ndarray): The input audio data.
    sample_rate (int): The sample rate of the audio.
    lowcut (float): The low cut frequency for the equalizer.
    highcut (float): The high cut frequency for the equalizer.
    
    Returns:
    numpy.ndarray: The equalized audio data.
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def apply_wiener_filter(data):
    """ 
    Apply Wiener filter to the audio data 
    
    Parameters:
    data (numpy.ndarray): The input audio data.
    
    Returns:
    numpy.ndarray: The audio data after applying the Wiener filter.
    """
    return wiener(data)

def upscale_mp3_to_flac(input_file_path, output_file_path_processed, max_iterations=400, threshold_value=0.4, gain_factor=22.8, reduction_profile=None, lowcut=5.0, highcut=150000.0, target_bitrate_kbps=1400, output_file_path_no_processing=None, use_wiener_filter=False):
    """
    Main function to upscale an MP3 file to FLAC format with optional processing.
    
    Parameters:
    input_file_path (str): Path to the input MP3 file.
    output_file_path_processed (str): Path to the output processed FLAC file.
    max_iterations (int): Maximum number of iterations for IST.
    threshold_value (float): Threshold value for IST.
    gain_factor (float): Gain factor for scaling amplitude.
    reduction_profile (list of tuple): Reduction profile for gain reduction.
    lowcut (float): Low cut frequency for equalizer.
    highcut (float): High cut frequency for equalizer.
    target_bitrate_kbps (int): Target bitrate in kbps (must be between 800 and 1400).
    output_file_path_no_processing (str, optional): Path to the output FLAC file without processing. Default is None.
    use_wiener_filter (bool): Flag to use Wiener filter or not. Default is False.
    """
    if reduction_profile is None:
        reduction_profile = [
            (5, 140, -28.4),
            (1000, 10000, 26.4),
        ]
    
    if not (800 <= target_bitrate_kbps <= 1400):
        raise ValueError("FLAC bitrate out of range. Please provide a value between 800 and 1400 kbps.")
    
    logger.info("Loading MP3 file...")
    sample_rate, samples, bitrate, audio = read_mp3(input_file_path)
    logger.info(f"Original MP3 bitrate: {bitrate / 1000:.2f} kbps")

    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))

    target_bitrate = target_bitrate_kbps * 1000
    upscale_factor = round(target_bitrate / bitrate)
    logger.info(f"Upscale factor set to: {upscale_factor}")

    if samples.ndim == 1:
        logger.info("Mono channel detected.")
        channels = samples[:, np.newaxis]
    else:
        logger.info("Stereo channels detected.")
        channels = samples

    if output_file_path_no_processing is not None:
        logger.info("Upscaling channels without processing...")
        upscaled_channels_no_processing = upscale_channels(channels, upscale_factor=upscale_factor, max_iter=max_iterations, threshold=threshold_value, apply_ist=False)
        new_sample_rate = sample_rate * upscale_factor
        write_flac(output_file_path_no_processing, new_sample_rate, upscaled_channels_no_processing)
        logger.info(f"Saved upscaled (no processing) FLAC file at {output_file_path_no_processing}")

    logger.info("Upscaling and processing channels...")
    upscaled_channels = upscale_channels(channels, upscale_factor=upscale_factor, max_iter=max_iterations, threshold=threshold_value, apply_ist=True)
    
    logger.info("Scaling amplitudes...")
    scaled_upscaled_channels = []
    for i, channel in enumerate(channels.T):
        scaled_channel = scale_amplitude(channel, upscaled_channels[:, i], gain_factor=gain_factor)
        scaled_upscaled_channels.append(scaled_channel)
    scaled_upscaled_channels = np.column_stack(scaled_upscaled_channels)

    logger.info("Applying gain reduction...")
    gain_reduced_channels = []
    for i in range(scaled_upscaled_channels.shape[1]):
        gain_reduced_channel = apply_gain_reduction(cp.asarray(scaled_upscaled_channels[:, i]), sample_rate * upscale_factor, reduction_profile)
        gain_reduced_channels.append(cp.asnumpy(gain_reduced_channel))
    gain_reduced_channels = np.column_stack(gain_reduced_channels)

    logger.info("Normalizing and equalizing...")
    normalized_upscaled_channels = []
    for i in range(gain_reduced_channels.shape[1]):
        normalized_channel = normalize_signal(gain_reduced_channels[:, i])
        equalized_channel = equalize_audio(normalized_channel, sample_rate * upscale_factor, lowcut, highcut)
        if use_wiener_filter:
            equalized_channel = apply_wiener_filter(equalized_channel)
        normalized_upscaled_channels.append(equalized_channel)
    normalized_upscaled_channels = np.column_stack(normalized_upscaled_channels)

    new_sample_rate = sample_rate * upscale_factor
    write_flac(output_file_path_processed, new_sample_rate, normalized_upscaled_channels)
    logger.info(f"Saved processed FLAC file at {output_file_path_processed}")
