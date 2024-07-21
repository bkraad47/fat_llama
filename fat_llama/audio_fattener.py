import numpy as np
import cupy as cp
from pydub import AudioSegment
import soundfile as sf
import os
from mutagen.mp3 import MP3
from scipy.signal import butter, lfilter, sosfilt, wiener
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_mp3(file_path):
    """ 
    Read an MP3 file and return the sample rate and data as a NumPy array.
    
    Parameters:
    - file_path (str): Path to the input MP3 file.
    
    Returns:
    - sample_rate (int): Sample rate of the audio.
    - samples (np.ndarray): Audio samples.
    - bitrate (int): Bitrate of the MP3 file.
    - audio (AudioSegment): Pydub AudioSegment object.
    """
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} not found.")
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
    Write data to a FLAC file.
    
    Parameters:
    - file_path (str): Path to the output FLAC file.
    - sample_rate (int): Sample rate of the audio.
    - data (np.ndarray): Audio data to write.
    """
    sf.write(file_path, data.astype(np.float32), sample_rate, format='FLAC', subtype='PCM_24')

def new_interpolation_algorithm(data, upscale_factor):
    """ 
    Interpolate data with scaled changes around center points.
    
    Parameters:
    - data (np.ndarray): Original audio data.
    - upscale_factor (int): Factor by which to upscale the data.
    
    Returns:
    - expanded_data (cp.ndarray): Interpolated audio data.
    """
    original_length = len(data)
    expanded_length = original_length * upscale_factor
    expanded_data = cp.zeros(expanded_length, dtype=cp.float32)
    
    for i in range(original_length):
        center_point = data[i]
        for j in range(upscale_factor):
            index = i * upscale_factor + j
            expanded_data[index] = center_point
    
    return expanded_data

def initialize_ist(data, threshold):
    """ 
    Initialize IST variables.
    
    Parameters:
    - data (cp.ndarray): Audio data.
    - threshold (float): Threshold value for IST.
    
    Returns:
    - data_thres (cp.ndarray): Thresholded data.
    """
    mask = cp.abs(data) > threshold
    data_thres = cp.where(mask, data, 0)
    return data_thres

def iterative_soft_thresholding(data, max_iter, threshold):
    """ 
    Perform IST on data using CuPy and cuFFT.
    
    Parameters:
    - data (cp.ndarray): Audio data.
    - max_iter (int): Maximum number of iterations for IST.
    - threshold (float): Threshold value for IST.
    
    Returns:
    - data_thres (cp.ndarray): IST processed data.
    """
    data_thres = initialize_ist(data, threshold)
    for _ in range(max_iter):
        data_fft = cp.fft.fft(data_thres)
        mask = cp.abs(data_fft) > threshold
        data_fft_thres = cp.where(mask, data_fft, 0)
        data_thres = cp.fft.ifft(data_fft_thres).real
    return data_thres

def upscale_channels(channels, upscale_factor, max_iter, threshold, apply_ist=True):
    """ 
    Process and upscale channels using the new interpolation and IST algorithms.
    
    Parameters:
    - channels (np.ndarray): Original audio channels.
    - upscale_factor (int): Factor by which to upscale the data.
    - max_iter (int): Maximum number of iterations for IST.
    - threshold (float): Threshold value for IST.
    - apply_ist (bool): Whether to apply IST or not.
    
    Returns:
    - processed_channels (cp.ndarray): Upscaled and processed audio channels.
    """
    processed_channels = []
    for channel in channels.T:
        # Interpolate data with the new algorithm
        logger.info("Interpolating data...")
        expanded_channel = new_interpolation_algorithm(cp.array(channel, dtype=cp.float32), upscale_factor)
        
        if apply_ist:
            logger.info("Performing IST...")
            # Apply IST algorithm
            ist_changes = iterative_soft_thresholding(expanded_channel, max_iter, threshold)
            expanded_channel = expanded_channel + ist_changes
        
        processed_channels.append(expanded_channel)
    
    return cp.column_stack(processed_channels)

def normalize_signal(signal):
    """ 
    Normalize signal to the range -1 to 1.
    
    Parameters:
    - signal (cp.ndarray): Audio signal.
    
    Returns:
    - normalized_signal (cp.ndarray): Normalized audio signal.
    """
    return signal / cp.max(cp.abs(signal))

def scale_amplitude(original, upscaled, gain_factor):
    """ 
    Scale the amplitude of the upscaled channels to match the original and apply gain factor.
    
    Parameters:
    - original (cp.ndarray): Original audio data.
    - upscaled (cp.ndarray): Upscaled audio data.
    - gain_factor (float): Gain factor to apply.
    
    Returns:
    - scaled_upscaled (cp.ndarray): Scaled upscaled audio data.
    """
    normalized_original = normalize_signal(original)
    normalized_upscaled = normalize_signal(upscaled)
    
    scale_factor = cp.max(cp.abs(normalized_original)) / cp.max(cp.abs(normalized_upscaled))
    scaled_upscaled = normalized_upscaled * scale_factor * gain_factor
    
    return scaled_upscaled

def apply_gain_reduction(data, sample_rate, reduction_profile):
    """ 
    Apply gain reduction based on the given reduction profile.
    
    Parameters:
    - data (cp.ndarray): Audio data.
    - sample_rate (int): Sample rate of the audio.
    - reduction_profile (list): List of tuples specifying the reduction profile.
    
    Returns:
    - reduced_data (cp.ndarray): Gain reduced audio data.
    """
    nyquist = 0.5 * sample_rate
    reduced_data = cp.copy(data)

    for (lowcut, highcut, reduction_factor) in reduction_profile:
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(2, [low, high], btype='band', output='sos')
        reduced_data += sosfilt(sos, data) * reduction_factor
    
    return reduced_data

def equalize_audio(data, sample_rate, lowcut, highcut):
    """ 
    Apply a bandpass filter to the audio data.
    
    Parameters:
    - data (cp.ndarray): Audio data.
    - sample_rate (int): Sample rate of the audio.
    - lowcut (float): Low cut frequency for the filter.
    - highcut (float): High cut frequency for the filter.
    
    Returns:
    - equalized_data (cp.ndarray): Equalized audio data.
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    y = lfilter(b, a, cp.asnumpy(data))  # lfilter works on numpy arrays
    return cp.array(y, dtype=cp.float32)

def apply_wiener_filter(data):
    """ 
    Apply Wiener filter to the audio data.
    
    Parameters:
    - data (cp.ndarray): Audio data.
    
    Returns:
    - wiener_data (cp.ndarray): Wiener filtered audio data.
    """
    return cp.array(wiener(cp.asnumpy(data)), dtype=cp.float32)

def upscale_mp3_to_flac(
        input_file_path,
        output_file_path_processed,
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
        output_file_path_no_processing=None,
        use_wiener_filter=False):
    """
    Upscale an MP3 file to FLAC with optional processing.
    
    Parameters:
    - input_file_path (str): Path to the input MP3 file. Mandatory.
    - output_file_path_processed (str): Path to the output processed FLAC file. Mandatory.
    - max_iterations (int): Number of iterations for IST. Default is 400.
    - threshold_value (float): Threshold value for IST. Default is 0.4.
    - gain_factor (float): Gain factor for scaling amplitude. Default is 22.8.
    - reduction_profile (list): Profile for gain reduction. Default is [(5, 140, -28.4), (1000, 10000, 26.4)].
    - lowcut (float): Low cut frequency for equalizer. Default is 5.0.
    - highcut (float): High cut frequency for equalizer. Default is 150000.0.
    - target_bitrate_kbps (int): Target bitrate in kbps. Default is 1400.
    - output_file_path_no_processing (str): Path to the output upscaled (no processing) FLAC file. Default is None.
    - use_wiener_filter (bool): Flag to use Wiener filter. Default is False.
    """
    if target_bitrate_kbps < 800 or target_bitrate_kbps > 1400:
        logger.error("Target bitrate must be between 800 and 1400 kbps.")
        raise ValueError("Target bitrate must be between 800 and 1400 kbps.")
    
    logger.info("Loading MP3 file...")
    sample_rate, samples, bitrate, audio = read_mp3(input_file_path)
    logger.info(f"Original MP3 bitrate: {bitrate / 1000:.2f} kbps")

    # Convert audio back to samples
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))

    # Calculate upscale factor to get closest to the target bitrate
    target_bitrate = target_bitrate_kbps * 1000  # Convert kbps to bits per second
    upscale_factor = round(target_bitrate / bitrate)
    logger.info(f"Upscale factor set to: {upscale_factor}")

    # Separate channels
    if samples.ndim == 1:  # Mono
        logger.info("Mono channel detected.")
        channels = samples[:, np.newaxis]
    else:  # Stereo
        logger.info("Stereo channels detected.")
        channels = samples

    # Upscale channels without processing
    if output_file_path_no_processing is not None:
        logger.info("Upscaling channels without processing...")
        upscaled_channels_no_processing = upscale_channels(channels, upscale_factor=upscale_factor, max_iter=max_iterations, threshold=threshold_value, apply_ist=False)

        # Write to FLAC with increased sample rate
        new_sample_rate = sample_rate * upscale_factor
        write_flac(output_file_path_no_processing, new_sample_rate, upscaled_channels_no_processing.get())
        logger.info(f"Saved upscaled (no processing) FLAC file at {output_file_path_no_processing}")

    # Upscale and process channels
    logger.info("Upscaling and processing channels...")
    upscaled_channels = upscale_channels(channels, upscale_factor=upscale_factor, max_iter=max_iterations, threshold=threshold_value)

    # Scale amplitudes of upscaled channels to match original channels and apply gain factor
    logger.info("Scaling amplitudes...")
    scaled_upscaled_channels = []
    for i, channel in enumerate(channels.T):
        scaled_channel = scale_amplitude(cp.array(channel, dtype=cp.float32), upscaled_channels[:, i], gain_factor=gain_factor)
        scaled_upscaled_channels.append(scaled_channel)
    scaled_upscaled_channels = cp.column_stack(scaled_upscaled_channels)

    # Apply frequency-specific gain reduction
    logger.info("Applying gain reduction...")
    gain_reduced_channels = []
    for i in range(scaled_upscaled_channels.shape[1]):
        gain_reduced_channel = apply_gain_reduction(scaled_upscaled_channels[:, i], sample_rate * upscale_factor, reduction_profile)
        gain_reduced_channels.append(gain_reduced_channel)
    gain_reduced_channels = cp.column_stack(gain_reduced_channels)

    # Normalize and equalize the upscaled channels
    logger.info("Normalizing and equalizing...")
    normalized_upscaled_channels = []
    for i in range(gain_reduced_channels.shape[1]):
        normalized_channel = normalize_signal(gain_reduced_channels[:, i])
        equalized_channel = equalize_audio(normalized_channel, sample_rate * upscale_factor, lowcut=lowcut, highcut=highcut)
        if use_wiener_filter:
            equalized_channel = apply_wiener_filter(equalized_channel)
        normalized_upscaled_channels.append(equalized_channel)
    normalized_upscaled_channels = cp.column_stack(normalized_upscaled_channels)

    # Write to FLAC with increased sample rate
    new_sample_rate = sample_rate * upscale_factor
    write_flac(output_file_path_processed, new_sample_rate, normalized_upscaled_channels.get())
    logger.info(f"Saved processed FLAC file at {output_file_path_processed}")
