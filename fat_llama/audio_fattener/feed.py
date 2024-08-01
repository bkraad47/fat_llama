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
    Read an audio file and return the sample rate and data as a NumPy array.
    
    Parameters:
    file_path (str): The path to the input audio file.
    format (str): The format of the input audio file (e.g., 'mp3', 'flac', 'ogg', 'wav').
    
    Returns:
    sample_rate (int): The sample rate of the audio file.
    samples (np.ndarray): The audio samples.
    bitrate (int): The bitrate of the audio file.
    audio (AudioSegment): The audio segment object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    # Define extra parameters for FFMPEG
    extra_params = ["-drc_scale", "0"]
    
    # Load the audio file with specified format and extra parameters
    audio = AudioSegment.from_file(file_path, format=format, parameters=extra_params)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
    sample_rate = audio.frame_rate
    bitrate = None
    
    # Retrieve bitrate information based on file format
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
        # Calculate bitrate for other formats
        duration_seconds = len(audio) / 1000.0
        bitrate = (len(samples) * 8) / duration_seconds
    
    # Reshape samples if the audio has two channels
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    
    return sample_rate, samples, bitrate, audio

def write_audio(file_path, sample_rate, data, format):
    """
    Write data to an audio file.
    
    Parameters:
    file_path (str): The path to the output audio file.
    sample_rate (int): The sample rate of the audio.
    data (np.ndarray): The audio data to write.
    format (str): The format of the output audio file (e.g., 'flac', 'wav').
    """
    if format == 'flac':
        sf.write(file_path, data.astype(np.float64), sample_rate, format='FLAC', subtype='PCM_24')
    elif format == 'wav':
        sf.write(file_path, data.astype(np.float64), sample_rate, format='WAV', subtype='PCM_24')
    else:
        raise ValueError(f"Unsupported target format: {format}")

def new_interpolation_algorithm(data, upscale_factor):
    """
    Interpolate data with scaled changes around center points.
    
    Parameters:
    data (cp.ndarray): The input audio data.
    upscale_factor (int): The factor by which to upscale the audio data.
    
    Returns:
    cp.ndarray: The upscaled audio data.
    """
    original_length = len(data)
    expanded_length = original_length * upscale_factor
    expanded_data = cp.zeros(expanded_length, dtype=cp.float64)
    
    # Duplicate each data point upscale_factor times
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
    data (cp.ndarray): The input audio data.
    threshold (float): The threshold value for IST.
    
    Returns:
    cp.ndarray: The thresholded audio data.
    """
    mask = cp.abs(data) > threshold
    data_thres = cp.where(mask, data, 0)
    return data_thres

def iterative_soft_thresholding(data, max_iter, threshold):
    """
    Perform IST on data using CuPy and cuFFT.
    
    Parameters:
    data (cp.ndarray): The input audio data.
    max_iter (int): The maximum number of iterations for IST.
    threshold (float): The threshold value for IST.
    
    Returns:
    cp.ndarray: The processed audio data after IST.
    """
    data_thres = initialize_ist(data, threshold)
    for _ in range(max_iter):
        data_fft = cp.fft.fft(data_thres)
        mask = cp.abs(data_fft) > threshold
        data_fft_thres = cp.where(mask, data_fft, 0)
        data_thres = cp.fft.ifft(data_fft_thres).real
        
        # Harmonic reconstruction
        harmonics = cp.sin(cp.linspace(0, 2 * cp.pi, len(data_thres)))
        data_thres += 0.1 * harmonics

    return data_thres

def lms_filter(signal, desired, mu=0.001, num_taps=32):
    """
    Apply an LMS adaptive filter using CuPy.
    
    Parameters:
    signal (cp.ndarray): The input audio signal.
    desired (cp.ndarray): The desired output signal.
    mu (float): The step size for the adaptive filter.
    num_taps (int): The number of filter taps.
    
    Returns:
    cp.ndarray: The filtered audio signal.
    """
    n = len(signal)
    w = cp.zeros(num_taps, dtype=cp.float64)
    filtered_signal = cp.zeros(n, dtype=cp.float64)

    for i in range(num_taps, n):
        # Extract a vector of the last 'num_taps' samples from the input signal
        x = signal[i:i-num_taps:-1]
        
        # Compute the filter output by taking the dot product of filter coefficients and input vector
        y = cp.dot(w, x)
        
        # Compute the error between the desired output and the filter output
        e = desired[i] - y
        
        # Update the filter coefficients using the LMS rule
        w += 2 * mu * e * x
        
        # Ensure the coefficients remain finite to avoid numerical issues
        w = cp.clip(w, -1e10, 1e10)
        
        # Store the filter output in the filtered signal
        filtered_signal[i] = y

    return filtered_signal

def upscale_channels(channels, upscale_factor, max_iter, threshold):
    """
    Process and upscale channels using the new interpolation and IST algorithms.
    
    Parameters:
    channels (cp.ndarray): The input audio channels.
    upscale_factor (int): The factor by which to upscale the audio data.
    max_iter (int): The maximum number of iterations for IST.
    threshold (float): The threshold value for IST.
    
    Returns:
    cp.ndarray: The upscaled and processed audio channels.
    """
    processed_channels = []
    for channel in channels.T:
        logger.info("Interpolating data...")
        expanded_channel = new_interpolation_algorithm(channel, upscale_factor)
        
        logger.info("Performing IST...")
        ist_changes = iterative_soft_thresholding(expanded_channel, max_iter, threshold)
        expanded_channel = expanded_channel.astype(cp.float64) + ist_changes

        processed_channels.append(expanded_channel)
    
    return cp.column_stack(processed_channels)

def normalize_signal(signal):
    """
    Normalize signal to the range -1 to 1.
    
    Parameters:
    signal (cp.ndarray): The input audio signal.
    
    Returns:
    cp.ndarray: The normalized audio signal.
    """
    return signal / cp.max(cp.abs(signal))

def upscale(
        input_file_path,
        output_file_path,
        source_format,
        target_format='flac',
        max_iterations=300,
        threshold_value=0.6,
        target_bitrate_kbps=1411,
        toggle_normalize=True,
        toggle_autoscale=True,
        toggle_adaptive_filter=True
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
    toggle_normalize (bool): Whether to normalize the audio. Defaults to True.
    toggle_autoscale (bool): Whether to autoscale the audio based on the original audio. Defaults to True.
    toggle_adaptive_filter (bool): Whether to apply adaptive filtering. Defaults to True.
    """
    # Validate target bitrate
    valid_bitrate_ranges = {
        'flac': (800, 1411),
        'wav': (800, 6444),
    }
    
    if target_format not in valid_bitrate_ranges:
        raise ValueError(f"Unsupported target format: {target_format}")
    
    min_bitrate, max_bitrate = valid_bitrate_ranges[target_format]
    
    if not (min_bitrate <= target_bitrate_kbps <= max_bitrate):
        raise ValueError(f"{target_format.upper()} bitrate out of range. Please provide a value between {min_bitrate} and {max_bitrate} kbps.")
    
    # Read the input audio file
    logger.info(f"Loading {source_format.upper()} file...")
    sample_rate, samples, bitrate, audio = read_audio(input_file_path, format=source_format)
    if bitrate:
        logger.info(f"Original {source_format.upper()} bitrate: {bitrate / 1000:.2f} kbps")
    
    samples = cp.array(samples, dtype=cp.float64)
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    
    # Determine the upscale factor
    target_bitrate = target_bitrate_kbps * 1000
    upscale_factor = round(target_bitrate / bitrate) if bitrate else 4
    logger.info(f"Upscale factor set to: {upscale_factor}")

    # Process and upscale the audio channels
    if samples.ndim == 1:
        logger.info("Mono channel detected.")
        channels = samples[:, cp.newaxis]
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
    
    # Autoscale amplitudes if enabled
    if toggle_autoscale:
        logger.info("Auto-scaling amplitudes based on original audio...")
        scaled_upscaled_channels = []
        for i, channel in enumerate(channels.T):
            scaled_channel = normalize_signal(upscaled_channels[:, i]) * cp.max(cp.abs(channel))
            scaled_upscaled_channels.append(scaled_channel)
        scaled_upscaled_channels = cp.column_stack(scaled_upscaled_channels)
    else:
        scaled_upscaled_channels = upscaled_channels

    # Normalize audio if enabled
    if toggle_normalize:
        logger.info("Normalizing audio...")
        normalized_upscaled_channels = []
        for i in range(scaled_upscaled_channels.shape[1]):
            normalized_channel = normalize_signal(scaled_upscaled_channels[:, i])
            normalized_upscaled_channels.append(normalized_channel)
        normalized_upscaled_channels = cp.column_stack(normalized_upscaled_channels)
    else:
        normalized_upscaled_channels = scaled_upscaled_channels

    # Apply adaptive filtering if enabled
    if toggle_adaptive_filter:
        logger.info("Applying adaptive filtering...")
        filtered_upscaled_channels = []
        for i in range(normalized_upscaled_channels.shape[1]):
            filtered_channel = lms_filter(normalized_upscaled_channels[:, i], normalized_upscaled_channels[:, i])
            filtered_upscaled_channels.append(filtered_channel)
        filtered_upscaled_channels = cp.column_stack(filtered_upscaled_channels)
    else:
        filtered_upscaled_channels = normalized_upscaled_channels

    # Write the processed audio to the output file
    new_sample_rate = sample_rate * upscale_factor
    write_audio(output_file_path, new_sample_rate, cp.asnumpy(filtered_upscaled_channels), target_format)
    logger.info(f"Saved processed {target_format.upper()} file at {output_file_path}")
