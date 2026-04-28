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

    # Disable dynamic range compression in the FFMPEG decoder.
    extra_params = ["-drc_scale", "0"]
    audio = AudioSegment.from_file(file_path, format=format, parameters=extra_params)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
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
        bitrate = (len(samples) * 8) / duration_seconds

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
    Bandlimited FFT-based (sinc) interpolation.

    Upsamples by zero-padding the spectrum in the frequency domain rather than
    doing zero-order hold. Zero-order hold creates spectral images (ghosts)
    mirrored around the original Nyquist frequency, which appear as an empty
    band and an inverted-looking spectrum above ~fs/2 in the upscaled file.
    This implementation avoids that entirely.

    Parameters:
    data (cp.ndarray): The input audio data (1-D).
    upscale_factor (int): The integer factor by which to upscale.

    Returns:
    cp.ndarray: The upscaled audio data of length len(data) * upscale_factor.
    """
    data = cp.asarray(data, dtype=cp.float64)
    original_length = int(len(data))
    if upscale_factor == 1:
        return data.copy()

    expanded_length = original_length * int(upscale_factor)

    # rFFT of the original signal -> compact one-sided spectrum.
    spectrum = cp.fft.rfft(data)

    # Build the zero-padded one-sided spectrum for the upsampled length.
    new_spectrum_length = expanded_length // 2 + 1
    padded_spectrum = cp.zeros(new_spectrum_length, dtype=spectrum.dtype)
    padded_spectrum[:spectrum.shape[0]] = spectrum

    # If the original length is even, the Nyquist bin is real-valued and shared;
    # halving it preserves energy symmetry after zero-padding.
    if original_length % 2 == 0 and spectrum.shape[0] > 0:
        padded_spectrum[spectrum.shape[0] - 1] *= 0.5

    # Inverse rFFT and compensate amplitude for the longer transform.
    expanded_data = cp.fft.irfft(padded_spectrum, n=expanded_length) * upscale_factor

    return expanded_data.astype(cp.float64)

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

    return data_thres

# Block-parallel CUDA kernel for LMS adaptive filtering.
#
# True LMS is a sequential recurrence (each sample's weight update depends on
# the previous one), so it can't be parallelised across all samples without
# changing the algorithm. The previous implementation respected that by
# launching the whole loop on a single GPU thread -- correct but unusably slow
# (tens of minutes on a laptop 3060 for a 15 s clip), because one CUDA core is
# slower than a single CPU core and we were leaving 99.9% of the GPU idle.
#
# This version uses *block-LMS*: the signal is split into independent chunks,
# each chunk is processed sequentially by one thread, and chunks run in
# parallel. Weight state resets at chunk boundaries. For audio enhancement
# this is a standard, well-behaved approximation -- the filter just re-adapts
# at each boundary, which is inaudible at chunk sizes of a few thousand
# samples.
_LMS_BLOCK_KERNEL = cp.RawKernel(
    r'''
    extern "C" __global__
    void lms_block_kernel(const double* signal, const double* desired,
                          double* filtered, double mu, int num_taps,
                          long long n, long long block_size) {
        long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        long long start = tid * block_size;
        if (start >= n) return;
        long long end = start + block_size;
        if (end > n) end = n;

        // Per-thread filter coefficients (kept in local memory).
        double w[256];
        for (int t = 0; t < num_taps; ++t) w[t] = 0.0;

        long long i0 = start < (long long)num_taps ? (long long)num_taps : start;
        for (long long i = i0; i < end; ++i) {
            double y = 0.0;
            for (int t = 0; t < num_taps; ++t) {
                y += w[t] * signal[i - t];
            }
            double e = desired[i] - y;
            double step = 2.0 * mu * e;
            for (int t = 0; t < num_taps; ++t) {
                double nw = w[t] + step * signal[i - t];
                if (nw >  1e10) nw =  1e10;
                if (nw < -1e10) nw = -1e10;
                w[t] = nw;
            }
            filtered[i] = y;
        }
    }
    ''',
    'lms_block_kernel',
)


def lms_filter(signal, desired, mu=0.001, num_taps=32, block_size=4096):
    """
    Apply a block-parallel LMS adaptive filter on the GPU.

    The signal is divided into independent blocks; each block is filtered
    sequentially by one CUDA thread, and blocks execute in parallel across
    the device. Filter weights reset at block boundaries (block-LMS).

    Parameters:
    signal (cp.ndarray): The input audio signal (float64, 1-D).
    desired (cp.ndarray): The desired output signal (float64, 1-D, same length).
    mu (float): The step size for the adaptive filter.
    num_taps (int): The number of filter taps. Must be <= 256.
    block_size (int): Samples processed per CUDA thread. Larger = fewer block
        boundaries but less parallelism. 4096 is a good balance for music.

    Returns:
    cp.ndarray: The filtered audio signal.
    """
    if num_taps > 256:
        raise ValueError("num_taps must be <= 256 for the LMS CUDA kernel.")
    if block_size <= num_taps:
        raise ValueError("block_size must be greater than num_taps.")

    signal = cp.ascontiguousarray(signal, dtype=cp.float64)
    desired = cp.ascontiguousarray(desired, dtype=cp.float64)
    n = int(signal.shape[0])
    filtered_signal = cp.zeros(n, dtype=cp.float64)

    num_blocks = (n + block_size - 1) // block_size
    threads_per_block = 64
    grid = (num_blocks + threads_per_block - 1) // threads_per_block

    _LMS_BLOCK_KERNEL(
        (grid,), (threads_per_block,),
        (signal, desired, filtered_signal,
         cp.float64(mu), cp.int32(num_taps),
         cp.int64(n), cp.int64(block_size)),
    )
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


# ----- AudioSR (ML-based bandwidth extension / super resolution) ---------------

# AudioSR operates internally at 48 kHz. It is a heavy optional dependency, so
# it is imported lazily inside ``apply_audiosr`` and only when the caller asks
# for it via ``toggle_audiosr=True``.
AUDIOSR_TARGET_SR = 48000


def _resample_linear(signal_np, src_sr, dst_sr):
    """
    Lightweight CPU resampler used to feed/receive audio from AudioSR.

    Uses ``numpy.interp`` so we don't add another hard dependency. Quality is
    sufficient here because AudioSR re-synthesises the high band itself.
    """
    if src_sr == dst_sr or signal_np.size == 0:
        return signal_np.astype(np.float64, copy=False)
    duration = signal_np.shape[0] / float(src_sr)
    src_t = np.linspace(0.0, duration, num=signal_np.shape[0], endpoint=False)
    dst_n = int(round(duration * dst_sr))
    dst_t = np.linspace(0.0, duration, num=dst_n, endpoint=False)
    return np.interp(dst_t, src_t, signal_np).astype(np.float64)


def apply_audiosr(
        samples_np,
        sample_rate,
        model_name='basic',
        ddim_steps=50,
        guidance_scale=3.5,
        seed=42,
        device=None,
):
    """
    Run pretrained AudioSR diffusion super-resolution on the given audio.

    AudioSR (Liu et al., "AudioSR: Versatile Audio Super-Resolution at Scale")
    is a latent-diffusion model that hallucinates plausible high-frequency
    content beyond the input bandwidth, producing 48 kHz output. This is the
    only stage in the pipeline capable of *creating* spectral content above the
    source's Nyquist (e.g. above the ~16 kHz MP3 lowpass).

    Parameters:
    samples_np (np.ndarray): Audio as float64, shape (N,) mono or (N, C) multi-channel.
    sample_rate (int): Input sample rate.
    model_name (str): AudioSR model variant ('basic' or 'speech').
    ddim_steps (int): Number of diffusion sampling steps. Higher = better, slower.
    guidance_scale (float): Classifier-free guidance scale.
    seed (int): RNG seed for reproducibility.
    device (str | None): Torch device override (e.g. 'cuda', 'cpu'). If None,
        AudioSR auto-selects.

    Returns:
    (np.ndarray, int): Processed audio at 48 kHz, same channel layout as input.
    """
    try:
        # Local imports so the package remains usable without these heavy deps.
        import torch  # noqa: F401  (audiosr requires it; surface the error early)
        from audiosr import build_model, super_resolution
    except ImportError as exc:
        raise ImportError(
            "AudioSR is not installed. Install it with `pip install audiosr` "
            "(also requires torch). Set toggle_audiosr=False to skip this stage."
        ) from exc

    import tempfile

    # Normalise input to (N, C) float64 for per-channel processing.
    if samples_np.ndim == 1:
        channels = samples_np[:, np.newaxis]
    else:
        channels = samples_np

    # AudioSR's public API consumes a file path, so each channel is written to
    # a temp WAV at AUDIOSR_TARGET_SR and read back.
    logger.info("Building AudioSR model (%s)...", model_name)
    audiosr_model = build_model(model_name=model_name, device=device)

    processed = []
    for ch_idx in range(channels.shape[1]):
        ch = channels[:, ch_idx]
        # AudioSR expects 48 kHz mono input internally; resample first.
        ch_48k = _resample_linear(ch, sample_rate, AUDIOSR_TARGET_SR)

        # Peak-normalise to avoid clipping during model write.
        peak = float(np.max(np.abs(ch_48k))) or 1.0
        ch_48k_norm = (ch_48k / peak).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, ch_48k_norm, AUDIOSR_TARGET_SR, subtype='PCM_16')

            logger.info(
                "Running AudioSR super-resolution on channel %d/%d (steps=%d)...",
                ch_idx + 1, channels.shape[1], ddim_steps,
            )
            sr_waveform = super_resolution(
                audiosr_model,
                tmp_path,
                seed=seed,
                guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                latent_t_per_second=12.8,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # AudioSR returns shape (1, 1, T) or (1, T); flatten to 1-D and rescale.
        sr_arr = np.asarray(sr_waveform).squeeze().astype(np.float64)
        sr_peak = float(np.max(np.abs(sr_arr))) or 1.0
        sr_arr = (sr_arr / sr_peak) * peak
        processed.append(sr_arr)

    # Pad/truncate to a common length (per-channel runs can differ by 1 sample).
    min_len = min(p.shape[0] for p in processed)
    processed = [p[:min_len] for p in processed]
    out = np.column_stack(processed) if len(processed) > 1 else processed[0]
    return out, AUDIOSR_TARGET_SR


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
        toggle_adaptive_filter=True,
        toggle_audiosr=False,
        audiosr_model='basic',
        audiosr_ddim_steps=50,
        audiosr_guidance_scale=3.5,
        audiosr_seed=42,
        audiosr_device=None,
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
    toggle_audiosr (bool): If True, run pretrained AudioSR diffusion super-resolution
        before the sinc/IST stage to synthesise plausible content above the source's
        Nyquist (e.g. above the ~16 kHz MP3 lowpass). Requires `audiosr` and `torch`.
        Defaults to False.
    audiosr_model (str): AudioSR model variant ('basic' or 'speech'). Defaults to 'basic'.
    audiosr_ddim_steps (int): Diffusion sampling steps. Defaults to 50.
    audiosr_guidance_scale (float): Classifier-free guidance scale. Defaults to 3.5.
    audiosr_seed (int): Random seed for AudioSR. Defaults to 42.
    audiosr_device (str | None): Torch device override (e.g. 'cuda', 'cpu'). If None,
        AudioSR auto-selects.
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

    # Optional: pretrained ML super-resolution to synthesise high-frequency
    # content that the source codec discarded. Runs on the raw NumPy samples
    # before the CuPy/sinc pipeline; output replaces ``samples``/``sample_rate``.
    if toggle_audiosr:
        logger.info("Running AudioSR ML super-resolution stage...")
        # ``samples`` from read_audio is integer PCM scaled to its dtype range;
        # convert to a normalised float view for the model.
        samples_float = np.asarray(samples, dtype=np.float64)
        peak = float(np.max(np.abs(samples_float))) or 1.0
        samples_float = samples_float / peak

        sr_audio, sr_rate = apply_audiosr(
            samples_float,
            sample_rate=sample_rate,
            model_name=audiosr_model,
            ddim_steps=audiosr_ddim_steps,
            guidance_scale=audiosr_guidance_scale,
            seed=audiosr_seed,
            device=audiosr_device,
        )

        # Restore original peak amplitude scale so downstream autoscale logic
        # remains meaningful relative to the source.
        samples = (sr_audio * peak).astype(np.float64)
        sample_rate = sr_rate
        logger.info("AudioSR output sample rate: %d Hz", sample_rate)

    samples = cp.array(samples, dtype=cp.float64)
    if audio.channels == 2 and samples.ndim == 1:
        samples = samples.reshape((-1, 2))

    # Determine the upscale factor.
    #
    # Previously this was ``round(target_bitrate / source_bitrate)``, which is
    # dimensionally wrong: a *bitrate* ratio was being used as a *sample-rate*
    # multiplier. For a typical 128 kbps MP3 with ``target_bitrate_kbps=1411``
    # that produced factor ~11 -> 485 kHz output, and at 900 kbps produced
    # factor ~7 -> ~309 kHz output (with an apparent FLAC bitrate over
    # 5000 kbps). We now derive the factor from a sane target sample rate
    # picked from the requested bitrate tier, and clamp it to [1, 4].
    if target_bitrate_kbps >= 1400:
        target_sr = 192000  # hi-res tier
    elif target_bitrate_kbps >= 1100:
        target_sr = 96000
    else:
        target_sr = 88200   # 2x of 44.1 kHz / close-to-2x of 48 kHz
    upscale_factor = max(1, min(round(target_sr / sample_rate), 4))
    logger.info(
        f"Upscale factor set to: {upscale_factor} "
        f"(target ~{target_sr} Hz, source {sample_rate} Hz)"
    )

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
