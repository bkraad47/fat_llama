import logging
import os

import cupy as cp
import numpy as np
import soundfile as sf
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE
from pydub import AudioSegment

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_audio(file_path, audio_format):
    """
    Read an audio file and return the sample rate and data as a NumPy array.

    Parameters:
    file_path (str): The path to the input audio file.
    audio_format (str): The format of the input audio file
        (e.g., 'mp3', 'flac', 'ogg', 'wav').

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
    audio = AudioSegment.from_file(
        file_path, format=audio_format, parameters=extra_params
    )
    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
    sample_rate = audio.frame_rate
    bitrate = None

    # Retrieve bitrate information based on file format
    if audio_format == 'mp3':
        mp3_info = MP3(file_path)
        bitrate = mp3_info.info.bitrate
    elif audio_format == 'flac':
        flac_info = FLAC(file_path)
        bitrate = flac_info.info.bitrate
    elif audio_format == 'ogg':
        ogg_info = OggVorbis(file_path)
        bitrate = ogg_info.info.bitrate
    elif audio_format == 'wav':
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


def write_audio(file_path, sample_rate, data, audio_format):
    """
    Write data to an audio file.

    Parameters:
    file_path (str): The path to the output audio file.
    sample_rate (int): The sample rate of the audio.
    data (np.ndarray): The audio data to write.
    audio_format (str): The format of the output audio file
        (e.g., 'flac', 'wav').
    """
    data = data.astype(np.float64)

    # soundfile expects floating-point input already scaled to [-1, 1] when
    # writing an integer PCM subtype; it silently clamps anything outside
    # that range instead of raising. read_audio() (and intermediate pipeline
    # stages) hand back data on the raw PCM/processing scale, not [-1, 1],
    # so peak-normalize here before writing to avoid clamping nearly every
    # sample to full scale. A zero/near-zero peak (silence) is left as-is to
    # avoid dividing by zero.
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak
    data = np.clip(data, -1.0, 1.0)

    if audio_format == 'flac':
        sf.write(
            file_path, data, sample_rate, format='FLAC', subtype='PCM_24'
        )
    elif audio_format == 'wav':
        sf.write(
            file_path, data, sample_rate, format='WAV', subtype='PCM_24'
        )
    else:
        raise ValueError(f"Unsupported target format: {audio_format}")


def new_interpolation_algorithm(data, upscale_factor):
    """
    Upsample a 1-D real signal via FFT-domain zero-padding (bandlimited /
    sinc interpolation).

    As of the cycle 3 fix, this replaces the prior zero-order-hold
    duplication (each sample repeated upscale_factor times), which was
    measured (audio-quality-checker, cycle 3) to inject strong mirrored
    spectral images at multiples of the original sample rate (e.g. near
    44.1/88.2/132.3 kHz for a 7x upscale of 44.1 kHz audio) rather than
    genuine added high-frequency detail -- confirmed to be zero-order-hold
    imaging, not reconstruction, because that energy sat exactly at
    predictable image frequencies with no dependence on the actual
    program content. Zero-order-hold also left iterative_soft_thresholding
    little headroom to add real detail: the ZOH-duplicated waveform
    shape dominated the signal, swamping IST's contribution.

    This implementation takes the real FFT of `data`, zero-pads the
    spectrum with additional high-frequency bins (all exactly zero, so no
    new spectral content is introduced), and inverse-FFTs back to a
    longer time-domain signal -- the standard Fourier/sinc method for
    bandlimited upsampling (the same technique used internally by e.g.
    `scipy.signal.resample`), computed here entirely with `cp.fft`
    (CuPy/CUDA), not scipy/numpy, to stay on the CUDA-only path. Measured
    (cycle 3): energy above the original Nyquist frequency drops from
    dominating the extended band (zero-order-hold) to ~1e-8 relative
    magnitude (FFT round-off) immediately after this step, leaving that
    band available for iterative_soft_thresholding to fill with
    genuinely reconstructed content instead of duplicate images.

    Parameters:
    data (cp.ndarray): The input audio data (single channel).
    upscale_factor (int): The factor by which to upscale the audio data.

    Returns:
    cp.ndarray: The upscaled audio data, band-limited to the original
        Nyquist frequency, length len(data) * upscale_factor.
    """
    data = data.astype(cp.float64)
    original_length = len(data)

    if upscale_factor == 1:
        return data.copy()

    expanded_length = original_length * upscale_factor

    spectrum = cp.fft.rfft(data)
    expanded_spectrum = cp.zeros(
        expanded_length // 2 + 1, dtype=cp.complex128
    )
    expanded_spectrum[:len(spectrum)] = spectrum

    expanded_data = cp.fft.irfft(expanded_spectrum, n=expanded_length)
    # irfft normalizes by the *output* length; rescale by upscale_factor
    # so the reconstructed waveform's amplitude matches the original
    # signal's amplitude instead of being attenuated by 1/upscale_factor.
    expanded_data *= upscale_factor

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
    threshold (float): The absolute threshold value for IST -- applied
        as-is to both `data`'s raw time-domain magnitude (via
        initialize_ist) and each iteration's raw FFT-bin magnitude, not
        scaled relative to `data`'s own amplitude.

    Known issue (investigated, not fixed as of cycle 3): `data` here is
    raw-PCM-scale (peak ~1e4-3e4), while `threshold`'s conventional
    default (0.6) is many orders of magnitude smaller. Measured (cycle 3,
    real ~15s input_test.mp3 channel): median FFT-bin magnitude ~9.3e4,
    so a threshold of 0.6 masks essentially nothing (only exact/
    near-zero bins) in both the time- and frequency-domain steps below --
    the "keep significant frequencies, discard noise" mechanism this
    function is meant to perform barely triggers at real audio's actual
    scale, leaving iteration_soft_thresholding to mostly perform
    near-lossless FFT/IFFT round trips rather than genuine sparse
    reconstruction. This cycle fixed the harmonic term's scale (see
    below) since that is IST's other, independently measurable source
    of added detail, but did not change `threshold`'s absolute-vs-
    relative semantics -- doing so changes default-value tuning and
    IST's masking behavior more broadly, and needs dedicated evaluation
    against real audio in its own cycle rather than folding it in here.

    Returns:
    cp.ndarray: The processed audio data after IST.
    """
    data_thres = initialize_ist(data, threshold)

    # The harmonic reconstruction term below is added to data_thres every
    # iteration, and data_thres carries forward from one iteration to the
    # next. Its FFT magnitude scales with array length, so it trivially
    # survives the fixed absolute `threshold` and is never removed by
    # masking -- left at a fixed per-iteration amplitude, its contribution
    # to the output would accumulate roughly linearly with max_iter instead
    # of converging (measured: max|output| grew from ~0.88 at max_iter=1 to
    # ~30.8 at max_iter=300, an unbounded, iteration-count-dependent rise).
    # Scaling by 1/max_iter keeps the *total* injected harmonic energy
    # constant regardless of how many iterations run -- identical behavior
    # to before at max_iter=1, but no longer scaling with iteration count --
    # without changing the FFT/IST method itself.
    #
    # As of the cycle 3 fix, that total is also scaled by `data`'s own peak
    # amplitude instead of being a fixed absolute constant (0.1). `data`
    # here is the already-interpolated, raw-PCM-scale channel (peak on the
    # order of 1e4-3e4 for 16-bit-sourced audio), not a normalized [-1, 1]
    # signal -- and this whole pipeline never normalizes before calling
    # this function. A fixed absolute total of 0.1 against a peak of ~3e4
    # is an ~1e-5 relative contribution: unmeasurable after the pipeline's
    # later autoscale/normalize steps, which only apply a global scalar
    # gain and cannot change that ratio. Measured directly (cycle 3): with
    # the old fixed-0.1 total, a 10,000x increase in signal peak shrank the
    # harmonic term's relative contribution by the same 10,000x (0.077 ->
    # 7.7e-6); scaling by peak keeps the relative contribution constant
    # (~0.10) regardless of the input's absolute scale, restoring a
    # genuinely audible (not swamped) contribution at real PCM scale.
    peak = float(cp.max(cp.abs(data)))
    harmonic_amplitude = (0.1 * peak) / max_iter if peak > 0 else 0.0

    for _ in range(max_iter):
        data_fft = cp.fft.fft(data_thres)
        mask = cp.abs(data_fft) > threshold
        data_fft_thres = cp.where(mask, data_fft, 0)
        data_thres = cp.fft.ifft(data_fft_thres).real

        # Harmonic reconstruction
        harmonics = cp.sin(cp.linspace(0, 2 * cp.pi, len(data_thres)))
        data_thres += harmonic_amplitude * harmonics

    return data_thres


def lms_filter(
    signal, desired, mu=0.001, num_taps=32, delay=1, return_weights=False
):
    """
    Apply an LMS adaptive filter using CuPy.

    As of the cycle 3 fix, this is a self-referential Adaptive Line
    Enhancer (ALE) by default (`delay=1`): the predictor's tap vector is
    drawn from `signal` lagged by `delay` samples rather than from
    `signal[i]` itself, so predicting `desired[i]` is a genuine (if
    small) estimation problem even when `signal is desired` -- the
    filter learns to predict each sample from its recent history,
    reinforcing quasi-periodic/tonal structure while treating
    lag-decorrelated content as unpredictable, a standard DSP technique
    (Widrow's Adaptive Line Enhancer), not a new algorithm class.

    Known issue (found and fixed in cycle 3): `upscale()` always calls
    this as `lms_filter(channel, channel)` -- signal and desired are the
    *same* array. With the prior `delay=0` behavior (tap 0 was always
    `signal[i]` itself, i.e. `desired[i]` exactly) and the cycle 1
    identity initialization (`w = [1, 0, ..., 0]`), `y == desired[i]`
    exactly on every single sample: the error term `e` was identically
    zero and the LMS update never changed `w` from its initial value --
    confirmed by direct measurement (cycle 3): the filtered output was
    bit-identical to the input and `w` stayed at `[1, 0, ..., 0]` after a
    full run. That made this stage an expensive (~18-19 of the pipeline's
    ~20 minute runtime) no-op. With `delay=1`, `w` measurably evolves
    (e.g. secondary taps moving from 0 to ~0.01-0.02 within 0.2s of
    44.1kHz audio) and the filtered output is no longer bit-identical to
    the input, while a highly-correlated-at-lag-1 signal (true of nearly
    all real audio) keeps the near-identity initialization close enough
    to the true minimum that no new warm-up dropout is introduced
    (measured warm-up RMS ratio ~1.00, same regression test as cycle 1).

    Parameters:
    signal (cp.ndarray): The input audio signal.
    desired (cp.ndarray): The desired output signal.
    mu (float): The step size for the adaptive filter.
    num_taps (int): The number of filter taps.
    delay (int): The ALE decorrelation lag, in samples, between the
        predictor's input taps and the sample being predicted. Must be
        >= 1 for `lms_filter(x, x, ...)` (signal is desired) to be a
        non-degenerate estimation problem; `0` reproduces the prior
        (now known-degenerate for that self-referential case) behavior.
    return_weights (bool): If True, return `(filtered_signal, w)` -- the
        final tap-weight vector alongside the filtered signal -- instead
        of just `filtered_signal`. Defaults to False to preserve the
        original single-array return for existing callers.

    Returns:
    cp.ndarray: The filtered audio signal (or `(filtered_signal, w)` if
        `return_weights` is True).
    """
    n = len(signal)
    # Initialize the direct-lag tap to 1 (all others 0) instead of an
    # all-zero weight vector. With delay >= 1, x[0] is signal[i - delay],
    # not signal[i] itself, so this is only a near pass-through (not an
    # exact one) for the self-referential case -- audio is highly
    # autocorrelated at small lags, so this still starts close to the true
    # optimum (avoiding the cycle 1 warm-up dropout) without being an exact
    # fixed point that blocks further adaptation (the cycle 3 bug). A
    # zero-initialized w makes every early output ~0 until enough
    # iterations accumulate to raise the weights, producing an audible
    # near-silent ramp-up at the start of the filtered signal.
    w = cp.zeros(num_taps, dtype=cp.float64)
    w[0] = 1.0
    filtered_signal = cp.zeros(n, dtype=cp.float64)
    start = num_taps + delay
    filtered_signal[:start] = signal[:start]

    for i in range(start, n):
        # Extract a vector of the last 'num_taps' samples from the signal,
        # lagged by 'delay' samples behind the sample being predicted.
        x = signal[i - delay:i - delay - num_taps:-1]

        # Compute the filter output: dot product of coefficients and input
        y = cp.dot(w, x)

        # Compute the error between the desired output and filter output
        e = desired[i] - y

        # Update the filter coefficients using the LMS rule
        w += 2 * mu * e * x

        # Ensure the coefficients remain finite to avoid numerical issues
        w = cp.clip(w, -1e10, 1e10)

        # Store the filter output in the filtered signal
        filtered_signal[i] = y

    if return_weights:
        return filtered_signal, w
    return filtered_signal


def upscale_channels(channels, upscale_factor, max_iter, threshold):
    """
    Process and upscale channels using the new interpolation and IST
    algorithms.

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
        expanded_channel = new_interpolation_algorithm(
            channel, upscale_factor
        )

        logger.info("Performing IST...")
        ist_changes = iterative_soft_thresholding(
            expanded_channel, max_iter, threshold
        )
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


def apply_original_nyquist_cutoff(
    signal, original_sample_rate, new_sample_rate
):
    """
    Zero out all spectral content above the original source's Nyquist
    frequency, as an unconditional final safety stage.

    Per the project's design (see `.claude/rules/project-mission.md`'s
    "no content above the original Nyquist frequency" constraint),
    fat_llama upscales precision/headroom within the original recording's
    real bandwidth -- it does not do bandwidth extension. The band above
    `original_sample_rate / 2` that an upsample opens up must stay
    silent, not merely "usually end up silent" depending on how earlier
    stages (interpolation, IST's harmonic term, autoscale, normalize,
    LMS adaptive filtering) happen to behave.

    As of the cycle-3 bandlimited-interpolation fix, that band already
    measures ~-136 dB (near the FFT noise floor) for a real end-to-end
    run -- this function's job is to make that a guarantee rather than
    an emergent property, so it still holds even if some future change
    to an earlier stage reintroduces energy there. It is applied
    unconditionally (no toggle), after every other processing stage, so
    no later step can reintroduce content past it.

    Implemented entirely with `cp.fft` (CuPy/CUDA), matching the rest of
    the pipeline's FFT/IST toolkit -- no scipy/numpy for this step, to
    stay on the CUDA-only path: `cp.fft.rfft` the signal, zero every bin
    whose frequency exceeds the original Nyquist frequency, then
    `cp.fft.irfft` back to the time domain at the same length.

    Parameters:
    signal (cp.ndarray): The fully processed signal, sampled at
        `new_sample_rate` (single channel).
    original_sample_rate (int): The sample rate of the original source
        audio, before upscaling. The cutoff frequency is
        `original_sample_rate / 2`, not derived from `new_sample_rate`.
    new_sample_rate (int or float): The sample rate `signal` is actually
        sampled at (i.e. `original_sample_rate * upscale_factor`).

    Returns:
    cp.ndarray: `signal` with all spectral content above
        `original_sample_rate / 2` removed, same length as `signal`.
    """
    signal = signal.astype(cp.float64)
    n = len(signal)
    if n == 0:
        return signal

    original_nyquist = original_sample_rate / 2.0
    spectrum = cp.fft.rfft(signal)
    freqs = cp.fft.rfftfreq(n, d=1.0 / new_sample_rate)
    spectrum = cp.where(freqs <= original_nyquist, spectrum, 0)

    return cp.fft.irfft(spectrum, n=n)


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
    Main function to upscale an audio file to a specified format with
    optional processing.

    Parameters:
    input_file_path (str): Path to the input audio file.
    output_file_path (str): Path to the output processed audio file.
    source_format (str): Format of the input audio file
        (e.g., 'mp3', 'wav', 'ogg', 'flac').
    target_format (str): Format of the output audio file
        (e.g., 'flac', 'wav').
    max_iterations (int): Maximum number of iterations for IST.
    threshold_value (float): Threshold value for IST.
    target_bitrate_kbps (int): Used only to derive the interpolation
        upscale_factor relative to the source file's own bitrate
        (upscale_factor = round(target_bitrate_kbps * 1000 / source
        bitrate)); must itself fall within the valid range for the
        target format (a sanity bound on this parameter, chosen to keep
        the derived upscale_factor reasonable). This is NOT a promise
        about the produced file's real bitrate: the output is always
        written as uncompressed PCM (see write_audio) at an upsampled
        sample rate, so its actual bitrate will be substantially higher
        than target_bitrate_kbps by design once upscale_factor > 1.
    toggle_normalize (bool): Whether to normalize the audio. Defaults to
        True.
    toggle_autoscale (bool): Whether to autoscale the audio based on the
        original audio. Defaults to True.
    toggle_adaptive_filter (bool): Whether to apply adaptive filtering.
        Defaults to True.
    """
    # Validate target_bitrate_kbps itself (the upscale_factor-derivation
    # knob below), not the eventual output file's real bitrate -- see the
    # target_bitrate_kbps docstring above for why those are different.
    valid_bitrate_ranges = {
        'flac': (800, 1411),
        'wav': (800, 6444),
    }

    if target_format not in valid_bitrate_ranges:
        raise ValueError(f"Unsupported target format: {target_format}")

    min_bitrate, max_bitrate = valid_bitrate_ranges[target_format]

    if not (min_bitrate <= target_bitrate_kbps <= max_bitrate):
        raise ValueError(
            f"{target_format.upper()} bitrate out of range. Please "
            f"provide a value between {min_bitrate} and {max_bitrate} kbps."
        )

    # Read the input audio file
    logger.info("Loading %s file...", source_format.upper())
    sample_rate, samples, bitrate, audio = read_audio(
        input_file_path, audio_format=source_format
    )
    if bitrate:
        logger.info(
            "Original %s bitrate: %.2f kbps",
            source_format.upper(), bitrate / 1000
        )

    samples = cp.array(samples, dtype=cp.float64)
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))

    # Determine the upscale factor
    target_bitrate = target_bitrate_kbps * 1000
    upscale_factor = round(target_bitrate / bitrate) if bitrate else 4
    logger.info("Upscale factor set to: %s", upscale_factor)

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
            scaled_channel = (
                normalize_signal(upscaled_channels[:, i])
                * cp.max(cp.abs(channel))
            )
            scaled_upscaled_channels.append(scaled_channel)
        scaled_upscaled_channels = cp.column_stack(scaled_upscaled_channels)
    else:
        scaled_upscaled_channels = upscaled_channels

    # Normalize audio if enabled
    if toggle_normalize:
        logger.info("Normalizing audio...")
        normalized_upscaled_channels = []
        for i in range(scaled_upscaled_channels.shape[1]):
            normalized_channel = normalize_signal(
                scaled_upscaled_channels[:, i]
            )
            normalized_upscaled_channels.append(normalized_channel)
        normalized_upscaled_channels = cp.column_stack(
            normalized_upscaled_channels
        )
    else:
        normalized_upscaled_channels = scaled_upscaled_channels

    # Apply adaptive filtering if enabled
    if toggle_adaptive_filter:
        logger.info("Applying adaptive filtering...")
        filtered_upscaled_channels = []
        for i in range(normalized_upscaled_channels.shape[1]):
            filtered_channel = lms_filter(
                normalized_upscaled_channels[:, i],
                normalized_upscaled_channels[:, i]
            )
            filtered_upscaled_channels.append(filtered_channel)
        filtered_upscaled_channels = cp.column_stack(
            filtered_upscaled_channels
        )
    else:
        filtered_upscaled_channels = normalized_upscaled_channels

    # Final safety stage: unconditionally guarantee no meaningful spectral
    # content survives above the *original* source's Nyquist frequency,
    # regardless of what interpolation, IST, autoscale, normalize, or LMS
    # did above -- fat_llama upscales precision/headroom within the
    # original recording's real bandwidth, it does not do bandwidth
    # extension (see apply_original_nyquist_cutoff's docstring). This has
    # no toggle and runs after every other processing stage, right before
    # write_audio, so nothing downstream can reintroduce content past it.
    new_sample_rate = sample_rate * upscale_factor
    logger.info(
        "Applying final Nyquist cutoff at %.1f Hz (original sample rate "
        "%s Hz)...", sample_rate / 2.0, sample_rate
    )
    cutoff_channels = []
    for i in range(filtered_upscaled_channels.shape[1]):
        cutoff_channels.append(
            apply_original_nyquist_cutoff(
                filtered_upscaled_channels[:, i],
                sample_rate,
                new_sample_rate,
            )
        )
    final_channels = cp.column_stack(cutoff_channels)

    # Write the processed audio to the output file
    write_audio(
        output_file_path,
        new_sample_rate,
        cp.asnumpy(final_channels),
        audio_format=target_format
    )
    logger.info(
        "Saved processed %s file at %s",
        target_format.upper(), output_file_path
    )
