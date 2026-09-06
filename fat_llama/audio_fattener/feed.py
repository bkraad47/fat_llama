import logging
import math
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

# --- Upscale-factor bounds -------------------------------------------------
# Hard upper bound on the interpolation factor derived from
# target_bitrate_kbps / source bitrate. See compute_upscale_factor() for the
# full rationale; in short, the derived ratio compares two incommensurable
# quantities (a target rate vs. a *compressed* source rate) and is unbounded
# above, so a low-bitrate source could otherwise drive a multi-MHz output
# sample rate and a proportional blow-up in memory and runtime.
MAX_UPSCALE_FACTOR = 8

# Hard upper bound on the produced file's sample rate, in Hz. 192 kHz is the
# highest sample rate in common consumer/professional use (and the ceiling of
# most DACs, players and editors); above it, output files are widely
# unplayable and gain nothing -- see compute_upscale_factor().
MAX_OUTPUT_SAMPLE_RATE = 192000

# Fallback factor used when the source file's bitrate cannot be determined.
DEFAULT_UPSCALE_FACTOR = 4

# --- LMS block-processing bounds -------------------------------------------
# Upper bound on the number of samples lms_filter() processes per weight
# update. The actual block size is derived per call from the signal's own
# power (see _derive_lms_block_size); this is only a ceiling.
LMS_MAX_BLOCK_SIZE = 4096

# Safety factor for the block-size stability bound: the per-block eigenvalue
# contraction 2 * mu * block_size * lambda_max is kept at or below this
# value, comfortably inside LMS's |1 - 2 * mu * L * lambda| < 1 convergence
# condition. See _derive_lms_block_size().
LMS_BLOCK_STABILITY_SAFETY = 0.5


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


def compute_upscale_factor(
    target_bitrate_kbps,
    source_bitrate,
    source_sample_rate,
    max_upscale_factor=MAX_UPSCALE_FACTOR,
    max_output_sample_rate=MAX_OUTPUT_SAMPLE_RATE,
):
    """
    Derive a bounded, realistic integer interpolation factor.

    The historical derivation was a single unguarded expression inside
    `upscale()`::

        upscale_factor = round(target_bitrate / bitrate) if bitrate else 4

    That ratio compares two incommensurable quantities -- a target rate
    expressed in the units of *uncompressed* audio (its 800-1411 kbps
    "flac" range is CD-PCM-shaped) against the source file's
    *compressed* bitrate -- so it is not a physically meaningful ratio
    and, more importantly, it is unbounded in both directions:

    * **No lower bound.** `round()` returns 0 whenever the source
      bitrate is more than twice the target (e.g. a 24-bit/96 kHz WAV
      source at 4608 kbps with `target_bitrate_kbps=900` gives
      `round(0.195) == 0`). A factor of 0 makes
      `new_interpolation_algorithm` produce a zero-length signal and
      `upscale()` write a file at 0 Hz -- a crash or an empty output.
    * **No upper bound.** A low-bitrate source inflates it without
      limit: an 8 kbps source with the same target gives 112, i.e. a
      ~4.9 MHz output sample rate and a ~112x blow-up in memory and
      runtime.
    * **No notion of a realistic output sample rate.** Measured against
      this project's own test asset (`input_test.mp3`, a 192 kbps
      44.1 kHz stereo MP3): `target_bitrate_kbps=900` gives
      `round(4.6875) == 5` and a 220500 Hz output, and the default 1411
      gives 7 and a 308700 Hz output whose (lossless-FLAC) bitrate
      measures 5294 kbps -- both reported by users as unrealistic, and
      neither playable on typical playback hardware.

    This function keeps the documented ratio (so `target_bitrate_kbps`
    remains a monotonically non-decreasing quality knob, and small,
    already-sane ratios are unaffected) but bounds it:

    1. Unknown/zero/negative/non-finite source bitrate falls back to
       `DEFAULT_UPSCALE_FACTOR` (unchanged behaviour).
    2. The factor is clamped to at least 1 (never 0 -- an upscale never
       shrinks or empties the signal) and at most
       `max_upscale_factor`.
    3. The factor is further reduced so the resulting output sample
       rate (`source_sample_rate * factor`) does not exceed
       `max_output_sample_rate`.

    Bound (3) costs no audio quality whatsoever in this pipeline:
    `apply_original_nyquist_cutoff` unconditionally zeroes every
    spectral bin above `source_sample_rate / 2` as the final stage
    (see `.claude/rules/project-mission.md`'s "no content above the
    original Nyquist frequency" constraint), so *any* factor of 2 or
    more already provides full headroom for every frequency the output
    is allowed to contain. A factor of 7 stores exactly the same
    information as a factor of 4 or 2 -- it just spends 1.75x/3.5x the
    bytes and the proportional extra runtime doing it.

    Parameters:
    target_bitrate_kbps (int): The caller's target bitrate knob, in
        kbps.
    source_bitrate (int, float or None): The source file's bitrate in
        bits/sec as reported by `read_audio`. `None`, 0, negative or
        non-finite values mean "undeterminable".
    source_sample_rate (int): The source file's sample rate in Hz, used
        for the output-sample-rate ceiling.
    max_upscale_factor (int): Hard ceiling on the returned factor.
    max_output_sample_rate (int or None): Ceiling on
        `source_sample_rate * factor`, in Hz. `None` disables the
        sample-rate ceiling.

    Returns:
    int: The interpolation factor, always >= 1.
    """
    usable_bitrate = (
        source_bitrate is not None
        and math.isfinite(float(source_bitrate))
        and float(source_bitrate) > 0
    )

    if usable_bitrate:
        raw_factor = int(
            round((target_bitrate_kbps * 1000) / float(source_bitrate))
        )
    else:
        raw_factor = int(DEFAULT_UPSCALE_FACTOR)
        logger.warning(
            "Source bitrate is undeterminable (%s); falling back to the "
            "default upscale factor of %s.",
            source_bitrate, raw_factor
        )

    factor = max(1, min(raw_factor, int(max_upscale_factor)))

    if (
        max_output_sample_rate
        and source_sample_rate
        and source_sample_rate > 0
    ):
        rate_limited_factor = int(
            max_output_sample_rate // source_sample_rate
        )
        factor = max(1, min(factor, rate_limited_factor))

    if factor != raw_factor:
        logger.warning(
            "Raw upscale factor %s (target %s kbps / source %s bps) is "
            "outside the realistic range; clamped to %s (max factor %s, "
            "max output sample rate %s Hz). Output sample rate will be "
            "%s Hz.",
            raw_factor, target_bitrate_kbps, source_bitrate, factor,
            max_upscale_factor, max_output_sample_rate,
            source_sample_rate * factor if source_sample_rate else None
        )

    logger.info(
        "Upscale factor: %s (raw ratio %s, source bitrate %s bps, source "
        "sample rate %s Hz, output sample rate %s Hz)",
        factor, raw_factor, source_bitrate, source_sample_rate,
        source_sample_rate * factor if source_sample_rate else None
    )

    return factor


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


def _derive_lms_block_size(
    signal,
    mu,
    num_taps,
    max_block=LMS_MAX_BLOCK_SIZE,
    safety=LMS_BLOCK_STABILITY_SAFETY,
):
    """
    Choose the largest LMS block size that provably stays stable.

    `lms_filter` processes the LMS recursion in blocks: within a block
    the tap weights are held frozen and the weight update applied at
    the end of the block is exactly the sum of the per-sample updates
    those frozen weights would have produced (standard Block LMS).
    That batching is what makes the filter fast on a GPU -- it replaces
    ~6 CuPy kernel launches *per audio sample* with ~10 per *block* --
    but it is only valid while the block is short enough that the
    weights genuinely would not have moved much across it.

    The mean-weight recursion for Block LMS is
    `E[w] <- (I - 2 * mu * L * R) E[w]`, so convergence requires
    `2 * mu * L * lambda_max < 2`, i.e. the usable block size shrinks as
    the step size and the signal's power grow. Using the standard bound
    `lambda_max <= trace(R) = num_taps * E[x^2]`, this returns

        L = clamp(safety / (2 * mu * num_taps * mean(signal^2)), 1,
                  max_block)

    which keeps the per-block contraction at or below `safety` (0.5 by
    default), a 4x margin inside the stability limit. Measured
    (NumPy reference implementation, 0.2 s two-tone 300/900 Hz signal at
    amplitude 0.5, mu=0.001, num_taps=32): the derived block size is 53
    and the block-processed output tracks the sample-by-sample
    recursion with correlation 0.99999 and a 1.3e-2 relative peak
    deviation, while a fixed block of 1024 for the same signal diverges
    outright (output peak 3.9e4 vs. the input's 0.5, correlation 0.13)
    -- i.e. this bound is load-bearing, not decorative.

    Parameters:
    signal (cp.ndarray): The filter's input signal, used only to
        estimate its power.
    mu (float): The LMS step size.
    num_taps (int): The number of filter taps.
    max_block (int): Ceiling on the returned block size.
    safety (float): Target per-block contraction factor
        (`2 * mu * L * lambda_max`), which must stay below 2 for
        stability.

    Returns:
    int: The block size, always >= 1 (1 reproduces the exact
        sample-by-sample LMS recursion).
    """
    max_block = max(1, int(max_block))

    if mu <= 0:
        return max_block

    power = float(cp.mean(cp.square(signal.astype(cp.float64))))
    trace_r = num_taps * power
    denominator = 2.0 * mu * trace_r

    if not math.isfinite(denominator) or denominator <= 0:
        # Silent (or non-finite) input: nothing to adapt to, so the
        # block size cannot destabilise anything.
        return max_block

    return int(max(1, min(max_block, math.floor(safety / denominator))))


def lms_filter(
    signal, desired, mu=0.001, num_taps=32, delay=1, return_weights=False,
    block_size=None
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

    Performance (fixed this cycle): this used to advance the recursion
    one sample at a time from Python, issuing roughly six CuPy kernel
    launches per output sample. Every one of those launches is a
    fixed ~10-20 microseconds of host-side latency that no GPU can
    amortise, so runtime was set by the *sample count*, not by the
    arithmetic: a 15 s stereo file at a 220500 Hz upscaled rate is
    6.7M samples, i.e. ~40M launches -- tens of minutes, and the
    dominant cost of the whole pipeline (previously measured at ~91% of
    a ~20 minute run). Users reported this as the upscale "hanging" on
    15 s of audio with `toggle_adaptive_filter=True`. It was never a
    numerical problem (no NaN/Inf accumulation, no denormal stall):
    the loop is launch-bound.

    The recursion is now advanced a *block* at a time (standard Block
    LMS): within a block the tap weights are frozen, every sample's
    filter output is computed with one batched matrix-vector product,
    and the end-of-block weight update is exactly the sum of the
    per-sample LMS updates those frozen weights would have produced.
    The update rule itself is therefore unchanged -- `block_size=1`
    reproduces the previous sample-by-sample behaviour bit for bit
    (verified to 1e-16 against a NumPy reference of the old loop) --
    and the block size is not a fixed guess but derived per call from
    the signal's own power so the frozen-weight approximation provably
    stays inside LMS's stability bound (see `_derive_lms_block_size`).
    For normalized audio that yields blocks of ~50-150 samples, cutting
    the Python-level iteration (and kernel launch) count by the same
    factor.

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
    block_size (int or None): Number of samples processed per weight
        update. `None` (the default) derives a stability-bounded block
        size from the signal itself via `_derive_lms_block_size`; `1`
        reproduces the exact sample-by-sample recursion (slow); larger
        values trade adaptation granularity for speed and, past the
        derived bound, diverge.

    Returns:
    cp.ndarray: The filtered audio signal (or `(filtered_signal, w)` if
        `return_weights` is True).
    """
    signal = signal.astype(cp.float64)
    desired = desired.astype(cp.float64)
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

    if n <= start:
        # Too short to run the recursion at all: pass the signal through
        # unchanged rather than indexing out of bounds.
        filtered_signal[:n] = signal[:n]
        if return_weights:
            return filtered_signal, w
        return filtered_signal

    filtered_signal[:start] = signal[:start]

    if block_size is None:
        block_size = _derive_lms_block_size(signal, mu, num_taps)
    block_size = max(1, int(block_size))

    # Offsets of a block's tap vectors relative to the block's first
    # predicted sample: row j, tap k is sample (j - delay - k). Computed
    # once and reused, so each block costs a constant handful of kernel
    # launches instead of a handful per sample.
    tap_offsets = (
        cp.arange(block_size, dtype=cp.int64)[:, None]
        - delay
        - cp.arange(num_taps, dtype=cp.int64)[None, :]
    )

    for block_start in range(start, n, block_size):
        block_end = min(block_start + block_size, n)
        block_len = block_end - block_start

        # Gather every tap vector in the block at once: row j holds the
        # num_taps samples ending 'delay' samples before the sample being
        # predicted, exactly as the per-sample loop's
        # signal[i - delay:i - delay - num_taps:-1] slice did.
        x_block = signal[tap_offsets[:block_len] + block_start]

        # Filter outputs for the whole block, computed with the weights
        # held at their block-entry value.
        y = x_block @ w

        # Errors between the desired output and the filter output
        e = desired[block_start:block_end] - y

        # Store the filter outputs in the filtered signal
        filtered_signal[block_start:block_end] = y

        # Update the filter coefficients using the LMS rule, accumulated
        # over the block: identical to applying the per-sample update
        # 2 * mu * e[j] * x[j] for each sample in turn while the weights
        # stay frozen.
        w = w + 2 * mu * (x_block.T @ e)

        # Ensure the coefficients remain finite to avoid numerical issues.
        # cp.clip alone cannot repair NaN (comparisons against NaN are
        # false, so it propagates straight through), which is why the
        # nan_to_num pass comes first.
        w = cp.clip(cp.nan_to_num(w), -1e10, 1e10)

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
        bitrate), then bounded -- see compute_upscale_factor); must
        itself fall within the valid range for the target format (a
        sanity bound on this parameter, chosen to keep the derived
        upscale_factor reasonable). This is NOT a promise about the
        produced file's real bitrate: the output is always written as
        uncompressed PCM (see write_audio) at an upsampled sample rate,
        so its actual bitrate will be substantially higher than
        target_bitrate_kbps by design once upscale_factor > 1. The
        derived factor is clamped to [1, MAX_UPSCALE_FACTOR] and
        further reduced so the output sample rate never exceeds
        MAX_OUTPUT_SAMPLE_RATE Hz, so this knob saturates rather than
        producing unrealistic output for atypical source bitrates.
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

    # Determine the upscale factor. compute_upscale_factor() bounds the
    # raw target/source bitrate ratio so an unusual (very low, very high,
    # or unreadable) source bitrate cannot produce a factor of 0 (empty
    # output) or an unrealistically large one (multi-hundred-kHz output
    # sample rates, proportionally huge files, and proportionally longer
    # runtime) -- see its docstring.
    upscale_factor = compute_upscale_factor(
        target_bitrate_kbps, bitrate, sample_rate
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
