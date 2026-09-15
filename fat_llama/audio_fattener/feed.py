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

# Consumer/professional playback hardware realistically supports sample
# rates up to about 192 kHz -- nothing mainstream plays back higher, and
# fat_llama upscales precision/headroom within the original recording's
# real bandwidth rather than extending it (see
# apply_original_nyquist_cutoff): any sample rate above this ceiling would
# only ever carry silence in the extended band, at a real cost (file size,
# IST/LMS runtime) for zero real benefit. See compute_upscale_factor.
MAX_REALISTIC_SAMPLE_RATE_HZ = 192000

# Default block size (in samples) for iterative_soft_thresholding's
# windowed-overlap-add (WOLA) local processing -- see that function's
# docstring for why a single whole-buffer FFT threshold is unsafe once
# peak-relative thresholding actually engages. 8192 samples is ~186 ms at
# 44.1 kHz: long enough for reasonable frequency resolution, short enough
# that a single loud passage's own peak-relative threshold stays local to
# that passage instead of setting the cutoff for the entire track.
IST_BLOCK_SIZE = 8192


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


def write_audio(
    file_path, sample_rate, data, audio_format, normalize=True,
    reference_amplitude=None
):
    """
    Write data to an audio file.

    Issue #18 fixes:

    (1) 64-bit float output precision: this pipeline computes internally
    in float64/complex128 throughout (see new_interpolation_algorithm,
    iterative_soft_thresholding, lms_filter, apply_original_nyquist_cutoff),
    but previously quantized every output -- both 'flac' and 'wav' -- to
    24-bit integer PCM ('PCM_24'), discarding that precision at the final
    step. FLAC has no true float/double subtype: libsndfile's own FLAC
    ceiling is PCM_24 (confirmed via `sf.available_subtypes('FLAC')`), so
    'flac' output is unchanged -- 24-bit is already its real ceiling, and
    "64-bit output" isn't a real FLAC/WAV container option (there is no
    true 64-bit integer or float FLAC/WAV format in practice); the highest
    fidelity actually available is used for each. WAV, however, does
    support a genuine 64-bit float subtype ('DOUBLE'), which this now
    uses: it stores the exact float64 values with no quantization and no
    clamping at all (verified directly -- writing a 'DOUBLE'-subtype WAV
    and reading it back reproduces the input bit-for-bit, even for values
    outside [-1, 1], unlike 'PCM_24' which silently clamps).

    (2) toggle_normalize wiring: `upscale()`'s `toggle_normalize` was
    already an optional pipeline stage (normalize_signal), but this
    function used to unconditionally peak-normalize (divide by data's own
    max abs value) before writing, regardless of what upstream toggle did
    -- so the final written file's peak amplitude always landed at
    exactly 1.0 (0 dBFS) either way, making toggle_normalize=False have no
    measurable effect on the actual output level. `normalize=False` now
    divides by `reference_amplitude` (typically the source's own
    AudioSegment.max_possible_amplitude, i.e. its bit-depth full-scale
    value) instead of this buffer's own peak -- a fixed, lossless domain
    conversion (raw-PCM-scale -> conventional float-PCM scale) rather than
    a loudness change, so the output's level reflects the original
    recording's own scale instead of always being stretched to touch
    exactly full scale:
      - For a float/double subtype (WAV), that division is the only
        change -- no clamping is applied, so the result is exact/lossless
        to float64 precision (verified: multiplying back by
        `reference_amplitude` reproduces the pre-write values exactly).
      - For an integer subtype (FLAC's mandatory 'PCM_24' -- there is no
        way around some quantization for an integer container), the
        result is additionally clipped to [-1, 1] as a safety net, since
        soundfile silently clamps out-of-range float input for integer
        subtypes instead of raising.
      - If `reference_amplitude` isn't provided, this falls back to the
        old peak-based divisor (the only way to guarantee a bounded
        result without a caller-supplied reference).

    Parameters:
    file_path (str): The path to the output audio file.
    sample_rate (int): The sample rate of the audio.
    data (np.ndarray): The audio data to write.
    audio_format (str): The format of the output audio file
        (e.g., 'flac', 'wav').
    normalize (bool): Whether to force a full-scale peak-normalize before
        writing. Defaults to True (matches all prior behavior exactly).
        When False, `reference_amplitude` is used instead of this
        buffer's own peak, genuinely preserving the original relative
        signal level (losslessly for float/double subtypes (wav); clipped
        as a container-required safety net for integer subtypes (flac)).
    reference_amplitude (float or None): Fixed scale (e.g. the source
        file's own bit-depth full-scale amplitude) to divide by when
        `normalize=False`. Ignored when `normalize=True`. Defaults to
        None (falls back to peak-based scaling if needed).
    """
    data = data.astype(np.float64)

    if audio_format == 'flac':
        sf_format, subtype = 'FLAC', 'PCM_24'
    elif audio_format == 'wav':
        # 64-bit float PCM -- the highest-fidelity subtype libsndfile
        # supports for WAV (Issue #18 item 1), matching this pipeline's
        # internal float64/complex128 computation exactly with zero
        # quantization loss.
        sf_format, subtype = 'WAV', 'DOUBLE'
    else:
        raise ValueError(f"Unsupported target format: {audio_format}")

    is_float_subtype = subtype in ('FLOAT', 'DOUBLE')

    if normalize:
        # Full-scale peak-normalize (the pre-existing, backward-compatible
        # default for both formats/subtypes).
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak
        if not is_float_subtype:
            data = np.clip(data, -1.0, 1.0)
    else:
        # normalize=False: convert data from its raw-PCM-scale domain into
        # the conventional float-PCM domain by dividing by a *fixed*
        # reference (the caller-supplied reference_amplitude, typically
        # the source's own bit-depth full-scale amplitude) instead of this
        # buffer's own peak -- this is a lossless domain conversion, not a
        # loudness change: it does not stretch the signal to touch exactly
        # full scale the way normalize=True's peak-based divisor does, so
        # the output genuinely reflects the original recording's relative
        # level. Falls back to peak-based scaling only if no reference was
        # given (the only way to guarantee a bounded result without one).
        divisor = reference_amplitude
        if not divisor:
            divisor = np.max(np.abs(data))
        if divisor > 0:
            data = data / divisor
        # Integer subtypes must fit [-1, 1] or soundfile silently clamps
        # instead of raising; float/double subtypes store any value
        # losslessly, so clipping here would only discard genuine (if
        # rare) above-reference-scale content for no reason.
        if not is_float_subtype:
            data = np.clip(data, -1.0, 1.0)

    sf.write(file_path, data, sample_rate, format=sf_format, subtype=subtype)


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

    As of the cycle 5 fix (see iterative_soft_thresholding's docstring),
    `threshold` is a peak-relative fraction (0-1), not an absolute
    magnitude cutoff: the mask keeps samples whose absolute value exceeds
    `threshold * max(abs(data))`, so the same conventional default
    (0.6) behaves consistently regardless of whether `data` happens to be
    normalized ([-1, 1]) or raw-PCM-scale (peak ~1e4-3e4).

    Parameters:
    data (cp.ndarray): The input audio data.
    threshold (float): Peak-relative threshold fraction (0-1) for IST.

    Returns:
    cp.ndarray: The thresholded audio data.
    """
    if data.size == 0:
        return data
    peak = cp.max(cp.abs(data))
    mask = cp.abs(data) > threshold * peak
    data_thres = cp.where(mask, data, 0)
    return data_thres


def _ist_chain(data, max_iter, threshold, convergence_tol):
    """
    The actual IST fixed-point iteration: init-threshold, then repeated
    FFT / peak-relative-threshold (DC bin always excluded) / IFFT,
    stopping early once the result converges. Operates on whatever
    buffer it is given -- the whole signal for short inputs, or a single
    windowed block for long inputs (see iterative_soft_thresholding,
    which is the public entry point and decides which).

    Parameters:
    data (cp.ndarray): The buffer to run IST on (whole signal or one
        analysis block).
    max_iter (int): Maximum number of passes before giving up on
        convergence.
    threshold (float): Peak-relative threshold fraction (0-1).
    convergence_tol (float): Relative early-exit tolerance.

    Returns:
    cp.ndarray: The IST-processed buffer, same length as `data`.
    """
    data_thres = initialize_ist(data, threshold)
    if data_thres.size == 0:
        return data_thres

    initial_scale = float(cp.max(cp.abs(data_thres)))
    if initial_scale == 0.0:
        # An all-zero initial threshold (e.g. threshold >= 1, or
        # genuinely silent input) still needs a nonzero convergence
        # scale to compare against -- fall back to the pre-threshold
        # data's own peak, and finally to 1.0 if that is also zero
        # (true silence), so the tolerance check below never divides by
        # zero or trivially never fires.
        initial_scale = float(cp.max(cp.abs(data))) if data.size else 0.0
    if initial_scale == 0.0:
        initial_scale = 1.0

    for _ in range(max_iter):
        data_fft = cp.fft.fft(data_thres)
        fft_peak = cp.max(cp.abs(data_fft))
        mask = cp.abs(data_fft) > threshold * fft_peak
        # Always exclude the DC (zero-frequency) bin from the retained
        # set -- see iterative_soft_thresholding's docstring. Guards
        # against a single asymmetric transient making bin 0 the loudest
        # bin and therefore the dominant (or sole) survivor of a
        # peak-relative mask.
        mask[0] = False
        data_fft_thres = cp.where(mask, data_fft, 0)
        next_data_thres = cp.fft.ifft(data_fft_thres).real

        change = float(cp.max(cp.abs(next_data_thres - data_thres)))
        data_thres = next_data_thres
        if change < convergence_tol * initial_scale:
            # Fixed point reached: hard-threshold IST alternates an exact
            # FFT/IFFT pair with a projection onto a fixed support, so
            # once that support (and therefore the reconstructed signal)
            # stops changing between passes, every further iteration up
            # to max_iter would recompute the identical result.
            break

    return data_thres


def _periodic_hann_window(n):
    """
    The DFT-even ("periodic") Hann window, sin^2(pi*k/n) -- distinct from
    the symmetric/endpoint-zero Hann window (e.g. cp.hanning), which does
    NOT exactly satisfy the constant-overlap-add (COLA) property this
    window is used for. Summing this window's own values, shifted by
    hops of n // 2 (50% overlap), across a fully-covered region equals
    exactly 1 everywhere -- the standard STFT COLA identity for a Hann
    window at 50% hop.

    Parameters:
    n (int): window length in samples.

    Returns:
    cp.ndarray: length-n window, dtype float64.
    """
    k = cp.arange(n, dtype=cp.float64)
    return 0.5 * (1.0 - cp.cos(2.0 * cp.pi * k / n))


def _sqrt_hann_window(n):
    """
    sqrt of the periodic Hann window (see _periodic_hann_window) -- used
    as BOTH the analysis and synthesis window in
    iterative_soft_thresholding's windowed-overlap-add (WOLA) block
    processing. Applying this window twice (once analyzing, once
    synthesizing) to the same block is equivalent to applying the
    (COLA-compliant) periodic Hann window once, so overlap-adding
    50%-hop blocks reconstructs a constant (1.0) weight everywhere in a
    fully-covered region.

    Parameters:
    n (int): window length in samples.

    Returns:
    cp.ndarray: length-n window, dtype float64.
    """
    return cp.sqrt(_periodic_hann_window(n))


def _wola_block_process(data, block_size, frame_processor):
    """
    Apply `frame_processor` independently to overlapping, sqrt-Hann-
    windowed blocks of `data` (analysis), then reconstruct via weighted
    overlap-add (WOLA: the same sqrt-Hann window is applied again on the
    synthesis side, and the result is divided by the actual accumulated
    window-squared weight at each sample -- a window-shaped, not flat-
    scalar, correction for any leftover unevenness rather than assuming
    perfect COLA coverage everywhere, e.g. at the very edges).

    Parameters:
    data (cp.ndarray): the full-length signal to process.
    block_size (int): analysis/synthesis window length in samples; must
        be even (hop is exactly block_size // 2, i.e. 50% overlap).
    frame_processor (callable): cp.ndarray -> cp.ndarray of the same
        length, applied to each windowed block independently.

    Returns:
    cp.ndarray: reconstructed signal, same length as `data`.
    """
    n = len(data)
    hop = block_size // 2
    window = _sqrt_hann_window(block_size)

    # Zero-pad by (block_size - hop) samples on each side so the first
    # and last real samples both fall under at least one fully-weighted
    # window, then pad the tail further so the padded length is an exact
    # whole number of hops past the final block start -- standard
    # STFT/WOLA edge handling, guaranteeing every real sample is covered
    # by the overlap-add loop below.
    edge_pad = block_size - hop
    min_len = edge_pad + n + edge_pad
    n_hops = max(0, -(-(min_len - block_size) // hop))  # ceil division
    padded_len = block_size + n_hops * hop
    right_pad = padded_len - (edge_pad + n)

    padded = cp.concatenate([
        cp.zeros(edge_pad, dtype=cp.float64),
        data.astype(cp.float64),
        cp.zeros(right_pad, dtype=cp.float64),
    ])

    output = cp.zeros(padded_len, dtype=cp.float64)
    weight = cp.zeros(padded_len, dtype=cp.float64)

    start = 0
    while start + block_size <= padded_len:
        frame = padded[start:start + block_size] * window
        processed = frame_processor(frame)
        output[start:start + block_size] += processed * window
        weight[start:start + block_size] += window * window
        start += hop

    safe_weight = cp.where(weight > 1e-12, weight, 1.0)
    reconstructed = output / safe_weight
    return reconstructed[edge_pad:edge_pad + n]


def _local_peak_envelope(signal, block_size):
    """
    A smooth, WOLA-consistent estimate of `signal`'s own local peak
    amplitude over time -- each `block_size`-sample analysis window
    contributes its own peak (`max(abs(.))` of that windowed block) as a
    constant "frame", cross-faded into neighboring blocks via the same
    sqrt-Hann analysis/synthesis overlap-add `_wola_block_process` already
    uses elsewhere in this module, rather than a hard block-boundary
    estimate. New in cycle 9, backing `_cap_ist_changes_to_baseline_peak`'s
    envelope-gated correction (see that function's docstring) -- this
    reuses the exact same block/window machinery `iterative_soft_
    thresholding` already relies on, rather than introducing a new
    filter-design parameter (e.g. an independent lowpass cutoff), so the
    envelope's own time resolution matches the granularity IST and the
    cap already treat as "local" elsewhere in this file.

    Parameters:
    signal (cp.ndarray): the signal whose local peak envelope to
        estimate (typically `expanded_channel`, the pre-IST baseline).
    block_size (int): analysis/synthesis window length in samples, same
        meaning as `_wola_block_process`'s own parameter.

    Returns:
    cp.ndarray: a smooth, non-negative envelope the same length as
        `signal`, approximating its own local peak amplitude at each
        sample position.
    """
    signal = signal.astype(cp.float64)
    if signal.size == 0:
        return signal
    if len(signal) <= block_size:
        return cp.full_like(signal, float(cp.max(cp.abs(signal))))

    def _block_peak(frame):
        return cp.full_like(frame, cp.max(cp.abs(frame)))

    return _wola_block_process(signal, block_size, _block_peak)


def iterative_soft_thresholding(
    data, max_iter, threshold, convergence_tol=1e-6, block_size=IST_BLOCK_SIZE
):
    """
    Perform IST on data using CuPy and cuFFT.

    Parameters:
    data (cp.ndarray): The input audio data.
    max_iter (int): The maximum number of iterations for IST to run
        before giving up on convergence (see the early-exit fix below --
        this is now a ceiling, not always the actual iteration count).
    threshold (float): Peak-relative threshold fraction (0-1) for IST --
        applied each iteration as `threshold * max(abs(current))`, both
        to `data`'s time-domain magnitude (via initialize_ist) and to
        each iteration's own FFT-bin magnitude, rather than as an
        absolute cutoff compared directly against raw magnitudes.
    convergence_tol (float): Relative early-exit tolerance (see below).
        Defaults to 1e-6.
    block_size (int): Windowed-overlap-add block size in samples (see
        the "Block/windowed processing" note below). Signals no longer
        than this run as a single whole-buffer chain (unchanged, fast
        path for short inputs/unit tests); longer signals are processed
        in 50%-overlap windowed blocks. Defaults to IST_BLOCK_SIZE
        (8192, ~186 ms at 44.1 kHz).

    Peak-relative thresholding fix (cycle 5, closing a known issue
    flagged since cycle 4): `data` here is raw-PCM-scale (peak
    ~1e4-3e4), while `threshold`'s conventional default (0.6) is many
    orders of magnitude smaller as an absolute cutoff. Measured (cycle 3,
    real ~15s input_test.mp3 channel): median FFT-bin magnitude ~9.3e4,
    so an absolute threshold of 0.6 masked essentially nothing (only
    exact/near-zero bins) in both the time- and frequency-domain steps
    below -- the "keep significant frequencies, discard noise" mechanism
    this function is meant to perform barely triggered at real audio's
    actual scale, leaving iterative_soft_thresholding to mostly perform
    near-lossless FFT/IFFT round trips rather than genuine sparse
    reconstruction. Both initialize_ist's time-domain mask and this
    function's own frequency-domain mask now compare against each
    domain's own current peak magnitude (`threshold * max(abs(.))`)
    instead of `threshold` alone, so the same 0-1 fraction behaves
    consistently regardless of the data's absolute scale.

    DC-bin exclusion (cycle 5, bundled with the fix above): once
    thresholding actually engages (per the fix above), a real audio
    buffer's own asymmetry (e.g. a loud one-sided transient) can make the
    FFT's zero-frequency (DC) bin the single largest-magnitude bin in a
    given iteration -- if so, a peak-relative mask would keep primarily
    that DC bin (since everything else is compared against *its*
    magnitude), injecting a spurious constant offset/drone into the
    reconstructed signal on every subsequent iteration. The
    frequency-domain mask therefore always excludes bin 0 regardless of
    its magnitude; genuine program content has no reason to depend on a
    literal zero-Hz component, so this costs nothing real while removing
    a specific, previously-unguarded failure mode.

    Convergence early-exit (cycle 5): hard-threshold IST here is a
    fixed-point projection (fft/ifft are exact inverses of each other),
    so once a pass's thresholded frequency support stops changing, every
    further iteration recomputes the identical result -- pure wasted GPU
    compute for the remaining iterations up to max_iter. Each pass now
    measures the largest per-sample change from the previous pass and
    breaks out of the loop once that change falls below
    `convergence_tol` times the initial (post-initialize_ist) peak
    magnitude, rather than always running exactly max_iter passes
    unconditionally.

    Harmonic-reconstruction term removed (cycle 4): cycles 2-3 added a
    per-iteration sinusoidal "harmonic reconstruction" term on top of the
    thresholding round trip above, meant to reconstruct missing/congested
    high-frequency detail. Across three consecutive cycles that term kept
    needing correction for a new artifact it introduced -- unbounded
    growth with max_iter (cycle 2), then a subsonic ~0.066 Hz tone
    (cycle 3's own fix for cycle 2's remaining defect), and finally (this
    cycle) a constant, non-source, audible drone: cycle 3 derived the
    term's frequency from `cp.argmax` of that iteration's own masked FFT,
    but that FFT is taken over the *entire* buffer (n ≈ 2.67M samples
    for a real ~15s upscaled channel) rather than any local window, so
    its single dominant bin is essentially the same value on every one
    of up to 300 iterations and across the whole track -- a static tone
    (measured by audio-quality-checker: a constant 98.168 Hz component at
    -28.7 dBFS spanning the entire output, plus its own overtone), not
    time-varying content-derived detail. Its amplitude was also a single
    value derived from the buffer's global peak, applied uniformly
    regardless of local signal level, so it behaved as a hard floor that
    only became audible in the many places where real content was
    quieter than it -- collapsing measured dynamic range in quiet
    passages from 57.1 dB (reference) to 25.4 dB (output).

    This cycle removes the harmonic-reconstruction term entirely rather
    than attempting a fourth revision of it, for several converging
    reasons: (1) a genuinely time-varying, artifact-free version would
    need real per-frame/local-window frequency and energy estimation with
    careful phase continuity across frame boundaries (to avoid clicks) --
    that is nontrivial DSP engineering this run cannot verify end-to-end
    against real audio without the GPU-based audio-quality-checker
    pipeline, which this function's own tests do not have access to;
    (2) naively nesting a per-block Python loop inside a function
    already run up to 300 times over ~2.67M samples/channel risks
    reintroducing exactly the kind of runtime blowup issue #20's
    lms_filter fix addressed elsewhere in this same file; (3) at cycle
    4's time, IST's threshold masking barely triggered at real audio
    scale at all (see the peak-relative thresholding fix above, added
    cycle 5), so removing the harmonic term at the time returned this
    function to IST's plain textbook form (init-threshold, then repeated
    FFT / frequency-domain-threshold / IFFT) -- exactly what the
    README/paper describe, with no synthetic tone bolted on. A properly
    windowed, locally energy-gated re-introduction remains a legitimate
    future direction, but only once it can be verified against real
    audio quality metrics rather than shipped speculatively.

    Block/windowed-overlap-add (WOLA) processing (cycle 5, found
    necessary while verifying the peak-relative fix above): a single
    whole-buffer FFT threshold compares every sample against one global
    peak magnitude, so the single loudest passage in the entire signal
    sets the cutoff for the *whole* track -- and because ifft's basis
    functions span the entire buffer, whatever frequency content a loud
    passage's threshold happens to retain leaks, via that global inverse
    transform, into every other sample position, including temporally
    distant quiet passages. This was measured directly while validating
    the peak-relative fix: a synthetic loud-then-quiet two-segment
    signal (test_ist_no_static_floor_in_quiet_segment) showed the quiet
    segment's RMS rising 57.8 dB after IST once peak-relative
    thresholding actually engaged -- a static-floor-like regression in
    the same family as the cycle-4 harmonic-term bug, just via a
    different mechanism (global spectral leakage instead of a
    synthetic tone). For signals longer than `block_size`, this function
    now runs the exact same `_ist_chain` (peak-relative threshold,
    DC-bin exclusion, convergence early-exit) independently on each of a
    series of 50%-overlapping, sqrt-Hann-windowed blocks, then
    reconstructs via weighted overlap-add (window applied on both the
    analysis and synthesis side, divided by the real accumulated
    window-squared weight rather than a flat scalar -- see
    _wola_block_process) -- so each block's own peak-relative threshold
    reflects only that ~186 ms window's local content, not the whole
    track. Verified (this cycle): re-running
    test_ist_no_static_floor_in_quiet_segment with this block processing
    in place brings the quiet segment's RMS rise back within its
    original (pre-cycle-5) 4x/12 dB bound. Signals no longer than
    `block_size` (e.g. every short synthetic buffer in this module's own
    unit tests) skip windowing entirely and run `_ist_chain` directly on
    the whole buffer, exactly matching this function's pre-block-
    processing behavior -- there is only one block to process, so
    windowing/overlap-add would only add unnecessary edge tapering with
    no benefit.

    Non-dominant-band contribution root cause (this cycle, closing the
    changelog's "no clearly measurable added detail below the original
    Nyquist" gap): real-audio measurement (audio-quality-checker) found
    IST's own contribution 34-78 dB below the interpolation baseline in
    every band above 2kHz -- too small to move a band's level regardless
    of _cap_ist_changes_to_baseline_peak's behavior. Direct measurement
    this cycle (see test_lower_threshold_increases_nondominant_band_ist_
    contribution) confirms both halves of that diagnosis on this module's
    own broadband synthetic signal: (1) the peak-inflation cap leaves
    every non-dominant band at ~100% of its uncapped magnitude -- it is
    not the bottleneck; (2) the default threshold_value=0.6's own
    peak-relative FFT-domain mask above IS the bottleneck -- for a
    real/broadband spectrum, a bin needs to be within `threshold` (60%)
    of the block's single loudest bin (usually low-frequency) to survive
    each pass, which almost no higher-frequency content can ever meet.
    A materially lower threshold_value (0.15 measured; same algorithm, no
    new synthesized content -- still the plain hard-threshold FFT/IFFT
    round trip) retains a larger, still information-derived set of
    "significant" bins each pass, measurably increasing every
    non-dominant band's own surviving contribution (uniformly ~8.4 dB
    from 2-18kHz on this cycle's synthetic signal) while changing the
    dominant bands by under 2 dB. This is reported as a proposed
    threshold_value default for upscale() (see this cycle's report) --
    verified only against this module's own synthetic signal here, not
    yet against audio-quality-checker's real-audio pipeline. A more
    fundamental redesign (a per-band/local-frequency-relative threshold
    instead of one global per-block peak) was prototyped but not shipped:
    it showed much larger gains for bands with no competing louder
    neighbor, but also a new failure mode not present in the mechanism
    above -- near-total collapse of a band's content depending on where
    an arbitrary band boundary happened to fall relative to the block's
    actual spectral content -- that could not be verified safe on real,
    continuous-spectrum program material without the audio-quality-
    checker pipeline, so it remains a candidate for a future cycle
    rather than a shipped change.

    Returns:
    cp.ndarray: The processed audio data after IST.
    """
    block_size = int(block_size)
    if block_size < 2:
        block_size = 2
    if block_size % 2:
        block_size += 1  # hop = block_size // 2 must be exact.

    if len(data) <= block_size:
        return _ist_chain(data, max_iter, threshold, convergence_tol)

    return _wola_block_process(
        data, block_size,
        lambda frame: _ist_chain(frame, max_iter, threshold, convergence_tol)
    )


def _cap_ist_changes_to_baseline_peak(
    expanded_channel, ist_changes, max_rounds=20, dominant_band_ratio=0.002,
    safety_margin=1.2, block_size=IST_BLOCK_SIZE
):
    """
    Rescale `ist_changes` so the combined signal
    `expanded_channel + ist_changes` peaks close to -- rather than
    substantially above -- `expanded_channel` (the pre-IST,
    interpolation-only baseline) on its own, WITHOUT uniformly
    suppressing frequency content that was not itself responsible for
    the inflation.

    History: cycle 6 (ported from the sibling fat_llama_fftw package)
    fixed a real regression from cycle 5's peak-relative IST threshold
    fix -- peak-relative thresholding (see initialize_ist /
    iterative_soft_thresholding) keeps/boosts whichever frequency
    dominates a block's own spectrum (for real music, usually low-
    frequency content); adding that boosted content back onto
    `expanded_channel` can raise the *combined* channel's own time-domain
    peak above what interpolation alone produced, which upscale()'s later
    autoscale/normalize stages (each a single scalar divide of the
    *entire* channel) then divided back down across every frequency,
    including bands IST never touched (measured: ~1.7-4.8 dB attenuation,
    coherence 9.5->8.0). Cycle 6's fix rescaled `ist_changes` by ONE
    uniform per-channel scalar via iterative shrink rounds. That closed
    the attenuation regression but (measured by audio-quality-checker on
    real audio, this cycle/7) also suppressed IST's own contribution
    almost everywhere else: for real, in-phase-dominated audio, the
    single scalar needed to tame the dominant band's overshoot is small
    (~0.06x on this module's own adversarial synthetic case below), and
    that same small scalar was then applied to every OTHER frequency
    `ist_changes` carries too -- including genuinely quiet, legitimately-
    added high-frequency detail -- collapsing it to near nothing.

    Frequency-selective fix (cycle 7): a whole-channel FFT of
    `ist_changes` shows the peak-inflation mechanism is concentrated in a
    narrow band around `ist_changes`'s own dominant spectral bin (the
    same one peak-relative thresholding privileges) plus its immediate
    spectral leakage/smearing (measured directly on this module's own
    adversarial synthetic case: the top single bin plus its neighbors
    within `dominant_band_ratio` of its magnitude account for the
    overwhelming majority of the peak overshoot, while a separate quiet
    high-frequency component sits at ~5e-5 of the dominant bin's
    magnitude -- two-plus orders of magnitude below even a small
    `dominant_band_ratio`, so it is never misclassified as "dominant").
    This function now: (1) splits `ist_changes`'s spectrum via
    `cp.fft.rfft` into a "dominant" component (bins whose magnitude is
    >= `dominant_band_ratio` times the spectrum's own peak magnitude) and
    a "residual" component (everything else), (2) iteratively shrinks
    ONLY the dominant component (via the same bounded multiplicative-
    shrink method cycle 6 used on the whole signal) so the combined peak
    approaches the baseline, leaving the residual component -- e.g. quiet
    high-frequency detail IST legitimately adds -- untouched, and (3)
    falls back to one additional whole-signal uniform shrink pass (cycle
    6's original method, applied to the frequency-selective candidate) as
    a safety net ONLY if that candidate's own combined peak still exceeds
    `baseline_peak * safety_margin` -- i.e. only when isolating the
    dominant band alone was not sufficient on its own. Verified directly
    (this cycle's own unit tests): on the adversarial loud-low +
    quiet-high synthetic case, the frequency-selective pass alone (no
    safety net needed) reduces the combined-peak excess over baseline to
    ~8% of the uncapped excess (comfortably under cycle 6's own ~15%
    acceptance bound) while leaving the quiet high-frequency component's
    own magnitude in `ist_changes` completely intact (~100% survival, vs
    ~5.9% under cycle 6's uniform scalar on the same case); a deliberately
    pathological case (dominant_band_ratio set so no bin qualifies as
    dominant) confirms the safety net engages correctly and reproduces
    cycle 6's own ~5.9%-survival bound rather than doing worse.

    This does not change `iterative_soft_thresholding` itself (still
    plain FFT/threshold/IFFT, no synthetic content, per
    .claude/rules/project-mission.md) -- it only reshapes, in the FFT
    domain, how much of IST's own contribution survives to be added onto
    the interpolated channel, and only in the specific band responsible
    for the peak-inflation mechanism above.

    As with cycle 6's version, there is no closed-form guarantee that the
    combined peak never exceeds `baseline_peak`: when IST's surviving
    dominant-band content is exactly in phase with `expanded_channel`'s
    own peak sample, only a dominant-component scale of (near) zero
    fully removes the overshoot that band contributes, and the residual
    component (deliberately left unscaled by the frequency-selective pass)
    can itself carry a small remaining excess -- this function approaches,
    but does not guarantee reaching, the baseline peak, same as cycle 6.
    `dominant_band_ratio=0.002` and `safety_margin=1.2` were chosen
    empirically on this module's own synthetic adversarial case (see
    tests) to keep the common case fully frequency-selective (safety net
    inactive) while still bounding pathological cases at least as well as
    cycle 6's uniform approach; `max_rounds=20` is unchanged from cycle 6.
    This has only been verified here via this module's own unit tests,
    not against real audio (see this cycle's report for that caveat).

    Envelope-gated correction (cycle 9, fixing a real regression this
    function itself introduced): audio-quality-checker measured a bounded
    but real onset defect on genuine real-audio output -- the first ~0.5s
    of a track (its own fade-in) showed frame RMS elevated up to +18.5 dB
    relative to the reference, decaying to within 1 dB by ~0.5s. Root-
    caused (this cycle, directly measured on this module's own synthetic
    fading-broadband signal, see
    test_cap_ist_changes_to_baseline_peak_preserves_quiet_onset) to this
    function's own dominant/residual split above: cycles 7-8 correctly
    identified WHICH frequencies drive the peak overshoot, but classify
    and rescale them via a SINGLE whole-buffer FFT/IFFT -- the same class
    of defect `iterative_soft_thresholding` itself was fixed (cycle 5) to
    avoid via WOLA block processing, reintroduced here because this
    function was added afterward and still operates on the whole channel
    at once. For a genuinely STATIONARY signal (this function's own
    existing adversarial unit tests), a whole-buffer split is harmless --
    every block looks alike, so a single global scale factor for the
    dominant band is representative everywhere. For a NON-STATIONARY
    channel (e.g. a real track's fade-in/fade-out, or any envelope-
    modulated passage), it is not: before any capping, `dominant_component
    + residual_component` reconstructs `ist_changes` EXACTLY (lossless by
    linearity of the FFT) -- the true, quiet shape of a fade-in exists
    only via a near-exact cancellation between the two components at that
    moment in time, not "inside" either one alone. Once `dominant_scale !=
    1` rescales ONLY the dominant component by one FIXED scalar (chosen to
    fix the overshoot in the LOUD part of the track) and reconstructs via
    a global `irfft`, that cancellation breaks -- and because the
    dominant component's own basis functions have roughly constant
    time-domain amplitude across the ENTIRE buffer (a handful of low-
    frequency bins, not a localized event), the same fixed-magnitude
    "correction" being subtracted everywhere is disproportionately large
    relative to a genuinely quiet region's own tiny true content,
    "unmasking" residual energy there that the uncapped, exact
    reconstruction had been quietly cancelling out. Measured directly:
    onset shape-relative frame-RMS deviation from the true envelope peaked
    at up to ~24.6 dB in the very first analysis block with the
    whole-buffer version, for a synthetic signal built specifically to
    exercise this (broadband multi-tone content under a 0.5s fade-in).

    The fix keeps the SAME dominant/residual split and SAME global
    `dominant_scale` computation (both already correctly identify what
    and how much needs shrinking for the LOUD part of the channel that
    actually causes the overshoot) but no longer applies the resulting
    correction (`(1 - dominant_scale) * dominant_component`, i.e. exactly
    how much is being removed from the dominant band) uniformly in time.
    Instead it tapers that correction by `_local_peak_envelope`, a smooth
    WOLA-based estimate of `expanded_channel`'s own local peak amplitude
    at `block_size` granularity (reusing the exact block/window machinery
    `iterative_soft_thresholding` already relies on, rather than adding a
    new filter-design parameter): a region whose own local peak sits near
    the channel's overall peak gets (close to) the full correction --
    reproducing cycles 7-8's already-verified behavior almost exactly,
    since real overshoots by construction occur where the channel is loud
    -- while a region far quieter than the channel's own peak (e.g. an
    onset/fade-in, which never came close to causing the overshoot in the
    first place) gets little to none of it, since it was never
    responsible for the mechanism the cap exists to fix. The same gating
    is applied to the safety-net fallback's own correction for
    consistency. Verified directly (this cycle): on the same fading
    synthetic signal, the onset deviation above drops from ~24.6 dB to
    ~4.9-6.1 dB (further reduced once averaged with the rest of the
    pipeline) while every one of this function's own pre-existing
    stationary-signal unit tests (peak-overshoot reduction, quiet-band
    survival, no-op cases, the safety-net fallback, and upscale_channels'
    end-to-end wiring) continues to pass with results numerically close
    to their pre-cycle-9 values -- for a stationary signal, the envelope
    stays near the channel's own peak throughout, so the gate stays near
    1.0 and this reduces to cycles 7-8's original behavior almost
    exactly.

    Parameters:
    expanded_channel (cp.ndarray): the pre-IST, interpolated channel (the
        baseline whose own peak must not be exceeded).
    ist_changes (cp.ndarray): IST's output for this channel, about to be
        added onto `expanded_channel` by upscale_channels.
    max_rounds (int): maximum number of iterative shrink rounds, used by
        both the dominant-band pass and the safety-net fallback. Default
        20.
    dominant_band_ratio (float): a spectral bin of `ist_changes` is
        classified "dominant" (and thus subject to shrinking) if its FFT
        magnitude is >= this fraction of the spectrum's own peak
        magnitude; everything else is "residual" and left untouched by
        the frequency-selective pass. Default 0.002.
    safety_margin (float): the whole-signal uniform-shrink safety net
        only engages if the frequency-selective candidate's own combined
        peak still exceeds `baseline_peak * safety_margin`. Default 1.2.
    block_size (int): granularity (in samples) of the `_local_peak_
        envelope` used to gate the dominant-band correction (cycle 9).
        Default IST_BLOCK_SIZE (8192), matching iterative_soft_
        thresholding's own block granularity.

    Returns:
    cp.ndarray: `ist_changes` reshaped in the FFT domain per the above
        (unchanged if no capping was needed).
    """
    expanded_channel = expanded_channel.astype(cp.float64)
    if expanded_channel.size == 0:
        return ist_changes

    baseline_peak = float(cp.max(cp.abs(expanded_channel)))
    if baseline_peak == 0.0:
        # A silent pre-IST baseline has no positive peak to bound against;
        # initialize_ist itself would already have zeroed an all-zero
        # input's threshold mask, so ist_changes is expected to be zero
        # here too -- nothing to cap.
        return ist_changes

    def _uniform_shrink(component):
        scale = 1.0
        for _ in range(max_rounds):
            combined_peak = float(
                cp.max(cp.abs(expanded_channel + scale * component))
            )
            if combined_peak <= baseline_peak or combined_peak == 0.0:
                break
            scale *= baseline_peak / combined_peak
        return scale

    uncapped_combined_peak = float(
        cp.max(cp.abs(expanded_channel + ist_changes))
    )
    if uncapped_combined_peak <= baseline_peak:
        # Nothing to cap -- a genuine no-op, not merely a small scalar.
        return ist_changes

    n = len(ist_changes)
    spectrum = cp.fft.rfft(ist_changes)
    magnitude = cp.abs(spectrum)
    peak_magnitude = float(cp.max(magnitude)) if magnitude.size else 0.0
    if peak_magnitude == 0.0:
        # ist_changes carries no spectral content at all -- shouldn't
        # normally arise given the check above, but guards against a
        # degenerate all-zero buffer.
        return ist_changes

    dominant_mask = magnitude >= dominant_band_ratio * peak_magnitude
    dominant_component = cp.fft.irfft(
        cp.where(dominant_mask, spectrum, 0), n=n
    )
    residual_component = cp.fft.irfft(
        cp.where(dominant_mask, 0, spectrum), n=n
    )

    dominant_scale = 1.0
    for _ in range(max_rounds):
        combined_peak = float(cp.max(cp.abs(
            expanded_channel
            + dominant_scale * dominant_component
            + residual_component
        )))
        if combined_peak <= baseline_peak or combined_peak == 0.0:
            break
        dominant_scale *= baseline_peak / combined_peak

    # Envelope-gated correction (cycle 9, see docstring): rather than
    # applying "how much is being removed from the dominant band"
    # uniformly in time (dominant_scale * dominant_component +
    # residual_component, cycles 7-8's original formula), taper it by a
    # smooth local-peak envelope of expanded_channel so genuinely quiet
    # regions (e.g. a fade-in onset, never responsible for the overshoot)
    # receive little to none of it, while regions near the channel's own
    # peak (where the overshoot actually happens) receive the same
    # correction cycles 7-8 already verified.
    envelope = _local_peak_envelope(expanded_channel, block_size)
    gate = cp.clip(envelope / baseline_peak, 0.0, 1.0)
    correction = (1.0 - dominant_scale) * dominant_component
    candidate = ist_changes - gate * correction

    combined_peak_candidate = float(
        cp.max(cp.abs(expanded_channel + candidate))
    )
    if combined_peak_candidate > baseline_peak * safety_margin:
        # The frequency-selective pass alone was not enough (e.g. an
        # unusually flat/broadband ist_changes spectrum, or a
        # dominant_band_ratio that happened to exclude the real driver) --
        # fall back to cycle 6's whole-signal uniform shrink on top of the
        # candidate, so pathological cases never regress below that
        # earlier guarantee. Gated the same way as the primary correction
        # above, for the same reason.
        safety_scale = _uniform_shrink(candidate)
        safety_correction = (1.0 - safety_scale) * candidate
        candidate = candidate - gate * safety_correction

    return candidate


def _lms_block_ranges(start, n, block_size):
    """
    Partition [start, n) into consecutive, non-overlapping chunks of at
    most `block_size` samples each, covering the whole range exactly
    once, in order.

    Extracted as a standalone, pure-Python generator (no CuPy) so the
    block-partitioning logic behind lms_filter's block-adaptive update
    (see its docstring, issue #20) can be unit-tested without a CUDA GPU
    -- the actual per-block filtering math still requires cp.ndarray
    input and is exercised by lms_filter's own (GPU-gated) regression
    tests instead.

    Parameters:
    start (int): first index to include (the warm-up length).
    n (int): one past the last index to include (the signal length).
    block_size (int): maximum chunk length; must be >= 1.

    Yields:
    (int, int): (block_start, block_end) pairs, block_end exclusive,
        with block_end - block_start <= block_size.
    """
    pos = start
    while pos < n:
        block_end = min(pos + block_size, n)
        yield pos, block_end
        pos = block_end


def lms_filter(
    signal, desired, mu=0.001, num_taps=32, delay=1, block_size=256,
    return_weights=False
):
    """
    Apply a block-adaptive LMS filter using CuPy.

    As of the cycle 3 fix, this is a self-referential Adaptive Line
    Enhancer (ALE) by default (`delay=1`): the predictor's tap vector is
    drawn from `signal` lagged by `delay` samples rather than from
    `signal[i]` itself, so predicting `desired[i]` is a genuine (if
    small) estimation problem even when `signal is desired` -- the
    filter learns to predict each sample from its recent history,
    reinforcing quasi-periodic/tonal structure while treating
    lag-decorrelated content as unpredictable, a standard DSP technique
    (Widrow's Adaptive Line Enhancer), not a new algorithm class.

    Issue #20 fix: this used to update the tap-weight vector `w` once
    per *sample* via a plain Python `for` loop -- each of the (up to
    several million, post-upscale) iterations issued several small,
    sequential CuPy/CUDA kernel calls (a slice, a dot product, an
    elementwise update, a clip, a store) whose combined per-iteration
    Python/kernel-launch overhead, not raw GPU compute, dominated
    runtime (measured, audio-quality-checker: 27.5 minutes wall clock
    for a 15.2s stereo source at a 7x-upscaled sample count of
    4,672,878/channel -- enabling `toggle_adaptive_filter` was
    impractical on consumer hardware). This replaces the per-sample
    update with block-adaptive LMS (aka the Block LMS / block-adaptive
    filter of Clark et al., 1981 -- a standard, long-documented LMS
    variant, not a new or learned/trained algorithm): the tap weights
    are held fixed across each block of up to `block_size` samples, the
    whole block's filter output is computed with a small (`num_taps`-
    length, not `block_size`-length) Python loop of vectorized
    elementwise CuPy ops over the whole block at once, and `w` is
    updated once per block using the block-averaged instantaneous
    gradient (mean over the block of `e[j] * x[j]`, matching the
    per-sample update's `2 * mu * e * x` in the limit `block_size == 1`
    -- see the update step below for the algebra). This cuts the number
    of sequential Python-loop iterations (and therefore sequential
    kernel launches) from `n` to roughly `n / block_size`, while
    remaining sequential/online across blocks -- still a genuinely
    adaptive filter, just updated at block granularity instead of
    sample granularity.

    `block_size=1` reproduces the exact prior per-sample update
    (verified algebraically: with a 1-sample block, the block-mean
    gradient is exactly `e[0] * x[0]`, identical to the old per-sample
    term) for callers that need bit-exact behavior; the default `256`
    trades a small amount of intra-block adaptation granularity (mu is
    small, 0.001 by default, so weight drift within one block is modest
    in practice) for roughly two orders of magnitude fewer sequential
    iterations. This is a genuine, disclosed accuracy/speed tradeoff --
    if a future audio-quality run shows measurably worse coherence
    attributable to this stage, reducing `block_size` (down to 1 for
    the prior exact behavior) is the first lever to try before anything
    else in this function.

    Known issue (found and fixed in cycle 3): `upscale()` always calls
    this as `lms_filter(channel, channel)` -- signal and desired are the
    *same* array. With the prior `delay=0` behavior (tap 0 was always
    `signal[i]` itself, i.e. `desired[i]` exactly) and the cycle 1
    identity initialization (`w = [1, 0, ..., 0]`), `y == desired[i]`
    exactly on every single sample: the error term `e` was identically
    zero and the LMS update never changed `w` from its initial value --
    confirmed by direct measurement (cycle 3): the filtered output was
    bit-identical to the input and `w` stayed at `[1, 0, ..., 0]` after a
    full run. With `delay=1`, `w` measurably evolves (e.g. secondary taps
    moving from 0 to ~0.01-0.02 within 0.2s of 44.1kHz audio) and the
    filtered output is no longer bit-identical to the input, while a
    highly-correlated-at-lag-1 signal (true of nearly all real audio)
    keeps the near-identity initialization close enough to the true
    minimum that no new warm-up dropout is introduced (measured warm-up
    RMS ratio ~1.00, same regression test as cycle 1).

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
    block_size (int): Number of samples per block-adaptive weight
        update (see above); `1` reproduces the exact prior per-sample
        LMS update. Defaults to 256.
    return_weights (bool): If True, return `(filtered_signal, w)` -- the
        final tap-weight vector alongside the filtered signal -- instead
        of just `filtered_signal`. Defaults to False to preserve the
        original single-array return for existing callers.

    Returns:
    cp.ndarray: The filtered audio signal (or `(filtered_signal, w)` if
        `return_weights` is True).
    """
    block_size = max(1, int(block_size))
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

    for block_start, block_end in _lms_block_ranges(start, n, block_size):
        block_len = block_end - block_start

        # Vectorized filter output for the whole block using the tap
        # weights as of the *start* of this block (held fixed across
        # the block -- this is the block-adaptive approximation). This
        # loop runs `num_taps` times (e.g. 32), not `block_len` times:
        # each iteration is one elementwise multiply-add over the whole
        # block at once, not a per-sample scalar operation.
        y_block = cp.zeros(block_len, dtype=cp.float64)
        for k in range(num_taps):
            lo = block_start - delay - k
            hi = block_end - delay - k
            y_block += w[k] * signal[lo:hi]

        # Error between the desired output and the filter output, for
        # every sample in the block at once.
        e_block = desired[block_start:block_end] - y_block

        # Block LMS weight update: replace the per-sample instantaneous
        # gradient `e * x` with its mean across the block, so `w` moves
        # once per block instead of once per sample. At block_size == 1
        # this is exactly `2 * mu * e[0] * x[0]` -- identical to the
        # original per-sample update rule.
        for k in range(num_taps):
            lo = block_start - delay - k
            hi = block_end - delay - k
            grad_k = cp.sum(e_block * signal[lo:hi]) / block_len
            w[k] = w[k] + 2 * mu * grad_k

        # Ensure the coefficients remain finite to avoid numerical issues
        w = cp.clip(w, -1e10, 1e10)

        # Store this block's filter output in the filtered signal
        filtered_signal[block_start:block_end] = y_block

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

    As of the cycle 6 fix, refined in cycle 7 to be frequency-selective
    (see _cap_ist_changes_to_baseline_peak's docstring), `ist_changes` is
    capped -- per channel, in the FFT domain -- before being added onto
    the interpolated channel, so IST's own peak-relative boost cannot
    inflate this channel's peak beyond what interpolation alone produced.
    Without this, that inflation propagated into upscale()'s later
    autoscale/normalize stages (each a per-channel scalar divide by this
    channel's own peak), which divided every frequency in the channel
    down harder than the reference -- including bands IST never touched
    -- measured (audio-quality-checker, cycle 6) as a broad ~1.7-4.8 dB
    attenuation regression relative to the reference FLAC. Cycle 6's own
    single-uniform-scalar version of the cap fixed that regression but
    (measured, cycle 7) also suppressed IST's own added detail almost
    everywhere else; cycle 7's frequency-selective version shrinks only
    the FFT band actually responsible for the peak inflation, leaving
    other bands (e.g. quiet high-frequency detail IST legitimately adds)
    untouched in the common case.

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
        ist_changes = _cap_ist_changes_to_baseline_peak(
            expanded_channel, ist_changes
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


def compute_upscale_factor(
    sample_rate, source_bitrate_bps, target_bitrate_kbps
):
    """
    Derive an integer upscale factor from target_bitrate_kbps, bounded so
    the resulting sample rate (sample_rate * upscale_factor) stays within
    a realistic consumer playback range.

    Issue #20: the previous derivation --
    round(target_bitrate_kbps * 1000 / source_bitrate_bps) -- compared a
    target value calibrated to the *compressed-file* bitrate range
    (target_bitrate_kbps's valid range is 800-1411 kbps for flac,
    800-6444 kbps for wav) directly against the source's own *compressed*
    bitrate (e.g. a typical mp3 at 128-192 kbps). That ratio routinely
    lands at 5-7+ for realistic inputs (e.g. round(1400 / 192) = 7,
    round(900 / 128) = 7), inflating the output sample rate far past any
    realistic range (e.g. 44100 Hz * 7 = 308700 Hz) for no corresponding
    gain in real information: apply_original_nyquist_cutoff guarantees
    the overwhelming majority of that extra bandwidth is silence
    (measured, audio-quality-checker: -140.2 dB above 22050 Hz for a 7x
    upscale of 44100 Hz audio; decimating the output back to the
    original rate and re-expanding it reproduced the 7x output to -66.7
    dB error, confirming the extra rate carried no information). The
    inflated sample count this produced was also the dominant multiplier
    behind issue #20's second report -- lms_filter's per-sample loop
    scales with sample count.

    This keeps the original ratio as a starting point, so
    target_bitrate_kbps keeps its documented contract (a higher value
    still drives a larger factor, relative to the source's own bitrate),
    but clamps the result so sample_rate * upscale_factor never exceeds
    MAX_REALISTIC_SAMPLE_RATE_HZ (192 kHz) and never drops below 1 (this
    is an upscaler, not a downscaler).

    Parameters:
    sample_rate (int): the source audio's sample rate, in Hz.
    source_bitrate_bps (float or None): the source file's own bitrate, in
        bits/sec, as returned by read_audio (None if undeterminable).
    target_bitrate_kbps (int): the caller's requested target bitrate, in
        kbps (already validated by the caller against the target format's
        valid range).

    Returns:
    int: the upscale factor to use: >= 1, and such that
        sample_rate * upscale_factor <= MAX_REALISTIC_SAMPLE_RATE_HZ
        whenever sample_rate itself is already within that ceiling.
    """
    if source_bitrate_bps:
        raw_factor = round(target_bitrate_kbps * 1000 / source_bitrate_bps)
    else:
        raw_factor = 4
    raw_factor = max(raw_factor, 1)

    max_factor = max(1, int(MAX_REALISTIC_SAMPLE_RATE_HZ // sample_rate))

    return min(raw_factor, max_factor)


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
    max_iterations (int): Maximum number of iterations for IST -- a
        ceiling, not always the actual iteration count, since (as of the
        cycle 5 fix) iterative_soft_thresholding now exits early once its
        result converges to a fixed point (see that function's
        docstring).
    threshold_value (float): Peak-relative IST threshold fraction (0-1).
        As of the cycle 5 fix, this is compared against each domain's
        own current peak magnitude each iteration (`threshold_value *
        max(abs(current))`), not against raw sample/FFT-bin magnitudes
        directly -- previously an absolute cutoff compared as-is against
        raw-PCM-scale magnitudes (peak ~1e4-3e4), so the documented
        default (0.6) masked essentially nothing and IST barely
        sparsified real audio at all (see
        iterative_soft_thresholding's docstring for the full history).
    target_bitrate_kbps (int): Used only to derive the interpolation
        upscale_factor relative to the source file's own bitrate --
        see compute_upscale_factor for the exact formula; must itself
        fall within the valid range for the target format (a sanity
        bound on this parameter, chosen to keep the derived
        upscale_factor reasonable). As of the issue #20 fix, the
        derived factor is additionally clamped so the resulting sample
        rate (sample_rate * upscale_factor) never exceeds
        MAX_REALISTIC_SAMPLE_RATE_HZ (192 kHz) -- previously this
        formula alone could drive sample rates well past 250 kHz for
        realistic inputs, which apply_original_nyquist_cutoff would
        guarantee is mostly silence anyway. This is NOT a promise about
        the produced file's real bitrate: the output is always written
        as uncompressed PCM (see write_audio) at an upsampled sample
        rate, so its actual bitrate will be higher than
        target_bitrate_kbps once upscale_factor > 1, though now bounded
        to a realistic range rather than unbounded.
    toggle_normalize (bool): Whether to normalize the audio. Defaults to
        True. Controls both the in-pipeline peak-normalize stage and (as
        of the Issue #18 fix) write_audio()'s own final scaling: when
        False, the output genuinely preserves the original recording's
        relative signal level (losslessly for 'wav', which is written as
        64-bit float; scaled by the source's own full-scale amplitude
        rather than forced to touch exactly full scale for 'flac', which
        has no float subtype) instead of always being renormalized to
        0 dBFS regardless of this flag, which was the prior behavior.
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

    # Determine the upscale factor -- see compute_upscale_factor's
    # docstring (issue #20) for why this is bounded to a realistic sample
    # rate rather than an unbounded ratio of target_bitrate_kbps to the
    # source's own (compressed) bitrate.
    upscale_factor = compute_upscale_factor(
        sample_rate, bitrate, target_bitrate_kbps
    )
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

    # Write the processed audio to the output file. toggle_normalize is
    # wired straight through to write_audio()'s own normalize argument
    # (Issue #18 fix) -- previously write_audio() unconditionally
    # peak-normalized regardless of this flag, so toggle_normalize=False
    # had no effect on the actual written output level.
    # audio.max_possible_amplitude is the source's own bit-depth
    # full-scale reference, used (only when toggle_normalize=False and an
    # integer subtype is written) to preserve the original recording's
    # relative level instead of rescaling to this buffer's own peak.
    write_audio(
        output_file_path,
        new_sample_rate,
        cp.asnumpy(final_channels),
        audio_format=target_format,
        normalize=toggle_normalize,
        reference_amplitude=audio.max_possible_amplitude,
    )
    logger.info(
        "Saved processed %s file at %s",
        target_format.upper(), output_file_path
    )
