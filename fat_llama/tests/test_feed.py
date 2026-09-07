import inspect
import math
import os
import unittest
from unittest import mock

import cupy as cp
import numpy as np
import soundfile as sf
from mutagen.flac import FLAC

from fat_llama.audio_fattener import feed as feed_module
from fat_llama.audio_fattener.feed import (
    MAX_REALISTIC_SAMPLE_RATE_HZ, _lms_block_ranges,
    apply_original_nyquist_cutoff, compute_upscale_factor,
    iterative_soft_thresholding, lms_filter, new_interpolation_algorithm,
    read_audio, upscale, write_audio
)


def _cuda_gpu_available():
    # This project is CUDA-only by design (no CPU fallback -- see
    # .claude/rules/project-mission.md); a missing/insufficient GPU driver
    # is an environmental limitation to report, not something to work
    # around with a CPU code path. GitHub's free-tier hosted CI runners
    # (ubuntu-latest) have no GPU hardware at all, so any test that
    # actually exercises cupy compute would otherwise crash there with a
    # raw CUDARuntimeError regardless of code correctness. This check lets
    # such tests skip cleanly with a clear reason in that environment,
    # while still running normally wherever a real CUDA GPU is present
    # (including this project's own local development machines).
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU_AVAILABLE = _cuda_gpu_available()
requires_gpu = unittest.skipUnless(
    GPU_AVAILABLE, "requires a CUDA-capable GPU (none available/functional)"
)


class TestAudioFattener(unittest.TestCase):

    def setUp(self):
        # Create a small example MP3 file for testing
        self.test_mp3_file = 'test_input.mp3'
        self.create_test_mp3(self.test_mp3_file)

    def tearDown(self):
        # Remove the test MP3 file and any generated FLAC files
        if os.path.exists(self.test_mp3_file):
            os.remove(self.test_mp3_file)
        if os.path.exists('output_processed.flac'):
            os.remove('output_processed.flac')

    def create_test_mp3(self, filename):
        from pydub.generators import Sine
        # 1 second of 440 Hz sine wave
        sine_wave = Sine(440).to_audio_segment(duration=1000)
        # export() returns the underlying file handle; pydub does not close
        # it for us, so do it explicitly to avoid leaking an open file
        # descriptor per test.
        out_handle = sine_wave.export(filename, format="mp3")
        out_handle.close()

    def test_read_audio(self):
        sample_rate, samples, bitrate, audio = read_audio(
            self.test_mp3_file, audio_format='mp3'
        )
        # Default sample rate for the generated sine wave
        self.assertEqual(sample_rate, 44100)
        # 1 second of audio at 44100 Hz
        self.assertEqual(len(samples), 44100)
        # A mono source must come back as a flat 1-D array, not an (N, 2)
        # reshape.
        self.assertEqual(audio.channels, 1)
        self.assertEqual(samples.ndim, 1)
        self.assertEqual(len(audio), 1000)  # duration in ms
        # The exact bitrate depends on the ffmpeg/LAME build, so assert the
        # encoder's default CBR band rather than one hard-coded magic
        # number.
        self.assertGreaterEqual(bitrate, 32000)
        self.assertLessEqual(bitrate, 320000)

        # The samples must actually carry the audio content, not just be the
        # right length: a 440 Hz sine must read back as a non-silent signal
        # whose dominant spectral peak is 440 Hz.
        self.assertGreater(np.max(np.abs(samples)), 0.0)
        self.assertTrue(np.all(np.isfinite(samples)))
        windowed = (samples - np.mean(samples)) * np.hanning(len(samples))
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
        dominant_freq = freqs[np.argmax(spectrum)]
        self.assertAlmostEqual(dominant_freq, 440.0, delta=5.0)

    def test_write_audio(self):
        sample_rate, samples, bitrate, audio = read_audio(
            self.test_mp3_file, audio_format='mp3'
        )
        output_file = 'test_output.flac'
        write_audio(output_file, sample_rate, samples, audio_format='flac')
        try:
            self.assertTrue(os.path.exists(output_file))

            info = sf.info(output_file)
            # Sample rate and channel count must be preserved by the round
            # trip.
            self.assertEqual(info.samplerate, sample_rate)
            self.assertEqual(info.channels, audio.channels)
            # The written container must actually be the 24-bit FLAC
            # write_audio() documents/requests (subtype='PCM_24'). Bit depth
            # is the one thing this MP3 -> FLAC step genuinely upscales
            # (fat_llama adds precision/headroom, not bandwidth), so a
            # silent regression to PCM_16 -- or to a FLAC-in-name-only
            # container -- would go unnoticed by every other assertion
            # here, all of which operate on soundfile's float view of the
            # samples and are indifferent to the stored bit depth.
            self.assertEqual(
                info.format, 'FLAC',
                "write_audio(audio_format='flac') did not produce a FLAC "
                f"container (got {info.format})."
            )
            self.assertEqual(
                info.subtype, 'PCM_24',
                "write_audio(audio_format='flac') did not write 24-bit PCM "
                f"(got {info.subtype}); bit-depth headroom is the "
                "precision upscale this path is meant to deliver."
            )
            flac_info = FLAC(output_file)
            self.assertEqual(
                flac_info.info.bits_per_sample, 24,
                "FLAC metadata reports "
                f"{flac_info.info.bits_per_sample} bits per sample rather "
                "than the 24 write_audio() requests."
            )
            # Duration must match the ~1 second input within a small
            # tolerance.
            self.assertAlmostEqual(
                info.duration, len(audio) / 1000.0, delta=0.05
            )

            written_data, written_sr = sf.read(output_file)
            self.assertEqual(written_sr, sample_rate)
            # The written audio must not be silence.
            self.assertGreater(np.max(np.abs(written_data)), 0.0)
            self.assertTrue(np.all(np.isfinite(written_data)))

            # The written file must carry the *same waveform*, not merely
            # some non-silent audio of the right length. write_audio() only
            # peak-normalizes, which is a pure scalar gain change, so the
            # normalized input and the written samples must match sample for
            # sample and the 440 Hz sine must survive the round trip.
            # Without this, the test would pass on any arbitrary non-silent
            # signal.
            normalized_input = samples / np.max(np.abs(samples))
            n = min(len(normalized_input), len(written_data))
            self.assertGreater(
                np.corrcoef(normalized_input[:n], written_data[:n])[0, 1],
                0.999,
                "Written FLAC waveform does not track the input waveform; "
                "write_audio() should only apply a scalar peak "
                "normalization."
            )

            windowed = (
                (written_data - np.mean(written_data))
                * np.hanning(len(written_data))
            )
            spectrum = np.abs(np.fft.rfft(windowed))
            freqs = np.fft.rfftfreq(len(written_data), 1.0 / written_sr)
            self.assertAlmostEqual(
                freqs[np.argmax(spectrum)], 440.0, delta=5.0,
                msg="Dominant frequency of the written FLAC is not the "
                    "440 Hz of the source sine wave."
            )
            # The written audio must not be catastrophically clipped:
            # write_audio() is documented to accept the samples produced by
            # read_audio(), whose magnitude is on the raw PCM scale (tens
            # of thousands), not normalized to [-1, 1]. If write_audio()
            # fails to normalize/scale before handing data to soundfile
            # with an integer subtype, nearly every sample gets clamped to
            # full scale, destroying the waveform.
            clipped_fraction = np.mean(np.abs(written_data) > 0.999)
            self.assertLess(
                clipped_fraction, 0.05,
                f"{clipped_fraction:.2%} of written samples are clipped to "
                "full scale; write_audio() likely wrote un-normalized/"
                "out-of-range data directly with an integer subtype "
                "instead of scaling it to [-1, 1] first."
            )
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_write_audio_normalize_false_preserves_relative_level(self):
        # Regression test for Issue #18 item (2): toggle_normalize is
        # documented ("Whether to normalize the audio") as an optional
        # pipeline stage, but write_audio() used to unconditionally
        # peak-normalize (divide by the buffer's own max abs value) no
        # matter what upstream toggle_normalize did -- so the final
        # written file's peak amplitude always landed at exactly 1.0
        # (0 dBFS) regardless of the toggle. Pure-numpy, no GPU needed:
        # write_audio() itself only ever receives plain np.ndarray data.
        sample_rate = 44100
        n = 4410
        t = np.arange(n) / sample_rate
        # A "quiet" raw-PCM-scale tone: peak well below a 16-bit source's
        # own full-scale reference (32768.0), the same relationship
        # upscale()'s pipeline produces when toggle_autoscale=True rescales
        # a channel back to its original (non-full-scale) peak and
        # toggle_normalize=False skips the subsequent peak-normalize step.
        reference_amplitude = 32768.0
        original_peak = 8192.0
        data = original_peak * np.sin(2 * np.pi * 440 * t)

        normalized_file = 'test_output_normalize_true.flac'
        preserved_file = 'test_output_normalize_false.flac'
        try:
            write_audio(
                normalized_file, sample_rate, data.copy(),
                audio_format='flac', normalize=True
            )
            write_audio(
                preserved_file, sample_rate, data.copy(),
                audio_format='flac', normalize=False,
                reference_amplitude=reference_amplitude
            )

            normalized_data, _ = sf.read(normalized_file)
            preserved_data, _ = sf.read(preserved_file)

            normalized_peak = float(np.max(np.abs(normalized_data)))
            preserved_peak = float(np.max(np.abs(preserved_data)))

            # normalize=True must keep the pre-existing behavior: the
            # written peak sits at (essentially) full scale.
            self.assertAlmostEqual(
                normalized_peak, 1.0, delta=0.01,
                msg="normalize=True should still peak-normalize to full "
                    "scale (pre-existing, backward-compatible behavior)."
            )
            # normalize=False must NOT be renormalized to full scale --
            # it should reflect the original signal's level relative to
            # reference_amplitude (8192 / 32768 = 0.25), not 1.0.
            expected_preserved_peak = original_peak / reference_amplitude
            self.assertAlmostEqual(
                preserved_peak, expected_preserved_peak, delta=0.01,
                msg="normalize=False did not preserve the original signal "
                    "level; write_audio() appears to still be forcing a "
                    "full-scale peak-normalize regardless of the flag."
            )
            self.assertLess(
                preserved_peak, normalized_peak - 0.1,
                "normalize=False produced a peak indistinguishable from "
                "normalize=True's full-scale peak -- toggle_normalize has "
                "no measurable effect on the written file."
            )
        finally:
            for f in (normalized_file, preserved_file):
                if os.path.exists(f):
                    os.remove(f)

    def test_write_audio_wav_uses_64bit_float_and_is_lossless(self):
        # Regression test for Issue #18 item (1): fat_llama's internal
        # computation is already float64/complex128 throughout feed.py,
        # but write_audio() was quantizing every output (both flac and
        # wav) to 24-bit integer PCM ('PCM_24') -- discarding that
        # precision at the very last step. FLAC has no true float/double
        # subtype (libsndfile's FLAC ceiling is PCM_24 -- confirmed via
        # sf.available_subtypes('FLAC')), so 24-bit is already FLAC's own
        # real ceiling and stays unchanged (see test_write_audio above).
        # WAV, however, supports a genuine 64-bit float subtype
        # ('DOUBLE'), which stores the exact float64 values with no
        # quantization or clamping at all (verified directly:
        # sf.write(..., subtype='DOUBLE') round-trips values bit-for-bit,
        # even ones outside [-1, 1], unlike PCM_24 which clamps).
        sample_rate = 44100
        n = 2000
        t = np.arange(n) / sample_rate
        reference_amplitude = 32768.0
        original_peak = 12345.6789
        data = original_peak * np.sin(2 * np.pi * 300 * t)

        output_file = 'test_output_wav_double.wav'
        try:
            write_audio(
                output_file, sample_rate, data.copy(), audio_format='wav',
                normalize=False, reference_amplitude=reference_amplitude
            )

            info = sf.info(output_file)
            self.assertEqual(
                info.subtype, 'DOUBLE',
                "write_audio(audio_format='wav') did not use the 64-bit "
                f"float 'DOUBLE' subtype (got {info.subtype}); WAV "
                "supports true 64-bit float and this pipeline computes "
                "in float64/complex128 throughout, so the final write "
                "should not quantize to a lower-precision integer subtype."
            )

            written_data, written_sr = sf.read(output_file, dtype='float64')
            self.assertEqual(written_sr, sample_rate)
            # Reconstruct the pre-write signal at write_audio's own scale
            # (data / reference_amplitude, same convention as the
            # normalize=False FLAC test above) and confirm the round trip
            # is lossless to float64 precision, not merely "close" the way
            # a 24-bit quantization step would be.
            expected = data / reference_amplitude
            np.testing.assert_allclose(
                written_data, expected, rtol=1e-9, atol=1e-12,
                err_msg="WAV+DOUBLE round trip is not losslessly precise; "
                        "write_audio() may still be rescaling/quantizing "
                        "the data instead of writing the true float64 "
                        "values."
            )
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_compute_upscale_factor_bounds_realistic_sample_rate(self):
        # Regression test for issue #20: upscale() with target_bitrate_kbps
        # values well within their documented valid range (800-1411 for
        # flac, 800-6444 for wav) against a typical compressed mp3 source
        # bitrate (128-192 kbps) used to derive an unbounded ratio
        # (round(target_bitrate_kbps * 1000 / source_bitrate)) that
        # regularly landed at 5-7+, driving output sample rates past
        # 250-300 kHz -- far outside any realistic playback/DAC rate.
        # compute_upscale_factor is pure Python (no CuPy/CUDA dependency),
        # so this exercises the actual shipped fix directly, without
        # needing a GPU.
        sample_rate = 44100

        # The issue's own reported scenario: target_bitrate_kbps=900 (and
        # a nearby 1400) against typical compressed mp3 bitrates. The old
        # formula (round(900_000 / 128_000) = 7, round(1_400_000 /
        # 192_000) = 7) drove sample_rate * 7 = 308700 Hz.
        for source_bitrate_bps, target_bitrate_kbps in (
            (128000, 900), (192000, 1400), (64000, 800), (320000, 1411),
        ):
            factor = compute_upscale_factor(
                sample_rate, source_bitrate_bps, target_bitrate_kbps
            )
            new_sample_rate = sample_rate * factor
            self.assertGreaterEqual(
                factor, 1,
                "compute_upscale_factor must never derive a downscale "
                f"(factor={factor} for source_bitrate={source_bitrate_bps}"
                f", target_bitrate_kbps={target_bitrate_kbps})."
            )
            self.assertLessEqual(
                new_sample_rate, MAX_REALISTIC_SAMPLE_RATE_HZ,
                f"compute_upscale_factor produced an unrealistic sample "
                f"rate ({new_sample_rate} Hz) for source_bitrate="
                f"{source_bitrate_bps}, target_bitrate_kbps="
                f"{target_bitrate_kbps}; this reproduces issue #20's "
                "unrealistic sample rate/bitrate report."
            )

        # Unknown source bitrate falls back to a small, bounded factor
        # rather than raising or defaulting to something unbounded.
        fallback_factor = compute_upscale_factor(sample_rate, None, 1400)
        self.assertGreaterEqual(fallback_factor, 1)
        self.assertLessEqual(
            sample_rate * fallback_factor, MAX_REALISTIC_SAMPLE_RATE_HZ
        )

        # An already-high-bitrate source (e.g. a lossless/high-bitrate
        # input) must clamp to a factor of 1 (no downscale), not derive a
        # fractional/zero factor.
        self.assertEqual(
            compute_upscale_factor(sample_rate, 1300000, 800), 1
        )

        # A source sample rate already at or above the realistic ceiling
        # must not be upscaled further.
        self.assertEqual(
            compute_upscale_factor(192000, 64000, 1400), 1
        )

    def test_lms_block_ranges_partitions_range_exactly(self):
        # Regression test for the block-partitioning logic behind
        # lms_filter's block-adaptive update (issue #20's runtime fix):
        # _lms_block_ranges must cover [start, n) exactly once, in order,
        # with no gaps or overlaps, and every block except possibly the
        # last must be exactly block_size long. This is pure Python (no
        # CuPy), so it directly exercises the shipped partitioning logic
        # without needing a GPU -- the per-block filtering math itself
        # still requires cp.ndarray input and is covered by lms_filter's
        # own (GPU-gated) regression tests.
        for start, n, block_size in (
            (33, 1000, 256), (0, 10, 3), (5, 5, 4), (0, 1, 1), (10, 11, 5),
        ):
            ranges = list(_lms_block_ranges(start, n, block_size))
            covered = []
            for index, (block_start, block_end) in enumerate(ranges):
                self.assertLess(block_start, block_end)
                self.assertLessEqual(block_end - block_start, block_size)
                # Every block except the last must be exactly block_size
                # long -- the whole point of the issue #20 runtime fix is
                # that the number of sequential (kernel-launching) Python
                # iterations drops to ~n / block_size. A partition that
                # emitted short blocks would silently give back that
                # speedup while still passing the coverage check below.
                if index < len(ranges) - 1:
                    self.assertEqual(
                        block_end - block_start, block_size,
                        f"_lms_block_ranges(start={start}, n={n}, "
                        f"block_size={block_size}) emitted a short "
                        f"non-final block [{block_start}, {block_end}); "
                        "only the final block may be shorter than "
                        "block_size."
                    )
                covered.extend(range(block_start, block_end))
            self.assertEqual(
                covered, list(range(start, n)),
                f"_lms_block_ranges(start={start}, n={n}, "
                f"block_size={block_size}) does not exactly partition "
                "[start, n) -- found a gap or overlap."
            )

    @requires_gpu
    def test_lms_filter_no_extended_warmup_dropout(self):
        # Regression test: lms_filter() used to zero-initialize both its
        # tap weights and its output buffer for the first num_taps samples,
        # causing the filtered signal to ramp up from near-silence over many
        # samples before tracking the actual input (observed in production
        # as a ~200ms, -82 dBFS dropout at the head of upscaled audio with
        # no corresponding silence in the source). upscale() calls
        # lms_filter(channel, channel, ...) -- signal and desired are the
        # same array -- so a properly warmed-up filter should already be
        # tracking the input's magnitude immediately past the initial taps,
        # not ramping up from zero.
        sr = 44100
        num_taps = 32
        t = cp.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
        signal = (
            0.5 * cp.sin(2 * cp.pi * 300 * t)
            + 0.2 * cp.sin(2 * cp.pi * 900 * t)
        )

        filtered = lms_filter(signal, signal, mu=0.001, num_taps=num_taps)

        signal_np = cp.asnumpy(signal)
        filtered_np = cp.asnumpy(filtered)

        early_filtered = filtered_np[num_taps:num_taps + 50]
        early_signal = signal_np[num_taps:num_taps + 50]
        early_filtered_rms = np.sqrt(np.mean(early_filtered ** 2))
        early_signal_rms = np.sqrt(np.mean(early_signal ** 2))

        self.assertGreater(
            early_filtered_rms / early_signal_rms, 0.5,
            f"Filtered output RMS immediately after warm-up "
            f"({early_filtered_rms:.4f}) is far below the input's own RMS "
            f"in that same window ({early_signal_rms:.4f}); lms_filter() "
            "is likely ramping up from a zero-initialized state instead "
            "of tracking the signal from (near) the first sample."
        )

    @requires_gpu
    def test_ist_harmonic_injection_bounded_across_iterations(self):
        # Regression test: iterative_soft_thresholding() used to add a
        # fixed 0.1-amplitude sinusoid every single iteration with no
        # decay/bound. Because data_thres carries forward from one
        # iteration to the next and the harmonic term's FFT magnitude
        # (proportional to array length) trivially survives the fixed
        # absolute `threshold`, that injection accumulated roughly
        # linearly with max_iter instead of converging -- measured
        # (audio-quality-checker, baseline max_iterations=300 pipeline
        # run): a +6.12 dB broadband in-band noise-floor rise relative to
        # the input. The output magnitude should stay bounded regardless
        # of how many iterations run, not scale with max_iter.
        n = 2000
        t = cp.linspace(0, 1, n, endpoint=False)
        data = (
            0.8 * cp.sin(2 * cp.pi * 300 * t)
            + 0.3 * cp.sin(2 * cp.pi * 700 * t)
        )

        out_few_iters = iterative_soft_thresholding(data.copy(), 5, 0.6)
        out_many_iters = iterative_soft_thresholding(data.copy(), 150, 0.6)

        max_abs_few = float(cp.max(cp.abs(out_few_iters)))
        max_abs_many = float(cp.max(cp.abs(out_many_iters)))

        self.assertLess(
            max_abs_many, max_abs_few * 2,
            "iterative_soft_thresholding()'s output magnitude grows with "
            f"max_iter (max|.|={max_abs_few:.3f} at 5 iterations vs "
            f"{max_abs_many:.3f} at 150 iterations) instead of staying "
            "bounded; the per-iteration harmonic injection term is likely "
            "accumulating unboundedly instead of being scaled/bounded "
            "across max_iter."
        )

    @requires_gpu
    def test_lms_filter_self_referential_call_genuinely_adapts(self):
        # Regression test for a cycle 3 finding: upscale() always calls
        # lms_filter(channel, channel) -- signal and desired are the SAME
        # array. Before this fix, tap 0 of the filter's input vector was
        # signal[i] itself (the exact sample being predicted), and the
        # cycle 1 identity initialization (w = [1, 0, ..., 0]) made
        # y == desired[i] exactly on every iteration: the LMS error term
        # was identically zero forever, so the weights never moved and the
        # filtered output was bit-identical to the input -- an expensive
        # (~18-19 of the pipeline's ~20 minute runtime) no-op, not an
        # adaptive filter. The fix introduces a decorrelation lag (delay)
        # between the predictor's taps and the sample being predicted, so
        # even in the self-referential case there is a real (if small)
        # estimation problem and the weights must move to reduce it.
        sr = 44100
        num_taps = 32
        t = cp.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
        signal = (
            0.5 * cp.sin(2 * cp.pi * 300 * t)
            + 0.2 * cp.sin(2 * cp.pi * 900 * t)
        )

        filtered, w_final = lms_filter(
            signal, signal, mu=0.001, num_taps=num_taps,
            return_weights=True
        )

        # The weights must have moved from the identity-pass-through
        # initialization -- if they haven't, the filter never adapted.
        w_initial = cp.zeros(num_taps, dtype=cp.float64)
        w_initial[0] = 1.0
        self.assertFalse(
            bool(cp.allclose(w_final, w_initial)),
            "lms_filter's tap weights are unchanged from their initial "
            "value after a full run with signal == desired; the adaptive "
            "filter did not adapt (likely a degenerate zero-error "
            "self-referential case, i.e. the cycle 3 no-op bug)."
        )

        # The filtered output must not be a bit-identical copy of the
        # input -- that was the direct, measurable symptom of the no-op
        # bug (e was identically zero, so y == desired[i] == signal[i]
        # exactly every sample).
        start = num_taps + 1
        self.assertFalse(
            bool(cp.allclose(filtered[start:], signal[start:])),
            "lms_filter's output is bit-identical to its input for the "
            "signal == desired case; the filter is acting as a pure "
            "pass-through instead of genuinely adapting."
        )

        # The fix must not reintroduce the cycle 1 warm-up dropout: the
        # filtered output must already be tracking the input's magnitude
        # immediately after warm-up, not ramping up from near-silence.
        signal_np = cp.asnumpy(signal)
        filtered_np = cp.asnumpy(filtered)
        early_filtered = filtered_np[start:start + 50]
        early_signal = signal_np[start:start + 50]
        early_filtered_rms = np.sqrt(np.mean(early_filtered ** 2))
        early_signal_rms = np.sqrt(np.mean(early_signal ** 2))
        self.assertGreater(
            early_filtered_rms / early_signal_rms, 0.5,
            "lms_filter's decorrelation-delay fix reintroduced a warm-up "
            "dropout: filtered RMS immediately after warm-up "
            f"({early_filtered_rms:.4f}) is far below the input's own "
            f"RMS in that window ({early_signal_rms:.4f})."
        )

    @requires_gpu
    def test_lms_filter_block_size_one_matches_reference_per_sample_update(
        self
    ):
        # Regression test for a coverage gap flagged by audio-quality-
        # checker: lms_filter's own docstring claims "block_size=1
        # reproduces the exact prior per-sample update" (verified there
        # only algebraically, in prose), but no test asserted it -- a
        # future change to the block-averaged gradient formula (e.g. an
        # off-by-one in block_len, or applying the block mean even when
        # block_len == 1) could silently break the claimed equivalence
        # while every other test (which only exercises the default
        # block_size=256) kept passing. This builds an independent
        # reference implementation of the exact per-sample LMS update
        # described in lms_filter's own comments (same w initialization,
        # same delay convention, same "2 * mu * e * x" update term, no
        # block averaging -- just a plain per-sample Python loop) and
        # checks it against lms_filter(..., block_size=1).
        sr = 44100
        num_taps = 16
        mu = 0.001
        delay = 1
        t = cp.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
        signal = (
            0.5 * cp.sin(2 * cp.pi * 300 * t)
            + 0.2 * cp.sin(2 * cp.pi * 900 * t)
        )

        def reference_per_sample_lms(sig, desired):
            n = len(sig)
            w = cp.zeros(num_taps, dtype=cp.float64)
            w[0] = 1.0
            filtered = cp.zeros(n, dtype=cp.float64)
            start = num_taps + delay
            filtered[:start] = sig[:start]
            for i in range(start, n):
                y = cp.float64(0.0)
                for k in range(num_taps):
                    y = y + w[k] * sig[i - delay - k]
                e = desired[i] - y
                for k in range(num_taps):
                    w[k] = w[k] + 2 * mu * e * sig[i - delay - k]
                w = cp.clip(w, -1e10, 1e10)
                filtered[i] = y
            return filtered, w

        ref_filtered, ref_w = reference_per_sample_lms(signal, signal)
        block_filtered, block_w = lms_filter(
            signal, signal, mu=mu, num_taps=num_taps, delay=delay,
            block_size=1, return_weights=True
        )

        self.assertTrue(
            bool(cp.allclose(block_filtered, ref_filtered, atol=1e-9)),
            "lms_filter(block_size=1) output does not match an "
            "independent per-sample LMS reference implementation of the "
            "same update rule; the docstring's claim that block_size=1 "
            "reproduces the exact prior per-sample update no longer "
            "holds."
        )
        self.assertTrue(
            bool(cp.allclose(block_w, ref_w, atol=1e-9)),
            "lms_filter(block_size=1) final tap weights do not match an "
            "independent per-sample LMS reference implementation; the "
            "block_size=1 equivalence claim does not hold for the "
            "weight-update path."
        )

    @requires_gpu
    def test_lms_filter_block_size_bounds_sequential_iterations(self):
        # Regression test for a coverage gap flagged by audio-quality-
        # checker: lms_filter's docstring claims the block-adaptive
        # rewrite (issue #20) "cuts the number of sequential Python-loop
        # iterations ... from n to roughly n / block_size" with a default
        # block_size of 256, but nothing asserted either the default or
        # the iteration count itself. A regression that silently made
        # every block length 1 sample (e.g. lms_filter no longer passing
        # its block_size argument through to _lms_block_ranges, or the
        # default being changed) would pass every other test in this
        # module (which only check output values, not how many
        # sequential iterations produced them) while quietly
        # reintroducing the exact per-sample-loop runtime issue that
        # issue #20 reported (27.5 minutes for a 15.2s source).
        default_block_size = inspect.signature(lms_filter).parameters[
            'block_size'
        ].default
        self.assertEqual(
            default_block_size, 256,
            "lms_filter's default block_size changed from the "
            "documented 256; this silently changes the default runtime/"
            "accuracy tradeoff described in its docstring."
        )

        sr = 44100
        num_taps = 16
        t = cp.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
        signal = (
            0.5 * cp.sin(2 * cp.pi * 300 * t)
            + 0.2 * cp.sin(2 * cp.pi * 900 * t)
        )
        n = len(signal)
        start = num_taps + 1  # delay defaults to 1

        call_counts = []

        def counting_block_ranges(block_start, block_n, block_size):
            ranges = list(
                _lms_block_ranges(block_start, block_n, block_size)
            )
            call_counts.append(len(ranges))
            return iter(ranges)

        with mock.patch.object(
            feed_module, '_lms_block_ranges',
            side_effect=counting_block_ranges
        ):
            lms_filter(signal, signal, num_taps=num_taps, block_size=256)
            lms_filter(signal, signal, num_taps=num_taps, block_size=1)

        iterations_default, iterations_per_sample = call_counts
        expected_default_iterations = math.ceil((n - start) / 256)

        self.assertEqual(
            iterations_default, expected_default_iterations,
            "lms_filter's default block_size=256 run did not perform "
            "the expected number of sequential block iterations "
            f"(expected {expected_default_iterations}, got "
            f"{iterations_default})."
        )
        self.assertEqual(
            iterations_per_sample, n - start,
            "lms_filter(block_size=1) did not perform exactly one "
            f"sequential iteration per sample (expected {n - start}, "
            f"got {iterations_per_sample}); this is exactly the "
            "per-sample-loop behavior issue #20's block-adaptive "
            "rewrite was meant to replace."
        )
        self.assertLess(
            iterations_default, iterations_per_sample / 100,
            "lms_filter's default block_size=256 only cut sequential "
            f"iterations from {iterations_per_sample} to "
            f"{iterations_default}, far short of the roughly "
            "two-orders-of-magnitude reduction issue #20's fix "
            "documents; this would erode the fix's runtime improvement."
        )

    @requires_gpu
    def test_ist_no_static_floor_in_quiet_segment(self):
        # Regression test for a cycle 4 finding: cycle 3's harmonic-
        # reconstruction term derived its frequency from cp.argmax of the
        # ENTIRE buffer's masked FFT every iteration. A whole-buffer FFT
        # has one global dominant bin, so for any real multi-thousand-
        # sample upscaled channel that "content-derived" frequency was
        # actually static across all iterations and the whole track -- a
        # constant, non-source tone (measured by audio-quality-checker: a
        # 98.168 Hz component at -28.7 dBFS spanning the entire output),
        # not time-varying detail. Its amplitude was also derived once
        # from the buffer's global peak and applied uniformly regardless
        # of local signal level, so it acted as a hard floor that
        # collapsed measured dynamic range in quiet passages from 57.1 dB
        # (reference) to 25.4 dB (output). This cycle removed the
        # harmonic term entirely (see iterative_soft_thresholding's own
        # docstring) rather than attempting a fourth revision of it.
        #
        # The test this replaces (test_ist_harmonic_term_lands_in_
        # audible_band) only ever exercised a short, single-segment
        # n=2000 buffer, where a single dominant bin is genuinely
        # representative of the whole signal -- it could never have
        # caught this failure mode. This test instead builds a two-
        # segment, longer buffer (a loud segment followed by a much
        # quieter one, both at real-PCM-like amplitude scale) and
        # exercises iterative_soft_thresholding's output the same way
        # upscale_channels actually uses it (added back onto the original
        # signal, not used standalone), then checks that the quiet
        # segment's level does not rise far above its pre-IST level --
        # i.e. that IST does not inject a static, content-independent
        # floor.
        sr = 44100
        t_loud = cp.arange(sr, dtype=cp.float64) / sr  # 1s
        t_quiet = cp.arange(sr, dtype=cp.float64) / sr  # 1s
        loud_segment = 20000.0 * cp.sin(2 * cp.pi * 400 * t_loud)
        quiet_segment = 5.0 * cp.sin(2 * cp.pi * 400 * t_quiet)
        data = cp.concatenate([loud_segment, quiet_segment])
        n_loud = len(loud_segment)

        max_iter = 20
        threshold = 0.6

        ist_changes = iterative_soft_thresholding(
            data.copy(), max_iter, threshold
        )
        # Matches upscale_channels' actual usage: the interpolated signal
        # plus IST's returned value, not IST's output taken standalone.
        combined = data + ist_changes

        quiet_rms_before = float(cp.sqrt(cp.mean(quiet_segment ** 2)))
        quiet_rms_after = float(cp.sqrt(cp.mean(combined[n_loud:] ** 2)))

        # A generous bound: allow up to a ~4x (12 dB) rise, which covers
        # this function's own separately-documented, pre-existing
        # near-lossless-round-trip "doubling" effect (threshold barely
        # masks anything at real PCM scale, so IST's output is close to a
        # second copy of the input added back on top) without allowing a
        # large, content-independent static floor like the one this test
        # guards against (measured, cycle 3 regression: >50 dB rise in an
        # equivalent synthetic case).
        self.assertLess(
            quiet_rms_after, quiet_rms_before * 4.0,
            "iterative_soft_thresholding (as combined by upscale_channels) "
            f"raised the quiet segment's RMS from {quiet_rms_before:.4g} "
            f"to {quiet_rms_after:.4g} -- a "
            f"{20 * math.log10(quiet_rms_after / quiet_rms_before):.1f} dB "
            "rise -- consistent with a static, content-independent tone/"
            "floor being injected rather than genuine local detail."
        )

    @requires_gpu
    def test_new_interpolation_algorithm_is_bandlimited(self):
        # Regression test for a cycle 3 finding: new_interpolation_
        # algorithm used zero-order-hold duplication (each sample
        # repeated upscale_factor times), which injects strong mirrored
        # spectral images above the original Nyquist frequency (measured,
        # audio-quality-checker: near 44.1/88.2/132.3 kHz for a 7x
        # upscale of 44.1 kHz audio) instead of genuine added detail, and
        # left iterative_soft_thresholding little headroom to add real
        # content since the ZOH-duplicated shape dominated the waveform.
        # A bandlimited (FFT zero-padding) interpolation should introduce
        # no new spectral content above the original Nyquist frequency.
        sr = 44100
        n = 4410  # 0.1s of audio
        t = cp.linspace(0, n / sr, n, endpoint=False)
        tone = 10000.0 * cp.sin(2 * cp.pi * 300 * t)
        upscale_factor = 7

        expanded = new_interpolation_algorithm(tone, upscale_factor)

        self.assertEqual(len(expanded), n * upscale_factor)
        self.assertTrue(
            bool(cp.all(cp.isfinite(expanded))),
            "new_interpolation_algorithm produced non-finite output."
        )

        new_sr = sr * upscale_factor
        spectrum = cp.abs(cp.fft.rfft(expanded))
        freqs = cp.fft.rfftfreq(len(expanded), 1.0 / new_sr)
        original_nyquist = sr / 2.0

        below_nyquist_peak = float(cp.max(spectrum[freqs <= original_nyquist]))
        above_nyquist_peak = float(cp.max(spectrum[freqs > original_nyquist]))

        # The imaging artifact this replaces would put energy comparable
        # to the below-Nyquist peak at mirrored image frequencies above
        # the original Nyquist; a genuinely bandlimited interpolation
        # should leave that band close to the FFT's own floating-point
        # noise floor, many orders of magnitude below the real content.
        self.assertLess(
            above_nyquist_peak, below_nyquist_peak * 1e-4,
            "new_interpolation_algorithm introduced significant spectral "
            f"energy above the original Nyquist frequency (peak "
            f"{above_nyquist_peak:.3g} vs in-band peak "
            f"{below_nyquist_peak:.3g}); this looks like zero-order-hold "
            "imaging rather than bandlimited interpolation."
        )

        # The interpolated tone must still be recognizable as the same
        # 300 Hz content, not distorted into something else.
        dominant_freq = float(
            freqs[cp.argmax(spectrum[freqs <= original_nyquist])]
        )
        self.assertAlmostEqual(dominant_freq, 300.0, delta=5.0)

    @requires_gpu
    def test_apply_original_nyquist_cutoff_removes_above_nyquist_content(
        self
    ):
        # Regression test for this cycle's fix: fat_llama's design (per
        # .claude/rules/project-mission.md's "no content above the
        # original Nyquist frequency" constraint) requires the band above
        # the *original* source's Nyquist frequency to be actively kept
        # silent, not merely left clean as an emergent property of
        # whichever stages happen to run beforehand. This test simulates
        # what would happen if some future upstream stage (IST's harmonic
        # term, autoscale, normalize, LMS) reintroduced genuine energy
        # above the original Nyquist: it builds a synthetic "already
        # fully processed" signal containing both an in-band tone (well
        # below the original Nyquist) and an out-of-band tone (above the
        # original Nyquist but below the upsampled Nyquist), then checks
        # that apply_original_nyquist_cutoff removes the out-of-band tone
        # to near the FFT noise floor while leaving the in-band tone
        # essentially untouched.
        original_sample_rate = 44100
        upscale_factor = 2
        new_sample_rate = original_sample_rate * upscale_factor
        duration = 0.05
        n = int(new_sample_rate * duration)
        t = cp.linspace(0, duration, n, endpoint=False)

        in_band_freq = 300.0  # well below the 22050 Hz original Nyquist
        above_nyquist_freq = 30000.0  # above 22050, below the 44100 new
        # Nyquist -- stands in for artifact energy some future stage
        # might reintroduce above the original Nyquist.
        signal = (
            cp.sin(2 * cp.pi * in_band_freq * t)
            + cp.sin(2 * cp.pi * above_nyquist_freq * t)
        )

        freqs = cp.fft.rfftfreq(n, d=1.0 / new_sample_rate)
        original_nyquist = original_sample_rate / 2.0
        in_band_mask = freqs <= original_nyquist
        above_mask = freqs > original_nyquist

        spectrum_before = cp.abs(cp.fft.rfft(signal))
        in_band_peak_before = float(cp.max(spectrum_before[in_band_mask]))
        above_peak_before = float(cp.max(spectrum_before[above_mask]))
        # Sanity check the synthetic signal actually carries comparable
        # energy in both bands before the cutoff -- otherwise this test
        # would pass trivially without exercising the fix.
        self.assertGreater(
            above_peak_before, in_band_peak_before * 0.5,
            "Test signal construction failed to place comparable energy "
            "above the original Nyquist frequency; the test would not "
            "meaningfully exercise apply_original_nyquist_cutoff."
        )

        cutoff_signal = apply_original_nyquist_cutoff(
            signal, original_sample_rate, new_sample_rate
        )
        self.assertEqual(len(cutoff_signal), n)
        self.assertTrue(bool(cp.all(cp.isfinite(cutoff_signal))))

        spectrum_after = cp.abs(cp.fft.rfft(cutoff_signal))
        in_band_peak_after = float(cp.max(spectrum_after[in_band_mask]))
        above_peak_after = float(cp.max(spectrum_after[above_mask]))

        self.assertLess(
            above_peak_after, in_band_peak_before * 1e-6,
            "apply_original_nyquist_cutoff left significant spectral "
            f"content above the original Nyquist frequency (peak "
            f"{above_peak_after:.3g} vs in-band peak "
            f"{in_band_peak_before:.3g} before cutoff); the guarantee "
            "that no content survives above the original Nyquist "
            "frequency does not hold."
        )
        # The in-band tone must survive essentially unchanged -- the
        # cutoff must not damage real, in-bandwidth content.
        self.assertAlmostEqual(
            in_band_peak_after, in_band_peak_before,
            delta=in_band_peak_before * 0.05,
            msg="apply_original_nyquist_cutoff altered in-band spectral "
                "content it should have left untouched."
        )
        dominant_freq = float(freqs[cp.argmax(spectrum_after)])
        self.assertAlmostEqual(dominant_freq, in_band_freq, delta=5.0)

    @requires_gpu
    def test_upscale_no_content_above_original_nyquist_frequency(self):
        # Regression test for this cycle's fix: verifies the guarantee
        # holds through a real (if fast/small) end-to-end upscale() call,
        # not just for the apply_original_nyquist_cutoff unit in
        # isolation -- confirming it is actually wired into the pipeline
        # as the final stage. Checked at two different upscale_factors
        # (via two target_bitrate_kbps values) since the cutoff's
        # correctness depends on the original/new sample-rate ratio, not
        # just a single hard-coded case. toggle_adaptive_filter=False and
        # max_iterations=2 keep this fast. Uses target_format='wav' rather
        # than 'flac' -- both are equally valid target_formats for this
        # check; as of the issue #20 fix, compute_upscale_factor's
        # realistic-sample-rate ceiling means this no longer risks
        # approaching FLAC's own separate ~655350 Hz format ceiling
        # either way.
        original_nyquist = 44100 / 2.0

        for target_bitrate_kbps in (800, 1400):
            output_file = (
                f'test_output_nyquist_{target_bitrate_kbps}.wav'
            )
            try:
                upscale(
                    input_file_path=self.test_mp3_file,
                    output_file_path=output_file,
                    source_format='mp3',
                    target_format='wav',
                    max_iterations=2,
                    threshold_value=0.6,
                    target_bitrate_kbps=target_bitrate_kbps,
                    toggle_normalize=True,
                    toggle_autoscale=True,
                    toggle_adaptive_filter=False,
                )

                out_data, out_sr = sf.read(output_file, always_2d=True)
                mono = out_data[:, 0]
                spectrum = np.abs(np.fft.rfft(mono))
                freqs = np.fft.rfftfreq(len(mono), 1.0 / out_sr)

                in_band_mask = freqs <= original_nyquist
                above_mask = freqs > original_nyquist
                in_band_peak = float(np.max(spectrum[in_band_mask]))

                # There must actually be an above-original-Nyquist band to
                # inspect -- otherwise this test would pass vacuously
                # without ever exercising the cutoff. Both bitrates here
                # drive an upscale_factor well above 1 against this
                # source's deterministic 64 kbps LAME CBR encode, so an
                # empty band means the upscale_factor derivation (or the
                # output sample rate) regressed, which is itself a failure
                # worth surfacing rather than silently skipping.
                self.assertTrue(
                    bool(np.any(above_mask)),
                    f"Output sample rate {out_sr} Hz leaves no band above "
                    f"the original {original_nyquist} Hz Nyquist frequency "
                    "to check; upscale() did not upsample, so this "
                    "regression test could not exercise the cutoff."
                )

                above_peak = float(np.max(spectrum[above_mask]))
                self.assertLess(
                    above_peak, in_band_peak * 1e-4,
                    f"upscale() (target_bitrate_kbps={target_bitrate_kbps}"
                    f") left significant spectral content above the "
                    f"original {original_nyquist} Hz Nyquist frequency "
                    f"(peak {above_peak:.3g} vs in-band peak "
                    f"{in_band_peak:.3g}); the final Nyquist cutoff stage "
                    "does not appear to be applied/effective."
                )
            finally:
                if os.path.exists(output_file):
                    os.remove(output_file)

    @requires_gpu
    def test_upscale_end_to_end_with_adaptive_filter_enabled(self):
        # Regression test for a coverage gap flagged by audio-quality-
        # checker after issue #20's block-adaptive lms_filter rewrite:
        # every existing end-to-end upscale() test passes
        # toggle_adaptive_filter=False to stay fast, so the exact stage
        # issue #20 rewrote (previously impractical to enable at all --
        # 27.5 minutes for the adaptive-filter stage alone on a 15.2s
        # source) had zero pipeline-level coverage; only isolated
        # lms_filter unit tests (built directly against synthetic
        # arrays, never routed through the real upscale() pipeline)
        # exercised it. This runs a real upscale() call with
        # toggle_adaptive_filter=True and a small max_iterations to stay
        # fast, and checks the same coherence properties the other e2e
        # tests check, specifically on the adaptive-filtered output, so
        # a wiring regression in upscale()'s lms_filter call (wrong
        # axis, shape mismatch, numerical instability at real pipeline
        # scale) would be caught even though it wouldn't show up in any
        # isolated lms_filter unit test.
        output_file = 'test_output_adaptive_filter.flac'
        try:
            upscale(
                input_file_path=self.test_mp3_file,
                output_file_path=output_file,
                source_format='mp3',
                target_format='flac',
                max_iterations=2,
                threshold_value=0.6,
                target_bitrate_kbps=800,
                toggle_normalize=True,
                toggle_autoscale=True,
                toggle_adaptive_filter=True,
            )

            info = sf.info(output_file)
            out_data, _ = sf.read(output_file, always_2d=True)

            # Container-level properties must survive the adaptive-filter
            # stage too, not just the sample values: lms_filter returns a
            # same-length array, so the output must keep the bounded
            # sample rate compute_upscale_factor derived and the input's
            # ~1 s duration. Without these, a stage that silently dropped
            # or duplicated samples (e.g. a block-partitioning off-by-one
            # at the tail) would still pass every check below.
            _, _, source_bitrate, _ = read_audio(
                self.test_mp3_file, audio_format='mp3'
            )
            expected_upscale_factor = compute_upscale_factor(
                44100, source_bitrate, 800
            )
            self.assertEqual(
                info.samplerate, 44100 * expected_upscale_factor,
                "Adaptive-filtered output sample rate does not match "
                "compute_upscale_factor's formula."
            )
            self.assertLessEqual(
                info.samplerate, MAX_REALISTIC_SAMPLE_RATE_HZ
            )
            self.assertAlmostEqual(
                info.duration, 1.0, delta=0.05,
                msg="Adaptive-filtered output duration does not match the "
                    "1 s input; lms_filter must not change the signal's "
                    "length."
            )

            self.assertTrue(
                np.all(np.isfinite(out_data)),
                "upscale() with toggle_adaptive_filter=True produced "
                "NaN/Inf samples."
            )
            self.assertGreater(
                np.sqrt(np.mean(out_data ** 2)), 1e-3,
                "upscale() with toggle_adaptive_filter=True produced "
                "(near) silent output."
            )
            self.assertLess(
                np.mean(np.abs(out_data) > 0.999), 0.05,
                "upscale() with toggle_adaptive_filter=True clipped "
                "more than 5% of output samples to full scale."
            )

            mono = out_data[:, 0]
            windowed = (mono - np.mean(mono)) * np.hanning(len(mono))
            spectrum = np.abs(np.fft.rfft(windowed))
            out_freqs = np.fft.rfftfreq(len(mono), 1.0 / info.samplerate)
            self.assertAlmostEqual(
                out_freqs[np.argmax(spectrum)], 440.0, delta=10.0,
                msg="Dominant frequency of the adaptive-filtered "
                    "upscaled output is not (close to) the source's "
                    "440 Hz tone; lms_filter's block-adaptive rewrite "
                    "may be distorting the signal when actually wired "
                    "into upscale()."
            )
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    @requires_gpu
    def test_upscale_toggle_normalize_false_preserves_output_level(self):
        # Regression test for Issue #18 item (2), wired end to end through
        # the real upscale() pipeline (not just write_audio() in
        # isolation): before the fix, write_audio() unconditionally
        # peak-normalized before writing regardless of toggle_normalize,
        # so a toggle_normalize=False run's final output peak was
        # indistinguishable from a toggle_normalize=True run's (both
        # landed at ~1.0 / 0 dBFS). Verified indirectly in this sandbox
        # (no functional CUDA device) via a numpy-backed cupy shim running
        # this exact unmodified source end to end: toggle_normalize=True
        # -> peak 1.0; toggle_normalize=False -> peak ~0.095 for a source
        # attenuated 20 dB below full scale, i.e. genuinely reflecting the
        # source's own relative level instead of being renormalized to
        # full scale.
        #
        # self.test_mp3_file (the shared per-test fixture) is generated at
        # pydub's default volume, which is already near full scale (peak
        # 32766 of a possible 32768) -- autoscale would put a
        # toggle_normalize=False run's output level near full scale too
        # purely because the *source* already is, making the "preserved
        # vs. renormalized to full scale" distinction unobservable. A
        # deliberately quieter (-20 dBFS) fixture makes it observable.
        from pydub.generators import Sine
        quiet_mp3_file = 'test_input_quiet.mp3'
        quiet_sine = Sine(440).to_audio_segment(duration=1000, volume=-20.0)
        out_handle = quiet_sine.export(quiet_mp3_file, format='mp3')
        out_handle.close()

        common_kwargs = dict(
            input_file_path=quiet_mp3_file,
            source_format='mp3',
            target_format='wav',
            max_iterations=2,
            threshold_value=0.6,
            target_bitrate_kbps=800,
            toggle_autoscale=True,
            toggle_adaptive_filter=False,
        )
        normalized_file = 'test_output_toggle_normalize_true.wav'
        preserved_file = 'test_output_toggle_normalize_false.wav'
        try:
            upscale(
                output_file_path=normalized_file, toggle_normalize=True,
                **common_kwargs
            )
            upscale(
                output_file_path=preserved_file, toggle_normalize=False,
                **common_kwargs
            )

            normalized_data, _ = sf.read(normalized_file)
            preserved_data, _ = sf.read(preserved_file)

            normalized_peak = float(np.max(np.abs(normalized_data)))
            preserved_peak = float(np.max(np.abs(preserved_data)))

            self.assertAlmostEqual(
                normalized_peak, 1.0, delta=0.01,
                msg="toggle_normalize=True should still peak-normalize "
                    "the final output to full scale (pre-existing, "
                    "backward-compatible behavior)."
            )
            self.assertLess(
                preserved_peak, normalized_peak - 0.1,
                "toggle_normalize=False produced a final output peak "
                "indistinguishable from toggle_normalize=True's -- the "
                "toggle has no measurable effect on the written file, "
                "meaning write_audio()'s own forced normalization is "
                "not respecting it."
            )
            self.assertGreater(
                preserved_peak, 0.0,
                "toggle_normalize=False produced silence."
            )
        finally:
            for f in (quiet_mp3_file, normalized_file, preserved_file):
                if os.path.exists(f):
                    os.remove(f)

    @requires_gpu
    def test_target_bitrate_kbps_drives_bounded_realistic_upscale_factor(
        self
    ):
        # Regression test for issue #20, documenting the fixed contract of
        # target_bitrate_kbps: it still drives upscale_factor relative to
        # the source file's own bitrate (see compute_upscale_factor), but
        # the derived factor is now clamped so the output sample rate
        # never exceeds MAX_REALISTIC_SAMPLE_RATE_HZ. Before this fix, the
        # unbounded ratio drove sample rates past 250-300 kHz and an
        # effective output bitrate of ~7822 kbps for a comparable input
        # (audio-quality-checker measurement); this test confirms both the
        # sample rate and the real output bitrate now stay within a
        # realistic range end-to-end, not just at the compute_upscale_
        # factor unit level.
        _, _, source_bitrate, _ = read_audio(
            self.test_mp3_file, audio_format='mp3'
        )
        target_bitrate_kbps = 1400  # near the top of the valid flac range,
        # deliberately chosen because it is exactly the kind of value that
        # used to drive an oversized upscale_factor (e.g. round(1400 / 192)
        # = 7) against a typical compressed source bitrate.
        expected_upscale_factor = compute_upscale_factor(
            44100, source_bitrate, target_bitrate_kbps
        )
        self.assertLessEqual(
            44100 * expected_upscale_factor, MAX_REALISTIC_SAMPLE_RATE_HZ,
            "Test setup assumption violated: compute_upscale_factor should "
            "never derive a sample rate above the realistic ceiling."
        )

        output_file = 'test_output_bitrate.flac'
        try:
            upscale(
                input_file_path=self.test_mp3_file,
                output_file_path=output_file,
                source_format='mp3',
                target_format='flac',
                max_iterations=2,
                threshold_value=0.6,
                target_bitrate_kbps=target_bitrate_kbps,
                toggle_normalize=True,
                toggle_autoscale=True,
                toggle_adaptive_filter=False,
            )

            info = sf.info(output_file)
            # The output sample rate must reflect compute_upscale_factor's
            # bounded formula (source_sample_rate * upscale_factor), and
            # must itself stay within the realistic ceiling end-to-end.
            self.assertEqual(
                info.samplerate, 44100 * expected_upscale_factor,
                "Output sample rate does not match compute_upscale_factor's "
                "formula."
            )
            self.assertLessEqual(
                info.samplerate, MAX_REALISTIC_SAMPLE_RATE_HZ,
                f"Output sample rate ({info.samplerate} Hz) exceeds the "
                "realistic playback ceiling; reproduces issue #20's "
                "unrealistic sample rate report."
            )

            # The upscaled output must also be coherent *audio*, not just a
            # file with the right header: without these, the test would pass
            # on an all-silent or all-NaN output of the correct sample rate.
            # Duration is preserved by construction (both the sample count
            # and the sample rate are multiplied by upscale_factor), so the
            # output must still be ~1 second long.
            self.assertAlmostEqual(
                info.duration, 1.0, delta=0.05,
                msg="Upscaled output duration does not match the 1 s input; "
                    "upscale() multiplies both sample count and sample rate "
                    "by upscale_factor, so duration must be preserved."
            )
            out_data, _ = sf.read(output_file, always_2d=True)
            self.assertEqual(out_data.shape[1], 1)
            self.assertTrue(
                np.all(np.isfinite(out_data)),
                "Upscaled output contains NaN/Inf samples."
            )
            self.assertGreater(
                np.sqrt(np.mean(out_data ** 2)), 1e-3,
                "Upscaled output is (near) silence; the pipeline produced no "
                "audible signal."
            )
            self.assertLess(
                np.mean(np.abs(out_data) > 0.999), 0.05,
                "More than 5% of upscaled samples are clipped to full scale."
            )
            # The 440 Hz tone of the source must survive the whole pipeline
            # (interpolation + IST + autoscale + normalize).
            mono = out_data[:, 0]
            windowed = (mono - np.mean(mono)) * np.hanning(len(mono))
            spectrum = np.abs(np.fft.rfft(windowed))
            out_freqs = np.fft.rfftfreq(len(mono), 1.0 / info.samplerate)
            self.assertAlmostEqual(
                out_freqs[np.argmax(spectrum)], 440.0, delta=5.0,
                msg="Dominant frequency of the upscaled output is not the "
                    "440 Hz of the source sine wave; the upscale pipeline "
                    "did not preserve the input's pitch."
            )

            real_bitrate_kbps = (
                os.path.getsize(output_file) * 8 / info.duration / 1000
            )
            # The output is still uncompressed-PCM-then-FLAC-compressed at
            # an upsampled rate, so its real bitrate need not equal
            # target_bitrate_kbps -- but as of the issue #20 fix it must
            # now stay within a realistic ceiling derived from the same
            # MAX_REALISTIC_SAMPLE_RATE_HZ bound: 24-bit mono PCM at that
            # rate is an absolute upper bound on what FLAC (lossless, so
            # never larger than raw PCM) could produce; a small margin
            # covers container/frame overhead.
            max_realistic_bitrate_kbps = (
                MAX_REALISTIC_SAMPLE_RATE_HZ * 24 / 1000
            )
            self.assertLessEqual(
                real_bitrate_kbps, max_realistic_bitrate_kbps * 1.05,
                f"Real output bitrate ({real_bitrate_kbps:.1f} kbps) "
                "exceeds the realistic ceiling implied by "
                f"MAX_REALISTIC_SAMPLE_RATE_HZ "
                f"({max_realistic_bitrate_kbps:.1f} kbps); reproduces "
                "issue #20's unrealistic bitrate report."
            )
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)


if __name__ == '__main__':
    unittest.main()
