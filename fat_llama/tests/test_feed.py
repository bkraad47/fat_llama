import os
import unittest

import cupy as cp
import numpy as np
import soundfile as sf

from fat_llama.audio_fattener.feed import (
    apply_original_nyquist_cutoff, iterative_soft_thresholding, lms_filter,
    new_interpolation_algorithm, read_audio, upscale, write_audio
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
    def test_ist_harmonic_amplitude_scales_with_signal_peak(self):
        # Regression test for a cycle 3 finding: iterative_soft_
        # thresholding's harmonic injection term used a fixed absolute
        # total amplitude (0.1, spread across max_iter iterations) instead
        # of one scaled to the input's own peak amplitude. upscale() never
        # normalizes before calling this function -- `data` is raw-PCM-
        # scale (peak ~1e4-3e4 for 16-bit-sourced audio) -- so a fixed
        # absolute total of 0.1 was an ~1e-5 relative contribution: far too
        # small to survive the pipeline's later autoscale/normalize steps
        # (which only apply a global scalar gain and cannot change that
        # ratio) as measurable added detail. This test isolates the
        # harmonic term's relative contribution using a threshold small
        # enough that essentially nothing gets masked in either the time
        # or frequency domain (so the FFT/IFFT round trip is
        # near-identity and the harmonic term is the dominant source of
        # change), and checks that the relative (not absolute) added
        # contribution is consistent across a large change in the input's
        # absolute scale.
        n = 2000
        t = cp.linspace(0, 1, n, endpoint=False)
        base_shape = (
            cp.sin(2 * cp.pi * 300 * t) + 0.3 * cp.sin(2 * cp.pi * 700 * t)
        )
        small_scale_signal = base_shape * 1.0
        large_scale_signal = base_shape * 10000.0
        negligible_threshold = 1e-9
        max_iter = 10

        out_small = iterative_soft_thresholding(
            small_scale_signal.copy(), max_iter, negligible_threshold
        )
        out_large = iterative_soft_thresholding(
            large_scale_signal.copy(), max_iter, negligible_threshold
        )

        added_small = float(
            cp.max(cp.abs(out_small - small_scale_signal))
        )
        added_large = float(
            cp.max(cp.abs(out_large - large_scale_signal))
        )
        relative_added_small = added_small / float(
            cp.max(cp.abs(small_scale_signal))
        )
        relative_added_large = added_large / float(
            cp.max(cp.abs(large_scale_signal))
        )

        self.assertGreater(
            relative_added_large, relative_added_small * 0.5,
            "iterative_soft_thresholding's harmonic contribution collapses "
            "relative to the signal's own peak as absolute scale grows "
            f"(relative added: {relative_added_small:.3g} at peak=1 vs "
            f"{relative_added_large:.3g} at peak=10000); the harmonic "
            "amplitude is likely a fixed absolute constant rather than "
            "one scaled to the input's own amplitude, making it "
            "unmeasurable at real (raw PCM-scale) audio amplitudes."
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
        # max_iterations=2 keep this fast (lms_filter's per-sample Python
        # loop is slow -- see feed.py's own notes). Uses target_format=
        # 'wav' rather than 'flac': this source's deterministic LAME CBR
        # encode (64 kbps) combined with a high target_bitrate_kbps drives
        # a large enough upscale_factor that the resulting sample rate
        # exceeds FLAC's ~655350 Hz format ceiling (a pre-existing,
        # unrelated libsndfile/FLAC limitation) -- WAV has no such low
        # ceiling and is an equally valid target_format for this check.
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
    def test_target_bitrate_kbps_drives_upscale_factor_not_output_bitrate(
        self
    ):
        # Regression test documenting the intended contract of
        # target_bitrate_kbps: it is used only to derive upscale_factor
        # relative to the source file's own bitrate
        # (upscale_factor = round(target_bitrate_kbps * 1000 / source
        # bitrate)); its 800-1411 (flac) valid range is a sanity bound on
        # this parameter itself, not a promise about the produced file's
        # real bitrate. The output is always written as uncompressed PCM
        # (write_audio) at an upsampled sample rate, so its actual bitrate
        # is, by design, substantially higher than target_bitrate_kbps
        # once upscale_factor > 1 -- audio-quality-checker measured this
        # divergence (target 1400 kbps vs. ~7822 kbps effective output) and
        # flagged it; this test confirms the divergence is a documented,
        # structural consequence of writing uncompressed high-resolution
        # PCM, not an upscale_factor computation bug.
        _, _, source_bitrate, _ = read_audio(
            self.test_mp3_file, audio_format='mp3'
        )
        target_bitrate_kbps = 800  # minimum of the valid flac range
        expected_upscale_factor = round(
            target_bitrate_kbps * 1000 / source_bitrate
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
            # The output sample rate must reflect the documented
            # upscale_factor formula (source_sample_rate * upscale_factor).
            self.assertEqual(
                info.samplerate, 44100 * expected_upscale_factor,
                "Output sample rate does not match the documented "
                "upscale_factor = round(target_bitrate_kbps * 1000 / "
                "source bitrate) formula."
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
            # By design (uncompressed PCM at an upsampled rate), the real
            # output bitrate must be well above target_bitrate_kbps even
            # at the minimum of the valid range -- this is the documented,
            # intentional divergence, not a defect to eliminate.
            self.assertGreater(
                real_bitrate_kbps, target_bitrate_kbps * 2,
                "Expected the real output bitrate to substantially exceed "
                "target_bitrate_kbps (uncompressed PCM at an upsampled "
                "rate); if this no longer holds, target_bitrate_kbps's "
                "documented contract (a upscale_factor-derivation knob, "
                "not an output bitrate promise) may have changed."
            )
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)


if __name__ == '__main__':
    unittest.main()
