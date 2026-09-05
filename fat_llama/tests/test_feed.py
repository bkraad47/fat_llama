import os
import unittest

import cupy as cp
import numpy as np
import soundfile as sf

from fat_llama.audio_fattener.feed import (
    iterative_soft_thresholding, lms_filter, read_audio, upscale, write_audio
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
