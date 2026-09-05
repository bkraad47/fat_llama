import unittest
import numpy as np
import cupy as cp
import os
import soundfile as sf
from unittest.mock import patch, MagicMock
from fat_llama.audio_fattener.feed import (
    read_audio, write_audio, new_interpolation_algorithm, lms_filter
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
        sine_wave = Sine(440).to_audio_segment(duration=1000)  # 1 second of 440 Hz sine wave
        sine_wave.export(filename, format="mp3")

    def test_read_audio(self):
        sample_rate, samples, bitrate, audio = read_audio(
            self.test_mp3_file, audio_format='mp3'
        )
        self.assertEqual(sample_rate, 44100)  # Default sample rate for the generated sine wave
        self.assertEqual(len(samples), 44100)  # 1 second of audio at 44100 Hz
        # A mono source must come back as a flat 1-D array, not an (N, 2) reshape.
        self.assertEqual(audio.channels, 1)
        self.assertEqual(samples.ndim, 1)
        self.assertEqual(len(audio), 1000)  # duration in ms
        # The exact bitrate depends on the ffmpeg/LAME build, so assert the
        # encoder's default CBR band rather than one hard-coded magic number.
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
            # Sample rate and channel count must be preserved by the round trip.
            self.assertEqual(info.samplerate, sample_rate)
            self.assertEqual(info.channels, audio.channels)
            # Duration must match the ~1 second input within a small tolerance.
            self.assertAlmostEqual(info.duration, len(audio) / 1000.0, delta=0.05)

            written_data, written_sr = sf.read(output_file)
            self.assertEqual(written_sr, sample_rate)
            # The written audio must not be silence.
            self.assertGreater(np.max(np.abs(written_data)), 0.0)
            # The written audio must not be catastrophically clipped: write_audio()
            # is documented to accept the samples produced by read_audio(), whose
            # magnitude is on the raw PCM scale (tens of thousands), not normalized
            # to [-1, 1]. If write_audio() fails to normalize/scale before handing
            # data to soundfile with an integer subtype, nearly every sample gets
            # clamped to full scale, destroying the waveform.
            clipped_fraction = np.mean(np.abs(written_data) > 0.999)
            self.assertLess(
                clipped_fraction, 0.05,
                f"{clipped_fraction:.2%} of written samples are clipped to full scale; "
                "write_audio() likely wrote un-normalized/out-of-range data directly "
                "with an integer subtype instead of scaling it to [-1, 1] first."
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
        signal = 0.5 * cp.sin(2 * cp.pi * 300 * t) + 0.2 * cp.sin(2 * cp.pi * 900 * t)

        filtered = lms_filter(signal, signal, mu=0.001, num_taps=num_taps)

        signal_np = cp.asnumpy(signal)
        filtered_np = cp.asnumpy(filtered)

        early_filtered = filtered_np[num_taps:num_taps + 50]
        early_signal = signal_np[num_taps:num_taps + 50]
        early_filtered_rms = np.sqrt(np.mean(early_filtered ** 2))
        early_signal_rms = np.sqrt(np.mean(early_signal ** 2))

        self.assertGreater(
            early_filtered_rms / early_signal_rms, 0.5,
            f"Filtered output RMS immediately after warm-up ({early_filtered_rms:.4f}) "
            f"is far below the input's own RMS in that same window ({early_signal_rms:.4f}); "
            "lms_filter() is likely ramping up from a zero-initialized state instead "
            "of tracking the signal from (near) the first sample."
        )

if __name__ == '__main__':
    unittest.main()
