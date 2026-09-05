import unittest
import numpy as np
import os
import soundfile as sf
from unittest.mock import patch, MagicMock
from fat_llama.audio_fattener.feed import (
    read_audio, write_audio, new_interpolation_algorithm
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
        sample_rate, samples, bitrate, audio = read_audio(self.test_mp3_file, format='mp3')
        self.assertEqual(sample_rate, 44100)  # Default sample rate for the generated sine wave
        self.assertEqual(len(samples), 44100)  # 1 second of audio at 44100 Hz
        self.assertEqual(bitrate, 63999)  # Bitrate of the generated MP3

    def test_write_audio(self):
        sample_rate, samples, bitrate, audio = read_audio(self.test_mp3_file, format='mp3')
        output_file = 'test_output.flac'
        write_audio(output_file, sample_rate, samples, format='flac')
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

if __name__ == '__main__':
    unittest.main()
