import unittest
import numpy as np
import os
from fat_llama.audio_fattener.feed import read_mp3, upscale_mp3_to_flac, write_flac

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
        if os.path.exists('output_upscaled_no_processing.flac'):
            os.remove('output_upscaled_no_processing.flac')

    def create_test_mp3(self, filename):
        from pydub.generators import Sine
        sine_wave = Sine(440).to_audio_segment(duration=1000)  # 1 second of 440 Hz sine wave
        sine_wave.export(filename, format="mp3")

    def test_read_mp3(self):
        sample_rate, samples, bitrate, audio = read_mp3(self.test_mp3_file)
        self.assertEqual(sample_rate, 44100)  # Default sample rate for the generated sine wave
        self.assertEqual(len(samples), 44100)  # 1 second of audio at 44100 Hz
        self.assertEqual(bitrate, 63999)  # Bitrate of the generated MP3

    def test_upscale_mp3_to_flac(self):
        upscale_mp3_to_flac(
            input_file_path=self.test_mp3_file,
            output_file_path_processed='output_processed.flac',
            max_iterations=10,  # Using a smaller number for faster testing
            threshold_value=0.1,
            gain_factor=1.0,
            reduction_profile=[
                (5, 140, -0.5),
                (1000, 10000, 0.5),
            ],
            lowcut=5.0,
            highcut=22000.0,
            target_bitrate_kbps=800,
            output_file_path_no_processing='output_upscaled_no_processing.flac',
            use_wiener_filter=False
        )
        self.assertTrue(os.path.exists('output_processed.flac'))
        self.assertTrue(os.path.exists('output_upscaled_no_processing.flac'))

    def test_write_flac(self):
        sample_rate, samples, bitrate, audio = read_mp3(self.test_mp3_file)
        output_file = 'test_output.flac'
        write_flac(output_file, sample_rate, samples)
        self.assertTrue(os.path.exists(output_file))
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == '__main__':
    unittest.main()
