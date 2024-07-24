import unittest
import numpy as np
import os
from fat_llama.audio_fattener.feed import read_audio, upscale, write_audio

class TestAudioFattener(unittest.TestCase):

    def setUp(self):
        # Create small example audio files for testing
        self.test_mp3_file = 'test_input.mp3'
        self.test_wav_file = 'test_input.wav'
        self.test_ogg_file = 'test_input.ogg'
        self.test_flac_file = 'test_input.flac'
        self.create_test_audio(self.test_mp3_file, 'mp3')
        self.create_test_audio(self.test_wav_file, 'wav')
        self.create_test_audio(self.test_ogg_file, 'ogg')
        self.create_test_audio(self.test_flac_file, 'flac')

    def tearDown(self):
        # Remove the test audio files and any generated audio files
        for file in [self.test_mp3_file, self.test_wav_file, self.test_ogg_file, self.test_flac_file, 'output_processed.flac']:
            if os.path.exists(file):
                os.remove(file)

    def create_test_audio(self, filename, format):
        from pydub.generators import Sine
        sine_wave = Sine(440).to_audio_segment(duration=1000)  # 1 second of 440 Hz sine wave
        sine_wave.export(filename, format=format)

    def test_read_audio(self):
        for file, format in [(self.test_mp3_file, 'mp3'), (self.test_wav_file, 'wav'), (self.test_ogg_file, 'ogg'), (self.test_flac_file, 'flac')]:
            sample_rate, samples, bitrate, audio = read_audio(file, format)
            self.assertEqual(sample_rate, 44100)  # Default sample rate for the generated sine wave
            if format == 'mp3':
                self.assertEqual(bitrate, 63999)  # Bitrate of the generated MP3
            else:
                self.assertIsNotNone(bitrate)  # Bitrate should be calculated for other formats

    def test_upscale(self):
        for file, format in [(self.test_mp3_file, 'mp3'), (self.test_wav_file, 'wav'), (self.test_ogg_file, 'ogg'), (self.test_flac_file, 'flac')]:
            upscale(
                input_file_path=file,
                output_file_path='output_processed.flac',
                source_format=format,
                target_format='flac',
                max_iterations=10,  # Using a smaller number for faster testing
                threshold_value=0.1,
                target_bitrate_kbps=800
            )
            self.assertTrue(os.path.exists('output_processed.flac'))
            os.remove('output_processed.flac')

    def test_write_audio(self):
        sample_rate, samples, bitrate, audio = read_audio(self.test_mp3_file, 'mp3')
        output_file = 'test_output.flac'
        write_audio(output_file, sample_rate, samples, 'flac')
        self.assertTrue(os.path.exists(output_file))
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == '__main__':
    unittest.main()
