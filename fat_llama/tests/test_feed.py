import unittest
import numpy as np
import os
from unittest.mock import patch, MagicMock
from fat_llama.audio_fattener.feed import read_audio, write_audio, new_interpolation_algorithm, initialize_ist, iterative_soft_thresholding, upscale_channels, normalize_signal

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
        self.assertTrue(os.path.exists(output_file))
        if os.path.exists(output_file):
            os.remove(output_file)

    def test_new_interpolation_algorithm(self):
        data = np.array([1, 2, 3, 4])
        upscale_factor = 2
        expected_result = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        result = new_interpolation_algorithm(data, upscale_factor)
        np.testing.assert_array_equal(result, expected_result)

    @patch('fat_llama.audio_fattener.feed.cp.where')
    @patch('fat_llama.audio_fattener.feed.cp.abs')
    @patch('fat_llama.audio_fattener.feed.cp.array')
    def test_initialize_ist(self, mock_cp_array, mock_cp_abs, mock_cp_where):
        mock_cp_array.side_effect = lambda x, dtype=None: np.array(x, dtype=dtype)
        mock_cp_abs.side_effect = lambda x: np.abs(x)
        mock_cp_where.side_effect = lambda condition, x, y: np.where(condition, x, y)

        data = np.array([0.1, 0.5, 0.8])
        threshold = 0.4
        expected_result = np.array([0.0, 0.5, 0.8])
        result = initialize_ist(data, threshold)
        np.testing.assert_array_equal(result, expected_result)

    @patch('fat_llama.audio_fattener.feed.cp.fft.fft')
    @patch('fat_llama.audio_fattener.feed.cp.fft.ifft')
    @patch('fat_llama.audio_fattener.feed.cp.where')
    @patch('fat_llama.audio_fattener.feed.cp.abs')
    @patch('fat_llama.audio_fattener.feed.cp.array')
    def test_iterative_soft_thresholding(self, mock_cp_array, mock_cp_abs, mock_cp_where, mock_cp_fft, mock_cp_ifft):
        mock_cp_array.side_effect = lambda x, dtype=None: np.array(x, dtype=dtype)
        mock_cp_abs.side_effect = lambda x: np.abs(x)
        mock_cp_where.side_effect = lambda condition, x, y: np.where(condition, x, y)
        mock_cp_fft.side_effect = lambda x: np.fft.fft(x)
        mock_cp_ifft.side_effect = lambda x: np.fft.ifft(x)

        data = np.array([1.0, 0.5, 0.2, 0.1])
        max_iter = 2
        threshold = 0.1
        result = iterative_soft_thresholding(data, max_iter, threshold)
        expected_result = np.array([1.0, 0.5, 0.2, 0])
        np.testing.assert_array_almost_equal(result, expected_result)

    @patch('fat_llama.audio_fattener.feed.cp.fft.fft')
    @patch('fat_llama.audio_fattener.feed.cp.fft.ifft')
    @patch('fat_llama.audio_fattener.feed.cp.where')
    @patch('fat_llama.audio_fattener.feed.cp.abs')
    @patch('fat_llama.audio_fattener.feed.cp.array')
    def test_upscale_channels(self, mock_cp_array, mock_cp_abs, mock_cp_where, mock_cp_fft, mock_cp_ifft):
        mock_cp_array.side_effect = lambda x, dtype=None: np.array(x, dtype=dtype)
        mock_cp_abs.side_effect = lambda x: np.abs(x)
        mock_cp_where.side_effect = lambda condition, x, y: np.where(condition, x, y)
        mock_cp_fft.side_effect = lambda x: np.fft.fft(x)
        mock_cp_ifft.side_effect = lambda x: np.fft.ifft(x)

        channels = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        upscale_factor = 2
        max_iter = 2
        threshold = 0.1
        result = upscale_channels(channels, upscale_factor, max_iter, threshold)
        expected_result = np.array([[ 2.,  4.],[ 2.,  4.],[ 6.,  8.],
                    [ 6.,  8.],
                    [10., 12.],
                    [10. ,12.],
                    [14., 16.],
                    [14., 16.]])
        np.testing.assert_array_almost_equal(result, expected_result)

if __name__ == '__main__':
    unittest.main()
