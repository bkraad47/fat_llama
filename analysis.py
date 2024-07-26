import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from pydub import AudioSegment
import soundfile as sf
import cupy as cp  # CuPy for GPU acceleration

def read_mp3(file_path):
    audio = AudioSegment.from_mp3(file_path)
    data = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        data = data.reshape((-1, 2))
        data = data.mean(axis=1)  # Convert to mono
    return data, audio.frame_rate

def read_flac(file_path):
    data, sample_rate = sf.read(file_path)
    if len(data.shape) == 2:
        data = data.mean(axis=1)  # Convert to mono if stereo
    return data, sample_rate

def normalize(signal):
    return signal / np.max(np.abs(signal))

def compare_signals(mp3, flac, sample_rate):
    # Normalize signals
    mp3 = normalize(mp3)
    flac = normalize(flac)
    
    min_len = min(len(mp3), len(flac))
    mp3 = mp3[:min_len]
    flac = flac[:min_len]
    time = np.arange(min_len) / sample_rate

    print(f"MP3 (first 10 samples): {mp3[:10]}")
    print(f"FLAC (first 10 samples): {flac[:10]}")
    
    # 1. Waveform Comparison
    plt.figure(figsize=(14, 7))
    plt.plot(time, mp3, label='MP3')
    plt.plot(time, flac, label='FLAC')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform Comparison')
    plt.show()

    # 2. Difference Signal
    difference_signal = mp3 - flac
    plt.figure(figsize=(14, 7))
    plt.plot(time, difference_signal, label='Difference Signal')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Difference Signal')
    plt.show()

    # 3. Mean Squared Error (MSE)
    mse = np.mean((mp3 - flac) ** 2)
    print(f"Mean Squared Error: {mse}")

    # 4. Spectrogram Comparison
    nperseg = 2048
    f1, t1, Sxx1 = signal.spectrogram(mp3, fs=sample_rate, nperseg=nperseg)
    f2, t2, Sxx2 = signal.spectrogram(flac, fs=sample_rate, nperseg=nperseg)

    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t1, f1, 10 * np.log10(Sxx1), vmin=-120, vmax=0)
    plt.title('Spectrogram of MP3')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='SNR (dB)')
    plt.yscale('log')
    plt.ylim(20, 25000)

    plt.subplot(2, 1, 2)
    plt.pcolormesh(t2, f2, 10 * np.log10(Sxx2), vmin=-120, vmax=0)
    plt.title('Spectrogram of FLAC')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='SNR (dB)')
    plt.yscale('log')
    plt.ylim(20, 25000)
    plt.tight_layout()
    plt.show()

    # 5. Cross-Correlation using GPU
    mp3_gpu = cp.asarray(mp3)
    flac_gpu = cp.asarray(flac)

    correlation_gpu = cp.correlate(mp3_gpu, flac_gpu, mode='full')
    correlation = cp.asnumpy(correlation_gpu)
    lags = np.arange(-len(mp3) + 1, len(flac))

    plt.figure(figsize=(14, 7))
    plt.plot(lags, correlation)
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation')
    plt.title('Cross-correlation between MP3 and FLAC')
    plt.show()

    # 6. Frequency Domain Analysis using cuFFT
    freq_mp3 = cp.fft.fft(mp3_gpu)
    freq_flac = cp.fft.fft(flac_gpu)

    freq_mp3 = cp.asnumpy(freq_mp3)
    freq_flac = cp.asnumpy(freq_flac)

    plt.figure(figsize=(14, 7))
    plt.plot(np.abs(freq_mp3), label='Frequency Content of MP3')
    plt.plot(np.abs(freq_flac), label='Frequency Content of FLAC')
    plt.legend()
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain Comparison')
    plt.show()

# Example usage:
if __name__ == "__main__":
    mp3_file_path = 'input_test.mp3'
    flac_file_path = 'output_test.flac'

    mp3, sample_rate_mp3 = read_mp3(mp3_file_path)
    flac, sample_rate_flac = read_flac(flac_file_path)

    # Resample if necessary to match sample rates
    if sample_rate_mp3 != sample_rate_flac:
        from scipy.signal import resample

        if len(mp3) > len(flac):
            flac = resample(flac, len(mp3))
        else:
            mp3 = resample(mp3, len(flac))

    sample_rate = sample_rate_mp3  # Assuming both are the same after resampling
    compare_signals(mp3, flac, sample_rate)
