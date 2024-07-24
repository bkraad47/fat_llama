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

def compare_signals(signal2, signal1):
    min_len = min(len(signal2), len(signal1))
    signal2 = signal2[:min_len]
    signal1 = signal1[:min_len]
    time = np.arange(min_len)

    print(f"Signal 2 (first 10 samples): {signal2[:10]}")
    print(f"Signal 1 (first 10 samples): {signal1[:10]}")
    
    # 1. Waveform Comparison
    plt.figure(figsize=(14, 7))
    plt.plot(time, signal2, label='Signal 2')
    plt.plot(time, signal1, label='Signal 1')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Waveform Comparison')
    plt.show()

    # 2. Difference Signal
    difference_signal = signal2 - signal1
    plt.figure(figsize=(14, 7))
    plt.plot(time, difference_signal, label='Difference Signal')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Difference Signal')
    plt.show()

    # 3. Mean Squared Error (MSE)
    mse = np.mean((signal2 - signal1) ** 2)
    print(f"Mean Squared Error: {mse}")

    # 4. Spectrogram Comparison
    f1, t1, Sxx1 = signal.spectrogram(signal2)
    f2, t2, Sxx2 = signal.spectrogram(signal1)

    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t1, f1, 10 * np.log10(Sxx1))
    plt.title('Spectrogram of Signal 2')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.subplot(2, 1, 2)
    plt.pcolormesh(t2, f2, 10 * np.log10(Sxx2))
    plt.title('Spectrogram of Signal 1')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()

    # 5. Cross-Correlation using GPU
    signal2_gpu = cp.asarray(signal2)
    signal1_gpu = cp.asarray(signal1)

    correlation_gpu = cp.correlate(signal2_gpu, signal1_gpu, mode='full')
    correlation = cp.asnumpy(correlation_gpu)
    lags = np.arange(-len(signal2) + 1, len(signal1))

    plt.figure(figsize=(14, 7))
    plt.plot(lags, correlation)
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation')
    plt.title('Cross-correlation between Signal 2 and Signal 1')
    plt.show()

    # 6. Frequency Domain Analysis using cuFFT
    freq2 = cp.fft.fft(signal2_gpu)
    freq1 = cp.fft.fft(signal1_gpu)

    freq2 = cp.asnumpy(freq2)
    freq1 = cp.asnumpy(freq1)

    plt.figure(figsize=(14, 7))
    plt.plot(np.abs(freq2), label='Frequency Content of Signal 2')
    plt.plot(np.abs(freq1), label='Frequency Content of Signal 1')
    plt.legend()
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain Comparison')
    plt.show()

# Example usage:
if __name__ == "__main__":
    mp3_file_path = 'input_test.mp3'
    flac_file_path = 'output_test.flac'

    signal1, sample_rate1 = read_mp3(mp3_file_path)
    signal2, sample_rate2 = read_flac(flac_file_path)

    # Resample if necessary to match sample rates
    if sample_rate1 != sample_rate2:
        from scipy.signal import resample

        if len(signal1) > len(signal2):
            signal2 = resample(signal2, len(signal1))
        else:
            signal1 = resample(signal1, len(signal2))

    compare_signals(signal2, signal1)
