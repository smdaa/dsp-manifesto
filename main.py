import numpy as np
import matplotlib.pyplot as plt


def triangular_wave(t, freq, amp, offset):
    y = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    y = amp * y + offset

    return y

def square_wave(t, freq, amp, offset):
    y = np.sign(np.sin(2 * np.pi * freq * t))
    y = amp * y + offset
    return y


def compute_spectrum(signal, sampling_rate):
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_shifted = np.fft.fftshift(fft_vals)
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    freqs = np.fft.fftshift(freqs)
    spectrum = np.abs(fft_shifted) / n

    return freqs, spectrum


fs = 500  # Sampling rate (Hz)
duration = 1.0  # Signal duration (seconds)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generate triangular wave
freq = 10
amp = 1.0
offset = 0.0
triangular_signal = triangular_wave(t, freq, amp, offset)

# Compute spectrum
freqs, spectrum = compute_spectrum(triangular_signal, fs)

plt.figure()
plt.plot(triangular_signal)
plt.grid()

plt.figure()
plt.stem(freqs, spectrum)
plt.grid()


# Generate square wave
freq = 10
amp = 1.0
offset = 0.0
square_signal = square_wave(t, freq, amp, offset)

# Compute spectrum
freqs, spectrum = compute_spectrum(square_signal, fs)

plt.figure()
plt.plot(square_signal)
plt.grid()

plt.figure()
plt.stem(freqs, spectrum)
plt.grid()

plt.show()
