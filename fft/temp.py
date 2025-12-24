import numpy as np
import matplotlib.pyplot as plt
import time

signal_sizes = range(100, 10_000, 10)

def direct_convolution(signal, kernel):
    size = len(signal)
    rolled_matrix = np.column_stack([np.roll(signal, -i) for i in range(size)])
    return rolled_matrix @ kernel

def fft_convolution(signal, kernel):
    return np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(kernel)))

direct_times = []
fft_times = []

for size in signal_sizes:
    signal = np.random.randn(size)
    kernel = np.ones(size)

    start_time = time.perf_counter()
    direct_result = direct_convolution(signal, kernel)
    direct_times.append(time.perf_counter() - start_time)

    start_time = time.perf_counter()
    fft_result = fft_convolution(signal, kernel)
    fft_times.append(time.perf_counter() - start_time)
    assert np.allclose(direct_result, fft_result, atol=1e-12), f"Mismatch at size={size}"


fig, ax = plt.subplots()
ax.loglog(signal_sizes, direct_times, "o-", label="Direct convolution")
ax.loglog(signal_sizes, fft_times, "o-", label="FFT convolution")
ax.set_xlabel("Signal size")
ax.set_ylabel("Runtime (seconds)")
ax.set_title("Convolution runtime scaling")
ax.legend()
ax.grid(True, which="both")
fig.tight_layout()
plt.show()
