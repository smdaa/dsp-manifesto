import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform


def hio_reconstruction(measured_magnitude, iterations=500, beta=0.9):
    rows, cols = measured_magnitude.shape
    # Initialize with random phase
    current_phase = np.exp(1j * 2 * np.pi * np.random.rand(rows, cols))
    G = measured_magnitude * current_phase

    g = np.real(np.fft.ifft2(G))
    g_prev = np.copy(g)

    for i in range(iterations):
        # 1. Fourier Projection
        G_prime = np.fft.fft2(g)
        G_prime = measured_magnitude * np.exp(1j * np.angle(G_prime))

        # 2. Inverse to Spatial Domain
        g_prime = np.real(np.fft.ifft2(G_prime))

        # 3. Constraint Enforcement (Object must be non-negative)
        feasible = g_prime >= 0
        g_next = np.zeros_like(g_prime)
        g_next[feasible] = g_prime[feasible]
        # Feedback mechanism to escape local minima
        g_next[~feasible] = g_prev[~feasible] - beta * g_prime[~feasible]

        g_prev = np.copy(g)
        g = g_next

    return g


# Prepare Data
image = data.camera().astype(float) / 255.0
image = transform.resize(image, (256, 256))

# Observed Magnitude Only
obs_mag = np.abs(np.fft.fft2(image))

# Reconstruct
result = hio_reconstruction(obs_mag)

plt.imshow(result, cmap="gray")
plt.title("Reconstructed from Magnitude Only")
plt.show()
