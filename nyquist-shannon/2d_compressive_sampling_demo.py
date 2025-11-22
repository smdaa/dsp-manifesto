import numpy as np
from scipy.fft import dctn, idctn
import matplotlib.pyplot as plt
from PIL import Image
import pylops
from pylops.optimization.sparsity import fista

# Load image
img = Image.open("lenna.png").convert("L")
img = np.asarray(img, dtype=float) / 255.0
height, width = img.shape
num_pixels = height * width

# Full 2D DCT
dct_coeffs = dctn(img, norm="ortho")
dct_mag = np.log1p(np.abs(dct_coeffs))
vmax = np.percentile(dct_mag, 99)

# Plot original + full DCT
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Original image and 2D DCT")
axs[0].imshow(img, cmap="gray")
axs[0].set_title("Original image")
axs[1].imshow(dct_mag, cmap="gray", vmin=0, vmax=vmax)
axs[1].set_title("2D DCT magnitude (log)")

# Sparsify - keep top % coefficients
keep_pct = 2
num_keep = int(keep_pct / 100 * num_pixels)
flat = np.abs(dct_coeffs).ravel()
thresh = np.partition(flat, -num_keep)[-num_keep]
mask = np.abs(dct_coeffs) >= thresh
dct_sparse = dct_coeffs * mask
img_reconstructed = idctn(dct_sparse, norm="ortho")

dct_sparse_mag = np.log1p(np.abs(dct_sparse))
vmax_sparse = np.percentile(dct_sparse_mag, 99)

# Plot reconstructed + sparse DCT
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(f"Reconstruction from top {keep_pct}% DCT coefficients")
axs[0].imshow(img_reconstructed, cmap="gray")
axs[0].set_title("Reconstructed image")
axs[1].imshow(dct_sparse_mag, cmap="gray", vmin=0, vmax=vmax_sparse)
axs[1].set_title("Sparse 2D DCT magnitude (log)")

# Compressive sensing with FISTA
sample_pct = 20
num_samples = int(sample_pct / 100 * num_pixels)
sample_idx = np.sort(np.random.choice(num_pixels, size=num_samples, replace=False))

# Define operators
dct_op = pylops.signalprocessing.DCT(dims=(height, width), dtype="float64")
restrict_op = pylops.Restriction(num_pixels, sample_idx, dtype="float64")

# Get measurements
measurements = restrict_op @ img.ravel()

# Sensing matrix A = M * IDCT
sensing_op = restrict_op * dct_op.H

# Reconstruct with FISTA
lambda_reg = 0.02
num_iters = 500
x_reconstructed, _, _ = fista(
    sensing_op, measurements, niter=num_iters, eps=lambda_reg, show=False
)
img_cs_reconstructed = (dct_op.H @ x_reconstructed).reshape(height, width)

# Create sampled image for visualization
sampled_img = np.zeros((height, width))
sampled_img.ravel()[sample_idx] = img.ravel()[sample_idx]

# Plot CS reconstruction
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(f"L1 CS reconstruction ({sample_pct}% samples)")
axs[0].imshow(sampled_img, cmap="gray")
axs[0].set_title("Sampled pixels")
axs[1].imshow(img_cs_reconstructed, cmap="gray")
axs[1].set_title("FISTA reconstruction")

# Print reconstruction quality
mse = np.mean((img - img_cs_reconstructed) ** 2)
psnr = 10 * np.log10(1.0 / mse)
print(f"Reconstruction MSE: {mse:.6f}")
print(f"Reconstruction PSNR: {psnr:.2f} dB")

plt.show()
