import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
import cvxpy as cp

np.random.seed(42)
plt.rcParams.update(
    {
        "figure.figsize": (10, 5),
        "axes.grid": True,
        "grid.alpha": 0.5,
    }
)

###############
# Definitions #
###############
n = 5000
m = 250
duration = 0.125
t = np.linspace(0, duration, n, endpoint=False)
fs = n / duration

f1, f2 = 1200, 2400
orig_signal = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.2 * np.sin(2 * np.pi * f2 * t)


def normalize(x):
    return x / np.max(np.abs(x))


def dct_freq_axis(num_samples, fs):
    return np.arange(num_samples) * fs / (2 * num_samples)


time_zoom = 0.01

########################
# Uniform downsampling #
########################
step = max(1, int(np.ceil(n / m)))
ds_signal = orig_signal[::step]
ds_t = t[::step]
m_ds = len(ds_signal)
fs_ds = fs / step

orig_dct = dct(orig_signal, norm="ortho")
ds_dct = dct(ds_signal, norm="ortho")

orig_dct_norm = normalize(orig_dct)
ds_dct_norm = normalize(ds_dct)

orig_freq = dct_freq_axis(n, fs)
ds_freq = dct_freq_axis(m_ds, fs_ds)

fig, axs = plt.subplots(1, 2)
fig.suptitle(f"Uniform downsampling (n={n}, m={m})")

axs[0].plot(t, orig_signal, alpha=0.75, label="Original")
axs[0].plot(ds_t, ds_signal, "o--", label="Uniform samples")
axs[0].set_title("Time domain")
axs[0].set_xlabel("Time (s)")
axs[0].set_xlim(0, min(time_zoom, duration))
axs[0].legend()

axs[1].plot(orig_freq, orig_dct_norm, alpha=0.75, label="Original")
axs[1].plot(ds_freq, ds_dct_norm, "--", label="Uniform samples")
axs[1].set_title("DCT domain")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].legend()

####################################
# Random sampling + L2/L1 recovery #
####################################
cs_idx = np.sort(np.random.choice(n, size=m, replace=False))
cs_b = orig_signal[cs_idx]

cs_A = dct(np.eye(n)[:, cs_idx], axis=0, norm="ortho").T

# L2 recovery
cs_x_l2 = np.linalg.pinv(cs_A) @ cs_b
cs_u_l2 = idct(cs_x_l2, norm="ortho")
cs_x_l2_norm = normalize(cs_x_l2)

fig, axs = plt.subplots(1, 2)
fig.suptitle(f"Random sampling + L2 recovery (n={n}, m={m})")

axs[0].plot(t, orig_signal, alpha=0.75, label="Original")
axs[0].scatter(t[cs_idx], cs_b, c="red", label="Random samples")
axs[0].plot(t, cs_u_l2, "--", label="L2 reconstruction")
axs[0].set_title("Time domain")
axs[0].set_xlabel("Time (s)")
axs[0].set_xlim(0, min(time_zoom, duration))
axs[0].legend()

axs[1].plot(orig_freq, orig_dct_norm, alpha=0.75, label="Original")
axs[1].plot(orig_freq, cs_x_l2_norm, "--", label="Recovered (L2)")
axs[1].set_title("DCT domain")
axs[1].set_ylabel("Normalized |DCT|")
axs[1].legend()

# L1 recovery
cs_x_var = cp.Variable(n)
cs_prob = cp.Problem(cp.Minimize(cp.norm1(cs_x_var)), [cs_A @ cs_x_var == cs_b])
cs_prob.solve()

cs_x_l1 = cs_x_var.value
cs_u_l1 = idct(cs_x_l1, norm="ortho")
cs_x_l1_norm = normalize(cs_x_l1)

fig, axs = plt.subplots(1, 2)
fig.suptitle(f"Random sampling + L1 recovery (n={n}, m={m})")

axs[0].plot(t, orig_signal, alpha=0.75, label="Original")
axs[0].scatter(t[cs_idx], cs_b, c="red", label="Random samples")
axs[0].plot(t, cs_u_l1, "--", label="L1 reconstruction")
axs[0].set_title("Time domain")
axs[0].set_xlabel("Time (s)")
axs[0].set_xlim(0, min(time_zoom, duration))
axs[0].legend()

axs[1].plot(orig_freq, orig_dct_norm, alpha=0.75, label="Original")
axs[1].plot(orig_freq, cs_x_l1_norm, "--", label="Recovered (L1)")
axs[1].set_title("DCT domain")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].legend()

plt.show()
