#!/usr/bin/env python3
"""Demonstration script: slicing generated data (3154, 4096) in chunks of width 14
and computing compute_median_ratio_spectrum to observe transient convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import lfilter, freqz
from radarsig.fpga_io import load_filter_coeffs_from_binary
from radarsig.fpga_analysis import compute_median_ratio_spectrum, compute_csd_spectrum


def main():
    # 1. Load filter coefficients from real binary data if available, else fallback
    coeff_path = Path("data/raw/fpga/iq/000_i.data")
    if coeff_path.exists():
        b, a = load_filter_coeffs_from_binary(coeff_path)
        print(f"Loaded coefficients from {coeff_path}")
    else:
        b = np.array([0.90186173, -2.703357699, 2.703357699, -0.90186173])
        a = np.array([1.0, -2.794421010, 2.605942211, -0.810075596])
        print("Using default coefficients")

    # 2. Generate synthetic data x of shape (3154, 4096)
    n_bins = 3154
    n_samples = 4096
    np.random.seed(42)
    x = np.random.randn(n_bins, n_samples) + 1j * np.random.randn(n_bins, n_samples)

    # Filter along axis=1 (simulating FPGA IIR filtering)
    y = lfilter(b, a, x, axis=1)

    # 3. Slicing along axis=1 with a width of 14 (like n_pulse=14)
    chunk_width = 14
    chunk_indices = [0, 1, 2, 5, 20, 100, 250]

    plt.figure(figsize=(12, 7))

    # Get theoretical reference curve
    fft_len = 256
    half_len = fft_len // 2 + 1

    w, h = freqz(b, a, worN=half_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_db = 20 * np.log10(np.abs(h))

    freqs = np.linspace(0, np.pi, half_len)

    plt.plot(
        freqs, h_db, label="Theoretical H(dB) [freqz]", color="black", linewidth=2.5, zorder=10
    )

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(chunk_indices)))

    for idx, c_idx in enumerate(chunk_indices):
        start = c_idx * chunk_width
        end = start + chunk_width
        if end > n_samples:
            break

        x_chunk = x[:, start:end]
        y_chunk = y[:, start:end]

        freqs, _, median_ratio_db = compute_median_ratio_spectrum(
            x_chunk, y_chunk, b, a, n_bins=n_bins, fft_len=256
        )

        plt.plot(
            freqs,
            median_ratio_db,
            label=f"Chunk {c_idx} (samples {start}–{end})",
            color=colors[idx],
            alpha=0.8,
        )

    plt.title("Median Ratio Spectrum Convergence Over Successive Chunks (Width=14)")
    plt.xlabel("Frequency (rad/sample)")
    plt.ylabel("Magnitude Ratio (dB)")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
