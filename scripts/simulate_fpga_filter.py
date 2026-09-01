#!/usr/bin/env python3
"""Demonstration script: slicing generated data (3154, 4096) in chunks of width 14
and computing compute_median_ratio_spectrum to observe transient convergence.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import lfilter, freqz
from radarsig.fpga_io import load_filter_coeffs_from_binary
from radarsig.fpga_analysis import compute_median_ratio_spectrum, compute_csd_spectrum


def main():
    parser = argparse.ArgumentParser(description="Simulate FPGA IIR filter transient convergence.")
    parser.add_argument(
        "--pair",
        type=str,
        default="000",
        help="IQ pair ID to load coefficients from (default: 000)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save output plots. If not specified, outputs are not saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively.",
    )
    args = parser.parse_args()

    # 1. Load filter coefficients from real binary data if available, else fallback
    coeff_path = Path(f"data/raw/fpga/iq/{args.pair}_i.data")
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

    fig, ax = plt.subplots(figsize=(8, 5))

    # Get theoretical reference curve
    fft_len = 256
    half_len = fft_len // 2 + 1

    w, h = freqz(b, a, worN=half_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_db = 20 * np.log10(np.abs(h))

    freqs = np.linspace(0, np.pi, half_len)

    ax.plot(freqs, h_db, label="Theoretical H(dB) [freqz]", color="black", linewidth=2.5, zorder=10)

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

        ax.plot(
            freqs,
            median_ratio_db,
            label=f"Chunk {c_idx} (samples {start}–{end})",
            color=colors[idx],
            alpha=0.8,
        )

    ax.set_title(f"Median Ratio Spectrum Convergence (Pair {args.pair}, Width=14)")
    ax.set_xlabel("Frequency (rad/sample)")
    ax.set_ylabel("Magnitude Ratio (dB)")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = args.out_dir / f"simulate_response_{args.pair}.png"
        fig.savefig(plot_path)
        print(f"Saved plot to {plot_path}")

    if args.show or args.out_dir is None:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
