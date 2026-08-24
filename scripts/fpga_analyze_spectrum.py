#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from radarsig.fpga_io import find_iq_pairs, load_fpga_ram_binary_to_iq, load_filter_coeffs_from_binary
from radarsig.fpga_analysis import compute_median_ratio_spectrum, compute_csd_spectrum, compute_coherence

# Method dispatcher definition
ANALYSIS_METHODS = {
    "median": {
        "func": compute_median_ratio_spectrum,
        "title_prefix": "IIR Filter Frequency Response (Median)",
        "filename_prefix": "filter_spectrum_median",
    },
    "csd": {
        "func": compute_csd_spectrum,
        "title_prefix": "IIR Filter Frequency Response (CSD)",
        "filename_prefix": "filter_spectrum_csd",
    },
    "coherence": {
        "func": compute_coherence,
        "title_prefix": "Magnitude-Squared Coherence & Theoretical Response",
        "filename_prefix": "filter_spectrum_coherence",
    },
}

def plot_result(freqs, result_data, method: str, pair_id: str, out_dir: Path):
    config = ANALYSIS_METHODS[method]
    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()

    if method in ("median", "csd"):
        h_db, emp_db = result_data
        ax1.plot(freqs, h_db, label="Theoretical ($|H|$) - freqz", color="black", linewidth=2, linestyle="--")
        label = "Empirical Median ($|Y|/|X|$)" if method == "median" else "Empirical CSD ($S_{yx}/S_{xx}$)"
        color = "blue" if method == "median" else "red"
        ax1.plot(freqs, emp_db, label=label, color=color, alpha=0.8)
        ax1.set_ylabel("Magnitude [dB]")
        ax1.legend(loc="upper right")
    elif method == "coherence":
        h_db, coherence = result_data
        line1 = ax1.plot(freqs, h_db, label="Theoretical ($|H|$) - freqz", color="black", linewidth=2, linestyle="--")
        ax1.set_ylabel("Magnitude [dB]", color="black")
        ax1.tick_params(axis='y', labelcolor="black")

        ax2 = ax1.twinx()
        line2 = ax2.plot(freqs, coherence, label="Coherence ($\gamma_{xy}^2$)", color="green", linewidth=2)
        ax2.set_ylabel("Coherence", color="green")
        ax2.tick_params(axis='y', labelcolor="green")
        ax2.set_ylim(-0.05, 1.05)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")

    ax1.set_title(f"{config['title_prefix']} - Pair {pair_id}")
    ax1.set_xlabel("Digital Frequency ($\omega$) [rad/sample]")
    ax1.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.tight_layout()

    plot_path = out_dir / f"{config['filename_prefix']}_{pair_id}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved plot to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch analyze IIR filter spectrum.")
    parser.add_argument("--dir", default="tmp/iq", help="Directory with IQ data")
    parser.add_argument("--method", choices=list(ANALYSIS_METHODS.keys()), default="median", help="Analysis method")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    try:
        pairs = find_iq_pairs(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not pairs:
        print(f"No IQ pairs found in {data_dir}")
        return

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    method_meta = ANALYSIS_METHODS[args.method]
    print(f"Found {len(pairs)} IQ pair(s). Processing method: {args.method}...")

    for pair_id, i_path, q_path in pairs:
        print(f"Processing pair: {pair_id}...")
        try:
            x, y = load_fpga_ram_binary_to_iq(i_path, q_path, offset_dtype=512, n_pulse=14)
            b, a = load_filter_coeffs_from_binary(i_path)
        except Exception as e:
            print(f"  Error loading {pair_id}: {e}")
            continue

        freqs, data1, data2 = method_meta["func"](x, y, b, a)
        plot_result(freqs, (data1, data2), args.method, pair_id, out_dir)

    print("Batch analysis complete.")

if __name__ == "__main__":
    main()
