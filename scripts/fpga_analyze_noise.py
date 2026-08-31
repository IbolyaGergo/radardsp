#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from radarsig.fpga_io import (
    load_fpga_ram_binary_to_iq,
    find_iq_pair_by_id,
    load_filter_coeffs_from_binary,
)
from radarsig.filters import compute_noise_power_gain_db

# NOTE: pairs 005-008 are omitted, because no filter was applied
RANGES = {
    "000": (2400, 2900),
    "001": (2100, 3000),
    "002": (1500, 2500),
    "003": (1500, 2500),
    "004": (1700, 2500),
    "009": (1700, 2500),
    "010": (1700, 2500),
    "011": (1700, 2500),
    "012": (1700, 2500),
}


def plot_pair_noise_stats(pair_id, emp_gains_db, theoretical_gain_db):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(emp_gains_db, "-o", label="Measured Median Gain")
    ax.axhline(
        theoretical_gain_db,
        color="red",
        linestyle="--",
        label=f"Theoretical Gain ({theoretical_gain_db:.2f} dB)",
    )
    ax.set_xlabel("Pulse Index")
    ax.set_ylabel("Noise Power Gain [dB]")
    ax.set_title(f"Noise Gain Statistics - Pair {pair_id}")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Analyze FPGA noise and compute statistics.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/fpga/iq/"),
        help="Directory containing raw FPGA IQ data.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to save noise statistics CSV. If not specified, CSV is not saved.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to save inspection plots. If not specified, plots are not saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display inspection plots interactively with plt.show().",
    )
    args = parser.parse_args()

    if args.plot_dir:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []

    for pair_id, (start, end) in RANGES.items():
        print(f"Processing pair {pair_id}, range ({start}, {end})...")
        try:
            i_path, q_path = find_iq_pair_by_id(args.data_dir, pair_id)
            x, y = load_fpga_ram_binary_to_iq(i_path, q_path, offset_dtype=512, n_pulse=14)
            b, a = load_filter_coeffs_from_binary(i_path)
            theoretical_gain_db = compute_noise_power_gain_db(b, a)
        except Exception as e:
            print(f"  Error loading {pair_id}: {e}")
            continue

        # Generate power/dB inspection plot across range bins for all pulses (OOP style)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
        with np.errstate(divide="ignore"):
            ax1.plot(20 * np.log10(np.abs(x)))
            ax2.plot(20 * np.log10(np.abs(y)))

        ax1.set_ylabel("Magnitude [dB]")
        ax1.set_title(f"x_{pair_id}")
        ax2.set_title(f"y_{pair_id}")
        for ax in (ax1, ax2):
            ax.axvline(start, color="red", linestyle="--", label="Noise Range Start")
            ax.axvline(end, color="green", linestyle="--", label="Noise Range End")
            ax.axvspan(start, end, color="gray", alpha=0.2)
            ax.set_xlabel("Range Bin")
            ax.legend(loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.7)
        fig.tight_layout()

        if args.plot_dir:
            plot_path = args.plot_dir / f"noise_range_{pair_id}.png"
            fig.savefig(plot_path)
            print(f"  Saved inspection plot to {plot_path}")

        if args.show or (not args.csv and not args.plot_dir):
            plt.show()

        plt.close(fig)

        # Compute statistics within the noise range
        for idx in range(x.shape[1]):
            x_seg = x[start:end, idx]
            y_seg = y[start:end, idx]

            y_power_mean = np.mean(np.abs(y_seg) ** 2)
            x_power_mean = np.mean(np.abs(x_seg) ** 2)
            mean_diff_db = 10 * np.log10(y_power_mean / x_power_mean)

            y_power_median = np.median(np.abs(y_seg) ** 2)
            x_power_median = np.median(np.abs(x_seg) ** 2)
            median_diff_db = 10 * np.log10(y_power_median / x_power_median)

            print(
                f"  Pulse {idx}: theoretical_gain_db={theoretical_gain_db:.2f} dB, mean_diff={mean_diff_db:.2f} dB, median_diff={median_diff_db:.2f} dB"
            )
            csv_rows.append(
                {
                    "pair_id": pair_id,
                    "pulse_idx": idx,
                    "range_start": start,
                    "range_end": end,
                    "theoretical_gain_db": f"{theoretical_gain_db:.4f}",
                    "mean_diff_db": f"{mean_diff_db:.4f}",
                    "median_diff_db": f"{median_diff_db:.4f}",
                }
            )

    # Write CSV stats if requested
    if args.csv and csv_rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "pair_id",
                    "pulse_idx",
                    "range_start",
                    "range_end",
                    "theoretical_gain_db",
                    "mean_diff_db",
                    "median_diff_db",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved noise statistics to {args.csv}")


if __name__ == "__main__":
    main()
