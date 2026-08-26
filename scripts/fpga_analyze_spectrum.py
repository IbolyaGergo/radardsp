#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import get_window
from radarsig.fpga_io import (
    load_fpga_ram_binary_to_iq,
    load_filter_coeffs_from_binary,
    find_iq_pairs,
)
from radarsig.fpga_analysis import (
    compute_median_ratio_spectrum,
    compute_csd_spectrum,
    compute_coherence,
)

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


def plot_result(results_list, method: str, pair_id: str, out_dir: Path):
    config = ANALYSIS_METHODS[method]
    plt.figure(figsize=(9, 5))
    ax1 = plt.gca()

    if method in ("median", "csd"):
        # Plot theoretical results
        _, freqs, h_db, _ = results_list[0]
        ax1.plot(
            freqs,
            h_db,
            label="Theoretical ($|H|$) - freqz",
            color="black",
            linewidth=2,
            linestyle="--",
            zorder=5,
        )

        # Plot empirical curves for each window
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
        for (win_name, _, _, emp_db), color in zip(results_list, colors):
            label_str = f"Empirical ({win_name})"
            ax1.plot(freqs, emp_db, label=label_str, color=color, alpha=0.8)

        ax1.set_ylabel("Magnitude [dB]")
        ax1.legend(loc="upper right")
    elif method == "coherence":
        # For coherence, take the first (or only) window result
        _, freqs, h_db, coherence, _ = results_list[0]
        line1 = ax1.plot(
            freqs,
            h_db,
            label="Theoretical ($|H|$) - freqz",
            color="black",
            linewidth=2,
            linestyle="--",
        )
        ax1.set_ylabel("Magnitude [dB]", color="black")
        ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twinx()
        line2 = ax2.plot(
            freqs, coherence, label=r"Coherence ($\gamma_{xy}^2$)", color="green", linewidth=2
        )
        ax2.set_ylabel("Coherence", color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.set_ylim(-0.05, 1.05)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")

    ax1.set_title(f"{config['title_prefix']} - Pair {pair_id}")
    ax1.set_xlabel(r"Digital Frequency ($\omega$) [rad/sample]")
    ax1.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"{config['filename_prefix']}_{pair_id}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved plot to {plot_path}")


def process_pair(pair_id, i_path, q_path, method, window_names, out_dir):
    try:
        x, y = load_fpga_ram_binary_to_iq(i_path, q_path, offset_dtype=512, n_pulse=14)
        b, a = load_filter_coeffs_from_binary(i_path)
    except Exception as e:
        print(f"  Error loading {pair_id}: {e}")
        return

    method_meta = ANALYSIS_METHODS[method]

    # Store results for each window: list of (window_name, freqs, h_db, emp_db)
    results_to_plot = []

    for window_name in window_names:
        window_arr = get_window(window_name, x.shape[1])
        freqs, h_db, emp_db = method_meta["func"](x, y, b, a, window=window_arr)

        results_to_plot.append((window_name, freqs, h_db, emp_db))

    plot_result(results_to_plot, method, pair_id, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Analyze IIR filter spectrum.")
    parser.add_argument("--dir", default="data/raw/fpga/iq", help="Directory with IQ data")
    parser.add_argument(
        "--method", choices=list(ANALYSIS_METHODS.keys()), default="median", help="Analysis method"
    )
    parser.add_argument("--pair", default=None, help="Specific pair ID to process (e.g. 004)")
    parser.add_argument("--out-dir", default=None, help="Output directory for plots")
    parser.add_argument(
        "--window",
        nargs="+",
        default=["hamming"],
        help="Window function name(s) (e.g., hamming hann boxcar blackman)",
    )

    args = parser.parse_args()

    data_dir = Path(args.dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"results/fpga_spectrum/{args.method}")

    if args.pair:
        i_path = data_dir / f"{args.pair}_i.data"
        q_path = data_dir / f"{args.pair}_q.data"
        if not i_path.exists() or not q_path.exists():
            print(f"Error: Pair {args.pair} files not found in {data_dir}")
            return
        print(f"Processing single pair: {args.pair} (method: {args.method})...")
        process_pair(args.pair, i_path, q_path, args.method, args.window, out_dir)
    else:
        try:
            pairs = find_iq_pairs(data_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        if not pairs:
            print(f"No IQ pairs found in {data_dir}")
            return
        print(f"Found {len(pairs)} IQ pair(s). Processing method: {args.method}...")
        for pair_id, i_path, q_path in pairs:
            print(f"Processing pair: {pair_id}...")
            process_pair(pair_id, i_path, q_path, args.method, out_dir)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
