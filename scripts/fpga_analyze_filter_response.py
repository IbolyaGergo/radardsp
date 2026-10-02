#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.fft import fft, fftshift
from scipy.signal import get_window, freqz
from radarsig.fpga_io import (
    load_fpga_ram_binary_to_iq,
    load_filter_coeffs_from_binary,
    find_iq_pairs,
)
from radarsig.fpga_analysis import (
    compute_median_ratio_spectrum,
    compute_csd_spectrum,
    compute_coherence,
    filter_valid_bins,
)


@dataclass
class FPGASpectrumResult:
    window_name: str
    freqs: np.ndarray
    h_db: np.ndarray
    emp_data: np.ndarray
    coherence: np.ndarray | None = None


def plot_median_spectrum(ax, results_list, pair_id):
    """Plot theoretical response, empirical median magnitude [dB], and dual-axis coherence."""
    base = results_list[0]
    line1 = ax.plot(
        base.freqs,
        base.h_db,
        label="Theoretical ($|H|$) - freqz",
        color="black",
        linewidth=2,
        linestyle="--",
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    emp_lines = []
    for res, color in zip(results_list, colors):
        line = ax.plot(
            res.freqs, res.emp_data, label=f"Empirical ({res.window_name})", color=color, alpha=0.8
        )
        emp_lines.extend(line)

    ax.set_ylabel("Magnitude [dB]", color="black")
    ax.tick_params(axis="y", labelcolor="black")

    ax2 = ax.twinx()
    coherence_lines = []
    for res, color in zip(results_list, colors):
        if res.coherence is not None:
            line = ax2.plot(
                res.freqs,
                res.coherence,
                label=r"Coherence ($\gamma_{xy}^2$) " + f"[{res.window_name}]",
                color="green",
                linewidth=1.5,
                linestyle="-.",
                alpha=0.8,
            )
            coherence_lines.extend(line)

    ax2.set_ylabel("Coherence", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(-0.05, 1.05)

    lines = line1 + emp_lines + coherence_lines
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="lower right")


def plot_coherence_spectrum(ax, results_list, pair_id):
    """Plot magnitude response with dual-axis coherence."""
    base = results_list[0]
    line1 = ax.plot(
        base.freqs,
        base.h_db,
        label="Theoretical ($|H|$) - freqz",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    ax.set_ylabel("Magnitude [dB]", color="black")
    ax.tick_params(axis="y", labelcolor="black")

    ax2 = ax.twinx()
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    coherence_lines = []

    for res, color in zip(results_list, colors):
        line = ax2.plot(
            res.freqs,
            res.emp_data,
            label=r"Coherence ($\gamma_{xy}^2$) " + f"[{res.window_name}]",
            color=color,
            linewidth=2,
            alpha=0.8,
        )
        coherence_lines.extend(line)

    ax2.set_ylabel("Coherence", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(-0.05, 1.05)

    lines = line1 + coherence_lines
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="lower right")


ANALYSIS_METHODS = {
    "median": {
        "func": compute_median_ratio_spectrum,
        "title_prefix": "IIR Filter Frequency Response (Median)",
        "filename_prefix": "filter_spectrum_median",
        "plotter": plot_median_spectrum,
    },
    "csd": {
        "func": compute_csd_spectrum,
        "title_prefix": "IIR Filter Frequency Response (CSD)",
        "filename_prefix": "filter_spectrum_csd",
        "plotter": plot_median_spectrum,
    },
    "coherence": {
        "func": compute_coherence,
        "title_prefix": "Magnitude-Squared Coherence & Theoretical Response",
        "filename_prefix": "filter_spectrum_coherence",
        "plotter": plot_coherence_spectrum,
    },
}


def plot_result(results_list, method: str, pair_id: str, out_file: Path | None):
    config = ANALYSIS_METHODS[method]
    plt.figure(figsize=(9, 5))
    ax = plt.gca()

    # Dispatch to the specific plotter function
    config["plotter"](ax, results_list, pair_id)

    ax.set_title(f"{config['title_prefix']} - Pair {pair_id}")
    ax.set_xlabel(r"Digital Frequency ($\omega$) [rad/sample]")
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.tight_layout()

    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file)
        plt.close()
        print(f"  Saved plot to {out_file}")
    else:
        plt.show()


def process_pair(pair_id, i_path, q_path, method, window_names, out_file):
    try:
        x, y = load_fpga_ram_binary_to_iq(i_path, q_path, offset_dtype=512, n_pulse=14)
        b, a = load_filter_coeffs_from_binary(i_path)
    except Exception as e:
        print(f"  Error loading {pair_id}: {e}")
        return

    method_meta = ANALYSIS_METHODS[method]

    # Separate theoretical filter response computation over whole unit circle [0, 2pi], shifted to [-pi, pi]
    fft_len = x.shape[1]
    w, h = freqz(b, a, worN=fft_len, whole=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_db = fftshift(20 * np.log10(np.abs(h)))
    freqs = np.linspace(-np.pi, np.pi, fft_len)

    x = x[:3165, :]
    y = y[:3165, :]

    x_valid, y_valid = filter_valid_bins(x, y, snr_threshold=1e-6)

    results_to_plot = []
    for window_name in window_names:
        window_arr = get_window(window_name, fft_len)

        x_win = x_valid * window_arr
        y_win = y_valid * window_arr

        if method == "median":
            emp_raw = method_meta["func"](x_win, y_win, window=window_arr)
            emp_data = fftshift(emp_raw)
            coh_raw = compute_coherence(x_win, y_win, fft_len=fft_len, window=window_arr)
            coherence = fftshift(coh_raw)
        elif method == "coherence":
            coh_raw = method_meta["func"](x_win, y_win, fft_len=fft_len, window=window_arr)
            emp_data = fftshift(coh_raw)
            coherence = None
        else:
            csd_raw = method_meta["func"](x_win, y_win, fft_len=fft_len, window=window_arr)
            emp_data = fftshift(csd_raw)
            coherence = None

        results_to_plot.append(
            FPGASpectrumResult(
                window_name=window_name,
                freqs=freqs,
                h_db=h_db,
                emp_data=emp_data,
                coherence=coherence,
            )
        )

    plot_result(results_to_plot, method, pair_id, out_file)


def main():
    parser = argparse.ArgumentParser(description="Analyze IIR filter spectrum.")
    parser.add_argument("--dir", default="data/raw/fpga/iq", help="Directory with IQ data")
    parser.add_argument(
        "--method", choices=list(ANALYSIS_METHODS.keys()), default="median", help="Analysis method"
    )
    parser.add_argument(
        "--pair",
        default="000",
        help="Pair ID to process (default: 000)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path for the plot. If not specified, displays interactively.",
    )
    parser.add_argument(
        "--window",
        nargs="+",
        default=["hamming"],
        help="Window function name(s) (e.g., hamming hann boxcar blackman)",
    )

    args = parser.parse_args()

    data_dir = Path(args.dir)

    i_path = data_dir / f"{args.pair}_i.data"
    q_path = data_dir / f"{args.pair}_q.data"
    if not i_path.exists() or not q_path.exists():
        print(f"Error: Pair {args.pair} files not found in {data_dir}")
        return
    print(f"Processing single pair: {args.pair} (method: {args.method})...")
    process_pair(args.pair, i_path, q_path, args.method, args.window, args.out)


if __name__ == "__main__":
    main()
