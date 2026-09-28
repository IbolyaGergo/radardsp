#! /usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from radarsig.signal_generator import generate_gaussian_doppler_signal
from radarsig.processing import (
    compute_pulse_pair_mean,
    compute_pulse_pair_variance,
    compute_pulse_pair_snr,
)

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
PRF = 2000.0
N_MC = 200  # number of Monte Carlo runs
N_PULSES_LIST = np.arange(10, 1001, 10)

TRUE_STD = PRF / 10.0
TRUE_MEAN = 0.0

ESTIMATOR_CONFIGS = {
    "mean": {
        "title": "Pulse-Pair Mean Estimator Convergence",
        "compute_est": lambda iq, prf: compute_pulse_pair_mean(iq, prf=prf),
        "true_value": lambda snr: TRUE_MEAN,
        "compute_norm_bias": lambda mean, true, prf: np.abs(mean - true) / prf,
        "compute_norm_std": lambda std, prf: std / prf,
        "labels": {
            "bias": "Normalized Bias",
            "std": "Normalized Standard Deviation",
        },
    },
    "width": {
        "title": "Pulse-Pair Spectral Width Estimator Convergence",
        "compute_est": lambda iq, prf: np.sqrt(
            np.maximum(0.0, compute_pulse_pair_variance(iq, prf=prf))
        ),
        "true_value": lambda snr: TRUE_STD,
        "compute_norm_bias": lambda mean, true, prf: np.abs(mean - true) / prf,
        "compute_norm_std": lambda std, prf: std / prf,
        "labels": {
            "bias": "Normalized Bias",
            "std": "Normalized Standard Deviation",
        },
    },
}


# run_monte_carlo() {{{1
def run_monte_carlo(
    n_pulses: int,
    prf: float,
    config: dict,
    snr: float | None,
    n_mc: int,
) -> tuple[float, float]:
    """
    Run Monte Carlo simulations to estimate the mean and standard deviation
    of the Doppler property estimator for a given number of pulses.
    """
    estimates = []
    if snr is None or np.isinf(snr):
        linear_snr = None
    else:
        linear_snr = 10 ** (snr / 10.0)

    for i in range(n_mc):
        iq = generate_gaussian_doppler_signal(
            n_pulses=n_pulses,
            prf=prf,
            fd=TRUE_MEAN,
            true_freq_variance=(TRUE_STD) ** 2,
            snr=linear_snr,
            seed=i,
        )

        est = config["compute_est"](iq, prf=prf)
        estimates.append(est)

    mean_est = np.mean(estimates)
    std_est = np.std(estimates)
    return float(mean_est), float(std_est)


# main() {{{1
def main():
    parser = argparse.ArgumentParser(description="Study pulse-pair estimator convergence.")
    parser.add_argument(
        "--estimator",
        choices=list(ESTIMATOR_CONFIGS.keys()),
        default="width",
        help="Estimator type to study",
    )
    parser.add_argument(
        "--metric",
        choices=["bias", "std"],
        default="std",
        help="Metric to plot (bias or std)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path for the plot. If not specified, displays interactively.",
    )
    parser.add_argument(
        "--snr",
        nargs="+",
        type=float,
        default=[float("inf"), 20.0, 10.0],
        help="SNR values in dB to simulate (use inf for noise-free)",
    )

    args = parser.parse_args()

    config = ESTIMATOR_CONFIGS[args.estimator]
    title = config["title"]

    results = []

    print(f"Running pulse-pair estimator convergence study for estimator: {args.estimator}...")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(args.snr)))

    for snr, color in zip(args.snr, colors):
        results = []
        snr_label = "Noise-Free" if np.isinf(snr) else f"{snr} dB"
        print(f"Running study for SNR: {snr_label}...")

        true_value = config["true_value"](snr)

        for n_pulses in N_PULSES_LIST:
            mean_est, std_est = run_monte_carlo(
                n_pulses=n_pulses,
                prf=PRF,
                config=config,
                snr=snr,
                n_mc=N_MC,
            )
            norm_bias = config["compute_norm_bias"](mean_est, true_value, PRF)
            norm_std = config["compute_norm_std"](std_est, PRF)
            results.append(
                {
                    "n_pulses": n_pulses,
                    "norm_bias": norm_bias,
                    "norm_std": norm_std,
                }
            )

        n_pulses_arr = np.array([r["n_pulses"] for r in results])
        y_arr = np.array(
            [r["norm_std"] if args.metric == "std" else r["norm_bias"] for r in results]
        )

        ax.plot(
            n_pulses_arr,
            y_arr,
            marker="o",
            linestyle="-",
            color=color,
            markersize=3,
            label=f"SNR: {snr_label}",
        )

    # Optional: Plot theoretical 1/sqrt(N) reference for std metric
    if args.metric == "std":
        ax.plot(
            N_PULSES_LIST,
            1.0 / np.sqrt(N_PULSES_LIST),
            label=r"$\frac{1}{\sqrt{\text{Number of Pulses}}}$ (Theory)",
            color="black",
            linestyle=":",
            linewidth=1.5,
        )

    metric_title_part = config["labels"][args.metric]

    ax.set_xlabel("Number of Pulses", fontsize=12)
    ax.set_ylabel(metric_title_part, fontsize=12)
    ax.set_title(f"{title} ({metric_title_part})", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    plt.tight_layout()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=300)
        plt.close()
        print(f"\nPlot saved to {args.out}")
    else:
        output_path = Path(f"pulse_pair_{args.estimator}_{args.metric}_convergence_study.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"\nPlot saved to {output_path}")
        plt.show()


if __name__ == "__main__":
    main()
