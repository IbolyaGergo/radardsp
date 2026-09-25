#! /usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from radarsig.signal_generator import generate_gaussian_doppler_signal
from radarsig.processing import compute_pulse_pair_mean, compute_pulse_pair_variance

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
PRF = 2000.0
N_MC = 200  # number of Monte Carlo runs
N_PULSES_LIST = np.arange(10, 1001, 10)

TRUE_STD = PRF / 10.0
TRUE_MEAN = 0.0


# run_monte_carlo() {{{1
def run_monte_carlo(
    n_pulses: int,
    prf: float,
    estimator_func,
    estimator_type: str,
    n_mc: int,
) -> tuple[float, float]:
    """
    Run Monte Carlo simulations to estimate the mean and standard deviation
    of the Doppler property estimator for a given number of pulses.
    """
    estimates = []
    for i in range(n_mc):
        iq = generate_gaussian_doppler_signal(
            n_pulses=n_pulses,
            prf=prf,
            fd=TRUE_MEAN,
            true_freq_variance=(TRUE_STD) ** 2,
            snr=None,
            seed=i,
        )

        est = estimator_func(iq, prf=prf)
        if estimator_type == "width":
            est = np.sqrt(est)
        estimates.append(est)

    mean_est = np.mean(estimates)
    std_est = np.std(estimates)
    return float(mean_est), float(std_est)


# main() {{{1
def main():
    parser = argparse.ArgumentParser(description="Study pulse-pair estimator convergence.")
    parser.add_argument(
        "--estimator",
        choices=["mean", "width"],
        default="width",
        help="Estimator type to study (mean or width)",
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
    args = parser.parse_args()

    # Estimator-specific setup
    if args.estimator == "width":
        true_value = TRUE_STD
        estimator_func = compute_pulse_pair_variance
        title = "Pulse-Pair Spectral Width Estimator Convergence"
    elif args.estimator == "mean":
        true_value = TRUE_MEAN
        estimator_func = compute_pulse_pair_mean
        title = "Pulse-Pair Mean Estimator Convergence"
    else:
        raise ValueError(f"Unknown estimator: {args.estimator}")

    results = []

    print(f"Running pulse-pair estimator convergence study for estimator: {args.estimator}...")
    for n_pulses in N_PULSES_LIST:
        mean_est, std_est = run_monte_carlo(
            n_pulses=n_pulses,
            prf=PRF,
            estimator_func=estimator_func,
            estimator_type=args.estimator,
            n_mc=N_MC,
        )

        norm_bias = np.abs(mean_est - true_value) / PRF
        norm_std = std_est / PRF

        results.append(
            {
                "n_pulses": n_pulses,
                "mean_est": mean_est,
                "std_est": std_est,
                "norm_bias": norm_bias,
                "norm_std": norm_std,
            }
        )

    # Plotting results using object-oriented matplotlib API
    fig, ax = plt.subplots(figsize=(9, 6))

    n_pulses_arr = np.array([r["n_pulses"] for r in results])

    if args.metric == "std":
        norm_std_arr = np.array([r["norm_std"] for r in results])
        ax.plot(
            n_pulses_arr,
            norm_std_arr,
            marker="o",
            linestyle="--",
            color="g",
            markersize=3,
            label="Normalized Standard Deviation",
        )
        metric_title_part = "Normalized Standard Deviation"
    elif args.metric == "bias":
        norm_bias_arr = np.array([r["norm_bias"] for r in results])
        ax.plot(
            n_pulses_arr,
            norm_bias_arr,
            marker="o",
            linestyle="-",
            color="b",
            markersize=3,
            label="Normalized Bias",
        )
        metric_title_part = "Normalized Bias"
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    ax.plot(
        n_pulses_arr,
        1.0 / np.sqrt(n_pulses_arr),
        label=r"$\frac{1}{\sqrt{\text{Number of Pulses}}}$",
        color="k",
        linestyle=":",
    )

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
