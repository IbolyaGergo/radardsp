#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from radarsig.signal_generator import generate_gaussian_doppler_signal
from radarsig.processing import compute_pulse_pair_mean, compute_pulse_pair_variance

# ==========================================
# CONFIGURATION
# ==========================================
DOPPLER_PROPERTY = "std"  # Options: "std" or "mean"
PRF = 2000.0
N_MC = 200  # number of Monte Carlo runs
N_PULSES_LIST = np.arange(10, 1001, 10)

TRUE_STD = PRF / 10.0
TRUE_MEAN = 0.0

# Property-specific setup
if DOPPLER_PROPERTY == "std":
    TRUE_VALUE = TRUE_STD
    ESTIMATOR_FUNC = compute_pulse_pair_variance
    TITLE = "Pulse-Pair Spectral Width Estimator Convergence"
elif DOPPLER_PROPERTY == "mean":
    TRUE_VALUE = TRUE_MEAN
    ESTIMATOR_FUNC = compute_pulse_pair_mean
    TITLE = "Pulse-Pair Mean Estimator Convergence"
else:
    raise ValueError(f"Unknown DOPPLER_PROPERTY: {DOPPLER_PROPERTY}")


# run_monte_carlo() {{{1
def run_monte_carlo(
    n_pulses: int,
    prf: float,
    estimator_func,
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
        if DOPPLER_PROPERTY == "std":
            est = np.sqrt(est)
        estimates.append(est)

    mean_est = np.mean(estimates)
    std_est = np.std(estimates)
    return float(mean_est), float(std_est)


# main() {{{1
def main():
    results = []

    print(f"Running pulse-pair estimator convergence study for property: {DOPPLER_PROPERTY}...")
    for n_pulses in N_PULSES_LIST:
        df = PRF / n_pulses

        mean_est, std_est = run_monte_carlo(
            n_pulses=n_pulses,
            prf=PRF,
            estimator_func=ESTIMATOR_FUNC,
            n_mc=N_MC,
        )

        rel_bias = np.abs(mean_est - TRUE_VALUE) / PRF
        rel_std = std_est / PRF

        results.append(
            {
                "n_pulses": n_pulses,
                "mean_est": mean_est,
                "std_est": std_est,
                "rel_bias": rel_bias,
                "rel_std": rel_std,
            }
        )

    # Plotting results using object-oriented matplotlib API
    fig, ax = plt.subplots(figsize=(9, 6))

    n_pulses_arr = np.array([r["n_pulses"] for r in results])
    rel_bias_arr = np.array([r["rel_bias"] for r in results])
    rel_std_arr = np.array([r["rel_std"] for r in results])

    ax.plot(
        n_pulses_arr,
        rel_bias_arr,
        marker="o",
        linestyle="-",
        color="b",
        markersize=3,
        label=f"Relative Bias",
    )

    ax.plot(
        n_pulses_arr,
        rel_std_arr,
        marker="o",
        linestyle="--",
        color="g",
        markersize=3,
        label=f"Relative Standard Deviation",
    )
    ax.plot(
        n_pulses_arr,
        1.0 / np.sqrt(n_pulses_arr),
        label=r"$\frac{1}{\sqrt{\text{Number of Pulses}}}$",
    )

    ax.set_xlabel("Number of Pulses", fontsize=12)
    ax.set_title(TITLE, fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    plt.tight_layout()
    output_path = f"pulse_pair_{DOPPLER_PROPERTY}_convergence_study.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
