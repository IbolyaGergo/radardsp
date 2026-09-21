import numpy as np
from radarsig.signal_generator import (
    generate_gaussian_doppler_signal,
    compute_pulse_pair_variance,
)


def test_compute_pulse_pair_variance():
    n_pulses = 1024
    prf = 100.0
    fd = 15.0
    true_freq_variance = 9.0  # Hz^2
    true_velocity_variance = 0.25 * true_freq_variance

    # Generate signal
    iq = generate_gaussian_doppler_signal(
        n_pulses=n_pulses,
        prf=prf,
        fd=fd,
        true_variance=true_freq_variance,
        seed=42,
    )

    # Estimate variance
    est_variance = compute_pulse_pair_variance(iq, prf=prf)
    np.testing.assert_equal(est_variance, 2.1502659330558944)
