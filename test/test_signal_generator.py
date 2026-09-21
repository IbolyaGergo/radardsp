import numpy as np
from radarsig.signal_generator import (
    generate_gaussian_doppler_signal,
    compute_pulse_pair_variance,
    compute_pulse_pair_snr,
)


# test_compute_pulse_pair_variance() {{{1
def test_compute_pulse_pair_variance():
    n_pulses = 1024
    prf = 100.0
    fd = 15.0
    true_freq_variance = 9.0  # Hz^2

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
    np.testing.assert_equal(est_variance, 4.0 * 2.1502659330558944)


# test_compute_pulse_pair_snr() {{{1
def test_compute_pulse_pair_snr():
    n_pulses = 1024
    prf = 100.0
    fd = 15.0
    true_freq_variance = 9.0  # Hz^2
    true_snr = 100

    # Generate signal
    iq = generate_gaussian_doppler_signal(
        n_pulses=n_pulses,
        prf=prf,
        fd=fd,
        true_variance=true_freq_variance,
        snr=true_snr,
        seed=42,
    )

    est_snr = compute_pulse_pair_snr(iq)
    np.testing.assert_allclose(est_snr, true_snr, rtol=0.1)
