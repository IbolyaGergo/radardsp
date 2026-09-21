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
        true_freq_variance=true_freq_variance,
        seed=42,
    )

    # Estimate variance
    est_variance = compute_pulse_pair_variance(iq, prf=prf)
    np.testing.assert_allclose(est_variance, true_freq_variance, rtol=0.05)


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
        true_freq_variance=true_freq_variance,
        snr=true_snr,
        seed=42,
    )

    est_snr = compute_pulse_pair_snr(iq)
    np.testing.assert_allclose(est_snr, true_snr, rtol=0.1)


# test_compute_pulse_pair_nd() {{{1
def test_compute_pulse_pair_nd():
    n_range = 100
    n_pulses = 1024
    prf = 100.0
    fd = 15.0
    true_freq_variance = 9.0
    true_snr = 100

    # Generate 2D IQ data of shape (n_range, n_pulses)
    iq_2d = np.array(
        [
            generate_gaussian_doppler_signal(
                n_pulses=n_pulses,
                prf=prf,
                fd=fd,
                true_freq_variance=true_freq_variance,
                snr=true_snr,
                seed=42 + i,
            )
            for i in range(n_range)
        ]
    )

    est_variance = compute_pulse_pair_variance(iq_2d, prf=prf)
    assert est_variance.shape == (n_range,)
    np.testing.assert_allclose(est_variance, true_freq_variance, rtol=0.3)

    est_snr = compute_pulse_pair_snr(iq_2d)
    assert est_snr.shape == (n_range,)
    np.testing.assert_allclose(est_snr, true_snr, rtol=0.4)
