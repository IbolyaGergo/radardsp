import pytest
import numpy as np
from radarsig.processing import (
    compute_pulse_phase_difference,
    compute_mean_phase_difference,
    compute_signal_quality_index,
    compute_pulse_pair_variance,
    compute_pulse_pair_snr,
)
from radarsig.signal_generator import generate_gaussian_doppler_signal


def test_compute_pulse_phase_difference_shape():
    n_range = 3
    n_pulse = 10
    iq_data = np.random.randn(n_range, n_pulse) + 1j * np.random.randn(n_range, n_pulse)

    delta_phi = compute_pulse_phase_difference(iq_data)
    assert delta_phi.shape == (n_range, n_pulse - 1)
    assert isinstance(delta_phi, np.ndarray)


def test_compute_pulse_phase_difference_values():
    # Test with known constant phase progression (e.g. delta = pi/4 per pulse)
    n_range = 1
    n_pulse = 4
    delta = np.pi / 4
    pulses = np.array([0.0, delta, 2 * delta, 3 * delta])
    iq_data = np.exp(1j * pulses).reshape(1, -1)

    delta_phi = compute_pulse_phase_difference(iq_data)

    # iq[:, :-1] * np.conj(iq[:, 1:]) -> angle(exp(j*p_t) * exp(-j*p_{t+1})) = p_t - p_{t+1} = -delta
    expected = np.full((1, n_pulse - 1), -delta)
    np.testing.assert_allclose(delta_phi, expected, atol=1e-7)


def test_compute_mean_phase_difference():
    n_range = 1
    n_pulse = 5
    delta = np.pi / 4
    pulses = np.array([0.0, delta, 2 * delta, 3 * delta, 4 * delta])
    iq_data = np.exp(1j * pulses).reshape(1, -1)

    mean_phi = compute_mean_phase_difference(iq_data)

    assert mean_phi.shape == (n_range,)
    np.testing.assert_allclose(mean_phi, [-delta], atol=1e-7)


def test_compute_signal_quality_index():
    n_range = 1
    n_pulse = 5
    # Constant signal (pure tone) should have SQI = 1.0
    iq_data = np.ones((n_range, n_pulse), dtype=complex)
    sqi = compute_signal_quality_index(iq_data)
    assert sqi.shape == (n_range,)
    np.testing.assert_allclose(sqi, [1.0], atol=1e-7)

    # Zero signal should handle division by zero gracefully (SQI = 0.0)
    iq_zero = np.zeros((n_range, n_pulse), dtype=complex)
    sqi_zero = compute_signal_quality_index(iq_zero)
    np.testing.assert_allclose(sqi_zero, [0.0], atol=1e-7)


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
