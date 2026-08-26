import pytest
import numpy as np
from radarsig.processing import (
    compute_pulse_phase_difference,
    compute_mean_phase_difference,
)


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
