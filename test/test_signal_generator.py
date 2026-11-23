"""Tests for radarsig.signal_generator.

Numeric examples follow
Curtis, C. D., 2018: Weather radar time series simulation: Improving accuracy and performance.
J. Atmos. Oceanic Technol., 35, 2169–2187, https://doi.org/10.1175/JTECH-D-17-0215.1.
"""

import numpy as np
import pytest
from scipy.fft import fft, fftfreq, fftshift

from radarsig.signal_generator import (
    _autocorr_threshold,
    _first_lag_below_threshold,
    _sim_length,
    generate_gaussian_doppler_signal,
)

# Parameters of the example used throughout the article.
N_PULSES = 16
PRF = 100.0  # Hz
NYQUIST_VELOCITY = 32.0  # m/s


def _width_hz(width_v: float) -> float:
    """Spectral width in Hz from a width given in velocity units (m/s).

    The article parameterises the spectrum by its mean/width ratio in velocity units; with
    the Nyquist velocity v_a that ratio gives width_f = prf * width_v / (2 * v_a).
    """
    return PRF * width_v / (2 * NYQUIST_VELOCITY)


# test_generate_gaussian_doppler_signal_width() {{{1
def test_generate_gaussian_doppler_signal_width():
    n_pulses = 1024
    prf = 100.0
    true_mean = 15.0
    width = 3.0  # Hz
    seed = 42

    iq = generate_gaussian_doppler_signal(
        n_pulses=n_pulses,
        prf=prf,
        mean=true_mean,
        width=width,
        seed=seed,
    )

    # Compute shifted FFT and Power Spectral Density (PSD)
    freqs = fftshift(fftfreq(n_pulses, d=1.0 / prf))
    spectrum = fftshift(fft(iq))
    psd = np.abs(spectrum) ** 2

    # Normalize PSD to form a discrete probability distribution
    psd_norm = psd / np.sum(psd)

    # Compute mean frequency and frequency variance from the PSD
    mean_freq = np.sum(freqs * psd_norm)
    est_variance = np.sum(psd_norm * (freqs - mean_freq) ** 2)

    # Verify center frequency and spectral variance
    np.testing.assert_allclose(mean_freq, true_mean, atol=2.0)
    np.testing.assert_allclose(est_variance, width**2, rtol=0.01)


# test_autocorr_threshold() {{{1
@pytest.mark.parametrize(
    ("width_v", "expected"),
    [
        (4.0, 25.0),  # formula gives 25.98 dB -> capped at 25 dB
        (0.5, 11.17727150175935),
    ],
    ids=["capped-at-max", "below-max"],
)
def test_autocorr_threshold(width_v, expected):
    threshold = _autocorr_threshold(N_PULSES, PRF, _width_hz(width_v))

    assert threshold == pytest.approx(expected)


# test_first_lag_below_threshold() {{{1
def test_first_lag_below_threshold():
    # With a 25 dB threshold and a 4 m/s width the first lag below it is 9 pulses.
    lag = _first_lag_below_threshold(PRF, _width_hz(4.0), threshold=25.0)

    assert lag == 9


# test_sim_length() {{{1
@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(25.0, 141), (None, 95)],
    ids=["fixed-threshold", "adaptive-threshold"],
)
def test_sim_length(threshold, expected):
    length = _sim_length(N_PULSES, PRF, _width_hz(0.5), autocorr_threshold=threshold)

    assert length == expected


# test_sim_length_is_at_least_one_extra_lag() {{{1
@pytest.mark.parametrize("width_v", [0.25, 0.5, 1.0, 4.0])
def test_sim_length_is_at_least_one_extra_lag(width_v):
    width_f = _width_hz(width_v)
    lag = _first_lag_below_threshold(PRF, width_f, _autocorr_threshold(N_PULSES, PRF, width_f))

    assert _sim_length(N_PULSES, PRF, width_f) >= N_PULSES + lag
