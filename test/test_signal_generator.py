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
    _gaussian_doppler_samples,
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


def _psd_mean_and_variance(iq: np.ndarray, prf: float) -> tuple[float, float]:
    """Mean frequency (Hz) and frequency variance (Hz^2) of the spectrum of `iq`."""
    freqs = fftshift(fftfreq(len(iq), d=1.0 / prf))
    psd = np.abs(fftshift(fft(iq))) ** 2
    psd_norm = psd / np.sum(psd)  # treat the PSD as a discrete distribution
    mean_freq = np.sum(freqs * psd_norm)
    return mean_freq, np.sum(psd_norm * (freqs - mean_freq) ** 2)


# test_generate_gaussian_doppler_signal_width() {{{1
def test_generate_gaussian_doppler_signal_width():
    n_pulses = 1024
    prf = 100.0
    true_mean = 15.0
    width = 3.0  # Hz
    n_seeds = 10

    estimates = [
        _psd_mean_and_variance(
            generate_gaussian_doppler_signal(
                n_pulses=n_pulses, prf=prf, mean=true_mean, width=width, seed=seed
            ),
            prf=prf,
        )
        for seed in range(n_seeds)
    ]
    mean_freq, est_variance = np.mean(estimates, axis=0)

    # A single periodogram-based estimate is noisy (relative std ~12 %) and biased ~7 % high
    # for this width, so assert on the mean over seeds with a matching tolerance.
    np.testing.assert_allclose(mean_freq, true_mean, atol=2.0)
    np.testing.assert_allclose(est_variance, width**2, rtol=0.15)


# test_generate_gaussian_doppler_signal_shape() {{{1
@pytest.mark.parametrize("avoid_wraparound", [True, False], ids=["trimmed", "legacy"])
def test_generate_gaussian_doppler_signal_shape(avoid_wraparound):
    iq = generate_gaussian_doppler_signal(
        n_pulses=N_PULSES,
        prf=PRF,
        mean=15.0,
        width=_width_hz(4.0),
        seed=0,
        avoid_wraparound=avoid_wraparound,
    )

    assert iq.shape == (N_PULSES,)


# test_generate_gaussian_doppler_signal_trims_sim_length() {{{1
def test_generate_gaussian_doppler_signal_trims_sim_length():
    width = _width_hz(4.0)
    mean = 15.0
    seed = 7
    # 25 samples are simulated for these parameters, so trimming is observable.
    n_sim = _sim_length(N_PULSES, PRF, width)
    assert n_sim > N_PULSES

    # Default: simulate n_sim samples, return the first n_pulses of them.
    np.random.seed(seed)  # mirrors the global seeding used by the generator
    expected = _gaussian_doppler_samples(n_sim, PRF, mean, width)
    iq = generate_gaussian_doppler_signal(
        n_pulses=N_PULSES, prf=PRF, mean=mean, width=width, seed=seed
    )
    np.testing.assert_array_equal(iq, expected[:N_PULSES])

    # Legacy: simulate exactly n_pulses samples, no trimming.
    np.random.seed(seed)
    expected_legacy = _gaussian_doppler_samples(N_PULSES, PRF, mean, width)
    iq_legacy = generate_gaussian_doppler_signal(
        n_pulses=N_PULSES, prf=PRF, mean=mean, width=width, seed=seed, avoid_wraparound=False
    )
    np.testing.assert_array_equal(iq_legacy, expected_legacy)


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
