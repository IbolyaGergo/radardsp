import numpy as np
import pytest
from scipy.fft import fft, fftfreq, fftshift

from radarsig.signal_generator import (
    _autocorr_threshold,
    _first_lag_below_threshold,
    _sim_length,
    generate_gaussian_doppler_signal,
)


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


"""Examples from the article
Curtis, C. D., 2018: Weather radar time series simulation: Improving accuracy and performance.
J. Atmos. Oceanic Technol., 35, 2169–2187, https://doi.org/10.1175/JTECH-D-17-0215.1."""


# test_autocorr_threshold() {{{1
def test_autocorr_threshold():
    n_pulses = 16
    v_a = 32
    prf = 100

    width_v = 4
    mean_width_ratio = v_a / width_v
    width_f = prf / (2 * mean_width_ratio)  # in Hz

    threshold = _autocorr_threshold(n_pulses, prf, width_f)
    assert threshold == pytest.approx(25.0)

    width_v = 0.5
    mean_width_ratio = v_a / width_v
    width_f = prf / (2 * mean_width_ratio)  # in Hz

    threshold = _autocorr_threshold(n_pulses, prf, width_f)
    assert threshold == pytest.approx(11.17727150175935)


# test_lag_below_threshold() {{{1
def test_lag_below_threshold():
    v_a = 32
    prf = 100

    width_v = 4
    mean_width_ratio = v_a / width_v
    width_f = prf / (2 * mean_width_ratio)  # in Hz

    threshold = 25  # dB

    assert _first_lag_below_threshold(prf, width_f, threshold) == 9


# test_sim_length() {{{1
def test_sim_length():
    n_pulses = 16
    v_a = 32
    prf = 100

    width_v = 0.5
    mean_width_ratio = v_a / width_v
    width_f = prf / (2 * mean_width_ratio)  # in Hz

    # if the threshold is 25 dB
    assert _sim_length(n_pulses, prf, width_f, autocorr_threshold=25) == 141
    # if threshold is according to the formula in the article
    assert _sim_length(n_pulses, prf, width_f) == 95
