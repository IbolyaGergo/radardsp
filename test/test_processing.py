import numpy as np
import pytest
from scipy.signal import get_window

from radarsig.processing import (
    compute_pulse_phase_difference,
    compute_pulse_pair_mean,
    compute_signal_quality_index,
    compute_pulse_pair_variance,
    compute_pulse_pair_width,
    compute_pulse_pair_snr,
    compute_pulse_pair_snr_db,
    compute_spectrum,
)
from radarsig.signal_generator import generate_gaussian_doppler_signal


# test_compute_pulse_phase_difference_shape() {{{1
def test_compute_pulse_phase_difference_shape():
    n_range = 3
    n_pulse = 10
    iq_data = np.random.randn(n_range, n_pulse) + 1j * np.random.randn(n_range, n_pulse)

    delta_phi = compute_pulse_phase_difference(iq_data)
    assert delta_phi.shape == (n_range, n_pulse - 1)
    assert isinstance(delta_phi, np.ndarray)


# test_compute_pulse_phase_difference_values() {{{1
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


# test_compute_pulse_pair_mean() {{{1
def test_compute_pulse_pair_mean():
    n_range = 1
    n_pulse = 5
    delta = np.pi / 4
    prf = 1
    pulses = np.array([0.0, delta, 2 * delta, 3 * delta, 4 * delta])
    iq_data = np.exp(1j * pulses).reshape(1, -1)

    mean_freq = compute_pulse_pair_mean(iq_data, prf=1)
    expected_mean_freq = -prf / (2 * np.pi) * delta

    assert mean_freq.shape == (n_range,)
    np.testing.assert_allclose(mean_freq, expected_mean_freq, atol=1e-7)


# test_compute_signal_quality_index() {{{1
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
    true_mean = 15.0
    width = 3.0  # Hz

    # Generate signal
    iq = generate_gaussian_doppler_signal(
        n_pulses=n_pulses,
        prf=prf,
        mean=true_mean,
        width=width,
        seed=42,
    )

    # Estimate variance
    est_variance = compute_pulse_pair_variance(iq, prf=prf)
    np.testing.assert_allclose(est_variance, width**2, rtol=0.05)


# test_compute_pulse_pair_width() {{{1
def test_compute_pulse_pair_width():
    n_pulses = 1024
    prf = 100.0
    true_mean = 15.0
    width = 3.0  # Hz

    # Generate signal
    iq = generate_gaussian_doppler_signal(
        n_pulses=n_pulses,
        prf=prf,
        mean=true_mean,
        width=width,
        seed=42,
    )

    # Estimate width
    est_width = compute_pulse_pair_width(iq, prf=prf)
    assert np.all(est_width >= 0.0)
    np.testing.assert_allclose(est_width, width, rtol=0.05)


# test_compute_pulse_pair_snr() {{{1
def test_compute_pulse_pair_snr():
    n_pulses = 1024
    prf = 100.0
    true_mean = 15.0
    width = 3.0  # Hz
    true_snr_db = 20.0  # dB, equivalent to a linear SNR of 100

    # Generate signal
    iq = generate_gaussian_doppler_signal(
        n_pulses=n_pulses,
        prf=prf,
        mean=true_mean,
        width=width,
        snr_db=true_snr_db,
        seed=42,
    )

    est_snr = compute_pulse_pair_snr(iq)
    np.testing.assert_allclose(est_snr, 10 ** (true_snr_db / 10.0), rtol=0.1)

    est_snr_db = compute_pulse_pair_snr_db(iq)
    np.testing.assert_allclose(est_snr_db, true_snr_db, atol=0.5)


# test_compute_pulse_pair_nd() {{{1
def test_compute_pulse_pair_nd():
    n_range = 100
    n_pulses = 1024
    prf = 100.0
    true_mean = 15.0
    width = 3.0  # Hz
    true_snr_db = 20.0  # dB, equivalent to a linear SNR of 100

    # Generate 2D IQ data of shape (n_range, n_pulses)
    iq_2d = np.array(
        [
            generate_gaussian_doppler_signal(
                n_pulses=n_pulses,
                prf=prf,
                mean=true_mean,
                width=width,
                snr_db=true_snr_db,
                seed=42 + i,
            )
            for i in range(n_range)
        ]
    )

    est_variance = compute_pulse_pair_variance(iq_2d, prf=prf)
    assert est_variance.shape == (n_range,)
    np.testing.assert_allclose(est_variance, width**2, rtol=0.3)

    est_snr_db = compute_pulse_pair_snr_db(iq_2d)
    assert est_snr_db.shape == (n_range,)
    np.testing.assert_allclose(est_snr_db, true_snr_db, atol=2.5)


# test_compute_spectrum() {{{1
def test_compute_spectrum():
    x = np.array([1.0, 2.0, 3.0, 4.0])

    freqs, spectrum = compute_spectrum(x)

    np.testing.assert_allclose(freqs, np.fft.fftshift(np.fft.fftfreq(x.size)))
    np.testing.assert_allclose(spectrum, np.fft.fftshift(np.fft.fft(x)))


# test_compute_spectrum_window() {{{1
def test_compute_spectrum_window():
    x = np.ones(8)
    window_values = get_window("hann", x.size, fftbins=True)

    _, spectrum = compute_spectrum(x, window="hann")

    np.testing.assert_allclose(
        spectrum, np.fft.fftshift(np.fft.fft(x * window_values)), atol=1e-12
    )


# test_compute_spectrum_sample_rate() {{{1
def test_compute_spectrum_sample_rate():
    x = np.ones(8)

    freqs, _ = compute_spectrum(x, sample_rate=100.0)

    expected = np.fft.fftshift(np.fft.fftfreq(x.size, d=1.0 / 100.0))
    np.testing.assert_allclose(freqs, expected)


# test_compute_spectrum_n() {{{1
def test_compute_spectrum_n():
    x = np.arange(4.0)

    freqs, spectrum = compute_spectrum(x, n=8)

    assert freqs.size == 8
    np.testing.assert_allclose(spectrum, np.fft.fftshift(np.fft.fft(x, n=8)))

    _, truncated = compute_spectrum(x, n=2)
    np.testing.assert_allclose(truncated, np.fft.fftshift(np.fft.fft(x, n=2)))


# test_compute_spectrum_nd() {{{1
def test_compute_spectrum_nd():
    x = np.arange(12.0).reshape(2, 6)

    freqs, spectrum = compute_spectrum(x)

    expected = np.fft.fftshift(np.fft.fft(x, axis=-1), axes=-1)
    np.testing.assert_allclose(spectrum, expected)
    assert freqs.shape == (6,)


# test_compute_spectrum_window_length_mismatch() {{{1
def test_compute_spectrum_window_length_mismatch():
    with pytest.raises(ValueError, match="window must have length 8"):
        compute_spectrum(np.ones(8), window=np.ones(7))
