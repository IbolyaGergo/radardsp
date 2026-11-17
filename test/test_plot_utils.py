import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import get_window

from radarsig.plot_utils import plot_magnitude_spectrum

plt.switch_backend("Agg")


# close_figures() {{{1
@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# _tone() {{{1
def _tone(n_samples: int, bin_index: int) -> np.ndarray:
    n = np.arange(n_samples)
    return np.exp(1j * 2 * np.pi * bin_index * n / n_samples)


# test_plot_magnitude_spectrum_peak_and_default_range() {{{1
def test_plot_magnitude_spectrum_peak_and_default_range():
    n_samples = 64
    bin_index = 7
    signal = _tone(n_samples, bin_index)

    fig, ax = plot_magnitude_spectrum(signal)

    freqs, magnitude = ax.lines[0].get_xydata().T
    assert freqs[0] == -n_samples / 2
    assert freqs[-1] == n_samples / 2 - 1
    assert freqs[np.argmax(magnitude)] == bin_index
    assert ax.get_xlabel() == "Frequency [bins]"
    assert ax.get_ylabel() == "Magnitude"
    assert fig is ax.get_figure()


# test_plot_magnitude_spectrum_sample_rate() {{{1
def test_plot_magnitude_spectrum_sample_rate():
    n_samples = 64
    bin_index = 7
    sample_rate = 100.0
    signal = _tone(n_samples, bin_index)

    _, ax = plot_magnitude_spectrum(signal, sample_rate=sample_rate)

    freqs, magnitude = ax.lines[0].get_xydata().T
    expected_peak = bin_index * sample_rate / n_samples
    np.testing.assert_allclose(freqs[np.argmax(magnitude)], expected_peak)
    assert ax.get_xlabel() == "Frequency [Hz]"


# test_plot_magnitude_spectrum_linear_magnitude() {{{1
def test_plot_magnitude_spectrum_linear_magnitude():
    signal = np.array([1.0, -1.0, 1.0, -1.0])

    _, ax = plot_magnitude_spectrum(signal)

    _, magnitude = ax.lines[0].get_xydata().T
    expected = np.abs(np.fft.fftshift(np.fft.fft(signal)))
    np.testing.assert_allclose(magnitude, expected)


# test_plot_magnitude_spectrum_db_scale() {{{1
def test_plot_magnitude_spectrum_db_scale():
    signal = np.array([10.0, 0.0, 0.0, 0.0])

    _, ax = plot_magnitude_spectrum(signal, scale="dB")

    _, magnitude = ax.lines[0].get_xydata().T
    expected = 20 * np.log10(np.maximum(np.abs(np.fft.fftshift(np.fft.fft(signal))), 1e-12))
    np.testing.assert_allclose(magnitude, expected)
    assert ax.get_ylabel() == "Magnitude [dB]"


# test_plot_magnitude_spectrum_window() {{{1
def test_plot_magnitude_spectrum_window():
    n_samples = 16
    signal = np.ones(n_samples)

    _, ax = plot_magnitude_spectrum(signal, window="hann")

    _, magnitude = ax.lines[0].get_xydata().T
    window_values = get_window("hann", n_samples, fftbins=True)
    expected = np.abs(np.fft.fftshift(np.fft.fft(signal * window_values)))
    np.testing.assert_allclose(magnitude, expected)


# test_plot_magnitude_spectrum_window_array_length_mismatch() {{{1
def test_plot_magnitude_spectrum_window_array_length_mismatch():
    with pytest.raises(ValueError, match="window must have length 8"):
        plot_magnitude_spectrum(np.ones(8), window=np.ones(7))


# test_plot_magnitude_spectrum_reuses_axes() {{{1
def test_plot_magnitude_spectrum_reuses_axes():
    fig, ax = plt.subplots()

    returned_fig, returned_ax = plot_magnitude_spectrum(np.ones(8), ax=ax)

    assert returned_fig is fig
    assert returned_ax is ax
    assert len(ax.lines) == 1


# test_plot_magnitude_spectrum_invalid_input() {{{1
def test_plot_magnitude_spectrum_invalid_input():
    with pytest.raises(ValueError, match="must be 1D"):
        plot_magnitude_spectrum(np.zeros((2, 4)))

    with pytest.raises(ValueError, match="scale must be"):
        plot_magnitude_spectrum(np.zeros(4), scale="wat")
