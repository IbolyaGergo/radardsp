import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import get_window

from radarsig.plot_utils import (
    plot_filter_magnitude_response,
    plot_magnitude_spectrum,
    plot_magnitude_spectrum_ratio,
)

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
    assert freqs[0] == -0.5
    assert freqs[-1] == 0.5 - 1 / n_samples
    assert freqs[np.argmax(magnitude)] == bin_index / n_samples
    assert ax.get_xlabel() == "Normalized frequency [cycles/sample]"
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
    np.testing.assert_allclose(magnitude, expected, atol=1e-12)


# test_plot_magnitude_spectrum_db_scale() {{{1
def test_plot_magnitude_spectrum_db_scale():
    signal = np.array([10.0, 0.0, 0.0, 0.0])

    _, ax = plot_magnitude_spectrum(signal, scale="dB")

    _, magnitude = ax.lines[0].get_xydata().T
    expected = 20 * np.log10(np.maximum(np.abs(np.fft.fftshift(np.fft.fft(signal))), 1e-12))
    np.testing.assert_allclose(magnitude, expected)
    assert ax.get_ylabel() == "Magnitude [dB]"


# test_plot_magnitude_spectrum_pad_to() {{{1
def test_plot_magnitude_spectrum_pad_to():
    b = np.array([0.25, 0.5, 0.25])
    pad_to = 64

    _, ax = plot_magnitude_spectrum(b, pad_to=pad_to)

    freqs, magnitude = ax.lines[0].get_xydata().T
    expected = np.abs(np.fft.fftshift(np.fft.fft(b, n=pad_to)))
    np.testing.assert_allclose(magnitude, expected)
    assert freqs[0] == -0.5
    assert freqs[-1] == 0.5 - 1 / pad_to


# test_plot_magnitude_spectrum_pad_to_too_small() {{{1
def test_plot_magnitude_spectrum_pad_to_too_small():
    with pytest.raises(ValueError, match="pad_to must be"):
        plot_magnitude_spectrum(np.ones(8), pad_to=4)


# test_plot_magnitude_spectrum_window() {{{1
def test_plot_magnitude_spectrum_window():
    n_samples = 16
    signal = np.ones(n_samples)

    _, ax = plot_magnitude_spectrum(signal, window="hann")

    _, magnitude = ax.lines[0].get_xydata().T
    window_values = get_window("hann", n_samples, fftbins=True)
    expected = np.abs(np.fft.fftshift(np.fft.fft(signal * window_values)))
    np.testing.assert_allclose(magnitude, expected, atol=1e-12)


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


# test_plot_magnitude_spectrum_ratio_scaling() {{{1
def test_plot_magnitude_spectrum_ratio_scaling():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=32) + 1j * rng.normal(size=32)
    test = 2.0 * reference

    _, ax = plot_magnitude_spectrum_ratio(reference, test)

    _, ratio = ax.lines[0].get_xydata().T
    np.testing.assert_allclose(ratio, 2.0)
    assert ax.get_xlabel() == "Normalized frequency [cycles/sample]"
    assert ax.get_ylabel() == "Magnitude ratio"


# test_plot_magnitude_spectrum_ratio_identity() {{{1
def test_plot_magnitude_spectrum_ratio_identity():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=32) + 1j * rng.normal(size=32)

    _, ax = plot_magnitude_spectrum_ratio(reference, reference)

    _, ratio = ax.lines[0].get_xydata().T
    np.testing.assert_allclose(ratio, 1.0)


# test_plot_magnitude_spectrum_ratio_window_and_pad_to() {{{1
def test_plot_magnitude_spectrum_ratio_window_and_pad_to():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=20) + 1j * rng.normal(size=20)
    test = rng.normal(size=20) + 1j * rng.normal(size=20)
    pad_to = 64
    window_values = get_window("hann", reference.size, fftbins=True)

    _, ax = plot_magnitude_spectrum_ratio(
        reference, test, scale="dB", window="hann", pad_to=pad_to
    )

    _, ratio_db = ax.lines[0].get_xydata().T
    reference_mag = np.abs(
        np.fft.fftshift(np.fft.fft(reference * window_values, n=pad_to))
    )
    test_mag = np.abs(
        np.fft.fftshift(np.fft.fft(test * window_values, n=pad_to))
    )
    np.testing.assert_allclose(ratio_db, 20 * np.log10(test_mag / reference_mag))


# test_plot_magnitude_spectrum_ratio_sample_rate() {{{1
def test_plot_magnitude_spectrum_ratio_sample_rate():
    reference = np.random.default_rng(3).normal(size=16)

    _, ax = plot_magnitude_spectrum_ratio(reference, reference, sample_rate=100.0)

    assert ax.get_xlabel() == "Frequency [Hz]"


# test_plot_magnitude_spectrum_ratio_shape_mismatch() {{{1
def test_plot_magnitude_spectrum_ratio_shape_mismatch():
    with pytest.raises(ValueError, match="same shape unless pad_to"):
        plot_magnitude_spectrum_ratio(np.ones(4), np.ones(5))

    with pytest.raises(ValueError, match="both signal lengths"):
        plot_magnitude_spectrum_ratio(np.ones(4), np.ones(5), pad_to=4)


# test_plot_magnitude_spectrum_ratio_unequal_lengths_with_pad_to() {{{1
def test_plot_magnitude_spectrum_ratio_unequal_lengths_with_pad_to():
    h = np.array([1.0, 0.5])
    x = np.random.default_rng(4).normal(size=20)
    y = np.convolve(x, h, mode="full")
    pad_to = y.size

    _, ax = plot_magnitude_spectrum_ratio(x, y, scale="dB", pad_to=pad_to)

    _, ratio_db = ax.lines[0].get_xydata().T
    expected_db = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(h, n=pad_to))))
    np.testing.assert_allclose(ratio_db, expected_db)


# test_plot_magnitude_spectrum_ratio_linear_scale() {{{1
def test_plot_magnitude_spectrum_ratio_linear_scale():
    rng = np.random.default_rng(5)
    reference = rng.normal(size=32) + 1j * rng.normal(size=32)
    test = 2.0 * reference

    _, ax = plot_magnitude_spectrum_ratio(reference, test, scale="linear")

    _, ratio = ax.lines[0].get_xydata().T
    np.testing.assert_allclose(ratio, 2.0)
    assert ax.get_ylabel() == "Magnitude ratio"


# test_plot_magnitude_spectrum_ratio_invalid_scale() {{{1
def test_plot_magnitude_spectrum_ratio_invalid_scale():
    with pytest.raises(ValueError, match="scale must be"):
        plot_magnitude_spectrum_ratio(np.ones(8), np.ones(8), scale="wat")


# test_plot_magnitude_spectrum_ratio_all_zero_bins() {{{1
def test_plot_magnitude_spectrum_ratio_all_zero_bins():
    _, ax = plot_magnitude_spectrum_ratio(np.zeros(8), np.zeros(8))

    _, ratio = ax.lines[0].get_xydata().T
    assert np.isnan(ratio).all()


# test_plot_magnitude_spectrum_invalid_input() {{{1
def test_plot_magnitude_spectrum_invalid_input():
    with pytest.raises(ValueError, match="must be 1D"):
        plot_magnitude_spectrum(np.zeros((2, 4)))

    with pytest.raises(ValueError, match="scale must be"):
        plot_magnitude_spectrum(np.zeros(4), scale="wat")


# test_plot_filter_magnitude_response() {{{1
def test_plot_filter_magnitude_response():
    fig, ax = plot_filter_magnitude_response([1.0], [1.0], worN=8)

    freqs, magnitude = ax.lines[0].get_xydata().T
    np.testing.assert_allclose(freqs, np.linspace(0.0, 0.5, 8, endpoint=False))
    np.testing.assert_allclose(magnitude, np.ones(8))
    assert ax.get_xlabel() == "Normalized frequency [cycles/sample]"
    assert ax.get_ylabel() == "Magnitude"
    assert fig is ax.get_figure()


# test_plot_filter_magnitude_response_db_scale() {{{1
def test_plot_filter_magnitude_response_db_scale():
    _, ax = plot_filter_magnitude_response([0.0], [1.0], scale="dB")

    _, magnitude = ax.lines[0].get_xydata().T
    np.testing.assert_allclose(magnitude, 20 * np.log10(1e-12))
    assert ax.get_ylabel() == "Magnitude [dB]"


# test_plot_filter_magnitude_response_sample_rate() {{{1
def test_plot_filter_magnitude_response_sample_rate():
    _, ax = plot_filter_magnitude_response([1.0], [1.0], worN=8, sample_rate=100.0)

    freqs, _ = ax.lines[0].get_xydata().T
    np.testing.assert_allclose(freqs, np.linspace(0.0, 50.0, 8, endpoint=False))
    assert ax.get_xlabel() == "Frequency [Hz]"


# test_plot_filter_magnitude_response_label() {{{1
def test_plot_filter_magnitude_response_label():
    _, ax = plot_filter_magnitude_response([1.0], [1.0], label="Ideal")

    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_texts()[0].get_text() == "Ideal"


# test_plot_filter_magnitude_response_invalid_scale() {{{1
def test_plot_filter_magnitude_response_invalid_scale():
    with pytest.raises(ValueError, match="scale must be"):
        plot_filter_magnitude_response([1.0], [1.0], scale="wat")
