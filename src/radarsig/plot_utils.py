import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import freqz, get_window
from typing import Callable, Any, Literal


# make_iq_plotter() {{{1
def make_iq_plotter(
    data: np.ndarray, use_global_limits: bool = False
) -> Callable[[plt.Axes, np.ndarray, Any], None]:
    """
    Creates a callback function for plotting IQ data with fixed global limits.

    Args:
        data: The full dataset to determine global axes limits.
        use_global_limits: If True, uses max(abs(data)) for limits.
                           If False, uses max(abs(data_slice)) dynamically.

    Returns:
        A callable suitable for RangeSlider.plot_callback.
    """
    global_max = np.max(np.abs(data)) * 1.1
    if np.isclose(global_max, 0.0):
        global_max = 1.0

    def callback(ax, data_slice, x):
        for i in range(data_slice.shape[0]):
            row = data_slice[i, :]
            i, q = row.real, row.imag
            ax.plot(i, q, "-o", alpha=0.5)

        if use_global_limits:
            limit = global_max
        else:
            limit = np.max(np.abs(data_slice)) * 1.1
            if np.isclose(limit, 0.0):
                limit = 1.0

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")

    return callback


# _format_freqz_axes() {{{1
def _format_freqz_axes(ax_mag: plt.Axes, ax_phase: plt.Axes):
    """Encapsulates all styling/aesthetics."""
    ax_mag.set_title("Frequency Response Comparison")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.grid(True)
    ax_phase.set_ylabel("Phase [rad]")
    ax_phase.set_xlabel("Frequency [rad/sample]")
    ax_phase.grid(True)


# plot_filter_response() {{{1
def plot_filter_response(
    b, a, worN: int = 512, axes=None, label: str | None = None
) -> tuple["fig", "axes"]:
    if axes is None:
        fig, axes = plt.subplots(2, 1, tight_layout=True, figsize=(8, 6))
        _format_freqz_axes(*axes)

    w, h = freqz(b, a, worN=worN)

    ax_mag, ax_phase = axes
    ax_mag.plot(w, 20 * np.log10(np.abs(h)), label=label)
    ax_phase.plot(w, np.unwrap(np.angle(h)), label=label)

    if label:
        ax_mag.legend()
        ax_phase.legend()

    return axes[0].get_figure(), axes


# plot_magnitude_spectrum() {{{1
def plot_magnitude_spectrum(
    signal: np.ndarray,
    *,
    sample_rate: float | None = None,
    scale: Literal["linear", "dB"] = "linear",
    window: str | tuple | np.ndarray | None = None,
    pad_to: int | None = None,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the magnitude spectrum of a 1D signal.

    Unlike matplotlib's ``Axes.magnitude_spectrum``, this function always
    returns a two-sided, unnormalized spectrum, applies no window by default,
    floors dB values at 1e-12, and returns the (figure, axes) pair.

    Args:
        signal: 1D complex or real time-series.
        sample_rate: Sampling rate in Hz. If None, the x-axis is in FFT bins
                     spanning -N/2 to N/2, where N is the FFT length. If given,
                     the x-axis is in Hz.
        scale: "linear" for magnitude, "dB" for 20*log10(magnitude).
        window: Window applied before the FFT. None means rectangular. Accepts
                any window name/tuple understood by scipy.signal.get_window, or
                a 1D array matching the signal length. No coherent-gain
                normalization is applied.
        pad_to: FFT length. If greater than the signal length, zero-pad before
                the FFT; must be at least the signal length. If None, use the
                signal length.
        ax: Axes to draw on. If None, a new figure and axes are created.
        label: Optional line label, shown in a legend when provided.

    Returns:
        The figure and axes containing the plot.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {signal.shape}")
    if scale not in ("linear", "dB"):
        raise ValueError(f"scale must be 'linear' or 'dB', got {scale!r}")

    n_samples = signal.size
    if pad_to is not None and pad_to < n_samples:
        raise ValueError(f"pad_to must be >= signal length {n_samples}, got {pad_to}")
    n_fft = n_samples if pad_to is None else pad_to

    if window is None:
        window_values = None
    elif isinstance(window, np.ndarray):
        if window.ndim != 1 or window.size != n_samples:
            raise ValueError(f"window must have length {n_samples}, got {window.shape}")
        window_values = window
    else:
        window_values = get_window(window, n_samples, fftbins=True)

    signal_for_fft = signal if window_values is None else signal * window_values
    spectrum = np.fft.fftshift(np.fft.fft(signal_for_fft, n=n_fft))

    if sample_rate is None:
        freqs = np.fft.fftshift(np.fft.fftfreq(n_fft)) * n_fft
        xlabel = "Frequency [bins]"
    else:
        freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sample_rate))
        xlabel = "Frequency [Hz]"

    magnitude = np.abs(spectrum)
    if scale == "dB":
        magnitude = 20 * np.log10(np.maximum(magnitude, 1e-12))
        ylabel = "Magnitude [dB]"
    else:
        ylabel = "Magnitude"

    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        fig = ax.get_figure()

    ax.plot(freqs, magnitude, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if label:
        ax.legend()

    return fig, ax
