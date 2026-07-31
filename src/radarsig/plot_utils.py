import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import freqz
from typing import Callable, Any


# make_iq_plotter() {{{1
def make_iq_plotter(data: np.ndarray, use_global_limits: bool = False) -> Callable[[plt.Axes, np.ndarray, Any], None]:
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
            ax.plot(i, q, '-o', alpha=0.5)

        if use_global_limits:
            limit = global_max
        else:
            limit = np.max(np.abs(data_slice)) * 1.1
            if np.isclose(limit, 0.0):
                limit = 1.0

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')

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
def plot_filter_response(b, a, worN: int = 512, axes=None, label: str | None =
                         None) -> tuple["fig", "axes"]:
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
