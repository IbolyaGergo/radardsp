import numpy as np

from matplotlib import pyplot as plt
from typing import Callable, Any


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
