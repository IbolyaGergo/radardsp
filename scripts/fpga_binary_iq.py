""" Script to load tmp/iq/*.data to complex np.ndarray. """
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from radarsig.fpga_io import (
    load_fpga_ram_binary_to_iq,
    load_filter_coeffs_from_binary,
)


def analyze_iq_data(
    x: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    threshold: float = 1e-3
) -> dict[str, dict[str, np.ndarray]]:
    """
    Analyze FPGA IQ channel data against filter coefficients using sliding window convolution.

    Parameters:
    -----------
    x : np.ndarray
        Complex array of shape (n_bins, n_samples) for channel X.
    y : np.ndarray
        Complex array of shape (n_bins, n_samples) for channel Y.
    b : np.ndarray
        FIR filter forward coefficients.
    a : np.ndarray
        IIR filter feedback coefficients.
    threshold : float, optional
        Error threshold for identifying failing range bins (default: 1e-3).

    Returns:
    --------
    dict[str, dict[str, np.ndarray]]
        A dictionary containing results for 'real' and 'imag' parts, each having:
        - 'diff': Normalized error difference array
        - 'x_sum': Filtered X signal sum
        - 'y_sum': Filtered Y signal sum
        - 'ref': Reference amplitude max
        - 'failing_bins': Array of range bin indices exceeding the threshold
    """

    n_tap = max([len(b), len(a)])

    results = {}
    for part in ('real', 'imag'):
        x_part = getattr(x, part)
        y_part = getattr(y, part)

        x_windows = sliding_window_view(x_part, window_shape=n_tap, axis=-1)
        y_windows = sliding_window_view(y_part, window_shape=n_tap, axis=-1)

        y_sum = np.sum(a * y_windows, axis=-1)
        x_sum = np.sum(b * x_windows, axis=-1)

        ref = np.max(np.abs(x_sum), axis=-1, keepdims=True)

        diff = np.zeros_like(y_sum, dtype=float)
        np.divide(y_sum - x_sum, ref, out=diff, where=np.abs(ref) > 1e-3)

        failing_bins = np.where(np.any(np.abs(diff) > threshold, axis=-1))[0]

        results[part] = {
            'diff': diff,
            'x_sum': x_sum,
            'y_sum': y_sum,
            'ref': ref,
            'failing_bins': failing_bins
        }
    return results

if __name__ == "__main__":
    x, y = load_fpga_ram_binary_to_iq("tmp/iq/012_i.data", "tmp/iq/012_q.data", offset_dtype=512)
    b, a = load_filter_coeffs_from_binary("tmp/iq/012_i.data")

    res = analyze_iq_data(x, y, b, a)
