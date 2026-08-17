""" Functions to analyze FPGA IQ channel data against filter coefficients. """
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

from radarsig.fpga_io import (
    load_fpga_ram_binary_to_iq,
    load_filter_coeffs_from_binary,
)


# analyze_iq_data() {{{1
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
        Complex array of shape (n_bins, n_samples).
    y : np.ndarray
        Complex array of shape (n_bins, n_samples).
    b : np.ndarray
        IIR filter forward coefficients.
    a : np.ndarray
        IIR filter feedback coefficients.
    threshold : float, optional
        Error threshold for identifying failing range bins (default: 1e-3).

    Returns:
    --------
    dict[str, dict[str, np.ndarray]]
        A dictionary containing results for 'real' and 'imag' parts, each having:
        - 'err_rel': Normalized error difference array
        - 'x_sum': Array of sum(a * x) for each window
        - 'y_sum': Array of sum(b * y) for each window
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

        err_rel = np.zeros_like(y_sum, dtype=float)
        np.divide(y_sum - x_sum, ref, out=err_rel, where=np.abs(ref) > 1e-3)

        failing_bins = np.where(np.any(np.abs(err_rel) > threshold, axis=-1))[0]

        results[part] = {
            'err_rel': err_rel,
            'x_sum': x_sum,
            'y_sum': y_sum,
            'ref': ref,
            'failing_bins': failing_bins
        }
    return results

# _compute_metrics() {{{1
def _compute_metrics(err_rel: np.ndarray) -> dict:
    """ Compute robust statistical metrics for an error array. """
    abs_err_rel = np.abs(err_rel)
    return {
        "mean": float(np.mean(abs_err_rel)),
        "median": float(np.median(abs_err_rel)),
        "p90": float(np.percentile(abs_err_rel, 90)),
        "p99": float(np.percentile(abs_err_rel, 99)),
        "max": float(np.max(abs_err_rel)),
    }

# analyze_iq_pair() {{{1
def analyze_iq_pair(
    i_path: Path | str,
    q_path: Path | str,
    offset_dtype: int = 512,
    n_pulse: int = 14,
    threshold: float = 1e-3,
) -> dict:
    """
    Load and analyze a single pair of FPGA IQ channel files.

    Parameters
    ----------
    i_path : Path | str
        Path to in-phase data file.
    q_path : Path | str
        Path to quadrature data file.
    offset_dtype : int, default=512
        Offset in elements for IQ binary loading.
    n_pulse : int, default=14
        Number of pulses for data reshaping.
    threshold : float, default=1e-3
        Error threshold for identifying failing range bins in analyze_iq_data.

    Returns
    -------
    dict
        Dictionary containing pair_id, real metrics, imag metrics, and full results.
    """
    i_path, q_path = Path(i_path), Path(q_path)

    x, y = load_fpga_ram_binary_to_iq(
        i_path, q_path, offset_dtype=offset_dtype, n_pulse=n_pulse
    )
    b, a = load_filter_coeffs_from_binary(i_path)

    results = analyze_iq_data(x, y, b, a, threshold=threshold)

    pair_id = i_path.stem.removesuffix("_i")

    return {
        "pair_id": pair_id,
        "real": _compute_metrics(results["real"]["err_rel"]),
        "imag": _compute_metrics(results["imag"]["err_rel"]),
        "results": results,
    }
