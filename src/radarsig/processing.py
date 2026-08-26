import numpy as np
from scipy.signal import lfilter
from typing import Dict, Any, Optional


def _apply_filter(data: np.ndarray, filt: Filter) -> np.ndarray:
    """Private helper to apply a filter to data."""
    return lfilter(filt.b, filt.a, data)


def _downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Private helper to downsample data along the sample dimension."""
    if data.ndim == 1:
        return data[::factor]
    return data[:, ::factor]


def compute_pulse_phase_difference(iq_data: np.ndarray) -> np.ndarray:
    """
    Computes the phase difference between consecutive pulses for each range bin.

    Parameters:
        iq_data (np.ndarray): Complex IQ data of shape (n_range_bin, n_pulse)

    Returns:
        np.ndarray: Phase differences of shape (n_range_bin, n_pulse - 1) in radians.
    """
    return np.angle(iq_data[:, :-1] * np.conj(iq_data[:, 1:]))


def compute_mean_phase_difference(iq_data: np.ndarray) -> np.ndarray:
    """
    Computes the mean phase difference across pulses for each range bin
    using the argument of the lag-1 autocorrelation.

    Parameters:
        iq_data (np.ndarray): Complex IQ data of shape (n_range_bin, n_pulse)

    Returns:
        np.ndarray: Mean phase difference of shape (n_range_bin,) in radians.
    """
    r1 = np.mean(iq_data[:, :-1] * np.conj(iq_data[:, 1:]), axis=1)
    return np.angle(r1)
