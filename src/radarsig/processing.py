import numpy as np
from scipy.signal import lfilter
from scipy.fft import fft, fftfreq, fftshift
from typing import Dict, Any, Optional


# _apply_filter() {{{1
def _apply_filter(data: np.ndarray, filt: Filter) -> np.ndarray:
    """Private helper to apply a filter to data."""
    return lfilter(filt.b, filt.a, data)


# _downsample() {{{1
def _downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Private helper to downsample data along the sample dimension."""
    if data.ndim == 1:
        return data[::factor]
    return data[:, ::factor]


# compute_pulse_phase_difference() {{{1
def compute_pulse_phase_difference(iq_data: np.ndarray) -> np.ndarray:
    """
    Computes the phase difference between consecutive pulses for each range bin.

    Parameters:
        iq_data (np.ndarray): Complex IQ data of shape (n_range_bin, n_pulse)

    Returns:
        np.ndarray: Phase differences of shape (n_range_bin, n_pulse - 1) in radians.
    """
    return np.angle(iq_data[:, :-1] * np.conj(iq_data[:, 1:]))


# compute_mean_phase_difference() {{{1
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


# compute_signal_quality_index() {{{1
def compute_signal_quality_index(iq_data: np.ndarray) -> np.ndarray:
    """
    Computes the Signal Quality Index (SQI) as the magnitude of the normalized
    lag-1 autocorrelation coefficient for each range bin.

    Parameters:
        iq_data (np.ndarray): Complex IQ data of shape (n_range_bin, n_pulse)

    Returns:
        np.ndarray: SQI of shape (n_range_bin,) ranging from 0 to 1.
    """
    num = np.abs(np.mean(iq_data[:, :-1] * np.conj(iq_data[:, 1:]), axis=1))
    den = np.abs(np.mean(iq_data * np.conj(iq_data), axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den == 0, 0.0, num / den)


# compute_fft() {{{1
def compute_fft(
    x: np.ndarray,
    fft_len: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the shifted FFT spectrum of a complex signal
    (1D or 2D) using a Hamming window.

    Parameters:
        x (np.ndarray): Complex input signal of shape (n_samples,) or (n_bins, n_samples).
        fft_len (int, optional): FFT length. Defaults to x.shape[-1].

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - freqs: Shifted frequency array of shape (fft_len,)
            - spectrum: Shifted FFT spectrum
    """
    if fft_len is None:
        fft_len = x.shape[-1]

    window = np.hamming(x.shape[-1])
    x_win = x * window

    x_fft = fftshift(fft(x_win, n=fft_len, axis=-1), axes=-1)
    freqs = fftshift(fftfreq(fft_len))

    return freqs, x_fft
