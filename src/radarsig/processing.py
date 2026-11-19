import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import freqz, get_window, lfilter


# _apply_filter() {{{1
def _apply_filter(data: np.ndarray, filt: Filter) -> np.ndarray:
    """Private helper to apply a filter to data."""
    return lfilter(filt.b, filt.a, data)


# _downsample() {{{1
def _downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Private helper to downsample data along the sample dimension (axis=-1)."""
    return data[..., ::factor]


# compute_pulse_phase_difference() {{{1
def compute_pulse_phase_difference(iq_data: np.ndarray) -> np.ndarray:
    """
    Computes the phase difference between consecutive pulses for each range bin.

    Parameters:
        iq_data (np.ndarray): Complex IQ data of shape (..., n_pulse)

    Returns:
        np.ndarray: Phase differences of shape (..., n_pulse - 1) in radians.
    """
    return np.angle(iq_data[..., :-1] * np.conj(iq_data[..., 1:]))


# compute_pulse_pair_mean() {{{1
def compute_pulse_pair_mean(iq_data: np.ndarray, prf: float) -> float | np.ndarray:
    """
    Estimate frequency mean from complex pulse returns using lag-1 autocorrelation pulse-pair
    method.

    Parameters:
        iq_data (np.ndarray): Complex IQ data of shape (..., n_pulse)
        prf: Pulse Repetition Frequency (Hz)

    Returns:
        estimated_mean: Estimated frequency mean
    """
    coeff = prf / (2.0 * np.pi)
    r1 = np.mean(iq_data[..., :-1] * np.conj(iq_data[..., 1:]), axis=-1)
    return coeff * np.angle(r1)


# compute_signal_quality_index() {{{1
def compute_signal_quality_index(iq_data: np.ndarray) -> np.ndarray:
    """
    Computes the Signal Quality Index (SQI) as the magnitude of the normalized
    lag-1 autocorrelation coefficient for each range bin.

    Parameters:
        iq_data (np.ndarray): Complex IQ data of shape (..., n_pulse)

    Returns:
        np.ndarray: SQI of shape (...,) ranging from 0 to 1.
    """
    num = np.abs(np.mean(iq_data[..., :-1] * np.conj(iq_data[..., 1:]), axis=-1))
    den = np.abs(np.mean(iq_data * np.conj(iq_data), axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den == 0, 0.0, num / den)


# compute_pulse_pair_variance() {{{1
def compute_pulse_pair_variance(iq: np.ndarray, prf: float) -> float | np.ndarray:
    """
    Estimate frequency variance from complex pulse returns using lag-1 and lag-2
    autocorrelation pulse-pair method.

    Parameters:
    - iq: Complex time-series array of pulse returns
    - prf: Pulse Repetition Frequency (Hz)

    Returns:
    - estimated_variance: Estimated frequency variance
    """

    # Autocorrelation at lag 1 and lag 2
    r1 = np.mean(iq[..., :-1] * np.conj(iq[..., 1:]), axis=-1)
    r2 = np.mean(iq[..., :-2] * np.conj(iq[..., 2:]), axis=-1)

    coeff = prf**2 / (6.0 * np.pi**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where(r2 == 0, 0.0, coeff * np.log(np.abs(r1 / r2)))

    return variance


# compute_pulse_pair_width() {{{1
def compute_pulse_pair_width(iq: np.ndarray, prf: float) -> float | np.ndarray:
    """
    Estimate spectral width (standard deviation, Hz) from complex pulse returns.

    Thin wrapper around compute_pulse_pair_variance(): the width is the square
    root of the estimated variance. The estimate is clipped at zero because the
    log-ratio variance estimator can be slightly negative under noise.

    Note: sqrt of a noisy variance estimate is biased for the standard deviation.

    Parameters:
    - iq: Complex time-series array of pulse returns
    - prf: Pulse Repetition Frequency (Hz)

    Returns:
    - estimated_width: Estimated frequency standard deviation (Hz)
    """
    variance = compute_pulse_pair_variance(iq, prf=prf)
    return np.sqrt(np.maximum(variance, 0.0))


# compute_pulse_pair_snr() {{{1
def compute_pulse_pair_snr(iq: np.ndarray) -> float | np.ndarray:
    # Autocorrelation at lag 0, 1 and 2
    r0 = np.mean(iq * np.conj(iq), axis=-1)
    r1 = np.mean(iq[..., :-1] * np.conj(iq[..., 1:]), axis=-1)
    r2 = np.mean(iq[..., :-2] * np.conj(iq[..., 2:]), axis=-1)

    nom = np.power(np.abs(r1), 4.0 / 3.0)
    denom = r0 * np.power(np.abs(r2), 1.0 / 3.0) - np.power(np.abs(r1), 4.0 / 3.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(denom == 0, 0.0, np.abs(nom / denom))

    return snr


# compute_pulse_pair_snr_db() {{{1
def compute_pulse_pair_snr_db(iq: np.ndarray) -> float | np.ndarray:
    """
    Estimate SNR in dB from complex pulse returns.

    Thin wrapper around compute_pulse_pair_snr(): the linear ratio is floored at
    1e-9 to avoid log(0) and then converted with 10*log10().

    Parameters:
    - iq: Complex time-series array of pulse returns

    Returns:
    - estimated_snr_db: Estimated SNR (dB)
    """
    snr = compute_pulse_pair_snr(iq)
    return 10.0 * np.log10(np.maximum(snr, 1e-9))


# _resolve_window() {{{1
def _resolve_window(
    window: str | tuple | np.ndarray | None, n_samples: int
) -> np.ndarray | None:
    """Resolve a window specification to an array, or None for rectangular."""
    if window is None:
        return None
    if isinstance(window, np.ndarray):
        if window.ndim != 1 or window.size != n_samples:
            raise ValueError(f"window must have length {n_samples}, got {window.shape}")
        return window
    return get_window(window, n_samples, fftbins=True)


# compute_spectrum() {{{1
def compute_spectrum(
    x: np.ndarray,
    *,
    n: int | None = None,
    window: str | tuple | np.ndarray | None = None,
    sample_rate: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the shifted FFT spectrum of a signal.

    Parameters:
        x: Complex or real signal of shape (..., n_samples).
        n: Length of the transformed axis, following numpy/scipy FFT semantics:
           shorter than the signal truncates, longer zero-pads. Defaults to the
           signal length.
        window: Window applied before the FFT. None means rectangular. Accepts
                any window name/tuple understood by scipy.signal.get_window, or
                a 1D array matching the signal length.
        sample_rate: Sampling rate in Hz. If None, frequencies are normalized
                     (cycles/sample); otherwise they are in Hz.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - freqs: Shifted frequency array of shape (n,)
            - spectrum: Shifted complex FFT spectrum
    """
    x = np.asarray(x)
    n_samples = x.shape[-1]
    if n is None:
        n = n_samples

    window_values = _resolve_window(window, n_samples)
    if window_values is not None:
        x = x * window_values

    spectrum = fftshift(fft(x, n=n, axis=-1), axes=-1)
    if sample_rate is None:
        freqs = fftshift(fftfreq(n))
    else:
        freqs = fftshift(fftfreq(n, d=1.0 / sample_rate))

    return freqs, spectrum


# compute_filter_response() {{{1
def compute_filter_response(
    b: np.ndarray,
    a: np.ndarray,
    *,
    worN: int = 512,
    sample_rate: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the complex frequency response of a digital filter.

    Parameters:
        b: Numerator coefficients.
        a: Denominator coefficients.
        worN: Number of frequencies to evaluate, following scipy.signal.freqz
              semantics. Frequencies span 0 up to (but excluding) Nyquist.
        sample_rate: Sampling rate in Hz. If None, frequencies are normalized
                     (cycles/sample); otherwise they are in Hz.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - freqs: Frequency array of shape (worN,)
            - response: Complex frequency response
    """
    w, h = freqz(b, a, worN=worN)
    if sample_rate is None:
        freqs = w / (2.0 * np.pi)
    else:
        freqs = w * sample_rate / (2.0 * np.pi)

    return freqs, h
