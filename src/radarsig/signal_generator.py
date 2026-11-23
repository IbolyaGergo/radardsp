import numpy as np
from scipy.fft import fftfreq, ifft


# _autocorr_threshold() {{{1
def _autocorr_threshold(n_pulses: int, prf: float, width: float) -> float:
    """Autocorrelation threshold (dB).

    From Curtis (2018):
    It is capped at 25 dB so that narrow spectra do not make the required
    series length grow without bound.
    """
    threshold = min(25.0, 10 + 5 / np.log(10) * (2 * np.pi * width * (n_pulses - 1) / prf) ** 2)
    return threshold


# _first_lag_below_threshold() {{{1
def _first_lag_below_threshold(prf: float, width: float, threshold: float) -> int:
    """Smallest lag (in pulses) at which the Gaussian autocorrelation has dropped below
    `threshold` (dB).

    For a Gaussian spectrum of standard deviation `width` (Hz) the autocorrelation is
    rho(tau) = exp(-2 * (pi * width * tau)**2), so setting 10 * log10(rho) = -threshold and
    converting to a lag (tau * prf) gives the expression below.
    """
    # int(...) so that callers get a plain integer length rather than a numpy scalar.
    return int(np.ceil(prf / (2 * np.pi * width) * np.sqrt(np.log(10) / 5 * threshold)))


# _sim_length() {{{1
def _sim_length(
    n_pulses: int, prf: float, width: float, autocorr_threshold: float | None = None
) -> int:
    """Number of samples to simulate for `n_pulses` pulses, so that the autocorrelation of the
    resulting series is not distorted by the series being circularly wrapped.

    Based on the article
    Curtis, C. D., 2018: Weather radar time series simulation: Improving accuracy and performance.
    J. Atmos. Oceanic Technol., 35, 2169–2187, https://doi.org/10.1175/JTECH-D-17-0215.1."""

    threshold = (
        autocorr_threshold
        if autocorr_threshold is not None
        else _autocorr_threshold(n_pulses, prf, width)
    )
    # Lag beyond which the autocorrelation is considered negligible.
    lag = _first_lag_below_threshold(prf, width, threshold)

    # The series is generated spectrally and is therefore periodic: keep enough samples that
    # the retained pulses are at least `lag` samples away from their wrapped copy, and that the
    # autocorrelation at lag `lag` still fits into the simulated window.
    return int(max(2 * lag + 1, n_pulses + lag))


# _gaussian_doppler_samples() {{{1
def _gaussian_doppler_samples(
    n_samples: int,
    prf: float,
    mean: float,
    width: float,
    snr_db: float | None = None,
) -> np.ndarray:
    """Complex Gaussian-spectrum time-series of `n_samples` samples.

    The series is generated in the frequency domain and is therefore periodic with period
    `n_samples`; deciding how much of it is kept is left to the caller (see
    generate_gaussian_doppler_signal).
    """
    dt = 1.0 / prf
    t = np.arange(n_samples) * dt
    freqs = fftfreq(n_samples, d=dt)

    # Desired Gaussian PSD centered at 0 Hz
    psd = np.exp(-(freqs**2) / (2 * width**2))
    s0_power = np.sum(psd)

    if snr_db is not None:
        # snr_db = 10 * log10(S_0 / N) => N = S_0 / 10**(snr_db / 10)
        snr = 10 ** (snr_db / 10.0)
        noise_power = s0_power / snr
        # divide noise across all frequency bins
        noise_floor = noise_power / n_samples
        psd_total = psd + noise_floor
    else:
        psd_total = psd

    # Generate complex white noise in frequency domain and shape with PSD
    noise_freq = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
    # psd represents power, so we take the sqrt to get the amplitude of the spectrum
    shaped_freq = noise_freq * np.sqrt(psd_total)

    # IFFT to time domain
    iq_base = ifft(shaped_freq) * np.sqrt(n_samples)

    # Apply Doppler shift (moving target)
    return iq_base * np.exp(1j * 2 * np.pi * mean * t)


# generate_gaussian_doppler_signal() {{{1
def generate_gaussian_doppler_signal(
    n_pulses: int,
    prf: float,
    mean: float,
    width: float,
    snr_db: float | None = None,
    seed: int | None = None,
    avoid_wraparound: bool = True,
    autocorr_threshold: float | None = None,
) -> np.ndarray:
    """
    Generate a complex time-series (pulse returns) whose power spectral density
    is a Gaussian centered at Doppler frequency mean, with specified width.

    Parameters:
    - n_pulses: Number of pulse samples
    - prf: Pulse Repetition Frequency (Hz)
    - mean: Mean Doppler frequency (Hz)
    - width: Spectral width (standard deviation, Hz) of the Gaussian spectrum.
      Equivalently, the frequency variance is width**2 (Hz^2).
    - snr_db: Signal-to-noise ratio in dB, defined as 10*log10(S_0 / N), where S_0 is the
      signal power and N is the noise power
    - seed: Random seed for reproducibility (optional)
    - avoid_wraparound: Simulate _sim_length(...) samples and return only the first
      n_pulses of them, so that the retained pulses are not correlated with their own
      wrapped copy. This costs _sim_length(...) / n_pulses more work. With False the
      series is simulated at exactly n_pulses samples and the wrap-around bias of
      Curtis (2018) is present. Note that the benefit is a property of the simulated
      process and cannot be measured from the returned samples alone.
    - autocorr_threshold: Autocorrelation threshold (dB) used by _sim_length(...); only
      relevant when avoid_wraparound is True. None selects the article's formula.

    Returns:
    - iq: Complex time-series array of shape (n_pulses,)
    """
    if seed is not None:
        np.random.seed(seed)

    # Simulate longer than needed so that the periodic nature of the spectrally generated
    # series does not distort the autocorrelation of the returned pulses.
    n_samples = (
        _sim_length(n_pulses, prf, width, autocorr_threshold) if avoid_wraparound else n_pulses
    )

    return _gaussian_doppler_samples(n_samples, prf, mean, width, snr_db)[:n_pulses]
