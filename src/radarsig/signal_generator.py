import numpy as np
from scipy.fft import fft, ifft, fftfreq


# generate_gaussian_doppler_signal() {{{1
def generate_gaussian_doppler_signal(
    n_pulses: int,
    prf: float,
    fd: float,
    true_freq_variance: float,
    wavelength: float = 1.0,
    snr: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a complex time-series (pulse returns) whose power spectral density
    is a Gaussian centered at Doppler frequency fd, with specified variance.

    Parameters:
    - n_pulses: Number of pulse samples
    - prf: Pulse Repetition Frequency (Hz)
    - fd: Mean Doppler frequency (Hz)
    - true_freq_variance: Frequency variance (Hz^2) of the Gaussian spectrum
    - wavelength: Radar wavelength (m), defaults to 1.0
    - snr: Signal To Noise ratio defined as S_0 / N, where S_0 is the signal power and N is the
      noise power
    - seed: Random seed for reproducibility (optional)

    Returns:
    - iq: Complex time-series array of shape (n_pulses,)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / prf
    t = np.arange(n_pulses) * dt
    freqs = fftfreq(n_pulses, d=dt)

    # Desired Gaussian PSD centered at 0 Hz
    psd = np.exp(-(freqs**2) / (2 * true_freq_variance))
    s0_power = np.sum(psd)

    if snr is not None:
        # snr = S_0 / N => N = S_0 / snr
        noise_power = s0_power / snr
        # divide noise across all frequency bins
        noise_floor = noise_power / n_pulses
        psd_total = psd + noise_floor
    else:
        psd_total = psd

    # Generate complex white noise in frequency domain and shape with PSD
    noise_freq = (np.random.randn(n_pulses) + 1j * np.random.randn(n_pulses)) / np.sqrt(2)
    # psd represents power, so we take the sqrt to get the amplitude of the spectrum
    shaped_freq = noise_freq * np.sqrt(psd_total)

    # IFFT to time domain
    iq_base = ifft(shaped_freq) * np.sqrt(n_pulses)

    # Apply Doppler shift (moving target)
    iq = iq_base * np.exp(1j * 2 * np.pi * fd * t)

    return iq


# compute_pulse_pair_variance() {{{1
def compute_pulse_pair_variance(iq: np.ndarray, wavelength: float = 1.0, prf: float = 1.0) -> float:
    """
    Estimate velocity variance from complex pulse returns using lag-1 and lag-2
    autocorrelation pulse-pair method.

    Parameters:
    - iq: Complex time-series array of pulse returns
    - wavelength: Radar wavelength (m)
    - prf: Pulse Repetition Frequency (Hz)

    Returns:
    - estimated_variance: Estimated frequency variance
    """

    # Autocorrelation at lag 1 and lag 2
    r1 = np.mean(iq[:-1] * np.conj(iq[1:]))
    r2 = np.mean(iq[:-2] * np.conj(iq[2:]))

    coeff = prf**2 / (6.0 * np.pi**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where(r2 == 0, 0.0, coeff * np.log(np.abs(r1 / r2)))

    return float(variance)


# compute_pulse_pair_snr() {{{1
def compute_pulse_pair_snr(iq: np.ndarray) -> float:
    # Autocorrelation at lag 0, 1 and 2
    r0 = np.mean(iq * np.conj(iq))
    r1 = np.mean(iq[:-1] * np.conj(iq[1:]))
    r2 = np.mean(iq[:-2] * np.conj(iq[2:]))

    nom = np.power(np.abs(r1), 4.0 / 3.0)
    denom = r0 * np.power(np.abs(r2), 1.0 / 3.0) - np.power(np.abs(r1), 4.0 / 3.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(denom == 0, 0.0, np.abs(nom / denom))

    return snr
