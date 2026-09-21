import numpy as np
from scipy.fft import fft, ifft, fftfreq


def generate_gaussian_doppler_signal(
    n_pulses: int,
    prf: float,
    fd: float,
    true_variance: float,
    wavelength: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a complex time-series (pulse returns) whose power spectral density
    is a Gaussian centered at Doppler frequency fd, with specified variance.

    Parameters:
    - n_pulses: Number of pulse samples
    - prf: Pulse Repetition Frequency (Hz)
    - fd: Mean Doppler frequency (Hz)
    - true_variance: Frequency variance (Hz^2) of the Gaussian spectrum
    - wavelength: Radar wavelength (m), defaults to 1.0
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
    psd = np.exp(-(freqs**2) / (2 * true_variance))

    # Generate complex white noise in frequency domain and shape with PSD
    noise_freq = (np.random.randn(n_pulses) + 1j * np.random.randn(n_pulses)) / np.sqrt(2)
    # psd represents power, so we take the sqrt to get the amplitude of the spectrum
    shaped_freq = noise_freq * np.sqrt(psd)

    # IFFT to time domain
    iq_base = ifft(shaped_freq) * np.sqrt(n_pulses)

    # Apply Doppler shift (moving target)
    iq = iq_base * np.exp(1j * 2 * np.pi * fd * t)

    return freqs, iq


def compute_pulse_pair_variance(iq: np.ndarray, wavelength: float = 1.0, prf: float = 1.0) -> float:
    """
    Estimate velocity variance from complex pulse returns using lag-1 and lag-2
    autocorrelation pulse-pair method.

    Parameters:
    - iq: Complex time-series array of pulse returns
    - wavelength: Radar wavelength (m)
    - prf: Pulse Repetition Frequency (Hz)

    Returns:
    - estimated_variance: Estimated velocity variance (m^2 / s^2)
    """
    tau = 1.0 / prf

    # Autocorrelation at lag 1 and lag 2
    r1 = np.mean(iq[:-1] * np.conj(iq[1:]))
    r2 = np.mean(iq[:-2] * np.conj(iq[2:]))

    coeff = 2.0 / (3.0 * tau**2) * (wavelength / (4.0 * np.pi)) ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where(r2 == 0, 0.0, coeff * np.log(np.abs(r1 / r2)))

    return float(variance)
