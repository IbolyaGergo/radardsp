import numpy as np
from scipy.fft import fft, ifft, fftfreq


# generate_gaussian_doppler_signal() {{{1
def generate_gaussian_doppler_signal(
    n_pulses: int,
    prf: float,
    mean: float,
    width: float,
    snr_db: float | None = None,
    seed: int | None = None,
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

    Returns:
    - iq: Complex time-series array of shape (n_pulses,)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / prf
    t = np.arange(n_pulses) * dt
    freqs = fftfreq(n_pulses, d=dt)

    # Desired Gaussian PSD centered at 0 Hz
    psd = np.exp(-(freqs**2) / (2 * width**2))
    s0_power = np.sum(psd)

    if snr_db is not None:
        # snr_db = 10 * log10(S_0 / N) => N = S_0 / 10**(snr_db / 10)
        snr = 10 ** (snr_db / 10.0)
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
    iq = iq_base * np.exp(1j * 2 * np.pi * mean * t)

    return iq
