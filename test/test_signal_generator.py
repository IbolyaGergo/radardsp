import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from radarsig.signal_generator import (
    generate_gaussian_doppler_signal,
)


# test_generate_gaussian_doppler_signal_variance() {{{1
def test_generate_gaussian_doppler_signal_variance():
    n_pulses = 1024
    prf = 100.0
    fd = 15.0
    true_freq_variance = 9.0  # Hz^2
    seed = 42

    iq = generate_gaussian_doppler_signal(
        n_pulses=n_pulses,
        prf=prf,
        fd=fd,
        true_freq_variance=true_freq_variance,
        seed=seed,
    )

    # Compute shifted FFT and Power Spectral Density (PSD)
    freqs = fftshift(fftfreq(n_pulses, d=1.0 / prf))
    spectrum = fftshift(fft(iq))
    psd = np.abs(spectrum) ** 2

    # Normalize PSD to form a discrete probability distribution
    psd_norm = psd / np.sum(psd)

    # Compute mean frequency and frequency variance from the PSD
    mean_freq = np.sum(freqs * psd_norm)
    est_variance = np.sum(psd_norm * (freqs - mean_freq) ** 2)

    # Verify center frequency and spectral variance
    np.testing.assert_allclose(mean_freq, fd, atol=2.0)
    np.testing.assert_allclose(est_variance, true_freq_variance, rtol=0.01)
