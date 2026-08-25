import numpy as np
from scipy.signal import lfilter
from radarsig.fpga_analysis import (
    analyze_iq_data,
    compute_median_ratio_spectrum,
    compute_csd_spectrum,
    compute_coherence,
)

def test_analyze_iq_data():
    b = np.array([0.8, 0.2], dtype=float)
    a = np.array([1.0, 0.0], dtype=float)

    rng = np.random.default_rng(42)
    x = rng.normal(size=(50, 256)) + 1j * rng.normal(size=(50, 256))

    # Construct y such that y[n] = b[0]*x[n] + b[1]*x[n+1] (matching x_sum)
    y = np.zeros_like(x)
    y[:, :-1] = b[0] * x[:, :-1] + b[1] * x[:, 1:]
    y[:, -1] = b[0] * x[:, -1]

    results = analyze_iq_data(x, y, b, a, threshold=1e-3)
    for part in ('real', 'imag'):
        err_rel_max = np.max(np.abs(results[part]['err_rel']))
        assert err_rel_max < 1e-12
        assert len(results[part]['failing_bins']) == 0


def test_compute_spectral_methods():
    b = np.array([0.8, 0.2], dtype=float)
    a = np.array([1.0], dtype=float)

    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 256)) + 1j * rng.normal(size=(100, 256))

    y_real = np.array([lfilter(b, a, row) for row in x.real])
    y_imag = np.array([lfilter(b, a, row) for row in x.imag])
    y = y_real + 1j * y_imag

    # Test median ratio spectrum
    freqs, h_db, median_ratio_db = compute_median_ratio_spectrum(x, y, b, a)
    rmse_median = np.sqrt(np.mean((h_db - median_ratio_db)**2))
    assert rmse_median < 1.5

    # Test CSD spectrum
    freqs_csd, _, h_csd_db = compute_csd_spectrum(x, y, b, a)
    rmse_csd = np.sqrt(np.mean((h_db_csd - h_csd_db)**2))
    assert rmse_csd < 1.5

    # Test coherence
    freqs_coh, _, coherence = compute_coherence(x, y, b, a)
    mean_coh = np.mean(coherence)
    assert mean_coh > 0.90
