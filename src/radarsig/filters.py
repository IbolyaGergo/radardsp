from pathlib import Path

import numpy as np
import tomllib
from scipy.signal import freqz, lfilter, tf2zpk


def load_filter_from_toml(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    b = np.array(data["b"])
    a = np.array(data["a"])

    return b, a


def compute_noise_power_gain_db(b: np.ndarray, a: np.ndarray, n_points: int = 512) -> float:
    """Compute theoretical noise power gain of an IIR filter in dB using freqz."""
    freqs, h = freqz(b, a, worN=n_points)
    h_power = np.abs(h) ** 2
    gain = np.trapezoid(h_power, freqs) / np.pi
    with np.errstate(divide="ignore"):
        return float(10 * np.log10(gain))


def step_response_settling(b, a, suppression_db=35.0, n_samples=4096):
    """
    Settling time from a unit-step response, expressed as DC suppression.

    The unit step represents a DC input. At each sample the instantaneous
    attenuation is 20*log10|y[n]| relative to the unit input, floored at
    -240 dB. The filter is considered settled at the first sample after which
    that attenuation is at most -suppression_db and stays there forever.

    suppression_db is an absolute suppression target (e.g. 35.0 for "DC
    suppressed by 35 dB or better"), which is the physically meaningful
    criterion when sizing warm-up against the clutter-to-signal ratio.

    Returns (settle_index, steady_state, attenuation_db) where attenuation_db
    is the per-sample 20*log10|y[n]|. settle_index is None if the target is not
    reached within n_samples.
    """
    if suppression_db <= 0:
        raise ValueError(f"suppression_db must be positive, got {suppression_db}")

    y = lfilter(b, a, np.ones(n_samples))

    # Steady state
    ss = np.sum(b) / np.sum(a)  # DC gain = H(z=1)
    attenuation_db = 20 * np.log10(np.maximum(np.abs(y), 1e-12))

    inside = attenuation_db <= -suppression_db
    outside_idx = np.flatnonzero(~inside)
    if outside_idx.size == 0:
        settle = 0
    elif outside_idx[-1] >= n_samples - 1:
        settle = None  # did not settle within n_samples
    else:
        settle = int(outside_idx[-1] + 1)

    return settle, ss, attenuation_db


def pole_settling_estimate(b, a, suppression_db=35.0):
    """
    Conservative estimate of the settling time from the slowest pole.

    The suppression target is absolute: |y[n]| <= 10**(-suppression_db/20).
    The unit-step transient is assumed to have amplitude at most 1 and to decay
    like |p_max|^n, so the instantaneous output is bounded by
    |y[n]| <= |y_ss| + |p_max|^n. The target is therefore guaranteed once
    |p_max|^n <= 10**(-suppression_db/20) - |y_ss|.

    Returns None if the steady-state suppression is not better than the target.
    FIR filters settle in len(b) - 1 samples.
    """
    if suppression_db <= 0:
        raise ValueError(f"suppression_db must be positive, got {suppression_db}")

    ss = abs(np.sum(b) / np.sum(a))  # |H(z=1)|
    threshold = 10 ** (-suppression_db / 20)
    if ss >= threshold:
        return None  # steady state never reaches the target suppression

    _, p, _ = tf2zpk(b, a)
    nonzero_poles = p[np.abs(p) > 0]
    if nonzero_poles.size == 0:
        return len(b) - 1  # FIR

    p_max = np.max(np.abs(nonzero_poles))
    return int(np.ceil(np.log(threshold - ss) / np.log(p_max)))
