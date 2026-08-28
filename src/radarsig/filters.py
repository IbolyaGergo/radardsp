import tomllib
import numpy as np
from pathlib import Path
from scipy.signal import freqz


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
