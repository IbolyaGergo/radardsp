import tomllib
import numpy as np
from scipy.signal import freqz
from matplotlib import pyplot as plt
from pathlib import Path

def load_filter_from_toml(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    b=np.array(data["b"])
    a=np.array(data["a"])

    return b, a

def _format_freqz_axes(ax_mag: plt.Axes, ax_phase: plt.Axes):
    """Encapsulates all styling/aesthetics."""
    ax_mag.set_title("Frequency Response Comparison")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.grid(True)
    ax_phase.set_ylabel("Phase [rad]")
    ax_phase.set_xlabel("Frequency [rad/sample]")
    ax_phase.grid(True)

def plot_filter_response(b, a, worN: int = 512, axes=None, label: str | None =
                         None) -> tuple["fig", "axes"]:
    if axes is None:
        fig, axes = plt.subplots(2, 1, tight_layout=True, figsize=(8, 6))
        _format_freqz_axes(*axes)

    w, h = freqz(b, a, worN=worN)

    ax_mag, ax_phase = axes
    ax_mag.plot(w, 20 * np.log10(np.abs(h)), label=label)
    ax_phase.plot(w, np.unwrap(np.angle(h)), label=label)

    if label:
        ax_mag.legend()
        ax_phase.legend()

    return axes[0].get_figure(), axes
