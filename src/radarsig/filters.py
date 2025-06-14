from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy.signal import freqz
from matplotlib import pyplot as plt

@dataclass
class Filter:
    """A container for filter coefficients."""
    a: np.ndarray
    b: np.ndarray
    label: str | None = None

    def get_response(self, n: int = 512):
        """Computes frequency response."""
        return freqz(self.b, self.a, worN=n)

def _format_freqz_axes(ax_mag: plt.Axes, ax_phase: plt.Axes):
    """Encapsulates all styling/aesthetics."""
    ax_mag.set_title("Frequency Response Comparison")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.grid(True)
    ax_phase.set_ylabel("Phase [rad]")
    ax_phase.set_xlabel("Frequency [rad/sample]")
    ax_phase.grid(True)

def _add_filter_to_axes(ax_mag: plt.Axes, ax_phase: plt.Axes, filt: Filter,
                       **kwargs) -> None:
    w, h = filt.get_response()

    ax_mag.plot(w, 20 * np.log10(np.abs(h)), label=filt.label, **kwargs)
    ax_phase.plot(w, np.unwrap(np.angle(h)), label=filt.label, **kwargs)

def plot_filters(filters: Iterable[Filter], n: int = 512, axes=None, **kwargs):
    """
    Plots magnitude and phase for multiple filters on subplots.
    
    Accepts optional `axes` (tuple of magnitude and phase axes) to add to an existing plot.
    Accepts arbitrary keyword arguments to pass to ax.plot().
    """
    # 1. If no axes provided, create them and apply formatting
    if axes is None:
        fig, axes = plt.subplots(2, 1, tight_layout=True, figsize=(8, 6))
        _format_freqz_axes(*axes)
    
    ax_mag, ax_phase = axes
    
    # 2. Plot the data
    for f in filters:
        _add_filter_to_axes(ax_mag, ax_phase, f, **kwargs)

    # 3. Update legends to include new lines
    if any(f.label for f in filters):
        ax_mag.legend()
        ax_phase.legend()

    return axes[0].get_figure(), axes
