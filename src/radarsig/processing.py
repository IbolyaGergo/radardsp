import numpy as np
from scipy.signal import lfilter
from typing import Dict, Any, Optional

def _apply_filter(data: np.ndarray, filt: Filter) -> np.ndarray:
    """Private helper to apply a filter to data."""
    return lfilter(filt.b, filt.a, data)

def _downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Private helper to downsample data along the sample dimension."""
    if data.ndim == 1:
        return data[::factor]
    return data[:, ::factor]



