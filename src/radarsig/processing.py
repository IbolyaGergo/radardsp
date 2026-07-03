import numpy as np
from scipy.signal import lfilter
from radarsig.filters import Filter
from typing import Dict, Any, Optional

def _apply_filter(data: np.ndarray, filt: Filter) -> np.ndarray:
    """Private helper to apply a filter to data."""
    return lfilter(filt.b, filt.a, data)

def _downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Private helper to downsample data along the sample dimension."""
    if data.ndim == 1:
        return data[::factor]
    return data[:, ::factor]

class SignalPipeline:
    """Encapsulates the signal processing pipeline state."""
    
    def __init__(self, filters: Dict[str, Filter]):
        self.filters = filters
        self.history: Dict[str, Any] = {}
        self.current_data: Optional[np.ndarray] = None

    def load_data(self, data: np.ndarray) -> "SignalPipeline":
        self.current_data = data
        return self

    def _execute_step(
        self,
        transform_func,
        *args,
        key: str,
        data: Optional[np.ndarray] = None,
        **kwargs
    ) -> "SignalPipeline":
        """Internal wrapper to handle boilerplate."""
        # 1. Resolve data
        input_data = data if data is not None else self.current_data
        if input_data is None:
            raise ValueError("No data provided or loaded in pipeline.")

        # 2. Transform
        result = transform_func(input_data, *args, **kwargs)

        # 3. Update state
        self.current_data = result
        self.history[key] = result

        return self

    def apply_filter(self, filter_name: str, *, key: str, data: Optional[np.ndarray] = None) -> "SignalPipeline":
        """Applies filter by name, stores in history under 'key', and returns self for chaining."""
        if filter_name not in self.filters:
            raise ValueError(f"Filter {filter_name} not found.")

        return self._execute_step(_apply_filter, self.filters[filter_name], key=key, data=data)

    def downsample(self, factor: int, *, key: str, data: Optional[np.ndarray] = None) -> "SignalPipeline":
        """Downsamples data, stores in history under 'key', and returns self for chaining."""
        return self._execute_step(_downsample, factor, key=key, data=data)
