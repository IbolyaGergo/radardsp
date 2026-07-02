import numpy as np
from scipy.signal import lfilter
from radarsig.filters import Filter
from typing import Dict, Any, Optional

def _apply_filter(data: np.ndarray, filt: Filter) -> np.ndarray:
    """Apply a filter to data."""
    return lfilter(filt.b, filt.a, data)

class SignalPipeline:
    """Encapsulates the signal processing pipeline state."""
    
    def __init__(self, filters: Dict[str, Filter]):
        self.filters = filters
        self.history: Dict[str, Any] = {}
        self.current_data: Optional[np.ndarray] = None

    def load_data(self, data: np.ndarray) -> "SignalPipeline":
        self.current_data = data
        return self

    def apply_filter(self, filter_name: str, key: str, data: Optional[np.ndarray] = None) -> "SignalPipeline":
        """Applies filter by name, stores in history under 'key', and returns self for chaining."""
        input_data = data if data is not None else self.current_data
        if input_data is None:
            raise ValueError("No data provided or loaded in pipeline.")
            
        if filter_name not in self.filters:
            raise ValueError(f"Filter {filter_name} not found.")
        
        filt = self.filters[filter_name]
        result = _apply_filter(input_data, filt)
        
        # Update state
        self.current_data = result
        self.history[key] = result
        
        return self
