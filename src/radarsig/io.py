import numpy as np
from radarsig import parsers
from pathlib import Path
from typing import NamedTuple

class PulseSet(NamedTuple):
    pulses: dict[str, np.ndarray]
    fs: float

def load_pulse_from_txt(path: Path, n_samples: int) -> dict[str, np.ndarray]:
    """Loads and parses a single pulse from a text file."""
    with open(path, 'r') as f:
        # Use existing parser
        data = parsers.hex_lines_to_dict(f)
    
    # Validation check
    for key, arr in data.items():
        if len(arr) != n_samples:
            raise ValueError(f"Shape mismatch in {path}: expected {n_samples}, got {len(arr)}")
            
    return data

def _aggregate_pulse_data(list_of_dicts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Helper to aggregate a list of pulse dictionaries into concatenated arrays."""
    if not list_of_dicts:
        return {}
        
    # Initialize buffers
    buffers = {key: [] for key in list_of_dicts[0].keys()}
    
    # Fill buffers
    for pulse_dict in list_of_dicts:
        for key, arr in pulse_dict.items():
            buffers[key].append(np.atleast_2d(arr))
            
    # Concatenate
    return {key: np.concatenate(arrs, axis=0) for key, arrs in buffers.items()}

def _get_pulse_idx(file_path: Path) -> int:
    """Helper to extract pulse index from filename."""
    return int(file_path.stem.split('_p')[-1])

def load_pulseset_from_txt(directory: str, data_pattern: str, n_samples: int, fs: float = 250_000_000.0) -> PulseSet:
    """Loads and aggregates all .txt pulses from a directory."""
    files = sorted(Path(directory).glob(data_pattern), key=_get_pulse_idx)
    
    # Load all pulses into a list of dictionaries
    pulses = [load_pulse_from_txt(f, n_samples) for f in files]
    
    # Delegate the complex aggregation logic to our helper
    return PulseSet(pulses=_aggregate_pulse_data(pulses), fs=fs)

def load_pulse_from_npz(path: Path, n_samples: int) -> tuple[dict[str, np.ndarray], float]:
    """Loads and validates a single pulse from an npz file."""
    with np.load(path) as data:
        fs = float(data['fs'])
        data_dict = {key: np.atleast_2d(data[key].copy()) for key in data.files if key != 'fs'}
        
        # Validation
        for key, arr in data_dict.items():
            if arr.shape[1] != n_samples:
                raise ValueError(f"Shape mismatch in {path}: expected {n_samples}, got {arr.shape[1]}")
                
        return data_dict, fs

def load_pulseset_from_npz(directory: str, data_pattern: str, n_samples: int) -> PulseSet:
    """Loads and aggregates all .npz pulses from a directory."""
    files = sorted(Path(directory).glob(data_pattern), key=_get_pulse_idx)
    
    pulse_results = []
    fs = None
    
    for file_path in files:
        try:
            data, current_fs = load_pulse_from_npz(file_path, n_samples)
            if fs is None:
                fs = current_fs
            pulse_results.append(data)
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            continue
            
    if not pulse_results:
        return PulseSet(pulses={}, fs=250_000_000.0)
        
    return PulseSet(pulses=_aggregate_pulse_data(pulse_results), fs=fs or 250_000_000.0)
