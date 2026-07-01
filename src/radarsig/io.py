import numpy as np
from pathlib import Path
import logging
from radarsig import parsers

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

def load_pulseset_from_txt(directory: str, data_pattern: str, n_samples: int) -> dict[str, np.ndarray]:
    """Loads and aggregates all .txt pulses from a directory."""
    files = sorted(Path(directory).glob(data_pattern), key=_get_pulse_idx)
    
    # Load all pulses into a list of dictionaries
    pulses = [load_pulse_from_txt(f, n_samples) for f in files]
    
    # Delegate the complex aggregation logic to our helper
    return _aggregate_pulse_data(pulses)

def load_pulses(directory: str, data_pattern: str, n_samples: int, bad_pulses: list[int] | None = None) -> dict[str, np.ndarray]:
    """
    Loads .npz files matching a glob pattern and aggregates them.

    Args:
        directory: Directory containing the files.
        data_pattern: A glob pattern string (e.g., '*_p*.npz').
        n_samples: Expected number of samples per pulse.
        bad_pulses: List of pulse indices to skip.

    Returns:
        A dictionary of concatenated numpy arrays.
    """
    bad_pulses = bad_pulses or []
    buffers = {}
    
    # Sort files numerically by the pulse index extracted from filename
    files = sorted(Path(directory).glob(data_pattern), key=_get_pulse_idx)
    
    if not files:
        logging.warning(f"No files found matching pattern: {directory}/{data_pattern}")
        return {}

    for file_path in files:
        # Extract pulse index using the same helper
        try:
            pulse_idx = _get_pulse_idx(file_path)
        except (ValueError, IndexError):
            logging.warning(f"Could not extract pulse index from {file_path}, skipping.")
            continue
            
        if pulse_idx in bad_pulses:
            continue

        try:
            with np.load(file_path) as data:
                for key in data.files:
                    if key not in buffers:
                        buffers[key] = []
                    
                    arr = np.atleast_2d(data[key].copy())
                    
                    if arr.shape[1] != n_samples:
                        raise ValueError(f"Shape mismatch in {file_path}: expected {n_samples}, got {arr.shape[1]}")
                        
                    buffers[key].append(arr)
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            continue
                
    if not buffers:
        return {}

    return {
        key: np.concatenate(list_of_arrs, axis=0)
        for key, list_of_arrs in buffers.items()
    }

def load_pulseset_from_npz(directory: str, data_pattern: str, n_samples: int, bad_pulses: list[int] | None = None) -> dict[str, np.ndarray]:
    """Alias for load_pulses to match the naming convention."""
    return load_pulses(directory, data_pattern, n_samples, bad_pulses)
