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

def _get_pulse_idx(file_path: Path) -> int:
    """Helper to extract pulse index from filename."""
    return int(file_path.stem.split('_p')[-1])

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
