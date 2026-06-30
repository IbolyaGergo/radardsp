import numpy as np
from typing import Iterable

def _to_int16(hi: str, lo: str) -> int:
    """Helper to convert a pair of hex strings to signed int16."""
    val = int(hi + lo, 16)
    if val >= 0x8000:
        val -= 0x10000
    return val

def parse_hex_line(line_str: str) -> dict[str, int]:
    """Parses a single hex string line into a dictionary of values."""
    parts = line_str.strip().split()
    if len(parts) < 13:
        raise ValueError(f"Line does not have enough parts: {len(parts)} < 13")

    # C -> parts[12], parts[11]
    # Tx -> parts[10], parts[9]
    # H_Hi -> parts[8], parts[7]
    # H_Lo -> parts[6], parts[5]
    # V_Hi -> parts[4], parts[3]
    # V_Lo -> parts[2], parts[1]

    return {
        'C': _to_int16(parts[12], parts[11]),
        'Tx': _to_int16(parts[10], parts[9]),
        'H_Hi': _to_int16(parts[8], parts[7]),
        'H_Lo': _to_int16(parts[6], parts[5]),
        'V_Hi': _to_int16(parts[4], parts[3]),
        'V_Lo': _to_int16(parts[2], parts[1]),
    }

def hex_lines_to_dict(lines: Iterable[str]) -> dict[str, np.ndarray]:
    """Parses an iterable of lines into a dictionary of arrays."""
    temp_data = {}
    
    for line in lines:
        if not line.strip():
            continue
        row = parse_hex_line(line)
        
        # Dynamically add keys as they appear
        for key, val in row.items():
            if key not in temp_data:
                temp_data[key] = []
            temp_data[key].append(val)
            
    # Convert lists to numpy arrays at the very end
    return {key: np.array(vals) for key, vals in temp_data.items()}
