import numpy as np


def parse_hex_line(line_str):
    parts = line_str.strip().split()

    # Helper to convert a pair of hex strings (e.g., '1B', '00') to int16
    def to_int16(hi, lo):
        # Join hex strings and convert to integer
        val = int(hi + lo, 16)
        # Handle signed 16-bit integer (if >= 0x8000, it's negative)
        if val >= 0x8000:
            val -= 0x10000
        return val

    # From FasterLoadHexdump.m
    # Based on the Octave script, skipping the first column:
    # Octave uses: col3, col2 | col5, col4 | col7, col6 | ...
    # Python list indices (0-based) for the remaining columns:
    # Mapping
    # C (Col 13/12) -> C
    # A (Col 11/10) -> Tx
    # B (Col 9/8) -> H_Hi
    # D (Col 7/6) -> H_Lo
    # F (Col 5/4) -> V_Hi
    # H (Col 3/2) -> V_Lo


    C = to_int16(parts[12], parts[11])
    H_Hi = to_int16(parts[8], parts[7])
    H_Lo = to_int16(parts[6], parts[5])
    Tx = to_int16(parts[10], parts[9])
    V_Hi = to_int16(parts[4], parts[3])
    V_Lo = to_int16(parts[2], parts[1])

    return {'C': C, 'Tx': Tx, 'H_Hi': H_Hi, 'H_Lo': H_Lo, 'V_Hi': V_Hi, 'V_Lo': V_Lo}

def read_pulse_file(file_path):
    """
    Reads a pulse file directly into columnar data.
    """
    # Initialize the structure with lists
    data = {'C': [], 'Tx': [], 'H_Hi': [], 'H_Lo': [], 'V_Hi': [], 'V_Lo': []}

    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue

            row = parse_hex_line(line)
            # Append values to the corresponding list
            for key in data:
                data[key].append(row[key])

    # Convert lists to numpy arrays
    return {key: np.array(vals) for key, vals in data.items()}
