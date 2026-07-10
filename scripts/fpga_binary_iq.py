import numpy as np

from pathlib import Path

def load_binary_to_array(path: Path, dtype=np.int32, offset_dtype: int = 0) -> np.ndarray:
    n_bytes_dtype = np.dtype(dtype).itemsize
    offset_bytes = offset_dtype * n_bytes_dtype
    return np.fromfile(path, dtype=dtype, offset=offset_bytes)

def process_channel_data(data: np.ndarray, n_rows: int = 14):
    """ Split even(x) and odd(y) and reshape. """
    x, y = data[0::2], data[1::2]

    n_cols = len(x) // n_rows
    n_elem = n_rows * n_cols

    return (
        x[:n_elem].reshape((n_rows, n_cols)),
        y[:n_elem].reshape((n_rows, n_cols))
    )

def load_fpga_ram_binary_to_iq(i_path: Path, q_path: Path, offset_dtype: int = 0, n_rows: int = 14):
    i_data = load_binary_to_array(i_path, offset_dtype=offset_dtype)
    q_data = load_binary_to_array(q_path, offset_dtype=offset_dtype)

    x_i, y_i = process_channel_data(i_data, n_rows=n_rows)
    x_q, y_q = process_channel_data(q_data, n_rows=n_rows)

    return (x_i + 1j * x_q), (y_i + 1j * y_q)
