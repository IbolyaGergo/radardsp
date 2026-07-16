""" Script to load tmp/iq/*.data to complex np.ndarray. """
import numpy as np

from pathlib import Path

def load_binary_to_array(path: Path, dtype=np.int32, offset_dtype: int = 0) -> np.ndarray:
    n_bytes_dtype = np.dtype(dtype).itemsize
    offset_bytes = offset_dtype * n_bytes_dtype
    return np.fromfile(path, dtype=dtype, offset=offset_bytes)

def process_channel_data(data: np.ndarray, n_pulse: int = 14):
    """ Split even(x) and odd(y) and reshape. """
    x, y = data[0::2], data[1::2]

    pulse_len = len(x) // n_pulse
    n_total = n_pulse * pulse_len

    return (
        x[:n_total].reshape((pulse_len, n_pulse)),
        y[:n_total].reshape((pulse_len, n_pulse))
    )

def load_fpga_ram_binary_to_iq(i_path: Path, q_path: Path, offset_dtype: int = 0, n_pulse: int = 14):
    i_data = load_binary_to_array(i_path, offset_dtype=offset_dtype)
    q_data = load_binary_to_array(q_path, offset_dtype=offset_dtype)

    x_i, y_i = process_channel_data(i_data, n_pulse=n_pulse)
    x_q, y_q = process_channel_data(q_data, n_pulse=n_pulse)

    return (x_i + 1j * x_q), (y_i + 1j * y_q)
