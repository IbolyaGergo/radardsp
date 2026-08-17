""" Functions to load tmp/iq/*.data to complex np.ndarray. """
import numpy as np

from pathlib import Path


# load_binary_to_array() {{{1
def load_binary_to_array(path: Path, dtype=np.int32, offset_dtype: int = 0) -> np.ndarray:
    n_bytes_dtype = np.dtype(dtype).itemsize
    offset_bytes = offset_dtype * n_bytes_dtype
    return np.fromfile(path, dtype=dtype, offset=offset_bytes)

# process_channel_data() {{{1
def process_channel_data(data: np.ndarray, n_pulse: int = 14):
    """ Split even(x) and odd(y) and reshape. """
    x, y = data[0::2], data[1::2]

    pulse_len = len(x) // n_pulse
    n_total = n_pulse * pulse_len

    return (
        x[:n_total].reshape((pulse_len, n_pulse)),
        y[:n_total].reshape((pulse_len, n_pulse))
    )

# process_filter_coeffs() {{{1
def process_filter_coeffs(data: np.ndarray, n_tap: int = 8):
    b = data[0::2][:n_tap].copy()
    a = data[1::2][:n_tap].copy()

    a[1:] = -a[1:]

    return b, a

# load_fpga_ram_binary_to_iq() {{{1
def load_fpga_ram_binary_to_iq(i_path: Path, q_path: Path, offset_dtype: int = 0, n_pulse: int = 14):
    i_data = load_binary_to_array(i_path, offset_dtype=offset_dtype)
    q_data = load_binary_to_array(q_path, offset_dtype=offset_dtype)

    x_i, y_i = process_channel_data(i_data, n_pulse=n_pulse)
    x_q, y_q = process_channel_data(q_data, n_pulse=n_pulse)

    return (x_i + 1j * x_q), (y_i + 1j * y_q)

# load_filter_coeffs_from_binary() {{{1
def load_filter_coeffs_from_binary(path: Path, n_tap: int = 8):
    data = load_binary_to_array(path)

    return process_filter_coeffs(data, n_tap=n_tap)

# find_iq_pairs() {{{1
def find_iq_pairs(data_dir: Path | str) -> list[tuple[str, Path, Path]]:
    """ Find matching IQ file pairs (*_i.data and *_q.data) in directory. """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pairs = []
    for i_path in sorted(directory.glob("*_i.data")):
        pair_id = i_path.stem.removesuffix("_i")
        q_path = directory / f"{pair_id}_q.data"
        if q_path.exists():
            pairs.append((pair_id, i_path, q_path))

    return pairs
