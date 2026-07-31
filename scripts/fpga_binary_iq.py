""" Script to load tmp/iq/*.data to complex np.ndarray. """
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from radarsig.fpga_io import (
    load_fpga_ram_binary_to_iq,
    load_filter_coeffs_from_binary,
)


if __name__ == "__main__":
    x, y = load_fpga_ram_binary_to_iq("tmp/iq/012_i.data", "tmp/iq/012_q.data", offset_dtype=512)
    b, a = load_filter_coeffs_from_binary("tmp/iq/012_i.data")

    res = analyze_iq_data(x, y, b, a)
