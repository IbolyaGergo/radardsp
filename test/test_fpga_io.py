import pytest
import numpy as np
from pathlib import Path
from radarsig.fpga_io import (
    load_binary_to_array,
    process_channel_data,
    process_filter_coeffs,
    find_iq_pairs,
)

# test_load_binary_to_array() {{{1
def test_load_binary_to_array(tmp_path):
    # Create synthetic int32 binary data
    bin_path = tmp_path / "test.data"
    data = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
    data.tofile(bin_path)

    # Test loading without offset
    loaded = load_binary_to_array(bin_path, dtype=np.int32, offset_dtype=0)
    np.testing.assert_array_equal(loaded, data)

    # Test loading with offset (e.g. offset of 2 int32 elements = 8 bytes)
    loaded_offset = load_binary_to_array(bin_path, dtype=np.int32, offset_dtype=2)
    np.testing.assert_array_equal(loaded_offset, np.array([30, 40, 50, 60], dtype=np.int32))

# test_process_channel_data() {{{1
def test_process_channel_data():
    # Verify even indices go to x and odd indices go to y
    data = np.array([10, 11, 20, 21, 30, 31, 40, 41], dtype=np.int32)
    x, y = process_channel_data(data, n_pulse=2)
    
    np.testing.assert_array_equal(x, [[10, 20], [30, 40]])
    np.testing.assert_array_equal(y, [[11, 21],[ 31, 41]])

# test_process_filter_coeffs() {{{1
def test_process_filter_coeffs():
    # Even indices for b, odd indices for a, with a[1:] negated
    data = np.array([10, 1, 20, -2, 30, 3, 40, -4], dtype=np.int32)
    b, a = process_filter_coeffs(data, n_tap=4)

    np.testing.assert_array_equal(b, [10, 20, 30, 40])
    np.testing.assert_array_equal(a, [1, 2, -3, 4])

# test_find_iq_pairs() {{{1
def test_find_iq_pairs(tmp_path):
    (tmp_path / "000_i.data").touch()
    (tmp_path / "000_q.data").touch()
    (tmp_path / "001_i.data").touch()
    (tmp_path / "001_q.data").touch()

    # Unmatched orphan file
    (tmp_path / "002_i.data").touch()

    pairs = find_iq_pairs(tmp_path)

    assert len(pairs) == 2
    assert pairs[0] == ("000", tmp_path / "000_i.data", tmp_path / "000_q.data")
    assert pairs[1] == ("001", tmp_path / "001_i.data", tmp_path / "001_q.data")
