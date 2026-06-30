import pytest
import numpy as np
from pathlib import Path
from radarsig.io import load_pulses

def test_load_pulses_basic(tmp_path):
    # Setup: Create a directory for dummy files
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Create dummy data files
    # Note: np.savez expects keyword arguments for the arrays
    np.savez(raw_dir / "data_p1.npz", Tx=np.array([1, 2, 3]))
    np.savez(raw_dir / "data_p2.npz", Tx=np.array([4, 5, 6]))

    # Execute
    # We use '*' as the pattern and pass the directory separately
    result = load_pulses(str(raw_dir), "data_p*.npz", n_samples=3)

    # Assert
    assert 'Tx' in result
    assert result['Tx'].shape == (2, 3)
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(result['Tx'], expected)

def test_load_pulses_skips_bad(tmp_path):
    # Setup
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    
    np.savez(raw_dir / "data_p1.npz", Tx=np.array([1, 1, 1]))
    np.savez(raw_dir / "data_p2.npz", Tx=np.array([2, 2, 2]))
    
    # Execute (skip pulse 2)
    result = load_pulses(str(raw_dir), "data_p*.npz", n_samples=3, bad_pulses=[2])
    
    # Assert
    assert result['Tx'].shape == (1, 3)
    np.testing.assert_allclose(result['Tx'], [[1, 1, 1]])
