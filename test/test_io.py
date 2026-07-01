import pytest
import numpy as np
import shutil
from pathlib import Path
from radarsig.io import (
    load_pulses, load_pulse_from_txt, load_pulseset_from_txt, _get_pulse_idx,
    load_pulseset_from_npz
)

MOCK_PULSE_PATH = Path("test/data_mock/sample_pulse.txt")
MOCK_PULSE_N_SAMPLES = 10

def test_load_pulse_from_txt_success():
    data = load_pulse_from_txt(MOCK_PULSE_PATH, n_samples=MOCK_PULSE_N_SAMPLES) 

    expected_keys = {'C', 'Tx', 'H_Hi', 'H_Lo', 'V_Hi', 'V_Lo'}
    assert set(data.keys()) == expected_keys

    for key in expected_keys:
        assert len(data[key]) == MOCK_PULSE_N_SAMPLES

def test_load_pulse_from_txt_n_samples_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch in test/data_mock/sample_pulse.txt"):
        load_pulse_from_txt(MOCK_PULSE_PATH, n_samples=MOCK_PULSE_N_SAMPLES-1)

def test_load_pulseset_from_txt(tmp_path):
    d = tmp_path / "data"
    d.mkdir()

    shutil.copy(MOCK_PULSE_PATH, d / "pulse_p1.txt")
    shutil.copy(MOCK_PULSE_PATH, d / "pulse_p2.txt")
    shutil.copy(MOCK_PULSE_PATH, d / "pulse_p9.txt")
    shutil.copy(MOCK_PULSE_PATH, d / "pulse_p10.txt")

    data = load_pulseset_from_txt(str(d), "*.txt", n_samples=MOCK_PULSE_N_SAMPLES)

    assert len(data['C'][0]) == 10

def test_pulse_sorting():
    files = [Path("pulse_p10.txt"), Path("pulse_p1.txt"), Path("pulse_p9.txt"), Path("pulse_p2.txt")]
    sorted_files = sorted(files, key=_get_pulse_idx)
    assert sorted_files == [Path("pulse_p1.txt"), Path("pulse_p2.txt"), Path("pulse_p9.txt"), Path("pulse_p10.txt")]

def test_raw_vs_converted_consistency():
    # Placeholder for actual implementation based on previous discussions
    pass

def test_raw_vs_converted_consistency_mocked(tmp_path):
    # 1. Setup: Load raw data from mock TXT
    raw_pulse = load_pulse_from_txt(MOCK_PULSE_PATH, n_samples=MOCK_PULSE_N_SAMPLES)
    
    # 2. Setup: Create a corresponding mock NPZ file
    npz_dir = tmp_path / "converted"
    npz_dir.mkdir()
    npz_path = npz_dir / "pulse_p1.npz"
    
    # Save the parsed raw_pulse dict to .npz
    # We use **raw_pulse to save all arrays in the dict
    np.savez(npz_path, **raw_pulse)
    
    # 3. Execution: Load via orchestrators
    # We copy the mock TXT to a new dir to mimic the "dataset" structure
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy(MOCK_PULSE_PATH, raw_dir / "pulse_p1.txt")
    
    raw_data = load_pulseset_from_txt(str(raw_dir), "*.txt", n_samples=MOCK_PULSE_N_SAMPLES)
    conv_data = load_pulseset_from_npz(str(npz_dir), "*.npz", n_samples=MOCK_PULSE_N_SAMPLES)
    
    # 4. Assert: Compare
    for key in raw_data:
        np.testing.assert_allclose(raw_data[key], conv_data[key])
