import pytest
import numpy as np
import shutil
from pathlib import Path
from radarsig.io import (
    load_pulses, load_pulse_from_txt, load_pulseset_from_txt, _get_pulse_idx
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
