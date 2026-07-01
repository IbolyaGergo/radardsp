import pytest
import numpy as np
from pathlib import Path
from radarsig.io import load_pulses, load_pulse_from_txt

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
