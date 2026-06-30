import pytest
import numpy as np
from radarsig.parsers import parse_hex_line, hex_lines_to_dict

def test_parse_hex_line():
    sample_line = "03000000 1B 00 00 00 C5 FF F7 FF 01 00 D8 FF 2E 75 00 00"
    # Expected output verified against Octave:
    # C=-40, A=1, B=-9, D=-59, F=0, H=27
    expected = {'C': -40, 'Tx': 1, 'H_Hi': -9, 'H_Lo': -59, 'V_Hi': 0, 'V_Lo': 27}

    assert parse_hex_line(sample_line) == expected

def test_hex_lines_to_dict():
    # Use a minimal line that satisfies the 13-part requirement
    # parts[1]..parts[12] -> V_Lo, V_Hi, H_Lo, H_Hi, Tx, C (in order of usage in _to_int16)
    # The example: 12 bytes = 6 pairs.
    # line: "header V_Lo V_Lo_hi V_Hi V_Hi_hi H_Lo H_Lo_hi H_Hi H_Hi_hi Tx Tx_hi C C_hi"
    # To get integer 1 for a pair: "01 00" (little endian hex) -> 0x0001 = 1
    # Line: "00 01 00 00 02 00 03 00 04 00 05 00 06 00" (14 parts)
    # V_Lo: "01 00" -> 1
    # V_Hi: "02 00" -> 2
    # H_Lo: "03 00" -> 3
    # H_Hi: "04 00" -> 4
    # Tx: "05 00" -> 5
    # C: "06 00" -> 6
    
    line = "00 01 00 02 00 03 00 04 00 05 00 06 00"
    lines = [line, "", line]
    
    result = hex_lines_to_dict(lines)
    
    assert isinstance(result, dict)
    for key in ['C', 'Tx', 'H_Hi', 'H_Lo', 'V_Hi', 'V_Lo']:
        assert key in result
        assert isinstance(result[key], np.ndarray)
        assert result[key].shape == (2,)
        
    assert result['V_Lo'][1] == 1
    assert result['V_Hi'][1] == 2
    assert result['H_Lo'][1] == 3
    assert result['H_Hi'][1] == 4
    assert result['Tx'][1] == 5
    assert result['C'][1] == 6
