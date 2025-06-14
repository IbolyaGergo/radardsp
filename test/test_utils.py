import pytest
from radarsig.utils import parse_hex_line

def test_parse_hex_line():
    sample_line = "03000000 1B 00 00 00 C5 FF F7 FF 01 00 D8 FF 2E 75 00 00"
    # Expected output verified against Octave:
    # C=-40, A=1, B=-9, D=-59, F=0, H=27
    expected = {'C': -40, 'Tx': 1, 'H_Hi': -9, 'H_Lo': -59, 'V_Hi': 0, 'V_Lo': 27}

    assert parse_hex_line(sample_line) == expected
