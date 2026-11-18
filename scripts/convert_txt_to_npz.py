#!/usr/bin/env python3
import sys

import numpy as np

from radarsig.io import load_pulse_from_txt


def main(input_file, output_file):
    data = load_pulse_from_txt(input_file)
    data["fs"] = 250e6  # 250 MHz samples/sec
    np.savez_compressed(output_file, **data)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_single.py <input_txt> <output_npz>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
