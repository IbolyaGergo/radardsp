import sys
import numpy as np
from radarsig.io import load_pulse_from_txt
from radarsig.parsers import hex_lines_to_dict

def main(input_file, output_file):
    data = load_pulse_from_txt(input_file, n_samples=30000)
    np.savez_compressed(output_file, **data)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_single.py <input_txt> <output_npz>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
