import sys
import numpy as np
from radarsig.utils import read_pulse_file

def main(input_file, output_file):
    data = read_pulse_file(input_file)
    np.savez_compressed(output_file, **data)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_single.py <input_txt> <output_npz>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
