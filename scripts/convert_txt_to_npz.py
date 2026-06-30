import sys
import numpy as np
from radarsig.parsers import hex_lines_to_dict

def main(input_file, output_file):
    with open(input_file, 'r') as f:
        data = hex_lines_to_dict(f)
    np.savez_compressed(output_file, **data)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_single.py <input_txt> <output_npz>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
