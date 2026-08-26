#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from radarsig.fpga_io import find_iq_pairs
from radarsig.fpga_analysis import analyze_iq_pair


def main():
    parser = argparse.ArgumentParser(description="Batch analyze FPGA IQ data.")
    parser.add_argument("--dir", default="data/raw/fpga/iq", help="Directory with IQ data")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    try:
        pairs = find_iq_pairs(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Header
    print(f"{'Pair':<6} | {'Real Med/P99':<15} | {'Imag Med/P99':<15}")
    print("-" * 45)

    for pair_id, i_path, q_path in pairs:
        try:
            res = analyze_iq_pair(i_path, q_path)
            r = res["real"]
            i = res["imag"]
            print(
                f"{pair_id:<6} | {r['median']:.2e}/{r['p99']:.2e} | {i['median']:.2e}/{i['p99']:.2e}"
            )
        except Exception as e:
            print(f"Error processing {pair_id}: {e}", file=sys.stderr)
            continue


if __name__ == "__main__":
    main()
