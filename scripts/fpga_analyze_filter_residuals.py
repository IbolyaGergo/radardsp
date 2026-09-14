#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from radarsig.fpga_analysis import analyze_iq_pair


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FPGA filter coefficient verification and error metrics for a single IQ pair."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/raw/fpga/iq"),
        help="Directory with IQ data",
    )
    parser.add_argument(
        "--pair",
        default="000",
        help="Pair ID to process (default: 000)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV file path. If not specified, prints result to stdout.",
    )
    args = parser.parse_args()

    data_dir = Path(args.dir)
    i_path = data_dir / f"{args.pair}_i.data"
    q_path = data_dir / f"{args.pair}_q.data"

    if not i_path.exists() or not q_path.exists():
        print(f"Error: Pair {args.pair} files not found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        res = analyze_iq_pair(i_path, q_path)
        r = res["real"]
        i = res["imag"]
    except Exception as e:
        print(f"Error processing {args.pair}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.out is not None:
        import csv

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["pair", "real_median", "real_p99", "imag_median", "imag_p99"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "pair": args.pair,
                    "real_median": r["median"],
                    "real_p99": r["p99"],
                    "imag_median": i["median"],
                    "imag_p99": i["p99"],
                }
            )
        print(f"Saved CSV result to {args.out}")
    else:
        print(f"{'Pair':<6} | {'Real Med/P99':<15} | {'Imag Med/P99':<15}")
        print("-" * 45)
        print(
            f"{args.pair:<6} | {r['median']:.2e}/{r['p99']:.2e} | {i['median']:.2e}/{i['p99']:.2e}"
        )


if __name__ == "__main__":
    main()
