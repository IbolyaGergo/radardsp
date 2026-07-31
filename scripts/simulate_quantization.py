#!/usr/bin/env python3
"""
Simulate Q16 fixed-point quantization for IIR filter coefficients from gcinit.sh
and analyze the impact on frequency response.
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.signal import freqz
from radarsig.plot_utils import plot_filter_response

# Hardcoded coefficients from gcinit.sh
# b: feedforward (K registers)
# a: feedback (C registers)
B_COEFFS = [0.90186173, -2.703357699, 2.703357699, -0.90186173]
A_COEFFS = [1.00000000, 2.794421010, -2.605942211, 0.810075596] # Note: C0 is 1.0 (unused)


def float_to_q16_int(val: float) -> int:
    """Scale float by 2^16 and round to integer."""
    return int(round(val * 65536))


def q16_int_to_float(int_val: int) -> float:
    """Convert Q16 integer back to float."""
    if int_val & 0x80000000:
        signed_val = int_val - 0x100000000
    else:
        signed_val = int_val
    return signed_val / 65536.0


def print_coefficient_table(name: str, coeffs: list[float]):
    print(f"\n--- {name} Coefficients ---")
    print(f"{'Index':<6} {'Original Float':<18} {'Q16 Dec':<12} {'Q16 Hex':<12} {'Quantized Float':<18} {'Error':<12}")
    print("-" * 80)
    for i, orig in enumerate(coeffs):
        q16_int = float_to_q16_int(orig)
        hex_str = f"0x{q16_int & 0xFFFFFFFF:08X}"
        quant_float = q16_int_to_float(q16_int & 0xFFFFFFFF)
        error = abs(orig - quant_float)
        print(f"{i:<6} {orig:<18.8f} {q16_int:<12} {hex_str:<12} {quant_float:<18.8f} {error:<12.2e}")


def analyze_frequency_response(b_orig, a_orig, b_quant, a_quant):
    w, h_orig = freqz(b_orig, a_orig, worN=1024)
    w, h_quant = freqz(b_quant, a_quant, worN=1024)

    mag_orig_db = 20 * np.log10(np.maximum(np.abs(h_orig), 1e-12))
    mag_quant_db = 20 * np.log10(np.maximum(np.abs(h_quant), 1e-12))

    diff = np.abs(mag_orig_db - mag_quant_db)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print("\n--- Frequency Response Analysis (Ideal vs Q16 Quantized) ---")
    print(f"Max magnitude difference (dB): {max_diff:.4f}")
    print(f"Mean magnitude difference (dB): {mean_diff:.4f}")

    print(f"\n{'Freq (rad/pi)':<18} {'Ideal (dB)':<15} {'Quantized (dB)':<15} {'Diff (dB)':<15}")
    print("-" * 65)
    indices = [0, 256, 512, 768, 1023]
    for idx in indices:
        freq_pi = w[idx] / np.pi
        print(f"{freq_pi:<18.4f} {mag_orig_db[idx]:<15.2f} {mag_quant_db[idx]:<15.2f} {diff[idx]:<15.2f}")


def main():
    parser = argparse.ArgumentParser(description="Simulate Q16 quantization and frequency response for IIR filter.")
    parser.parse_args()

    print("IIR Filter Coefficient Q16 Quantization & Frequency Response Simulation")
    print("Format: Q16 (Fraction bits = 16, Scale = 65536)")

    print_coefficient_table("Numerator (b / K)", B_COEFFS)
    print_coefficient_table("Denominator (a / C)", A_COEFFS)

    b_quant = [q16_int_to_float(float_to_q16_int(v) & 0xFFFFFFFF) for v in B_COEFFS]
    a_quant = [q16_int_to_float(float_to_q16_int(v) & 0xFFFFFFFF) for v in A_COEFFS]

    analyze_frequency_response(B_COEFFS, A_COEFFS, b_quant, a_quant)

    # Plot and save comparison
    fig, axes = plot_filter_response(B_COEFFS, A_COEFFS, label="Ideal (Float)")
    plot_filter_response(b_quant, a_quant, axes=axes, label="Q16 Quantized")

    plot_path = Path("tmp/filter_response_comparison.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300)
    print(f"\nSaved frequency response comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
