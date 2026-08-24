#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz, fftconvolve
from pathlib import Path
from radarsig.fpga_io import find_iq_pairs, load_fpga_ram_binary_to_iq, load_filter_coeffs_from_binary


#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from pathlib import Path
from radarsig.fpga_io import find_iq_pairs, load_fpga_ram_binary_to_iq, load_filter_coeffs_from_binary

def main():
   data_dir = Path("tmp/iq")
   try:
       pairs = find_iq_pairs(data_dir)
   except FileNotFoundError as e:
       print(f"Error: {e}")
       return

   if not pairs:
       print(f"No IQ pairs found in {data_dir}")
       return

   out_dir = Path("tmp")
   out_dir.mkdir(exist_ok=True)

   window = np.hamming(14)
   fft_len = 256
   half_len = fft_len // 2 + 1
   freqs = np.linspace(0, np.pi, half_len)

   print(f"Found {len(pairs)} IQ pair(s). Processing...")

   for pair_id, i_path, q_path in pairs:
       print(f"Processing pair: {pair_id}...")
       try:
           x, y = load_fpga_ram_binary_to_iq(i_path, q_path, offset_dtype=512, n_pulse=14)
           b, a = load_filter_coeffs_from_binary(i_path)
       except Exception as e:
           print(f"  Error loading {pair_id}: {e}")
           continue

       # from Matej's script, 3165 is the measured data. The rest is irrelevant.
       n_bins = 3165
       x_sub = x[:n_bins, :]
       y_sub = y[:n_bins, :]

       ratio_db_list = []
       for k in range(n_bins):
           x_win = x_sub[k] * window
           y_win = y_sub[k] * window

           X_fft = np.fft.fft(x_win, n=fft_len)[:half_len]
           Y_fft = np.fft.fft(y_win, n=fft_len)[:half_len]

           with np.errstate(divide='ignore', invalid='ignore'):
               ratio_db = 20 * np.log10(np.abs(Y_fft)) - 20 * np.log10(np.abs(X_fft))
           ratio_db_list.append(ratio_db)

       ratio_db_array = np.array(ratio_db_list)
       median_ratio_db = np.median(ratio_db_array, axis=0)

       # Theoretical response
       w, h = freqz(b, a, worN=half_len)
       h_db = 20 * np.log10(np.abs(h))

       # Plotting
       plt.figure(figsize=(8, 5))
       plt.plot(freqs, h_db, label="Theoretical ($|H|$) - freqz", color="black", linewidth=2, linestyle="--")
       plt.plot(freqs, median_ratio_db, label="Empirical Median ($|Y|/|X|$)", color="blue", alpha=0.8)

       plt.title(f"IIR Filter Frequency Response - Pair {pair_id}")
       plt.xlabel("Digital Frequency ($\omega$) [rad/sample]")
       plt.ylabel("Magnitude [dB]")
       plt.grid(True, which="both", linestyle=":", alpha=0.7)
       plt.legend()
       plt.tight_layout()

       plot_path = out_dir / f"filter_spectrum_{pair_id}.png"
       plt.savefig(plot_path)
       plt.close()
       print(f"  Saved plot to {plot_path}")

   print("Batch analysis complete.")

if __name__ == "__main__":
   main()
