# FPGA IIR Filtering & Spectral Analysis Insights

This document captures theoretical and practical insights regarding FPGA IIR filter execution, fixed-point quantization noise, transient responses, and spectral estimation methods (`compute_median_ratio_spectrum` vs. `compute_csd_spectrum`).

---

## 1. Fixed-Point Quantization in FPGA IIR Filters
FPGAs implement digital filters using fixed-point arithmetic (e.g., 16 fractional bits as seen in `gcinit.sh`) rather than double-precision floating-point. 

* **Coefficient Quantization:** Filter coefficients $b$ and $a$ are rounded to discrete fixed-point values, slightly altering pole/zero locations and frequency responses.
* **Internal Rounding Noise:** Every multiplication and accumulation step requires rounding or truncation back to the register bit-width. This injects quantization noise modeled as additive white noise with variance:
  $$\sigma_q^2 = \frac{\Delta^2}{12}$$
  where $\Delta = 2^{-N}$ ($N$ is the number of fractional bits).
* **Feedback Amplification ($1/A$):** In IIR filters, internal quantization noise is fed back through the denominator $A(z)$. Consequently, noise is heavily amplified at the filter's resonance/passband peaks.

---

## 2. IIR Filter Transients vs. Steady-State (`freqz`)
* **`freqz`** assumes an infinite-duration input and computes the ideal steady-state frequency response.
* **`lfilter`** on finite signals starts with zero initial conditions. The beginning of the filtered signal exhibits **start-up transients (ringing)**. 
* Slicing signals into very short blocks (e.g., width 14) without accounting for filter memory or transient settling will yield spectra that deviate significantly from theoretical curves. Longer sequences or steady-state segments are required for proper convergence.

---

## 3. Median Ratio Spectrum vs. CSD Spectrum on Real Radar Data
When comparing empirical FPGA data against theoretical models, two primary spectrum estimators behave very differently:

### A. Median Ratio Spectrum (`compute_median_ratio_spectrum`)
* **Mechanism:** Computes $\frac{|Y_k(\omega)|}{|X_k(\omega)|}$ for each range bin $k$ independently, then takes the **median** across all bins.
* **Characteristics:** Highly robust. Because it uses the median, it ignores outliers, strong ground clutter, non-stationary weather echoes, and dead/noisy bins, resulting in a clean response that matches theoretical expectations well.

### B. Cross-Spectral Density (CSD) Spectrum (`compute_csd_spectrum`)
* **Mechanism:** Computes:
  $$H_{\text{csd}}(\omega) = \frac{\sum_k Y_k(\omega) X_k^*(\omega)}{\sum_k X_k(\omega) X_k^*(\omega)}$$
* **Characteristics:** Sensitive to non-stationarities. Strong clutter or high-power point targets dominate the linear sums ($\sum$), leading to skew, bias, and passband fluctuations (e.g., ~10 dB ripples observed in real measurements).
