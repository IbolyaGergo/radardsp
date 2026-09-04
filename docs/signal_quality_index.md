# Signal Quality Index (SQI) in Doppler Radar

This document summarizes the mathematical definition, literature terminology, placement in signal processing pipelines, and practical interpretation of the **Signal Quality Index (SQI)** in Doppler and meteorological radar.

---

## 1. Mathematical Definition
The Signal Quality Index (SQI) is defined as the magnitude of the normalized lag-1 complex autocorrelation coefficient of the IQ time series (computed pulse-to-pulse for each range gate):

$$SQI = \frac{\left| \sum_{m=0}^{M-2} x(m) x^*(m+1) \right|}{\sum_{m=0}^{M-1} |x(m)|^2}$$

In Python / NumPy:
```python
num = np.abs(np.mean(iq_data[:, :-1] * np.conj(iq_data[:, 1:]), axis=1))
den = np.abs(np.mean(iq_data * np.conj(iq_data), axis=1))
sqi = np.where(den == 0, 0.0, num / den)
```

- **Values Range:** $0 \le SQI \le 1$.
- **Interpretation:** 
  - $SQI \approx 1$: High coherence / pure tone / stationary or slow-moving target (e.g., ground clutter or strong weather echo).
  - $SQI \approx 0$: Incoherent / white noise or weak signal.

---

## 2. Naming Conventions in Literature & Terminology
While the acronym **SQI** was popularized in modern digital signal processors (such as commercial Sigmet/Vaisala RVP processors and meteorological radar packages like Py-ART and NCAR/LROSE), older classic literature (such as *Doppler Radar and Weather Observations* by Doviak & Zrnic) refers to the underlying mathematical quantity by several other names:

* **Normalized Lag-1 Complex Autocorrelation Coefficient Magnitude** ($|\rho_v(1)|$)
* **Pulse-Pair Coherence** or **Temporal Coherence**
* **Lag-1 Correlation Coefficient**
* **Normalized Coherent Power**

---

## 3. Placement in the Signal Processing Pipeline: Before or After Clutter Filtering?
In Doppler radar architectures, SQI is computed **before** applying ground clutter filters (such as MTI or IIR notch filters).

* **Why before?**
  1. **Clutter Discrimination:** Ground clutter is highly coherent and stationary (yielding high SQI and near-zero velocity). Computing SQI on raw IQ data helps classifiers distinguish between noise, weather, and ground clutter.
  2. **Filter Preservation:** Ground clutter filters alter phase and amplitude relationships (frequency response modification). Filtering first would distort the intrinsic pulse-to-pulse coherence of the received signal.
  3. **Adaptive Processing:** SQI is often used in fuzzy-logic or thresholding schemes to decide whether a ground clutter filter *needs* to be applied to a specific range gate.
