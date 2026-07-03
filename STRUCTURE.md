# Project Structure

- `data/`
  - `raw/`: Raw input data (e.g., `.txt` files).
  - `converted/`: Processed data in format suitable for further analysis (e.g., `.npz`).
- `filters/`: Human-readable filter coefficients (TOML files).
- `scripts/`: Automation scripts for running the processing pipeline.
- `src/radarsig/`: Main source code.
  - `__init__.py`: Package initialization.
  - `filters.py`: `Filter` dataclass with TOML loading capabilities.
  - `parsers.py`: Parse hex data from `*.txt` files.
  - `processing.py`: Fluent `SignalPipeline` class for chained signal processing.
  - `io.py`: I/O for reading `*.txt` and parsed `*.npz`.
- `test/`: Test suite using `pytest`.

## Fluent Pipeline Architecture
The pipeline uses a chained interface to manage state and history:

```python
pipeline = SignalPipeline(filters=my_filters, fs=250e6)
pipeline.load_data(raw_data) \
        .apply_filter("bp_1", key="step1") \
        .downsample(2, key="step2")
```
Intermediate results are stored in `pipeline.history`.

## Pipeline Execution Flow
The pipeline follows this sequence (based on `Processing.m`):
1. `load_data` (raw input, fs=250MHz)
2. `apply_filter` (BP Filter)
3. `downsample` (Factor=2, fs=125MHz)
4. `mix` (Frequency=...)
5. `apply_filter` (LP Filter - *TO BE IMPLEMENTED*)
6. `downsample` (Factor=16 - *TO BE IMPLEMENTED*)
