# Project Structure

- `data/`
  - `raw/`: Raw input data (e.g., `.txt` files).
  - `converted/`: Processed data in format suitable for further analysis (e.g., `.npz`).
- `filters/`: Human-readable filter coefficients (TOML files).
- `scripts/`: Automation scripts for converting files, analyzing measurement
  data.
- `src/radarsig/`: Main source code.
  - `__init__.py`: Package initialization.
  - `filters.py`: Utils for filters, e.g., load b, a coeffs from toml.
  - `parsers.py`: Parse hex data from `*.txt` files.
  - `processing.py`: Signal processing functions, downsampling, apply filters,
    compute Doppler moments...
  - `io.py`: I/O for reading `*.txt` and parsed `*.npz`.
- `test/`: Test suite using `pytest`.
