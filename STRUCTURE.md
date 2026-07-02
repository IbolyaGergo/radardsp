# Project Structure

- `data/`
  - `raw/`: Raw input data (e.g., `.txt` files).
  - `converted/`: Parsed data in format suitable for further analysis (e.g., `.npz`).
- `scripts/`: Automation scripts for running the processing pipeline.
- `src/radarsig/`: Main source code.
  - `__init__.py`: Package initialization.
  - `data.py`: Data structures containing pulse data.
  - `filters.py`: Filter data structure containing FIR/IIR filters.
  - `io.py`: I/O - load data from txt or npz.
  - `parsers.py`: Parsing raw txt files.
- `test/`: Test suite.
  - `data_mock/`: Mock data for unit and integration testing.

## Hybrid Approach Architecture
- **Functional Core**: Pure signal processing functions in `processing.py` using `scipy.signal` and `numpy`. These are easily testable.
- **Configurable Pipeline**: A class in `processing.py` that encapsulates the state and sequence of operations, holding filter coefficients and intermediate results.
