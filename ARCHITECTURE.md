# Technical Architecture & Design Patterns

## 1. Core Architectural Principle: "Plain Data, Powerful Functions"
*   **Data Structures**: Use simple, "dumb" dataclasses (e.g., `PulseData`) as containers. They should not contain complex business logic or perform I/O.
*   **Processing Functions**: Logic resides in pure, modular functions that accept `np.ndarray` and return `np.ndarray`.
*   **Separation of Concerns**: Avoid deep class hierarchies (e.g., avoid a generic `Signal` base class). Favor composition and simple, predictable transformations.

## 2. Data Handling & Ingestion
*   **Storage**: Utilize `.npz` for fast, efficient, and native-Python data storage.
*   **Robustness**: Perform explicit loading (e.g., `.copy()`) within loading functions to ensure data is copied into memory before file streams are closed.

## 4. I/O Design Pattern: Primitive vs. Orchestrator
To maintain consistency and testability, all I/O functions must follow a two-layer structure:

1. **Primitive Loaders (`load_pulse_from_<format>`)**:
    - **Scope**: Handles exactly one file.
    - **Responsibility**: Opening the file, parsing content, validating schema (`n_samples`), and returning the data.
    - **Goal**: Highly testable atomic units.

2. **Orchestrator Loaders (`load_pulseset_from_<format>`)**:
    - **Scope**: Handles a directory/dataset.
    - **Responsibility**: Globbing, sorting, filtering, and invoking the primitive loaders for each file.
    - **Goal**: High-level data management.

## 5. Aggregation Strategy
- **Centralized Aggregation**: Use a consistent helper (e.g., `_aggregate_pulse_data`) to restructure lists of dictionaries into the final concatenated format. This ensures identical output structures across different input formats (NPZ vs. TXT).
