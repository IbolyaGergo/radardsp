# Agent Behavioral Guidelines

Read *.md

## Communication & Process
- **Collaborative Flow**: Always suggest and discuss proposed changes. Never modify files without explicit user confirmation.
- **Incrementalism**: Avoid "big bang" refactoring. Prefer small, verifiable steps.
- **Separation of Concerns**: Enforce strict boundaries between logic and core data processing/simulation logic.

## Technical Preferences
- **Dependencies**: Minimize additional dependencies. Pandas is permitted for data handling, but avoid unnecessary overhead.
- **CLI Parsing**: Standardize on `argparse`.
- **Refactoring Goal**: Focus on improving user experience (UX) and testability.
- **Maintainability**: Prioritize readable, modular code over clever optimizations.

## Execution
- **Validation**: When proposing changes, outline how they can be verified.

## Development Methodology
- **Incrementalism**: Always decompose tasks into atomic, verifiable steps. Discuss and verify one step at a time.
- **TDD (Test-Driven Development) LIGHT**: Prioritize writing tests for the expected behavior (even simple mock tests), however, not strictly before implementing functions as in classic TDD. We write a text when our ideas are developed enough and we want to fix a behaviour.
- **Readable over Clever**: Prioritize code clarity. Do not abstract code (e.g., creating private helpers) unless it clearly improves maintainability. If the existing code is readable, keep it direct.
- **Naming Conventions**: Use distinct, typo-proof naming for functions based on their scope (e.g., `load_pulse_...` for a single file vs. `load_pulseset_...` for a collection/dataset).
- **Pragmatic Testing**: Avoid over-engineering tests. Prefer simple, hermetic tests using mock data or `tmp_path` over heavy mocking frameworks unless absolutely necessary.
