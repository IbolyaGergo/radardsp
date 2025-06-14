# Agent Behavioral Guidelines

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
