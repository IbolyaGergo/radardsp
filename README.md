# Radar Signal Processing Pipeline

A modular, reproducible, and tinker-friendly signal processing framework for radar pulse data.

## Current Pipeline Status
- **Implemented Stages**: Load Data -> Apply Filter -> Downsample -> Mix.
- **Pending Stages**: LP Filter -> Second Downsampling -> IQ Demodulation integration.
- **Immediate Task**: Resolve the order-of-operations (ensure filtering happens at the correct sample rate) and verify complex data handling.

## Documentation
- [Project Structure](STRUCTURE.md)
- [Architecture & Design](ARCHITECTURE.md)
