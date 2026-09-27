# FACS Analysis Workstation

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A desktop workstation for analyzing facial action units from video and webcam using Py-Feat Detectorv2. Designed for research and technical experimentation in facial expression measurement, temporal dynamics, and measurement quality assessment.

**Important:** This tool analyzes visible facial actions detected by a neural network. It does not establish identity, detect deception, classify genuine vs. posed expressions without validation, or provide access to mental states.

## Features

**Detection**
- 20 Facial Action Units (FACS coding system)
- 7 emotion probabilities with ambiguity detection
- Valence and arousal scores
- Head pose (pitch, roll, yaw) with quality assessment
- Gaze direction tracked separately from expression
- 68-point 2D landmarks and 478-point 3D face mesh
- Blendshape values when available
- Multi-face tracking with visual continuity (not biometric identity)

**Temporal Analysis**
- AU onset, peak, and offset detection with duration and amplitude
- Rise and fall rates with validity flags for short intervals
- Hysteresis-based activation to reduce jitter
- Coactivation analysis (which AUs occur together)
- Directed transitions between AUs
- Temporal sequences and local activity clusters
- Per-detection measurement quality scores

**Calibration & Interpretation**
- Neutral reference profiles from stable frames (median/MAD based)
- Pose-aware measurement quality without treating pose as failure
- Emotion ambiguity indicators instead of single-label classification
- Robust rolling baselines for normalization
- Signal smoothing (exponential and Savitzky-Golay)

**Review & Export**
- Post-analysis playback with timeline and frame stepping
- Event-based navigation
- Multiple formats: CSV, JSON, and master `analysis.json`
- Session metadata including detector version and parameters
- Built-in deterministic validation tests

## Requirements

- Python 3.11+
- NVIDIA GPU recommended (CPU fallback available)
- ~2 GB disk space for Py-Feat models (downloaded on first use)

## Installation

Clone the repository:

```bash
git clone https://github.com/testcom314/FACS.git
cd FACS
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

**For CPU:**

```bash
pip install -r requirements-cpu.txt
```

**For CUDA 13.0 (NVIDIA GPU):**

```bash
pip install -r requirements-cuda130.txt
```

Py-Feat will download pretrained models (~500 MB) on first run. Internet connection is required for initial setup.

## Quick Start

Launch the application:

```bash
python run.py
```

The main window provides these tabs:

- Quick View: Summary for current track
- Overview: Emotion probabilities, state, events
- FACS: All 20 AU values
- Face / Mesh: Landmarks and 3D geometry
- Blendshapes: Blendshape outputs
- Raw / Events: Detector output and events
- Graphs: Valence/arousal history
- Temporal Review: Episode and sequence browser
- Settings: Thresholds and options

### Basic Workflow

1. Select a source (webcam or video file).
2. Configure settings if needed. The defaults work for most cases.
3. Click **Start analysis**.
4. Results are saved to `facs_output/session_YYYY-MM-DD_HH-MM-SS/`.
5. Open a completed session for playback and review.

### Configuration

Key settings are exposed in the Settings tab:

- Face detection threshold: `0.30` (default)
- AU activation threshold: `0.15` (default)
- Detection interval: `1` (process every frame)
- Smoothing factor: `0.35` (exponential smoothing)

See [CONFIGURATION.md](CONFIGURATION.md) for the complete parameter list.

## Session Output

Each analysis creates a timestamped directory:

```text
facs_output/session_YYYY-MM-DD_HH-MM-SS/
├── original.mp4              # Original video
├── tracked.mp4               # Annotated video with overlays
├── data.csv                  # Per-frame detector records
├── data.json                 # Per-frame records in JSON format
├── events.csv                # AU activations and rapid changes
├── episodes.csv              # Temporal AU events with rates
├── sequences.csv             # Clustered AU activity periods
├── coactivation.csv          # AU co-occurrence statistics
├── transitions.csv           # Directed AU transitions
├── summary.json              # Session-level statistics
├── temporal_summary.json     # Per-track temporal statistics
├── session_info.json         # Configuration and metadata
└── analysis.json             # Complete analysis in one file
```

The primary file for analysis is `analysis.json`. It combines session information, frame data, events, and temporal analysis. Individual files are retained for inspection and compatibility.

## Documentation

- [THEORY_AND_METHODS.md](THEORY_AND_METHODS.md) — What is measured and why; AU interpretation; measurement model
- [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) — Implementation details, processing pipeline, and architecture
- [DATA_FORMAT.md](DATA_FORMAT.md) — CSV/JSON field reference and schema
- [CONFIGURATION.md](CONFIGURATION.md) — Runtime parameters and tuning guide
- [VALIDATION.md](VALIDATION.md) — Internal tests and external validation strategy
- [DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md) — Design decisions and project evolution
- [RESEARCH_NOTES.md](RESEARCH_NOTES.md) — Literature and references

## Measurement Model

The application separates measurements into layers:

**Raw Layer:** Detector outputs including 20 AUs, 7 emotion probabilities, valence/arousal, pose, gaze, landmarks, mesh, and available blendshapes.

**Quality Layer:** Measurement conditions including face size, detector confidence, pose, gaze, visibility, and tracking state. Pose can affect AU estimate reliability, while gaze is recorded separately.

**Temporal Layer:** AU episodes with onset, peak, offset, duration, amplitude, and rate information. Same-frame activations are grouped, and rates are marked invalid when the available interval is too short.

**Derived Layer:** Smoothed values, normalized deviations, calibration/reference measurements, and emotion ambiguity indicators.

Raw detector values are preserved. Later processing layers do not overwrite them.

## Limitations

- Detector accuracy inherits the limitations of the Py-Feat model.
- Pose, lighting, occlusion, and face visibility can degrade measurement quality.
- Tracking is visual continuity tracking, not biometric identity. Track IDs represent continuity within an analysis session.
- The seven emotion categories cannot represent every possible facial configuration.
- Standard 30 FPS video provides relatively limited temporal resolution for microexpression research. High-speed datasets commonly use substantially higher frame rates.
- Coactivation counts depend on recording duration, thresholds, and recording conditions.
- No external validation against human FACS annotations is currently included.
- The system cannot detect deception, determine whether an expression is genuine or posed without appropriate validation data, or establish a person's mental state.

See [THEORY_AND_METHODS.md](THEORY_AND_METHODS.md) for a detailed discussion of these limitations.

## Validation

Internal tests verify application-level behavior including:

- Tracking stability
- Episode detection
- Hysteresis logic
- Same-frame AU grouping
- Temporal sequence separation
- Signal smoothing
- Rate handling
- Data-processing consistency

Run the tests through **Settings > Run internal validation tests**.

These tests validate the application's processing logic. They do not establish the accuracy of the underlying Py-Feat detector.

External validation requires independently annotated datasets. Examples include DISFA for spontaneous facial action measurements and CASME II/SAMM for high-speed microexpression research.

## Architecture

- `FACS_Level2.py` — GUI, detector adapter, tracking, temporal analysis, calibration, measurement quality, export, and playback
- `run.py` — Application entry point
- Single-module design currently prioritizes experimental iteration over extensive refactoring

Stack:

- PySide6 — GUI
- PyTorch — model inference
- Py-Feat — facial analysis
- NumPy / Pandas — numerical and tabular data
- OpenCV — video processing
- SciPy — signal processing

## Contributing

Improvements are welcome. Current development areas include:

- Split the monolithic module into focused submodules
- Add a dedicated pytest test suite for tracking and temporal algorithms
- Pin dependency versions
- Add a headless CLI and validation mode
- External validation against independently annotated datasets such as DISFA
- Improve cross-video and cross-subject evaluation

Contributions should preserve the distinction between raw detector measurements and derived analysis.

## License

MIT License

Copyright (c) 2026 Adithya S

See [LICENSE](LICENSE) for the complete license text.

## Citation

Built on:

- [Py-Feat](https://py-feat.org) — Multi-task facial analysis
- Facial Action Coding System (Ekman & Friesen) — AU taxonomy
- PyTorch
- OpenCV
- NumPy
- Pandas
- SciPy

Research informing the project includes:

- Cross, Acevedo, Hunter (2023) — Automated facial coding limitations
- DISFA — Spontaneous facial action database
- CASME II / SAMM — Microexpression datasets
