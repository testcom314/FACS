# FACS Analysis Workstation

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
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

Py-Feat will download pretrained models (~500 MB) on first run. Internet connection required for initial setup.

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

1. Select source (webcam or video file)
2. Configure settings if needed (defaults work for most cases)
3. Click "Start analysis"
4. Results saved to `facs_output/session_YYYY-MM-DD_HH-MM-SS/`
5. Open session for playback and review

### Configuration

Key settings are exposed in the Settings tab:
- Face detection threshold: 0.30 (default)
- AU activation threshold: 0.15 (default)
- Detection interval: 1 (process every frame)
- Smoothing factor: 0.35 (exponential smoothing)

See [CONFIGURATION.md](CONFIGURATION.md) for complete parameter list.

## Session Output

Each analysis creates a timestamped directory:

```
facs_output/session_YYYY-MM-DD_HH-MM-SS/
├── original.mp4              # Original video
├── tracked.mp4               # Annotated video with overlays
├── data.csv                  # Per-frame detector records
├── data.json                 # Same as CSV, JSON format
├── events.csv                # AU activations and rapid changes
├── episodes.csv              # Temporal AU events with rates
├── sequences.csv             # Clustered AU activity periods
├── coactivation.csv          # AU co-occurrence statistics
├── transitions.csv           # Directed AU pairs
├── summary.json              # Session-level statistics
├── temporal_summary.json     # Per-track temporal stats
├── session_info.json         # Configuration and metadata
└── analysis.json             # Complete analysis in one file
```

The primary file for analysis is `analysis.json`, which combines session info, frame data, events, and temporal analysis. Individual files are kept for inspection and compatibility.

## Documentation

- [THEORY_AND_METHODS.md](THEORY_AND_METHODS.md) — What is measured and why; AU interpretation; measurement model
- [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) — Implementation details, processing pipeline, architecture
- [DATA_FORMAT.md](DATA_FORMAT.md) — CSV/JSON field reference and schema
- [CONFIGURATION.md](CONFIGURATION.md) — Runtime parameters and tuning guide
- [VALIDATION.md](VALIDATION.md) — Internal tests and external validation strategy
- [DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md) — Design decisions and evolution
- [RESEARCH_NOTES.md](RESEARCH_NOTES.md) — Literature and references

## Measurement Model

The application separates measurements into layers:

**Raw Layer:** Detector outputs (20 AUs, 7 emotion probabilities, valence/arousal, pose, gaze, landmarks, mesh).

**Quality Layer:** Measurement conditions (face size, pose, confidence, gaze, tracking state). Pose affects AU estimate reliability; gaze is recorded separately.

**Temporal Layer:** AU episodes with onset, peak, offset, duration, amplitude, and rates. Same-frame activations are grouped; rates marked invalid for short intervals.

**Derived Layer:** Smoothed values, normalized deviations, calibration adjustments, emotion ambiguity indicators.

Raw detector values are always preserved. Later layers do not overwrite them.

## Limitations

- Detector accuracy inherits Py-Feat model limitations
- Pose, lighting, and occlusion can degrade measurement quality
- Visual tracking, not biometric identity (Track IDs represent continuity)
- 7 emotion categories cannot represent every facial configuration
- 30 FPS video is insufficient for microexpression research (requires 120+ FPS)
- Coactivation counts are recording-dependent (duration, threshold, conditions)
- No external validation against human FACS annotations
- Cannot detect deception, classify genuine vs. posed expressions without validation, or establish mental states

See [THEORY_AND_METHODS.md](THEORY_AND_METHODS.md) for detailed discussion.

## Validation

Internal tests verify tracking stability, episode detection, hysteresis logic, and smoothing correctness. Run via **Settings > Run internal validation tests**.

External validation requires independently annotated datasets (DISFA for spontaneous AU intensity, CASME II/SAMM for high-speed microexpressions).

## Architecture

- `FACS_Level2.py` — GUI, detector adapter, tracking, temporal analysis, calibration, export, playback
- `run.py` — Entry point
- Single-module design prioritizes experimental iteration over refactoring

Stack: PySide6 (GUI), PyTorch (inference), NumPy/Pandas (data), OpenCV (video), SciPy (signal processing).

## Contributing

Improvements welcome. Priority areas:
- Split monolithic module into focused submodules
- Add pytest test suite for tracking and temporal algorithms
- Pin dependency versions
- Add headless CLI and validation mode
- External validation against DISFA

## License

MIT License — See [LICENSE](LICENSE)

## Citation

Built on:
- Py-Feat (https://py-feat.org) — Multi-task face detection
- Facial Action Coding System (Ekman & Friesen) — AU taxonomy
- PyTorch, OpenCV, NumPy, Pandas, SciPy

Research informed by:
- Cross, Acevedo, Hunter (2023) — Automated facial coding limitations
- DISFA — Spontaneous facial action database
- CASME II / SAMM — Microexpression datasets
