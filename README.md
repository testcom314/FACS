# FACS Analysis Workstation

A desktop application for frame-by-frame analysis of facial action units in recorded video and from a webcam. The application keeps the detector output as the primary dataset and adds temporal analysis, tracking, measurement-quality information, calibration, and review tools around it.

The project is intended for technical experimentation, FACS-oriented analysis, and inspection of facial movement over time. It is not a lie detector and it does not establish a person's internal emotional state.

## Main capabilities

- Py-Feat Detectorv2 inference on CPU or CUDA
- 20 facial action units
- seven detector emotion probabilities
- valence and arousal
- head pose and gaze
- 68-point landmarks
- 478-point 3D face mesh
- blendshape output when supplied by the detector
- multiple-face detection and visual track IDs
- frame-level smoothing and temporal AU analysis
- AU onset, apex/peak, offset, duration and amplitude
- rate-of-change and validity flags
- AU coactivation and directed transitions
- temporal clusters and sequences
- measurement quality and pose-aware quality checks
- neutral/reference calibration using median and MAD statistics
- emotion ambiguity information rather than a single unconditional label
- recorded-video playback with timeline and temporal review
- CSV and JSON exports, including a master `analysis.json`
- internal validation tests

## Requirements

Python 3.11 or newer is recommended. The project has separate dependency files for CPU and CUDA 13.0 installations.

For a CUDA 13.0 installation:

```text
pip install -r requirements-cuda130.txt
```

For a CPU installation:

```text
pip install -r requirements-cpu.txt
```

Py-Feat downloads its pretrained model files on first use and caches them locally. An internet connection is therefore normally required for the first detector initialization.

## Running

```text
python run.py
```

The application opens the main workstation window. A recorded video can be selected from the file controls, or the webcam mode can be used for live analysis.

Results are written under `facs_output/`, with a separate directory for each analysis session.

## Session files

A normal completed session contains:

```text
session_YYYY-MM-DD_HH-MM-SS/
├── original.mp4
├── tracked.mp4
├── data.csv
├── data.json
├── events.csv
├── episodes.csv
├── sequences.csv
├── coactivation.csv
├── transitions.csv
├── summary.json
├── temporal_summary.json
├── session_info.json
└── analysis.json
```

The individual files are kept for inspection and compatibility. `analysis.json` combines the main session information, frame records, events and temporal analysis in one file.

## Project layout

- `FACS_Level2.py` contains the application and analysis pipeline.
- `run.py` is the entry point.
- `requirements*.txt` describe the supported dependency sets.
- `TECHNICAL_DOCUMENTATION.md` describes the implementation.
- `THEORY_AND_METHODS.md` describes the measurement model and interpretation rules.
- `DATA_FORMAT.md` documents exported fields and file relationships.
- `VALIDATION.md` describes internal tests and external validation plans.
- `DEVELOPMENT_HISTORY.md` records the decisions and changes that shaped the current version.
- `RESEARCH_NOTES.md` lists the external literature and the conclusions used when designing the temporal and quality layers.

## Interpretation

The detector's AU values are measurements produced by a trained model. The temporal layer describes changes in those measurements. Neither layer should be treated as a direct measurement of private mental state. Head rotation, occlusion, lighting, tracking errors and other recording conditions can affect the result.

The application therefore keeps raw detector values separate from smoothed values and derived values. Calibration changes derived reference values only; it does not rewrite detector probabilities.
