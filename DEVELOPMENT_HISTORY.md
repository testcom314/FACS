# Development History

## Initial application

The project started as a small Python/OpenCV/Py-Feat application for displaying facial action units and emotion predictions from a webcam or video. The early version had a short AU list, basic face detection and a video overlay.

## Detector and CUDA work

The project moved to Py-Feat Detectorv2 so the main detection pass could provide AUs, emotion probabilities, valence/arousal, pose, gaze, landmarks, mesh and blendshapes from one multitask model.

The development machine uses an RTX 5060 Laptop GPU. The original PyTorch CUDA build did not include the GPU's `sm_120` architecture. The working setup was changed to the CUDA 13.0 PyTorch build. Device checks were added so an incompatible CUDA build falls back to CPU instead of failing later inside inference.

## Data expansion

The application was expanded from a small AU display into a full frame dataset. The stored data now includes all 20 Detectorv2 AUs, all seven emotion probabilities, valence/arousal, pose, gaze, 68 landmarks, 478 mesh coordinates and blendshapes.

## Level 2 temporal analysis

The next stage added AU onset/offset tracking, duration, peak intensity, rate of change, rapid-change events and short-burst candidates. This was motivated by the need to study facial movement rather than only frame averages.

An early implementation produced unrealistic rate values when onset and peak occurred on the same frame. The rate calculation was changed so the event is retained but the rate is marked invalid when there is not enough elapsed time.

## Review and output system

The application gained recorded-video playback, frame stepping, timeline navigation, person/track selection and a Temporal Review tab. Separate CSV/JSON files were kept while a master `analysis.json` was added for analysis scripts.

## Phase 1 and Phase 2 temporal work

The temporal system was then expanded with:

- hysteresis
- rolling baselines
- smoothed signals
- measurement-quality fields
- raw/measured/derived separation
- simultaneous AU grouping
- coactivation
- directed transitions
- local temporal sequences
- episode filtering
- timing and rate validity flags

An earlier sequence implementation could merge nearly an entire recording into one sequence. The grouping rules were tightened so local temporal clusters are separated by time gaps.

## Tracking changes

The detector's per-frame face order was not treated as a stable identity. A visual tracker was added using box overlap, predicted movement, distance and size constraints. Track IDs represent continuity through the video, not identity recognition.

## Pose, gaze and calibration

Testing showed that changing head direction can affect expression estimates. Instead of altering the detector output to compensate for pose, the project added a measurement-condition layer. Pose contributes to quality, gaze is kept separately, and raw measurements remain untouched.

A reference calibration system was added using stable frames from a completed session. Median/MAD statistics provide a robust neutral reference for derived deviations. Calibration is not retraining.

## Emotion display changes

The detector can return mixed emotion probabilities. The interface therefore stopped treating the largest probability as an unquestionable answer. It now exposes ambiguity and retains the complete probability distribution.

## Reliability fixes

Several failures during development came from state being available in the worker but not in the main window, missing episode fields such as `offset_time`, a temporal interaction variable being used before assignment, and a missing `FEAT_VERSION` definition. These were fixed in the source rather than hidden with broad exception handling.

The latest source also initializes frame dimensions before quality calculations. This fixes the `FACSWorker` `frame_width` error that occurred after the calibration/quality changes.

## Current state

The project is a working FACS-oriented analysis workstation prototype with a substantially larger temporal and review layer than the original viewer. The raw detector data path is the most established part. Temporal episode detection, tracking, sequence segmentation and scientific validation remain areas where external annotated data are needed.
