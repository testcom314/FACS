# Development History

## 1. Initial prototype

The project started as a small Py-Feat facial-analysis viewer. The first goal was simply to get facial Action Units onto the screen and understand what the detector actually returned.

The early version had a much smaller AU list, basic webcam/video input and simple output.

The main lesson from this stage was that the detector output was considerably richer than the initial GUI exposed. Throwing away the extra fields would have made later analysis impossible.

## 2. Detectorv2 and CUDA

The application moved to Py-Feat Detectorv2 so that AU, emotion, valence/arousal, pose, gaze, mesh and other outputs could be obtained from the same multitask model.

The development machine uses an RTX 5060 Laptop GPU. The first PyTorch CUDA build did not contain the required architecture support. Moving to the CUDA 13.0 PyTorch build resolved the GPU compatibility problem.

The project consequently records GPU/device information instead of assuming that CUDA availability alone guarantees compatibility.

## 3. Tensor input and fallback handling

Py-Feat's tensor input path initially exposed an image-loading incompatibility where a NumPy array reached a code path expecting an image filename or file-like object.

The application now prepares a BCHW tensor for Detectorv2 and retains a disk-image fallback for compatible image-input failures.

This failure also demonstrated why detector exceptions should not be hidden behind a generic “no face” result.

## 4. Full data extraction

The frame record was expanded to preserve:

- 20 AUs
- 7 emotion outputs
- valence
- arousal
- pose
- gaze
- 68 landmarks
- 478 mesh points
- blendshape fields
- face detection metadata

Raw output remains separate from derived analysis.

## 5. Temporal analysis

The next major change was moving from frame-by-frame display to temporal FACS analysis.

Episodes were added with onset, peak and offset information, followed by amplitude, duration and rate calculations.

Early versions produced absurd rates when the peak occurred in the same frame as onset. The current implementation explicitly marks these intervals as invalid instead of dividing by an almost-zero time difference.

## 6. Tracking

Face detection order is not a stable identity signal. A person appearing in slot 0 on one frame can appear in slot 1 on the next.

A local geometric tracker was therefore introduced. It predicts bounding-box movement, assigns detections globally and maintains track state through short detector gaps.

The IDs are intentionally called tracks rather than identities.

## 7. Interaction analysis

The project then added:

- simultaneous AU groups
- AU coactivation
- directed AU transitions
- temporal clusters
- activity sequences

An early sequence implementation could join almost an entire recording into one sequence. Synthetic tests were added specifically to catch this behavior.

## 8. Measurement quality

Testing prerecorded material showed that head orientation and gaze could change the reliability of facial measurements. The solution was not to invent a pose-correction formula. Instead, pose, gaze, face size and detector confidence were recorded as measurement conditions.

AU-specific quality fields were added so that eye-related measurements could be treated differently from mouth-related measurements.

## 9. Calibration

A neutral/reference calibration layer was added to provide subject/session-relative deviations. Calibration is kept separate from raw detector output and does not retrain or modify the detector.

## 10. Validation

The application gained deterministic synthetic tests for tracking and temporal logic. The tests are deliberately built on the same processing classes used by the application rather than maintaining a separate toy implementation.

External validation against human-coded datasets remains a separate research task.

## 11. Current architecture

The current prototype is still centered in a large Python file. This is not an ideal final architecture, but it keeps the experimental processing path visible while the algorithms are being developed.

The next major software-engineering step would be separating the detector interface, tracking, temporal analysis, storage and GUI into modules while keeping the same data schema.
