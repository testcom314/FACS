# Technical Documentation

## 1. System overview

The application is a PySide6 desktop workstation around Py-Feat Detectorv2. It processes webcam frames or prerecorded video, stores the detector outputs, maintains local face tracks, derives temporal features and writes a complete analysis session.

The main implementation is currently concentrated in `FACS_Level2.py`. This is deliberate at the prototype stage: the processing path is easier to inspect while the analysis methods are still changing.

## 2. Processing pipeline

```text
Input frame
    |
    v
RGB conversion / tensor preparation
    |
    v
Py-Feat Detectorv2
    |
    +--> 20 AUs
    +--> 7 emotion outputs
    +--> valence / arousal
    +--> head pose
    +--> gaze
    +--> 68 landmarks
    +--> 478-point mesh
    +--> blendshapes
    |
    v
Face records
    |
    v
MultiFaceTracker
    |
    v
Measurement-quality assessment
    |
    +--> pose / gaze context
    +--> face-size quality
    +--> AU-specific quality
    |
    v
Temporal processing
    |
    +--> smoothing
    +--> baselines
    +--> episodes
    +--> rapid changes
    +--> short bursts
    +--> simultaneous activations
    +--> coactivation
    +--> transitions
    +--> sequences
    |
    v
Optional neutral-reference calibration
    |
    v
Tracked video + CSV/JSON output
```

The raw detector measurements are kept alongside derived fields. Derived processing should never overwrite the original AU or emotion outputs.

## 3. Detector input

Detectorv2 receives an RGB tensor in BCHW layout. The frame is converted from NumPy HWC representation to a contiguous PyTorch tensor and a batch dimension is added.

The application retains a disk-image fallback because some Py-Feat/PyTorch image-loading paths can reject a NumPy array even though the logical input is an image. If the tensor route fails for the known image-input failure modes, the frame can be written temporarily and passed through an image-compatible path.

This fallback is recorded in the session summary as `fallback_disk_input`.

## 4. GPU selection

The application checks CUDA availability and device information before processing. The current workstation was developed around an NVIDIA RTX 5060 Laptop GPU using a PyTorch build with CUDA 13.0 support for the GPU's architecture.

The session records the selected device and GPU name. CPU fallback remains available.

GPU availability is not assumed to mean that a particular PyTorch wheel contains a kernel for the installed GPU architecture. That distinction caused an earlier failure and is why device selection is explicit.

## 5. Tracking algorithm

The tracker uses local geometric tracking rather than identity recognition.

Conceptually:

```python
predictions = [predict_box(track) for track in active_tracks]

costs = build_cost_matrix(
    predicted_boxes=predictions,
    detections=detections,
    iou_weight=..., 
    distance_weight=...,
    size_weight=...
)

matches = global_assignment(costs)

for track, detection in matches:
    track.update(detection)
    track.update_velocity()

for unmatched_track in unmatched_tracks:
    unmatched_track.missing_frames += 1

for unmatched_detection in unmatched_detections:
    create_track(unmatched_detection)
```

The actual implementation uses predicted bounding boxes, geometric gates, velocity smoothing and global assignment when SciPy is available.

Important tracker settings currently include:

```text
maximum missing frames: 45
minimum track hits: 2
velocity alpha: 0.35
minimum IoU gate: 0.02
maximum normalized distance gate: 1.15
maximum size-change gate: 0.65
```

These are engineering parameters, not universal tracking constants.

## 6. Episode detection

The temporal state is maintained independently for each local track.

A simplified version of the logic is:

```python
value = current_AU_value

if not active and value >= activation_threshold:
    start_episode()

elif active and value < deactivation_threshold:
    close_episode()

elif active:
    update_peak_if_needed()
```

The actual implementation also uses recent history, smoothed values, timing checks and minimum episode requirements.

When an episode closes:

```python
amplitude = peak - baseline
duration = offset_time - onset_time

rise_dt = peak_time - onset_time
fall_dt = offset_time - peak_time

if rise_dt >= MIN_RATE_INTERVAL_SECONDS:
    rise_rate = amplitude / rise_dt
else:
    rise_rate = None

if fall_dt >= MIN_RATE_INTERVAL_SECONDS:
    fall_rate = amplitude / fall_dt
else:
    fall_rate = None
```

This prevents single-frame peaks from generating meaningless numerical rates.

## 7. Signal processing

The application keeps raw AU values and creates smoothed values separately.

Exponential smoothing is used for live per-person values:

```text
smooth_t = alpha * current_t + (1 - alpha) * smooth_(t-1)
```

The default smoothing factor is 0.35.

The temporal pipeline can also use Savitzky-Golay filtering where SciPy is available. Peak detection support is included through SciPy's signal-processing functions.

Smoothing is analysis-dependent. It is not presented as a correction to the detector.

## 8. Measurement quality implementation

The current face-level quality calculation uses:

```text
visibility =
    0.45 * detector confidence
  + 0.25 * face-size quality
  + 0.20 * pose quality
  + 0.10 * roll quality
```

The result is classified as:

```text
GOOD       >= 0.80
FAIR       >= 0.60
LIMITED    >= 0.40
POOR       <  0.40
```

Pose quality uses soft and hard angular limits. Gaze is stored separately and produces a context category:

```text
CENTERED
AVERTED
STRONGLY_AVERTED
```

Eye-related AUs receive an AU-specific quality calculation that considers gaze. Mouth-related AUs are primarily affected by pose in the current model.

## 9. Emotion interpretation layer

The detector outputs are normalized before calculating secondary interpretation fields. The application then finds the highest and second-highest outputs and calculates their margin and entropy.

```python
margin = top - second
```

A small margin or high entropy produces `MIXED / AMBIGUOUS`.

The raw emotion values are never replaced by the interpretation score.

## 10. Calibration

A calibration profile represents a reference condition. It can be loaded from the application's calibration file and is included in session metadata when active.

The calibration layer calculates deviations from the reference and robust standardized values. It does not change the detector output.

A calibration profile should therefore be considered part of the experimental setup. Changing the reference changes the meaning of calibrated deviations.

## 11. Temporal interactions

The interaction layer produces three main structures:

```text
activation_groups
coactivation
transitions
```

`activation_groups` preserves activations that occur within the configured simultaneous-frame tolerance.

`coactivation` records pairs of AUs that are active together. The output includes overlap information and conditional measures where available.

`transitions` records directed AU activation relationships and timing information. A transition is an observed temporal relationship, not a causal relationship.

## 12. Sequences

Sequences are built from temporally related AU activity. The current configuration distinguishes local clusters from longer sequences and uses gaps between activity to avoid treating an entire recording as one expression.

The summary explicitly labels sequences as activity sequences rather than a single expression or emotion.

## 13. Error handling

Errors in frame processing are caught at the per-frame detection stage and reported through the GUI status signal. The rest of the session can continue when possible.

Model initialization and overall worker failures are handled separately. Output writers and video captures are released in the worker's final cleanup path.

Known detector input incompatibilities can trigger the disk-image fallback described above.

A failure in metadata or derived analysis should not be confused with a failure of GPU inference. During development, for example, an undefined `FEAT_VERSION` variable caused analysis output to fail after processing logic had already been reached. That was a metadata bug, not a detector failure.

## 14. Memory and storage

The main memory cost is not just the video. Every detected face becomes a Python dictionary containing AU values, emotions, pose, gaze, landmarks, mesh coordinates and blendshapes. The 478-point mesh alone contributes 1,434 numeric fields when x/y/z are all retained.

The application therefore stores large sessions in memory before writing the final analysis files. A long 1080p video with multiple faces can use substantially more RAM than the compressed source video size suggests.

The exact RAM requirement depends on face count, frame count and the amount of data returned by the installed Py-Feat version. A fixed “10 minute at 1080p = X GB” figure would be misleading because the source resolution is not the main driver of the stored record size.

For large experiments, splitting videos into shorter sessions is safer.

GPU memory is primarily determined by the loaded Py-Feat model, PyTorch runtime and temporary tensors. It should be measured on the actual machine rather than documented as a fixed number.

## 15. Output session

A completed session contains:

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
├── temporal_summary.json
├── summary.json
├── session_info.json
└── analysis.json
```

`analysis.json` is the master file. The other files are retained because they are easier to inspect, process and import independently.

## 16. Internal validation

The application contains deterministic local validation tests. They test the analysis machinery rather than claiming detector accuracy.

The current tests include:

- tracker ID stability under crossing/dropped detections
- no duplicate assignment within a frame
- synthetic episode detection
- sane rate limits
- prevention of a giant continuous sequence
- same-frame activation grouping
- smoothed AU values staying inside the detector domain
- separation of distant temporal sequences

The validation code uses the same temporal classes and functions used by normal analysis. It does not maintain a simplified second implementation just for tests.
