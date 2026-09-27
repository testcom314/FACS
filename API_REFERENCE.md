# Programmatic Reference

The GUI is the primary interface, but the processing code exposes several useful classes and methods.

## FACSWorker

```python
FACSWorker(source, is_webcam, settings)
```

`FACSWorker` is a `QThread` subclass that manages video capture, detector inference, tracking, temporal analysis and output.

Important attributes include:

```text
raw_rows
event_rows
temporal_episodes
temporal_sequences
temporal_interactions
person_temporal_stats
tracker
session_dir
frame_number
video_duration_seconds
```

Important methods include:

```text
detect_frame(rgb)
face_record(row, person_id)
measurement_quality(rec)
emotion_quality(rec)
apply_calibration(rec)
temporal_update(rec)
build_temporal_interactions()
write_outputs()
```

Because `FACSWorker` inherits from `QThread`, it should normally be used through the application's Qt event loop rather than treated as an ordinary synchronous Python class.

## MultiFaceTracker

`MultiFaceTracker` maintains local track IDs using bounding-box geometry and motion prediction.

It is not a face-recognition system.

The tracker report is available from the tracker object and is also included in session summaries.

## SignalSmoother

`SignalSmoother` maintains exponentially smoothed values for named signals.

Conceptually:

```text
new = alpha * value + (1 - alpha) * previous
```

The raw signal should remain available for analysis.

## Internal validation

```python
from FACS_Level2 import run_internal_validation

report = run_internal_validation()
print(report)
```

The validation routine is deterministic and does not require a video file.

## Constants

Important exported module-level configuration includes:

```text
AU_NAMES
EMOTION_NAMES
FACE_DETECTION_THRESHOLD
DETECTION_INTERVAL
DEFAULT_SMOOTHING
MIN_EPISODE_DURATION
MIN_EPISODE_AMPLITUDE
ACTIVATION_HYSTERESIS
BASELINE_WINDOW
BASELINE_MIN_SAMPLES
MAX_CLUSTER_GAP
MAX_SEQUENCE_GAP
MIN_TRANSITION_DELAY
```

These are implementation parameters. Scripts that depend on them should record their values rather than assuming they will never change.

## Data access

For most analysis scripts, loading `analysis.json` is preferable to importing the GUI classes. It provides a stable session-level representation and avoids starting a Qt application just to read results.
