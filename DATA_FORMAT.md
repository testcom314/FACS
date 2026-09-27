# Data Format

## 1. Session files

### `analysis.json`

Master analysis file containing:

```text
session_info
summary
frame_data
events
temporal_analysis
```

It is the easiest file to archive because it contains the complete session in one JSON document.

### `data.csv`

One row per detected face record. Contains raw detector outputs and derived per-frame fields.

### `data.json`

JSON representation of the same frame records. Non-finite numeric values are converted to `null` during JSON serialization.

### `events.csv`

Discrete event records such as AU activation/deactivation, rapid changes, short-burst candidates and derived-expression events.

### `episodes.csv`

Temporal AU episodes with onset, peak and offset information.

### `sequences.csv`

Groups of temporally related AU activity.

### `coactivation.csv`

Pairwise AU overlap statistics.

### `transitions.csv`

Directed AU transition statistics.

### `temporal_summary.json`

Temporal parameters, per-person statistics, episodes, sequences and interaction results.

### `summary.json`

Compact session-level summary.

### `session_info.json`

Run configuration and environment information, including detector version, device, tracking settings and temporal parameters.

## 2. Per-frame identity and timing

| Field | Meaning |
|---|---|
| `frame_number` | Sequential frame number used by the analysis |
| `timestamp` | Video-relative timestamp in seconds |
| `face_id` | Local face/track identifier in the record |
| `person_id` | Local tracking ID, not biometric identity |
| `face_x`, `face_y` | Face bounding-box position |
| `face_width`, `face_height` | Face bounding-box dimensions |
| `face_confidence` | Detector face confidence/output |

## 3. Action Units

The 20 AU fields are continuous detector outputs.

| Field | Description |
|---|---|
| `AU01` | Inner Brow Raiser |
| `AU02` | Outer Brow Raiser |
| `AU04` | Brow Lowerer |
| `AU05` | Upper Lid Raiser |
| `AU06` | Cheek Raiser |
| `AU07` | Lid Tightener |
| `AU09` | Nose Wrinkler |
| `AU10` | Upper Lip Raiser |
| `AU11` | Nasolabial Deepener |
| `AU12` | Lip Corner Puller |
| `AU14` | Dimpler |
| `AU15` | Lip Corner Depressor |
| `AU17` | Chin Raiser |
| `AU20` | Lip Stretcher |
| `AU23` | Lip Tightener |
| `AU24` | Lip Pressor |
| `AU25` | Lips Part |
| `AU26` | Jaw Drop |
| `AU28` | Lip Suck |
| `AU43` | Eyes Closed |

Derived AU fields include smoothed values, rolling baseline values and deviations where the corresponding processing stage produces them.

A field such as `AU12 = 0.67` must not be rewritten in documentation as “67% confidence” unless the detector documentation explicitly defines that output as a probability. In this project it is safer to call it an AU model output.

## 4. Emotion fields

The detector produces:

```text
emotion_neutral
emotion_happy
emotion_sad
emotion_surprise
emotion_fear
emotion_disgust
emotion_anger
```

The application additionally calculates:

```text
emotion_top
emotion_top_probability
emotion_second
emotion_second_probability
emotion_margin
emotion_entropy
emotion_interpretation_score
emotion_interpretation_status
emotion_top3
```

The top label is not treated as a statement about the person's internal state.

Example:

```json
{
  "emotion_happy": 0.45,
  "emotion_surprise": 0.30,
  "emotion_fear": 0.15,
  "emotion_top": "Happy",
  "emotion_second": "Surprise",
  "emotion_margin": 0.15
}
```

## 5. Pose and gaze

The raw detector fields include:

```text
Pitch
Roll
Yaw
X
Y
Z
gaze_pitch
gaze_yaw
gaze_angle
```

Derived fields include:

```text
pose_pitch_deg
pose_yaw_deg
pose_roll_deg
pose_magnitude_deg
pose_quality
gaze_pitch_deg
gaze_yaw_deg
gaze_angle_deg
gaze_context
```

## 6. Measurement quality

The following fields describe observability conditions:

```text
face_size_quality
measurement_quality_score
measurement_quality
pose_quality
au_quality_AU01 ... au_quality_AU43
```

`measurement_quality_score` is an application heuristic, not a measured detector accuracy percentage.

## 7. Landmarks, mesh and blendshapes

The frame record stores 68 two-dimensional landmark pairs:

```text
x_0 ... x_67
y_0 ... y_67
```

Detectorv2 mesh output is stored as:

```text
mesh_x_0 ... mesh_x_477
mesh_y_0 ... mesh_y_477
mesh_z_0 ... mesh_z_477
```

Blendshape-like columns returned by the detector are copied into fields beginning with `blendshape_`.

## 8. Temporal fields

Depending on the processing stage, frame records can contain fields describing:

```text
temporal_active_aus
temporal_active_au_count
temporal_strongest_au
temporal_fastest_au
temporal_fastest_rate
```

These are derived features and should not be confused with raw detector output.

## 9. Calibration fields

When a calibration profile is active, the application can record reference-based AU deviations and robust standardized values.

The important distinction is:

```text
raw AU value        = detector output
calibration value   = deviation from a reference condition
```

Calibration does not modify the raw field.

## 10. Episode record

A typical episode contains fields such as:

```json
{
  "episode_id": 1,
  "person_id": 1,
  "au": "AU12",
  "onset_time": 0.5,
  "onset_frame": 15,
  "peak_time": 1.1,
  "peak_frame": 33,
  "offset_time": 1.8,
  "offset_frame": 54,
  "baseline": 0.05,
  "peak": 0.67,
  "amplitude": 0.62,
  "duration": 1.3,
  "rise_rate": 0.477,
  "fall_rate": 0.477,
  "rise_rate_valid": true,
  "fall_rate_valid": true,
  "single_frame_peak": false,
  "valid": true
}
```

The exact set of fields can change with the temporal-analysis version. `temporal_analysis_version` in session metadata should therefore be recorded when comparing datasets.

## 11. Rate fields

`rise_rate` and `fall_rate` are intentionally nullable.

If the onset-to-peak or peak-to-offset interval is shorter than the configured minimum rate interval, the corresponding rate is not considered valid.

This is preferable to reporting an enormous numerical value generated by division by a very small time interval.

## 12. Session metadata

The current implementation records items including:

```text
schema_version
 temporal_analysis_version
py_feat_version
device
gpu
source
face_detection_threshold
smoothing
detection_interval
tracking settings
identity_recognition
calibration state
pose/gaze quality limits
minimum episode duration
minimum episode amplitude
activation hysteresis
minimum rate interval
```

## 13. Example master structure

```json
{
  "session_info": {},
  "summary": {},
  "frame_data": [],
  "events": [],
  "temporal_analysis": {
    "parameters": {},
    "per_person": {},
    "episodes": [],
    "sequences": [],
    "interactions": {
      "coactivation": [],
      "transitions": [],
      "activation_groups": []
    }
  }
}
```
