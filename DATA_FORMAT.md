# Data Format

## Session directory

Each completed run is stored in a timestamped directory under `facs_output/`.

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

## analysis.json

`analysis.json` is the main machine-readable entry point.

```json
{
  "session_info": {},
  "summary": {},
  "frame_data": [],
  "events": [],
  "temporal_analysis": {}
}
```

The file uses JSON `null` for values that cannot be represented as finite JSON numbers.

## data.csv / data.json

These contain the per-frame/per-face detector records. Depending on the installed detector version, the record can contain:

- frame number
- timestamp
- track/person ID
- face bounding box and confidence
- 20 AU values
- seven emotion probabilities
- valence
- arousal
- head pose
- gaze
- 68 landmark coordinates
- 478 mesh coordinates
- blendshape values
- smoothed AU values
- AU baseline and deviation fields
- measurement quality
- pose/gaze quality
- emotion interpretation fields

The raw detector values are the primary measurement layer. Derived fields are named separately.

## events.csv

Event records include AU activation/deactivation, rapid changes, short-burst candidates, derived expression events and other temporal notifications. Each event includes a timestamp and track where available.

A rapid-change event is not a microexpression label. It indicates that the measured AU value changed quickly enough to cross the configured event rule.

## episodes.csv

An episode is an AU event with a temporal extent. Typical fields include:

```text
person_id
AU
episode_id
onset_time
peak_time
offset_time
onset_frame
peak_frame
offset_frame
baseline
peak_value
amplitude
duration
rise_rate
fall_rate
rise_rate_valid
fall_rate_valid
single_frame_peak
measurement_quality
```

## sequences.csv

Contains temporal clusters built from nearby AU episodes. A sequence can contain several AUs and stores its time range, participating actions and episode count.

## coactivation.csv

Contains AU pairs observed as active at the same time within valid measurement frames.

## transitions.csv

Contains directed temporal AU pairs. Important fields include source AU, destination AU and transition delay. Same-frame activations are excluded from directional ordering.

## summary.json

Contains session-level counts and processing information, including source video duration, number of processed frames, face/track counts, event counts, detector version and timing information.

## temporal_summary.json

Contains temporal-analysis parameters and per-track statistics. This is useful when the full frame dataset is not required.

## session_info.json

Contains configuration and environment information relevant to reproducing a session, including schema version, temporal-analysis version, device information, Py-Feat version and calibration state.

## calibration_profile.json

Stored under `facs_output/` when a reference profile is created. It contains the profile schema, reference frame count and AU/pose/gaze reference statistics. It does not contain rewritten detector probabilities.

## Versioning

Current export schema: `4`.

Current temporal analysis version: `4.3`.

When fields or interpretation rules change, the schema/version fields should be updated rather than silently changing the meaning of an existing field.
