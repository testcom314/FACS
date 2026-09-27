# Technical Documentation

## 1. Purpose

The application is a local desktop workstation for extracting and reviewing facial action data from video. The core design goal is to retain the complete detector output while adding analysis that is useful for studying facial movement over time.

The project grew from a small Py-Feat/OpenCV viewer into a larger analysis tool. The current source is deliberately kept as a single main Python module so the processing path can be followed without jumping through a large package hierarchy.

## 2. Processing path

For each input frame the pipeline follows this order:

1. Read a frame from OpenCV.
2. Convert the frame to the representation expected by Py-Feat.
3. Run `Detectorv2` on the selected device.
4. Extract face boxes and the available per-face outputs.
5. Assign detections to visual tracks.
6. Store raw AU, emotion, valence/arousal, pose, gaze, landmark, mesh and blendshape values.
7. Calculate measurement-quality fields.
8. Apply an optional neutral/reference calibration to derived fields.
9. Update smoothed values and temporal AU state.
10. Record events, episodes, interactions and sequences.
11. Draw the review overlay and write the processed video.
12. Write the session data when processing finishes.

Raw detector values are not replaced by the later stages.

## 3. Py-Feat integration

The application uses `Detectorv2`, Py-Feat's multitask detector. Current Py-Feat documentation describes Detectorv2 as a single network that predicts 20 AUs, seven emotion classes, valence/arousal, gaze, head pose, 68-point landmarks, a 478-point 3D mesh and blendshapes. The exact available fields depend on the installed Py-Feat version.

The installed Py-Feat version is written into the session metadata. This matters because model outputs and mesh model versions can change between releases.

The optional identity branch is not used by this project. Track IDs are created from the face boxes over time and are not biometric identities.

## 4. GPU selection

The application checks CUDA availability and the device's compute capability before selecting CUDA. This was added because a CUDA-enabled PyTorch build can still fail when its compiled architecture list does not contain the installed GPU architecture. If CUDA cannot be used, the application falls back to CPU.

The CUDA requirements file targets the PyTorch CUDA 13.0 build used during development. The CPU requirements file provides a separate installation path.

## 5. Tracking

The detector reports faces independently on each frame. Detection row order is not an identity guarantee, so the application maintains visual tracks.

Tracking uses predicted box position, overlap, distance and size-change constraints. Track segments are intentionally treated as visual continuity rather than identity recognition. A person leaving the frame for long enough can receive a new track when they return.

The UI calls these values `Track 1`, `Track 2`, etc. This avoids implying that the program knows who the person is.

## 6. Measurement quality

A facial action estimate is more useful when the conditions under which it was measured are also recorded. The quality layer considers factors such as face size, tracking state and head pose.

Head pose is treated as a measurement-quality condition because facial action estimates can degrade as the face rotates away from the view used by the detector. Gaze is stored separately. Looking away is not automatically classified as an expression failure.

Quality is available at the general measurement level and, where appropriate, at the AU region level. Eye-related AUs and mouth-related AUs can be affected differently by visibility and pose.

The quality layer does not alter the raw detector output. It provides context for later analysis and filtering.

## 7. Emotion interpretation

The detector returns probabilities for seven labels. The application records the complete probability vector rather than retaining only the largest class.

The display also calculates the leading class, the second class and the separation between them. When the distribution is weak or closely split, the interface reports `Mixed / uncertain` rather than presenting the largest class as a definitive description.

This is intentional. A classifier must select a class even when several facial actions are present at once. The UI should not turn that forced selection into a stronger claim than the underlying probabilities support.

## 8. Calibration

Calibration is a reference adjustment, not model retraining. A completed session can provide a neutral/reference profile from stable, high-quality frames.

For each AU the profile stores a robust central value and a robust spread estimate. The implementation uses the median and median absolute deviation (MAD), with a minimum scale floor to prevent extremely small spreads from producing meaningless large standardized deviations.

For pose and gaze, calibration stores a reference orientation and reports later differences from that reference.

The raw AU values and raw emotion probabilities remain unchanged. Calibration fields are stored separately.

## 9. Temporal AU analysis

The temporal layer treats each AU as a time series rather than a collection of unrelated frames. It records activation and deactivation using separate thresholds. This hysteresis reduces repeated on/off switching when a value is close to the activation threshold.

An episode contains:

- person/track ID
- AU
- onset time and frame
- peak time and frame
- offset time and frame
- baseline
- peak value
- amplitude
- duration
- rise rate
- fall rate
- rate validity flags
- single-frame-peak information
- measurement-quality information

A rate is not reported when the time interval is too small. A one-frame peak can be real as a sampled observation, but its derivative cannot be treated as a reliable continuous-time rate.

Episodes also have minimum duration and amplitude thresholds. These are intended to remove trivial detector jitter rather than claim that all short events are noise.

## 10. Smoothing

The live signal display uses exponential smoothing. Temporal analysis can also use Savitzky-Golay filtering when enough samples are available. Smoothed values are derived values; the raw detector series remains available.

Derivatives are calculated only when the sampling interval is valid. Values are constrained to the detector's AU range after smoothing so filtering cannot introduce values outside the expected interval.

## 11. Simultaneous AUs

Several AUs can begin within the same frame or within a small timing tolerance. The sequence layer therefore groups near-simultaneous activations instead of forcing an arbitrary order between them.

This matters because sorting same-frame events by dictionary or detector order would create an artificial temporal sequence.

## 12. Coactivation

Coactivation counts how often two AUs are active during the same valid observations. The export contains the AU pair and its count. The count is descriptive and depends on the duration, sampling rate and activation threshold of the session. It should not be interpreted as a general population association.

## 13. Directed transitions

A transition records one AU becoming active before another AU within a configured temporal window. The analysis stores the direction and delay. Same-frame activations are not treated as directional transitions.

This allows questions such as whether AU A tends to precede AU B in one recording. It does not establish a causal relationship between the facial actions.

## 14. Temporal clusters and sequences

Nearby AU episodes can be grouped into local temporal clusters. A sequence contains a set of related episodes that occur within configured time gaps.

Earlier versions grouped too much of a recording into one sequence. The current implementation uses shorter temporal gaps and onset timing so a long recording can contain multiple local sequences rather than one container spanning everything.

Sequence analysis remains a derived layer and should be validated against manually annotated examples before being used as a research endpoint.

## 15. Output architecture

The project writes both individual files and a master file. The separate files make it easy to inspect or import one data type. `analysis.json` provides a single entry point for analysis scripts.

The master structure is:

```json
{
  "session_info": {},
  "summary": {},
  "frame_data": [],
  "events": [],
  "temporal_analysis": {}
}
```

The current schema version is 4 and the temporal analysis version is 4.3.

## 16. Review interface

The main window contains live/processing information and separate analysis views. The current interface includes overview information, FACS values, face/mesh data, blendshapes, raw/events, graphs, temporal review and settings.

Recorded sessions can be reopened without rerunning the detector. The review controls allow timeline scrubbing, frame stepping, track selection and navigation through temporal events.

## 17. Internal validation

The application has a validation button that runs deterministic local tests against the temporal machinery. The tests cover the basic signal/episode logic and selected data handling paths. A JSON report is saved under `facs_output/validation/`.

The internal tests are not a substitute for an external benchmark. A proper accuracy study requires videos with independent frame-level or event-level annotations.

## 18. Error handling

The worker reports detection failures without silently replacing the entire analysis with fabricated values. CPU fallback is available for CUDA initialization failures. JSON export converts non-finite numeric values to `null` rather than emitting invalid JSON.

The worker initializes dimensions before any quality calculation. This avoids a failure mode where the quality layer attempts to use `frame_width` or `frame_height` before the input stream has supplied them.

## 19. Performance

Processing speed depends on video resolution, face count, GPU model, Py-Feat version and whether all mesh/blendshape outputs are being retained. Processing time and video time are stored separately. A run that takes 60 seconds to process a 14-second video is not a 60-second video.

The application reports processing timing so performance can be evaluated without confusing compute time with source-video duration.
