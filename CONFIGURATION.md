# Configuration

## 1. Main defaults

The current source defines the following important defaults:

| Setting | Default | Purpose |
|---|---:|---|
| Face detection threshold | 0.30 | Detector face acceptance threshold |
| Detection interval | 1 | Process every frame |
| Display history | 180 | Live graph/history length |
| Smoothing factor | 0.35 | Exponential smoothing |
| AU activation threshold | 0.15 | Start an AU episode |
| Hysteresis | 0.03 | Gap between activation and deactivation |
| Minimum episode duration | 2/30 s | Reject extremely short episodes at 30 FPS |
| Minimum episode amplitude | 0.035 | Reject very small AU excursions |
| Baseline window | 45 frames | Rolling baseline history |
| Minimum baseline samples | 12 | Minimum samples before baseline is trusted |
| Maximum temporal gap | 0.25 s | Temporal continuity limit |
| Cluster gap | 0.35 s | Local activity clustering |
| Sequence gap | 0.75 s | Sequence grouping |
| Simultaneous tolerance | 0.5/30 s | Treat near-same-frame activations as simultaneous |
| Minimum rate interval | 2/30 s | Minimum timing interval for rates |
| Maximum missing track frames | 45 | Track persistence |

## 2. Choosing the face threshold

Lowering the face threshold can help with small or weak detections, but it also allows more borderline detections.

Raising it is useful when only strong detections should enter the analysis.

A lower threshold is not automatically more accurate. It changes the trade-off between missed detections and false detections.

## 3. Choosing the AU threshold

The default AU activation threshold is 0.15.

Lower it when subtle changes are important and additional false episode candidates are acceptable.

Raise it when the analysis is intended to focus on stronger AU activity.

The deactivation threshold remains below the activation threshold because hysteresis is used.

## 4. Detection interval

`1` means every frame is processed.

`2` means every second frame is processed, reducing computation but also reducing temporal resolution.

For temporal research, processing every frame is preferable when hardware allows it.

Skipping frames should not be presented as equivalent to recording at a higher frame rate.

## 5. Smoothing

The live exponential smoother uses:

```text
smooth = alpha * current + (1 - alpha) * previous
```

Higher alpha follows the current frame more closely and reacts faster. Lower alpha produces more persistence from previous values.

The displayed/derived smoothed series should not replace the raw detector series in exported data.

## 6. Episode duration and amplitude

The default minimum episode duration is two frames at 30 FPS, approximately 66.7 ms.

The default minimum amplitude is 0.035 on the detector's AU numerical scale.

These values are engineering filters, not universal FACS definitions.

For high-speed data, the minimum duration should be reconsidered relative to the actual frame rate.

## 7. Tracking

If track IDs split too often, the relevant gates are:

- maximum missing frames
- distance gate
- IoU gate
- size-change gate
- velocity smoothing

Changing these values affects track continuity and can therefore affect every downstream temporal measurement.

Tracking should be fixed before interpreting temporal episodes from a multi-face recording.

## 8. Pose and gaze

The current quality system uses soft and hard pose limits rather than deleting rotated-face measurements.

The important principle is:

```text
pose changes the measurement context
pose does not rewrite the detector output
```

Gaze is kept separate from expression. Looking away should not automatically be labeled as a failed expression measurement.

## 9. Calibration

Calibration should be performed under a clearly defined reference condition. The resulting profile becomes part of the analysis context.

A calibration profile should not be treated as a universal correction that transfers unchanged between cameras, lighting conditions or sessions.

## 10. Practical decision tree

```text
Are faces missing?
    |
    +-- yes --> inspect face size, lighting and detection threshold
    |
    +-- no --> continue

Are track IDs unstable?
    |
    +-- yes --> fix tracking before temporal interpretation
    |
    +-- no --> continue

Are AU values noisy?
    |
    +-- yes --> inspect raw values first, then adjust smoothing/filtering
    |
    +-- no --> continue

Are there too many tiny episodes?
    |
    +-- yes --> increase minimum amplitude/duration or activation threshold
    |
    +-- no --> continue

Are short events important?
    |
    +-- yes --> use the highest available frame rate and avoid aggressive smoothing
    |
    +-- no --> normal temporal filtering may be sufficient
```

## 11. Common symptoms

| Symptom | Likely cause | First check |
|---|---|---|
| No faces | Detection threshold, source or model issue | Face confidence and video input |
| Too many episodes | Threshold too low or noisy signal | Raw AU series and amplitude filter |
| Missed subtle events | Threshold too high or smoothing too strong | Raw AU series |
| Track IDs change | Matching gates or occlusion | Tracker report and video |
| Emotion changes rapidly | Mixed model outputs or poor observability | Top-two margin, entropy and quality |
| Large rate values | Very short timing interval | `rise_rate_valid` / `fall_rate_valid` |
| One giant sequence | Sequence gap logic or sustained AU activity | `sequences.csv` |
| Rotated face produces unstable AUs | Pose reduces observability | Pose quality and AU-specific quality |
