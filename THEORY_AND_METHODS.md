# Theory and Methods

## 1. What is being measured

The application is built around facial Action Units (AUs) rather than treating an emotion label as the primary measurement.

The current detector exposes 20 AUs:

`AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10, AU11, AU12, AU14, AU15, AU17, AU20, AU23, AU24, AU25, AU26, AU28, AU43`.

The AU values are continuous outputs from the trained detector. They are not manually coded FACS intensity scores and should not be described as ground-truth measurements. A value such as `AU01 = 0.72` means that the detector produced a value of 0.72 for AU01 in that frame. The number should be treated as a model output on the detector's numerical scale, not automatically as “72% certainty” or “72% muscle activation.”

The application also records seven emotion outputs, valence, arousal, head pose, gaze, 68 landmark coordinates, a 478-point face mesh and available blendshape outputs.

Keeping these measurements separate is intentional. An emotion classifier can summarize several facial actions into one label, while the AU layer preserves more of the observable information.

## 2. Facial Action Coding System

FACS describes visible facial movement using Action Units. Examples include AU01 (Inner Brow Raiser), AU06 (Cheek Raiser), AU12 (Lip Corner Puller), AU25 (Lips Part) and AU43 (Eyes Closed).

The software uses the detector's AU outputs as the starting signal and then performs temporal analysis on those signals. It does not claim to replace a trained human FACS coder.

## 3. Why emotion labels are not treated as ground truth

The seven emotion outputs are useful as model estimates, but they are not an independent measurement of a person's internal emotional state.

For example, a frame could produce something like:

```text
Happy     0.45
Surprise  0.30
Fear      0.15
Other     0.10
```

The interface can show Happy as the highest model output while still exposing the other probabilities, the top-two margin and an ambiguity status. This avoids turning a close classification into a false statement such as “the person is happy.”

The same principle applies to temporal analysis. A rapid AU change is a property of the recorded signal. It is not evidence by itself of a microexpression, deception, sincerity or a particular emotion.

## 4. Temporal analysis

A single frame is often less informative than the way an AU changes over time. The application therefore records onset, peak and offset information when an AU forms a valid episode.

An episode can contain:

- onset time and frame
- peak time and frame
- offset time and frame
- baseline
- peak value
- amplitude
- duration
- rise rate when the timing interval is long enough to support it
- fall rate when the timing interval is long enough to support it
- validity flags
- single-frame peak information

The current implementation also records rapid changes, short bursts, simultaneous AU activation, coactivation and directed transitions.

## 5. Hysteresis

Episode detection uses separate activation and deactivation thresholds. This prevents an AU that is hovering around one threshold from repeatedly switching between active and inactive.

Conceptually:

```text
if AU_value >= activation_threshold:
    active = True
elif AU_value < deactivation_threshold:
    active = False
# otherwise keep the previous state
```

The default activation threshold is 0.15. The deactivation threshold is the activation threshold minus 0.03, subject to a lower bound in the implementation.

This is signal-state hysteresis. It is not a claim that 0.15 is a universal FACS threshold.

## 6. Rates and short events

A rate is calculated from a change in AU value divided by the elapsed time. The timing interval must be long enough for the result to be meaningful.

At 30 FPS, one frame is approximately 33.3 ms. If an AU rises from 0.1 to 0.9 in one frame, the numerical slope is about 24 units/s. The application does not treat a one-frame rise as a reliable temporal rate. The current minimum rate interval is two frames, approximately 66.7 ms at 30 FPS.

When the interval is too short, `rise_rate` or `fall_rate` is stored as `null` and the corresponding validity flag is false. This prevents the earlier failure mode where a nearly zero denominator produced absurd values such as tens of thousands of units per second.

High-speed microexpression datasets such as CASME II use much finer temporal sampling, commonly around 200 FPS. A 30 FPS webcam or ordinary video therefore cannot provide the same temporal resolution.

## 7. Baselines and calibration

Two ideas are kept separate:

1. A rolling baseline used by temporal analysis.
2. An optional neutral reference calibration profile.

The rolling baseline is intended to describe recent signal level. The current implementation uses a robust median over a bounded history, with a minimum number of samples before treating the estimate as established.

Calibration is reference normalization. It does not retrain Py-Feat, alter the detector weights or turn the application into a personalized emotion classifier.

A calibrated value can therefore be interpreted as a deviation from the person's recorded reference condition. It should not be interpreted as an absolute physiological measurement.

Median Absolute Deviation (MAD) is used for robust spread estimation. A scale floor prevents extremely small variation around the baseline from producing enormous normalized values.

## 8. Head pose and gaze

Head pose is treated as a measurement-condition variable. The application records pitch, yaw and roll and derives a pose magnitude used in quality scoring.

The application does not “correct” the AU values by mathematically forcing a rotated face to look frontal. That would create a second model layered on top of the detector without validation.

Gaze is recorded separately. Looking away is not itself an expression failure, so gaze is used as context and affects AU-specific observability differently by facial region.

The current quality model gives eye-related AUs more sensitivity to gaze than mouth-related AUs.

## 9. Measurement quality

The quality score is intended to answer:

> How observable and usable was this face measurement under the recorded conditions?

It combines detector confidence, approximate face size, head-pose quality and roll quality. The current weighting is:

```text
45% face confidence
25% face size
20% pitch/yaw pose quality
10% roll quality
```

The resulting categories are `GOOD`, `FAIR`, `LIMITED` and `POOR`.

This score is not an accuracy estimate. A `GOOD` frame is not proof that the AU values are correct. It simply means the recorded conditions satisfy the application's measurement-quality heuristics more closely.

## 10. Emotion interpretation quality

The raw emotion outputs are preserved. A separate interpretation layer calculates:

- top emotion
- top probability/output
- second emotion
- second probability/output
- top-two margin
- normalized entropy
- interpretation score
- interpretation status
- top three outputs

If the top two outputs are close or the distribution is sufficiently spread out, the status becomes `MIXED / AMBIGUOUS`.

This is deliberately different from modifying the detector probabilities. The raw values remain untouched.

## 11. Tracking

Detector output order cannot be assumed to represent stable identity across frames. The application therefore assigns local track IDs using face-box geometry and motion prediction.

A track ID means:

```text
Track 1
Track 2
Track 3
```

It does not mean a recognized person. Identity recognition is disabled.

The tracker predicts bounding-box position, computes matching costs from geometry, performs global assignment when SciPy's assignment routine is available, updates motion state and creates or retires tracks as needed.

A track can still be wrong. The output should therefore be treated as a tracking hypothesis, not biometric identity.

## 12. Coactivation and transitions

Coactivation asks which AUs tend to be active during the same frames. Raw overlap counts alone can be misleading because a frequently active AU will overlap with many other AUs simply by being common.

The application therefore records additional conditional and overlap information where available.

Transitions are directed temporal relationships. For example:

```text
AU07 -> AU15
```

means that AU07 activation was followed by AU15 activation within the configured transition window. It does not mean AU07 caused AU15.

Activations detected within the same-frame tolerance are grouped as simultaneous rather than being assigned an arbitrary order.

## 13. Temporal sequences

A sequence is a local period of facial-action activity assembled from temporally related AU events. It is not automatically an “expression” and it is not an emotion episode.

This distinction matters because a person can keep one AU active while other actions appear and disappear. A naive sequence algorithm can then merge most of a recording into one giant event. The current implementation explicitly tests against that failure mode.

## 14. Microexpressions and genuine expressions

The project does not claim to determine whether somebody is lying, faking an emotion or experiencing a particular internal state.

Microexpression research is concerned with very short, often subtle facial events and their temporal structure. A conventional 30 FPS recording provides roughly 33 ms between frames, so short events can be undersampled or represented by only one or two frames.

Likewise, spontaneous versus posed expression analysis requires appropriately labeled data and subject-independent validation. The current application provides temporal measurements that could be used for such research, but it does not contain a validated “genuine” score.

## 15. Research basis

The design was influenced by work on automated facial coding, temporal facial expression analysis, FACS event detection, AU intensity measurement and microexpression spotting.

Useful starting points include:

- Cross et al. (2023), *A Critique of Automated Approaches to Code Facial Expressions: What Do Researchers Need to Know?*
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10514002/
- Valstar et al. work on temporal phases of facial action units.
- DISFA and DISFA+ for spontaneous and intensity-labelled AU data.
- CASME II, SAMM and SMIC for high-speed microexpression research.
- SciPy signal-processing documentation for smoothing and peak detection.

These sources are used to motivate the measurement and validation design, not to imply that this application has reproduced their reported accuracy.
