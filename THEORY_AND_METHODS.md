# Theory and Methods

## 1. What is being measured

The central measurement in this project is the facial Action Unit (AU). An AU represents a visually defined facial action in the Facial Action Coding System. The application records detector estimates of AU activation and intensity over time.

An AU is not an emotion. A detector can report several AUs simultaneously, and the same AU can occur in different contexts. The application therefore keeps AU measurements as the primary analysis layer and treats emotion probabilities as a separate model output.

## 2. The 20 AUs

The current Detectorv2 output used by the application includes:

| AU | Action |
|---|---|
| AU01 | Inner Brow Raiser |
| AU02 | Outer Brow Raiser |
| AU04 | Brow Lowerer |
| AU05 | Upper Lid Raiser |
| AU06 | Cheek Raiser |
| AU07 | Lid Tightener |
| AU09 | Nose Wrinkler |
| AU10 | Upper Lip Raiser |
| AU11 | Nasolabial Deepener |
| AU12 | Lip Corner Puller |
| AU14 | Dimpler |
| AU15 | Lip Corner Depressor |
| AU17 | Chin Raiser |
| AU20 | Lip Stretcher |
| AU23 | Lip Tightener |
| AU24 | Lip Pressor |
| AU25 | Lips Part |
| AU26 | Jaw Drop |
| AU28 | Lip Suck |
| AU43 | Eyes Closed |

The detector also returns seven emotion probabilities, valence/arousal, pose, gaze, landmarks, mesh and blendshapes.

## 3. Why the project does not treat emotion labels as ground truth

A facial expression recognition model can map facial observations to emotion categories, but the category is a model output rather than a direct reading of subjective experience. Research on automated facial coding has repeatedly highlighted validity, reliability in natural conditions and the theoretical assumptions involved in converting facial movement into emotion labels.

For this reason the application keeps the full probability distribution and displays ambiguity. A frame can contain a combination of facial actions that does not fit cleanly into one of seven labels.

## 4. Temporal information

A single frame cannot describe how a facial action developed. The temporal layer therefore uses four useful landmarks:

- onset: the action begins to cross the activation condition
- peak/apex: the strongest observed point in the episode
- offset: the action falls below the deactivation condition
- duration: elapsed time from onset to offset

The project also records amplitude, rise rate and fall rate when the sampling interval supports those calculations.

This is closer to the way dynamic facial behavior is normally studied than simply counting positive frames. Timing, speed, amplitude and irregularity can contain information that an average AU value loses.

## 5. Hysteresis

Two thresholds are used instead of one. An AU must reach the activation threshold to start an episode, but it can remain active until it falls below a lower deactivation threshold.

Without hysteresis, a noisy signal near a single threshold can produce a sequence such as:

```text
active -> inactive -> active -> inactive -> active
```

without a meaningful change in the face. Hysteresis reduces this switching.

## 6. Rates and short events

A common mistake in frame-based analysis is to calculate a derivative across a single frame interval and interpret the resulting large number as a meaningful movement speed. At 30 FPS, a one-frame interval is about 33.3 ms. A small numerical change divided by that interval can produce a very large value.

The implementation therefore marks rates invalid when the time interval is too short. One-frame peaks are retained as observations, but they are not presented as reliable continuous-time rates.

This distinction is especially important for discussions of microexpressions. A short detector event is not automatically a microexpression. Microexpression datasets such as CASME II and SAMM use high-speed video and explicit onset/apex/offset annotations. Ordinary 30 FPS video has much less temporal resolution.

## 7. Baseline and calibration

A raw AU value is useful, but a reference level can help describe change within a session. The calibration layer estimates a neutral/reference baseline from stable frames. Median and MAD are used because they are less sensitive to a few unusually expressive frames than a simple mean and standard deviation.

For an AU value x, the derived quantities are conceptually:

```text
deviation = x - median_baseline
robust_deviation = deviation / max(scale, minimum_scale)
```

The minimum scale prevents an almost constant AU from producing enormous standardized values because the estimated spread is nearly zero.

Calibration does not modify the detector's raw output.

## 8. Pose and gaze

Head pose and gaze answer different questions. Head pose describes the orientation of the face relative to the camera. Gaze describes the direction of the eyes.

Head rotation can make some facial actions harder to measure because parts of the face become less visible or depart from the conditions represented in training data. The project therefore uses pose as a measurement-quality input.

Gaze is retained as its own signal. Looking down or to the side is not automatically treated as an expression error. The resulting analysis can therefore distinguish:

```text
facial action estimate
measurement conditions
gaze direction
```

## 9. Coactivation

Facial actions commonly occur together. Coactivation counts simultaneous valid activation of AU pairs. The result is a within-session description. It is not a normative statement about how AUs should combine in a population.

Counts also depend on the length of the recording and activation threshold, so comparisons between sessions should use normalized measures or matched recording conditions.

## 10. Transitions

A directed transition A -> B is recorded when A activates before B within a configured window. The delay is measured from the activation times.

A same-frame activation is simultaneous, not A -> B. Assigning an arbitrary order would create information that was never observed.

Transitions should be treated as temporal associations, not causes.

## 11. Sequences

Sequences are collections of nearby AU episodes. They are useful for finding local periods of facial activity, but sequence segmentation is a design choice rather than a property that the detector directly provides.

The current implementation uses local time gaps and onset timing. Long recordings can therefore contain multiple sequences. Sequence segmentation remains a candidate area for comparison with manually annotated data.

## 12. Genuine versus posed expressions

The project originally included the question of whether dynamic facial analysis could help distinguish genuine from deliberately produced expressions. The answer is more limited than a simple classifier suggests.

Onset timing, duration, peak intensity, rise/fall speed, AU combinations and temporal irregularity can be useful variables when comparing labeled spontaneous and posed material. They are not, by themselves, a reliable truthfulness detector.

A defensible experiment would require independent labels, subject-independent train/test splits and a dataset containing spontaneous and posed examples. The current application does not claim to perform that classification.

## 13. Validation strategy

There are three useful levels of validation:

### Signal-level tests

Generate known synthetic AU-like signals and check whether onset, peak, offset, duration and rate calculations recover the known values within tolerance.

### Annotation-level tests

Compare detected events against human FACS annotations. Event-level measures should include interval overlap and timing agreement rather than relying only on frame-level binary accuracy.

### Cross-dataset tests

A model or threshold that works on one recording condition can behave differently elsewhere. DISFA provides spontaneous facial-action intensity annotations, while datasets such as CASME II and SAMM provide high-speed micro-expression timing annotations. These are useful for testing different parts of the pipeline.

## 14. Limitations

The following are known limitations rather than hidden assumptions:

- Detector outputs inherit the limitations of the pretrained Py-Feat model.
- Camera angle, lighting, occlusion and face size can affect measurement quality.
- A visual track is not a biometric identity.
- Seven emotion categories cannot represent every possible facial configuration.
- A short event in a normal-speed video is not automatically a microexpression.
- Coactivation counts are recording-dependent.
- Sequence segmentation is heuristic and requires external validation.
- Calibration is a reference normalization step, not model personalization.
- Internal tests verify software behavior, not scientific accuracy.
- Facial measurements alone do not provide access to private mental states.
