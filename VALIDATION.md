# Validation

## Internal validation

The application contains an `Run internal validation tests` button. It runs deterministic checks against the temporal processing code and writes:

```text
facs_output/validation/validation_report.json
```

The report contains the number of tests, passed tests, failed tests and individual results.

The validation path is kept inside the application so it exercises the same worker methods used by normal analysis instead of maintaining a second simplified implementation.

## What should be tested

### 1. Episode timing

Use synthetic signals with known onset, peak and offset locations. Confirm that the detected episode falls within the expected tolerance.

### 2. Hysteresis

Use a signal that oscillates around the activation threshold. Confirm that it does not create a large number of artificial on/off events.

### 3. Rate handling

Test both normal multi-frame rises and single-frame peaks. Normal intervals should produce rates; intervals below the configured minimum should be marked invalid rather than producing enormous values.

### 4. Baseline stability

Test a mostly constant signal with a short outlier. The median/MAD baseline should remain close to the stable part of the signal.

### 5. Simultaneous activation

Start two or more AUs on the same frame. The output should keep them simultaneous instead of inventing an ordering.

### 6. Transition direction

Create A -> B and B -> A examples with known delays. Confirm that the direction and delay are preserved.

### 7. Tracking

Use known moving boxes or a recorded multi-face clip. Check that continuous faces retain their track and that a track is not silently treated as a permanent identity.

### 8. JSON validity

Export records containing NaN or infinity internally and verify that the resulting JSON contains `null` rather than invalid numeric tokens.

### 9. Calibration

Create a profile from stable frames and verify that raw AU and emotion values are unchanged while calibration fields are added.

### 10. Pose quality

Use frontal and rotated examples and verify that quality fields respond to pose while gaze remains a separate signal.

## External validation

Internal tests only establish that the software behaves as designed. They do not establish detector accuracy.

A research-grade evaluation should use independent annotations. DISFA contains frame-level manual presence/absence/intensity coding for spontaneous facial action. CASME II and SAMM provide high-speed micro-expression recordings with onset, apex and offset annotations. These datasets address different questions and should not be treated as interchangeable benchmarks.

For event detection, interval overlap and timing agreement are more informative than a simple frame-level F1 score when the objective is to find temporal episodes.

## Current known validation gaps

- No claim of scientific accuracy is made from the built-in tests.
- The current project has not been calibrated against a human-coded dataset in this package.
- Cross-camera and cross-dataset performance has not been established.
- Genuine-versus-posed expression classification is not implemented or validated.
- The normal project video rate is not sufficient by itself for microexpression claims.
