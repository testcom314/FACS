# Validation

## 1. What internal validation means

The internal validation system checks whether the application's own analysis logic behaves as intended. It does not measure Py-Feat's facial-expression accuracy.

A passing synthetic episode test means the episode code can recover a known synthetic pattern. It does not mean the detector will recover every real human expression.

## 2. Running internal validation

The application exposes internal validation through the GUI. The report is written as JSON under the application's validation output when the validation action is used.

The report has this general structure:

```json
{
  "schema_version": 1,
  "total": 8,
  "passed": 8,
  "failed": 0,
  "results": [
    {
      "name": "tracker keeps IDs bounded",
      "passed": true,
      "detail": "..."
    }
  ]
}
```

The exact number of tests can change as the implementation changes. Do not hard-code a requirement such as “11/12 means accurate.” The useful question is whether all current deterministic logic tests pass.

## 3. Current internal tests

The validation code currently exercises:

1. Tracker ID bounds during crossing faces and a short detector dropout.
2. No duplicate assignment within a frame.
3. Detection of a synthetic AU12 episode.
4. Sane rise/fall rates.
5. Prevention of a giant sequence caused by continuous activity.
6. Same-frame activation grouping.
7. Smoothed AUs remaining in the detector's 0–1 domain.
8. Separation of distant temporal events.

The synthetic data is generated inside the application. No network connection is required for these checks.

## 4. What internal validation cannot prove

It cannot prove:

- AU accuracy against human FACS coding
- emotion recognition accuracy
- cross-camera robustness
- demographic robustness
- genuine versus posed expression classification
- microexpression detection accuracy
- causal relationships between AUs

Those require external datasets and independent evaluation.

## 5. External AU validation

A proper external study should use frame-level human annotations such as DISFA/DISFA+ or another dataset with appropriate AU labels.

Possible measurements include:

- AU presence agreement
- AU intensity correlation where compatible labels exist
- temporal onset/offset agreement
- event-level precision and recall
- agreement across subjects not used for tuning

Class imbalance should be considered. A single frame-level F1 number can hide important differences between common and rare AU events.

## 6. Temporal validation

For onset/apex/offset work, datasets with explicit temporal annotations such as CASME II and SAMM are useful references.

A sensible test is:

```text
annotated onset / apex / offset
             |
             v
run the same video through the application
             |
             v
compare detected episode intervals
```

For high-speed datasets, compare the detector's temporal resolution with the source frame rate before interpreting errors.

Downsampling a 200 FPS dataset to 30 FPS changes the problem. It should be treated as a separate experiment, not as equivalent data.

## 7. Manual spot checks

A simple practical validation method is to select short clips and manually inspect the corresponding frames and AU curves.

For each selected event:

1. Watch the original video.
2. Locate the relevant frame interval.
3. Inspect raw AU values.
4. Inspect smoothed values.
5. Check the episode onset, peak and offset.
6. Check measurement quality and pose.
7. Compare the tracked face against the video.

This is not a replacement for a formal benchmark, but it is useful for finding implementation errors.

## 8. Reproducibility

A validation run should record:

- Python version
- Py-Feat version
- PyTorch version
- CUDA version
- GPU
- video frame rate
- video resolution
- analysis settings
- schema version
- temporal-analysis version

The application stores many of these values in session metadata.

## 9. Known gaps

The project still needs independent validation against human-coded data before numerical accuracy claims should be made.

In particular, there is currently no validated classifier for “genuine” versus “posed” expression and no basis for using the application as a lie detector.
