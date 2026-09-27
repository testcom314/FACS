# Research Notes

These notes record the external material used to shape the analysis design. They are included so the project decisions can be traced to published work and upstream documentation rather than appearing as arbitrary thresholds.

## Py-Feat

Py-Feat's current documentation describes Detectorv2 as a multitask detector producing 20 AUs, seven emotion classes, valence/arousal, gaze, head pose, a 478-point 3D mesh and blendshapes. The video tutorial also describes frame-by-frame and batched video processing.

- Py-Feat image detection: https://py-feat.org/basic_tutorials/Detecting_Images/
- Py-Feat video detection: https://py-feat.org/basic_tutorials/Detecting_Videos/
- Py-Feat model overview: https://py-feat.org/pages/models/

## Automated facial coding limitations

Cross, Acevedo and Hunter (2023) discuss validity, reliability under non-ideal conditions and the theoretical assumptions involved in automated facial coding. A major point relevant to this project is that facial movement measurements and inferred emotion labels should not be treated as the same thing.

The paper also discusses problems caused by lighting, pose, occlusion and other conditions that differ from controlled datasets. These concerns are the reason the project stores measurement-quality fields instead of silently treating every frame as equally reliable.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10514002/

## DISFA

DISFA is a spontaneous facial-action database in which each video frame was manually coded for AU presence, absence and intensity. It is a useful reference for evaluating AU intensity and frame-level measurements against independent human coding.

Source: https://ieeexplore.ieee.org/document/6475933

## Microexpression datasets

CASME II and SAMM use high-speed recordings and provide onset/apex/offset annotations for microexpression samples. Their temporal resolution is substantially higher than ordinary 30 FPS video. This is one reason the project does not label every short 30 FPS AU episode as a microexpression.

CASME II: https://doi.org/10.1007/s11042-013-1647-6

SAMM: https://doi.org/10.1007/s11042-016-3594-1

## Signal processing

Savitzky-Golay filtering and peak detection are standard tools for smoothing and locating structure in sampled signals. The project uses these ideas for derived temporal analysis while preserving the unsmoothed detector values.

SciPy signal filtering: https://docs.scipy.org/doc/scipy/reference/signal.html

## Design consequences

The research led to several concrete implementation choices:

1. Keep AU values as the primary data layer.
2. Keep emotion probabilities instead of only the top class.
3. Store pose and gaze separately from expression.
4. Record quality alongside measurements.
5. Use onset/peak/offset rather than only frame counts.
6. Reject or flag mathematically unstable rate estimates.
7. Treat same-frame AUs as simultaneous.
8. Treat transitions as temporal associations, not causes.
9. Do not call short 30 FPS events microexpressions without suitable evidence.
10. Require independent annotation for claims about accuracy or genuine/posed classification.
