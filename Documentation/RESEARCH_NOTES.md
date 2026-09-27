# Research Notes

This file records the research that influenced the design. It is not an accuracy claim for the application.

## Automated facial coding

Cross et al. (2023), *A Critique of Automated Approaches to Code Facial Expressions: What Do Researchers Need to Know?*

https://pmc.ncbi.nlm.nih.gov/articles/PMC10514002/

This work is useful for understanding the distinction between automated facial measurements and psychological interpretation. It is one reason the project keeps AU measurements separate from emotion labels and avoids treating automated output as ground truth.

## Dynamic emotion expression

Research on dynamic facial expression emphasizes that timing, speed, duration and amplitude can carry information that is lost in a single still image.

This motivates the project's onset/peak/offset and temporal-analysis layers.

## Microexpression datasets

CASME II:

https://facedb.seu.edu.cn/TopFolder/Database/casme2.html

CASME II contains high-speed facial recordings with temporal annotations such as onset, apex and offset. Its frame rate is substantially higher than ordinary webcam video, which is important when discussing very short facial events.

SAMM and SMIC are additional microexpression datasets commonly used for temporal facial-expression research.

## DISFA

DISFA provides spontaneous facial behavior with frame-level AU annotations and intensity information.

https://www.cs.rochester.edu/u/qyou/face/DISFA/

It is a useful reference for AU validation because it provides human-coded labels rather than relying on another automated detector as the ground truth.

## FACS

The Facial Action Coding System defines Action Units and their appearance criteria. The project uses the FACS vocabulary for labeling detector outputs but does not claim that the continuous detector value is equivalent to a trained coder's ordinal FACS intensity score.

## Temporal AU detection

Research on temporal AU detection has used explicit onset/apex/offset phases and models such as HMMs. Other work uses optical flow, local appearance descriptors and temporal sequence models.

These approaches are relevant to future versions of the project, but the current implementation remains a deterministic signal-processing system rather than an HMM or learned temporal model.

## Signal processing

SciPy's signal-processing tools are used for smoothing and peak analysis where appropriate:

https://docs.scipy.org/doc/scipy/reference/signal.html

In particular, the project uses Savitzky-Golay filtering and peak-detection functionality when available.

## Interpretation limits

The following claims are deliberately outside the scope of the current software:

- detecting lies
- determining whether an expression is “real” with a single score
- determining internal emotional state from a face alone
- diagnosing psychological or medical conditions
- treating an emotion classifier's top label as ground truth
- calling every rapid facial change a microexpression

These limitations are part of the design rather than disclaimers added after the fact.
