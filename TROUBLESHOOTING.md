# Troubleshooting

## Import errors

### `No module named feat`

Install the project requirements in the active virtual environment.

```powershell
pip install -r requirements-cpu.txt
```

or the CUDA environment when appropriate.

### PySide6 or Qt errors

Make sure the virtual environment contains the same PySide6 installation used by the project. A warning such as:

```text
QFont::setPointSize: Point size <= 0 (-1)
```

is normally a Qt font warning rather than a detector failure. The application explicitly sets a normal application font.

## CUDA problems

Check:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_capability() if torch.cuda.is_available() else None)"
```

If the installed PyTorch wheel does not support the GPU architecture, CUDA can be installed correctly and still fail at runtime. Use the CUDA requirements file appropriate for the project or fall back to CPU.

## Detection errors

### `numpy.ndarray object has no attribute 'read'`

This indicates that a Py-Feat image-loading path received a NumPy array where it expected an image/file object. The current application prepares a Detectorv2 tensor and has a disk-image fallback for compatible failures.

### `FACSWorker object has no attribute frame_width`

The measurement-quality stage needs the current video dimensions. The worker now initializes `frame_width` and `frame_height` before processing and updates them after opening the source.

This was a worker-state initialization bug, not a face-detection problem.

### `FEAT_VERSION is not defined`

The session metadata records the installed Py-Feat version. The current source obtains it through `importlib.metadata` and falls back to `unknown` when the package metadata is unavailable.

## No faces

Check:

- video actually contains visible faces
- face size is large enough
- lighting is adequate
- face detection threshold is not unnecessarily high
- the detector loaded successfully

## Track IDs change

Track IDs are local tracking IDs, not identity recognition. Occlusion, rapid movement, large pose changes and missed detections can still cause track changes.

Inspect the tracked video and tracker report before interpreting temporal data across a track boundary.

## Too many short episodes

Inspect the raw AU series first. If the detector output itself is noisy, changing temporal thresholds only hides the underlying issue.

Possible adjustments include:

- increasing minimum episode amplitude
- increasing minimum episode duration
- raising the AU activation threshold
- reducing aggressive sensitivity to short bursts

## Absurd rise/fall rates

Check:

```text
rise_rate_valid
fall_rate_valid
single_frame_peak
```

A very short interval should result in a null/invalid rate rather than a huge numerical value.

## Emotion labels change rapidly

Do not assume this means the person's emotion changed every frame.

Check:

```text
emotion_top_probability
emotion_second_probability
emotion_margin
emotion_entropy
emotion_interpretation_status
measurement_quality
pose
```

Mixed facial actions and uncertain detector outputs can produce rapidly changing top labels.

## Large output files

The application stores dense face data, including a 478-point mesh and many other fields. Long videos and multiple faces therefore produce large JSON files.

For large experiments, use shorter clips or process sessions separately.

## GPU memory errors

Try a shorter clip, lower the source resolution, or use CPU. The exact GPU memory requirement depends on the installed model and runtime and should be measured on the target machine.

## Reporting a bug

Include:

```text
Python version
Py-Feat version
PyTorch version
CUDA version
GPU
OS
full traceback
video format/resolution/FPS
session_info.json
```

Do not report only “it crashed.” Computers have enough opportunities to be mysterious without us withholding the evidence.
