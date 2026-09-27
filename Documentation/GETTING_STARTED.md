# Getting Started

## 1. Install

Use Python 3.11 or newer.

Create a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

For CPU-only installation:

```powershell
pip install -r requirements-cpu.txt
```

For the CUDA 13.0 environment used during development:

```powershell
pip install -r requirements-cuda130.txt
```

## 2. Run

```powershell
python run.py
```

The application opens the FACS Analysis Workstation.

## 3. First video test

For a first run, use a short video with a clearly visible face. A 10–60 second clip is enough to verify the pipeline without generating a huge analysis file.

Select the video, start analysis and wait for the session to finish.

Results are written below:

```text
facs_output/
└── session_YYYY-MM-DD_HH-MM-SS/
```

## 4. What to inspect first

Start with:

```text
summary.json
session_info.json
analysis.json
episodes.csv
```

For a frame-by-frame inspection, use `data.csv`.

For temporal relationships, inspect `episodes.csv`, `coactivation.csv`, `transitions.csv` and `sequences.csv`.

## 5. Simple Python inspection

```python
import json

with open("facs_output/session_YYYY-MM-DD_HH-MM-SS/analysis.json", encoding="utf-8") as f:
    data = json.load(f)

print("Frames:", len(data["frame_data"]))
print("Episodes:", len(data["temporal_analysis"]["episodes"]))

for episode in data["temporal_analysis"]["episodes"]:
    if episode.get("au") == "AU12":
        print(
            episode.get("onset_time"),
            episode.get("offset_time"),
            episode.get("amplitude")
        )
```

## 6. First sanity check

Before interpreting an interesting expression, check:

1. Was the face tracked continuously?
2. Was measurement quality acceptable?
3. What were the raw AU values?
4. What was the head pose?
5. Was the emotion output ambiguous?
6. Does the event appear across multiple frames?
7. Does the tracked video actually show the corresponding movement?

This prevents a single noisy model output from becoming an entire theory about what somebody was feeling.
