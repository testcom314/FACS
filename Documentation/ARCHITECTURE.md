# Refactored Project Layout

The original monolithic `FACS_Level2.py` has been split by responsibility while keeping the existing public behavior and the historical entry point.

```text
FACS_Refactored/
├── FACS_Level2.py              # compatibility entry point
├── run.py                      # normal entry point
├── ARCHITECTURE.md
└── facs_workstation/
    ├── __init__.py
    ├── dependencies.py         # controlled third-party import order
    ├── config.py               # constants and analysis parameters
    ├── utils.py                # shared helpers and calibration I/O
    ├── signal_processing.py    # smoothing
    ├── tracking.py             # multi-face visual tracker
    ├── worker.py               # detection + analysis + export worker
    ├── widgets.py              # reusable Qt widgets
    ├── validation.py           # deterministic internal tests
    ├── main_window.py          # GUI and playback logic
    └── main.py                 # application entry point
```

The split is intentionally conservative. Core analysis code was moved rather than redesigned so that behavior remains comparable to the source version. The next natural refactoring step would be to split `worker.py` itself into detector, temporal-analysis, and export components once behavior is fully regression-tested.
