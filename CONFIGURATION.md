# Configuration Reference

The main runtime settings are defined near the top of `FACS_Level2.py`. The GUI exposes the settings that are intended to be changed during normal use.

| Setting | Current value | Purpose |
|---|---:|---|
| Face detection threshold | 0.30 | Minimum detector face confidence |
| Detection interval | 1 | Process every frame by default |
| Display history | 180 | Number of recent display samples |
| Smoothing | 0.35 | Exponential smoothing factor |
| Activation hysteresis | 0.03 | Separation between activation and deactivation thresholds |
| Baseline window | 45 frames | Rolling baseline window |
| Baseline minimum samples | 12 | Minimum samples before baseline use |
| Minimum episode duration | 2 frames at 30 FPS | Removes trivial one-sample episodes |
| Minimum episode amplitude | 0.035 | Minimum AU excursion |
| Maximum cluster gap | 0.35 s | Groups nearby AU episodes |
| Maximum sequence gap | 0.75 s | Limits sequence merging |
| Minimum transition delay | 1 frame at 30 FPS | Excludes same-frame ordering |
| Tracking max missing frames | 45 | Track persistence |
| Tracking distance gate | 1.15 | Normalized movement gate |
| Tracking minimum IoU | 0.02 | Box overlap gate |
| Tracking max size change | 0.65 | Rejects implausible box changes |
| Temporal max gap | 0.25 s | Breaks temporal continuity |
| Baseline scale floor | 0.02 | Prevents unstable normalized deviations |

Thresholds are implementation parameters, not universal FACS constants. They should be validated against the intended recording conditions before being treated as research settings.
