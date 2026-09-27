"""General-purpose helpers shared by the application."""

import json
import math

import numpy as np

from .dependencies import torch
from .config import CALIBRATION_PATH, OUTPUT_ROOT

def safe_float(v, default=np.nan):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def get_cuda_info():
    if not torch.cuda.is_available():
        return {"available": False, "device": "cpu", "name": "CPU", "capability": None, "arch": []}
    try:
        idx = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(idx)
        arch = torch.cuda.get_arch_list()
        target = f"sm_{cap[0]}{cap[1]}"
        usable = target in arch
        return {
            "available": usable,
            "device": "cuda" if usable else "cpu",
            "name": torch.cuda.get_device_name(idx),
            "capability": cap,
            "arch": arch,
        }
    except Exception:
        return {"available": False, "device": "cpu", "name": "CPU", "capability": None, "arch": []}

def choose_device():
    info = get_cuda_info()
    return info["device"], info

def percent(v):
    x = safe_float(v, 0.0)
    return clamp(x * 100.0, 0.0, 100.0)

def emotion_color(name):
    # UI color only; not an inference claim.
    return {
        "Happy": "#55d68a", "Sad": "#6aa9ff", "Anger": "#ff6b6b",
        "Fear": "#c58cff", "Disgust": "#83d16a", "Surprise": "#ffc857",
        "Neutral": "#aab4c0",
    }.get(name, "#aab4c0")

def row_to_dict(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.floating, float)):
            out[str(k)] = None if not math.isfinite(float(v)) else float(v)
        elif isinstance(v, (np.integer, int)):
            out[str(k)] = int(v)
        else:
            out[str(k)] = str(v) if v is not None else None
    return out

def robust_median(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.median(vals)) if vals else None

def robust_mad(values, center=None):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return None
    c = float(np.median(vals)) if center is None else float(center)
    return float(np.median(np.abs(np.asarray(vals) - c)))

def load_calibration_profile():
    try:
        if CALIBRATION_PATH.exists():
            data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("schema_version") == 1:
                return data
    except Exception:
        pass
    return None

def save_calibration_profile(profile):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")
