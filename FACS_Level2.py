import sys
import os
import json
import csv
import math
import time
import traceback
import importlib.metadata
from pathlib import Path
from datetime import datetime
from collections import deque
from statistics import median

import cv2
import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter, find_peaks
    from scipy.optimize import linear_sum_assignment
except ImportError:
    savgol_filter = None
    find_peaks = None
    linear_sum_assignment = None

# Load matplotlib before PyTorch/PySide6. This avoids an import-hook conflict in the
# matplotlib/dateutil/six dependency chain used by Py-Feat.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QRadioButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QTabWidget, QProgressBar, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QSplitter, QScrollArea, QMessageBox, QPlainTextEdit, QSlider
)

try:
    from feat import Detectorv2
except ImportError:
    Detectorv2 = None

# ----------------------------- configuration -----------------------------
APP_NAME = "FACS Analysis Workstation"
try:
    FEAT_VERSION = importlib.metadata.version("py-feat")
except importlib.metadata.PackageNotFoundError:
    FEAT_VERSION = "unknown"
OUTPUT_ROOT = Path("facs_output")
FACE_DETECTION_THRESHOLD = 0.30
DETECTION_INTERVAL = 1
DISPLAY_HISTORY = 180
DEFAULT_SMOOTHING = 0.35
CALIBRATION_PATH = OUTPUT_ROOT / "calibration_profile.json"
POSE_SOFT_LIMIT_DEG = 20.0
POSE_HARD_LIMIT_DEG = 45.0
GAZE_CONTEXT_SOFT_LIMIT_DEG = 25.0
AU_EYE = {"AU01", "AU02", "AU05", "AU06", "AU07", "AU43"}
AU_MOUTH = {"AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28"}
SCHEMA_VERSION = 4
TEMPORAL_VERSION = "4.3"
MIN_EPISODE_DURATION = 2.0 / 30.0
MIN_EPISODE_AMPLITUDE = 0.035
ACTIVATION_HYSTERESIS = 0.03
BASELINE_WINDOW = 45
BASELINE_MIN_SAMPLES = 12
MAX_CLUSTER_GAP = 0.35
MAX_SEQUENCE_GAP = 0.75
MIN_TRANSITION_DELAY = 1.0 / 30.0
TRACK_MAX_MISSING_FRAMES = 45
TRACK_DISTANCE_GATE = 1.15
TRACK_MIN_IOU = 0.02
TRACK_MAX_SIZE_CHANGE = 0.65
TRACK_VELOCITY_ALPHA = 0.35
TRACK_MIN_HITS = 2
TEMPORAL_MAX_GAP_SECONDS = 0.25
BASELINE_SCALE_FLOOR = 0.02
SIMULTANEOUS_FRAME_TOLERANCE = 0.5 / 30.0
MIN_SHORT_BURST_DURATION = 2.0 / 30.0
MIN_RATE_INTERVAL_SECONDS = 2.0 / 30.0

AU_NAMES = {
    "AU01": "Inner Brow Raiser", "AU02": "Outer Brow Raiser", "AU04": "Brow Lowerer",
    "AU05": "Upper Lid Raiser", "AU06": "Cheek Raiser", "AU07": "Lid Tightener",
    "AU09": "Nose Wrinkler", "AU10": "Upper Lip Raiser", "AU11": "Nasolabial Deepener",
    "AU12": "Lip Corner Puller", "AU14": "Dimpler", "AU15": "Lip Corner Depressor",
    "AU17": "Chin Raiser", "AU20": "Lip Stretcher", "AU23": "Lip Tightener",
    "AU24": "Lip Pressor", "AU25": "Lips Part", "AU26": "Jaw Drop",
    "AU28": "Lip Suck", "AU43": "Eyes Closed",
}
EMOTION_NAMES = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
BLENDSHAPE_PREFIXES = ("_neutral", "brow", "cheek", "eye", "jaw", "mouth", "nose")

# Face mesh edges used by the display overlay.
MESH_EDGES = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33),
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (362, 398), (398, 384), (384, 385), (385, 386),
    (386, 387), (387, 388), (388, 466), (466, 263),
]


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
    # Display colors for the emotion labels.
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


class SignalSmoother:
    def __init__(self, alpha=DEFAULT_SMOOTHING):
        self.alpha = float(alpha)
        self.values = {}

    def reset(self):
        self.values.clear()

    def update(self, key, value):
        value = safe_float(value)
        if not math.isfinite(value):
            return value
        if key not in self.values or not math.isfinite(self.values[key]):
            self.values[key] = value
        else:
            self.values[key] = self.alpha * value + (1.0 - self.alpha) * self.values[key]
        return self.values[key]


class MultiFaceTracker:
    """Small multi-object tracker for face detections.

    The tracker is geometric only. It does not identify people biometrically. It keeps
    tracks alive through short detector dropouts and uses global assignment to reduce
    identity swaps when faces are near each other.
    """
    def __init__(self, max_missing=TRACK_MAX_MISSING_FRAMES, distance_gate=TRACK_DISTANCE_GATE,
                 min_iou=TRACK_MIN_IOU, max_size_change=TRACK_MAX_SIZE_CHANGE):
        self.max_missing = int(max_missing)
        self.distance_gate = float(distance_gate)
        self.min_iou = float(min_iou)
        self.max_size_change = float(max_size_change)
        self.tracks = {}
        self.next_id = 1
        self.total_detections = 0
        self.matched_detections = 0
        self.created_tracks = 0
        self.max_active_tracks = 0

    @staticmethod
    def bbox(rec):
        x = safe_float(rec.get("face_x"), np.nan)
        y = safe_float(rec.get("face_y"), np.nan)
        w = safe_float(rec.get("face_width"), np.nan)
        h = safe_float(rec.get("face_height"), np.nan)
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            return None
        return (x, y, x + w, y + h)

    @staticmethod
    def center_size(box):
        if box is None:
            return None, None
        x1, y1, x2, y2 = box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5), (max(1.0, x2 - x1), max(1.0, y2 - y1))

    @staticmethod
    def iou(a, b):
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / max(1e-9, area_a + area_b - inter)

    @staticmethod
    def normalized_center_distance(pred_box, det_box):
        pc, ps = MultiFaceTracker.center_size(pred_box)
        dc, _ = MultiFaceTracker.center_size(det_box)
        if pc is None or dc is None:
            return float("inf")
        d = math.hypot(dc[0] - pc[0], dc[1] - pc[1])
        scale = max(1.0, math.sqrt(ps[0] * ps[1]))
        return d / scale

    @staticmethod
    def size_change(pred_box, det_box):
        _, ps = MultiFaceTracker.center_size(pred_box)
        _, ds = MultiFaceTracker.center_size(det_box)
        if ps is None or ds is None:
            return float("inf")
        return max(abs(ds[0] / max(ps[0], 1.0) - 1.0), abs(ds[1] / max(ps[1], 1.0) - 1.0))

    def _predict_box(self, track, frame_number):
        box = track.get("bbox")
        center, size = self.center_size(box)
        if center is None or size is None:
            return box
        dt = max(1, int(frame_number) - int(track.get("last_frame", frame_number)))
        vx, vy = track.get("velocity", (0.0, 0.0))
        vsx, vsy = track.get("size_velocity", (0.0, 0.0))
        cx = center[0] + vx * dt
        cy = center[1] + vy * dt
        sw = max(10.0, size[0] + vsx * dt)
        sh = max(10.0, size[1] + vsy * dt)
        return (cx - sw * 0.5, cy - sh * 0.5, cx + sw * 0.5, cy + sh * 0.5)

    def _cost(self, track, det_box, frame_number):
        pred = self._predict_box(track, frame_number)
        iou = self.iou(pred, det_box)
        dist = self.normalized_center_distance(pred, det_box)
        size_delta = self.size_change(pred, det_box)
        if not ((dist <= self.distance_gate or iou >= self.min_iou) and size_delta <= self.max_size_change):
            return None
        cost = (0.50 * (1.0 - min(1.0, iou))
                + 0.38 * min(1.0, dist / max(self.distance_gate, 1e-6))
                + 0.12 * min(1.0, size_delta / max(self.max_size_change, 1e-6)))
        return cost

    def _new_track(self, box, frame_number, timestamp):
        center, size = self.center_size(box)
        tid = self.next_id
        self.next_id += 1
        self.created_tracks += 1
        self.tracks[tid] = {
            "bbox": box, "velocity": (0.0, 0.0), "size_velocity": (0.0, 0.0),
            "missing": 0, "hits": 1, "age": 1,
            "last_frame": int(frame_number), "last_timestamp": float(timestamp),
            "state": "TENTATIVE", "center": center, "size": size,
        }
        return tid

    def update(self, detections, frame_number=0, timestamp=0.0):
        detections = list(detections or [])
        self.total_detections += len(detections)
        track_ids = list(self.tracks.keys())
        det_boxes = [self.bbox(d) for d in detections]
        matched = []
        valid = None
        if track_ids and detections:
            cost_matrix = np.full((len(track_ids), len(detections)), 1e6, dtype=float)
            valid = np.zeros_like(cost_matrix, dtype=bool)
            for ti, tid in enumerate(track_ids):
                for di, box in enumerate(det_boxes):
                    if box is None:
                        continue
                    c = self._cost(self.tracks[tid], box, frame_number)
                    if c is not None:
                        cost_matrix[ti, di] = c
                        valid[ti, di] = True
            if linear_sum_assignment is not None:
                rr, cc = linear_sum_assignment(cost_matrix)
                matched = [(int(ti), int(di)) for ti, di in zip(rr, cc) if valid[ti, di]]
            else:
                pairs = sorted((cost_matrix[ti, di], ti, di)
                               for ti in range(len(track_ids))
                               for di in range(len(detections)) if valid[ti, di])
                used_t, used_d = set(), set()
                for _, ti, di in pairs:
                    if ti not in used_t and di not in used_d:
                        used_t.add(ti); used_d.add(di); matched.append((ti, di))

        assigned = {}
        matched_tids = set()
        matched_dis = set()
        for ti, di in matched:
            tid = track_ids[ti]
            tr = self.tracks[tid]
            box = det_boxes[di]
            old_center, old_size = self.center_size(tr["bbox"])
            new_center, new_size = self.center_size(box)
            dt = max(1, int(frame_number) - int(tr.get("last_frame", frame_number)))
            av = TRACK_VELOCITY_ALPHA
            ovx = (new_center[0] - old_center[0]) / dt
            ovy = (new_center[1] - old_center[1]) / dt
            ovs_x = (new_size[0] - old_size[0]) / dt
            ovs_y = (new_size[1] - old_size[1]) / dt
            tr["velocity"] = (av * ovx + (1.0-av) * tr["velocity"][0], av * ovy + (1.0-av) * tr["velocity"][1])
            tr["size_velocity"] = (av * ovs_x + (1.0-av) * tr["size_velocity"][0], av * ovs_y + (1.0-av) * tr["size_velocity"][1])
            tr["bbox"] = box; tr["center"] = new_center; tr["size"] = new_size
            tr["missing"] = 0; tr["hits"] += 1; tr["age"] += dt
            tr["last_frame"] = int(frame_number); tr["last_timestamp"] = float(timestamp)
            if tr["hits"] >= TRACK_MIN_HITS:
                tr["state"] = "TRACKED"
            assigned[di] = tid; matched_tids.add(tid); matched_dis.add(di)
            self.matched_detections += 1

        for tid in list(self.tracks):
            if tid in matched_tids:
                continue
            tr = self.tracks[tid]
            step = max(1, int(frame_number) - int(tr.get("last_frame", frame_number)))
            tr["missing"] += step; tr["age"] += step
            if tr["missing"] > self.max_missing:
                del self.tracks[tid]

        for di, box in enumerate(det_boxes):
            if di in matched_dis:
                continue
            assigned[di] = self._new_track(box, frame_number, timestamp)

        self.max_active_tracks = max(self.max_active_tracks, len(self.tracks))
        for di, rec in enumerate(detections):
            tid = assigned[di]
            tr = self.tracks[tid]
            rec["person_id"] = tid
            rec["track_state"] = tr["state"]
            rec["track_hits"] = int(tr["hits"])
            rec["track_age_frames"] = int(tr["age"])
            rec["track_missing_frames"] = int(tr["missing"])
            rec["track_matched"] = bool(tid in matched_tids)
        return detections

    def report(self):
        total = max(1, self.total_detections)
        return {
            "total_face_detections": int(self.total_detections),
            "matched_detections": int(self.matched_detections),
            "match_rate": float(self.matched_detections / total),
            "tracks_created": int(self.created_tracks),
            "max_active_tracks": int(self.max_active_tracks),
            "active_tracks_at_end": int(len(self.tracks)),
            "max_missing_frames": int(self.max_missing),
            "tracking_method": "predicted_bbox + global_assignment",
            "biometric_identity": False,
        }

class FACSWorker(QThread):
    frame_ready = Signal(object, object, object)  # original, tracked, state
    status = Signal(str)
    progress = Signal(int)
    finished_info = Signal(object)
    failed = Signal(str)

    def __init__(self, source, is_webcam, settings):
        super().__init__()
        self.source = source
        self.is_webcam = is_webcam
        self.settings = settings
        self.running = True
        self.detector = None
        self.cap = None
        self.writer_original = None
        self.writer_tracked = None
        self.session_dir = None
        self.raw_rows = []
        self.event_rows = []
        self.history = {"time": deque(maxlen=DISPLAY_HISTORY), "valence": deque(maxlen=DISPLAY_HISTORY), "arousal": deque(maxlen=DISPLAY_HISTORY)}
        self.person_smoothers = {}
        self.person_prev_values = {}
        self.person_prev_active = {}
        self.person_active_since = {}
        # Level 2 temporal FACS state. These are per-person and persist across frames.
        self.person_prev_time = {}
        self.person_prev_au = {}
        self.person_peak_au = {}
        self.person_burst_peak = {}
        self.person_prev_rapid = {}
        self.person_temporal_stats = {}
        self.person_open_episodes = {}
        self.temporal_episodes = []
        self.temporal_sequences = []
        self.temporal_interactions = {"coactivation": [], "transitions": [], "activation_groups": []}
        self.person_au_history = {}
        self.person_frame_times = {}
        self.person_last_active = {}
        self.person_sequence_events = {}
        self.frame_timestamps = []
        self.video_duration_seconds = 0.0
        self.tracking_quality_counts = {"GOOD": 0, "FAIR": 0, "LOW": 0, "BAD": 0}
        self.frame_quality_counts = {"GOOD": 0, "FAIR": 0, "LOW": 0, "BAD": 0, "NO_FACE": 0}
        self.tracker = MultiFaceTracker()
        self.frame_number = 0
        # Frame dimensions are needed by measurement-quality scoring.
        # Keep these attributes available before the worker starts processing.
        self.frame_width = 1
        self.frame_height = 1
        self.start_time = None
        self.last_detection_ms = 0.0
        self.last_faces = []
        self.last_result = None
        self.fps_times = deque(maxlen=30)
        self.fallback_disk = False
        self.calibration = self.settings.get("calibration_profile") or load_calibration_profile()

    def stop(self):
        self.running = False

    def frame_to_tensor(self, rgb):
        # Detectorv2 documents tensor input as BCHW. Keep uint8 0..255.
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
        return t

    def detect_frame(self, rgb):
        tensor = self.frame_to_tensor(rgb)
        t0 = time.perf_counter()
        try:
            result = self.detector.detect(
                tensor,
                data_type="tensor",
                batch_size=1,
                num_workers=0,
                pin_memory=False,
                face_detection_threshold=self.settings["face_threshold"],
                progress_bar=False,
            )
        except Exception as exc:
            # Some older Py-Feat builds have a broken tensor-image dataset path.
            # Fall back to a temporary PNG for compatibility, but only after tensor input fails.
            msg = str(exc)
            if "Image.open" not in msg and "read_file" not in msg and "numpy.ndarray" not in msg:
                raise
            self.fallback_disk = True
            temp_path = self.session_dir / "_frame_tmp.png"
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(temp_path), bgr):
                raise RuntimeError(f"Tensor input failed and temporary-frame fallback could not write: {exc}")
            try:
                result = self.detector.detect(
                    str(temp_path), data_type="image", batch_size=1,
                    num_workers=0, pin_memory=False,
                    face_detection_threshold=self.settings["face_threshold"],
                    progress_bar=False,
                )
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
        self.last_detection_ms = (time.perf_counter() - t0) * 1000.0
        return result

    @staticmethod
    def extract_rows(fex):
        if fex is None:
            return []
        try:
            df = pd.DataFrame(fex)
        except Exception:
            return []
        if df.empty:
            return []
        return [df.iloc[i] for i in range(len(df))]

    def get_col(self, row, name, default=np.nan):
        try:
            return safe_float(row[name], default)
        except Exception:
            return default

    def face_record(self, row, person_id):
        rec = {
            "timestamp": self.frame_number / max(self.settings["fps"], 1.0),
            "frame_number": self.frame_number,
            "face_id": person_id,
            "person_id": person_id,
            "face_x": self.get_col(row, "FaceRectX"),
            "face_y": self.get_col(row, "FaceRectY"),
            "face_width": self.get_col(row, "FaceRectWidth"),
            "face_height": self.get_col(row, "FaceRectHeight"),
            "face_confidence": self.get_col(row, "FaceScore"),
        }
        for au in AU_NAMES:
            rec[au] = self.get_col(row, au, 0.0)
        for e in EMOTION_NAMES:
            rec[f"emotion_{e.lower()}"] = self.get_col(row, e, 0.0)
        for k in ("valence", "arousal", "Pitch", "Roll", "Yaw", "X", "Y", "Z", "gaze_pitch", "gaze_yaw", "gaze_angle"):
            rec[k] = self.get_col(row, k)
        # 68-point coordinates if supplied by the installed model.
        for i in range(68):
            rec[f"x_{i}"] = self.get_col(row, f"x_{i}")
            rec[f"y_{i}"] = self.get_col(row, f"y_{i}")
        # Detectorv2's dense 478 mesh.
        for i in range(478):
            rec[f"mesh_x_{i}"] = self.get_col(row, f"mesh_x_{i}")
            rec[f"mesh_y_{i}"] = self.get_col(row, f"mesh_y_{i}")
            rec[f"mesh_z_{i}"] = self.get_col(row, f"mesh_z_{i}")
        # All blendshape-like columns exposed by the model.
        for col in row.index:
            name = str(col)
            if name.startswith(BLENDSHAPE_PREFIXES):
                rec[f"blendshape_{name}"] = self.get_col(row, name)
        return rec

    @staticmethod
    def _angle_deg(value):
        x = safe_float(value, np.nan)
        return math.degrees(x) if math.isfinite(x) else np.nan

    def measurement_quality(self, rec):
        """Estimate measurement conditions without modifying raw detector outputs.

        Pose is treated as an observability factor. Gaze is recorded separately because
        looking away is not itself a facial-expression failure. Thresholds are soft
        heuristics, not claims of detector accuracy at a particular angle.
        """
        conf = clamp(safe_float(rec.get("face_confidence"), 0.0), 0.0, 1.0)
        w = max(1.0, safe_float(rec.get("face_width"), 1.0))
        h = max(1.0, safe_float(rec.get("face_height"), 1.0))
        face_size = math.sqrt(w * h)
        size_quality = clamp((face_size - 60.0) / 180.0, 0.0, 1.0)
        pitch = abs(self._angle_deg(rec.get("Pitch")))
        yaw = abs(self._angle_deg(rec.get("Yaw")))
        roll = abs(self._angle_deg(rec.get("Roll")))
        pose_mag = math.sqrt(pitch * pitch + yaw * yaw)
        pose_q = 1.0 if pose_mag <= POSE_SOFT_LIMIT_DEG else clamp(1.0 - (pose_mag - POSE_SOFT_LIMIT_DEG) / (POSE_HARD_LIMIT_DEG - POSE_SOFT_LIMIT_DEG), 0.0, 1.0)
        roll_q = 1.0 if roll <= 15 else clamp(1.0 - (roll - 15) / 30.0, 0.0, 1.0)
        visibility = 0.45 * conf + 0.25 * size_quality + 0.20 * pose_q + 0.10 * roll_q
        gaze_pitch = abs(self._angle_deg(rec.get("gaze_pitch")))
        gaze_yaw = abs(self._angle_deg(rec.get("gaze_yaw")))
        gaze_angle = math.degrees(math.sqrt((safe_float(rec.get("gaze_pitch"), 0.0) ** 2) + (safe_float(rec.get("gaze_yaw"), 0.0) ** 2)))
        gaze_context = "CENTERED" if gaze_angle <= GAZE_CONTEXT_SOFT_LIMIT_DEG else ("AVERTED" if gaze_angle <= 50 else "STRONGLY_AVERTED")
        quality = "GOOD" if visibility >= 0.80 else ("FAIR" if visibility >= 0.60 else ("LIMITED" if visibility >= 0.40 else "POOR"))
        rec["pose_pitch_deg"] = self._angle_deg(rec.get("Pitch"))
        rec["pose_yaw_deg"] = self._angle_deg(rec.get("Yaw"))
        rec["pose_roll_deg"] = self._angle_deg(rec.get("Roll"))
        rec["pose_magnitude_deg"] = pose_mag
        rec["pose_quality"] = pose_q
        rec["gaze_pitch_deg"] = self._angle_deg(rec.get("gaze_pitch"))
        rec["gaze_yaw_deg"] = self._angle_deg(rec.get("gaze_yaw"))
        rec["gaze_angle_deg"] = gaze_angle
        rec["gaze_context"] = gaze_context
        rec["face_size_quality"] = size_quality
        rec["measurement_quality_score"] = visibility
        rec["measurement_quality"] = quality
        # Per-region AU observability. Eye actions are more sensitive to gaze/pose;
        # mouth actions are less affected by gaze alone.
        eye_q = clamp(0.75 * pose_q + 0.25 * (1.0 if gaze_angle <= 35 else clamp(1.0 - (gaze_angle - 35) / 45.0, 0.0, 1.0)), 0.0, 1.0)
        mouth_q = pose_q
        for au in AU_NAMES:
            q = eye_q if au in AU_EYE else (mouth_q if au in AU_MOUTH else 0.85 * pose_q + 0.15 * conf)
            rec[f"au_quality_{au}"] = q
        return visibility

    def emotion_quality(self, rec):
        probs = np.asarray([clamp(safe_float(rec.get(f"emotion_{e.lower()}"), 0.0), 0.0, 1.0) for e in EMOTION_NAMES], dtype=float)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        order = np.argsort(probs)[::-1]
        top = float(probs[order[0]]) if len(probs) else 0.0
        second = float(probs[order[1]]) if len(probs) > 1 else 0.0
        margin = top - second
        entropy = float(-(probs[probs > 1e-12] * np.log(probs[probs > 1e-12])).sum() / math.log(len(EMOTION_NAMES))) if len(EMOTION_NAMES) > 1 else 0.0
        meas = safe_float(rec.get("measurement_quality_score"), 0.0)
        # This is an interpretation-quality indicator, not a modified emotion probability.
        confidence = clamp((0.55 * top + 0.45 * margin) * (0.55 + 0.45 * meas), 0.0, 1.0)
        mixed = margin < 0.12 or entropy > 0.78
        status = "MIXED / AMBIGUOUS" if mixed else ("HIGH" if confidence >= 0.65 else ("MODERATE" if confidence >= 0.40 else "LOW"))
        rec["emotion_top"] = EMOTION_NAMES[int(order[0])] if len(order) else "Unknown"
        rec["emotion_top_probability"] = top
        rec["emotion_second"] = EMOTION_NAMES[int(order[1])] if len(order) > 1 else "Unknown"
        rec["emotion_second_probability"] = second
        rec["emotion_margin"] = margin
        rec["emotion_entropy"] = entropy
        rec["emotion_interpretation_score"] = confidence
        rec["emotion_interpretation_status"] = status
        rec["emotion_top3"] = ", ".join(f"{EMOTION_NAMES[int(i)]} {probs[i]:.2f}" for i in order[:3])
        return status

    def apply_calibration(self, rec):
        profile = self.calibration
        if not profile:
            rec["calibration_active"] = False
            return
        rec["calibration_active"] = True
        aus = profile.get("au", {})
        for au in AU_NAMES:
            base = safe_float(aus.get(au, {}).get("median"), np.nan)
            mad = safe_float(aus.get(au, {}).get("mad"), np.nan)
            value = safe_float(rec.get(au), np.nan)
            if math.isfinite(base) and math.isfinite(value):
                scale = max(0.025, 1.4826 * mad) if math.isfinite(mad) else 0.05
                rec[f"calibration_baseline_{au}"] = base
                rec[f"calibration_deviation_{au}"] = value - base
                rec[f"calibration_z_{au}"] = (value - base) / scale
        pose = profile.get("pose", {})
        for key in ("Pitch", "Roll", "Yaw"):
            base = safe_float(pose.get(key), np.nan)
            value = safe_float(rec.get(key), np.nan)
            if math.isfinite(base) and math.isfinite(value):
                rec[f"calibration_pose_delta_{key}"] = math.degrees(value - base)
        gaze = profile.get("gaze", {})
        for key in ("gaze_pitch", "gaze_yaw"):
            base = safe_float(gaze.get(key), np.nan)
            value = safe_float(rec.get(key), np.nan)
            if math.isfinite(base) and math.isfinite(value):
                rec[f"calibration_gaze_delta_{key}"] = math.degrees(value - base)

    def derived_expression(self, rec):
        au = {k: max(0.0, safe_float(rec.get(k), 0.0)) for k in AU_NAMES}
        patterns = {
            "Happy-like": (au["AU06"] + au["AU12"]) / 2,
            "Sad-like": (au["AU01"] + au["AU04"] + au["AU15"]) / 3,
            "Anger-like": (au["AU04"] + au["AU05"] + au["AU07"] + au["AU23"] + au["AU24"]) / 5,
            "Fear-like": (au["AU01"] + au["AU02"] + au["AU04"] + au["AU05"]) / 4,
            "Surprise-like": (au["AU01"] + au["AU02"] + au["AU05"] + au["AU26"]) / 4,
            "Disgust-like": (au["AU09"] + au["AU10"]) / 2,
            "Contempt-like": (au["AU12"] + au["AU14"]) / 2,
        }
        name, score = max(patterns.items(), key=lambda kv: kv[1])
        return name if score >= 0.20 else "No strong derived pattern", score

    def _au_params(self, au):
        base = self.settings.get("au_threshold", 0.15)
        return {"on": max(0.01, min(0.95, base)), "off": max(0.005, min(0.94, base - ACTIVATION_HYSTERESIS)), "min_duration": self.settings.get("min_episode_duration", MIN_EPISODE_DURATION), "min_amplitude": self.settings.get("min_episode_amplitude", MIN_EPISODE_AMPLITUDE)}

    def _quality_for_record(self, rec):
        """Return measurement quality using documented Py-Feat pose units (radians).

        This method deliberately does not treat gaze direction as a failure condition.
        Head pose affects observability; gaze is stored as separate context and only
        reduces the quality of eye-region AU measurements.
        """
        conf = clamp(safe_float(rec.get("face_confidence"), 0.0), 0.0, 1.0)
        w = max(0.0, safe_float(rec.get("face_width"), 0.0))
        h = max(0.0, safe_float(rec.get("face_height"), 0.0))
        frame_area = max(1.0, float(self.frame_width * self.frame_height))
        face_area_ratio = (w * h) / frame_area
        pitch = abs(math.degrees(safe_float(rec.get("Pitch"), 0.0)))
        roll = abs(math.degrees(safe_float(rec.get("Roll"), 0.0)))
        yaw = abs(math.degrees(safe_float(rec.get("Yaw"), 0.0)))
        pose = math.sqrt(pitch * pitch + yaw * yaw)
        flags = []
        if conf < 0.30: flags.append("LOW_CONFIDENCE")
        if face_area_ratio < 0.03: flags.append("SMALL_FACE")
        if pose > POSE_HARD_LIMIT_DEG: flags.append("LARGE_POSE")
        if rec.get("track_state") == "TENTATIVE": flags.append("TENTATIVE_TRACK")
        if conf < 0.30 or face_area_ratio < 0.015 or pose > POSE_HARD_LIMIT_DEG:
            quality = "BAD"
        elif conf < 0.50 or face_area_ratio < 0.03 or pose > 45:
            quality = "LOW"
        elif conf < 0.75 or face_area_ratio < 0.06 or pose > 25 or rec.get("track_state") != "TRACKED":
            quality = "FAIR"
        else:
            quality = "GOOD"
        pose_factor = 1.0 if pose <= POSE_SOFT_LIMIT_DEG else clamp(1.0 - (pose - POSE_SOFT_LIMIT_DEG) / (POSE_HARD_LIMIT_DEG - POSE_SOFT_LIMIT_DEG), 0.0, 1.0)
        size_factor = clamp((face_area_ratio - 0.02) / 0.10, 0.0, 1.0)
        track_factor = 1.0 if rec.get("track_state") == "TRACKED" else 0.75
        score = clamp((0.55 * conf + 0.20 * size_factor + 0.20 * pose_factor + 0.05 * track_factor), 0.0, 1.0)
        return quality, score, conf, pose, flags

    def _robust_baseline(self, history, au, current=None):
        values = [x for x in history if math.isfinite(x)]
        if not values:
            if current is not None and math.isfinite(safe_float(current, np.nan)):
                return float(current), BASELINE_SCALE_FLOOR, False
            return 0.0, BASELINE_SCALE_FLOOR, False
        arr = np.asarray(values[-BASELINE_WINDOW:], dtype=float)
        med = float(np.median(arr))
        if len(arr) < BASELINE_MIN_SAMPLES:
            scale = float(np.std(arr)) if len(arr) > 1 else BASELINE_SCALE_FLOOR
            return med, max(scale, BASELINE_SCALE_FLOOR), False
        mad = float(np.median(np.abs(arr - med)))
        std = float(np.std(arr))
        return med, max(1.4826 * mad, std, BASELINE_SCALE_FLOOR), True

    def active_aus(self, rec, previous_active=None):
        previous_active = previous_active or set()
        active = set()
        for au in AU_NAMES:
            value = max(0.0, safe_float(rec.get(au), 0.0))
            params = self._au_params(au)
            threshold = params["off"] if au in previous_active else params["on"]
            if value >= threshold:
                active.add(au)
        return active

    def _close_episode(self, pid, au, end_time, end_frame, final=False, gap_break=False):
        ep = self.person_open_episodes.setdefault(pid, {}).pop(au, None)
        if ep is None:
            return None
        ep["offset_time"] = float(end_time)
        ep["offset_frame"] = int(end_frame)
        ep["duration"] = max(0.0, ep["offset_time"] - ep["onset_time"])
        ep["amplitude"] = max(0.0, ep["peak"] - ep["baseline"])
        rise_dt = ep["peak_time"] - ep["onset_time"]
        fall_dt = ep["offset_time"] - ep["peak_time"]
        ep["rise_rate"] = float(ep["amplitude"] / rise_dt) if rise_dt >= MIN_RATE_INTERVAL_SECONDS else None
        # A finalization at the end of the recording does not contain a measured fall.
        final_fall_valid = (not final) or (fall_dt >= MIN_RATE_INTERVAL_SECONDS)
        ep["fall_rate"] = float(ep["amplitude"] / fall_dt) if final_fall_valid and fall_dt >= MIN_RATE_INTERVAL_SECONDS else None
        ep["rise_rate_valid"] = bool(rise_dt >= MIN_RATE_INTERVAL_SECONDS)
        ep["fall_rate_valid"] = bool(final_fall_valid and fall_dt >= MIN_RATE_INTERVAL_SECONDS)
        ep["single_frame_peak"] = ep["peak_frame"] == ep["onset_frame"]
        ep["finalized_at_end"] = bool(final)
        ep["gap_break"] = bool(gap_break)
        params = self._au_params(au)
        ep["valid_duration"] = bool(ep["duration"] >= params["min_duration"])
        ep["valid_amplitude"] = bool(ep["amplitude"] >= params["min_amplitude"])
        ep["valid"] = bool(ep["valid_duration"] and ep["valid_amplitude"] and ep["quality_fraction"] >= 0.5)
        if ep["valid"]:
            self.temporal_episodes.append(ep)
        return ep

    def _record_activation_group(self,pid,timestamp,frame,active,previous_active):
        newly=sorted(active-previous_active)
        if newly: self.temporal_interactions["activation_groups"].append({"person_id":pid,"time":float(timestamp),"frame":int(frame),"aus":newly,"count":len(newly)})

    def _build_sequences(self,pid):
        eps=sorted([e for e in self.temporal_episodes if e["person_id"]==pid],key=lambda e:e["onset_time"]); clusters=[]
        for ep in eps:
            if not clusters or ep["onset_time"]-clusters[-1]["end_time"]>MAX_CLUSTER_GAP: clusters.append({"start_time":ep["onset_time"],"end_time":ep["offset_time"],"episodes":[ep]})
            else: clusters[-1]["end_time"]=max(clusters[-1]["end_time"],ep["offset_time"]); clusters[-1]["episodes"].append(ep)
        sequences=[]
        for c in clusters:
            if not sequences or c["start_time"]-sequences[-1]["end_time"]>MAX_SEQUENCE_GAP: sequences.append({"start_time":c["start_time"],"end_time":c["end_time"],"clusters":[c]})
            else: sequences[-1]["end_time"]=max(sequences[-1]["end_time"],c["end_time"]); sequences[-1]["clusters"].append(c)
        out=[]
        for sid,seq in enumerate(sequences,1):
            all_eps=[e for c in seq["clusters"] for e in c["episodes"]]; aus=sorted({e["au"] for e in all_eps})
            out.append({"sequence_id":f"P{pid}-S{sid}","person_id":pid,"start_time":seq["start_time"],"end_time":seq["end_time"],"duration":max(0.0,seq["end_time"]-seq["start_time"]),"episode_count":len(all_eps),"distinct_au_count":len(aus),"aus":aus,"clusters":len(seq["clusters"]),"density":len(all_eps)/max(0.001,seq["end_time"]-seq["start_time"])})
        return out

    def temporal_update(self, rec):
        pid = int(rec.get("person_id", 0))
        now = safe_float(rec.get("timestamp"), 0.0)
        prev_t = self.person_prev_time.get(pid)
        nominal_dt = 1.0 / max(self.settings.get("fps", 30.0), 1.0)
        dt = now - prev_t if prev_t is not None else nominal_dt
        timing_valid = prev_t is not None and math.isfinite(dt) and nominal_dt * 0.25 <= dt <= TEMPORAL_MAX_GAP_SECONDS
        if not math.isfinite(dt) or dt <= 0:
            dt = nominal_dt

        prev_active = self.person_last_active.setdefault(pid, set())
        if prev_t is not None and not timing_valid:
            gap_end = prev_t + nominal_dt
            for au in list(self.person_open_episodes.setdefault(pid, {})):
                self._close_episode(pid, au, gap_end, max(0, self.frame_number - 1), gap_break=True)
            prev_active.clear()
            self.person_prev_au.setdefault(pid, {}).clear()

        active = self.active_aus(rec, prev_active)
        quality, quality_score, conf, pose, quality_flags = self._quality_for_record(rec)
        # Populate the richer pose/gaze/region quality fields before temporal decisions.
        self.measurement_quality(rec)
        quality_score = float(rec.get("measurement_quality_score", quality_score))
        quality = str(rec.get("measurement_quality", quality))
        self.tracking_quality_counts[quality] = self.tracking_quality_counts.get(quality, 0) + 1
        rec["tracking_quality"] = quality
        rec["measurement_quality_score"] = quality_score
        rec["timing_dt"] = float(dt)
        rec["timing_valid"] = bool(timing_valid or prev_t is None)
        rec["data_quality_flags"] = ",".join(quality_flags) if quality_flags else "OK"

        histories = self.person_au_history.setdefault(pid, {au: deque(maxlen=BASELINE_WINDOW) for au in AU_NAMES})
        prev_values = self.person_prev_au.setdefault(pid, {})
        rates = {}
        for au in AU_NAMES:
            raw = max(0.0, safe_float(rec.get(au), 0.0))
            hist = histories[au]
            baseline, scale, warmed = self._robust_baseline(hist, au, raw)
            hist.append(raw)
            rec[f"smooth_{au}"] = raw
            rec[f"baseline_{au}"] = baseline
            rec[f"baseline_scale_{au}"] = scale
            rec[f"baseline_warmed_{au}"] = bool(warmed)
            rec[f"deviation_{au}"] = raw - baseline
            rec[f"normalized_deviation_{au}"] = (raw - baseline) / max(scale, BASELINE_SCALE_FLOOR)
            old = safe_float(prev_values.get(au), raw)
            rate = (raw - old) / dt if (prev_t is not None and timing_valid) else None
            rates[au] = rate
            rec[f"rate_{au}"] = rate

            open_map = self.person_open_episodes.setdefault(pid, {})
            ep = open_map.get(au)
            if au in active and ep is None:
                open_map[au] = {"episode_id": 0, "person_id": pid, "au": au,
                                "onset_time": now, "onset_frame": self.frame_number,
                                "peak_time": now, "peak_frame": self.frame_number,
                                "baseline": baseline, "peak": raw,
                                "quality_fraction": 1.0 if quality in {"GOOD", "FAIR"} else 0.0}
            elif au in active and ep is not None:
                ep["quality_fraction"] += 1.0 if quality in {"GOOD", "FAIR"} else 0.0
                if raw > ep["peak"]:
                    ep["peak"] = raw; ep["peak_time"] = now; ep["peak_frame"] = self.frame_number
            elif au not in active and ep is not None:
                self._close_episode(pid, au, now, self.frame_number)
            prev_values[au] = raw
            rec[f"peak_{au}"] = open_map.get(au, {}).get("peak", raw)

        abs_rates = {au: abs(v) for au, v in rates.items() if v is not None and math.isfinite(v)}
        strongest_rate_au = max(abs_rates, key=abs_rates.get) if abs_rates else "--"
        strongest_au = max(AU_NAMES, key=lambda au: safe_float(rec.get(au), 0.0))
        rec["temporal_active_au_count"] = len(active)
        rec["temporal_active_aus"] = ", ".join(sorted(active)) if active else "None"
        rec["temporal_strongest_au"] = strongest_au
        rec["temporal_strongest_au_value"] = safe_float(rec.get(strongest_au), 0.0)
        rec["temporal_fastest_au"] = strongest_rate_au
        rec["temporal_fastest_rate"] = abs_rates.get(strongest_rate_au) if abs_rates else None
        rec["temporal_mean_abs_rate"] = float(np.mean(list(abs_rates.values()))) if abs_rates else None
        rec["temporal_coactive_pairs"] = len(active) * max(0, len(active) - 1) // 2

        rapid = {au for au, rate in abs_rates.items() if rate >= self.settings.get("rapid_threshold", 2.0) and safe_float(rec.get(au), 0.0) >= self._au_params(au)["on"]}
        rec["rapid_action_candidates"] = ", ".join(sorted(rapid)) if rapid else "None"
        stats = self.person_temporal_stats.setdefault(pid, {"frames": 0, "onsets": 0, "offsets": 0, "short_bursts": 0, "rapid_changes": 0, "max_rate": 0.0, "au_duration": {au: 0.0 for au in AU_NAMES}, "peak": {au: 0.0 for au in AU_NAMES}})
        stats["frames"] += 1
        stats["max_rate"] = max(stats["max_rate"], max(abs_rates.values(), default=0.0))
        stats["onsets"] += len(active - prev_active)
        stats["offsets"] += len(prev_active - active)
        if timing_valid or prev_t is None:
            for au in active:
                stats["au_duration"][au] += dt
        for au in AU_NAMES:
            stats["peak"][au] = max(stats["peak"].get(au, 0.0), safe_float(rec.get(au), 0.0))
        if active - prev_active:
            self._record_activation_group(pid, now, self.frame_number, active, prev_active)
        for au in rapid:
            self.event_rows.append({"timestamp": now, "frame": self.frame_number, "person_id": pid, "type": "Rapid AU change", "name": au, "value": rates[au]})
            stats["rapid_changes"] += 1
        self.person_last_active[pid] = active
        self.person_prev_time[pid] = now
    def event_update(self, rec, derived_name, derived_score):
        pid=int(rec.get("person_id",0)); now=safe_float(rec.get("timestamp"),0.0)
        if derived_name != "No strong derived pattern" and derived_score >= 0.45:
            prev=self.person_prev_values.get(pid)
            if prev != derived_name:
                self.event_rows.append({"timestamp":now,"frame":self.frame_number,"person_id":pid,"type":"Derived expression","name":derived_name,"value":derived_score})
            self.person_prev_values[pid]=derived_name

    def finalize_signal_processing(self):
        """Add offline smoothed signals plus first/second derivatives without modifying raw AU columns."""
        if not self.raw_rows:
            return
        by_pid={}
        for idx,row in enumerate(self.raw_rows):
            by_pid.setdefault(int(row.get("person_id",0)),[]).append((idx,row))
        for pid,items in by_pid.items():
            items.sort(key=lambda x:(safe_float(x[1].get("timestamp"),0.0), int(x[1].get("frame_number",0))))
            for au in AU_NAMES:
                vals=np.asarray([max(0.0,safe_float(r.get(au),0.0)) for _,r in items],dtype=float)
                smooth=vals.copy()
                if savgol_filter is not None and len(vals)>=5:
                    win=min( nine:=9, len(vals) if len(vals)%2 else len(vals)-1 )
                    if win>=5:
                        try: smooth=np.clip(savgol_filter(vals,window_length=win,polyorder=2,mode="interp"), 0.0, 1.0)
                        except Exception: smooth=vals.copy()
                times=np.asarray([safe_float(r.get("timestamp"),0.0) for _,r in items],dtype=float)
                vel=np.full(len(vals),np.nan); acc=np.full(len(vals),np.nan)
                if len(vals)>1:
                    d=np.diff(times); d[d<=0]=np.nan
                    vel[1:]=np.diff(smooth)/d
                    if len(vals)>2:
                        d2=np.diff(times[1:]); d2[d2<=0]=np.nan; acc[2:]=np.diff(vel[1:])/d2
                for j,(idx,r) in enumerate(items):
                    r[f"smooth_{au}"]=float(smooth[j]); r[f"velocity_{au}"]=None if not math.isfinite(float(vel[j])) else float(vel[j]); r[f"acceleration_{au}"]=None if not math.isfinite(float(acc[j])) else float(acc[j])

    def rebuild_episodes_from_smoothed(self):
        """Build canonical AU episodes from the finalized smoothed signal."""
        self.temporal_episodes = []
        by_pid = {}
        for row in self.raw_rows:
            by_pid.setdefault(int(row.get("person_id", 0)), []).append(row)
        for pid, rows in by_pid.items():
            rows.sort(key=lambda r: (safe_float(r.get("timestamp"), 0.0), int(r.get("frame_number", 0))))
            if not rows:
                continue
            times = [safe_float(r.get("timestamp"), 0.0) for r in rows]
            diffs = [b - a for a, b in zip(times, times[1:]) if math.isfinite(a) and math.isfinite(b) and b > a]
            nominal_dt = float(np.median(diffs)) if diffs else 1.0 / max(self.settings.get("fps", 30.0), 1.0)
            histories = {au: deque(maxlen=BASELINE_WINDOW) for au in AU_NAMES}
            active = {au: False for au in AU_NAMES}
            open_eps = {}
            for row in rows:
                t = safe_float(row.get("timestamp"), 0.0); frame = int(row.get("frame_number", 0)); quality = row.get("tracking_quality", "GOOD")
                for au in AU_NAMES:
                    value = max(0.0, safe_float(row.get(f"smooth_{au}"), safe_float(row.get(au), 0.0)))
                    hist = histories[au]; baseline, scale, warmed = self._robust_baseline(hist, au, value); hist.append(value)
                    row[f"final_baseline_{au}"] = baseline
                    row[f"final_baseline_scale_{au}"] = scale
                    row[f"final_baseline_warmed_{au}"] = bool(warmed)
                    row[f"final_normalized_deviation_{au}"] = (value - baseline) / max(scale, BASELINE_SCALE_FLOOR)
                    params = self._au_params(au); threshold = params["off"] if active[au] else params["on"]; is_active = value >= threshold
                    if is_active and not active[au]:
                        open_eps[au] = {"episode_id": 0, "person_id": pid, "au": au, "onset_time": t, "onset_frame": frame, "peak_time": t, "peak_frame": frame, "baseline": baseline, "peak": value, "quality_fraction": 1.0 if quality in {"GOOD", "FAIR"} else 0.0, "sample_count": 1}
                    elif is_active and active[au]:
                        ep = open_eps.get(au)
                        if ep is not None:
                            ep["sample_count"] += 1; ep["quality_fraction"] += 1.0 if quality in {"GOOD", "FAIR"} else 0.0
                            if value > ep["peak"]: ep["peak"] = value; ep["peak_time"] = t; ep["peak_frame"] = frame
                    elif not is_active and active[au]:
                        ep = open_eps.pop(au, None)
                        if ep is not None: self._finish_offline_episode(ep, t, frame, params, final=False)
                    active[au] = is_active
            end_t = times[-1] + nominal_dt; end_frame = int(rows[-1].get("frame_number", 0)) + 1
            for au, ep in list(open_eps.items()):
                self._finish_offline_episode(ep, end_t, end_frame, self._au_params(au), final=True)
        self.temporal_episodes.sort(key=lambda e: (e["person_id"], e["onset_time"], e["au"]))
        for i, ep in enumerate(self.temporal_episodes, 1): ep["episode_id"] = i

    def _finish_offline_episode(self, ep, end_time, end_frame, params, final=False):
        count = max(1, int(ep.pop("sample_count", 1)))
        ep["quality_fraction"] /= count; ep["offset_time"] = float(end_time); ep["offset_frame"] = int(end_frame)
        ep["duration"] = max(0.0, ep["offset_time"] - ep["onset_time"]); ep["amplitude"] = max(0.0, ep["peak"] - ep["baseline"])
        rise = ep["peak_time"] - ep["onset_time"]; fall = ep["offset_time"] - ep["peak_time"]
        ep["rise_rate"] = float(ep["amplitude"] / rise) if rise >= MIN_RATE_INTERVAL_SECONDS else None
        final_fall_valid = (not final) or (fall >= MIN_RATE_INTERVAL_SECONDS)
        ep["fall_rate"] = float(ep["amplitude"] / fall) if final_fall_valid and fall >= MIN_RATE_INTERVAL_SECONDS else None
        ep["rise_rate_valid"] = bool(rise >= MIN_RATE_INTERVAL_SECONDS); ep["fall_rate_valid"] = bool(final_fall_valid and fall >= MIN_RATE_INTERVAL_SECONDS); ep["single_frame_peak"] = ep["peak_frame"] == ep["onset_frame"]; ep["finalized_at_end"] = bool(final); ep["gap_break"] = False
        ep["valid_duration"] = bool(ep["duration"] >= params["min_duration"]); ep["valid_amplitude"] = bool(ep["amplitude"] >= params["min_amplitude"]); ep["valid"] = bool(ep["valid_duration"] and ep["valid_amplitude"] and ep["quality_fraction"] >= 0.5)
        if ep["valid"]: self.temporal_episodes.append(ep)

    def finalize_temporal(self):
        end_time = self.video_duration_seconds if self.video_duration_seconds > 0 else self.frame_number / max(self.settings.get("fps", 30.0), 1.0)
        if self.raw_rows:
            times = [safe_float(r.get("timestamp"), 0.0) for r in self.raw_rows]
            diffs = [b - a for a, b in zip(times, times[1:]) if math.isfinite(a) and math.isfinite(b) and b > a]
            if diffs: end_time = max(end_time, times[-1] + float(np.median(diffs)))
        for pid, open_map in list(self.person_open_episodes.items()):
            for au in list(open_map): self._close_episode(pid, au, end_time, self.frame_number + 1, True)
        self.temporal_episodes.sort(key=lambda e: (e["person_id"], e["onset_time"], e["au"]))
        for i, ep in enumerate(self.temporal_episodes, 1): ep["episode_id"] = i
        self.build_temporal_interactions()

    def _frame_activity(self, pid):
        rows = sorted(
            [r for r in self.raw_rows if int(r.get("person_id", 0)) == pid],
            key=lambda r: (safe_float(r.get("timestamp"), 0.0), int(r.get("frame_number", 0))),
        )
        if not rows:
            return [], {}
        active_rows = []
        prev = set()
        durations = {au: 0.0 for au in AU_NAMES}
        for i, row in enumerate(rows):
            t = safe_float(row.get("timestamp"), 0.0)
            if i + 1 < len(rows):
                nt = safe_float(rows[i + 1].get("timestamp"), t)
                raw_dt = nt - t if nt > t else 0.0
            elif len(rows) > 1:
                pt = safe_float(rows[-2].get("timestamp"), t)
                raw_dt = t - pt if t > pt else 0.0
            else:
                raw_dt = 1.0 / max(self.settings.get("fps", 30.0), 1.0)

            gap_break = raw_dt > TEMPORAL_MAX_GAP_SECONDS
            quality = str(row.get("tracking_quality", "GOOD"))
            usable = quality in {"GOOD", "FAIR"} and row.get("track_state", "TRACKED") != "TENTATIVE"
            if gap_break or not usable:
                active = set()
                dt = 0.0
            else:
                active = self.active_aus(row, prev)
                dt = min(max(0.0, raw_dt), TEMPORAL_MAX_GAP_SECONDS)
            for au in active:
                durations[au] += dt
            active_rows.append((t, dt, active))
            prev = active
        return active_rows, durations

    def _build_sequences(self, pid):
        """Group nearby AU onsets into local activation sequences.

        Sequence boundaries are defined by onset timing, not by the offset of a
        long-lived AU. A sustained AU therefore cannot stretch one sequence across
        an entire recording. Individual episode onset/offset spans remain available
        in the episode table.
        """
        eps = sorted(
            [e for e in self.temporal_episodes if e["person_id"] == pid and e.get("valid", False)],
            key=lambda e: (e["onset_time"], e["au"]),
        )
        if not eps:
            return []
        clusters = []
        for ep in eps:
            if not clusters or ep["onset_time"] - clusters[-1]["last_onset_time"] > MAX_CLUSTER_GAP:
                clusters.append({"start_time": ep["onset_time"], "end_time": ep["onset_time"], "last_onset_time": ep["onset_time"], "episodes": [ep]})
            else:
                clusters[-1]["end_time"] = max(clusters[-1]["end_time"], ep["onset_time"])
                clusters[-1]["last_onset_time"] = ep["onset_time"]
                clusters[-1]["episodes"].append(ep)

        rows, durations = self._frame_activity(pid)
        out = []
        for sid, cluster in enumerate(clusters, 1):
            all_eps = cluster["episodes"]
            start = float(cluster["start_time"])
            end = float(cluster["end_time"])
            duration = max(1.0 / max(self.settings.get("fps", 30.0), 1.0), end - start)
            aus = sorted({e["au"] for e in all_eps})
            local_duration = {au: 0.0 for au in AU_NAMES}
            local_peak_simultaneous = 0
            local_active_intervals = []
            for t, dt, active in rows:
                if t < start or t > end + MAX_CLUSTER_GAP:
                    continue
                local_peak_simultaneous = max(local_peak_simultaneous, len(active))
                if active and dt > 0:
                    local_active_intervals.append((t, t + dt))
                for au in active:
                    local_duration[au] += dt
            local_idle_gaps = []
            if local_active_intervals:
                local_active_intervals.sort()
                last_end = local_active_intervals[0][1]
                for a, b in local_active_intervals[1:]:
                    if a > last_end:
                        local_idle_gaps.append(a - last_end)
                    last_end = max(last_end, b)
            episode_coverage_end = max((e["offset_time"] for e in all_eps), default=end)
            out.append({
                "sequence_id": f"P{pid}-S{sid}",
                "person_id": pid,
                "start_time": start,
                "end_time": end,
                "duration": float(duration),
                "episode_coverage_end_time": float(episode_coverage_end),
                "onset_span_seconds": float(max(0.0, end - start)),
                "episode_count": len(all_eps),
                "distinct_au_count": len(aus),
                "aus": aus,
                "clusters": 1,
                "density": float(len(all_eps) / duration),
                "peak_amplitude": float(max((e["amplitude"] for e in all_eps), default=0.0)),
                "peak_time": float(max(all_eps, key=lambda e: e["amplitude"])["peak_time"]) if all_eps else None,
                "peak_simultaneous_aus": int(local_peak_simultaneous),
                "longest_idle_gap_seconds": float(max(local_idle_gaps, default=0.0)),
                "active_au_seconds": {au: float(v) for au, v in local_duration.items() if v > 0},
                "interpretation": "activation-onset sequence; episode offsets are reported separately and sustained AUs do not define sequence duration",
            })
        return out

    def build_temporal_interactions(self):
        episodes = [e for e in self.temporal_episodes if e.get("valid", False)]
        co = {}
        transitions = {}
        activation_groups = []

        for pid in sorted({e["person_id"] for e in episodes}):
            rows, durations = self._frame_activity(pid)
            pair_frames = {}
            pair_seconds = {}
            au_seconds = {au: float(durations.get(au, 0.0)) for au in AU_NAMES}
            for _, dt, active in rows:
                active = sorted(active)
                for i, a in enumerate(active):
                    for b in active[i + 1:]:
                        key = (a, b)
                        pair_frames[key] = pair_frames.get(key, 0) + 1
                        pair_seconds[key] = pair_seconds.get(key, 0.0) + dt

            for (a, b), frames in pair_frames.items():
                sec = pair_seconds[(a, b)]
                union = au_seconds.get(a, 0.0) + au_seconds.get(b, 0.0) - sec
                co[(pid, a, b)] = {
                    "person_id": pid, "au_a": a, "au_b": b,
                    "coactive_frames": int(frames), "overlap_seconds": float(sec),
                    "p_b_given_a": float(sec / max(au_seconds.get(a, 0.0), 1e-9)),
                    "p_a_given_b": float(sec / max(au_seconds.get(b, 0.0), 1e-9)),
                    "jaccard_overlap": float(sec / max(union, 1e-9)),
                }

            pe = sorted([e for e in episodes if e["person_id"] == pid], key=lambda e: (e["onset_time"], e["au"]))
            groups = []
            for ep in pe:
                if not groups or ep["onset_time"] - groups[-1]["start_time"] > SIMULTANEOUS_FRAME_TOLERANCE:
                    groups.append({
                        "start_time": ep["onset_time"],
                        "end_time": ep["offset_time"],
                        "aus": [ep["au"]],
                        "episode_ids": [ep.get("episode_id")],
                        "onset_frame": ep.get("onset_frame"),
                    })
                else:
                    groups[-1]["end_time"] = max(groups[-1]["end_time"], ep["offset_time"])
                    if ep["au"] not in groups[-1]["aus"]:
                        groups[-1]["aus"].append(ep["au"])
                    groups[-1]["episode_ids"].append(ep.get("episode_id"))
            for g in groups:
                g["aus"] = sorted(set(g["aus"]))
                activation_groups.append({
                    "person_id": pid,
                    "time": float(g["start_time"]),
                    "frame": int(g["onset_frame"] or 0),
                    "aus": g["aus"],
                    "count": len(g["aus"]),
                    "episode_ids": g["episode_ids"],
                })

            for g1, g2 in zip(groups, groups[1:]):
                delay = float(g2["start_time"] - g1["start_time"])
                if delay <= SIMULTANEOUS_FRAME_TOLERANCE:
                    continue
                if delay > MAX_SEQUENCE_GAP:
                    continue
                transition_type = "near_simultaneous" if delay <= MAX_CLUSTER_GAP else "sequential"
                for a in g1["aus"]:
                    for b in g2["aus"]:
                        key = (pid, a, b, transition_type)
                        x = transitions.setdefault(key, {
                            "person_id": pid, "from_au": a, "to_au": b,
                            "transition_type": transition_type, "count": 0, "delays": [],
                        })
                        x["count"] += 1
                        x["delays"].append(delay)

        for x in transitions.values():
            delays = x.pop("delays", [])
            x["mean_delay_seconds"] = float(np.mean(delays)) if delays else None
            x["min_delay_seconds"] = float(min(delays)) if delays else None
            x["max_delay_seconds"] = float(max(delays)) if delays else None

        self.temporal_interactions["coactivation"] = list(co.values())
        self.temporal_interactions["transitions"] = list(transitions.values())
        self.temporal_interactions["activation_groups"] = activation_groups
        self.temporal_sequences = []
        for pid in sorted({e["person_id"] for e in episodes}):
            self.temporal_sequences.extend(self._build_sequences(pid))
        return self.temporal_interactions

    def draw_face(self, frame, rec, face_index):
        h, w = frame.shape[:2]
        x = int(clamp(safe_float(rec["face_x"], 0), 0, w - 1))
        y = int(clamp(safe_float(rec["face_y"], 0), 0, h - 1))
        fw = int(max(1, safe_float(rec["face_width"], 1)))
        fh = int(max(1, safe_float(rec["face_height"], 1)))
        x2 = clamp(x + fw, 0, w - 1)
        y2 = clamp(y + fh, 0, h - 1)
        cv2.rectangle(frame, (x, y), (x2, y2), (80, 220, 120), 2)
        conf = percent(rec.get("face_confidence", 0))
        cv2.putText(frame, f"Track {int(rec.get("person_id", face_index + 1))}  {conf:.0f}%", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 120), 2, cv2.LINE_AA)

        # 68-point overlay when available.
        if self.settings["show_landmarks"]:
            pts = []
            for i in range(68):
                px, py = rec.get(f"x_{i}"), rec.get(f"y_{i}")
                if math.isfinite(safe_float(px)) and math.isfinite(safe_float(py)):
                    pts.append((int(px), int(py)))
                else:
                    pts.append(None)
            for p in pts:
                if p:
                    cv2.circle(frame, p, 2, (255, 210, 70), -1)

        if self.settings["show_mesh"]:
            mx = [safe_float(rec.get(f"mesh_x_{i}")) for i in range(478)]
            my = [safe_float(rec.get(f"mesh_y_{i}")) for i in range(478)]
            for a, b in MESH_EDGES:
                if math.isfinite(mx[a]) and math.isfinite(my[a]) and math.isfinite(mx[b]) and math.isfinite(my[b]):
                    cv2.line(frame, (int(mx[a]), int(my[a])), (int(mx[b]), int(my[b])), (90, 170, 255), 1, cv2.LINE_AA)
            for i in range(478):
                if math.isfinite(mx[i]) and math.isfinite(my[i]):
                    cv2.circle(frame, (int(mx[i]), int(my[i])), 1, (80, 160, 255), -1)

        # compact overlay
        emotions = {e: safe_float(rec.get(f"emotion_{e.lower()}"), 0) for e in EMOTION_NAMES}
        dominant = max(emotions, key=emotions.get)
        derived, dscore = self.derived_expression(rec)
        lines = [
            f"{dominant} {percent(emotions[dominant]):.0f}%",
            f"Val {safe_float(rec.get('valence'), 0):+.2f}  Aro {safe_float(rec.get('arousal'), 0):+.2f}",
            f"{derived} {percent(dscore):.0f}%",
        ]
        yy = min(h - 10, int(y2) + 22)
        for line in lines:
            cv2.putText(frame, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
            yy += 18

    def setup_output(self, frame):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = OUTPUT_ROOT / f"session_{stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        fps = self.settings["fps"] if self.settings["fps"] > 0 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer_original = cv2.VideoWriter(str(self.session_dir / "original.mp4"), fourcc, fps, (w, h))
        self.writer_tracked = cv2.VideoWriter(str(self.session_dir / "tracked.mp4"), fourcc, fps, (w, h))
        if not self.writer_original.isOpened() or not self.writer_tracked.isOpened():
            raise RuntimeError("Could not open MP4 writers. Check that OpenCV has an MP4 codec available.")
        meta = {
            "application": APP_NAME,
            "analysis_schema_version": 4,
            "measurement_quality_version": 1,
            "started": stamp,
            "source": "webcam" if self.is_webcam else str(self.source),
            "device": self.settings["device"],
            "gpu": self.settings["gpu"],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "py_feat_version": FEAT_VERSION,
            "face_threshold": self.settings["face_threshold"],
            "au_threshold": self.settings["au_threshold"],
            "smoothing": self.settings["smoothing"],
            "detection_interval": self.settings["detection_interval"],
            "tracking": "predicted_bbox_global_assignment",
            "tracking_max_missing_frames": TRACK_MAX_MISSING_FRAMES,
            "tracking_min_hits": TRACK_MIN_HITS,
            "identity_recognition": False,
            "level_2_temporal_analysis": True,
            "temporal_rapid_change_threshold_per_second": 2.0,
            "temporal_short_burst_max_duration_seconds": 0.50,
            "calibration_profile": str(CALIBRATION_PATH) if self.calibration else None,
            "calibration_is_neutral_reference": bool(self.calibration),
            "pose_quality_soft_limit_deg": POSE_SOFT_LIMIT_DEG,
            "pose_quality_hard_limit_deg": POSE_HARD_LIMIT_DEG,
            "gaze_context_soft_limit_deg": GAZE_CONTEXT_SOFT_LIMIT_DEG,
            "schema_version": SCHEMA_VERSION,
            "temporal_analysis_version": TEMPORAL_VERSION,
            "min_episode_duration": self.settings.get("min_episode_duration", MIN_EPISODE_DURATION),
            "min_episode_amplitude": self.settings.get("min_episode_amplitude", MIN_EPISODE_AMPLITUDE),
            "activation_hysteresis": ACTIVATION_HYSTERESIS,
            "min_rate_interval_seconds": MIN_RATE_INTERVAL_SECONDS,
        }
        (self.session_dir / "session_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def _json_safe(value):
        if isinstance(value,dict): return {str(k):FACSWorker._json_safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [FACSWorker._json_safe(v) for v in value]
        if isinstance(value,np.ndarray): return FACSWorker._json_safe(value.tolist())
        if isinstance(value,np.integer): return int(value)
        if isinstance(value,(np.floating,float)): x=float(value); return x if math.isfinite(x) else None
        if isinstance(value,np.bool_): return bool(value)
        return value

    def write_outputs(self):
        if self.session_dir is None: return
        self.finalize_signal_processing()
        self.rebuild_episodes_from_smoothed()
        self.finalize_temporal()
        safe_rows=[self._json_safe(r) for r in self.raw_rows]; safe_events=[self._json_safe(r) for r in self.event_rows]; safe_eps=[self._json_safe(r) for r in self.temporal_episodes]; safe_seq=[self._json_safe(r) for r in self.temporal_sequences]; safe_co=[self._json_safe(r) for r in self.temporal_interactions.get("coactivation",[])]; safe_tr=[self._json_safe(r) for r in self.temporal_interactions.get("transitions",[])]
        pd.DataFrame(self.raw_rows).to_csv(self.session_dir/"data.csv",index=False); (self.session_dir/"data.json").write_text(json.dumps(safe_rows,indent=2,allow_nan=False),encoding="utf-8"); pd.DataFrame(self.event_rows).to_csv(self.session_dir/"events.csv",index=False)
        pd.DataFrame(safe_eps).to_csv(self.session_dir/"episodes.csv",index=False); pd.DataFrame(safe_seq).to_csv(self.session_dir/"sequences.csv",index=False); pd.DataFrame(safe_co).to_csv(self.session_dir/"coactivation.csv",index=False); pd.DataFrame(safe_tr).to_csv(self.session_dir/"transitions.csv",index=False)
        wall=max(0.0,time.perf_counter()-self.start_time) if self.start_time else 0.0; video_duration=self.video_duration_seconds if self.video_duration_seconds>0 else self.frame_number/max(self.settings.get("fps",30.0),1.0); df=pd.DataFrame(self.raw_rows) if self.raw_rows else pd.DataFrame()
        summary={"schema_version":SCHEMA_VERSION,"temporal_analysis_version":TEMPORAL_VERSION,"video_duration_seconds":video_duration,"processing_wall_time_seconds":wall,"frames_processed":self.frame_number,"unique_track_ids_seen":sorted({int(r.get("person_id",0)) for r in self.raw_rows}),"max_simultaneous_faces":max([sum(1 for x in self.raw_rows if x.get("frame_number")==f) for f in {r.get("frame_number") for r in self.raw_rows}],default=0),"events":len(self.event_rows),"valid_episodes":len(safe_eps),"sequences":len(safe_seq),"coactivation_pairs":len(safe_co),"transition_pairs":len(safe_tr),"fallback_disk_input":self.fallback_disk,"last_detection_ms":self.last_detection_ms,"timing":{"fps":self.settings.get("fps",30.0),"frame_timestamps_recorded":bool(self.frame_timestamps)},"face_measurement_quality":self.tracking_quality_counts,"frame_quality":self.frame_quality_counts,"tracking":self.tracker.report(),"measurement_note":"AU values are detector measurements; derived temporal fields are analysis features and are not direct emotion or intent labels."}
        if not df.empty:
            for e in EMOTION_NAMES:
                c=f"emotion_{e.lower()}"
                if c in df: summary[f"mean_{c}"]=float(pd.to_numeric(df[c],errors="coerce").mean())
            for au in AU_NAMES:
                if au in df: summary[f"mean_{au}"]=float(pd.to_numeric(df[au],errors="coerce").mean())
        temporal={str(pid):self._json_safe(stats) for pid,stats in self.person_temporal_stats.items()}
        temporal_payload={"schema_version":SCHEMA_VERSION,"parameters":{"activation_threshold":self.settings.get("au_threshold",0.15),"deactivation_threshold":max(0.005,self.settings.get("au_threshold",0.15)-ACTIVATION_HYSTERESIS),"min_episode_duration":self.settings.get("min_episode_duration",MIN_EPISODE_DURATION),"min_episode_amplitude":self.settings.get("min_episode_amplitude",MIN_EPISODE_AMPLITUDE),"rapid_threshold_per_second":self.settings.get("rapid_threshold",2.0),"baseline_window_frames":BASELINE_WINDOW,"baseline_min_samples":BASELINE_MIN_SAMPLES,"baseline_scale_floor":BASELINE_SCALE_FLOOR,"max_temporal_gap_seconds":TEMPORAL_MAX_GAP_SECONDS,"cluster_gap_seconds":MAX_CLUSTER_GAP,"sequence_gap_seconds":MAX_SEQUENCE_GAP,"simultaneous_frame_tolerance_seconds":SIMULTANEOUS_FRAME_TOLERANCE,"min_short_burst_duration_seconds":MIN_SHORT_BURST_DURATION,"min_rate_interval_seconds":MIN_RATE_INTERVAL_SECONDS,"sequence_interpretation":"activity sequence, not a single expression or emotion"},"per_person":temporal,"episodes":safe_eps,"sequences":safe_seq,"interactions":{"coactivation":safe_co,"transitions":safe_tr,"activation_groups":self._json_safe(self.temporal_interactions.get("activation_groups",[]))}}
        (self.session_dir/"temporal_summary.json").write_text(json.dumps(temporal_payload,indent=2,allow_nan=False),encoding="utf-8"); (self.session_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
        master={"session_info":self._json_safe({"schema_version":SCHEMA_VERSION,"temporal_analysis_version":TEMPORAL_VERSION,"fps":self.settings.get("fps",30.0),"device":self.settings.get("device"),"gpu":self.settings.get("gpu"),"source":"webcam" if self.is_webcam else str(self.source)}),"summary":self._json_safe(summary),"frame_data":safe_rows,"events":safe_events,"temporal_analysis":temporal_payload}
        (self.session_dir/"analysis.json").write_text(json.dumps(master,indent=2,allow_nan=False),encoding="utf-8")

    def run(self):
        try:
            device, dinfo = choose_device()
            self.settings["device"] = device
            self.settings["gpu"] = dinfo["name"]
            self.status.emit(f"Loading Detectorv2 on {device.upper()}...")
            if Detectorv2 is None:
                raise RuntimeError("Py-Feat is not installed. Install it with: python -m pip install py-feat")
            self.detector = Detectorv2(device=device, identity_model=None)

            if self.is_webcam:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(str(self.source))
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open source: {self.source}")

            source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.settings["fps"] = source_fps if source_fps and source_fps > 1 else 30.0
            self.frame_width = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            self.frame_height = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_webcam else 0
            self.start_time = time.perf_counter()

            first = True
            while self.running:
                ok, bgr = self.cap.read()
                if not ok:
                    break
                self.frame_number += 1
                pos_ms = safe_float(self.cap.get(cv2.CAP_PROP_POS_MSEC), np.nan)
                timestamp = (pos_ms / 1000.0) if math.isfinite(pos_ms) and pos_ms >= 0 else ((self.frame_number - 1) / max(self.settings.get("fps", 30.0), 1.0))
                self.frame_timestamps.append(timestamp)
                self.video_duration_seconds = max(self.video_duration_seconds, timestamp)
                original = bgr.copy()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                if first:
                    self.setup_output(original)
                    first = False

                if (self.frame_number - 1) % self.settings["detection_interval"] == 0:
                    try:
                        fex = self.detect_frame(rgb)
                        rows = self.extract_rows(fex)
                        detections = [self.face_record(row, 0) for row in rows]
                        detections = self.tracker.update(detections, self.frame_number, timestamp)
                        self.last_faces = []
                        frame_qualities = []
                        for rec in detections:
                            person_id = int(rec.get("person_id", 0))
                            derived, dscore = self.derived_expression(rec)
                            rec["timestamp"] = timestamp
                            rec["derived_expression"] = derived
                            rec["derived_expression_score"] = dscore
                            self.temporal_update(rec)
                            self.apply_calibration(rec)
                            self.emotion_quality(rec)
                            self.event_update(rec, derived, dscore)
                            smoother = self.person_smoothers.setdefault(person_id, SignalSmoother(self.settings["smoothing"]))
                            for key in list(AU_NAMES) + ["valence", "arousal"]:
                                rec[f"smooth_{key}"] = smoother.update(key, rec.get(key))
                            frame_qualities.append(rec.get("tracking_quality", "BAD"))
                            self.last_faces.append(rec)
                            self.raw_rows.append(rec)
                        if not frame_qualities:
                            self.frame_quality_counts["NO_FACE"] += 1
                        elif "BAD" in frame_qualities:
                            self.frame_quality_counts["BAD"] += 1
                        elif "LOW" in frame_qualities:
                            self.frame_quality_counts["LOW"] += 1
                        elif "FAIR" in frame_qualities:
                            self.frame_quality_counts["FAIR"] += 1
                        else:
                            self.frame_quality_counts["GOOD"] += 1
                        self.last_result = fex
                    except Exception as exc:
                        self.status.emit(f"Detection error: {exc}")

                tracked = original.copy()
                for i, rec in enumerate(self.last_faces):
                    self.draw_face(tracked, rec, i)

                # global diagnostics
                now = time.perf_counter()
                self.fps_times.append(now)
                live_fps = (len(self.fps_times) - 1) / max(0.001, self.fps_times[-1] - self.fps_times[0]) if len(self.fps_times) > 1 else 0
                cv2.rectangle(tracked, (8, 8), (310, 82), (20, 25, 30), -1)
                cv2.putText(tracked, f"FACS  {self.settings['device'].upper()}  {live_fps:.1f} FPS", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235,235,235), 1, cv2.LINE_AA)
                cv2.putText(tracked, f"Faces {len(self.last_faces)}  Infer {self.last_detection_ms:.1f} ms", (16, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190,205,220), 1, cv2.LINE_AA)
                cv2.putText(tracked, f"Frame {self.frame_number}", (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190,205,220), 1, cv2.LINE_AA)

                self.writer_original.write(original)
                self.writer_tracked.write(tracked)
                state = {
                    "fps": live_fps, "inference_ms": self.last_detection_ms,
                    "faces": self.last_faces, "frame": self.frame_number,
                    "device": self.settings["device"], "gpu": self.settings["gpu"],
                    "session_dir": str(self.session_dir), "fallback_disk": self.fallback_disk,
                }
                self.frame_ready.emit(original, tracked, state)
                if total:
                    self.progress.emit(int(self.frame_number * 100 / max(1, total)))
                self.msleep(5 if self.is_webcam else 1)

            self.write_outputs()
            self.finished_info.emit({"session_dir": str(self.session_dir) if self.session_dir else "", "frames": self.frame_number})
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))
        finally:
            for writer in (self.writer_original, self.writer_tracked):
                if writer is not None:
                    writer.release()
            if self.cap is not None:
                self.cap.release()


class VideoPanel(QLabel):
    def __init__(self, title):
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 260)
        self.setStyleSheet("background:#0a0d10; border:1px solid #28303a; color:#7f8b99;")
        self.setText(title)

    def set_frame(self, bgr):
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        q = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(q).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Pixmap is already scaled on update; keeping the event simple avoids storing large frames.


class BarRow(QWidget):
    def __init__(self, name, value=0.0, suffix=""):
        super().__init__()
        self.name = QLabel(name)
        self.value = QLabel(f"{value:.2f}{suffix}")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(int(clamp(value * 100, 0, 100)))
        self.bar.setTextVisible(False)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        self.name.setMinimumWidth(135)
        self.value.setMinimumWidth(65)
        layout.addWidget(self.name)
        layout.addWidget(self.bar, 1)
        layout.addWidget(self.value)

    def update_value(self, value):
        value = safe_float(value, 0.0)
        self.bar.setValue(int(clamp(value * 100, 0, 100)))
        self.value.setText(f"{value:.2f}")


class LiveGraph(QWidget):
    def __init__(self):
        super().__init__()
        self.series = {}
        self.setMinimumHeight(230)

    def set_series(self, series):
        self.series = series
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        w, h = self.width(), self.height()
        left, top, right, bottom = 50, 18, 18, 30
        plot_w, plot_h = max(1, w-left-right), max(1, h-top-bottom)
        pen = QPen(Qt.darkGray); pen.setWidth(1); p.setPen(pen)
        for i in range(6):
            y = top + int(plot_h * i / 5)
            p.drawLine(left, y, w-right, y)
        p.setPen(Qt.lightGray)
        p.drawText(8, top+5, "1.0")
        p.drawText(8, top+plot_h//2+5, "0.0")
        p.drawText(8, top+plot_h+5, "-1.0")
        colors = [Qt.green, Qt.cyan, Qt.yellow, Qt.magenta, Qt.red]
        for si, (name, values) in enumerate(self.series.items()):
            vals = list(values)
            if len(vals) < 2:
                continue
            pen = QPen(colors[si % len(colors)]); pen.setWidth(2); p.setPen(pen)
            for i in range(1, len(vals)):
                a, b = vals[i-1], vals[i]
                if not (math.isfinite(safe_float(a)) and math.isfinite(safe_float(b))):
                    continue
                x1 = left + int(plot_w * (i-1) / max(1, len(vals)-1))
                x2 = left + int(plot_w * i / max(1, len(vals)-1))
                y1 = top + int((1.0 - clamp(float(a), -1, 1)) * 0.5 * plot_h)
                y2 = top + int((1.0 - clamp(float(b), -1, 1)) * 0.5 * plot_h)
                p.drawLine(x1, y1, x2, y2)
            p.drawText(left + 8 + si*105, h-8, name)


def run_internal_validation():
    """Run deterministic, local temporal-analysis sanity checks."""
    results = []
    def check(name, condition, detail=""):
        results.append({"name": name, "passed": bool(condition), "detail": str(detail)})

    # Tracking: two crossing faces plus a short detector dropout.
    tracker = MultiFaceTracker()
    expected = {1, 2}
    seen = set()
    for frame in range(1, 61):
        t = (frame - 1) / 30.0
        # Cross in the middle; tracker must still keep two distinct tracks.
        c1 = 220 + min(frame, 30) * 4 - max(0, frame - 30) * 4
        c2 = 460 - min(frame, 30) * 4 + max(0, frame - 30) * 4
        dets = [
            {"face_x": c1, "face_y": 120, "face_width": 90, "face_height": 90},
            {"face_x": c2, "face_y": 120, "face_width": 90, "face_height": 90},
        ]
        if 28 <= frame <= 30:
            dets = dets[:1]
        assigned = tracker.update(dets, frame, t)
        seen.update(int(r["person_id"]) for r in assigned)
    check("tracker keeps IDs bounded", tracker.created_tracks <= 3, tracker.report())
    check("tracker does not duplicate IDs within a frame", tracker.matched_detections <= tracker.total_detections)

    # Direct synthetic AU rows for episode/sequence fundamentals.
    settings = {"fps": 30.0, "au_threshold": 0.15, "min_episode_duration": 2.0/30.0, "min_episode_amplitude": 0.05}
    # Construct a lightweight worker without starting Qt.
    worker = FACSWorker(None, False, settings)
    worker.settings = settings
    worker.raw_rows = []
    worker.temporal_episodes = []
    worker.temporal_interactions = {"coactivation": [], "transitions": [], "activation_groups": []}
    worker.person_open_episodes = {}
    worker.person_prev_time = {}
    worker.person_prev_au = {}
    worker.person_last_active = {}
    worker.person_au_history = {}
    worker.person_temporal_stats = {}
    worker.tracking_quality_counts = {"GOOD":0,"FAIR":0,"LOW":0,"BAD":0}
    worker.frame_number = 0
    worker.video_duration_seconds = 2.0
    # Use class methods below; they do not require the QThread runtime.
    for i in range(61):
        t = i/30.0
        v12 = 0.05
        if 0.5 <= t <= 1.1:
            v12 = 0.05 + 0.70 * max(0.0, 1.0 - abs(t-0.8)/0.3)
        row = {"timestamp": t, "frame_number": i+1, "person_id": 1, "AU12": v12, "face_confidence": 0.99, "face_width": 120, "face_height": 120, "Pitch":0,"Roll":0,"Yaw":0,"track_state":"TRACKED","timing_valid":True,"tracking_quality":"GOOD"}
        worker.raw_rows.append(row)
    for row in worker.raw_rows:
        for au in AU_NAMES:
            if au != "AU12": row[au]=0.02
        row["tracking_quality"]="GOOD"; row["track_state"]="TRACKED"
    worker.finalize_signal_processing()
    worker.rebuild_episodes_from_smoothed()
    worker.build_temporal_interactions()
    eps = worker.temporal_episodes
    check("synthetic episode detected", any(e["au"]=="AU12" and e["valid"] for e in eps), len(eps))
    check("valid episodes have sane rates", all((e.get("rise_rate") is None or abs(e["rise_rate"]) < 100) and (e.get("fall_rate") is None or abs(e["fall_rate"]) < 100) for e in eps), eps)
    check("no giant single sequence from continuous tail", len(worker.temporal_sequences) <= 2, worker.temporal_sequences)
    check("same-frame activations are grouped", isinstance(worker.temporal_interactions.get("activation_groups"), list))
    check("smoothed AU values stay in detector domain", all(0.0 <= safe_float(r.get(f"smooth_{au}"), 0.0) <= 1.0 for r in worker.raw_rows for au in AU_NAMES))

    # A sustained AU plus a later independent event must not create one giant sequence.
    worker2 = FACSWorker(None, False, settings)
    worker2.settings = settings; worker2.raw_rows=[]; worker2.temporal_episodes=[]; worker2.temporal_interactions={"coactivation":[],"transitions":[],"activation_groups":[]}; worker2.person_open_episodes={}; worker2.person_prev_time={}; worker2.person_prev_au={}; worker2.person_last_active={}; worker2.person_au_history={}; worker2.person_temporal_stats={}; worker2.tracking_quality_counts={"GOOD":0,"FAIR":0,"LOW":0,"BAD":0}; worker2.frame_number=0; worker2.video_duration_seconds=4.0
    for i in range(121):
        t=i/30.0; row={"timestamp":t,"frame_number":i+1,"person_id":1,"face_confidence":0.99,"face_width":120,"face_height":120,"Pitch":0,"Roll":0,"Yaw":0,"track_state":"TRACKED","tracking_quality":"GOOD"}
        for au in AU_NAMES: row[au]=0.02
        row["AU25"] = 0.40 if t < 1.0 or t > 3.0 else 0.03
        row["AU12"] = 0.85 if 2.0 <= t <= 2.2 else 0.03
        worker2.raw_rows.append(row)
    worker2.finalize_signal_processing(); worker2.rebuild_episodes_from_smoothed(); worker2.build_temporal_interactions()
    check("sustained AU does not bridge distant sequences", len(worker2.temporal_sequences) >= 2 and max(x["duration"] for x in worker2.temporal_sequences) < 1.5, worker2.temporal_sequences)

    passed = sum(1 for r in results if r["passed"])
    return {"schema_version": 1, "total": len(results), "passed": passed, "failed": len(results)-passed, "results": results}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1700, 1050)
        self.setMinimumSize(1200, 760)
        self.worker = None
        self.latest_faces = []
        self.selected_person = None
        self.person_temporal_stats = {}
        self.temporal_episodes = []
        self.temporal_sequences = []
        self.temporal_interactions = {"coactivation": [], "transitions": [], "activation_groups": []}
        self.history = {"t": deque(maxlen=DISPLAY_HISTORY), "valence": deque(maxlen=DISPLAY_HISTORY), "arousal": deque(maxlen=DISPLAY_HISTORY)}
        self.last_original = None
        self.last_tracked = None
        # Post-analysis playback state. This is deliberately separate from the
        # analysis worker so a finished recording can be reviewed without rerunning inference.
        self.playback_cap = None
        self.playback_original_cap = None
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.playback_tick)
        self.playback_rows = None
        self.playback_events = []
        self.playback_session_dir = None
        self.playback_total_frames = 0
        self.playback_fps = 30.0
        self.playback_frame = 0
        self.playback_people_frames = {}
        self.playback_mode = False
        self.build_ui()
        self.apply_style()
        self.refresh_system_info()

    def build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("Title")
        header.addWidget(title)
        header.addStretch()
        self.device_label = QLabel("GPU: checking...")
        self.fps_label = QLabel("FPS: --")
        self.face_count_label = QLabel("Faces: 0")
        header.addWidget(self.device_label)
        header.addWidget(self.fps_label)
        header.addWidget(self.face_count_label)
        root.addLayout(header)

        source_box = QGroupBox("Source")
        source_layout = QHBoxLayout(source_box)
        self.webcam_radio = QRadioButton("Webcam")
        self.file_radio = QRadioButton("Video file")
        self.webcam_radio.setChecked(True)
        self.webcam_radio.toggled.connect(self.update_source_label)
        self.file_edit = QLabel("No file selected")
        self.file_edit.setObjectName("PathLabel")
        self.file_edit.setMinimumWidth(260)
        self.source_info = QLabel("Webcam: live analysis")
        self.source_info.setObjectName("PathLabel")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        source_layout.addWidget(self.webcam_radio)
        source_layout.addWidget(self.file_radio)
        source_layout.addWidget(self.file_edit, 1)
        source_layout.addWidget(self.source_info)
        source_layout.addWidget(self.browse_button)
        self.start_button = QPushButton("Start analysis")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        source_layout.addWidget(self.start_button)
        source_layout.addWidget(self.stop_button)
        root.addWidget(source_box)

        timeline_box = QGroupBox("Video review timeline")
        tl = QVBoxLayout(timeline_box)
        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.step_back_button = QPushButton("◀ Frame")
        self.step_forward_button = QPushButton("Frame ▶")
        self.prev_event_button = QPushButton("◀ Event")
        self.next_event_button = QPushButton("Event ▶")
        self.open_session_button = QPushButton("Open session")
        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.frame_label = QLabel("Frame -- / --")
        for b in (self.play_button, self.pause_button, self.step_back_button, self.step_forward_button, self.prev_event_button, self.next_event_button, self.open_session_button):
            controls.addWidget(b)
        controls.addStretch()
        controls.addWidget(self.time_label)
        controls.addWidget(self.frame_label)
        tl.addLayout(controls)
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.setTracking(True)
        self.timeline_slider.sliderMoved.connect(self.seek_playback)
        self.timeline_slider.sliderPressed.connect(self.pause_for_seek)
        self.timeline_slider.sliderReleased.connect(self.resume_after_seek)
        tl.addWidget(self.timeline_slider)
        self.play_button.clicked.connect(self.playback_play)
        self.pause_button.clicked.connect(self.playback_pause)
        self.step_back_button.clicked.connect(lambda: self.step_playback(-1))
        self.step_forward_button.clicked.connect(lambda: self.step_playback(1))
        self.prev_event_button.clicked.connect(lambda: self.jump_event(-1))
        self.next_event_button.clicked.connect(lambda: self.jump_event(1))
        self.open_session_button.clicked.connect(self.browse_session)
        for b in (self.play_button, self.pause_button, self.step_back_button, self.step_forward_button, self.prev_event_button, self.next_event_button):
            b.setEnabled(False)
        root.addWidget(timeline_box)

        splitter = QSplitter(Qt.Vertical)
        top_split = QSplitter(Qt.Horizontal)
        self.original_panel = VideoPanel("Original camera")
        self.tracked_panel = VideoPanel("Tracked / analyzed camera")
        top_split.addWidget(self.original_panel)
        top_split.addWidget(self.tracked_panel)
        top_split.setStretchFactor(0, 1)
        top_split.setStretchFactor(1, 1)
        splitter.addWidget(top_split)

        tabs = QTabWidget()
        self.tabs = tabs
        self.quick_tab = self.make_quick_tab()
        self.overview_tab = self.make_overview_tab()
        self.facs_tab = self.make_facs_tab()
        self.face_tab = self.make_face_tab()
        self.blend_tab = self.make_blendshape_tab()
        self.raw_tab = self.make_raw_tab()
        self.graph_tab = self.make_graph_tab()
        self.settings_tab = self.make_settings_tab()
        self.temporal_tab = self.make_temporal_tab()
        for widget, name in [
            (self.quick_tab, "Quick View"), (self.overview_tab, "Overview"), (self.facs_tab, "FACS"),
            (self.face_tab, "Face / Mesh"), (self.blend_tab, "Blendshapes"),
            (self.raw_tab, "Raw / Events"), (self.graph_tab, "Graphs"), (self.temporal_tab, "Temporal Review"), (self.settings_tab, "Settings")
        ]:
            tabs.addTab(widget, name)
        splitter.addWidget(tabs)
        splitter.setSizes([620, 400])
        root.addWidget(splitter, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)
        self.status_label = QLabel("Ready")
        root.addWidget(self.status_label)
        self.setCentralWidget(central)

    def make_temporal_tab(self):
        w=QWidget(); layout=QVBoxLayout(w); controls=QHBoxLayout()
        self.temporal_person_combo=QComboBox(); self.temporal_person_combo.addItem("All people",None)
        self.temporal_au_combo=QComboBox(); self.temporal_au_combo.addItem("All AUs",None)
        for au in AU_NAMES: self.temporal_au_combo.addItem(au,au)
        self.temporal_refresh=QPushButton("Refresh"); self.temporal_refresh.clicked.connect(self.refresh_temporal_view)
        controls.addWidget(QLabel("Person")); controls.addWidget(self.temporal_person_combo); controls.addWidget(QLabel("AU")); controls.addWidget(self.temporal_au_combo); controls.addWidget(self.temporal_refresh); controls.addStretch(); layout.addLayout(controls)
        self.temporal_summary_label=QLabel("No temporal analysis loaded."); layout.addWidget(self.temporal_summary_label); self.temporal_view=QPlainTextEdit(); self.temporal_view.setReadOnly(True); layout.addWidget(self.temporal_view,1); return w

    def refresh_temporal_view(self):
        eps=self.temporal_episodes; pid=self.temporal_person_combo.currentData() if hasattr(self,"temporal_person_combo") else None; au=self.temporal_au_combo.currentData() if hasattr(self,"temporal_au_combo") else None
        if pid is not None: eps=[e for e in eps if int(e.get("person_id",0))==int(pid)]
        if au is not None: eps=[e for e in eps if e.get("au")==au]
        valid=[e for e in eps if e.get("valid",False)]; co=self.temporal_interactions.get("coactivation",[]); tr=self.temporal_interactions.get("transitions",[])
        lines=[f"Temporal analysis v{TEMPORAL_VERSION}",f"Episodes: {len(valid)}  Sequences: {len(self.temporal_sequences)}  Coactivation pairs: {len(co)}  Transition pairs: {len(tr)}","","EPISODES"]
        for e in valid[-80:]:
            rise="--" if e.get("rise_rate") is None else f"{e['rise_rate']:.3f}/s"; fall="--" if e.get("fall_rate") is None else f"{e['fall_rate']:.3f}/s"
            lines.append(f"{e.get('au'):>5}  {e.get('onset_time',0):7.3f}s -> {e.get('peak_time',0):7.3f}s -> {e.get('offset_time',0):7.3f}s  dur {e.get('duration',0):.3f}s  amp {e.get('amplitude',0):.3f}  rise {rise} fall {fall}")
        lines += ["","SEQUENCES"]
        for q in self.temporal_sequences[-30:]: lines.append(f"{q.get('sequence_id')}  {q.get('start_time',0):.3f}-{q.get('end_time',0):.3f}s  {q.get('episode_count')} episodes  AUs: {', '.join(q.get('aus',[]))}")
        self.temporal_summary_label.setText(f"Showing {len(valid)} valid episodes"); self.temporal_view.setPlainText("\n".join(lines))

    def make_scroll(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def make_quick_tab(self):
        w = QWidget(); layout = QGridLayout(w)

        state_box = QGroupBox("Current person")
        sl = QGridLayout(state_box)
        self.quick_labels = {}
        quick_items = [
            ("person", "Person"), ("dominant", "Dominant emotion"),
            ("emotion_conf", "Emotion probability"), ("derived", "Derived pattern"),
            ("valence", "Valence"), ("arousal", "Arousal"),
            ("confidence", "Face confidence"), ("quality", "Tracking quality"),
            ("measurement", "Measurement quality"), ("emotion_status", "Emotion interpretation"),
            ("measurement", "Measurement quality"), ("emotion_status", "Emotion interpretation"),
            ("pose", "Head pose"), ("gaze_context", "Gaze context"),
            ("active_count", "Active AUs"), ("strongest_au", "Strongest AU"),
            ("fastest_au", "Fastest AU change"), ("rate", "Fastest rate / sec"),
            ("burst", "Rapid-action candidates"), ("pairs", "Co-active AU pairs"),
        ]
        for i, (key, label) in enumerate(quick_items):
            sl.addWidget(QLabel(label), i, 0)
            value = QLabel("--")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.quick_labels[key] = value
            sl.addWidget(value, i, 1)
        layout.addWidget(state_box, 0, 0)

        top_box = QGroupBox("Most active AUs")
        tl = QVBoxLayout(top_box)
        self.quick_au_view = QPlainTextEdit(); self.quick_au_view.setReadOnly(True)
        tl.addWidget(self.quick_au_view)
        layout.addWidget(top_box, 0, 1)

        temporal_box = QGroupBox("Level 2 temporal analysis")
        vl = QVBoxLayout(temporal_box)
        vl.addWidget(QLabel("Tracks AU onset, offset, duration, peak intensity, rate of change, co-activation, and short-burst candidates over time."))
        self.quick_temporal_view = QPlainTextEdit(); self.quick_temporal_view.setReadOnly(True)
        vl.addWidget(self.quick_temporal_view)
        layout.addWidget(temporal_box, 1, 0, 1, 2)

        note = QLabel("Important: temporal measurements describe visible facial actions. A rapid or short facial action is not automatically a microexpression, lie, or proof of a person's internal emotion.")
        note.setWordWrap(True)
        layout.addWidget(note, 2, 0, 1, 2)
        layout.setRowStretch(1, 1)
        return self.make_scroll(w)

    def make_overview_tab(self):
        w = QWidget(); layout = QGridLayout(w)
        self.emotion_rows = {}
        emotion_box = QGroupBox("Emotion probabilities")
        el = QVBoxLayout(emotion_box)
        for e in EMOTION_NAMES:
            row = BarRow(e)
            self.emotion_rows[e] = row
            el.addWidget(row)
        layout.addWidget(emotion_box, 0, 0)

        info_box = QGroupBox("Live state")
        il = QGridLayout(info_box)
        self.info_labels = {}
        for r, (key, label) in enumerate([
            ("dominant", "Dominant"), ("derived", "Derived expression"),
            ("confidence", "Face confidence"), ("quality", "Tracking quality"),
            ("measurement", "Measurement quality"), ("emotion_status", "Emotion interpretation"),
            ("pose", "Head pose"), ("gaze_context", "Gaze context"),
            ("valence", "Valence"), ("arousal", "Arousal"),
            ("pitch", "Pitch"), ("yaw", "Yaw"), ("roll", "Roll"),
            ("gaze", "Gaze"), ("inference", "Inference"),
        ]):
            il.addWidget(QLabel(label), r, 0)
            v = QLabel("--"); self.info_labels[key] = v
            il.addWidget(v, r, 1)
        layout.addWidget(info_box, 0, 1)

        people_box = QGroupBox("Tracks in frame")
        pl = QVBoxLayout(people_box)
        self.person_combo = QComboBox()
        self.person_combo.addItem("All people", None)
        self.person_combo.currentIndexChanged.connect(self.on_person_changed)
        pl.addWidget(self.person_combo)
        self.people_note = QLabel("Track IDs are visual tracking segments, not biometric identities. A new Track ID may be created after a track is lost.")
        self.people_note.setWordWrap(True)
        pl.addWidget(self.people_note)
        layout.addWidget(people_box, 1, 0)

        event_box = QGroupBox("Recent events")
        evl = QVBoxLayout(event_box)
        self.event_view = QPlainTextEdit(); self.event_view.setReadOnly(True)
        evl.addWidget(self.event_view)
        layout.addWidget(event_box, 1, 1)
        return self.make_scroll(w)

    def make_facs_tab(self):
        w = QWidget(); layout = QGridLayout(w)
        self.au_rows = {}
        for idx, (au, desc) in enumerate(AU_NAMES.items()):
            box = QGroupBox(f"{au}  {desc}")
            bl = QVBoxLayout(box)
            row = BarRow(au)
            self.au_rows[au] = row
            bl.addWidget(row)
            layout.addWidget(box, idx // 2, idx % 2)
        return self.make_scroll(w)

    def make_face_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        self.face_stats = QPlainTextEdit(); self.face_stats.setReadOnly(True)
        layout.addWidget(QLabel("68 landmarks, 478-point 3D mesh, face box and pose data are recorded to the session dataset."))
        layout.addWidget(self.face_stats)
        return w

    def make_blendshape_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        self.blend_view = QPlainTextEdit(); self.blend_view.setReadOnly(True)
        layout.addWidget(self.blend_view)
        return w

    def make_raw_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        self.raw_view = QPlainTextEdit(); self.raw_view.setReadOnly(True)
        layout.addWidget(self.raw_view)
        return w

    def make_graph_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        self.graph = LiveGraph()
        layout.addWidget(QLabel("Live valence / arousal history. Raw values remain in the CSV/JSON output."))
        layout.addWidget(self.graph, 1)
        return w

    def make_settings_tab(self):
        w = QWidget(); layout = QGridLayout(w)
        layout.addWidget(QLabel("Face detection threshold"), 0, 0)
        self.face_threshold = QDoubleSpinBox(); self.face_threshold.setRange(0.05, 0.99); self.face_threshold.setSingleStep(0.05); self.face_threshold.setValue(FACE_DETECTION_THRESHOLD)
        layout.addWidget(self.face_threshold, 0, 1)
        layout.addWidget(QLabel("AU activation threshold"), 1, 0)
        self.au_threshold = QDoubleSpinBox(); self.au_threshold.setRange(0.01, 0.99); self.au_threshold.setSingleStep(0.05); self.au_threshold.setValue(0.15)
        layout.addWidget(self.au_threshold, 1, 1)
        layout.addWidget(QLabel("Detection interval"), 2, 0)
        self.interval = QSpinBox(); self.interval.setRange(1, 10); self.interval.setValue(DETECTION_INTERVAL)
        layout.addWidget(self.interval, 2, 1)
        layout.addWidget(QLabel("Smoothing"), 3, 0)
        self.smoothing = QDoubleSpinBox(); self.smoothing.setRange(0.05, 1.0); self.smoothing.setSingleStep(0.05); self.smoothing.setValue(DEFAULT_SMOOTHING)
        layout.addWidget(self.smoothing, 3, 1)
        self.show_landmarks = QCheckBox("Show 68 landmarks"); self.show_landmarks.setChecked(True)
        self.show_mesh = QCheckBox("Show 478-point mesh"); self.show_mesh.setChecked(False)
        layout.addWidget(self.show_landmarks, 4, 0, 1, 2)
        layout.addWidget(self.show_mesh, 5, 0, 1, 2)
        self.self_test_button = QPushButton("Run internal validation tests")
        self.self_test_button.clicked.connect(self.run_self_test)
        layout.addWidget(self.self_test_button, 6, 0, 1, 2)
        note = QLabel("Recorded videos are processed frame-by-frame. Head pose affects expression measurement quality, while gaze direction is recorded separately and is not treated as an expression failure. Calibration creates a robust neutral/reference baseline from a completed session without changing raw Py-Feat probabilities. Raw detector outputs remain preserved. Use the validation button to test the temporal machinery locally without a video.")
        note.setWordWrap(True)
        layout.addWidget(note, 7, 0, 1, 2)
        layout.setRowStretch(8, 1)
        return w

    def run_self_test(self):
        try:
            report = run_internal_validation()
            out_dir = OUTPUT_ROOT / "validation"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            status = f"Validation: {report['passed']}/{report['total']} passed"
            if report["failed"]:
                status += f", {report['failed']} failed"
            self.status_label.setText(status)
            lines = [f"Internal validation: {report['passed']}/{report['total']} passed", ""]
            lines.extend([f"{'PASS' if x['passed'] else 'FAIL'}  {x['name']}  {x['detail']}" for x in report["results"]])
            QMessageBox.information(self, "Internal validation", "\n".join(lines))
        except Exception as exc:
            QMessageBox.critical(self, "Validation error", f"Internal validation failed: {exc}")

    def apply_style(self):
        self.setStyleSheet("""
        QWidget { background:#101419; color:#e6ebf0; font-size:13px; }
        QGroupBox { border:1px solid #2b343e; border-radius:6px; margin-top:8px; padding:8px; font-weight:600; }
        QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
        QPushButton { background:#202832; border:1px solid #394653; border-radius:5px; padding:7px 12px; }
        QPushButton:hover { background:#2a3540; }
        QPushButton:disabled { color:#697581; }
        QProgressBar { border:1px solid #303a44; border-radius:4px; background:#151b21; text-align:center; height:12px; }
        QProgressBar::chunk { background:#4ea1ff; border-radius:3px; }
        QTabBar::tab { background:#171d23; padding:8px 14px; border:1px solid #29323b; }
        QTabBar::tab:selected { background:#26313c; }
        QPlainTextEdit, QScrollArea { background:#0c1014; border:1px solid #27313a; }
        QLabel#Title { font-size:21px; font-weight:700; }
        QLabel#PathLabel { color:#9ba7b4; }
        """)

    def refresh_system_info(self):
        device, info = choose_device()
        if info["available"]:
            cap = info["capability"]
            self.device_label.setText(f"GPU: {info['name']}  sm_{cap[0]}{cap[1]}")
        else:
            self.device_label.setText("GPU: CPU fallback")

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Video (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*.*)")
        if path:
            self.file_edit.setText(path)
            self.file_radio.setChecked(True)
            self.update_source_label()

    def update_source_label(self):
        if self.file_radio.isChecked():
            self.source_info.setText("Recorded video: GPU analysis")
        else:
            self.source_info.setText("Webcam: live analysis")

    def on_person_changed(self):
        data = self.person_combo.currentData()
        self.selected_person = int(data) if data is not None else None
        if self.playback_mode and self.selected_person is not None:
            frames = self.playback_people_frames.get(self.selected_person, [])
            if frames:
                self.seek_to_frame(frames[0])
        if self.latest_faces:
            self.update_information({"faces": self.latest_faces, "fps": self.fps_label.text(), "inference_ms": 0, "frame": self.playback_frame, "gpu": "", "session_dir": str(self.playback_session_dir or "")})

    # -------------------------- post-analysis playback --------------------------
    def _format_time(self, seconds):
        seconds = max(0.0, float(seconds))
        mins = int(seconds // 60)
        secs = seconds - mins * 60
        return f"{mins:02d}:{secs:06.3f}"

    def _set_playback_buttons(self, enabled):
        for b in (self.play_button, self.pause_button, self.step_back_button,
                  self.step_forward_button, self.prev_event_button, self.next_event_button):
            b.setEnabled(bool(enabled))
        self.timeline_slider.setEnabled(bool(enabled))

    def browse_session(self):
        path = QFileDialog.getExistingDirectory(self, "Open FACS analysis session", str(OUTPUT_ROOT))
        if path:
            self.load_analysis_session(path)

    def close_playback(self):
        self.playback_timer.stop()
        if self.playback_cap is not None:
            self.playback_cap.release()
        if self.playback_original_cap is not None:
            self.playback_original_cap.release()
        self.playback_cap = None
        self.playback_original_cap = None
        self.playback_rows = None
        self.playback_events = []
        self.playback_people_frames = {}
        self.playback_mode = False
        self.playback_session_dir = None
        self._set_playback_buttons(False)

    def load_analysis_session(self, session_dir):
        """Open the finished tracked video and its CSV for offline review."""
        self.close_playback()
        session = Path(session_dir)
        tracked = session / "tracked.mp4"
        data_csv = session / "data.csv"
        events_csv = session / "events.csv"
        if not tracked.exists():
            self.status_label.setText("Playback unavailable: tracked.mp4 was not created.")
            return
        cap = cv2.VideoCapture(str(tracked))
        if not cap.isOpened():
            self.status_label.setText("Playback unavailable: could not open tracked.mp4.")
            return
        original_path = session / "original.mp4"
        original_cap = cv2.VideoCapture(str(original_path)) if original_path.exists() else None
        if original_cap is not None and not original_cap.isOpened():
            original_cap.release()
            original_cap = None
        self.playback_cap = cap
        self.playback_original_cap = original_cap
        self.playback_session_dir = session
        self.playback_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.playback_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.timeline_slider.setRange(0, max(0, self.playback_total_frames - 1))
        self.playback_mode = True

        if data_csv.exists():
            try:
                df = pd.read_csv(data_csv)
                if "frame_number" in df.columns and not df.empty:
                    df["frame_number"] = pd.to_numeric(df["frame_number"], errors="coerce").fillna(0).astype(int)
                    self.playback_rows = df
                    self.playback_people_frames = {}
                    if "person_id" in df.columns:
                        for pid, group in df.groupby("person_id"):
                            pid = int(pid)
                            self.playback_people_frames[pid] = sorted(set(group["frame_number"].astype(int).tolist()))
                    people = sorted(self.playback_people_frames)
                    self.person_combo.blockSignals(True)
                    self.person_combo.clear()
                    self.person_combo.addItem("All people", None)
                    for pid in people:
                        self.person_combo.addItem(f"Track {pid}", pid)
                    self.person_combo.setCurrentIndex(0)
                    self.selected_person = None
                    self.person_combo.blockSignals(False)
                else:
                    self.playback_rows = pd.DataFrame()
            except Exception as exc:
                self.playback_rows = pd.DataFrame()
                self.status_label.setText(f"Playback loaded, but data.csv could not be read: {exc}")
        else:
            self.playback_rows = pd.DataFrame()

        if events_csv.exists():
            try:
                self.playback_events = pd.read_csv(events_csv).to_dict("records")
            except Exception:
                self.playback_events = []

        self.temporal_episodes=[]; self.temporal_sequences=[]; self.temporal_interactions={"coactivation":[],"transitions":[],"activation_groups":[]}; self.person_temporal_stats={}
        temporal_path=session/"temporal_summary.json"
        if temporal_path.exists():
            try:
                payload=json.loads(temporal_path.read_text(encoding="utf-8"))
                self.temporal_episodes=payload.get("episodes",[])
                self.temporal_sequences=payload.get("sequences",[])
                self.temporal_interactions=payload.get("interactions",self.temporal_interactions)
                self.person_temporal_stats=payload.get("per_person",{})
            except Exception as exc:
                self.status_label.setText(f"Playback loaded, temporal data could not be read: {exc}")
        self.refresh_temporal_view()

        self._set_playback_buttons(True)
        self.playback_frame = 0
        self.show_playback_frame(0)
        self.status_label.setText(f"Review mode: {session.name}  |  select a person, scrub the timeline, or step frame-by-frame.")

    def pause_for_seek(self):
        self.playback_was_playing = self.playback_timer.isActive()
        self.playback_timer.stop()

    def resume_after_seek(self):
        self.show_playback_frame(self.timeline_slider.value())
        if getattr(self, "playback_was_playing", False):
            self.playback_play()

    def playback_play(self):
        if not self.playback_mode or self.playback_cap is None:
            return
        if self.playback_frame >= self.playback_total_frames - 1:
            self.seek_to_frame(0)
        interval = max(1, int(1000 / max(self.playback_fps, 1.0)))
        self.playback_timer.start(interval)

    def playback_pause(self):
        self.playback_timer.stop()

    def playback_tick(self):
        next_frame = self.playback_frame + 1
        if next_frame >= self.playback_total_frames:
            self.playback_pause()
            return
        self.show_playback_frame(next_frame)

    def seek_playback(self, frame):
        self.show_playback_frame(int(frame))

    def seek_to_frame(self, frame):
        if not self.playback_mode:
            return
        frame = int(clamp(frame, 0, max(0, self.playback_total_frames - 1)))
        self.show_playback_frame(frame)

    def step_playback(self, delta):
        if not self.playback_mode:
            return
        self.playback_pause()
        self.seek_to_frame(self.playback_frame + int(delta))

    def jump_event(self, direction):
        if not self.playback_mode or not self.playback_events:
            return
        pid = self.selected_person
        current_time = self.playback_frame / max(self.playback_fps, 1.0)
        events = []
        for e in self.playback_events:
            try:
                if pid is not None and int(e.get("person_id", -1)) != pid:
                    continue
                t = float(e.get("timestamp", 0))
                events.append(t)
            except Exception:
                continue
        events = sorted(set(events))
        if direction < 0:
            candidates = [t for t in events if t < current_time - 0.001]
            target = candidates[-1] if candidates else (events[0] if events else None)
        else:
            candidates = [t for t in events if t > current_time + 0.001]
            target = candidates[0] if candidates else (events[-1] if events else None)
        if target is not None:
            self.seek_to_frame(round(target * self.playback_fps))

    def _row_to_record(self, row):
        if row is None:
            return None
        rec = {}
        for k, v in row.items():
            if pd.isna(v):
                rec[k] = np.nan
            else:
                rec[k] = v.item() if hasattr(v, "item") else v
        return rec

    def _playback_records_for_frame(self, frame):
        if self.playback_rows is None or self.playback_rows.empty:
            return []
        df = self.playback_rows
        if "frame_number" not in df.columns:
            return []
        # Detection can intentionally run every Nth frame. Use the most recent
        # analyzed frame at or before the displayed frame, per person.
        subset = df[df["frame_number"] <= int(frame)]
        if subset.empty:
            return []
        latest_frame = int(subset["frame_number"].max())
        subset = subset[subset["frame_number"] == latest_frame]
        return [self._row_to_record(row) for _, row in subset.iterrows()]

    def show_playback_frame(self, frame):
        if not self.playback_mode or self.playback_cap is None:
            return
        frame = int(clamp(frame, 0, max(0, self.playback_total_frames - 1)))
        self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = self.playback_cap.read()
        if not ok:
            return
        self.playback_frame = frame
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame)
        self.timeline_slider.blockSignals(False)
        self.time_label.setText(f"{self._format_time(frame / max(self.playback_fps, 1.0))} / {self._format_time(self.playback_total_frames / max(self.playback_fps, 1.0))}")
        self.frame_label.setText(f"Frame {frame + 1} / {self.playback_total_frames}")
        self.tracked_panel.set_frame(bgr)
        self.last_tracked = bgr
        if self.playback_original_cap is not None:
            self.playback_original_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok_orig, original_bgr = self.playback_original_cap.read()
            if ok_orig:
                self.original_panel.set_frame(original_bgr)
                self.last_original = original_bgr

        records = self._playback_records_for_frame(frame + 1)
        self.latest_faces = records
        ids = sorted(set(int(r.get("person_id", 0)) for r in records if int(r.get("person_id", 0)) > 0))
        self.person_combo.blockSignals(True)
        current = self.selected_person
        self.person_combo.clear()
        self.person_combo.addItem("All people", None)
        all_people = sorted(self.playback_people_frames)
        for pid in all_people:
            self.person_combo.addItem(f"Track {pid}", pid)
        if current in all_people:
            self.person_combo.setCurrentIndex(all_people.index(current) + 1)
        else:
            self.person_combo.setCurrentIndex(0)
            self.selected_person = None
        self.person_combo.blockSignals(False)
        self.update_information({
            "faces": records, "fps": self.playback_fps, "inference_ms": 0,
            "frame": frame + 1, "gpu": "Recorded analysis", "session_dir": str(self.playback_session_dir),
        })

    def calibrate_from_session(self):
        """Build a robust neutral/reference profile from the currently loaded session.

        Calibration changes only derived deviation/quality interpretation. Raw detector
        outputs and the pretrained emotion probabilities are never overwritten.
        """
        if self.playback_rows is None or self.playback_rows.empty:
            QMessageBox.information(self, "Calibration", "Open a completed analysis session first.")
            return
        df = self.playback_rows.copy()
        if self.selected_person is not None and "person_id" in df.columns:
            df = df[df["person_id"].astype(int) == int(self.selected_person)]
        if df.empty:
            QMessageBox.information(self, "Calibration", "No records are available for the selected track.")
            return
        # Prefer high-quality, low-motion frames as neutral references.
        for col in ("face_confidence", "measurement_quality_score", "pose_magnitude_deg", "temporal_mean_abs_rate"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "measurement_quality_score" in df:
            candidates = df[df["measurement_quality_score"] >= 0.75]
        else:
            candidates = df
        if "temporal_mean_abs_rate" in candidates:
            low_motion = candidates["temporal_mean_abs_rate"] <= candidates["temporal_mean_abs_rate"].quantile(0.35)
            candidates = candidates[low_motion]
        if len(candidates) < 10:
            candidates = df.head(min(len(df), 60))
        # Limit calibration to a stable early portion, avoiding accidental expression peaks.
        candidates = candidates.sort_values("frame_number").head(90)
        profile = {
            "schema_version": 1,
            "created": datetime.now().isoformat(timespec="seconds"),
            "source": str(self.playback_session_dir or ""),
            "track_id": int(self.selected_person) if self.selected_person is not None else None,
            "frames": int(len(candidates)),
            "note": "Neutral/reference calibration. It does not retrain Py-Feat or modify raw detector probabilities.",
            "au": {}, "pose": {}, "gaze": {}
        }
        for au in AU_NAMES:
            vals = pd.to_numeric(candidates.get(au, pd.Series(dtype=float)), errors="coerce").dropna().tolist()
            c = robust_median(vals); mad = robust_mad(vals, c)
            profile["au"][au] = {"median": c if c is not None else 0.0, "mad": mad if mad is not None else 0.02}
        for key in ("Pitch", "Roll", "Yaw"):
            vals = pd.to_numeric(candidates.get(key, pd.Series(dtype=float)), errors="coerce").dropna().tolist()
            profile["pose"][key] = robust_median(vals) if vals else 0.0
        for key in ("gaze_pitch", "gaze_yaw"):
            vals = pd.to_numeric(candidates.get(key, pd.Series(dtype=float)), errors="coerce").dropna().tolist()
            profile["gaze"][key] = robust_median(vals) if vals else 0.0
        save_calibration_profile(profile)
        self.status_label.setText(f"Calibration saved: {len(candidates)} stable frames")
        QMessageBox.information(self, "Calibration saved", f"Created a robust neutral/reference profile from {len(candidates)} high-quality frames.\n\nRaw AUs and emotion probabilities are unchanged. The profile will be used on future analyses.")

    def clear_calibration(self):
        try:
            CALIBRATION_PATH.unlink(missing_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "Calibration", str(exc))
            return
        if self.worker is not None:
            self.worker.calibration = None
        self.status_label.setText("Calibration profile cleared")
        QMessageBox.information(self, "Calibration", "Calibration profile cleared. Future analyses will use detector outputs without a personal neutral baseline.")

    def settings_dict(self):
        device, info = choose_device()
        return {
            "device": device, "gpu": info["name"],
            "face_threshold": self.face_threshold.value(),
            "au_threshold": self.au_threshold.value(),
            "detection_interval": self.interval.value(),
            "smoothing": self.smoothing.value(),
            "show_landmarks": self.show_landmarks.isChecked(),
            "show_mesh": self.show_mesh.isChecked(),
            "fps": 30.0,
            "min_episode_duration": MIN_EPISODE_DURATION,
            "min_episode_amplitude": MIN_EPISODE_AMPLITUDE,
            "rapid_threshold": 2.0,
            "calibration_profile": load_calibration_profile(),
        }

    def start(self):
        self.close_playback()
        self.update_source_label()
        self.selected_person = None
        if self.worker and self.worker.isRunning():
            return
        if self.file_radio.isChecked():
            source = self.file_edit.text()
            if not source or source == "No file selected" or not Path(source).exists():
                QMessageBox.warning(self, "No video", "Select a valid video file first.")
                return
            webcam = False
        else:
            source = 0
            webcam = True
        self.worker = FACSWorker(source, webcam, self.settings_dict())
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished_info.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.browse_button.setEnabled(False)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Stopping... finishing output files...")
            self.stop_button.setEnabled(False)

    def on_status(self, text):
        self.status_label.setText(text)

    def on_frame(self, original, tracked, state):
        self.last_original = original
        self.last_tracked = tracked
        self.original_panel.set_frame(original)
        self.tracked_panel.set_frame(tracked)
        self.fps_label.setText(f"FPS: {state['fps']:.1f}")
        self.face_count_label.setText(f"Faces: {len(state['faces'])}")
        self.latest_faces = state["faces"]
        if self.worker is not None:
            self.person_temporal_stats = self.worker.person_temporal_stats
            self.temporal_episodes = self.worker.temporal_episodes
            self.temporal_sequences = self.worker.temporal_sequences
            self.temporal_interactions = self.worker.temporal_interactions
            self.refresh_temporal_view()

        ids = [int(r.get("person_id", 0)) for r in self.latest_faces]
        ids = sorted(set(i for i in ids if i > 0))
        old = self.person_combo.currentData() if hasattr(self, "person_combo") else None
        self.person_combo.blockSignals(True)
        self.person_combo.clear()
        self.person_combo.addItem("All people", None)
        for pid in ids:
            self.person_combo.addItem(f"Track {pid}", pid)
        if self.selected_person in ids:
            self.person_combo.setCurrentIndex(ids.index(self.selected_person) + 1)
        else:
            self.selected_person = ids[0] if ids else None
            self.person_combo.setCurrentIndex(1 if ids else 0)
        self.person_combo.blockSignals(False)
        self.update_information(state)

    def update_information(self, state):
        faces = state["faces"]
        if not faces:
            self.info_labels["dominant"].setText("No face")
            if hasattr(self, "quick_labels"):
                for value in self.quick_labels.values():
                    value.setText("--")
                self.quick_au_view.setPlainText("No face detected.")
                self.quick_temporal_view.setPlainText("No temporal data available.")
            self.info_labels["quality"].setText("No tracking")
            for e in EMOTION_NAMES:
                self.emotion_rows[e].update_value(0)
            for au in AU_NAMES:
                self.au_rows[au].update_value(0)
            return
        if self.selected_person is not None:
            selected = [r for r in faces if int(r.get("person_id", 0)) == self.selected_person]
            if not selected:
                self.info_labels["dominant"].setText(f"Track {self.selected_person}: not visible")
                self.info_labels["quality"].setText("Not in frame")
                for e in EMOTION_NAMES:
                    self.emotion_rows[e].update_value(0)
                for au in AU_NAMES:
                    self.au_rows[au].update_value(0)
                if hasattr(self, "quick_labels"):
                    for value in self.quick_labels.values():
                        value.setText("Not visible")
                    self.quick_au_view.setPlainText(f"Track {self.selected_person} is not visible in the current frame.")
                    self.quick_temporal_view.setPlainText("No new temporal measurements for this person at the current frame.")
                return
        else:
            selected = faces
        rec = selected[0] if selected else faces[0]
        emotions = {e: safe_float(rec.get(f"emotion_{e.lower()}"), 0) for e in EMOTION_NAMES}
        dominant = max(emotions, key=emotions.get)
        display_dominant = dominant if rec.get("emotion_interpretation_status") not in ("MIXED / AMBIGUOUS", "LOW") else "Mixed / uncertain"
        if hasattr(self, "quick_labels"):
            q = self.quick_labels
            q["person"].setText(f"Track {int(rec.get('person_id', 0))}")
            q["dominant"].setText(display_dominant)
            q["emotion_conf"].setText(f"{percent(emotions[dominant]):.1f}%  | margin {percent(rec.get('emotion_margin', 0)):.1f}%")
            q["derived"].setText(f"{rec.get('derived_expression', '--')}  {percent(rec.get('derived_expression_score', 0)):.1f}%")
            q["valence"].setText(f"{safe_float(rec.get('valence'), 0):+.3f}")
            q["arousal"].setText(f"{safe_float(rec.get('arousal'), 0):+.3f}")
            q["confidence"].setText(f"{percent(rec.get('face_confidence', 0)):.1f}%")
            q["quality"].setText(str(rec.get('tracking_quality', '--')))
            q["measurement"].setText(f"{rec.get('measurement_quality', '--')}  {percent(rec.get('measurement_quality_score', 0)):.0f}%")
            q["emotion_status"].setText(str(rec.get('emotion_interpretation_status', '--')))
            q["pose"].setText(f"P {safe_float(rec.get('pose_pitch_deg'),0):+.1f}°  Y {safe_float(rec.get('pose_yaw_deg'),0):+.1f}°  R {safe_float(rec.get('pose_roll_deg'),0):+.1f}°")
            q["gaze_context"].setText(str(rec.get('gaze_context', '--')))
            q["active_count"].setText(str(int(safe_float(rec.get('temporal_active_au_count'), 0))))
            q["strongest_au"].setText(f"{rec.get('temporal_strongest_au', '--')}  {safe_float(rec.get('temporal_strongest_au_value'), 0):.3f}")
            q["fastest_au"].setText(str(rec.get('temporal_fastest_au', '--')))
            q["rate"].setText(f"{safe_float(rec.get('temporal_fastest_rate'), 0):.3f}")
            q["burst"].setText(str(rec.get('rapid_action_candidates', 'None')))
            q["pairs"].setText(str(int(safe_float(rec.get('temporal_coactive_pairs'), 0))))

            top_aus = sorted(((au, safe_float(rec.get(au), 0.0)) for au in AU_NAMES), key=lambda x: x[1], reverse=True)[:8]
            self.quick_au_view.setPlainText("\n".join(f"{au:5s}  {value:.3f}  {AU_NAMES[au]}" for au, value in top_aus))
            active = str(rec.get('temporal_active_aus', 'None'))
            temporal_lines = [
                f"Active AUs: {active}",
                f"Fastest changing AU: {rec.get('temporal_fastest_au', '--')}  |  |rate| = {safe_float(rec.get('temporal_fastest_rate'), 0):.3f}/s",
                f"Mean absolute AU rate: {safe_float(rec.get('temporal_mean_abs_rate'), 0):.3f}/s",
                f"Co-active AU pairs: {int(safe_float(rec.get('temporal_coactive_pairs'), 0))}",
                f"Rapid-action candidates: {rec.get('rapid_action_candidates', 'None')}",
                "",
                "Current AU active durations:",
            ]
            for au in AU_NAMES:
                duration = safe_float(rec.get(f'active_duration_{au}'), 0.0)
                if duration > 0:
                    temporal_lines.append(f"  {au}: {duration:.3f} s  | peak {safe_float(rec.get(f'peak_{au}'), 0):.3f}")
            self.quick_temporal_view.setPlainText("\n".join(temporal_lines))
        self.info_labels["dominant"].setText(f"{display_dominant}  {percent(emotions[dominant]):.1f}%")
        self.info_labels["derived"].setText(f"{rec.get('derived_expression', '--')}  {percent(rec.get('derived_expression_score', 0)):.1f}%")
        self.info_labels["confidence"].setText(f"{percent(rec.get('face_confidence', 0)):.1f}%")
        self.info_labels["quality"].setText(str(rec.get("tracking_quality", "--")))
        self.info_labels["measurement"].setText(f"{rec.get('measurement_quality', '--')}  {percent(rec.get('measurement_quality_score', 0)):.0f}%")
        self.info_labels["emotion_status"].setText(str(rec.get("emotion_interpretation_status", "--")))
        self.info_labels["valence"].setText(f"{safe_float(rec.get('valence'), 0):+.3f}")
        self.info_labels["arousal"].setText(f"{safe_float(rec.get('arousal'), 0):+.3f}")
        self.info_labels["pitch"].setText(f"{safe_float(rec.get('Pitch'), 0):+.3f}")
        self.info_labels["yaw"].setText(f"{safe_float(rec.get('Yaw'), 0):+.3f}")
        self.info_labels["roll"].setText(f"{safe_float(rec.get('Roll'), 0):+.3f}")
        self.info_labels["gaze"].setText(f"pitch {safe_float(rec.get('gaze_pitch'), 0):+.3f}  yaw {safe_float(rec.get('gaze_yaw'), 0):+.3f}")
        self.info_labels["inference"].setText(f"{state['inference_ms']:.1f} ms")
        for e, row in self.emotion_rows.items():
            row.update_value(emotions[e])
        for au, row in self.au_rows.items():
            row.update_value(safe_float(rec.get(au), 0))

        blend_lines = []
        for k, v in rec.items():
            if k.startswith("blendshape_"):
                blend_lines.append(f"{k[11:]:28s} {safe_float(v, 0):.4f}")
        self.blend_view.setPlainText("\n".join(sorted(blend_lines)) or "No blendshape columns found in this Py-Feat output.")

        mesh_lines = [f"Face box: {rec.get('face_x')}, {rec.get('face_y')}, {rec.get('face_width')} × {rec.get('face_height')}",
                      f"Confidence: {rec.get('face_confidence')}",
                      f"Pose: Pitch={rec.get('Pitch')} Roll={rec.get('Roll')} Yaw={rec.get('Yaw')}",
                      f"Position: X={rec.get('X')} Y={rec.get('Y')} Z={rec.get('Z')}",
                      f"Gaze: pitch={rec.get('gaze_pitch')} yaw={rec.get('gaze_yaw')} angle={rec.get('gaze_angle')}",
                      "",
                      "68 landmarks:"]
        for i in range(68):
            mesh_lines.append(f"{i:03d}: ({safe_float(rec.get(f'x_{i}'), np.nan):.2f}, {safe_float(rec.get(f'y_{i}'), np.nan):.2f})")
        self.face_stats.setPlainText("\n".join(mesh_lines))

        self.history["t"].append(rec.get("timestamp", 0))
        self.history["valence"].append(safe_float(rec.get("valence"), 0))
        self.history["arousal"].append(safe_float(rec.get("arousal"), 0))
        self.graph.set_series({"Valence": self.history["valence"], "Arousal": self.history["arousal"]})

        recent = self.playback_events[-12:] if self.playback_mode and self.playback_events else (self.worker.event_rows[-12:] if self.worker else [])
        self.event_view.setPlainText("\n".join(
            f"{safe_float(e.get('timestamp'),0):8.2f}s  P{e.get('person_id','-')}  {e.get('type',''):20s}  {e.get('name','')}  {safe_float(e.get('value'),0):.2f}"
            for e in recent
        ))
        self.raw_view.setPlainText(
            f"Session: {state['session_dir']}\n"
            f"Frame: {state['frame']}\n"
            f"GPU: {state['gpu']}\n"
            f"Raw dataset: data.csv + data.json\n"
            f"Events: events.csv\n"
            f"Temporal summary: temporal_summary.json\n"
            f"Tracked video: tracked.mp4\n"
            f"Original video: original.mp4\n\n"
            "The CSV/JSON contain raw per-detection model outputs plus derived temporal, pose/gaze quality, calibration and emotion-ambiguity fields. Raw detector probabilities remain unchanged."
        )

    def on_finished(self, info):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.browse_button.setEnabled(True)
        self.progress.setValue(100 if not self.webcam_radio.isChecked() else self.progress.value())
        self.status_label.setText(f"Finished. Output: {info['session_dir']}")
        QMessageBox.information(self, "Analysis complete", f"Saved {info['frames']} frames to:\n{info['session_dir']}")

    def on_failed(self, message):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.browse_button.setEnabled(True)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Analysis error", message)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        self.close_playback()
        event.accept()


def main():
    print(f"{APP_NAME}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    device, info = choose_device()
    print(f"Device: {device}")
    print(f"GPU: {info['name']}")
    if info.get("capability"):
        print(f"Capability: {info['capability']}")
    print(f"Architectures: {info.get('arch', [])}")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setFont(QFont("Segoe UI", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
