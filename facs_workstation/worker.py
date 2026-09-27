"""Analysis worker: detection, measurement, temporal analysis, export, and video processing."""

import json
import math
import os
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import median

from .dependencies import (
    cv2, np, pd, savgol_filter, find_peaks, torch, QThread, Signal, Detectorv2
)
from .config import *
from .utils import *
from .signal_processing import SignalSmoother
from .tracking import MultiFaceTracker

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
        # Initialize them here so validation and any early calls are safe.
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
