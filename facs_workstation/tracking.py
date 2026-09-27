"""Multi-face visual continuity tracking."""

import math

from .dependencies import np, linear_sum_assignment

from .utils import safe_float
from .config import (TRACK_MAX_MISSING_FRAMES, TRACK_DISTANCE_GATE, TRACK_MIN_IOU,
    TRACK_MAX_SIZE_CHANGE, TRACK_VELOCITY_ALPHA, TRACK_MIN_HITS)

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
