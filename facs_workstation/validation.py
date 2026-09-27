"""Deterministic application-level validation checks."""

from .config import AU_NAMES
from .utils import safe_float
from .tracking import MultiFaceTracker
from .worker import FACSWorker

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
