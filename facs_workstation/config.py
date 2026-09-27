"""Application constants and analysis parameters."""

from pathlib import Path
import importlib.metadata

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

# MediaPipe face mesh edges. This is intentionally a moderate subset for speed.
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

