"""Main Qt window and workstation presentation logic."""

import json
import math
import os
from collections import deque
from pathlib import Path
from datetime import datetime

from .dependencies import (
    cv2, np, pd, torch, Qt,
    QMainWindow, QWidget, QLabel, QPushButton, QRadioButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QTabWidget, QProgressBar,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QSplitter, QScrollArea,
    QMessageBox, QPlainTextEdit, QSlider, QFont, QTimer
)
from .config import *
from .utils import safe_float, choose_device, percent,clamp,emotion_color, row_to_dict, load_calibration_profile, save_calibration_profile
from .worker import FACSWorker
from .widgets import VideoPanel, BarRow, LiveGraph
from .validation import run_internal_validation

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
