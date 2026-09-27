"""Small reusable Qt widgets used by the workstation UI."""

import math

from .dependencies import cv2, np, Qt, QLabel, QWidget, QImage, QPixmap, QPainter, QPen, QProgressBar, QHBoxLayout
from .utils import safe_float, clamp

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
