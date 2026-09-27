"""Third-party dependency imports in a controlled order."""

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

# Py-Feat imports matplotlib/dateutil/six internally. Load that dependency chain
# before PyTorch and PySide6 so their import hooks do not conflict.
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
