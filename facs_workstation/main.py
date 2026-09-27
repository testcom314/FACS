"""Application entry point."""

import sys

from .dependencies import QApplication, QFont, torch
from .config import APP_NAME
from .utils import choose_device
from .main_window import MainWindow


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
