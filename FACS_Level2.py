"""Compatibility entry point for the refactored FACS Analysis Workstation."""

from facs_workstation.config import *
from facs_workstation.utils import *
from facs_workstation.signal_processing import SignalSmoother
from facs_workstation.tracking import MultiFaceTracker
from facs_workstation.worker import FACSWorker
from facs_workstation.widgets import VideoPanel, BarRow, LiveGraph
from facs_workstation.validation import run_internal_validation
from facs_workstation.main_window import MainWindow
from facs_workstation.main import main

if __name__ == "__main__":
    main()
