from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from fits_3d_viewer.config import AppConfig
from fits_3d_viewer.main_window import MainWindow


def run() -> None:
    cfg = AppConfig.load()
    app = QApplication(sys.argv)
    w = MainWindow(cfg)
    w.resize(1680, 900)
    w.show()
    sys.exit(app.exec())

