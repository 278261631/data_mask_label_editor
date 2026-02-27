from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from mask_label_editor.config import AppConfig
from mask_label_editor.main_window import MainWindow


def run() -> None:
    cfg = AppConfig.load()
    app = QApplication(sys.argv)
    w = MainWindow(cfg)
    w.resize(1280, 800)
    w.show()
    sys.exit(app.exec())

