from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QSplitter,
    QToolBar,
)

from fits_3d_viewer.canvas import ImageCanvas
from fits_3d_viewer.config import AppConfig
from fits_3d_viewer.file_browser import FileBrowser, TileGroup
from fits_3d_viewer.fits_io import read_fits_image, to_uint8_view
from fits_3d_viewer.view3d import Dual3DView


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("FITS 3D Viewer")
        self._cfg = cfg

        self._ref_path: Path | None = None
        self._aligned_path: Path | None = None
        self._current_tile: TileGroup | None = None
        self._raw_ref: np.ndarray | None = None
        self._raw_aligned: np.ndarray | None = None

        self._canvas = ImageCanvas()
        self._canvas.set_mode("view3d")
        self._canvas.cursor_pixel.connect(self._on_cursor_pixel)
        self._canvas.view3d_click.connect(self._on_view3d_click)

        self._view3d = Dual3DView()
        self._view3d.patch_size_changed.connect(self._on_patch_size_changed)
        self._view3d.set_patch_size(self._cfg.patch_size)

        self._file_browser = FileBrowser()
        self._file_browser.tile_selected.connect(self._on_tile_selected)
        self._file_browser.setMinimumWidth(220)
        self._file_browser.setMaximumWidth(360)

        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._on_blink_tick)
        self._blinking = False

        self._build_ui()
        self._setup_shortcuts()

        if cfg.data_dir:
            self._file_browser.set_data_dir(cfg.data_dir)

    def _build_ui(self) -> None:
        tb = QToolBar("tools")
        tb.setMovable(False)
        self.addToolBar(tb)

        act_set_dir = QAction("📁 设置数据目录", self)
        act_set_dir.triggered.connect(self.set_data_dir_dialog)
        tb.addAction(act_set_dir)

        act_refresh = QAction("🔄 刷新", self)
        act_refresh.setShortcut(QKeySequence("F5"))
        act_refresh.triggered.connect(self._refresh_file_list)
        tb.addAction(act_refresh)

        self._image_name_label = QLabel("  显示: --")
        tb.addWidget(self._image_name_label)

        self._center_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._center_splitter.addWidget(self._canvas)
        self._center_splitter.addWidget(self._view3d)
        self._center_splitter.setStretchFactor(0, 3)
        self._center_splitter.setStretchFactor(1, 2)
        self._center_splitter.setSizes([900, 450])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._file_browser)
        splitter.addWidget(self._center_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 1200])
        self.setCentralWidget(splitter)

        self._status_coord = QLabel("x=-, y=-")
        self._status_value = QLabel("")
        self._status_file = QLabel("")
        self.statusBar().addWidget(self._status_coord, 0)
        self.statusBar().addWidget(self._status_value, 0)
        self.statusBar().addPermanentWidget(self._status_file, 1)

    def _setup_shortcuts(self) -> None:
        sc_prev = QShortcut(QKeySequence("PgUp"), self)
        sc_prev.activated.connect(self._file_browser.go_prev)
        sc_next = QShortcut(QKeySequence("PgDown"), self)
        sc_next.activated.connect(self._file_browser.go_next)

    def _fits_filter(self) -> str:
        return "FITS (*.fits *.fit *.fts);;All (*.*)"

    def set_data_dir_dialog(self) -> None:
        start = self._cfg.data_dir or str(Path.cwd())
        p = QFileDialog.getExistingDirectory(self, "选择数据目录", start)
        if not p:
            return
        self._cfg.data_dir = str(Path(p))
        self._cfg.save()
        self._file_browser.set_data_dir(self._cfg.data_dir)
        self.statusBar().showMessage(f"数据目录: {self._cfg.data_dir}")

    def _refresh_file_list(self) -> None:
        if self._cfg.data_dir:
            self._file_browser.set_data_dir(self._cfg.data_dir)
            self.statusBar().showMessage("已刷新文件列表")

    def _on_patch_size_changed(self, size: int) -> None:
        self._cfg.patch_size = int(size)
        self._cfg.save()

    def _load_reference(self, path: Path) -> None:
        img = read_fits_image(path)
        self._raw_ref = np.squeeze(img.data).astype(np.float64)
        gray8 = to_uint8_view(img.data)
        self._canvas.load_base_gray8(gray8, slot="a")
        self._canvas.fit_view()
        self._ref_path = path
        self._image_name_label.setText("  显示: reference")
        self._status_file.setText(f"📷 {path.name}")
        self._view3d.set_data(self._raw_ref, self._raw_aligned)

    def _load_aligned(self, path: Path) -> None:
        img = read_fits_image(path)
        self._raw_aligned = np.squeeze(img.data).astype(np.float64)
        gray8 = to_uint8_view(img.data)
        self._canvas.load_base_gray8(gray8, slot="b")
        self._aligned_path = path
        self._view3d.set_data(self._raw_ref, self._raw_aligned)

    def _on_tile_selected(self, tile: TileGroup) -> None:
        self._current_tile = tile
        self._canvas.clear_all()
        self._ref_path = None
        self._aligned_path = None
        self._raw_ref = None
        self._raw_aligned = None

        if tile.reference:
            self._load_reference(tile.reference)
        if tile.aligned:
            self._load_aligned(tile.aligned)

        self._image_name_label.setText("  显示: reference")
        self.setWindowTitle(f"FITS 3D Viewer - {tile.tile_id}")

    def _on_view3d_click(self, x: int, y: int) -> None:
        patch_size = self._view3d.get_patch_size()
        self._canvas.show_region_rect(x, y, patch_size)
        self._view3d.update_view(x, y)
        self.statusBar().showMessage(f"3D 查看: 中心({x}, {y})  {patch_size}×{patch_size} px")

    def _on_cursor_pixel(self, x: int, y: int, _code: int) -> None:
        self._status_coord.setText(f"x={x}, y={y}")
        val_str = ""
        if self._raw_ref is not None and 0 <= y < self._raw_ref.shape[0] and 0 <= x < self._raw_ref.shape[1]:
            val_str = f"ref={self._raw_ref[y, x]:.1f}"
        if self._raw_aligned is not None and 0 <= y < self._raw_aligned.shape[0] and 0 <= x < self._raw_aligned.shape[1]:
            if val_str:
                val_str += f"  ali={self._raw_aligned[y, x]:.1f}"
            else:
                val_str = f"ali={self._raw_aligned[y, x]:.1f}"
        self._status_value.setText(val_str)

    def _on_blink_tick(self) -> None:
        self._canvas.toggle_base_image()
        name = self._canvas.current_base_name()
        self._image_name_label.setText(f"  显示: {name}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._blink_timer.stop()
        event.accept()
