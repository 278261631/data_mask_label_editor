from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QSpinBox,
    QSplitter,
    QToolBar,
)

from fits_3d_viewer.background import (
    BG_METHOD_MESH,
    BG_METHOD_MORPH,
    BG_METHOD_ORIGINAL,
    BG_METHOD_PIPELINE,
    BG_METHOD_POLY2,
    BG_METHOD_RPCA,
    BG_METHOD_WAVELET,
    remove_background,
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
        self._disp_ref: np.ndarray | None = None
        self._disp_aligned: np.ndarray | None = None

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
        self._compare_original = False

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

        tb.addSeparator()
        tb.addWidget(QLabel("去背景:"))
        self._bg_method_combo = QComboBox()
        self._bg_method_combo.addItem("原图", BG_METHOD_ORIGINAL)
        self._bg_method_combo.addItem("分块背景建模", BG_METHOD_MESH)
        self._bg_method_combo.addItem("形态学背景 (Rolling/Top-hat)", BG_METHOD_MORPH)
        self._bg_method_combo.addItem("多项式曲面拟合", BG_METHOD_POLY2)
        self._bg_method_combo.addItem("小波多尺度分离", BG_METHOD_WAVELET)
        self._bg_method_combo.addItem("低秩+稀疏 (RPCA)", BG_METHOD_RPCA)
        self._bg_method_combo.addItem("推荐流程(稳健组合)", BG_METHOD_PIPELINE)
        self._bg_method_combo.currentIndexChanged.connect(self._on_bg_method_changed)
        tb.addWidget(self._bg_method_combo)

        tb.addWidget(QLabel("尺度:"))
        self._bg_scale_spin = QSpinBox()
        self._bg_scale_spin.setRange(8, 256)
        self._bg_scale_spin.setValue(self._cfg.bg_scale)
        self._bg_scale_spin.valueChanged.connect(self._on_bg_scale_changed)
        tb.addWidget(self._bg_scale_spin)

        saved_idx = self._bg_method_combo.findData(self._cfg.bg_method)
        if saved_idx < 0:
            saved_idx = 0
        self._bg_method_combo.setCurrentIndex(saved_idx)

        self._act_compare_original = QAction("对比原图 (C)", self)
        self._act_compare_original.setCheckable(True)
        self._act_compare_original.setShortcut(QKeySequence("C"))
        self._act_compare_original.toggled.connect(self._on_compare_original_toggled)
        tb.addAction(self._act_compare_original)

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

    def _on_bg_method_changed(self, _index: int) -> None:
        self._cfg.bg_method = str(self._bg_method_combo.currentData())
        self._cfg.save()
        self._recompute_background_view()

    def _on_bg_scale_changed(self, value: int) -> None:
        self._cfg.bg_scale = int(value)
        self._cfg.save()
        self._recompute_background_view()

    def _on_compare_original_toggled(self, on: bool) -> None:
        self._compare_original = bool(on)
        self._recompute_background_view()

    def _load_reference(self, path: Path) -> None:
        img = read_fits_image(path)
        self._raw_ref = np.squeeze(img.data).astype(np.float64)
        self._ref_path = path
        self._status_file.setText(f"📷 {path.name}")
        self._recompute_background_view()
        self._canvas.fit_view()

    def _load_aligned(self, path: Path) -> None:
        img = read_fits_image(path)
        self._raw_aligned = np.squeeze(img.data).astype(np.float64)
        self._aligned_path = path
        self._recompute_background_view()

    def _on_tile_selected(self, tile: TileGroup) -> None:
        self._current_tile = tile
        self._canvas.clear_all()
        self._ref_path = None
        self._aligned_path = None
        self._raw_ref = None
        self._raw_aligned = None
        self._disp_ref = None
        self._disp_aligned = None

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
        if self._disp_ref is not None and 0 <= y < self._disp_ref.shape[0] and 0 <= x < self._disp_ref.shape[1]:
            val_str = f"ref={self._disp_ref[y, x]:.1f}"
        if self._disp_aligned is not None and 0 <= y < self._disp_aligned.shape[0] and 0 <= x < self._disp_aligned.shape[1]:
            if val_str:
                val_str += f"  ali={self._disp_aligned[y, x]:.1f}"
            else:
                val_str = f"ali={self._disp_aligned[y, x]:.1f}"
        self._status_value.setText(val_str)

    def _recompute_background_view(self) -> None:
        method = str(self._bg_method_combo.currentData())
        effective_method = BG_METHOD_ORIGINAL if self._compare_original else method
        scale = int(self._bg_scale_spin.value())

        self._disp_ref = None
        self._disp_aligned = None

        if self._raw_ref is not None:
            self._disp_ref = remove_background(self._raw_ref, method=effective_method, scale=scale)
            self._canvas.load_base_gray8(to_uint8_view(self._disp_ref), slot="a")
        if self._raw_aligned is not None:
            self._disp_aligned = remove_background(self._raw_aligned, method=effective_method, scale=scale)
            self._canvas.load_base_gray8(to_uint8_view(self._disp_aligned), slot="b")

        self._view3d.set_data(self._disp_ref, self._disp_aligned)
        method_name = self._bg_method_combo.currentText()
        if self._compare_original:
            self.statusBar().showMessage(f"对比原图中 (C 关闭)  | 当前方法: {method_name}  尺度: {scale}")
        else:
            self.statusBar().showMessage(f"去背景方法: {method_name}  尺度: {scale}")

    def _on_blink_tick(self) -> None:
        self._canvas.toggle_base_image()
        name = self._canvas.current_base_name()
        self._image_name_label.setText(f"  显示: {name}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._blink_timer.stop()
        event.accept()
