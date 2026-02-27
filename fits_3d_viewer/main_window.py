from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from fits_3d_viewer.background import (
    BG_METHOD_MESH,
    BG_METHOD_MORPH,
    BG_METHOD_ORIGINAL,
    BG_METHOD_PIPELINE,
    BG_METHOD_POLY2,
    BG_METHOD_RPCA,
    BG_METHOD_WAVELET,
    remove_background_with_params,
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
        self._active_bg_method = self._cfg.bg_method

        self._build_ui()
        self._setup_shortcuts()
        self._set_active_bg_method(self._cfg.bg_method, save=False)

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

        self._method_panel = self._build_method_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._file_browser)
        splitter.addWidget(self._center_splitter)
        splitter.addWidget(self._method_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([250, 1000, 340])
        self.setCentralWidget(splitter)

        self._status_coord = QLabel("x=-, y=-")
        self._status_value = QLabel("")
        self._status_file = QLabel("")
        self.statusBar().addWidget(self._status_coord, 0)
        self.statusBar().addWidget(self._status_value, 0)
        self.statusBar().addPermanentWidget(self._status_file, 1)

    def _build_method_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(420)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self._active_method_label = QLabel("当前方法: -")
        self._active_method_label.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        lay.addWidget(self._active_method_label)

        # Original
        g0 = QGroupBox("原图")
        g0l = QHBoxLayout(g0)
        b0 = QPushButton("应用")
        b0.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_ORIGINAL))
        g0l.addWidget(b0)
        lay.addWidget(g0)

        # Mesh
        g1 = QGroupBox("1) 分块背景建模")
        g1l = QHBoxLayout(g1)
        self._mesh_box = QSpinBox()
        self._mesh_box.setRange(16, 512)
        self._mesh_box.setValue(self._cfg.mesh_box)
        self._mesh_clip = QDoubleSpinBox()
        self._mesh_clip.setRange(1.5, 8.0)
        self._mesh_clip.setSingleStep(0.1)
        self._mesh_clip.setValue(self._cfg.mesh_clip_sigma)
        b1 = QPushButton("应用")
        b1.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_MESH))
        g1l.addWidget(QLabel("box")); g1l.addWidget(self._mesh_box)
        g1l.addWidget(QLabel("sigma")); g1l.addWidget(self._mesh_clip)
        g1l.addWidget(b1)
        lay.addWidget(g1)

        # Morphology
        g2 = QGroupBox("2) 形态学背景")
        g2l = QHBoxLayout(g2)
        self._morph_radius = QSpinBox()
        self._morph_radius.setRange(2, 256)
        self._morph_radius.setValue(self._cfg.morph_radius)
        b2 = QPushButton("应用")
        b2.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_MORPH))
        g2l.addWidget(QLabel("radius")); g2l.addWidget(self._morph_radius); g2l.addWidget(b2)
        lay.addWidget(g2)

        # Poly
        g3 = QGroupBox("3) 多项式曲面拟合")
        g3l = QHBoxLayout(g3)
        self._poly_sample = QSpinBox()
        self._poly_sample.setRange(5000, 500000)
        self._poly_sample.setSingleStep(5000)
        self._poly_sample.setValue(self._cfg.poly_sample_size)
        self._poly_clip = QDoubleSpinBox()
        self._poly_clip.setRange(1.5, 8.0)
        self._poly_clip.setSingleStep(0.1)
        self._poly_clip.setValue(self._cfg.poly_clip_sigma)
        b3 = QPushButton("应用")
        b3.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_POLY2))
        g3l.addWidget(QLabel("sample")); g3l.addWidget(self._poly_sample)
        g3l.addWidget(QLabel("sigma")); g3l.addWidget(self._poly_clip)
        g3l.addWidget(b3)
        lay.addWidget(g3)

        # Wavelet
        g4 = QGroupBox("4) 小波多尺度")
        g4l = QHBoxLayout(g4)
        self._wav_sigma = QDoubleSpinBox()
        self._wav_sigma.setRange(2.0, 64.0)
        self._wav_sigma.setSingleStep(0.5)
        self._wav_sigma.setValue(self._cfg.wavelet_base_sigma)
        self._wav_levels = QSpinBox()
        self._wav_levels.setRange(2, 8)
        self._wav_levels.setValue(self._cfg.wavelet_levels)
        b4 = QPushButton("应用")
        b4.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_WAVELET))
        g4l.addWidget(QLabel("sigma")); g4l.addWidget(self._wav_sigma)
        g4l.addWidget(QLabel("levels")); g4l.addWidget(self._wav_levels)
        g4l.addWidget(b4)
        lay.addWidget(g4)

        # RPCA
        g5 = QGroupBox("5) 低秩+稀疏")
        g5l = QHBoxLayout(g5)
        self._rpca_rank = QSpinBox()
        self._rpca_rank.setRange(1, 10)
        self._rpca_rank.setValue(self._cfg.rpca_rank_keep)
        b5 = QPushButton("应用")
        b5.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_RPCA))
        g5l.addWidget(QLabel("rank")); g5l.addWidget(self._rpca_rank); g5l.addWidget(b5)
        lay.addWidget(g5)

        # Pipeline
        g6 = QGroupBox("6) 推荐流程")
        g6l = QVBoxLayout(g6)
        row1 = QHBoxLayout()
        self._pipe_box = QSpinBox(); self._pipe_box.setRange(16, 512); self._pipe_box.setValue(self._cfg.pipeline_box)
        self._pipe_clip = QDoubleSpinBox(); self._pipe_clip.setRange(1.5, 8.0); self._pipe_clip.setSingleStep(0.1); self._pipe_clip.setValue(self._cfg.pipeline_clip_sigma)
        row1.addWidget(QLabel("box")); row1.addWidget(self._pipe_box); row1.addWidget(QLabel("sigma")); row1.addWidget(self._pipe_clip)
        row2 = QHBoxLayout()
        self._pipe_denoise = QDoubleSpinBox(); self._pipe_denoise.setRange(0.5, 20.0); self._pipe_denoise.setSingleStep(0.1); self._pipe_denoise.setValue(self._cfg.pipeline_denoise_sigma)
        self._pipe_mix = QDoubleSpinBox(); self._pipe_mix.setRange(0.0, 1.0); self._pipe_mix.setSingleStep(0.05); self._pipe_mix.setValue(self._cfg.pipeline_mix_alpha)
        self._pipe_gain = QDoubleSpinBox(); self._pipe_gain.setRange(0.5, 40.0); self._pipe_gain.setSingleStep(0.5); self._pipe_gain.setValue(self._cfg.pipeline_asinh_gain)
        row2.addWidget(QLabel("denoise")); row2.addWidget(self._pipe_denoise)
        row2.addWidget(QLabel("mix")); row2.addWidget(self._pipe_mix)
        row2.addWidget(QLabel("gain")); row2.addWidget(self._pipe_gain)
        b6 = QPushButton("应用推荐流程")
        b6.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_PIPELINE))
        g6l.addLayout(row1); g6l.addLayout(row2); g6l.addWidget(b6)
        lay.addWidget(g6)

        lay.addStretch(1)
        return panel

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

    def _save_bg_params_to_cfg(self) -> None:
        self._cfg.mesh_box = int(self._mesh_box.value())
        self._cfg.mesh_clip_sigma = float(self._mesh_clip.value())
        self._cfg.morph_radius = int(self._morph_radius.value())
        self._cfg.poly_sample_size = int(self._poly_sample.value())
        self._cfg.poly_clip_sigma = float(self._poly_clip.value())
        self._cfg.wavelet_base_sigma = float(self._wav_sigma.value())
        self._cfg.wavelet_levels = int(self._wav_levels.value())
        self._cfg.rpca_rank_keep = int(self._rpca_rank.value())
        self._cfg.pipeline_box = int(self._pipe_box.value())
        self._cfg.pipeline_clip_sigma = float(self._pipe_clip.value())
        self._cfg.pipeline_denoise_sigma = float(self._pipe_denoise.value())
        self._cfg.pipeline_mix_alpha = float(self._pipe_mix.value())
        self._cfg.pipeline_asinh_gain = float(self._pipe_gain.value())
        self._cfg.bg_method = self._active_bg_method
        self._cfg.save()

    def _set_active_bg_method(self, method: str, save: bool = True) -> None:
        self._active_bg_method = method
        self._active_method_label.setText(f"当前方法: {method}")
        if save:
            self._save_bg_params_to_cfg()
        self._recompute_background_view()

    def _params_for_method(self, method: str) -> dict[str, float | int]:
        if method == BG_METHOD_MESH:
            return {"box": int(self._mesh_box.value()), "clip_sigma": float(self._mesh_clip.value())}
        if method == BG_METHOD_MORPH:
            return {"radius": int(self._morph_radius.value())}
        if method == BG_METHOD_POLY2:
            return {
                "sample_size": int(self._poly_sample.value()),
                "clip_sigma": float(self._poly_clip.value()),
            }
        if method == BG_METHOD_WAVELET:
            return {
                "base_sigma": float(self._wav_sigma.value()),
                "levels": int(self._wav_levels.value()),
            }
        if method == BG_METHOD_RPCA:
            return {"rank_keep": int(self._rpca_rank.value())}
        if method == BG_METHOD_PIPELINE:
            return {
                "box": int(self._pipe_box.value()),
                "clip_sigma": float(self._pipe_clip.value()),
                "denoise_sigma": float(self._pipe_denoise.value()),
                "mix_alpha": float(self._pipe_mix.value()),
                "asinh_gain": float(self._pipe_gain.value()),
            }
        return {}

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
        method = self._active_bg_method
        effective_method = BG_METHOD_ORIGINAL if self._compare_original else method
        params = self._params_for_method(effective_method)

        self._disp_ref = None
        self._disp_aligned = None

        if self._raw_ref is not None:
            self._disp_ref = remove_background_with_params(self._raw_ref, method=effective_method, params=params)
            self._canvas.load_base_gray8(to_uint8_view(self._disp_ref), slot="a")
        if self._raw_aligned is not None:
            self._disp_aligned = remove_background_with_params(
                self._raw_aligned, method=effective_method, params=params
            )
            self._canvas.load_base_gray8(to_uint8_view(self._disp_aligned), slot="b")

        self._view3d.set_data(self._disp_ref, self._disp_aligned)
        method_name = method
        if self._compare_original:
            self.statusBar().showMessage(f"对比原图中 (C 关闭)  | 当前方法: {method_name}")
        else:
            self.statusBar().showMessage(f"去背景方法: {method_name}")

    def _on_blink_tick(self) -> None:
        self._canvas.toggle_base_image()
        name = self._canvas.current_base_name()
        self._image_name_label.setText(f"  显示: {name}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._cfg.window_width = int(self.width())
        self._cfg.window_height = int(self.height())
        self._cfg.save()
        self._blink_timer.stop()
        event.accept()
