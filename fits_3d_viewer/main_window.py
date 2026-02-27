from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from fits_3d_viewer.canvas import MaskCanvas
from fits_3d_viewer.codebook import load_codebook
from fits_3d_viewer.config import AppConfig
from fits_3d_viewer.file_browser import FileBrowser, TileGroup
from fits_3d_viewer.fits_io import (
    read_fits_image,
    read_mask_image,
    to_uint8_view,
    write_mask_image,
)
from fits_3d_viewer.labels import Label, default_labels, ensure_unique_codes
from fits_3d_viewer.view3d import Dual3DView


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("FITS Mask Label Editor")
        self._cfg = cfg

        self._ref_path: Path | None = None
        self._aligned_path: Path | None = None
        self._mask_path: Path | None = None
        self._codebook_path: Path | None = None
        self._current_tile: TileGroup | None = None
        self._source_mask: np.ndarray | None = None
        self._suspend_dirty_tracking = False

        self._mask_header = None

        # 原始 16-bit 全图数据（用于 3D 视图）
        self._raw_ref: np.ndarray | None = None
        self._raw_aligned: np.ndarray | None = None

        self._labels: list[Label] = default_labels(alpha=255)
        self._cluster_rng = np.random.default_rng(20260227)
        self._last_cluster_code: int | None = None
        self._cluster_base_mask: np.ndarray | None = None

        # ---------- 当前模式 ----------
        self._mode = "view3d"  # "view3d" | "edit_mask"

        # ---------- Canvas ----------
        self._canvas = MaskCanvas()
        self._canvas.set_mode("view3d")
        self._canvas.mask_changed.connect(self._on_mask_changed)
        self._canvas.cursor_pixel.connect(self._on_cursor_pixel)
        self._canvas.color_picked.connect(self._on_color_picked)
        self._canvas.view3d_click.connect(self._on_view3d_click)

        self._dirty = False
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._on_blink_tick)
        self._blinking = False

        # ---------- 3D View ----------
        self._view3d = Dual3DView()

        # ---------- File browser ----------
        self._file_browser = FileBrowser()
        self._file_browser.tile_selected.connect(self._on_tile_selected)
        self._file_browser.setMinimumWidth(220)
        self._file_browser.setMaximumWidth(360)

        # ---------- Label list ----------
        self._label_list = QListWidget()
        self._label_list.currentRowChanged.connect(self._on_label_changed)

        # ---------- Brush slider ----------
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(1, 100)
        self._brush_slider.setValue(8)
        self._brush_slider.valueChanged.connect(self._on_brush_changed)
        self._brush_label = QLabel("笔刷: 8")

        # ---------- Alpha slider ----------
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.setValue(50)
        self._alpha_slider.valueChanged.connect(self._on_alpha_changed)
        self._alpha_label = QLabel("透明度: 50%")

        # ---------- Cluster params ----------
        self._cluster_k_spin = QSpinBox()
        self._cluster_k_spin.setRange(2, 16)
        self._cluster_k_spin.setValue(5)
        self._cluster_sample_spin = QSpinBox()
        self._cluster_sample_spin.setRange(1000, 2000000)
        self._cluster_sample_spin.setSingleStep(1000)
        self._cluster_sample_spin.setValue(20000)
        self._cluster_iter_spin = QSpinBox()
        self._cluster_iter_spin.setRange(5, 200)
        self._cluster_iter_spin.setValue(20)
        self._cluster_seed_min_spin = QSpinBox()
        self._cluster_seed_min_spin.setRange(5, 50000)
        self._cluster_seed_min_spin.setValue(20)
        self._cluster_radius_spin = QSpinBox()
        self._cluster_radius_spin.setRange(0, 7)
        self._cluster_radius_spin.setValue(1)
        self._cluster_precision_spin = QDoubleSpinBox()
        self._cluster_precision_spin.setRange(0.0, 1.0)
        self._cluster_precision_spin.setDecimals(3)
        self._cluster_precision_spin.setSingleStep(0.01)
        self._cluster_precision_spin.setValue(0.12)
        self._cluster_recall_spin = QDoubleSpinBox()
        self._cluster_recall_spin.setRange(0.0, 1.0)
        self._cluster_recall_spin.setDecimals(3)
        self._cluster_recall_spin.setSingleStep(0.01)
        self._cluster_recall_spin.setValue(0.08)
        self._cluster_write_bg_only = QCheckBox("只写入背景(0)+当前类")
        self._cluster_write_bg_only.setChecked(True)
        self._cluster_rerun_btn = QPushButton("重跑上次类聚")
        self._cluster_rerun_btn.setEnabled(False)
        self._cluster_rerun_btn.clicked.connect(self._rerun_last_cluster)

        # ---------- Eraser toggle ----------
        self._eraser_toggle = QAction("橡皮 (E)", self)
        self._eraser_toggle.setCheckable(True)
        self._eraser_toggle.setShortcut(QKeySequence("E"))
        self._eraser_toggle.toggled.connect(self._canvas.set_eraser)

        # ---------- Build UI ----------
        self._build_ui()
        self._setup_shortcuts()
        self._rebuild_label_list()

        # 自动加载 codebook
        self._try_auto_load_codebook()

        # 如果有数据目录，刷新文件浏览器
        if cfg.data_dir:
            self._file_browser.set_data_dir(cfg.data_dir)

        # 初始模式 UI 状态
        self._apply_mode_ui()

    # ================================================================
    #                           UI
    # ================================================================

    def _build_ui(self) -> None:
        # ---------- Toolbar ----------
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

        # ---- 模式切换按钮 ----
        self._act_mode_view3d = QAction("🔍 3D查看", self)
        self._act_mode_view3d.setCheckable(True)
        self._act_mode_view3d.setChecked(True)
        self._act_mode_view3d.triggered.connect(lambda: self._switch_mode("view3d"))
        tb.addAction(self._act_mode_view3d)

        # ---------- Image name label ----------
        self._image_name_label = QLabel("  显示: --")
        tb.addWidget(self._image_name_label)

        # ---------- Right side panel (edit mode) ----------
        self._side_edit = QWidget()
        self._side_edit.setMinimumWidth(180)
        self._side_edit.setMaximumWidth(260)
        side_layout = QVBoxLayout(self._side_edit)
        side_layout.setContentsMargins(6, 6, 6, 6)

        side_layout.addWidget(QLabel("标签 (数字键切换)"))
        side_layout.addWidget(self._label_list, 1)

        side_layout.addSpacing(6)
        side_layout.addWidget(self._brush_label)
        side_layout.addWidget(self._brush_slider)

        side_layout.addSpacing(6)
        side_layout.addWidget(self._alpha_label)
        side_layout.addWidget(self._alpha_slider)

        side_layout.addSpacing(6)
        cluster_box = QGroupBox("类聚参数")
        cluster_form = QFormLayout(cluster_box)
        cluster_form.setContentsMargins(8, 6, 8, 6)
        cluster_form.setHorizontalSpacing(8)
        cluster_form.setVerticalSpacing(4)
        cluster_form.addRow("簇数 k", self._cluster_k_spin)
        cluster_form.addRow("采样上限", self._cluster_sample_spin)
        cluster_form.addRow("迭代次数", self._cluster_iter_spin)
        cluster_form.addRow("最小种子", self._cluster_seed_min_spin)
        cluster_form.addRow("局部半径", self._cluster_radius_spin)
        cluster_form.addRow("精度阈值", self._cluster_precision_spin)
        cluster_form.addRow("召回阈值", self._cluster_recall_spin)
        cluster_form.addRow("", self._cluster_write_bg_only)
        cluster_form.addRow("", self._cluster_rerun_btn)
        side_layout.addWidget(cluster_box)

        side_layout.addStretch(1)

        # ---------- Right side panel (view3d mode): empty placeholder ----------
        self._side_view3d = QWidget()
        self._side_view3d.setMinimumWidth(180)
        self._side_view3d.setMaximumWidth(260)
        v3d_layout = QVBoxLayout(self._side_view3d)
        v3d_layout.setContentsMargins(6, 6, 6, 6)
        v3d_layout.addStretch(1)

        # ---------- Stacked widget for right panel ----------
        self._side_stack = QStackedWidget()
        self._side_stack.addWidget(self._side_view3d)   # index 0
        self._side_stack.addWidget(self._side_edit)      # index 1
        # 右侧提示面板已关闭：收缩为 0 宽，避免留下空白占位
        self._side_stack.setMinimumWidth(0)
        self._side_stack.setMaximumWidth(0)
        self._side_stack.setVisible(False)

        # ---------- 中间区域: canvas 左侧 + 3D 右侧（用水平 splitter） ----------
        self._center_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._center_splitter.addWidget(self._canvas)
        self._center_splitter.addWidget(self._view3d)
        self._center_splitter.setStretchFactor(0, 3)
        self._center_splitter.setStretchFactor(1, 2)
        self._center_splitter.setSizes([900, 450])

        # ---------- 顶层 splitter ----------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._file_browser)
        splitter.addWidget(self._center_splitter)
        splitter.addWidget(self._side_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(2, True)
        splitter.setSizes([250, 1000, 0])

        self.setCentralWidget(splitter)

        # ---------- Status bar ----------
        self._status_coord = QLabel("x=-, y=-")
        self._status_code = QLabel("code=-")
        self._status_mode = QLabel("[3D查看]")
        self._status_mode.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        self._status_file = QLabel("")
        self.statusBar().addWidget(self._status_mode, 0)
        self.statusBar().addWidget(self._status_coord, 0)
        self.statusBar().addWidget(self._status_code, 0)
        self.statusBar().addPermanentWidget(self._status_file, 1)

    def _setup_shortcuts(self) -> None:
        # PgUp/PgDn 上下翻页
        sc_prev = QShortcut(QKeySequence("PgUp"), self)
        sc_prev.activated.connect(self._file_browser.go_prev)
        sc_next = QShortcut(QKeySequence("PgDown"), self)
        sc_next.activated.connect(self._file_browser.go_next)

    def _adjust_brush(self, delta: int) -> None:
        new_r = max(1, min(100, self._brush_slider.value() + delta))
        self._brush_slider.setValue(new_r)

    # ================================================================
    #                    Mode switching
    # ================================================================

    def _switch_mode(self, mode: str) -> None:
        self._mode = mode
        self._canvas.set_mode(mode)
        self._apply_mode_ui()

    def _toggle_mode(self) -> None:
        new_mode = "edit_mask" if self._mode == "view3d" else "view3d"
        self._switch_mode(new_mode)

    def _apply_mode_ui(self) -> None:
        # 固定只保留 3D 查看模式
        self._mode = "view3d"
        self._act_mode_view3d.setChecked(True)
        self._side_stack.setCurrentIndex(0)
        self._view3d.setVisible(True)
        self._status_mode.setText("[3D查看]")
        self._status_mode.setStyleSheet("color: #4fc3f7; font-weight: bold;")

    # ================================================================
    #                    Dialogs / Actions
    # ================================================================

    def _fits_filter(self) -> str:
        return "FITS (*.fits *.fit *.fts);;All (*.*)"

    def _mask_filter(self) -> str:
        return "Mask files (*.png *.fits *.fit *.fts *.bmp *.tif *.tiff);;All (*.*)"

    def set_data_dir_dialog(self) -> None:
        start = self._cfg.data_dir or str(Path.cwd())
        p = QFileDialog.getExistingDirectory(self, "选择数据目录", start)
        if not p:
            return
        self._cfg.data_dir = str(Path(p))
        self._cfg.save()
        self._file_browser.set_data_dir(self._cfg.data_dir)
        self._try_auto_load_codebook()
        self.statusBar().showMessage(f"数据目录: {self._cfg.data_dir}")

    def _try_auto_load_codebook(self) -> None:
        if not self._cfg.data_dir:
            return
        root = Path(self._cfg.data_dir)
        candidates = [
            root / "mask_codebook.json",
            root / "codebook.json",
            root / "labels.json",
        ]
        for c in candidates:
            if c.exists():
                self._load_codebook(c)
                return

    def open_image_dialog(self) -> None:
        start = str(self._ref_path.parent if self._ref_path else Path(self._cfg.data_dir))
        p, _ = QFileDialog.getOpenFileName(self, "选择图像FITS", start, self._fits_filter())
        if not p:
            return
        self._load_reference(Path(p))

    def open_mask_dialog(self) -> None:
        start = str(self._mask_path.parent if self._mask_path else Path(self._cfg.data_dir))
        p, _ = QFileDialog.getOpenFileName(self, "选择Mask文件", start, self._mask_filter())
        if not p:
            return
        self._load_mask(Path(p))

    def open_codebook_dialog(self) -> None:
        start = str(self._codebook_path.parent if self._codebook_path else Path(self._cfg.data_dir))
        p, _ = QFileDialog.getOpenFileName(
            self, "选择mask_codebook", start, "JSON/CSV/FITS (*.json *.csv *.fits *.fit *.fts);;All (*.*)"
        )
        if not p:
            return
        self._load_codebook(Path(p))

    def save_mask(self) -> None:
        if self._mask_path is None:
            return self.save_mask_as()
        self._do_save(self._mask_path)

    def save_mask_as(self) -> None:
        if self._mask_path:
            start = str(self._mask_path)
        else:
            start = str(Path(self._cfg.data_dir) / "mask.png")
        p, _ = QFileDialog.getSaveFileName(self, "保存Mask文件", start, self._mask_filter())
        if not p:
            return
        self._do_save(Path(p))

    # ================================================================
    #                    Load / Save
    # ================================================================

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

    def _set_canvas_mask_programmatically(self, mask: np.ndarray) -> None:
        """加载/切换显示时更新画布，不把程序行为记为用户编辑。"""
        self._suspend_dirty_tracking = True
        try:
            self._canvas.load_mask(mask)
            self._view3d.set_mask(mask)
        finally:
            self._suspend_dirty_tracking = False

    def _load_mask(self, path: Path, check_dirty: bool = True) -> None:
        if check_dirty and self._dirty and not self._confirm_discard():
            return
        self._reset_cluster_rerun_state()
        mask = read_mask_image(path)
        self._mask_header = None
        if path.suffix.lower() in {".fits", ".fit", ".fts"}:
            fi = read_fits_image(path)
            self._mask_header = fi.header
        self._set_canvas_mask_programmatically(mask)
        self._canvas.set_custom_overlay_argb(None)
        self._mask_path = path
        self._source_mask = np.ascontiguousarray(mask.astype(np.int32, copy=False))
        self._dirty = False
        self._status_file.setText(
            f"📷 {self._ref_path.name if self._ref_path else '--'} | 🎭 {path.name}"
        )
        self.statusBar().showMessage(f"已加载mask: {path.name}  ({mask.shape[1]}×{mask.shape[0]})")

    def _load_codebook(self, path: Path) -> None:
        labels = load_codebook(path, alpha=255)
        labels = ensure_unique_codes(labels)
        self._labels = labels
        self._canvas.set_labels(labels)
        self._view3d.set_labels(labels)
        self._rebuild_label_list()
        self._codebook_path = path
        self.statusBar().showMessage(f"已加载codebook: {path.name}（{len(labels)}类）")

    def _do_save(self, path: Path) -> None:
        mask = self._canvas.get_mask()
        if mask is None:
            QMessageBox.warning(self, "无法保存", "当前没有mask数据。")
            return
        write_mask_image(path, mask, header=self._mask_header, overwrite=True)
        self._mask_path = path
        self._dirty = False
        self.statusBar().showMessage(f"✅ 已保存: {path.name}")

    def _on_tile_selected(self, tile: TileGroup) -> None:
        if self._dirty and not self._confirm_discard():
            return

        self._reset_cluster_rerun_state()
        self._current_tile = tile
        self._canvas.clear_all()
        self._ref_path = None
        self._aligned_path = None
        self._mask_path = None
        self._source_mask = None
        self._raw_ref = None
        self._raw_aligned = None
        self._dirty = False

        if tile.reference:
            self._load_reference(tile.reference)
        if tile.mask:
            self._load_mask(tile.mask)

        if not tile.has_mask:
            if tile.reference:
                fi = read_fits_image(tile.reference)
                h, w = fi.data.shape[:2]
                empty_mask = np.zeros((h, w), dtype=np.int32)
                self._set_canvas_mask_programmatically(empty_mask)
                self._dirty = False
                self.statusBar().showMessage(f"已创建空白mask ({w}×{h})")

        # 默认进入 3D 查看模式
        self._switch_mode("view3d")

        self._image_name_label.setText("  显示: reference")
        self.setWindowTitle(f"FITS Mask Label Editor - {tile.tile_id}")

    # ================================================================
    #                    3D View interaction
    # ================================================================

    def _on_view3d_click(self, x: int, y: int) -> None:
        """3D 模式下点击图像时的处理。"""
        patch_size = self._view3d.get_patch_size()
        self._canvas.show_region_rect(x, y, patch_size)
        self._view3d.update_view(x, y)
        self.statusBar().showMessage(f"3D 查看: 中心({x}, {y})  {patch_size}×{patch_size} px")

    # ================================================================
    #                    Blink / Toggle
    # ================================================================

    def _toggle_image(self) -> None:
        name = self._canvas.toggle_base_image()
        self._image_name_label.setText(f"  显示: {name}")

    def _toggle_blink(self, on: bool) -> None:
        self._blinking = on
        if on:
            self._blink_timer.start()
        else:
            self._blink_timer.stop()

    def _on_blink_tick(self) -> None:
        self._canvas.toggle_base_image()
        name = self._canvas.current_base_name()
        self._image_name_label.setText(f"  显示: {name}")

    def _refresh_file_list(self) -> None:
        if self._cfg.data_dir:
            self._file_browser.set_data_dir(self._cfg.data_dir)
            self.statusBar().showMessage("已刷新文件列表")

    # ================================================================
    #                    Callbacks
    # ================================================================

    def _on_mask_changed(self) -> None:
        if self._suspend_dirty_tracking:
            return
        self._dirty = True
        cur = self._canvas.get_mask()
        if cur is not None:
            self._source_mask = cur

    def _on_cursor_pixel(self, x: int, y: int, code: int) -> None:
        self._status_coord.setText(f"x={x}, y={y}")
        if code >= 0:
            label_name = "?"
            for la in self._labels:
                if la.code == code:
                    label_name = la.name
                    break
            self._status_code.setText(f"code={code} ({label_name})")
        else:
            # 3D 模式下可能没有 mask
            # 显示原始像素值
            val_str = ""
            if self._raw_ref is not None and 0 <= y < self._raw_ref.shape[0] and 0 <= x < self._raw_ref.shape[1]:
                val_str = f"ref={self._raw_ref[y, x]:.1f}"
            if self._raw_aligned is not None and 0 <= y < self._raw_aligned.shape[0] and 0 <= x < self._raw_aligned.shape[1]:
                if val_str:
                    val_str += f"  ali={self._raw_aligned[y, x]:.1f}"
                else:
                    val_str = f"ali={self._raw_aligned[y, x]:.1f}"
            self._status_code.setText(val_str if val_str else "")

    def _on_color_picked(self, code: int) -> None:
        for i, la in enumerate(self._labels):
            if la.code == code:
                self._label_list.setCurrentRow(i)
                self.statusBar().showMessage(f"取色: {la.name} (code={code})")
                return
        self.statusBar().showMessage(f"取色: 未知标签 (code={code})")

    def _on_label_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._labels):
            return
        la = self._labels[idx]
        self._canvas.set_current_code(la.code)
        if self._eraser_toggle.isChecked():
            self._eraser_toggle.setChecked(False)

    def _on_brush_changed(self, v: int) -> None:
        self._canvas.set_brush_radius(v)
        self._brush_label.setText(f"笔刷: {v}")

    def _on_alpha_changed(self, v: int) -> None:
        alpha = float(v) / 100.0
        self._canvas.set_overlay_opacity(alpha)
        self._alpha_label.setText(f"透明度: {v}%")

    def _select_label_by_number(self, num: int) -> None:
        for i, la in enumerate(self._labels):
            if la.code == num:
                self._label_list.setCurrentRow(i)
                return
        if num < len(self._labels):
            self._label_list.setCurrentRow(num)

    # ================================================================
    #                    Label list
    # ================================================================

    def _on_cluster_for_label(self, code: int) -> None:
        base = self._canvas.get_mask()
        if base is None:
            QMessageBox.information(self, "类聚", "当前没有可操作的 mask。")
            return
        if self._mode != "edit_mask":
            self._switch_mode("edit_mask")
        # 保存本次类聚前的基线，供“重跑上次类聚”回退后再计算。
        self._cluster_base_mask = base
        self._last_cluster_code = int(code)
        self._cluster_rerun_btn.setEnabled(True)
        self._run_kmeans_expand_for_code(code)

    def _rerun_last_cluster(self) -> None:
        if self._last_cluster_code is None:
            self.statusBar().showMessage("还没有可重跑的类聚记录。")
            return
        if self._cluster_base_mask is None:
            self.statusBar().showMessage("缺少类聚基线，先执行一次标签类聚。")
            return
        if self._mode != "edit_mask":
            self._switch_mode("edit_mask")
        base = self._cluster_base_mask.copy()
        self._set_canvas_mask_programmatically(base)
        self._canvas.set_custom_overlay_argb(None)
        self._source_mask = base
        self._dirty = True
        self._run_kmeans_expand_for_code(int(self._last_cluster_code))

    def _reset_cluster_rerun_state(self) -> None:
        self._last_cluster_code = None
        self._cluster_base_mask = None
        self._cluster_rerun_btn.setEnabled(False)

    @staticmethod
    def _box_mean(img: np.ndarray, radius: int = 1) -> np.ndarray:
        """积分图实现均值滤波，避免引入 scipy 依赖。"""
        r = max(0, int(radius))
        if r == 0:
            return np.asarray(img, dtype=np.float32)
        x = np.asarray(img, dtype=np.float32)
        x_pad = np.pad(x, ((r, r), (r, r)), mode="reflect")
        ii = np.pad(x_pad, ((1, 0), (1, 0)), mode="constant")
        ii = np.cumsum(np.cumsum(ii, axis=0), axis=1)
        k = 2 * r + 1
        area = float(k * k)
        s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
        return s / area

    @classmethod
    def _local_mean_std(cls, img: np.ndarray, radius: int = 1) -> tuple[np.ndarray, np.ndarray]:
        mu = cls._box_mean(img, radius=radius)
        mu2 = cls._box_mean(np.asarray(img, dtype=np.float32) ** 2, radius=radius)
        var = np.maximum(mu2 - mu * mu, 1e-8)
        return mu.astype(np.float32, copy=False), np.sqrt(var, dtype=np.float32)

    def _build_cluster_features(self, stat_radius: int) -> np.ndarray | None:
        if self._raw_ref is None or self._raw_aligned is None:
            return None
        ref = np.asarray(self._raw_ref, dtype=np.float32)
        ali = np.asarray(self._raw_aligned, dtype=np.float32)
        if ref.shape != ali.shape:
            return None
        diff = ref - ali
        adiff = np.abs(diff)
        r = max(0, int(stat_radius))
        ref_mu, ref_std = self._local_mean_std(ref, radius=r)
        ali_mu, ali_std = self._local_mean_std(ali, radius=r)
        ad_mu = self._box_mean(adiff, radius=r)
        feat = np.stack(
            [ref, ali, diff, adiff, ref_mu, ref_std, ali_mu, ali_std, ad_mu],
            axis=-1,
        )
        return feat.astype(np.float32, copy=False)

    def _get_cluster_params(self) -> dict[str, float | int | bool]:
        k = int(self._cluster_k_spin.value())
        sample_n = int(self._cluster_sample_spin.value())
        max_iter = int(self._cluster_iter_spin.value())
        min_seed = int(self._cluster_seed_min_spin.value())
        stat_radius = int(self._cluster_radius_spin.value())
        p_th = float(self._cluster_precision_spin.value())
        r_th = float(self._cluster_recall_spin.value())
        bg_only = bool(self._cluster_write_bg_only.isChecked())
        return {
            "k": max(2, k),
            "sample_n": max(1, sample_n),
            "max_iter": max(1, max_iter),
            "min_seed": max(1, min_seed),
            "stat_radius": max(0, stat_radius),
            "precision_th": min(max(p_th, 0.0), 1.0),
            "recall_th": min(max(r_th, 0.0), 1.0),
            "bg_only": bg_only,
        }

    def _kmeans_fit(self, x: np.ndarray, k: int, max_iter: int = 25) -> np.ndarray:
        n = x.shape[0]
        if n == 0:
            raise ValueError("kmeans 输入为空")
        if n <= k:
            return x.copy()
        idx = self._cluster_rng.choice(n, size=k, replace=False)
        centers = np.ascontiguousarray(x[idx], dtype=np.float32)
        for _ in range(max_iter):
            d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(d2, axis=1)
            new_centers = centers.copy()
            for ci in range(k):
                sel = labels == ci
                if not np.any(sel):
                    new_centers[ci] = x[self._cluster_rng.integers(0, n)]
                else:
                    new_centers[ci] = x[sel].mean(axis=0)
            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if shift < 1e-4:
                break
        return centers

    @staticmethod
    def _assign_to_centers(x: np.ndarray, centers: np.ndarray, chunk: int = 250000) -> np.ndarray:
        n = x.shape[0]
        out = np.empty(n, dtype=np.int16)
        for i in range(0, n, chunk):
            j = min(n, i + chunk)
            d2 = ((x[i:j, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            out[i:j] = np.argmin(d2, axis=1).astype(np.int16, copy=False)
        return out

    def _run_kmeans_expand_for_code(self, code: int) -> None:
        p = self._get_cluster_params()
        k = int(p["k"])
        sample_n_cfg = int(p["sample_n"])
        max_iter = int(p["max_iter"])
        min_seed = int(p["min_seed"])
        stat_radius = int(p["stat_radius"])
        precision_th = float(p["precision_th"])
        recall_th = float(p["recall_th"])
        bg_only = bool(p["bg_only"])

        cur = self._canvas.get_mask()
        if cur is None:
            QMessageBox.information(self, "类聚", "当前没有可操作的 mask。")
            return
        if self._raw_ref is None or self._raw_aligned is None:
            QMessageBox.information(self, "类聚", "需要同时加载 reference 与 aligned 才能类聚。")
            return
        if cur.shape != self._raw_ref.shape or cur.shape != self._raw_aligned.shape:
            QMessageBox.warning(self, "类聚", "mask 与图像尺寸不一致，无法类聚。")
            return

        seed = cur == int(code)
        seed_count = int(np.count_nonzero(seed))
        if seed_count < min_seed:
            QMessageBox.information(
                self,
                "类聚",
                f"标签 code={code} 的已标注像素太少（{seed_count}），至少需要 {min_seed}。",
            )
            return

        feat = self._build_cluster_features(stat_radius=stat_radius)
        if feat is None:
            QMessageBox.warning(self, "类聚", "无法构建类聚特征。")
            return

        h, w, d = feat.shape
        x_all = feat.reshape(-1, d)
        x_mean = x_all.mean(axis=0, keepdims=True)
        x_std = x_all.std(axis=0, keepdims=True)
        x_std = np.where(x_std < 1e-6, 1.0, x_std)
        x_all = (x_all - x_mean) / x_std

        n = x_all.shape[0]
        sample_n = min(sample_n_cfg, n)
        if sample_n < k:
            sample_n = k
        sample_idx = self._cluster_rng.choice(n, size=sample_n, replace=False)
        x_sample = x_all[sample_idx]

        centers = self._kmeans_fit(x_sample, k=k, max_iter=max_iter)
        labels_all = self._assign_to_centers(x_all, centers=centers).reshape(h, w)
        seed_labels = labels_all[seed]
        seed_hist = np.bincount(seed_labels, minlength=k)
        best = int(np.argmax(seed_hist))

        total_hist = np.bincount(labels_all.reshape(-1), minlength=k)
        cluster_precision = seed_hist / np.maximum(total_hist, 1)
        cluster_recall = seed_hist / max(seed_count, 1)

        target_clusters = [best]
        for ci in range(k):
            if ci == best:
                continue
            if cluster_precision[ci] >= precision_th and cluster_recall[ci] >= recall_th:
                target_clusters.append(ci)
        target_clusters_set = set(target_clusters)

        candidate = np.isin(labels_all, list(target_clusters_set))
        if bg_only:
            writable = (cur == 0) | (cur == int(code))
        else:
            writable = np.ones_like(cur, dtype=bool)
        new_hits = candidate & writable
        add_only = new_hits & (cur != int(code))
        add_count = int(np.count_nonzero(add_only))
        if add_count <= 0:
            self.statusBar().showMessage(f"类聚完成：code={code}，没有新增像素。")
            return

        out = cur.copy()
        out[new_hits] = int(code)
        self._set_canvas_mask_programmatically(out)
        self._canvas.set_custom_overlay_argb(None)
        self._source_mask = out
        self._dirty = True
        self.statusBar().showMessage(
            f"类聚完成：code={code}，新增 {add_count} px（簇: {sorted(target_clusters_set)}）"
        )

    def _rebuild_label_list(self) -> None:
        self._label_list.blockSignals(True)
        self._label_list.clear()
        for la in self._labels:
            item = QListWidgetItem()
            item.setSizeHint(QPixmap(1, 28).size())
            self._label_list.addItem(item)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(6)
            r, g, b, a = la.color_rgba
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: rgba({r},{g},{b},{a}); border: 1px solid #444; border-radius: 2px;"
            )
            text = QLabel(f"{la.code}: {la.name}")
            btn = QPushButton("类聚")
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda _=False, c=la.code: self._on_cluster_for_label(c))
            row_layout.addWidget(swatch, 0)
            row_layout.addWidget(text, 1)
            row_layout.addWidget(btn, 0)
            self._label_list.setItemWidget(item, row)
        self._label_list.blockSignals(False)
        if self._labels:
            self._label_list.setCurrentRow(0)
            self._canvas.set_current_code(self._labels[0].code)

    # ================================================================
    #                    Misc
    # ================================================================

    def _confirm_discard(self) -> bool:
        r = QMessageBox.question(
            self,
            "未保存的修改",
            "当前 mask 有未保存修改，是否丢弃并继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return r == QMessageBox.StandardButton.Yes

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._dirty and not self._confirm_discard():
            event.ignore()
            return
        self._blink_timer.stop()
        event.accept()
