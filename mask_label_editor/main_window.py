from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence, QPixmap, QIcon, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSlider,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from mask_label_editor.canvas import MaskCanvas
from mask_label_editor.codebook import load_codebook
from mask_label_editor.config import AppConfig
from mask_label_editor.file_browser import FileBrowser, TileGroup
from mask_label_editor.fits_io import (
    read_fits_image,
    read_mask_image,
    to_uint8_view,
    write_mask_image,
)
from mask_label_editor.labels import Label, default_labels, ensure_unique_codes
from mask_label_editor.view3d import Dual3DView


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("FITS Mask Label Editor")
        self._cfg = cfg

        self._ref_path: Path | None = None
        self._aligned_path: Path | None = None
        self._mask_path: Path | None = None
        self._prob_path: Path | None = None
        self._codebook_path: Path | None = None
        self._current_tile: TileGroup | None = None
        self._source_mask: np.ndarray | None = None
        self._source_prob: np.ndarray | None = None
        self._prob_display_mode: str = "mask"
        self._suspend_dirty_tracking = False

        self._mask_header = None

        # 原始 16-bit 全图数据（用于 3D 视图）
        self._raw_ref: np.ndarray | None = None
        self._raw_aligned: np.ndarray | None = None

        self._labels: list[Label] = default_labels(alpha=255)

        # ---------- 当前模式 ----------
        self._mode = "view3d"  # "view3d" | "edit_mask"

        # ---------- Canvas ----------
        self._canvas = MaskCanvas()
        self._canvas.set_overlay_opacity(cfg.overlay_alpha)
        self._canvas.set_brush_radius(cfg.brush_radius)
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
        self._brush_slider.setValue(cfg.brush_radius)
        self._brush_slider.valueChanged.connect(self._on_brush_changed)
        self._brush_label = QLabel(f"笔刷: {cfg.brush_radius}")

        # ---------- Alpha slider ----------
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.setValue(int(cfg.overlay_alpha * 100))
        self._alpha_slider.valueChanged.connect(self._on_alpha_changed)
        self._alpha_label = QLabel(f"透明度: {int(cfg.overlay_alpha * 100)}%")

        # ---------- Prob overlay mode ----------
        self._prob_mode_label = QLabel("叠加:")
        self._prob_mode_combo = QComboBox()
        self._prob_mode_combo.addItems(["Mask", "Prob-Argmax", "Prob-Heatmap", "Confidence"])
        self._prob_mode_combo.currentTextChanged.connect(self._on_prob_mode_changed)

        self._prob_class_label = QLabel("类别:")
        self._prob_class_combo = QComboBox()
        self._prob_class_combo.currentIndexChanged.connect(self._on_prob_class_changed)
        self._prob_class_combo.setEnabled(False)

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

        # 恢复上次打开的文件
        if cfg.last_codebook_path and Path(cfg.last_codebook_path).exists():
            self._load_codebook(Path(cfg.last_codebook_path))

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

        self._act_mode_edit = QAction("✏️ Mask编辑", self)
        self._act_mode_edit.setCheckable(True)
        self._act_mode_edit.setChecked(False)
        self._act_mode_edit.triggered.connect(lambda: self._switch_mode("edit_mask"))
        tb.addAction(self._act_mode_edit)

        tb.addSeparator()

        act_open_img = QAction("打开图像FITS", self)
        act_open_img.triggered.connect(self.open_image_dialog)
        tb.addAction(act_open_img)

        act_open_mask = QAction("打开Mask", self)
        act_open_mask.triggered.connect(self.open_mask_dialog)
        tb.addAction(act_open_mask)

        act_open_codebook = QAction("打开Codebook", self)
        act_open_codebook.triggered.connect(self.open_codebook_dialog)
        tb.addAction(act_open_codebook)

        tb.addSeparator()

        act_save = QAction("💾 保存 (Ctrl+S)", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.save_mask)
        tb.addAction(act_save)

        act_save_as = QAction("另存为", self)
        act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_save_as.triggered.connect(self.save_mask_as)
        tb.addAction(act_save_as)

        act_undo = QAction("↩ 撤销 (Ctrl+Z)", self)
        act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        act_undo.triggered.connect(self._canvas.undo)
        tb.addAction(act_undo)

        tb.addSeparator()
        tb.addAction(self._eraser_toggle)

        self._act_blink = QAction("👁 闪烁 (B)", self)
        self._act_blink.setCheckable(True)
        self._act_blink.setShortcut(QKeySequence("B"))
        self._act_blink.toggled.connect(self._toggle_blink)
        tb.addAction(self._act_blink)

        act_toggle_mask_prob = QAction("🎭⇆📈 切换Mask/Prob (P)", self)
        act_toggle_mask_prob.setShortcut(QKeySequence("P"))
        act_toggle_mask_prob.triggered.connect(self._toggle_mask_prob)
        tb.addAction(act_toggle_mask_prob)

        tb.addSeparator()
        tb.addWidget(self._prob_mode_label)
        tb.addWidget(self._prob_mode_combo)
        tb.addWidget(self._prob_class_label)
        tb.addWidget(self._prob_class_combo)

        act_toggle = QAction("⇆ 切换 (Tab)", self)
        act_toggle.setShortcut(QKeySequence("Tab"))
        act_toggle.triggered.connect(self._toggle_image)
        tb.addAction(act_toggle)

        act_fit = QAction("适应窗口 (F)", self)
        act_fit.setShortcut(QKeySequence("F"))
        act_fit.triggered.connect(self._canvas.fit_view)
        tb.addAction(act_fit)

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

        # ---------- Help text ----------
        help_text = QLabel(
            "快捷键:\n"
            "  左键=绘制  中键=平移\n"
            "  右键=取色  Ctrl+滚轮=缩放\n"
            "  Shift+滚轮=笔刷大小\n"
            "  Tab=切换图像  B=闪烁对比\n"
            "  P=切换Mask/Prob\n"
            "  E=橡皮  F=适应窗口\n"
            "  M=切换模式\n"
            "  1-9=选择标签  0=背景\n"
            "  PgUp/PgDn=上/下一张"
        )
        help_text.setStyleSheet("color: #888; font-size: 10px;")
        help_text.setWordWrap(True)
        side_layout.addWidget(help_text)

        # ---------- Right side panel (view3d mode): empty placeholder ----------
        self._side_view3d = QWidget()
        self._side_view3d.setMinimumWidth(180)
        self._side_view3d.setMaximumWidth(260)
        v3d_layout = QVBoxLayout(self._side_view3d)
        v3d_layout.setContentsMargins(6, 6, 6, 6)
        v3d_info = QLabel(
            "3D 查看模式\n\n"
            "左键点击图像查看局部\n"
            "3D 像素分布\n\n"
            "快捷键:\n"
            "  中键=平移\n"
            "  Ctrl+滚轮=缩放\n"
            "  Tab=切换图像\n"
            "  B=闪烁对比\n"
            "  P=切换Mask/Prob\n"
            "  F=适应窗口\n"
            "  M=切换到编辑模式\n"
            "  PgUp/PgDn=上/下一张"
        )
        v3d_info.setStyleSheet("color: #aaa; font-size: 11px;")
        v3d_info.setWordWrap(True)
        v3d_layout.addWidget(v3d_info)
        v3d_layout.addStretch(1)

        # ---------- Stacked widget for right panel ----------
        self._side_stack = QStackedWidget()
        self._side_stack.addWidget(self._side_view3d)   # index 0
        self._side_stack.addWidget(self._side_edit)      # index 1

        # ---------- 中间区域: canvas 上面 + 3D 下面（用垂直 splitter） ----------
        self._center_splitter = QSplitter(Qt.Orientation.Vertical)
        self._center_splitter.addWidget(self._canvas)
        self._center_splitter.addWidget(self._view3d)
        self._center_splitter.setStretchFactor(0, 2)
        self._center_splitter.setStretchFactor(1, 1)
        self._center_splitter.setSizes([500, 300])

        # ---------- 顶层 splitter ----------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._file_browser)
        splitter.addWidget(self._center_splitter)
        splitter.addWidget(self._side_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([250, 800, 200])

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
        # 数字键 0-9 选择标签
        for i in range(10):
            sc = QShortcut(QKeySequence(str(i)), self)
            sc.activated.connect(lambda idx=i: self._select_label_by_number(idx))

        # PgUp/PgDn 上下翻页
        sc_prev = QShortcut(QKeySequence("PgUp"), self)
        sc_prev.activated.connect(self._file_browser.go_prev)
        sc_next = QShortcut(QKeySequence("PgDown"), self)
        sc_next.activated.connect(self._file_browser.go_next)

        # +/- 调整笔刷大小
        sc_inc = QShortcut(QKeySequence("+"), self)
        sc_inc.activated.connect(lambda: self._adjust_brush(2))
        sc_inc2 = QShortcut(QKeySequence("="), self)
        sc_inc2.activated.connect(lambda: self._adjust_brush(2))
        sc_dec = QShortcut(QKeySequence("-"), self)
        sc_dec.activated.connect(lambda: self._adjust_brush(-2))

        # M 键切换模式
        sc_mode = QShortcut(QKeySequence("M"), self)
        sc_mode.activated.connect(self._toggle_mode)

        # P 键切换 Mask / Prob 叠加
        sc_prob = QShortcut(QKeySequence("P"), self)
        sc_prob.activated.connect(self._toggle_mask_prob)

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
        is_view3d = (self._mode == "view3d")

        self._act_mode_view3d.setChecked(is_view3d)
        self._act_mode_edit.setChecked(not is_view3d)

        if is_view3d:
            self._side_stack.setCurrentIndex(0)
            self._view3d.setVisible(True)
            self._status_mode.setText("[3D查看]")
            self._status_mode.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        else:
            self._side_stack.setCurrentIndex(1)
            self._view3d.setVisible(False)
            self._canvas.hide_region_rect()
            self._status_mode.setText("[Mask编辑]")
            self._status_mode.setStyleSheet("color: #ff9800; font-weight: bold;")

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
        mask = read_mask_image(path)
        self._mask_header = None
        if path.suffix.lower() in {".fits", ".fit", ".fts"}:
            fi = read_fits_image(path)
            self._mask_header = fi.header
        self._set_canvas_mask_programmatically(mask)
        self._canvas.set_custom_overlay_argb(None)
        self._mask_path = path
        self._prob_path = None
        self._source_mask = np.ascontiguousarray(mask.astype(np.int32, copy=False))
        self._dirty = False
        self._status_file.setText(
            f"📷 {self._ref_path.name if self._ref_path else '--'} | 🎭 {path.name}"
        )
        self.statusBar().showMessage(f"已加载mask: {path.name}  ({mask.shape[1]}×{mask.shape[0]})")

    def _load_prob_npz(self, path: Path, check_dirty: bool = True) -> None:
        """加载 prob.npz 并将概率张量转换为 argmax 标签图叠加显示。"""
        if check_dirty and self._dirty and not self._confirm_discard():
            return
        with np.load(path, allow_pickle=False) as obj:
            keys = list(obj.files)
            if not keys:
                raise ValueError(f"prob.npz 不包含数组: {path}")
            preferred = ("prob", "probs", "pred", "arr_0")
            key = next((k for k in preferred if k in obj.files), keys[0])
            arr = np.asarray(obj[key])

        label_map = self._prob_to_label_map(arr)
        self._mask_header = None
        self._set_canvas_mask_programmatically(label_map)
        self._canvas.set_custom_overlay_argb(None)
        self._mask_path = None
        self._prob_path = path
        self._source_prob = np.asarray(arr)
        self._source_mask = np.ascontiguousarray(label_map.astype(np.int32, copy=False))
        self._refresh_prob_class_combo()
        self._dirty = False
        self._status_file.setText(
            f"📷 {self._ref_path.name if self._ref_path else '--'} | 📈 {path.name}"
        )
        h, w = label_map.shape
        self.statusBar().showMessage(f"已加载prob(argmax): {path.name}  ({w}×{h})")

    @staticmethod
    def _prob_to_label_map(prob: np.ndarray) -> np.ndarray:
        """将 prob 数组转换为 HxW 的 int32 标签图。"""
        arr = np.asarray(prob)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            # 若已经是 2D，则视作 label map 或单通道概率图（四舍五入到整数）
            if np.issubdtype(arr.dtype, np.integer):
                return np.ascontiguousarray(arr.astype(np.int32, copy=False))
            return np.ascontiguousarray(np.rint(arr).astype(np.int32, copy=False))
        if arr.ndim != 3:
            raise ValueError(f"不支持的 prob 维度: {arr.shape}")

        # 估计通道维：优先取最小维作为类别通道（通常类别数远小于 H/W）
        ch_axis = int(np.argmin(arr.shape))
        label_map = np.argmax(arr, axis=ch_axis)
        return np.ascontiguousarray(label_map.astype(np.int32, copy=False))

    def _load_codebook(self, path: Path) -> None:
        labels = load_codebook(path, alpha=255)
        labels = ensure_unique_codes(labels)
        self._labels = labels
        self._canvas.set_labels(labels)
        self._view3d.set_labels(labels)
        self._rebuild_label_list()
        self._codebook_path = path
        self._cfg.last_codebook_path = str(path)
        self._cfg.save()
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

        self._current_tile = tile
        self._canvas.clear_all()
        self._ref_path = None
        self._aligned_path = None
        self._mask_path = None
        self._prob_path = None
        self._source_mask = None
        self._source_prob = None
        self._raw_ref = None
        self._raw_aligned = None
        self._dirty = False

        if tile.reference:
            self._load_reference(tile.reference)
        if tile.aligned:
            self._load_aligned(tile.aligned)
        if tile.prob_npz:
            # 有 prob 时默认进入 prob 视图
            self._load_prob_npz(tile.prob_npz)
        elif tile.mask:
            self._load_mask(tile.mask)

        if not tile.has_mask and tile.prob_npz is None:
            if tile.reference:
                fi = read_fits_image(tile.reference)
                h, w = fi.data.shape[:2]
                empty_mask = np.zeros((h, w), dtype=np.int32)
                self._set_canvas_mask_programmatically(empty_mask)
                self._dirty = False
                self.statusBar().showMessage(f"已创建空白mask ({w}×{h})")

        # 默认进入 3D 查看模式
        self._switch_mode("view3d")

        # 默认叠加模式：
        # - 有 prob.npz: Prob-Heatmap，类别优先 1（不足则回退 0）
        # - 无 prob.npz: Mask
        if tile.prob_npz is not None:
            default_cls = 1 if self._prob_class_combo.count() > 1 else 0
            if self._prob_class_combo.count() > 0:
                self._prob_class_combo.setCurrentIndex(default_cls)
            self._prob_mode_combo.setCurrentText("Prob-Heatmap")
        else:
            self._prob_mode_combo.setCurrentText("Mask")

        self._image_name_label.setText("  显示: reference")
        self.setWindowTitle(f"FITS Mask Label Editor - {tile.tile_id}")
        self._apply_prob_display_mode()

    def _toggle_mask_prob(self) -> None:
        """在当前 tile 的 mask 与 prob(argmax) 之间切换显示。"""
        tile = self._current_tile
        if tile is None:
            self.statusBar().showMessage("请先选择一个 tile")
            return
        if tile.mask is None or tile.prob_npz is None:
            self.statusBar().showMessage("当前 tile 需要同时存在 mask 和 prob.npz 才能切换")
            return

        if self._prob_path is not None:
            self._load_mask(tile.mask, check_dirty=False)
            self._prob_mode_combo.setCurrentText("Mask")
        else:
            # 切到 prob 前保留当前未保存绘制，便于切回继续编辑
            cur = self._canvas.get_mask()
            if cur is not None:
                self._source_mask = cur
            self._load_prob_npz(tile.prob_npz, check_dirty=False)
            self._prob_mode_combo.setCurrentText("Prob-Argmax")

    def _on_prob_mode_changed(self, text: str) -> None:
        mode_map = {
            "Mask": "mask",
            "Prob-Argmax": "argmax",
            "Prob-Heatmap": "heatmap",
            "Confidence": "confidence",
        }
        self._prob_display_mode = mode_map.get(text, "mask")
        self._apply_prob_display_mode()

    def _on_prob_class_changed(self, _: int) -> None:
        if self._prob_display_mode == "heatmap":
            self._apply_prob_display_mode()

    def _refresh_prob_class_combo(self) -> None:
        self._prob_class_combo.blockSignals(True)
        self._prob_class_combo.clear()
        c = self._prob_channel_count(self._source_prob)
        if c <= 0:
            self._prob_class_combo.addItem("0")
        else:
            for i in range(c):
                self._prob_class_combo.addItem(str(i))
        self._prob_class_combo.blockSignals(False)

    def _apply_prob_display_mode(self) -> None:
        mode = self._prob_display_mode
        self._prob_class_combo.setEnabled(mode == "heatmap")

        if mode == "mask":
            if self._source_mask is not None:
                self._set_canvas_mask_programmatically(self._source_mask)
            self._canvas.set_custom_overlay_argb(None)
            return

        if self._source_prob is None:
            if self._source_mask is not None:
                self._set_canvas_mask_programmatically(self._source_mask)
            self._canvas.set_custom_overlay_argb(None)
            return

        prob3d = self._normalize_prob3d(self._source_prob)
        if prob3d is None:
            return

        if mode == "argmax":
            label_map = np.argmax(prob3d, axis=0).astype(np.int32, copy=False)
            self._source_mask = np.ascontiguousarray(label_map)
            self._set_canvas_mask_programmatically(self._source_mask)
            self._canvas.set_custom_overlay_argb(None)
            return

        if mode == "confidence":
            conf = np.max(prob3d, axis=0)
            overlay = self._make_heatmap_argb(conf)
            label_map = np.argmax(prob3d, axis=0).astype(np.int32, copy=False)
            self._source_mask = np.ascontiguousarray(label_map)
            self._set_canvas_mask_programmatically(self._source_mask)
            self._canvas.set_custom_overlay_argb(overlay)
            return

        ch_count = prob3d.shape[0]
        idx = self._prob_class_combo.currentIndex()
        if idx < 0 or idx >= ch_count:
            idx = 0
        hm = prob3d[idx]
        overlay = self._make_heatmap_argb(hm)
        label_map = np.argmax(prob3d, axis=0).astype(np.int32, copy=False)
        self._source_mask = np.ascontiguousarray(label_map)
        self._set_canvas_mask_programmatically(self._source_mask)
        self._canvas.set_custom_overlay_argb(overlay)

    @staticmethod
    def _prob_channel_count(prob: np.ndarray | None) -> int:
        if prob is None:
            return 0
        p = np.asarray(prob)
        p = np.squeeze(p)
        if p.ndim == 3:
            return int(np.min(p.shape))
        return 0

    @staticmethod
    def _normalize_prob3d(prob: np.ndarray) -> np.ndarray | None:
        arr = np.asarray(prob)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return np.ascontiguousarray(arr.astype(np.float32, copy=False)[None, ...])
        if arr.ndim != 3:
            return None
        ch_axis = int(np.argmin(arr.shape))
        if ch_axis == 0:
            out = arr
        elif ch_axis == 1:
            out = np.transpose(arr, (1, 0, 2))
        else:
            out = np.transpose(arr, (2, 0, 1))
        return np.ascontiguousarray(out.astype(np.float32, copy=False))

    @staticmethod
    def _make_heatmap_argb(v: np.ndarray) -> np.ndarray:
        x = np.asarray(v, dtype=np.float32)
        if x.size == 0:
            return np.zeros((0, 0), dtype=np.uint32)
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            x = np.zeros_like(x, dtype=np.float32)
        else:
            x = (x - lo) / (hi - lo)
            x = np.clip(x, 0.0, 1.0)

        r = np.clip(2.0 * x - 0.5, 0.0, 1.0)
        g = np.clip(2.0 * x, 0.0, 1.0) * np.clip(2.0 - 2.0 * x, 0.0, 1.0)
        b = np.clip(1.5 - 2.0 * x, 0.0, 1.0)
        a = np.full_like(x, 0.75, dtype=np.float32)

        rr = (r * 255.0).astype(np.uint32)
        gg = (g * 255.0).astype(np.uint32)
        bb = (b * 255.0).astype(np.uint32)
        aa = (a * 255.0).astype(np.uint32)
        return (aa << 24) | (rr << 16) | (gg << 8) | bb

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
        self._cfg.brush_radius = int(v)
        self._cfg.save()
        self._brush_label.setText(f"笔刷: {v}")

    def _on_alpha_changed(self, v: int) -> None:
        alpha = float(v) / 100.0
        self._canvas.set_overlay_opacity(alpha)
        self._cfg.overlay_alpha = alpha
        self._cfg.save()
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

    def _rebuild_label_list(self) -> None:
        self._label_list.blockSignals(True)
        self._label_list.clear()
        for la in self._labels:
            r, g, b, a = la.color_rgba
            pix = QPixmap(16, 16)
            pix.fill(QColor(r, g, b, a))
            icon = QIcon(pix)
            item = QListWidgetItem(icon, f"{la.code}: {la.name}")
            self._label_list.addItem(item)
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
