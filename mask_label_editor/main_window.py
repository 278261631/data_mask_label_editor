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


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("FITS Mask Label Editor")
        self._cfg = cfg

        self._ref_path: Path | None = None
        self._aligned_path: Path | None = None
        self._mask_path: Path | None = None
        self._codebook_path: Path | None = None

        self._mask_header = None  # 保存 FITS mask 的 header

        self._labels: list[Label] = default_labels(alpha=255)

        # ---------- Canvas ----------
        self._canvas = MaskCanvas()
        self._canvas.set_overlay_opacity(cfg.overlay_alpha)
        self._canvas.set_brush_radius(cfg.brush_radius)
        self._canvas.mask_changed.connect(self._on_mask_changed)
        self._canvas.cursor_pixel.connect(self._on_cursor_pixel)
        self._canvas.color_picked.connect(self._on_color_picked)

        self._dirty = False
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._on_blink_tick)
        self._blinking = False

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

        self._act_blink = QAction("👁 闪烁对比 (B)", self)
        self._act_blink.setCheckable(True)
        self._act_blink.setShortcut(QKeySequence("B"))
        self._act_blink.toggled.connect(self._toggle_blink)
        tb.addAction(self._act_blink)

        act_toggle = QAction("⇆ 切换图像 (Tab)", self)
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

        # ---------- Right side panel ----------
        side = QWidget()
        side.setMinimumWidth(180)
        side.setMaximumWidth(260)
        side_layout = QVBoxLayout(side)
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
            "  E=橡皮  F=适应窗口\n"
            "  1-9=选择标签  0=背景\n"
            "  PgUp/PgDn=上/下一张"
        )
        help_text.setStyleSheet("color: #888; font-size: 10px;")
        help_text.setWordWrap(True)
        side_layout.addWidget(help_text)

        # ---------- Layout with splitter ----------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._file_browser)
        splitter.addWidget(self._canvas)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 0)  # file browser: fixed
        splitter.setStretchFactor(1, 1)  # canvas: stretch
        splitter.setStretchFactor(2, 0)  # side panel: fixed
        splitter.setSizes([250, 700, 200])

        self.setCentralWidget(splitter)

        # ---------- Status bar ----------
        self._status_coord = QLabel("x=-, y=-")
        self._status_code = QLabel("code=-")
        self._status_file = QLabel("")
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

    def _adjust_brush(self, delta: int) -> None:
        new_r = max(1, min(100, self._brush_slider.value() + delta))
        self._brush_slider.setValue(new_r)

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
        """自动查找数据目录中的 mask_codebook.json。"""
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
        gray8 = to_uint8_view(img.data)
        self._canvas.load_base_gray8(gray8, slot="a")
        self._canvas.fit_view()
        self._ref_path = path
        self._image_name_label.setText(f"  显示: reference")
        self._status_file.setText(f"📷 {path.name}")

    def _load_aligned(self, path: Path) -> None:
        img = read_fits_image(path)
        gray8 = to_uint8_view(img.data)
        self._canvas.load_base_gray8(gray8, slot="b")
        self._aligned_path = path

    def _load_mask(self, path: Path) -> None:
        if self._dirty and not self._confirm_discard():
            return
        mask = read_mask_image(path)
        self._mask_header = None  # PNG 没有 FITS header
        if path.suffix.lower() in {".fits", ".fit", ".fts"}:
            from mask_label_editor.fits_io import read_fits_image as _rfi
            fi = _rfi(path)
            self._mask_header = fi.header
        self._canvas.load_mask(mask)
        self._mask_path = path
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
        """从文件列表选择一个 tile 时自动加载整组文件。"""
        if self._dirty and not self._confirm_discard():
            return

        self._canvas.clear_all()
        self._ref_path = None
        self._aligned_path = None
        self._mask_path = None
        self._dirty = False

        if tile.reference:
            self._load_reference(tile.reference)
        if tile.aligned:
            self._load_aligned(tile.aligned)
        if tile.mask:
            self._load_mask(tile.mask)

        if not tile.has_mask:
            # 没有现成 mask，创建空白 mask
            if tile.reference:
                from mask_label_editor.fits_io import read_fits_image as _rfi
                fi = _rfi(tile.reference)
                h, w = fi.data.shape[:2]
                empty_mask = np.zeros((h, w), dtype=np.int32)
                self._canvas.load_mask(empty_mask)
                self._dirty = False
                self.statusBar().showMessage(f"已创建空白mask ({w}×{h})")

        self._image_name_label.setText(f"  显示: reference")
        self.setWindowTitle(f"FITS Mask Label Editor - {tile.tile_id}")

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
        self._dirty = True

    def _on_cursor_pixel(self, x: int, y: int, code: int) -> None:
        self._status_coord.setText(f"x={x}, y={y}")
        # 找到对应的标签名
        label_name = "?"
        for la in self._labels:
            if la.code == code:
                label_name = la.name
                break
        self._status_code.setText(f"code={code} ({label_name})")

    def _on_color_picked(self, code: int) -> None:
        """右键取色：选择对应标签。"""
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
        # 切掉橡皮模式
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
        """数字键 0-9 选择对应 code 的标签。"""
        # 先尝试按 code 匹配
        for i, la in enumerate(self._labels):
            if la.code == num:
                self._label_list.setCurrentRow(i)
                return
        # fallback: 按列表索引
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
            # 创建带颜色方块的 item
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
