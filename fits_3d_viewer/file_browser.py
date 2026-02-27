from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass
class TileGroup:
    """表示一组对齐的 tile 文件：reference、aligned。"""

    tile_id: str  # e.g. "tile0001"
    reference: Path | None = None
    aligned: Path | None = None
    pred_png: Path | None = None
    display_name: str = ""
    predict_display: str = "-"
    data_kind: str = "data"

    @property
    def has_pair(self) -> bool:
        return self.reference is not None and self.aligned is not None

    @property
    def has_pred_png(self) -> bool:
        return self.pred_png is not None

    @property
    def is_predict(self) -> bool:
        return self.pred_png is not None


def discover_tiles(data_dir: str | Path) -> list[TileGroup]:
    """扫描目录并按“单个 FITS 文件”返回条目。"""
    root = Path(data_dir)
    result: list[TileGroup] = []
    search_dirs = [root]
    tiles_dir = root / "tiles"
    if tiles_dir.is_dir():
        search_dirs.append(tiles_dir)

    fit_exts = (".fits", ".fit", ".fts")
    def _find_existing(candidates: list[Path]) -> Path | None:
        for p in candidates:
            if p.exists() and p.is_file():
                return p
        return None

    def _prefix(stem: str) -> str:
        # 兼容原命名：xxx_1_reference / xxx_2_aligned
        return re.sub(r"_(1_reference|2_aligned)$", "", stem, flags=re.IGNORECASE)

    for d in search_dirs:
        for f in sorted(d.iterdir()):
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if not name_lower.endswith(fit_exts):
                continue

            # 列表只显示图像 FITS，避免把概率派生文件本身作为主图条目
            if "_prob" in name_lower:
                continue

            stem = f.stem
            prefix = _prefix(stem)
            g = TileGroup(tile_id=stem, reference=f)

            if "_1_reference" in name_lower:
                ali_candidates = [f.with_name(f"{prefix}_2_aligned{ext}") for ext in fit_exts]
                g.aligned = _find_existing(ali_candidates)

            g.pred_png = _find_existing([f.with_name(f"{prefix}_pred.png")])

            status_parts = ["F"]
            if g.aligned:
                status_parts.append("A")
            pred_parts = []
            if g.pred_png:
                pred_parts.append("pred.png")

            g.predict_display = "/".join(pred_parts) if pred_parts else "-"
            g.data_kind = "predict" if g.is_predict else "data"
            g.display_name = f"{f.name} [{'/' .join(status_parts)}]"
            result.append(g)

    return sorted(result, key=lambda x: (x.reference or Path("")).name.lower())


class FileBrowser(QWidget):
    """文件浏览面板：显示数据目录中的单文件 FITS 列表。"""

    tile_selected = Signal(object)  # emits TileGroup

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tiles: list[TileGroup] = []
        self._current_idx: int = -1
        self._data_dir: str = ""

        self._dir_label = QLabel("未设置数据目录")
        self._dir_label.setWordWrap(True)
        self._dir_label.setStyleSheet("color: #888; font-size: 11px;")
        self._pred_info_label = QLabel("数据类型: - | 预测文件: -")
        self._pred_info_label.setWordWrap(True)
        self._pred_info_label.setStyleSheet("color: #666; font-size: 11px;")

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_row_changed)

        self._btn_prev = QPushButton("◀ 上一个")
        self._btn_next = QPushButton("下一个 ▶")
        self._btn_prev.clicked.connect(self.go_prev)
        self._btn_next.clicked.connect(self.go_next)

        nav = QHBoxLayout()
        nav.setContentsMargins(0, 0, 0, 0)
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._btn_next)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("文件列表"))
        layout.addWidget(self._dir_label)
        layout.addWidget(self._pred_info_label)
        layout.addWidget(self._list, 1)
        layout.addLayout(nav)

    def set_data_dir(self, data_dir: str) -> None:
        self._data_dir = data_dir
        self._dir_label.setText(data_dir)
        self.refresh()

    def refresh(self) -> None:
        if not self._data_dir:
            return
        self._tiles = discover_tiles(self._data_dir)
        self._list.blockSignals(True)
        self._list.clear()
        for g in self._tiles:
            item = QListWidgetItem(g.display_name)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._current_idx = -1
        self._pred_info_label.setText("文件类型: - | 预测文件: -")

    def select_index(self, idx: int) -> None:
        if 0 <= idx < len(self._tiles):
            self._list.setCurrentRow(idx)

    def go_prev(self) -> None:
        if self._current_idx > 0:
            self.select_index(self._current_idx - 1)

    def go_next(self) -> None:
        if self._current_idx < len(self._tiles) - 1:
            self.select_index(self._current_idx + 1)

    def current_tile(self) -> TileGroup | None:
        if 0 <= self._current_idx < len(self._tiles):
            return self._tiles[self._current_idx]
        return None

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._tiles):
            return
        self._current_idx = row
        tile = self._tiles[row]
        self._pred_info_label.setText(f"文件类型: {tile.data_kind} | 预测文件: {tile.predict_display}")
        self.tile_selected.emit(tile)
