from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    """表示一组对齐的 tile 文件：reference、aligned、mask。"""

    tile_id: str  # e.g. "tile0001"
    reference: Path | None = None
    aligned: Path | None = None
    mask: Path | None = None
    display_name: str = ""

    @property
    def has_pair(self) -> bool:
        return self.reference is not None and self.aligned is not None

    @property
    def has_mask(self) -> bool:
        return self.mask is not None


def discover_tiles(data_dir: str | Path) -> list[TileGroup]:
    """扫描目录（含子目录 tiles/），按 前缀+tile编号 分组。"""
    root = Path(data_dir)
    groups: dict[str, TileGroup] = {}

    # 扫描所有 fits 和 png 文件
    search_dirs = [root]
    tiles_dir = root / "tiles"
    if tiles_dir.is_dir():
        search_dirs.append(tiles_dir)

    for d in search_dirs:
        for f in sorted(d.iterdir()):
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            # 提取 前缀 + tile 编号，例如:
            #   GY3_K073-2_20250719_170125_-13._tile0001_1_reference.fits
            #   前缀 = "gy3_k073-2_20250719_170125_-13._tile0001"
            m = re.search(r"^(.+?tile[_]?\d+)", name_lower)
            if m is None:
                continue
            # 用前缀+tile编号作为唯一分组 key
            group_key = m.group(1)
            # 提取纯 tile 编号用于显示
            tm = re.search(r"(tile[_]?\d+)", group_key)
            tile_id = tm.group(1).replace("_", "") if tm else group_key

            if group_key not in groups:
                groups[group_key] = TileGroup(tile_id=tile_id)

            g = groups[group_key]

            if "_1_reference" in name_lower and name_lower.endswith((".fits", ".fit", ".fts")):
                g.reference = f
            elif "_2_aligned" in name_lower and name_lower.endswith((".fits", ".fit", ".fts")):
                g.aligned = f
            elif "_mask" in name_lower:
                g.mask = f

    # 设置 display_name 并排序
    result = sorted(groups.values(), key=lambda g: (g.reference or g.aligned or g.mask or Path("")).name)
    for g in result:
        ref_name = g.reference.stem if g.reference else ""
        # 去掉 _1_reference 后缀作为显示名
        display = re.sub(r"_1_reference$", "", ref_name, flags=re.IGNORECASE)
        if not display:
            display = g.tile_id
        status_parts = []
        if g.reference:
            status_parts.append("R")
        if g.aligned:
            status_parts.append("A")
        if g.mask:
            status_parts.append("M")
        g.display_name = f"{display} [{'/'.join(status_parts)}]"

    return result


class FileBrowser(QWidget):
    """文件浏览面板：显示数据目录中的 tile 列表。"""

    tile_selected = Signal(object)  # emits TileGroup

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tiles: list[TileGroup] = []
        self._current_idx: int = -1
        self._data_dir: str = ""

        self._dir_label = QLabel("未设置数据目录")
        self._dir_label.setWordWrap(True)
        self._dir_label.setStyleSheet("color: #888; font-size: 11px;")

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
        layout.addWidget(QLabel("Tile 列表"))
        layout.addWidget(self._dir_label)
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
            # 颜色标记：有 mask 的绿色，没有的灰色
            if g.has_mask:
                item.setForeground(Qt.GlobalColor.darkGreen)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._current_idx = -1

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
        self.tile_selected.emit(self._tiles[row])
