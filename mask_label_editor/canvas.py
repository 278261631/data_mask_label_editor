from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from mask_label_editor.labels import Label, labels_to_lut
from mask_label_editor.qt_image import argb32_to_qimage, gray8_to_qimage


@dataclass
class CanvasState:
    brush_radius: int = 8
    current_code: int = 1
    eraser: bool = False


class MaskCanvas(QGraphicsView):
    mask_changed = Signal()
    cursor_pixel = Signal(int, int, int)  # x, y, code
    color_picked = Signal(int)  # code under cursor (right-click pick)
    view3d_click = Signal(int, int)  # x, y — 3D查看模式下的点击坐标

    def __init__(self) -> None:
        super().__init__()

        self.setRenderHints(self.renderHints())
        # 默认不设 ScrollHandDrag，由中键处理平移
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._base_item = QGraphicsPixmapItem()
        self._overlay_item = QGraphicsPixmapItem()
        self._overlay_item.setOpacity(0.5)

        self._scene.addItem(self._base_item)
        self._scene.addItem(self._overlay_item)

        # 笔刷光标（圆圈）
        pen = QPen(QColor(255, 255, 0, 180), 1.0)
        pen.setCosmetic(True)  # 不随缩放变化线宽
        self._cursor_circle = QGraphicsEllipseItem()
        self._cursor_circle.setPen(pen)
        self._cursor_circle.setBrush(Qt.BrushStyle.NoBrush)
        self._cursor_circle.setZValue(100)
        self._cursor_circle.setVisible(False)
        self._scene.addItem(self._cursor_circle)

        # 3D 查看区域高亮框
        rect_pen = QPen(QColor(0, 255, 255, 200), 2.0)
        rect_pen.setCosmetic(True)
        self._region_rect = self._scene.addRect(QRectF(), rect_pen, Qt.BrushStyle.NoBrush)
        self._region_rect.setZValue(99)
        self._region_rect.setVisible(False)

        # 交互模式: "view3d" | "edit_mask"
        self._mode: str = "view3d"

        self._state = CanvasState()

        # 双图支持：两张底图，可切换
        self._img_gray8_a: np.ndarray | None = None  # reference
        self._img_gray8_b: np.ndarray | None = None  # aligned
        self._showing_b: bool = False  # 当前显示哪张

        self._overlay_argb: np.ndarray | None = None  # hold buffer
        self._custom_overlay_argb: np.ndarray | None = None

        self._mask: np.ndarray | None = None  # int32
        self._lut: np.ndarray | None = None  # uint32 LUT
        self._labels: list[Label] = []

        self._painting = False
        self._panning = False
        self._pan_start = QPointF()
        self._undo_stack: list[np.ndarray] = []
        self._max_undo = 30

        # 外部可挂钩
        self.on_info: Callable[[str], None] | None = None

    # -------------------- public API --------------------

    def set_mode(self, mode: str) -> None:
        """切换交互模式: 'view3d' 或 'edit_mask'。"""
        self._mode = mode
        if mode == "view3d":
            self._cursor_circle.setVisible(False)
            # mask overlay 保持显示，用于提示哪些区域需要查看
        else:
            self._region_rect.setVisible(False)

    def get_mode(self) -> str:
        return self._mode

    def show_region_rect(self, cx: int, cy: int, size: int) -> None:
        """在画布上显示 3D 查看区域的高亮框。"""
        half = size // 2
        self._region_rect.setRect(QRectF(cx - half, cy - half, size, size))
        self._region_rect.setVisible(True)

    def hide_region_rect(self) -> None:
        self._region_rect.setVisible(False)

    def set_overlay_opacity(self, alpha01: float) -> None:
        self._overlay_item.setOpacity(float(np.clip(alpha01, 0.0, 1.0)))

    def set_brush_radius(self, r: int) -> None:
        self._state.brush_radius = max(1, int(r))
        self._update_cursor_circle()

    def set_current_code(self, code: int) -> None:
        self._state.current_code = int(code)

    def set_eraser(self, enabled: bool) -> None:
        self._state.eraser = bool(enabled)
        self._update_cursor_color()

    def set_labels(self, labels: list[Label]) -> None:
        self._labels = labels
        self._rebuild_lut()
        self._refresh_overlay(full=True)

    def load_base_gray8(self, gray8: np.ndarray, slot: str = "a") -> None:
        """加载底图到 slot='a' (reference) 或 'b' (aligned)。"""
        buf = np.ascontiguousarray(gray8, dtype=np.uint8)
        if slot == "b":
            self._img_gray8_b = buf
        else:
            self._img_gray8_a = buf

        # 如果是当前显示的 slot，刷新
        if (slot == "a" and not self._showing_b) or (slot == "b" and self._showing_b):
            self._show_base(buf)
        elif slot == "a" and self._showing_b is False:
            self._show_base(buf)
        # 第一次加载时默认显示 a
        if slot == "a" and self._img_gray8_b is None:
            self._showing_b = False
            self._show_base(buf)

    def toggle_base_image(self) -> str:
        """切换 reference / aligned 底图，返回当前显示的 slot 名。"""
        if self._img_gray8_a is None and self._img_gray8_b is None:
            return "none"
        if self._showing_b:
            if self._img_gray8_a is not None:
                self._showing_b = False
                self._show_base(self._img_gray8_a)
                return "reference"
            return "aligned"
        else:
            if self._img_gray8_b is not None:
                self._showing_b = True
                self._show_base(self._img_gray8_b)
                return "aligned"
            return "reference"

    def current_base_name(self) -> str:
        return "aligned" if self._showing_b else "reference"

    def _show_base(self, gray8: np.ndarray) -> None:
        qimg = gray8_to_qimage(gray8)
        self._base_item.setPixmap(QPixmap.fromImage(qimg))
        self._overlay_item.setOffset(0, 0)
        self._base_item.setOffset(0, 0)
        self._scene.setSceneRect(0, 0, qimg.width(), qimg.height())

    def fit_view(self) -> None:
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def load_mask(self, mask: np.ndarray) -> None:
        m = np.asarray(mask)
        if m.ndim != 2:
            raise ValueError("mask must be HxW")
        self._mask = np.ascontiguousarray(m.astype(np.int32))
        self._rebuild_lut()
        self._ensure_overlay_buffer()
        self._refresh_overlay(full=True)
        self._undo_stack.clear()
        self.mask_changed.emit()

    def set_custom_overlay_argb(self, argb: np.ndarray | None) -> None:
        """设置/清除自定义叠加层（0xAARRGGBB）。"""
        if argb is None:
            self._custom_overlay_argb = None
            self._refresh_overlay(full=True)
            return
        arr = np.asarray(argb)
        if arr.ndim != 2:
            raise ValueError("custom overlay must be HxW uint32 argb")
        self._custom_overlay_argb = np.ascontiguousarray(arr.astype(np.uint32, copy=False))
        qimg = argb32_to_qimage(self._custom_overlay_argb)
        self._overlay_item.setPixmap(QPixmap.fromImage(qimg))

    def get_mask(self) -> np.ndarray | None:
        return None if self._mask is None else self._mask.copy()

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def undo(self) -> None:
        if not self._undo_stack or self._mask is None:
            return
        self._mask = self._undo_stack.pop()
        self._refresh_overlay(full=True)
        self.mask_changed.emit()

    def clear_all(self) -> None:
        """清空所有图像和 mask 数据。"""
        self._img_gray8_a = None
        self._img_gray8_b = None
        self._showing_b = False
        self._mask = None
        self._overlay_argb = None
        self._custom_overlay_argb = None
        self._undo_stack.clear()
        self._base_item.setPixmap(QPixmap())
        self._overlay_item.setPixmap(QPixmap())

    # -------------------- internals --------------------

    def _rebuild_lut(self) -> None:
        if self._mask is None:
            lut_tuple = labels_to_lut(self._labels, max_code=max([la.code for la in self._labels], default=7))
            self._lut = np.asarray(lut_tuple, dtype=np.uint32)
            return

        mask_max = int(np.max(self._mask)) if self._mask.size else 0
        label_max = max([la.code for la in self._labels], default=0)
        max_code = max(mask_max, label_max, 0)
        lut_tuple = labels_to_lut(self._labels, max_code=max_code)
        self._lut = np.asarray(lut_tuple, dtype=np.uint32)

        # 未定义 code 的像素用洋红
        if mask_max > label_max and self._lut.size > 0:
            unknown = 0x80FF00FF  # AARRGGBB
            for code in range(label_max + 1, self._lut.size):
                self._lut[code] = unknown

    def _ensure_overlay_buffer(self) -> None:
        if self._mask is None:
            return
        h, w = self._mask.shape
        if self._overlay_argb is None or self._overlay_argb.shape != (h, w):
            self._overlay_argb = np.zeros((h, w), dtype=np.uint32)

    def _refresh_overlay(self, full: bool = False, roi: tuple[int, int, int, int] | None = None) -> None:
        if self._mask is None or self._lut is None:
            return
        if self._custom_overlay_argb is not None:
            qimg = argb32_to_qimage(self._custom_overlay_argb)
            self._overlay_item.setPixmap(QPixmap.fromImage(qimg))
            return
        self._ensure_overlay_buffer()
        assert self._overlay_argb is not None
        h, w = self._mask.shape

        if full or roi is None:
            m = np.clip(self._mask, 0, self._lut.size - 1).astype(np.int32, copy=False)
            self._overlay_argb[:, :] = self._lut[m]
        else:
            x0, y0, x1, y1 = roi
            x0 = max(0, min(w, x0))
            x1 = max(0, min(w, x1))
            y0 = max(0, min(h, y0))
            y1 = max(0, min(h, y1))
            if x1 <= x0 or y1 <= y0:
                return
            m = np.clip(self._mask[y0:y1, x0:x1], 0, self._lut.size - 1).astype(np.int32, copy=False)
            self._overlay_argb[y0:y1, x0:x1] = self._lut[m]

        qimg = argb32_to_qimage(self._overlay_argb)
        self._overlay_item.setPixmap(QPixmap.fromImage(qimg))

    def _scene_pos_to_pixel(self, pos: QPointF) -> tuple[int, int] | None:
        if self._mask is None:
            return None
        x = int(np.floor(pos.x()))
        y = int(np.floor(pos.y()))
        h, w = self._mask.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return x, y

    def _paint_circle(self, x: int, y: int) -> tuple[int, int, int, int] | None:
        if self._mask is None:
            return None
        h, w = self._mask.shape
        r = int(self._state.brush_radius)
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        if x1 <= x0 or y1 <= y0:
            return None

        yy, xx = np.ogrid[y0:y1, x0:x1]
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
        new_code = 0 if self._state.eraser else int(self._state.current_code)
        region = self._mask[y0:y1, x0:x1]
        region[inside] = new_code
        return x0, y0, x1, y1

    def _update_cursor_circle(self) -> None:
        r = self._state.brush_radius
        self._cursor_circle.setRect(QRectF(-r, -r, r * 2, r * 2))

    def _update_cursor_color(self) -> None:
        if self._state.eraser:
            pen = QPen(QColor(255, 100, 100, 200), 1.5)
        else:
            pen = QPen(QColor(255, 255, 0, 200), 1.0)
        pen.setCosmetic(True)
        self._cursor_circle.setPen(pen)

    # -------------------- Qt events --------------------

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+滚轮：缩放
            factor = 1.2 if delta > 0 else 1 / 1.2
            self.scale(factor, factor)
            event.accept()
            return
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Shift+滚轮：调整笔刷大小
            new_r = self._state.brush_radius + (1 if delta > 0 else -1)
            new_r = max(1, min(100, new_r))
            self.set_brush_radius(new_r)
            event.accept()
            return
        super().wheelEvent(event)

    def _pos_to_img_pixel(self, event) -> tuple[int, int] | None:
        """从事件位置获取图像像素坐标（不依赖 mask 是否存在）。"""
        sp = self.mapToScene(event.position().toPoint())
        x = int(np.floor(sp.x()))
        y = int(np.floor(sp.y()))
        sr = self._scene.sceneRect()
        if x < 0 or y < 0 or x >= sr.width() or y >= sr.height():
            return None
        return x, y

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            # 中键平移
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # ---------- view3d 模式 ----------
        if self._mode == "view3d":
            if event.button() == Qt.MouseButton.LeftButton:
                pix = self._pos_to_img_pixel(event)
                if pix:
                    self.view3d_click.emit(pix[0], pix[1])
                event.accept()
                return
            super().mousePressEvent(event)
            return

        # ---------- edit_mask 模式 ----------
        if event.button() == Qt.MouseButton.RightButton and self._mask is not None:
            # 右键取色
            sp = self.mapToScene(event.position().toPoint())
            pix = self._scene_pos_to_pixel(sp)
            if pix:
                x, y = pix
                code = int(self._mask[y, x])
                self.color_picked.emit(code)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self._mask is not None:
            # 左键绘制
            self._painting = True
            self._undo_stack.append(self._mask.copy())
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack = self._undo_stack[-self._max_undo:]
            self._do_paint_at_event(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        sp = self.mapToScene(event.position().toPoint())

        # 根据模式决定光标显示
        if self._mode == "edit_mask":
            self._cursor_circle.setPos(sp)
            self._cursor_circle.setVisible(True)
        else:
            self._cursor_circle.setVisible(False)

        # 发出坐标信号
        pix = self._scene_pos_to_pixel(sp)
        if pix and self._mask is not None:
            x, y = pix
            code = int(self._mask[y, x])
            self.cursor_pixel.emit(x, y, code)
        elif self._mode == "view3d":
            # view3d 模式下即使没有 mask 也发坐标
            px = self._pos_to_img_pixel(event)
            if px:
                self.cursor_pixel.emit(px[0], px[1], -1)

        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return

        if self._painting:
            self._do_paint_at_event(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        if self._mode == "edit_mask":
            if event.button() == Qt.MouseButton.LeftButton and self._painting:
                self._painting = False
                self.mask_changed.emit()
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._cursor_circle.setVisible(False)
        super().leaveEvent(event)

    def enterEvent(self, event) -> None:
        self._cursor_circle.setVisible(True)
        super().enterEvent(event)

    def _do_paint_at_event(self, event) -> None:
        sp = self.mapToScene(event.position().toPoint())
        pix = self._scene_pos_to_pixel(sp)
        if pix is None:
            return
        x, y = pix
        roi = self._paint_circle(x, y)
        if roi is not None:
            self._refresh_overlay(roi=roi)
