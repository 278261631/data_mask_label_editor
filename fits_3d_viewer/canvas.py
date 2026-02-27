from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from fits_3d_viewer.qt_image import gray8_to_qimage


class ImageCanvas(QGraphicsView):
    """仅用于图像浏览与 3D 取点的画布。"""

    cursor_pixel = Signal(int, int, int)  # x, y, code(-1)
    view3d_click = Signal(int, int)  # x, y

    def __init__(self) -> None:
        super().__init__()

        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._base_item = QGraphicsPixmapItem()
        self._scene.addItem(self._base_item)

        rect_pen = QPen(QColor(0, 255, 255, 200), 2.0)
        rect_pen.setCosmetic(True)
        self._region_rect = self._scene.addRect(QRectF(), rect_pen, Qt.BrushStyle.NoBrush)
        self._region_rect.setZValue(99)
        self._region_rect.setVisible(False)

        self._mode: str = "view3d"

        self._img_gray8_a: np.ndarray | None = None
        self._img_gray8_b: np.ndarray | None = None
        self._showing_b: bool = False

        self._panning = False
        self._pan_start = QPointF()

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def show_region_rect(self, cx: int, cy: int, size: int) -> None:
        half = size // 2
        self._region_rect.setRect(QRectF(cx - half, cy - half, size, size))
        self._region_rect.setVisible(True)

    def load_base_gray8(self, gray8: np.ndarray, slot: str = "a") -> None:
        buf = np.ascontiguousarray(gray8, dtype=np.uint8)
        if slot == "b":
            self._img_gray8_b = buf
        else:
            self._img_gray8_a = buf

        if (slot == "a" and not self._showing_b) or (slot == "b" and self._showing_b):
            self._show_base(buf)
        elif slot == "a" and self._showing_b is False:
            self._show_base(buf)
        if slot == "a" and self._img_gray8_b is None:
            self._showing_b = False
            self._show_base(buf)

    def toggle_base_image(self) -> str:
        if self._img_gray8_a is None and self._img_gray8_b is None:
            return "none"
        if self._showing_b:
            if self._img_gray8_a is not None:
                self._showing_b = False
                self._show_base(self._img_gray8_a)
                return "reference"
            return "aligned"
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
        self._base_item.setOffset(0, 0)
        self._scene.setSceneRect(0, 0, qimg.width(), qimg.height())

    def fit_view(self) -> None:
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def clear_all(self) -> None:
        self._img_gray8_a = None
        self._img_gray8_b = None
        self._showing_b = False
        self._base_item.setPixmap(QPixmap())
        self._region_rect.setVisible(False)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if delta > 0 else 1 / 1.2
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def _pos_to_img_pixel(self, event) -> tuple[int, int] | None:
        sp = self.mapToScene(event.position().toPoint())
        x = int(np.floor(sp.x()))
        y = int(np.floor(sp.y()))
        sr = self._scene.sceneRect()
        if x < 0 or y < 0 or x >= sr.width() or y >= sr.height():
            return None
        return x, y

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            pix = self._pos_to_img_pixel(event)
            if pix:
                self.view3d_click.emit(pix[0], pix[1])
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pix = self._pos_to_img_pixel(event)
        if pix:
            self.cursor_pixel.emit(pix[0], pix[1], -1)

        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)
