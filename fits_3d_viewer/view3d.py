"""局部 3D 像素视图：显示 reference 图的 3D surface。"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("QtAgg")  # noqa: E402  must be before pyplot import

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel, QHBoxLayout, QSpinBox
from PySide6.QtCore import Qt, Signal

class Dual3DView(QWidget):
    """显示 reference 的局部 3D surface plot。"""
    patch_size_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._patch_size = 30  # 默认 30×30
        self._ref_data: np.ndarray | None = None   # 原始 16-bit 全图

        # matplotlib figure —— 上下两个子图
        # 上：自动 z 轴；下：固定 z=0~65535
        self._fig = Figure(figsize=(10, 7), dpi=100, facecolor="#2b2b2b")
        self._ax_ref = self._fig.add_subplot(2, 1, 1, projection="3d", facecolor="#1e1e1e")
        self._ax_ref_fixed = self._fig.add_subplot(2, 1, 2, projection="3d", facecolor="#1e1e1e")
        self._fig.subplots_adjust(left=0.04, right=0.98, bottom=0.03, top=0.95, hspace=0.25)

        self._mpl_canvas = FigureCanvas(self._fig)
        self._mpl_canvas.setMinimumHeight(520)

        # 顶部信息栏
        self._info_label = QLabel("Click image to inspect a local region")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("color: #ccc; font-size: 12px; padding: 4px;")

        # Patch size 调节
        size_label = QLabel("Patch size:")
        size_label.setStyleSheet("color: #ccc;")
        self._size_spin = QSpinBox()
        self._size_spin.setRange(10, 100)
        self._size_spin.setValue(self._patch_size)
        self._size_spin.setSuffix(" px")
        self._size_spin.valueChanged.connect(self._on_size_changed)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self._info_label, 1)
        top_bar.addWidget(size_label)
        top_bar.addWidget(self._size_spin)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top_bar)
        layout.addWidget(self._mpl_canvas, 1)

        self._style_axes(self._ax_ref, "Auto Z-axis")
        self._style_axes(self._ax_ref_fixed, "Fixed Z-axis (0-65535)")
        self._mpl_canvas.draw_idle()

    # ----------------------------------------------------------------
    #  公共 API
    # ----------------------------------------------------------------

    def set_data(self, ref: np.ndarray | None, aligned: np.ndarray | None = None) -> None:
        """设置原始 16-bit 全图数据。"""
        self._ref_data = ref

    def get_patch_size(self) -> int:
        return self._patch_size

    def set_patch_size(self, val: int) -> None:
        v = max(10, min(100, int(val)))
        if v == self._patch_size:
            return
        self._patch_size = v
        self._size_spin.blockSignals(True)
        self._size_spin.setValue(v)
        self._size_spin.blockSignals(False)

    def update_view(self, cx: int, cy: int) -> None:
        """以 (cx, cy) 为中心提取 patch 并绘制 3D surface。"""
        half = self._patch_size // 2

        self._ax_ref.cla()
        self._ax_ref_fixed.cla()

        drawn = False
        if self._ref_data is not None:
            patch_r = self._extract_patch(self._ref_data, cx, cy, half)
            if patch_r is not None:
                self._plot_surface(self._ax_ref, patch_r, "Auto Z-axis", cx, cy)
                self._plot_surface(
                    self._ax_ref_fixed,
                    patch_r,
                    "Fixed Z-axis (0-65535)",
                    cx,
                    cy,
                    zlim=(0.0, 65535.0),
                )
                drawn = True
            else:
                self._style_axes(self._ax_ref, "No data")
                self._style_axes(self._ax_ref_fixed, "No data")
        else:
            self._style_axes(self._ax_ref, "Not loaded")
            self._style_axes(self._ax_ref_fixed, "Not loaded")

        if drawn:
            self._info_label.setText(
                f"Center: ({cx}, {cy})  |  Patch: {self._patch_size}x{self._patch_size} px"
            )

        self._mpl_canvas.draw_idle()

    # ----------------------------------------------------------------
    #  内部方法
    # ----------------------------------------------------------------

    def _on_size_changed(self, val: int) -> None:
        self._patch_size = val
        self.patch_size_changed.emit(self._patch_size)

    def _extract_patch(self, data: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray | None:
        h, w = data.shape[:2]
        x0 = cx - half
        x1 = cx + half
        y0 = cy - half
        y1 = cy + half
        x0c = max(0, x0)
        x1c = min(w, x1)
        y0c = max(0, y0)
        y1c = min(h, y1)
        if x1c <= x0c or y1c <= y0c:
            return None
        return data[y0c:y1c, x0c:x1c].astype(np.float64)

    def _plot_surface(
        self,
        ax,
        patch: np.ndarray,
        title: str,
        cx: int,
        cy: int,
        zlim: tuple[float, float] | None = None,
    ) -> None:
        ph, pw = patch.shape
        half = self._patch_size // 2
        X = np.arange(cx - half, cx - half + pw)
        Y = np.arange(cy - half, cy - half + ph)
        X, Y = np.meshgrid(X, Y)

        ax.plot_surface(X, Y, patch, cmap="coolwarm", edgecolor="none",
                        alpha=1.0, rstride=1, cstride=1, antialiased=False)

        if zlim is not None:
            ax.set_zlim(zlim[0], zlim[1])

        ax.set_title(title, color="white", fontsize=11, pad=2)
        ax.set_xlabel("X", color="#aaa", fontsize=8, labelpad=1)
        ax.set_ylabel("Y", color="#aaa", fontsize=8, labelpad=1)
        ax.set_zlabel("Value", color="#aaa", fontsize=8, labelpad=1)

        # 暗色主题样式
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#555")
        ax.yaxis.pane.set_edgecolor("#555")
        ax.zaxis.pane.set_edgecolor("#555")
        ax.tick_params(colors="#999", labelsize=7)
        ax.grid(True, alpha=0.3)

    def _style_axes(self, ax, title: str) -> None:
        ax.set_title(title, color="white", fontsize=11, pad=2)
        ax.tick_params(colors="#999", labelsize=7)
        try:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        except Exception:
            pass
