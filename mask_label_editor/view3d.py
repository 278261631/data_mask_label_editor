"""局部 3D 像素视图：并排显示 reference 和 aligned 两张图的 3D surface。"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("QtAgg")  # noqa: E402  must be before pyplot import

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.figure import Figure

from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel, QHBoxLayout, QSpinBox
from PySide6.QtCore import Qt

from mask_label_editor.labels import Label


class Dual3DView(QWidget):
    """并排显示 reference / aligned 的局部 3D surface plot。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._patch_size = 30  # 默认 30×30
        self._ref_data: np.ndarray | None = None   # 原始 16-bit 全图
        self._aligned_data: np.ndarray | None = None
        self._mask_data: np.ndarray | None = None   # mask 全图 (int32)
        self._labels: list[Label] = []

        # matplotlib figure —— 1 行 2 列子图
        self._fig = Figure(figsize=(10, 4), dpi=100, facecolor="#2b2b2b")
        self._ax_ref = self._fig.add_subplot(1, 2, 1, projection="3d", facecolor="#1e1e1e")
        self._ax_ali = self._fig.add_subplot(1, 2, 2, projection="3d", facecolor="#1e1e1e")
        self._fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92, wspace=0.15)

        self._mpl_canvas = FigureCanvas(self._fig)
        self._mpl_canvas.setMinimumHeight(300)

        # 顶部信息栏
        self._info_label = QLabel("点击图像选择查看区域")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("color: #ccc; font-size: 12px; padding: 4px;")

        # Patch size 调节
        size_label = QLabel("区域大小:")
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

        self._style_axes(self._ax_ref, "Reference")
        self._style_axes(self._ax_ali, "Aligned")
        self._mpl_canvas.draw_idle()

    # ----------------------------------------------------------------
    #  公共 API
    # ----------------------------------------------------------------

    def set_data(self, ref: np.ndarray | None, aligned: np.ndarray | None) -> None:
        """设置原始 16-bit 全图数据。"""
        self._ref_data = ref
        self._aligned_data = aligned

    def set_mask(self, mask: np.ndarray | None) -> None:
        """设置 mask 全图数据。"""
        self._mask_data = mask

    def set_labels(self, labels: list[Label]) -> None:
        """设置标签列表（用于 mask 着色）。"""
        self._labels = labels

    def get_patch_size(self) -> int:
        return self._patch_size

    def update_view(self, cx: int, cy: int) -> None:
        """以 (cx, cy) 为中心提取 patch 并绘制 3D surface。"""
        half = self._patch_size // 2

        # 提取 mask patch 用于着色
        mask_patch = None
        if self._mask_data is not None:
            mask_patch = self._extract_patch(self._mask_data, cx, cy, half)

        self._ax_ref.cla()
        self._ax_ali.cla()

        drawn = False
        if self._ref_data is not None:
            patch_r = self._extract_patch(self._ref_data, cx, cy, half)
            if patch_r is not None:
                fc = self._build_facecolors(mask_patch, patch_r) if mask_patch is not None else None
                self._plot_surface(self._ax_ref, patch_r, "Reference", cx, cy,
                                   facecolors=fc)
                drawn = True
            else:
                self._style_axes(self._ax_ref, "Reference (无数据)")
        else:
            self._style_axes(self._ax_ref, "Reference (未加载)")

        if self._aligned_data is not None:
            patch_a = self._extract_patch(self._aligned_data, cx, cy, half)
            if patch_a is not None:
                fc = self._build_facecolors(mask_patch, patch_a) if mask_patch is not None else None
                self._plot_surface(self._ax_ali, patch_a, "Aligned", cx, cy,
                                   facecolors=fc)
                drawn = True
            else:
                self._style_axes(self._ax_ali, "Aligned (无数据)")
        else:
            self._style_axes(self._ax_ali, "Aligned (未加载)")

        if drawn:
            self._info_label.setText(
                f"中心: ({cx}, {cy})  |  区域: {self._patch_size}×{self._patch_size} px"
            )

        self._mpl_canvas.draw_idle()

    # ----------------------------------------------------------------
    #  内部方法
    # ----------------------------------------------------------------

    def _on_size_changed(self, val: int) -> None:
        self._patch_size = val

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

    def _build_label_hue_map(self) -> dict[int, tuple[float, float, float]]:
        """从 labels 构建 code -> RGB(0~1) 色调映射。
        code=0 (background/normal) 如果原色太暗，改用浅蓝色调以在暗背景上清晰可见。
        """
        cmap: dict[int, tuple[float, float, float]] = {}
        for la in self._labels:
            r, g, b, _a = la.color_rgba
            # 如果是 code=0 且颜色太暗（灰/黑），替换为浅蓝色调
            if la.code == 0 and (r + g + b) < 400:
                cmap[la.code] = (0.2, 0.4, 1.0)  # 蓝色
            else:
                cmap[la.code] = (r / 255.0, g / 255.0, b / 255.0)
        return cmap

    def _build_facecolors(self, mask_patch: np.ndarray, img_patch: np.ndarray) -> np.ndarray:
        """
        根据 mask 色调 + 像素亮度 构建 facecolors。
        - mask code 决定色调（红/绿/灰等）
        - 像素值决定该色调的明暗变化
        facecolors 尺寸是 (H-1, W-1, 4)。
        """
        ph, pw = mask_patch.shape
        hue_map = self._build_label_hue_map()

        # 默认色调（未定义的 code）：中灰
        default_hue = (0.5, 0.5, 0.5)

        # 将像素值归一化到 0~1 作为亮度因子
        vmin = np.nanmin(img_patch)
        vmax = np.nanmax(img_patch)
        if vmax <= vmin:
            vmax = vmin + 1.0
        brightness = (img_patch - vmin) / (vmax - vmin)  # 0~1

        # 将亮度映射到 0.25~1.0 范围，避免太暗看不清
        brightness = 0.25 + brightness * 0.75

        # facecolors: (H-1, W-1, 4)
        fh = max(ph - 1, 1)
        fw = max(pw - 1, 1)
        fc = np.zeros((fh, fw, 4), dtype=np.float64)
        for iy in range(fh):
            for ix in range(fw):
                code = int(mask_patch[iy, ix])
                hr, hg, hb = hue_map.get(code, default_hue)
                bri = float(brightness[iy, ix])
                fc[iy, ix] = (hr * bri, hg * bri, hb * bri, 1.0)
        return fc

    def _plot_surface(self, ax, patch: np.ndarray, title: str, cx: int, cy: int,
                      facecolors: np.ndarray | None = None) -> None:
        ph, pw = patch.shape
        half = self._patch_size // 2
        X = np.arange(cx - half, cx - half + pw)
        Y = np.arange(cy - half, cy - half + ph)
        X, Y = np.meshgrid(X, Y)

        if facecolors is not None and facecolors.shape[0] == ph - 1 and facecolors.shape[1] == pw - 1:
            ax.plot_surface(X, Y, patch, facecolors=facecolors, edgecolor="none",
                            alpha=0.9, rstride=1, cstride=1, antialiased=False, shade=True)
        else:
            ax.plot_surface(X, Y, patch, cmap="coolwarm", edgecolor="none",
                            alpha=0.9, rstride=1, cstride=1, antialiased=False)

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
