from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def gray8_to_qimage(gray: np.ndarray) -> QImage:
    """
    gray: (H,W) uint8
    返回的 QImage 依赖底层 numpy buffer，调用方必须持有 gray 引用以防被 GC。
    """
    if gray.dtype != np.uint8:
        raise TypeError("gray must be uint8")
    if gray.ndim != 2:
        raise ValueError("gray must be HxW")
    h, w = gray.shape
    bytes_per_line = w
    return QImage(gray.data, w, h, bytes_per_line, QImage.Format_Grayscale8)


def argb32_to_qimage(argb: np.ndarray) -> QImage:
    """
    argb: (H,W) uint32, 0xAARRGGBB
    返回的 QImage 依赖底层 numpy buffer，调用方必须持有 argb 引用以防被 GC。
    """
    if argb.dtype != np.uint32:
        raise TypeError("argb must be uint32")
    if argb.ndim != 2:
        raise ValueError("argb must be HxW")
    h, w = argb.shape
    bytes_per_line = w * 4
    return QImage(argb.data, w, h, bytes_per_line, QImage.Format_ARGB32)

