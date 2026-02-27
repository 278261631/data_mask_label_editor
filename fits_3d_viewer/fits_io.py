from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


@dataclass
class FitsImage:
    path: Path
    data: np.ndarray
    header: Any


def read_fits_image(path: str | Path) -> FitsImage:
    p = Path(path)
    with fits.open(p) as hdul:
        # 通常主 HDU 就是图像；若不是，尝试找第一个 image HDU
        hdu = None
        for candidate in hdul:
            if getattr(candidate, "data", None) is None:
                continue
            if candidate.data is None:
                continue
            if isinstance(candidate.data, np.ndarray):
                hdu = candidate
                break
        if hdu is None:
            raise ValueError(f"No image data found in FITS: {p}")
        data = np.asarray(hdu.data)
        header = hdu.header

    # 常见 FITS 可能是 (1,H,W) 或 (H,W,1) 之类，这里尽量 squeeze
    data = np.squeeze(data)
    return FitsImage(path=p, data=data, header=header)


def write_fits_image(path: str | Path, data: np.ndarray, header: Any | None = None, overwrite: bool = True) -> None:
    p = Path(path)
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(p, overwrite=overwrite)


def robust_minmax_u16(x: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> tuple[float, float]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo = np.percentile(x, lo_pct)
    hi = np.percentile(x, hi_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def to_uint8_view(x: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    x = np.asarray(x)
    if vmin is None or vmax is None:
        vmin2, vmax2 = robust_minmax_u16(x)
        vmin = vmin if vmin is not None else vmin2
        vmax = vmax if vmax is not None else vmax2
    x2 = (x.astype(np.float32) - float(vmin)) / (float(vmax) - float(vmin))
    x2 = np.clip(x2, 0.0, 1.0)
    return (x2 * 255.0).astype(np.uint8)

