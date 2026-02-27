from __future__ import annotations

from collections import deque

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.stats import sigma_clip


BG_METHOD_ORIGINAL = "original"
BG_METHOD_MESH = "mesh_sigma_clip"
BG_METHOD_MORPH = "morph_opening"
BG_METHOD_POLY2 = "poly2d"
BG_METHOD_WAVELET = "wavelet_multiscale"
BG_METHOD_RPCA = "rpca_lowrank_sparse"
BG_METHOD_PIPELINE = "robust_pipeline"


def _fill_nan_with_median(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    out = arr.copy()
    m = np.isfinite(out)
    if not np.any(m):
        return np.zeros_like(out, dtype=np.float64)
    med = float(np.median(out[m]))
    out[~m] = med
    return out


def _upsample_bilinear(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    gy, gx = grid.shape
    if gy == h and gx == w:
        return grid.astype(np.float64, copy=False)
    x_src = np.linspace(0.0, 1.0, gx)
    y_src = np.linspace(0.0, 1.0, gy)
    x_dst = np.linspace(0.0, 1.0, w)
    y_dst = np.linspace(0.0, 1.0, h)
    grid_x = np.empty((gy, w), dtype=np.float64)
    for i in range(gy):
        grid_x[i, :] = np.interp(x_dst, x_src, grid[i, :])
    out = np.empty((h, w), dtype=np.float64)
    for j in range(w):
        out[:, j] = np.interp(y_dst, y_src, grid_x[:, j])
    return out


def _estimate_source_mask(data: np.ndarray, sigma: float = 3.5, dilate_radius: int = 2) -> np.ndarray:
    """粗略源掩膜：sigma-clipping + 轻度膨胀，避免目标参与背景估计。"""
    img = _fill_nan_with_median(data)
    clipped = sigma_clip(img, sigma=3.0, maxiters=3, masked=True)
    vals = np.asarray(clipped.compressed(), dtype=np.float64)
    if vals.size == 0:
        vals = img.reshape(-1)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-6
    robust_std = 1.4826 * mad
    src = img > (med + sigma * robust_std)
    if dilate_radius <= 0:
        return src
    return _max_filter2d(src.astype(np.float64), 2 * dilate_radius + 1) > 0.5


def estimate_background_mesh(data: np.ndarray, box: int = 64, clip_sigma: float = 3.0) -> np.ndarray:
    """分块稳健中值背景 + mask + 插值重建背景图。"""
    img = _fill_nan_with_median(data)
    h, w = img.shape
    b = max(8, int(box))
    source_mask = _estimate_source_mask(img, sigma=max(2.5, clip_sigma), dilate_radius=2)

    ys = list(range(0, h, b))
    xs = list(range(0, w, b))
    gy = len(ys)
    gx = len(xs)
    grid = np.zeros((gy, gx), dtype=np.float64)

    for iy, y0 in enumerate(ys):
        y1 = min(h, y0 + b)
        for ix, x0 in enumerate(xs):
            x1 = min(w, x0 + b)
            patch = img[y0:y1, x0:x1]
            patch_mask = source_mask[y0:y1, x0:x1]
            patch_bg = patch[~patch_mask]
            if patch_bg.size < max(16, (patch.size // 10)):
                patch_bg = patch.reshape(-1)
            clipped = sigma_clip(patch_bg, sigma=clip_sigma, maxiters=3, masked=True)
            vals = np.asarray(clipped.compressed(), dtype=np.float64)
            if vals.size == 0:
                vals = patch_bg.reshape(-1)
            grid[iy, ix] = float(np.median(vals))
    return _upsample_bilinear(grid, h, w)


def _min_filter1d(x: np.ndarray, k: int) -> np.ndarray:
    n = x.size
    r = max(1, int(k))
    if n == 0:
        return x.copy()
    pad = r // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty(n, dtype=np.float64)
    dq: deque[int] = deque()
    for i in range(xp.size):
        while dq and dq[0] <= i - r:
            dq.popleft()
        while dq and xp[dq[-1]] >= xp[i]:
            dq.pop()
        dq.append(i)
        if i >= r - 1 and (i - r + 1) < n:
            out[i - r + 1] = xp[dq[0]]
    return out


def _max_filter1d(x: np.ndarray, k: int) -> np.ndarray:
    n = x.size
    r = max(1, int(k))
    if n == 0:
        return x.copy()
    pad = r // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty(n, dtype=np.float64)
    dq: deque[int] = deque()
    for i in range(xp.size):
        while dq and dq[0] <= i - r:
            dq.popleft()
        while dq and xp[dq[-1]] <= xp[i]:
            dq.pop()
        dq.append(i)
        if i >= r - 1 and (i - r + 1) < n:
            out[i - r + 1] = xp[dq[0]]
    return out


def _min_filter2d(img: np.ndarray, k: int) -> np.ndarray:
    tmp = np.empty_like(img, dtype=np.float64)
    out = np.empty_like(img, dtype=np.float64)
    for y in range(img.shape[0]):
        tmp[y, :] = _min_filter1d(img[y, :], k)
    for x in range(img.shape[1]):
        out[:, x] = _min_filter1d(tmp[:, x], k)
    return out


def _max_filter2d(img: np.ndarray, k: int) -> np.ndarray:
    tmp = np.empty_like(img, dtype=np.float64)
    out = np.empty_like(img, dtype=np.float64)
    for y in range(img.shape[0]):
        tmp[y, :] = _max_filter1d(img[y, :], k)
    for x in range(img.shape[1]):
        out[:, x] = _max_filter1d(tmp[:, x], k)
    return out


def estimate_background_morphology(data: np.ndarray, radius: int = 24) -> np.ndarray:
    """形态学 opening 近似 Rolling Ball 背景。"""
    img = _fill_nan_with_median(data)
    k = max(3, int(radius) * 2 + 1)
    erosion = _min_filter2d(img, k)
    opening = _max_filter2d(erosion, k)
    return opening


def estimate_background_poly2d(
    data: np.ndarray,
    sample_size: int = 60000,
    clip_sigma: float = 3.0,
    seed: int = 20260227,
) -> np.ndarray:
    """二次多项式曲面背景拟合。"""
    img = _fill_nan_with_median(data)
    h, w = img.shape

    yy, xx = np.indices((h, w), dtype=np.float64)
    clipped = sigma_clip(img, sigma=clip_sigma, maxiters=3, masked=True)
    good = ~np.asarray(clipped.mask, dtype=bool)
    gy = yy[good]
    gx = xx[good]
    gz = img[good]
    if gz.size < 64:
        return np.full_like(img, float(np.median(img)), dtype=np.float64)

    if gz.size > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(gz.size, size=sample_size, replace=False)
        gx = gx[idx]
        gy = gy[idx]
        gz = gz[idx]

    # 标准化坐标，提升拟合稳定性
    xn = (gx / max(w - 1, 1)) * 2.0 - 1.0
    yn = (gy / max(h - 1, 1)) * 2.0 - 1.0
    A = np.column_stack([np.ones_like(xn), xn, yn, xn * yn, xn * xn, yn * yn])
    coef, *_ = np.linalg.lstsq(A, gz, rcond=None)

    xxn = (xx / max(w - 1, 1)) * 2.0 - 1.0
    yyn = (yy / max(h - 1, 1)) * 2.0 - 1.0
    bg = (
        coef[0]
        + coef[1] * xxn
        + coef[2] * yyn
        + coef[3] * xxn * yyn
        + coef[4] * xxn * xxn
        + coef[5] * yyn * yyn
    )
    return np.asarray(bg, dtype=np.float64)


def estimate_background_wavelet(data: np.ndarray, base_sigma: float = 8.0, levels: int = 4) -> np.ndarray:
    """多尺度分解：取最粗尺度作为背景估计。"""
    img = _fill_nan_with_median(data)
    cur = img
    sigma = max(2.0, float(base_sigma))
    lv = max(2, int(levels))
    for i in range(lv):
        s = sigma * (2 ** i)
        kernel = Gaussian2DKernel(s)
        cur = np.asarray(
            convolve_fft(
                cur,
                kernel,
                normalize_kernel=True,
                boundary="fill",
                fill_value=float(np.median(cur)),
            ),
            dtype=np.float64,
        )
    return cur


def estimate_background_rpca(data: np.ndarray, rank_keep: int = 3) -> np.ndarray:
    """
    RPCA 简化版：先降采样，再做低秩近似，最后上采样回原图作为背景。
    注：为了交互速度，采用截断 SVD 近似背景（工程上可用）。
    """
    img = _fill_nan_with_median(data)
    h, w = img.shape
    max_side = max(h, w)
    scale = max(1, int(np.ceil(max_side / 256)))
    hs = max(8, h // scale)
    ws = max(8, w // scale)
    # 区块均值下采样
    y_edges = np.linspace(0, h, hs + 1, dtype=int)
    x_edges = np.linspace(0, w, ws + 1, dtype=int)
    small = np.zeros((hs, ws), dtype=np.float64)
    for iy in range(hs):
        y0, y1 = y_edges[iy], y_edges[iy + 1]
        for ix in range(ws):
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            patch = img[y0:y1, x0:x1]
            small[iy, ix] = float(np.mean(patch))

    u, s, vt = np.linalg.svd(small, full_matrices=False)
    r = max(1, min(int(rank_keep), s.size))
    low = (u[:, :r] * s[:r]) @ vt[:r, :]
    bg = _upsample_bilinear(low, h, w)
    return bg


def _asinh_enhance(x: np.ndarray, gain: float = 8.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    lo = float(np.percentile(v, 1.0))
    hi = float(np.percentile(v, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    xn = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.arcsinh(gain * xn) / np.arcsinh(gain)


def process_robust_pipeline(
    data: np.ndarray,
    box: int = 48,
    clip_sigma: float = 3.0,
    denoise_sigma: float | None = None,
    mix_alpha: float = 0.7,
    asinh_gain: float = 8.0,
) -> np.ndarray:
    """推荐流程：mask + mesh + sigma clip + 轻度去噪 + asinh 对比增强。"""
    img = _fill_nan_with_median(data)
    box = max(16, int(box))
    bg = estimate_background_mesh(img, box=box, clip_sigma=float(clip_sigma))
    resid = img - bg

    # 轻度去噪（高斯）
    if denoise_sigma is None:
        sigma_dn = max(1.0, box / 24.0)
    else:
        sigma_dn = max(0.5, float(denoise_sigma))
    den = np.asarray(
        convolve_fft(
            resid,
            Gaussian2DKernel(sigma_dn),
            normalize_kernel=True,
            boundary="fill",
            fill_value=float(np.median(resid)),
        ),
        dtype=np.float64,
    )
    # 保留细节：与原残差按比例融合
    a = float(np.clip(mix_alpha, 0.0, 1.0))
    mix = a * resid + (1.0 - a) * den

    # 显示增强（不改变输入文件）
    enh = _asinh_enhance(mix, gain=max(0.5, float(asinh_gain)))
    # 映射回近似原量纲，便于 3D 对比（保留正负）
    v = mix[np.isfinite(mix)]
    amp = float(np.percentile(np.abs(v), 99.0)) if v.size else 1.0
    amp = max(amp, 1.0)
    return (enh - 0.5) * 2.0 * amp


def remove_background(data: np.ndarray, method: str, scale: int) -> np.ndarray:
    img = np.asarray(data, dtype=np.float64)
    if method == BG_METHOD_ORIGINAL:
        return img
    if method == BG_METHOD_MESH:
        bg = estimate_background_mesh(img, box=max(16, int(scale)))
        return img - bg
    if method == BG_METHOD_MORPH:
        bg = estimate_background_morphology(img, radius=max(2, int(scale // 2)))
        return img - bg
    if method == BG_METHOD_POLY2:
        sample = max(10000, int(scale) * 1200)
        bg = estimate_background_poly2d(img, sample_size=sample)
        return img - bg
    if method == BG_METHOD_WAVELET:
        levels = int(np.clip(np.log2(max(8, int(scale))) - 1, 2, 6))
        bg = estimate_background_wavelet(img, base_sigma=max(2.0, float(scale) / 6.0), levels=levels)
        return img - bg
    if method == BG_METHOD_RPCA:
        rank_keep = int(np.clip(int(scale) // 32 + 1, 1, 6))
        bg = estimate_background_rpca(img, rank_keep=rank_keep)
        return img - bg
    if method == BG_METHOD_PIPELINE:
        return process_robust_pipeline(img, box=max(16, int(scale)))
    return img


def remove_background_with_params(data: np.ndarray, method: str, params: dict[str, float | int]) -> np.ndarray:
    img = np.asarray(data, dtype=np.float64)
    if method == BG_METHOD_ORIGINAL:
        return img
    if method == BG_METHOD_MESH:
        box = int(params.get("box", 48))
        clip_sigma = float(params.get("clip_sigma", 3.0))
        return img - estimate_background_mesh(img, box=max(16, box), clip_sigma=clip_sigma)
    if method == BG_METHOD_MORPH:
        radius = int(params.get("radius", 24))
        return img - estimate_background_morphology(img, radius=max(2, radius))
    if method == BG_METHOD_POLY2:
        sample_size = int(params.get("sample_size", 60000))
        clip_sigma = float(params.get("clip_sigma", 3.0))
        return img - estimate_background_poly2d(
            img, sample_size=max(5000, sample_size), clip_sigma=max(1.5, clip_sigma)
        )
    if method == BG_METHOD_WAVELET:
        base_sigma = float(params.get("base_sigma", 8.0))
        levels = int(params.get("levels", 4))
        return img - estimate_background_wavelet(img, base_sigma=max(2.0, base_sigma), levels=max(2, levels))
    if method == BG_METHOD_RPCA:
        rank_keep = int(params.get("rank_keep", 3))
        return img - estimate_background_rpca(img, rank_keep=max(1, min(10, rank_keep)))
    if method == BG_METHOD_PIPELINE:
        box = int(params.get("box", 48))
        clip_sigma = float(params.get("clip_sigma", 3.0))
        denoise_sigma = float(params.get("denoise_sigma", max(1.0, box / 24.0)))
        mix_alpha = float(params.get("mix_alpha", 0.7))
        asinh_gain = float(params.get("asinh_gain", 8.0))
        return process_robust_pipeline(
            img,
            box=max(16, box),
            clip_sigma=max(1.5, clip_sigma),
            denoise_sigma=max(0.5, denoise_sigma),
            mix_alpha=float(np.clip(mix_alpha, 0.0, 1.0)),
            asinh_gain=max(0.5, asinh_gain),
        )
    return img

