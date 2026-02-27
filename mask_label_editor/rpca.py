from __future__ import annotations

import numpy as np


def _soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - float(t), 0.0)


def robust_pca_inexact_alm(
    d: np.ndarray,
    lam: float | None = None,
    max_iter: int = 120,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Inexact ALM 求解 RPCA: min ||L||_* + lam * ||S||_1, s.t. D = L + S

    返回:
    - L: 低秩分量
    - S: 稀疏分量
    - iters: 实际迭代次数
    - err: 终止时相对残差
    """
    x = np.asarray(d, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"RPCA 输入必须是二维矩阵，当前 shape={x.shape}")
    m, n = x.shape
    if m == 0 or n == 0:
        raise ValueError("RPCA 输入为空矩阵")

    lam_eff = float(1.0 / np.sqrt(max(m, n))) if lam is None else float(lam)
    lam_eff = max(lam_eff, 1e-9)

    # 初始化（参考 Candes 等人的 inexact ALM 实现）
    norm2 = float(np.linalg.norm(x, 2))
    norm_inf = float(np.linalg.norm(x.reshape(-1), np.inf)) / lam_eff
    dual_norm = max(norm2, norm_inf, 1e-8)
    y = x / dual_norm

    mu = 1.25 / max(norm2, 1e-8)
    mu_bar = mu * 1e7
    rho = 1.5

    l = np.zeros_like(x)
    s = np.zeros_like(x)
    norm_x = float(np.linalg.norm(x, ord="fro"))
    norm_x = max(norm_x, 1e-12)

    err = 1.0
    iters = 0
    for i in range(max(1, int(max_iter))):
        iters = i + 1
        u, sigma, vh = np.linalg.svd(x - s + (1.0 / mu) * y, full_matrices=False)
        sigma_shrink = np.maximum(sigma - 1.0 / mu, 0.0)
        rank = int(np.count_nonzero(sigma_shrink > 0))
        if rank > 0:
            l = (u[:, :rank] * sigma_shrink[:rank]) @ vh[:rank, :]
        else:
            l.fill(0.0)

        s = _soft_threshold(x - l + (1.0 / mu) * y, lam_eff / mu)
        z = x - l - s
        y = y + mu * z
        err = float(np.linalg.norm(z, ord="fro") / norm_x)
        if err < float(tol):
            break
        mu = min(mu * rho, mu_bar)

    return l, s, iters, err

