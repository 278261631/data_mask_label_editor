from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


DEFAULT_DATA_DIR = r"D:\github\SiameseNetwork_fits_diff\data"


def _config_path() -> Path:
    # 放在仓库根目录（package 上一级），避免从其它 cwd 启动时找不到配置
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    return repo_root / "config.json"


@dataclass
class AppConfig:
    data_dir: str = DEFAULT_DATA_DIR
    window_width: int = 1680
    window_height: int = 900
    patch_size: int = 30
    bg_method: str = "original"
    bg_scale: int = 48
    mesh_box: int = 48
    mesh_clip_sigma: float = 3.0
    morph_radius: int = 24
    poly_sample_size: int = 60000
    poly_clip_sigma: float = 3.0
    wavelet_base_sigma: float = 8.0
    wavelet_levels: int = 4
    rpca_rank_keep: int = 3
    pipeline_box: int = 48
    pipeline_clip_sigma: float = 3.0
    pipeline_denoise_sigma: float = 2.0
    pipeline_mix_alpha: float = 0.7
    pipeline_asinh_gain: float = 8.0

    def __post_init__(self) -> None:
        self.window_width = max(960, min(4096, int(self.window_width)))
        self.window_height = max(640, min(2160, int(self.window_height)))

        try:
            v = int(self.patch_size)
        except Exception:
            v = 30
        self.patch_size = max(10, min(100, v))

        # 兼容旧配置名
        legacy_map = {
            "mesh_median": "mesh_sigma_clip",
            "gaussian_lowpass": "mesh_sigma_clip",
        }
        self.bg_method = legacy_map.get(self.bg_method, self.bg_method)
        allowed_methods = {
            "original",
            "mesh_sigma_clip",
            "morph_opening",
            "poly2d",
            "wavelet_multiscale",
            "rpca_lowrank_sparse",
            "robust_pipeline",
        }
        if self.bg_method not in allowed_methods:
            self.bg_method = "original"

        try:
            s = int(self.bg_scale)
        except Exception:
            s = 48
        self.bg_scale = max(8, min(256, s))

        self.mesh_box = max(16, int(self.mesh_box))
        self.mesh_clip_sigma = min(max(float(self.mesh_clip_sigma), 1.5), 8.0)
        self.morph_radius = max(2, int(self.morph_radius))
        self.poly_sample_size = max(5000, int(self.poly_sample_size))
        self.poly_clip_sigma = min(max(float(self.poly_clip_sigma), 1.5), 8.0)
        self.wavelet_base_sigma = min(max(float(self.wavelet_base_sigma), 2.0), 64.0)
        self.wavelet_levels = min(max(int(self.wavelet_levels), 2), 8)
        self.rpca_rank_keep = min(max(int(self.rpca_rank_keep), 1), 10)
        self.pipeline_box = max(16, int(self.pipeline_box))
        self.pipeline_clip_sigma = min(max(float(self.pipeline_clip_sigma), 1.5), 8.0)
        self.pipeline_denoise_sigma = min(max(float(self.pipeline_denoise_sigma), 0.5), 20.0)
        self.pipeline_mix_alpha = min(max(float(self.pipeline_mix_alpha), 0.0), 1.0)
        self.pipeline_asinh_gain = min(max(float(self.pipeline_asinh_gain), 0.5), 40.0)

    @classmethod
    def load(cls) -> "AppConfig":
        p = _config_path()
        if not p.exists():
            return cls()
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            base = asdict(cls())
            # 过滤未知字段，避免旧配置或手改配置导致启动失败
            merged = {**base, **{k: v for k, v in obj.items() if k in base}}
            return cls(**merged)
        except Exception:
            # 配置损坏就回退默认
            return cls()

    def save(self) -> None:
        p = _config_path()
        p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

