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
    last_image_path: str = ""
    last_mask_path: str = ""
    last_codebook_path: str = ""
    overlay_alpha: float = 0.5
    brush_radius: int = 8

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

