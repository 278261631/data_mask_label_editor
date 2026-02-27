from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _config_path() -> Path:
    return Path(__file__).resolve().parent / "config.json"


@dataclass
class FitsDisplayConfig:
    # 来自 fits_io.py 的显示拉伸默认值
    lo_pct: float = 1.0
    hi_pct: float = 99.0
    # 来自主窗口的叠加透明度默认值
    overlay_alpha: float = 0.5


@dataclass
class View3DConfig:
    # 来自 view3d.py 的默认配置
    patch_size: int = 30
    figure_width: float = 10.0
    figure_height: float = 4.0
    figure_dpi: int = 100
    surface_cmap: str = "coolwarm"
    surface_alpha: float = 1.0
    grid_alpha: float = 0.3


@dataclass
class AppConfig:
    data_dir: str = ""
    last_image_path: str = ""
    last_mask_path: str = ""
    fits_display: FitsDisplayConfig = field(default_factory=FitsDisplayConfig)
    view3d: View3DConfig = field(default_factory=View3DConfig)

    @classmethod
    def load(cls) -> "AppConfig":
        p = _config_path()
        if not p.exists():
            return cls()
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return cls.from_dict(obj if isinstance(obj, dict) else {})
        except Exception:
            # 配置损坏时回退默认，避免启动失败
            return cls()

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "AppConfig":
        base = cls()
        fits_raw = obj.get("fits_display", {})
        view3d_raw = obj.get("view3d", {})

        fits_cfg = FitsDisplayConfig(
            lo_pct=float(fits_raw.get("lo_pct", base.fits_display.lo_pct)),
            hi_pct=float(fits_raw.get("hi_pct", base.fits_display.hi_pct)),
            overlay_alpha=float(fits_raw.get("overlay_alpha", base.fits_display.overlay_alpha)),
        )
        view3d_cfg = View3DConfig(
            patch_size=int(view3d_raw.get("patch_size", base.view3d.patch_size)),
            figure_width=float(view3d_raw.get("figure_width", base.view3d.figure_width)),
            figure_height=float(view3d_raw.get("figure_height", base.view3d.figure_height)),
            figure_dpi=int(view3d_raw.get("figure_dpi", base.view3d.figure_dpi)),
            surface_cmap=str(view3d_raw.get("surface_cmap", base.view3d.surface_cmap)),
            surface_alpha=float(view3d_raw.get("surface_alpha", base.view3d.surface_alpha)),
            grid_alpha=float(view3d_raw.get("grid_alpha", base.view3d.grid_alpha)),
        )
        return cls(
            data_dir=str(obj.get("data_dir", base.data_dir)),
            last_image_path=str(obj.get("last_image_path", base.last_image_path)),
            last_mask_path=str(obj.get("last_mask_path", base.last_mask_path)),
            fits_display=fits_cfg,
            view3d=view3d_cfg,
        )

    def save(self) -> None:
        p = _config_path()
        p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
