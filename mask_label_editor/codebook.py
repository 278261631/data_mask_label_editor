from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from mask_label_editor.labels import Label, default_labels, ensure_unique_codes, make_label


def load_codebook(path: str | Path, alpha: int = 128) -> list[Label]:
    """
    尝试从 json/csv/fits 读取 labels。
    - JSON: {"labels":[{"code":1,"name":"x","color":"#ff0000"}]}
    - CSV: code,name,color 或 code,name,r,g,b
    - FITS: 二进制表格 HDU，尝试识别列名（code/name/color 或 r/g/b）
    """
    p = Path(path)
    if not p.exists():
        return default_labels(alpha=alpha)

    suffix = p.suffix.lower()
    try:
        if suffix in {".json"}:
            return _load_json(p, alpha=alpha)
        if suffix in {".csv"}:
            return _load_csv(p, alpha=alpha)
        if suffix in {".fits", ".fit", ".fts"}:
            return _load_fits_table(p, alpha=alpha)
    except Exception:
        # 任何解析失败都回退默认，避免 GUI 直接崩
        return default_labels(alpha=alpha)

    # 未知后缀：尝试 JSON
    try:
        return _load_json(p, alpha=alpha)
    except Exception:
        return default_labels(alpha=alpha)


def save_codebook_json(path: str | Path, labels: list[Label]) -> None:
    p = Path(path)
    obj = {"labels": [asdict(l) for l in labels]}
    # 把 rgba 拆成更易读的 hex + alpha
    for item, l in zip(obj["labels"], labels):
        r, g, b, a = l.color_rgba
        item["color"] = f"#{r:02x}{g:02x}{b:02x}"
        item["alpha"] = a
        item.pop("color_rgba", None)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(p: Path, alpha: int) -> list[Label]:
    obj = json.loads(p.read_text(encoding="utf-8"))

    # 支持 mask_codebook.json 的 mask_types 格式
    if "mask_types" in obj:
        labels_raw = obj["mask_types"]
    elif "labels" in obj:
        labels_raw = obj["labels"]
    elif isinstance(obj, list):
        labels_raw = obj
    else:
        labels_raw = []

    # 为没有颜色信息的 codebook 分配可区分颜色
    _auto_palette = [
        "#442044",  # 0: normal/background - 
        "#34c759",  # 1: good - 绿色
        "#ff3b30",  # 2: bad - 红色
        "#0a84ff",  # 3: 蓝
        "#ff9f0a",  # 4: 橙
        "#bf5af2",  # 5: 紫
        "#64d2ff",  # 6: 浅蓝
        "#ffd60a",  # 7: 黄
    ]

    out: list[Label] = []
    for item in labels_raw:
        code = int(item.get("code", item.get("id", 0)))
        name = str(item.get("name", f"code_{code}"))
        if "color" in item:
            color = str(item["color"])
        elif "r" in item and "g" in item and "b" in item:
            r = int(item.get("r", 0))
            g = int(item.get("g", 0))
            b = int(item.get("b", 0))
            color = (r, g, b)
        else:
            # 自动分配颜色
            if code < len(_auto_palette):
                color = _auto_palette[code]
            else:
                color = _hsv_to_rgb_hex((code * 47 % 360) / 360.0, 0.9, 0.95)
        # code=0 使用其他类别一半透明度
        default_a = max(1, int(alpha) // 2) if code == 0 else alpha
        a = int(item.get("alpha", default_a))
        out.append(make_label(code, name, color, alpha=a))
    out = ensure_unique_codes(out)
    return out if out else default_labels(alpha=alpha)


def _load_csv(p: Path, alpha: int) -> list[Label]:
    out: list[Label] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = int(row.get("code") or row.get("id") or row.get("label") or 0)
            name = row.get("name") or row.get("class") or f"code_{code}"
            if "color" in row and row["color"]:
                color: Any = row["color"]
            else:
                r = int(row.get("r") or 0)
                g = int(row.get("g") or 0)
                b = int(row.get("b") or 0)
                color = (r, g, b)
            a = int(row.get("alpha") or alpha)
            out.append(make_label(code, name, color, alpha=a))
    out = ensure_unique_codes(out)
    return out if out else default_labels(alpha=alpha)


def _load_fits_table(p: Path, alpha: int) -> list[Label]:
    with fits.open(p) as hdul:
        table_hdu = None
        for h in hdul:
            if isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                table_hdu = h
                break
        if table_hdu is None or table_hdu.data is None:
            return default_labels(alpha=alpha)

        data = table_hdu.data
        colnames = {c.lower(): c for c in data.names}  # type: ignore[attr-defined]

        def getcol(*names: str) -> str | None:
            for n in names:
                if n.lower() in colnames:
                    return colnames[n.lower()]
            return None

        code_col = getcol("code", "id", "label", "class_id")
        name_col = getcol("name", "class", "label_name")
        color_col = getcol("color", "hex", "hexcolor")
        r_col = getcol("r", "red")
        g_col = getcol("g", "green")
        b_col = getcol("b", "blue")
        a_col = getcol("alpha", "a")

        if code_col is None:
            return default_labels(alpha=alpha)

        out: list[Label] = []
        for i in range(len(data)):
            code = int(data[code_col][i])
            name = str(data[name_col][i]) if name_col else f"code_{code}"
            if color_col:
                color = str(data[color_col][i])
            elif r_col and g_col and b_col:
                color = (int(data[r_col][i]), int(data[g_col][i]), int(data[b_col][i]))
            else:
                # 没颜色就给个可区分的默认色（由 default_labels 覆盖不到时兜底）
                hue = (code * 47) % 360
                color = _hsv_to_rgb_hex(hue / 360.0, 0.9, 0.95)
            a = int(data[a_col][i]) if a_col else alpha
            out.append(make_label(code, name, color, alpha=a))
        out = ensure_unique_codes(out)
        return out if out else default_labels(alpha=alpha)


def _hsv_to_rgb_hex(h: float, s: float, v: float) -> str:
    # 小工具：避免引入额外依赖
    h = float(h) % 1.0
    s = float(np.clip(s, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

