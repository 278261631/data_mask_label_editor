from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Label:
    code: int
    name: str
    color_rgba: tuple[int, int, int, int]  # 0-255


def _parse_hex_color(s: str) -> tuple[int, int, int]:
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) != 6:
        raise ValueError(f"Invalid color hex: {s!r}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return r, g, b


def make_label(code: int, name: str, color: str | tuple[int, int, int], alpha: int = 128) -> Label:
    if isinstance(color, str):
        r, g, b = _parse_hex_color(color)
    else:
        r, g, b = color
    return Label(code=int(code), name=str(name), color_rgba=(int(r), int(g), int(b), int(alpha)))


def default_labels(alpha: int = 128) -> list[Label]:
    # 基础调色板：背景 + 若干可区分颜色
    palette = [
        ("background", "#808080"),
        ("class1", "#ff3b30"),
        ("class2", "#34c759"),
        ("class3", "#0a84ff"),
        ("class4", "#ff9f0a"),
        ("class5", "#bf5af2"),
        ("class6", "#64d2ff"),
        ("class7", "#ffd60a"),
    ]
    out: list[Label] = []
    for i, (name, color) in enumerate(palette):
        # code=0 使用其他类别一半透明度
        a = max(1, int(alpha) // 2) if i == 0 else alpha
        out.append(make_label(i, name, color, alpha=a))
    return out


def labels_to_lut(labels: Iterable[Label], max_code: int | None = None) -> "tuple[int, ...]":
    """
    返回用于快速映射的 LUT（长度为 max_code+1），每个元素是 0xAARRGGBB。
    """
    labels_list = list(labels)
    if not labels_list:
        labels_list = default_labels(alpha=128)
    if max_code is None:
        max_code = max(l.code for l in labels_list)

    lut = [0] * (max_code + 1)
    for l in labels_list:
        if l.code < 0:
            continue
        if l.code > max_code:
            continue
        r, g, b, a = l.color_rgba
        lut[l.code] = (a << 24) | (r << 16) | (g << 8) | b
    return tuple(lut)


def ensure_unique_codes(labels: Iterable[Label]) -> list[Label]:
    seen: set[int] = set()
    out: list[Label] = []
    for l in labels:
        if l.code in seen:
            continue
        seen.add(l.code)
        out.append(l)
    return out


def label_name_map(labels: Iterable[Label]) -> Mapping[int, str]:
    return {l.code: l.name for l in labels}

